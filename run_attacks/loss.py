import os
import sys
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs import ensure_dataset_ready


def load_dataset(dataset_name: str, base_dir: str = ".") -> pd.DataFrame:
    dataset_path = os.path.join(base_dir, dataset_name + ".csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    return pd.read_csv(dataset_path, header=None)


def prepare_tabular_arrays(df: pd.DataFrame):
    """Convert mixed-type tabular data into numeric arrays for training."""
    data = df.copy()

    for col in data.columns[:-1]:
        if pd.api.types.is_string_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
            try:
                data[col] = pd.to_numeric(data[col], errors="raise")
            except (ValueError, TypeError):
                data[col] = data[col].astype("category").cat.codes

    label_col = data.columns[-1]
    if pd.api.types.is_string_dtype(data[label_col]) or pd.api.types.is_object_dtype(data[label_col]):
        try:
            data[label_col] = pd.to_numeric(data[label_col], errors="raise")
        except (ValueError, TypeError):
            data[label_col] = data[label_col].astype("category").cat.codes

    arr = data.to_numpy()
    y = arr[:, -1]
    X = arr[:, :-1].astype(np.float32)
    return X, y


def main(dataset_name: str = "locations", model_name: str = None, seed: int = None):
    torch.backends.cudnn.benchmark = True

    config_filename = f"{dataset_name}_{model_name}.yaml" if model_name else f"{dataset_name}.yaml"
    config_file = f"ml_privacy_meter/configs/{config_filename}"
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    check_configs(configs)
    initialize_seeds(configs["run"]["random_seed"])

    model_name_for_logs = str(configs["train"]["model_name"]).lower()
    base_log_dir = os.path.join("ml_privacy_meter", "logs", dataset_name, model_name_for_logs)
    if seed is not None:
        base_log_dir = os.path.join(base_log_dir, f"seed{seed}")
    log_dir = os.path.join(base_log_dir, "loss")
    rmia_log_dir = os.path.join(base_log_dir, "rmia")
    configs["run"]["log_dir"] = log_dir
    configs["audit"]["algorithm"] = "LOSS"
    # LOSS does not use reference models.
    configs["audit"]["num_ref_models"] = 0
    num_reference_models = 0

    report_dir = os.path.join(log_dir, "report")
    directories = {
        "log_dir": log_dir,
        "report_dir": report_dir,
        "signal_dir": os.path.join(log_dir, "signals"),
        "data_dir": configs["data"]["data_dir"],
    }
    create_directories(directories)

    logger = setup_log(
        directories["report_dir"], "time_analysis", configs["run"]["time_log"]
    )

    start_time = time.time()

    df = load_dataset(dataset_name=dataset_name, base_dir=directories["data_dir"])
    print(f"Loaded {dataset_name} dataset with shape: {df.shape}")
    X, y = prepare_tabular_arrays(df)
    if seed is not None:
        perm_path = os.path.join(rmia_log_dir, "splits", "dataset_permutation.npy")
        if not os.path.exists(perm_path):
            raise FileNotFoundError(
                f"Dataset permutation not found: {perm_path}. Run RMIA with --seed {seed} first."
            )
        order = np.load(perm_path)
        X, y = X[order], y[order]
        print(f"[SEED] Applied dataset permutation for seed={seed}")
    training_size = int(len(y) * 0.75)

    dataset = TabularDataset(X[:training_size], y[:training_size])

    num_experiments = configs["run"]["num_experiments"]
    # LOSS only needs the target model — one pair is sufficient.
    num_model_pairs = max(math.ceil(num_experiments / 2.0), 1)

    # LOSS always reuses the RMIA target model — no separate training needed.
    models_list, memberships = load_models(rmia_log_dir, dataset, num_model_pairs, configs, logger)
    if models_list is None or memberships is None:
        raise FileNotFoundError(
            f"Could not load RMIA models from {rmia_log_dir}/models. Run RMIA first."
        )

    auditing_dataset, auditing_membership = sample_auditing_dataset(
        configs, dataset, logger, memberships
    )

    # LOSS only scores each target model independently — no reference models needed.
    # Trim to just the target models so we don't waste time computing signals for the
    # paired model that was trained alongside each target but is never used by LOSS.
    target_models_list = models_list[:num_experiments]
    auditing_membership = auditing_membership[:num_experiments, :]

    # RMIA signals are identical to what LOSS would compute (same model, same auditing data).
    # Reuse the cached RMIA signal file directly to skip redundant inference.
    rmia_signal_path = os.path.join(rmia_log_dir, "signals", "rmia_signals.npy")
    loss_signal_path = os.path.join(directories["signal_dir"], "loss_signals.npy")
    if not os.path.exists(loss_signal_path) and os.path.exists(rmia_signal_path):
        rmia_sigs = np.load(rmia_signal_path)
        if rmia_sigs.shape[0] == len(auditing_dataset) and rmia_sigs.shape[1] >= num_experiments:
            logger.info("Reusing RMIA signals for LOSS (no inference needed)")
            os.makedirs(directories["signal_dir"], exist_ok=True)
            np.save(loss_signal_path, rmia_sigs[:, :num_experiments])

    logger.info("Preparing LOSS signals for auditing dataset")
    signals = get_model_signals(target_models_list, auditing_dataset, configs, logger)

    target_model_indices = list(range(num_experiments))
    mia_score_list, membership_list = audit_models(
        f"{directories['report_dir']}/exp",
        target_model_indices,
        signals,
        None,  # LOSS does not use population signals
        auditing_membership,
        num_reference_models,
        logger,
        configs,
    )

    if len(target_model_indices) > 1:
        get_average_audit_results(
            directories["report_dir"], mia_score_list, membership_list, logger
        )

    logger.info("LOSS runtime: %0.5f seconds", time.time() - start_time)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LOSS membership inference attack.")
    parser.add_argument("--dataset", type=str, default="locations", help="Dataset name to use")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for config generation. If omitted, a missing config is auto-created with model='mlp'.",
    )
    parser.add_argument("--skip-config", action="store_true", help="Skip rewriting the YAML config if it already exists.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed matching a prior RMIA seeded run. Reads models and signals from seed<seed>/ subdirectory.")
    args = parser.parse_args()

    ensure_dataset_ready(dataset_name=args.dataset, model_name=args.model, algorithm="LOSS", skip_if_exists=args.skip_config)

    sys.path.append(str(Path(__file__).parent.parent / "ml_privacy_meter"))

    from dataset import TabularDataset
    from audit import get_average_audit_results, audit_models, sample_auditing_dataset
    from get_signals import get_model_signals
    from models.utils import load_models
    from util import check_configs, setup_log, initialize_seeds, create_directories

    main(dataset_name=args.dataset, model_name=args.model, seed=args.seed)
