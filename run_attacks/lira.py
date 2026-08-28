"""
LiRA (Likelihood-Ratio Attack) implementation within ml_privacy_meter.

Uses the RMIA trained models instead of training separate shadow models.
Loads target + shadow models from the RMIA model directory for fair comparison.

Key differences from RMIA:
- Uses shadow model distributions (no population dataset needed)
- Computes per-sample statistics from shadow models
- Likelihood-ratio based scoring using Gaussian distributions

Key points preserved:
- Uses identical dataset splits as RMIA (75/25 train/auditing split)
- Uses same auditing samples for fair comparison
- Generates same index comparisons for reproducibility
"""

import sys
import os
import yaml
import pandas as pd
import time
import numpy as np
import torch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
from configs import ensure_dataset_ready


def load_dataset(dataset_name: str, base_dir: str = ".") -> pd.DataFrame:
    dataset_path = os.path.join(base_dir, dataset_name + '.csv')
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


def main(dataset_name: str = "locations", model_name: str = None, online: bool = False, mode: str = "load", defense: str = "none", seed: int = None):
    # Enable cudnn benchmark for faster training if inputs are consistent
    torch.backends.cudnn.benchmark = True

    config_filename = f"{dataset_name}_{model_name}.yaml" if model_name else f"{dataset_name}.yaml"
    config_file = f"ml_privacy_meter/configs/{config_filename}"

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    if online:
        configs["audit"]["online_attack"] = True

    check_configs(configs)
    initialize_seeds(configs["run"]["random_seed"])

    model_name_for_logs = str(configs["train"]["model_name"]).lower()
    base_log_dir = os.path.join("ml_privacy_meter", "logs", dataset_name, model_name_for_logs)
    if seed is not None:
        base_log_dir = os.path.join(base_log_dir, f"seed{seed}")
    log_dir = os.path.join(base_log_dir, "lira")
    rmia_log_dir = os.path.join(base_log_dir, "rmia")
    configs["run"]["log_dir"] = log_dir

    report_suffix = "_online" if online else ""
    if defense != "none":
        report_dir = os.path.join(log_dir, f"defense_{defense}", f"report{report_suffix}")
    else:
        report_dir = os.path.join(log_dir, f"report{report_suffix}")
    directories = {
        "log_dir": log_dir,
        "report_dir": report_dir,
        "signal_dir": os.path.join(log_dir, "signals"),
        "data_dir": configs["data"]["data_dir"],
    }
    create_directories(directories)

    if mode == "signal":
        import shutil
        signal_dir = directories["signal_dir"]
        if os.path.exists(signal_dir):
            shutil.rmtree(signal_dir)
        os.makedirs(signal_dir, exist_ok=True)

    logger = setup_log(
        directories["report_dir"],
        "time_analysis",
        configs["run"].get("time_log", True)
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
    num_reference_models = configs["audit"]["num_ref_models"]

    import math
    num_model_pairs = max(math.ceil(num_experiments / 2.0), num_reference_models + 1)
    models_list, memberships = load_models(rmia_log_dir, dataset, 2 * num_model_pairs, configs, logger)
    if models_list is None or memberships is None:
        raise FileNotFoundError(
            f"Could not load RMIA models from {rmia_log_dir}/models. Run RMIA first."
        )

    auditing_dataset, auditing_membership = sample_auditing_dataset(
        configs, dataset, logger, memberships
    )

    # Reuse RMIA signals if available (same model, same auditing data).
    rmia_signal_path = os.path.join(rmia_log_dir, "signals", "rmia_signals.npy")
    lira_signal_path = os.path.join(directories["signal_dir"], "lira_signals.npy")
    if not os.path.exists(lira_signal_path) and os.path.exists(rmia_signal_path):
        rmia_sigs = np.load(rmia_signal_path)
        if rmia_sigs.shape[0] == len(auditing_dataset):
            logger.info("Reusing RMIA signals for LiRA (no inference needed)")
            os.makedirs(directories["signal_dir"], exist_ok=True)
            np.save(lira_signal_path, rmia_sigs)

    configs["audit"]["algorithm"] = "LIRA"

    if defense != "none":
        from run_defenses.hamp_inference import make_hamp_wrapper, log_defense_accuracy
        configs["run"]["log_dir"] = os.path.join(log_dir, f"defense_{defense}")
        orig_target = models_list[0]
        models_list = [make_hamp_wrapper(m, dataset.data, model_name_for_logs) for m in models_list]
        log_defense_accuracy(orig_target, models_list[0],
                             X[training_size:], y[training_size:],
                             directories["report_dir"], logger)

    logger.info("Preparing model signals for auditing dataset")
    all_signals = get_model_signals(models_list, auditing_dataset, configs, logger)

    target_model_indices = list(range(num_experiments))
    mia_score_list, membership_list = audit_models(
        f"{directories['report_dir']}/exp",
        target_model_indices,
        all_signals,
        None,  # No population signals needed for LiRA
        auditing_membership,
        num_reference_models,
        logger,
        configs,
    )

    if len(target_model_indices) > 1:
        get_average_audit_results(
            directories["report_dir"], mia_score_list, membership_list, logger
        )

    if seed is not None and defense == "none":
        from run_attacks.seed_summary import update_seed_row
        attack_label = "lira_online" if online else "lira"
        update_seed_row(
            attack_label, seed,
            Path(directories["report_dir"]),
            Path(base_log_dir).parent,
        )

    logger.info("Total runtime: %0.5f seconds", time.time() - start_time)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LiRA attack within ml_privacy_meter.")
    parser.add_argument("--dataset", type=str, default="locations", help="Dataset name to use")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for config generation. If omitted, a missing config is auto-created with model='mlp'.",
    )
    parser.add_argument("--skip-config", action="store_true", help="Skip rewriting the YAML config if it already exists.")
    parser.add_argument(
        "--online",
        action="store_true",
        help="Run LiRA in online mode (uses both IN and OUT shadow models). Results go to report_online/.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="load",
        choices=["signal", "load"],
        help="'signal' deletes existing signals and recomputes them from RMIA models; 'load' reuses existing signals (default).",
    )
    parser.add_argument("--defense", type=str, default="none", choices=["none", "hamp"],
                        help="Apply a test-time defense before computing attack signals.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed matching a prior RMIA seeded run. Reads models and signals from seed<seed>/ subdirectory.")
    args = parser.parse_args()

    ensure_dataset_ready(dataset_name=args.dataset, model_name=args.model, algorithm="LIRA", skip_if_exists=args.skip_config)

    sys.path.append(str(Path(__file__).parent.parent / "ml_privacy_meter"))

    from dataset import TabularDataset
    from audit import get_average_audit_results, audit_models, sample_auditing_dataset
    from get_signals import get_model_signals
    from models.utils import load_models
    from util import check_configs, setup_log, initialize_seeds, create_directories

    main(dataset_name=args.dataset, model_name=args.model, online=args.online, mode=args.mode, defense=args.defense, seed=args.seed)
