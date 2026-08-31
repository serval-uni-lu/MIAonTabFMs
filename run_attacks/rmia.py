import shutil
import sys
import os
import yaml
import pandas as pd
import time
import numpy as np
import torch
import torch.multiprocessing
import math
import re
from pathlib import Path

torch.multiprocessing.set_sharing_strategy('file_system')
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs import ensure_dataset_ready


def load_dataset(dataset_name: str, base_dir: str = ".") -> pd.DataFrame:
    """
    Loads and preprocesses a dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'locations', 'purchases10').
        base_dir (str): Base directory where the datasets are located.

    Returns:
        pd.DataFrame: Preprocessed dataset with labels in the last column.
    """
    dataset_path = os.path.join(base_dir, dataset_name + '.csv')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    df = pd.read_csv(dataset_path, header=None)

    #df = df.loc[:1200, :]  # Limit to first 1200 for tabfpn
    return df


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


def load_proxy_signals(dataset_name: str, proxy_model_name: str, context_pct: float) -> tuple:
    """
    Load pre-computed signals and memberships from a proxy model's log directory.

    Returns (signals, pop_signals, memberships) where:
      - signals shape: (n_samples, n_models)
      - pop_signals shape: (n_pop, n_models)
      - memberships shape: (n_models, n_samples)
    """
    proxy_base = os.path.join("ml_privacy_meter", "logs", dataset_name, proxy_model_name.lower())
    if context_pct < 100.0:
        proxy_log_dir = os.path.join(proxy_base, f"rmia_ctx{int(context_pct)}")
    else:
        proxy_log_dir = os.path.join(proxy_base, "rmia")

    sig_path  = os.path.join(proxy_log_dir, "signals", "rmia_signals.npy")
    pop_path  = os.path.join(proxy_log_dir, "signals", "rmia_signals_pop.npy")
    mem_path  = os.path.join(proxy_log_dir, "models", "memberships.npy")

    for p in (sig_path, pop_path, mem_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Proxy model file not found: {p}\n"
                f"Run 'python rmia.py --dataset {dataset_name} --model {proxy_model_name}' first."
            )

    return np.load(sig_path), np.load(pop_path), np.load(mem_path)


def _rmia_log_subdir(context_pct: float, seed: int = None) -> str:
    base = f"rmia_ctx{int(context_pct)}" if context_pct < 100.0 else "rmia"
    if seed is not None:
        return os.path.join(f"seed{seed}", base)
    return base




def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())


def _load_or_create_seed_permutation(log_dir: str, n_samples: int, seed: int) -> np.ndarray:
    split_dir = os.path.join(log_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)
    path = os.path.join(split_dir, "dataset_permutation.npy")
    if os.path.exists(path):
        try:
            order = np.load(path)
            valid = len(order) == n_samples and set(order.tolist()) == set(range(n_samples))
        except Exception as exc:
            valid = False
            print(f"[SEED] Could not load saved dataset permutation at {path}: {exc}")
        if valid:
            print(f"[SEED] Loaded saved dataset permutation from {path}")
            return order

        backup_path = f"{path}.invalid"
        try:
            os.replace(path, backup_path)
            print(f"[SEED] Moved invalid dataset permutation to {backup_path}")
        except OSError as exc:
            print(f"[SEED] Could not move invalid dataset permutation {path}: {exc}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    np.save(path, order)
    print(f"[SEED] Created dataset permutation at {path}")
    return order


def summarize_seed_results(report_dirs: list, seeds: list, summary_dir: str,
                           online: bool = False, context_pct: float = 100.0) -> None:
    from run_attacks.seed_summary import update_seed_row
    base_label = f"rmia_ctx{int(context_pct)}" if context_pct < 100.0 else "rmia"
    attack_label = f"{base_label}_online" if online else base_label
    for seed, report_dir in zip(seeds, report_dirs):
        update_seed_row(attack_label, seed, Path(report_dir), Path(summary_dir))


def _quantile_calibrate_cols(
    proxy_signals: np.ndarray,
    proxy_pop_signals: np.ndarray,
    target_pop_signals: np.ndarray,
) -> tuple:
    """
    Calibrate proxy reference signals to match the target architecture's probability scale.

    Why this is needed
    ------------------
    RMIA's formula contains an additive floor term:
        mean_x = (1+a)/2 * mean_out_ref(x)  +  (1-a)/2
    This floor (e.g. 0.35 when a=0.3) assumes target and reference probabilities live on
    the same scale.  When the proxy architecture is more confident than the target
    (e.g. TabPFN gives 0.91 where RF gives 0.62 for the same sample), the floor term
    becomes small relative to the proxy signal, inflating the denominator and collapsing
    the RMIA ratio below 1.0 even for members — inverting the membership signal.

    Fix: quantile calibration
    --------------------------
    For each proxy column, map its values so that the population distribution of the
    calibrated proxy matches the population distribution of the target (col 0).
    Concretely: p_calibrated = F_target_pop^{-1}( F_proxy_pop( p_proxy ) )
    where F is the empirical CDF.  This is a monotone mapping so the within-column
    ordering (and therefore the IN/OUT membership signal) is preserved, but the
    probability scale now matches the target architecture.

    Args:
        proxy_signals:     (n_samples, n_proxy_cols) — proxy auditing-set signals (cols 2+)
        proxy_pop_signals: (n_pop,     n_proxy_cols) — proxy population signals
        target_pop_signals:(n_pop,     1)            — target (col 0) population signals,
                           used as the reference distribution to calibrate toward

    Returns:
        (calibrated_signals, calibrated_pop_signals)
    """
    target_pop_sorted = np.sort(target_pop_signals[:, 0])
    n_target = len(target_pop_sorted)

    cal_signals     = np.empty_like(proxy_signals,     dtype=np.float64)
    cal_pop_signals = np.empty_like(proxy_pop_signals, dtype=np.float64)

    for col in range(proxy_signals.shape[1]):
        proxy_pop_col = proxy_pop_signals[:, col]

        # For each proxy value, find its percentile in the proxy population,
        # then map that percentile to the corresponding value in the target population.
        def _map(values):
            # Percentile of each value within the proxy population (linear interpolation)
            pcts = np.searchsorted(np.sort(proxy_pop_col), values, side="right") / len(proxy_pop_col)
            pcts = np.clip(pcts, 0.0, 1.0)
            # Map percentile to target population quantile
            indices = pcts * (n_target - 1)
            lo = np.floor(indices).astype(int)
            hi = np.minimum(lo + 1, n_target - 1)
            frac = indices - lo
            return target_pop_sorted[lo] * (1 - frac) + target_pop_sorted[hi] * frac

        cal_signals[:, col]     = _map(proxy_signals[:, col])
        cal_pop_signals[:, col] = _map(proxy_pop_col)

    return cal_signals, cal_pop_signals


def build_proxy_combined_signals(
    target_signals: np.ndarray,
    target_pop_signals: np.ndarray,
    target_memberships: np.ndarray,
    proxy_signals: np.ndarray,
    proxy_pop_signals: np.ndarray,
    proxy_memberships: np.ndarray,
) -> tuple:
    """
    Combine target model signals with cross-architecture proxy reference model signals.

    Layout of the combined array:
      col 0  = target model          (from target, col 0)
      col 1  = target's paired model (from target, col 1) — excluded from references by RMIA
      col 2+ = proxy reference models (from proxy, cols 2+) — quantile-calibrated to
               match the target architecture's probability scale

    Args:
        target_*: signals/memberships from the target model's run directory
        proxy_*:  signals/memberships from the proxy model's run directory

    Returns:
        (combined_signals, combined_pop_signals, combined_memberships)
    """
    # Calibrate proxy cols 2+ to match the target's probability distribution (col 0).
    cal_proxy_signals, cal_proxy_pop_signals = _quantile_calibrate_cols(
        proxy_signals[:, 2:],
        proxy_pop_signals[:, 2:],
        target_pop_signals[:, :1],   # col 0 only, as reference distribution
    )

    combined_signals     = np.concatenate([target_signals[:, :2],     cal_proxy_signals],     axis=1)
    combined_pop_signals = np.concatenate([target_pop_signals[:, :2], cal_proxy_pop_signals], axis=1)
    # memberships: rows 0-1 from target, rows 2+ from proxy
    combined_memberships = np.concatenate([target_memberships[:2, :], proxy_memberships[2:, :]], axis=0)
    return combined_signals, combined_pop_signals, combined_memberships


def main(dataset_name: str = "locations", mode: str = "train", context_pct: float = 100.0, gpu: str = None, model_name: str = None, proxy_model: str = None, online: bool = False, num_ref_override: int = None, seed: int = None, skip_existing: bool = False):
    """
    Main entry point for running ML Privacy Meter experiments.

    Args:
        dataset_name (str): Name of the dataset to load and process.
        config_file (str): Path to the YAML configuration file.
    """
    # Enable cudnn benchmark for faster training if inputs are consistent
    torch.backends.cudnn.benchmark = True

    config_filename = f"{dataset_name}_{model_name}.yaml" if model_name else f"{dataset_name}.yaml"
    config_file = f"ml_privacy_meter/configs/{config_filename}"

    # Load configs
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    if seed is not None:
        configs["run"]["random_seed"] = seed
        configs["run"]["seed"] = seed
        configs["run"]["context_pct"] = context_pct
        configs.setdefault("train", {})["random_state"] = seed

    # If a GPU was requested, override the YAML device fields so all trainers
    # and auditing code use cuda:0 (the first visible GPU after CUDA_VISIBLE_DEVICES is set).
    if gpu is not None:
        configs.setdefault("train", {})["device"] = "cuda:0"
        configs.setdefault("audit", {})["device"] = "cuda:0"

    # CLI --online flag overrides the config value
    if online:
        configs["audit"]["online_attack"] = True

    # CLI --num-ref overrides the YAML num_ref_models for sweep experiments
    if num_ref_override is not None:
        configs["audit"]["num_ref_models"] = num_ref_override

    # Validate configurations
    check_configs(configs)

    # Initialize seeds for reproducibility
    initialize_seeds(configs["run"]["random_seed"])

    # RMIA logs to: ml_privacy_meter/logs/<dataset>/<model>/rmia/
    # When context_pct < 100 a separate subdir is used so full-data runs are not overwritten.
    model_name_for_logs = str(configs["train"]["model_name"]).lower()
    base_log_dir = os.path.join("ml_privacy_meter", "logs", dataset_name, model_name_for_logs)
    model_log_dir = os.path.join(base_log_dir, _rmia_log_subdir(context_pct, seed))
    log_dir = model_log_dir
    configs["run"]["log_dir"] = log_dir
    # Proxy runs get their own report dir so normal RMIA results are not overwritten.
    # Online runs also get their own dir so offline results are not overwritten.
    # Ref-sweep runs get their own dir (report_ref{k}) to avoid overwriting the baseline.
    online_attack = configs["audit"].get("online_attack", False)
    report_suffix = "_online" if online_attack else ""
    if num_ref_override is not None:
        report_suffix += f"_ref{num_ref_override}"
    baseline_report_dir = os.path.join(log_dir, f"report{report_suffix}")
    report_dir = (
        os.path.join(base_log_dir, "rmia_proxy", proxy_model.lower(), f"report{report_suffix}")
        if proxy_model is not None
        else baseline_report_dir
    )
    directories = {
        "log_dir": log_dir,
        "report_dir": report_dir,
        "signal_dir": os.path.join(log_dir, "signals"),
        "data_dir": configs["data"]["data_dir"],
    }
    create_directories(directories)

    # Set up logger — unique name per seed so in-process --seeds loops don't
    # share handler state between seeds.
    _logger_name = f"time_analysis_seed{seed}" if seed is not None else None
    logger = setup_log(
        directories["report_dir"], "time_analysis", configs["run"]["time_log"],
        logger_name=_logger_name,
    )

    start_time = time.time()

    # Load the dataset
    dataset_dir = directories["data_dir"]
    df = load_dataset(dataset_name=dataset_name, base_dir=dataset_dir)
    print(f"Loaded {dataset_name} dataset with shape: {df.shape}")
    X, y = prepare_tabular_arrays(df)
    if seed is not None:
        order = _load_or_create_seed_permutation(model_log_dir, len(y), seed)
        X = X[order]
        y = y[order]
        print(f"[SEED] Applied saved dataset permutation for seed={seed}")
    else:
        order = np.arange(len(y))
    training_size = int(len(y) * 0.75)  # Splitting to create a population dataset

    # context_pct controls what fraction of the training pool is given to the model.
    # The population always uses the held-out 25% so the reference distribution is fixed.
    context_size = max(2, int(training_size * context_pct / 100.0))
    if context_pct < 100.0:
        print(f"Context size: {context_size}/{training_size} samples ({context_pct:.1f}% of training pool)")
    if seed is not None:
        split_dir = os.path.join(log_dir, "splits")
        np.save(os.path.join(split_dir, "context_indices_original.npy"), order[:context_size])
        np.save(os.path.join(split_dir, "population_indices_original.npy"), order[training_size:])

    dataset = TabularDataset(X[:context_size], y[:context_size])
    population = TabularDataset(X[training_size:], y[training_size:])

    num_experiments = configs["run"]["num_experiments"]
    num_reference_models = configs["audit"]["num_ref_models"]
    num_model_pairs = max(math.ceil(num_experiments / 2.0), num_reference_models + 1) # 2 model pairs = 4 models

    # train models
    baseline_time = time.time()
    if mode == "signal":
        signals_dir = directories['signal_dir']
        if os.path.exists(signals_dir):
            shutil.rmtree(signals_dir)
        os.makedirs(signals_dir, exist_ok=True)

        num_total_pairs = 2 * num_model_pairs
        models_list, memberships = load_models(model_log_dir, dataset, num_total_pairs, configs, logger)
        if models_list is None or memberships is None:
            raise FileNotFoundError(
                f"--mode signal requires pre-trained models in {model_log_dir}/models "
                f"(expected {num_total_pairs} models), but none were found. "
                "Run with --mode train first."
            )
        print(f"[INFO] Models loaded from {model_log_dir}/models — signals deleted, recomputing.")
        logger.info("--mode signal: models loaded, signals will be recomputed.")

    elif mode == "train":
        signals_dir = directories['signal_dir']
        if os.path.exists(signals_dir):
            shutil.rmtree(signals_dir)
        os.makedirs(signals_dir, exist_ok=True)

        _models_dir = os.path.join(log_dir, "models")
        _models_exist = os.path.exists(_models_dir) and bool(os.listdir(_models_dir)) if os.path.exists(_models_dir) else False

        if _models_exist and skip_existing:
            # --skip-existing: models are on disk, reuse them and only recompute signals.
            models_list, memberships = load_models(log_dir, dataset, 2 * num_model_pairs, configs, logger)
            print(f"[INFO] --skip-existing: models found in {_models_dir} — skipping training, recomputing signals.")
            logger.info("--skip-existing: reusing existing models, skipping training.")
        else:
            # Default: always train from scratch (delete any existing models first).
            if _models_exist:
                shutil.rmtree(_models_dir)
                logger.info("--mode train: deleted existing models in %s — retraining from scratch.", _models_dir)

            # Split dataset for training two models per pair
            import inspect as _inspect
            _sdt_params = _inspect.signature(split_dataset_for_training).parameters
            _split_kwargs = {}
            if 'labels' in _sdt_params:
                _split_kwargs['labels'] = dataset.targets
            if 'random_seed' in _sdt_params:
                _split_kwargs['random_seed'] = seed
            data_splits, memberships = split_dataset_for_training(
                len(dataset), num_model_pairs,
                **_split_kwargs
            )
            if seed is not None:
                split_dir = os.path.join(log_dir, "splits")
                os.makedirs(split_dir, exist_ok=True)
                np.save(os.path.join(split_dir, "memberships.npy"), memberships)
                np.savez(
                    os.path.join(split_dir, "model_pair_splits.npz"),
                    **{f"model_{idx}_train_indices_context": split_info["train"] for idx, split_info in enumerate(data_splits)},
                    **{f"model_{idx}_test_indices_context": split_info["test"] for idx, split_info in enumerate(data_splits)},
                )
            # Create auditing_dataset now (memberships known) so signals can be computed
            # immediately after each model is trained, freeing model RAM on the fly.
            auditing_dataset, auditing_membership = sample_auditing_dataset(configs, dataset, logger, memberships)
            _algo = configs["audit"]["algorithm"].lower()
            _sig_path = os.path.join(signals_dir, f"{_algo}_signals.npy")
            _pop_path = os.path.join(signals_dir, f"{_algo}_signals_pop.npy")
            _sigs, _pop_sigs = [], []

            def _on_model_trained(model, model_idx):
                _sigs.append(compute_signal_one_model(model, auditing_dataset, configs, logger))
                _pop_sigs.append(compute_signal_one_model(model, population, configs, logger, is_population=True))

            models_list = train_models(
                log_dir, dataset, data_splits, memberships, configs, logger,
                on_model_trained=_on_model_trained,
            )
            import gc as _gc
            np.save(_sig_path, np.concatenate(_sigs, axis=1)); del _sigs; _gc.collect()
            np.save(_pop_path, np.concatenate(_pop_sigs, axis=1)); del _pop_sigs; _gc.collect()
            logger.info("Signals computed and saved during training.")
            logger.info(
                "Model training + signals took %0.1f seconds", time.time() - baseline_time
            )

    else:
        num_total_pairs = 2 * num_model_pairs
        models_list, memberships = load_models(
            model_log_dir, dataset, num_total_pairs, configs, logger
        )
        if models_list is None or memberships is None:
            raise FileNotFoundError(
                f"Could not load models from {model_log_dir}/models. "
                f"Expected at least {num_total_pairs} models. "
                "Run with --mode train first."
            )

    print(f"Script finished in {time.time() - start_time:.2f} seconds.")

    # Auditing dataset — already created during interleaved training; compute for other modes.
    if "auditing_dataset" not in locals():
        auditing_dataset, auditing_membership = sample_auditing_dataset(
            configs, dataset, logger, memberships
        )

    # compute signals — check what already exists to avoid loading models unnecessarily
    baseline_time = time.time()
    algo = configs["audit"]["algorithm"].lower()
    signal_dir = os.path.join(configs["run"]["log_dir"], "signals")
    sig_path = os.path.join(signal_dir, f"{algo}_signals.npy")
    pop_path = os.path.join(signal_dir, f"{algo}_signals_pop.npy")
    sig_exists = os.path.exists(sig_path)
    pop_exists = os.path.exists(pop_path)

    if sig_exists and not pop_exists:
        logger.info("Auditing signals found; computing only population signals.")
        signals = get_model_signals(models_list, auditing_dataset, configs, logger)
        del models_list  # free RAM before population inference
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        models_list_pop, _ = load_models(model_log_dir, dataset, 2 * num_model_pairs, configs, logger)
        population_signals = get_model_signals(models_list_pop, population, configs, logger, is_population=True)
        del models_list_pop; gc.collect()
    else:
        logger.info("Preparing signals for auditing dataset")
        signals = get_model_signals(models_list, auditing_dataset, configs, logger)
        logger.info("Preparing population signals")
        population_signals = get_model_signals(models_list, population, configs, logger, is_population=True)
    logger.info("Preparing signals took %0.5f seconds", time.time() - baseline_time)

    # If a proxy model is specified, replace reference columns with its pre-computed signals.
    if proxy_model is not None:
        print(f"[INFO] Using '{proxy_model}' as proxy/reference model for target '{model_name_for_logs}'.")
        proxy_sigs, proxy_pop_sigs, proxy_mems = load_proxy_signals(dataset_name, proxy_model, context_pct)
        signals, population_signals, auditing_membership = build_proxy_combined_signals(
            signals, population_signals, auditing_membership,
            proxy_sigs, proxy_pop_sigs, proxy_mems,
        )
        logger.info("Proxy signals combined: signals=%s, pop=%s, memberships=%s",
                    signals.shape, population_signals.shape, auditing_membership.shape)

    # Perform the privacy audit
    baseline_time = time.time()
    target_model_indices = list(range(num_experiments)) # --> [0]

    mia_score_list, membership_list = audit_models(
            f"{directories['report_dir']}/exp",
            target_model_indices,
            signals,
            population_signals,
            auditing_membership,
            num_reference_models,
            logger,
            configs,
        )

    if len(target_model_indices) > 1:
        logger.info(
            "Auditing privacy risk took %0.1f seconds", time.time() - baseline_time
        )

    # Get average audit results across all experiments
    if len(target_model_indices) > 1:
        get_average_audit_results(
            directories["report_dir"], mia_score_list, membership_list, logger
        )

    logger.info("Total runtime: %0.5f seconds", time.time() - start_time)
    return directories["report_dir"]



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ML Privacy Meter experiments.")
    parser.add_argument("--dataset", type=str, default="locations", help="Dataset name to use")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "load", "signal"],
                        help="train: train models + compute signals; load: load models + load signals; signal: load models + recompute signals (no training).")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for config generation. If omitted, a missing config is auto-created with model='mlp'.",
    )
    parser.add_argument(
        "--context-pct",
        type=float,
        default=100.0,
        help="Percentage of the training pool to use as context (1-100). Default 100 uses all.",
    )
    parser.add_argument(
        "--gpu", "--gpus",
        dest="gpu",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to expose (e.g. '0' or '0,1'). Sets CUDA_VISIBLE_DEVICES and overrides the YAML device to cuda:0.",
    )
    parser.add_argument(
        "--proxy-model",
        type=str,
        default=None,
        help=(
            "Use a different model's pre-trained signals as reference models in RMIA. "
            "E.g. --model lightgbm --proxy-model tabpfn uses lightgbm as target (col 0) "
            "and tabpfn's signals as references (cols 1+). Requires --mode audit."
        ),
    )
    parser.add_argument(
        "--skip-config",
        action="store_true",
        help="Skip rewriting the YAML config if it already exists.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Run RMIA in online mode: P(x) = (P_in + P_out) / 2. Reuses existing signals; results go to report_online/.",
    )
    parser.add_argument(
        "--num-ref",
        type=int,
        default=None,
        help=(
            "Override num_ref_models from the config. Results go to report_ref{N}/ so the "
            "baseline report is not overwritten. Pair this with --mode load to reuse "
            "existing models and cached signals without retraining."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Run the default RMIA case with one explicit random seed. "
            "Results go to seed<seed>/rmia/."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "Comma-separated random seeds for repeated default RMIA runs, e.g. "
            "'12345,23456,34567,45678,56789'. Writes mean/std CSVs under the model directory."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "In --mode train: if models are already on disk, skip retraining and only recompute signals. "
            "Without this flag, --mode train always deletes existing models and trains from scratch."
        ),
    )
    args = parser.parse_args()

    # --online reuses existing models/signals — force load mode unless the user
    # explicitly asked for train (which would wipe the signals directory).
    if args.online and args.mode == "train":
        args.mode = "load"

    # Must be set before any CUDA/torch initialization.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    ensure_dataset_ready(dataset_name=args.dataset, model_name=args.model, algorithm="RMIA", skip_if_exists=args.skip_config)

    # Add the tool directory to sys.path
    sys.path.append(str(Path(__file__).parent.parent / "ml_privacy_meter"))

    from dataset import TabularDataset
    from audit import get_average_audit_results, audit_models, sample_auditing_dataset
    from get_signals import get_model_signals, compute_signal_one_model
    from models.utils import load_models, train_models, split_dataset_for_training
    from util import check_configs, setup_log, initialize_seeds, create_directories

    _model_name = args.model or "mlp"

    try:
        if args.seeds is not None:
            if args.proxy_model is not None:
                raise ValueError("--seeds is not supported with --proxy-model.")
            seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
            if not seeds:
                raise ValueError("--seeds must contain at least one integer seed.")
            report_dirs = []
            for seed in seeds:
                print(f"[SEED] Running RMIA seed={seed}")
                report_dirs.append(
                    main(
                        dataset_name=args.dataset,
                        mode=args.mode,
                        context_pct=args.context_pct,
                        gpu=args.gpu,
                        model_name=args.model,
                        proxy_model=args.proxy_model,
                        online=args.online,
                        num_ref_override=args.num_ref,
                        seed=seed,
                        skip_existing=args.skip_existing,
                    )
                )
                # Explicitly close the seed-specific logger so its file handler
                # is released before the next seed opens its own log file.
                import logging as _logging
                _sl = _logging.getLogger(f"time_analysis_seed{seed}")
                for _h in list(_sl.handlers):
                    _sl.removeHandler(_h)
                    _h.close()
            summary_dir = os.path.join(
                "ml_privacy_meter",
                "logs",
                args.dataset,
                _model_name.lower(),
            )
            summarize_seed_results(report_dirs, seeds, summary_dir, online=args.online,
                                   context_pct=args.context_pct)
            print(f"[SEED] Wrote seed summaries to {summary_dir}")
        else:
            if args.seed is not None and args.proxy_model is not None:
                raise ValueError("--seed is not supported together with --proxy-model.")
            main(
                dataset_name=args.dataset,
                mode=args.mode,
                context_pct=args.context_pct,
                gpu=args.gpu,
                model_name=args.model,
                proxy_model=args.proxy_model,
                online=args.online,
                num_ref_override=args.num_ref,
                seed=args.seed,
                skip_existing=args.skip_existing,
            )
    except Exception as e:
        import traceback
        _failed_dir = Path("results_visualizations")
        _failed_dir.mkdir(parents=True, exist_ok=True)
        _failed_file = _failed_dir / "rmia_failed_runs.csv"
        # Append one row: dataset, model, error
        header = not _failed_file.exists()
        with _failed_file.open("a") as _fh:
            if header:
                _fh.write("dataset_name,model_name,error\n")
            _error_msg = str(e).replace(",", ";").replace("\n", " ")
            _fh.write(f"{args.dataset},{_model_name},{_error_msg}\n")
        traceback.print_exc()
        print(f"[FAILED] {args.dataset} + {_model_name}: {e}")
        raise SystemExit(1)
