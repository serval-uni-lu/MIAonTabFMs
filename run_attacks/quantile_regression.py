import shutil
import sys
import os
import math
import yaml
import pandas as pd
import time
import numpy as np
import torch
from pathlib import Path

#import polars as pl
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs import ensure_dataset_ready


DATA_DIR_CANDIDATES = [
    Path("data/original"),
    Path("data/data_tabarena"),
]


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



def f_score(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    return np.log(prob) - np.log(1 - prob)
    

import optuna
from catboost import CatBoostRegressor
from catboost.metrics import RMSEWithUncertainty
from sklearn.model_selection import train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_qmia_attack(
    target_signals: np.ndarray,
    population_signals: np.ndarray,
    X_audit,
    X_population,
    y_population,
    seed: int,
    n_trials: int,
    n_jobs: int,
    save_dir: str = None,
    model_idx: int = 0):

    rng = np.random.RandomState(seed)

    # 1. transform signals into scores (for training the quantile regression model)
    #    - for each sample, compute the logit of the probability and then flip the sig based on the true label
    #   - get_model_signals should already be returning the true-class probability, so we can directly apply the logit transformation to get the f_score
    y_score_population = f_score(population_signals)
    y_score_audit = f_score(target_signals)

    # 2. train quantile regression catboost model (on the population set)
    #    - define the objective function for optuna
    def objective(trial):
        param = {
            "depth": trial.suggest_int("depth", 1, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 1e4, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1, log=True),
            "iterations": trial.suggest_int("iterations", 1, 1000, log=True),
        }

        param["thread_count"] = 1
        param["objective"] = "RMSEWithUncertainty"
        param["posterior_sampling"] = True
        param["random_seed"] = seed
        param["allow_writing_files"] = False
        eval_metric = RMSEWithUncertainty()

        # Split the data randomly into a training set for the quantile regression,
        # and a validation set for evaluation. The performance of the quantile regression model
        # on the validation set is reported to Optuna.
        _X_train, _X_valid, _y_train, _y_valid = train_test_split(
            X_population,
            y_score_population,
            test_size=0.2,
            random_state=rng.randint(0, 1000),
            stratify=y_population,
        )
        clf = CatBoostRegressor(**param)
        try:
            clf.fit(_X_train, _y_train, verbose=0)
            _y_pred_valid = clf.predict(_X_valid, prediction_type="RawFormulaVal")
            score = eval_metric.eval(label=_y_valid.T, approx=_y_pred_valid.T)
            return score
        except Exception as _e:
            print(f"[Optuna trial failed] {_e}")
            return np.inf

    # Create a study for tuning hyperparameters
    study = optuna.create_study(
        direction="minimize", sampler=None, pruner=optuna.pruners.HyperbandPruner
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    if study.best_value == np.inf:
        raise RuntimeError("All Optuna trials failed — check the '[Optuna trial failed]' lines above.")

    if save_dir is not None:
        import json
        os.makedirs(save_dir, exist_ok=True)
        params_path = os.path.join(save_dir, f"best_params_{model_idx}.json")
        with open(params_path, "w") as f:
            json.dump(study.best_trial.params, f, indent=2)

    # Define the objective function that trains a quantile regression model with the best hyperparameters
    def detailed_objective(trial):
        param = {
            "depth": trial.suggest_int("depth", 1, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 1e4, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1, log=True),
            "iterations": trial.suggest_int("iterations", 1, 1000, log=True),
        }

        param["thread_count"] = 1
        param["objective"] = "RMSEWithUncertainty"
        param["posterior_sampling"] = True
        param["random_seed"] = seed
        param["allow_writing_files"] = False

        clf = CatBoostRegressor(**param)
        clf.fit(X_population, y_score_population, verbose=0)

        conf_test = clf.predict(X_audit, prediction_type="RawFormulaVal")

        return conf_test

    y_conf = detailed_objective(study.best_trial)

    mu = y_conf[:, 0]
    sigma = np.exp(np.clip(y_conf[:, 1] / 2, -10, 10))

    from scipy.stats import norm as _norm
    quantile_threshold = mu + sigma * _norm.ppf(0.95)
    return y_score_audit - quantile_threshold



def main(dataset_name: str = "locations", mode: str = "load", gpu: str = None, model_name: str = None, defense: str = "none", seed: int = None):
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

    # If a GPU was requested, override the YAML device fields so all trainers
    # and auditing code use cuda:0 (the first visible GPU after CUDA_VISIBLE_DEVICES is set).
    if gpu is not None:
        configs.setdefault("train", {})["device"] = "cuda:0"
        configs.setdefault("audit", {})["device"] = "cuda:0"

    # Validate configurations
    check_configs(configs)

    # Initialize seeds for reproducibility
    initialize_seeds(configs["run"]["random_seed"])

    # Quantile regression logs to: ml_privacy_meter/logs/<dataset>/<model>/quantile_reg/
    model_name_for_logs = str(configs["train"]["model_name"]).lower()
    base_log_dir = os.path.join("ml_privacy_meter", "logs", dataset_name, model_name_for_logs)
    if seed is not None:
        base_log_dir = os.path.join(base_log_dir, f"seed{seed}")
    log_dir = os.path.join(base_log_dir, "quantile_reg")
    rmia_log_dir = os.path.join(base_log_dir, "rmia")
    configs["run"]["log_dir"] = log_dir
    report_dir = (
        os.path.join(log_dir, f"defense_{defense}", "report")
        if defense != "none"
        else os.path.join(log_dir, "report")
    )
    directories = {
        "log_dir": log_dir,
        "report_dir": report_dir,
        "signal_dir": os.path.join(log_dir, "signals"),
        "data_dir": configs["data"]["data_dir"],
    }
    create_directories(directories)

    # Set up logger
    logger = setup_log(
        directories["report_dir"], "time_analysis", configs["run"]["time_log"]
    )

    start_time = time.time()

    # Load the dataset
    dataset_dir = directories["data_dir"]
    df = load_dataset(dataset_name=dataset_name, base_dir=dataset_dir)
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
    training_size = int(len(y) * 0.75)  # Splitting to create a population dataset

    dataset = TabularDataset(X[:training_size], y[:training_size])
    population = TabularDataset(X[training_size:], y[training_size:])

    max_audit_samples = configs["data"].get("max_audit_samples", None)
    _trim_idx = None
    if max_audit_samples is not None and len(dataset) > max_audit_samples:
        rng = np.random.default_rng(configs["run"]["random_seed"])
        _trim_idx = np.sort(rng.choice(len(dataset), size=max_audit_samples, replace=False))
        dataset = TabularDataset(dataset.data[_trim_idx], dataset.targets[_trim_idx])


    num_experiments = configs["run"]["num_experiments"]
    num_model_pairs = max(math.ceil(num_experiments / 2.0), 1)

    baseline_time = time.time()
    if mode == "signal":
        signals_dir = directories["signal_dir"]
        if os.path.exists(signals_dir):
            shutil.rmtree(signals_dir)
        os.makedirs(signals_dir, exist_ok=True)

    models_list, memberships = load_models(rmia_log_dir, dataset, num_model_pairs, configs, logger)
    if models_list is None or memberships is None:
        raise FileNotFoundError(
            f"Could not load RMIA models from {rmia_log_dir}/models. Run RMIA first."
        )
    if _trim_idx is not None and memberships.shape[1] != len(dataset):
        memberships = memberships[:, _trim_idx]

    if defense != "none":
        from run_defenses.hamp_inference import make_hamp_wrapper, log_defense_accuracy
        configs["run"]["log_dir"] = os.path.join(log_dir, f"defense_{defense}")
        orig_target = models_list[0]
        models_list = [make_hamp_wrapper(models_list[0], dataset.data, model_name_for_logs)] + models_list[1:]
        log_defense_accuracy(orig_target, models_list[0],
                             X[training_size:], y[training_size:],
                             directories["report_dir"], logger)

    print(f"Script finished in {time.time() - start_time:.2f} seconds.")

    # Auditing dataset
    auditing_dataset, auditing_membership = sample_auditing_dataset(
        configs, dataset, logger, memberships
    )

    # Extract raw feature arrays aligned with computed signals
    X_population_array = population.data
    y_population_array = population.targets.astype(np.float64)

    if hasattr(auditing_dataset, 'indices'):
        X_audit_array = auditing_dataset.dataset.data[auditing_dataset.indices]
    else:
        X_audit_array = auditing_dataset.data

    # compute signals — check what already exists to avoid loading models unnecessarily
    baseline_time = time.time()
    algo = configs["audit"]["algorithm"].lower()
    signal_dir = directories["signal_dir"]
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
        models_list_pop, _ = load_models(log_dir, dataset, 2, configs, logger)
        population_signals = get_model_signals(models_list_pop, population, configs, logger, is_population=True)
        del models_list_pop; gc.collect()
    else:
        logger.info("Preparing signals for auditing dataset")
        signals = get_model_signals(models_list, auditing_dataset, configs, logger)
        logger.info("Preparing population signals")
        population_signals = get_model_signals(models_list, population, configs, logger, is_population=True)
    logger.info("Preparing signals took %0.5f seconds", time.time() - baseline_time)

    
    # Perform the privacy audit
    baseline_time = time.time()
    target_model_indices = list(range(num_experiments)) # --> [0]

    mia_score_list, membership_list = [], []

    for target_model_idx in target_model_indices:
        logger.info("Running QMIA on target model %d", target_model_idx)
        mia_scores = run_qmia_attack(
            target_signals=signals[:, target_model_idx],
            population_signals=population_signals[:, target_model_idx],
            X_audit=X_audit_array,
            X_population=X_population_array,
            y_population=y_population_array,
            seed=configs["run"]["random_seed"],
            n_trials=configs["audit"].get("qmia_n_trials", 200),
            n_jobs=configs["audit"].get("qmia_n_jobs", 30),
            save_dir=directories["signal_dir"],
            model_idx=target_model_idx,
        )

        if len(target_model_indices) > 1:
            logger.info(
                "Auditing privacy risk took %0.1f seconds", time.time() - baseline_time
            )

        target_memberships = auditing_membership[target_model_idx]
        mia_score_list.append(mia_scores.copy())
        membership_list.append(target_memberships.copy())

        get_audit_results(f"{directories['report_dir']}/exp", target_model_idx, mia_scores, target_memberships, logger,)
        logger.info("Total runtime: %0.5f seconds", time.time() - start_time)

    # Get average audit results across all experiments
    if len(target_model_indices) > 1:
        get_average_audit_results(
            directories["report_dir"], mia_score_list, membership_list, logger
        )




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run QMIA attack on a single dataset/model.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--model", type=str, default=None, help="Model name.")
    parser.add_argument("--mode", type=str, default="load", choices=["signal", "load"],
                        help="'signal' recomputes signals from RMIA models; 'load' reuses them.")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU IDs to expose via CUDA_VISIBLE_DEVICES (e.g. '0' or '0,1').")

    parser.add_argument("--skip-config", action="store_true",
                        help="Skip rewriting the YAML config if it already exists.")
    parser.add_argument("--defense", type=str, default="none", choices=["none", "hamp"],
                        help="Apply a test-time defense before computing attack signals.")
    parser.add_argument("--max-audit-samples", type=int, default=None,
                        help="Cap the audit dataset to this many samples.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed matching a prior RMIA seeded run. Reads models and signals from seed<seed>/ subdirectory.")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    sys.path.append(str(Path(__file__).parent.parent / "ml_privacy_meter"))

    from dataset import TabularDataset
    from audit import get_average_audit_results, sample_auditing_dataset, get_audit_results
    from get_signals import get_model_signals
    from models.utils import load_models
    from util import check_configs, setup_log, initialize_seeds, create_directories

    ensure_dataset_ready(dataset_name=args.dataset, model_name=args.model, algorithm="QUANTILE-REG", skip_if_exists=args.skip_config)

    if args.max_audit_samples is not None:
        config_filename = f"{args.dataset}_{args.model}.yaml" if args.model else f"{args.dataset}.yaml"
        config_file = f"ml_privacy_meter/configs/{config_filename}"
        with open(config_file, "r") as f:
            _cfg = yaml.load(f, Loader=yaml.Loader)
        _cfg.setdefault("data", {})["max_audit_samples"] = args.max_audit_samples
        with open(config_file, "w") as f:
            yaml.dump(_cfg, f, sort_keys=False)

    main(dataset_name=args.dataset, mode=args.mode, gpu=args.gpu, model_name=args.model, defense=args.defense, seed=args.seed)
