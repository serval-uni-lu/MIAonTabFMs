import difflib
import os
import re
from typing import Optional, Dict, List

import pandas as pd
import numpy as np
import yaml


def write_config_yaml(
    config_dir: str,
    filename: str = "config.yaml",
    model_name: str = "mlp",
    algorithm: str = "RMIA",
    run_config: Optional[Dict] = None,
    audit_config: Optional[Dict] = None,
    train_config: Optional[Dict] = None,
    data_config: Optional[Dict] = None,
) -> str:
    """
    Writes a YAML configuration file with model-specific training parameters.
    
    Args:
        config_dir (str): Directory to save the YAML file.
        filename (str): Name of the YAML file.
        model_name (str): Model type ("mlp", "lightgbm", "tabpfn", "real-tabpfn").
        run_config, audit_config, train_config, data_config (dict, optional):
            Overrides for sections.
    
    Returns:
        str: Path to the YAML file.
    """
    os.makedirs(config_dir, exist_ok=True)
    filepath = os.path.join(config_dir, filename)
    torch_transformer_models = ["tabnet"]
    nn_models = ["mlp", "cnn", "wide_resnet"]
    ensembling_models = ["rf"]
    boosting_models = ["lightgbm", "xgboost"]
    foundation_models = ["tabpfn", "real-tabpfn", "tabicl", "tabdpt", "tarte"]
    # Default config sections
    audit_device = 0 if model_name.lower() in ["tabpfn", "real-tabpfn", "tabicl", "tabdpt", "tarte"] else "cpu"
    # audit_batch_size = 512 if model_name.lower() in ["tabpfn", "real-tabpfn", "tabicl", "tabdpt", "tarte"] else 5000
    audit_batch_size = 5000

    config = {
        "run": {
            "random_seed": 12345,
            "log_dir": f"ml_privacy_meter/logs/{filename.split('.')[0]}/{model_name}",
            "time_log": True,
            "num_experiments": 1,
        },
        "audit": {
            "privacy_game": "privacy_loss_model",
            "algorithm": algorithm,
            "num_ref_models": 1,
            "device": audit_device,
            "report_log": "report_rmia",
            "batch_size": audit_batch_size,
            "n_jobs": -1,
        },
        "data": {
            "dataset": "locations",
            "data_dir": "data/original",
        },
    }

    # Model-specific defaults
    if model_name.lower() in nn_models:
        model_defaults = {
            "model_name": model_name,
            "device": "cpu",
            "batch_size": 256,
            "optimizer": "SGD",
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "epochs": 100,
            "hyperparameter_tuning": True,
            "tuning_n_trials": 30,
            "tuning_cv": 3,
        }
    elif model_name.lower() in boosting_models:
        model_defaults = {
            "model_name": model_name,
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 5,
            "random_state": 42,
            "n_jobs": -1,
            "hyperparameter_tuning": True,
            "tuning_n_trials": 30,
            "tuning_cv": 3,
        }
    elif model_name.lower() in ensembling_models:
        model_defaults = {
            "model_name": model_name,
            "n_estimators": 200,
            "max_depth": None,
            "random_state": 42,
            "n_jobs": -1,
            "hyperparameter_tuning": True,
            "tuning_n_trials": 30,
            "tuning_cv": 3,
        }
    elif model_name.lower() in foundation_models:
        model_defaults = {
            "model_name": model_name,
            "device": 0,
        }
    elif model_name.lower() in torch_transformer_models:
        model_defaults = {
            "model_name": model_name,
            "n_d": 32,
            "n_a": 32,
            "n_steps": 3,
            "gamma": 1.3,
            "batch_size": 256,
            "virtual_batch_size": 64,
            "max_epochs": 200,
            "patience": 15,
            "hyperparameter_tuning": True,
            "tuning_n_trials": 30,
            "tuning_cv": 3,
        }
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    config["train"] = model_defaults

    # Merge user overrides
    if isinstance(run_config, dict):
        config["run"].update(run_config)
    if isinstance(audit_config, dict):
        config["audit"].update(audit_config)
    if isinstance(train_config, dict):
        config["train"].update(train_config)
    if isinstance(data_config, dict):
        config["data"].update(data_config)

    # Write YAML
    with open(filepath, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    return filepath


def add_shape(num_features, num_classes, dataset_name):
    """
    Adds new shape to INPUT_OUTPUT_SHAPE in utils.py
    if it is not already present.
    """
    utils_path = "ml_privacy_meter/models/utils.py"
    if not os.path.exists(utils_path):
        print(f"utils.py not found: {utils_path}")
        return

    with open(utils_path, "r") as f:
        content = f.read()

    if dataset_name == 'locations':
        num_classes = num_classes+1

    insert_line = f'    "{dataset_name}": [{num_features}, {num_classes}],'

    # If already exists, update only if the values differ
    existing_pattern = re.compile(rf'"{re.escape(dataset_name)}"\s*:\s*\[(\d+),\s*(\d+)\]')
    m = existing_pattern.search(content)
    if m:
        existing_features, existing_classes = int(m.group(1)), int(m.group(2))
        if existing_features == num_features and existing_classes == num_classes:
            print(f"'{dataset_name}' entry already correct. No changes made.")
            return
        print(f"'{dataset_name}' shape changed: [{existing_features}, {existing_classes}] → [{num_features}, {num_classes}]. Updating.")
        new_content = existing_pattern.sub(f'"{dataset_name}": [{num_features}, {num_classes}]', content, count=1)
        with open(utils_path, "w") as f:
            f.write(new_content)
        print("utils.py updated successfully.\n")
        return

    # New entry — insert at top of dict
    pattern = r"(INPUT_OUTPUT_SHAPE\s*=\s*\{)"
    new_content = re.sub(pattern, lambda m: f"{m.group(1)}\n{insert_line}", content, count=1)
    
    with open(utils_path, "w") as f:
        f.write(new_content)

    print("utils.py updated successfully.\n")


def resolve_dataset_file(dataset_name: str, data_dir: Optional[str] = None) -> str:
    candidates: List[str] = []
    if data_dir:
        candidates.append(data_dir)
    candidates.extend([
        "data/original",
        "data/data_tabarena",
    ])

    # Keep insertion order while avoiding duplicates.
    seen = set()
    deduped_candidates = []
    for cand in candidates:
        if cand not in seen:
            deduped_candidates.append(cand)
            seen.add(cand)

    for base in deduped_candidates:
        file_path = os.path.join(base, f"{dataset_name}.csv")
        if os.path.exists(file_path):
            return file_path

    expected = ", ".join(
        os.path.join(base, f"{dataset_name}.csv") for base in deduped_candidates
    )

    available = []
    for base in deduped_candidates:
        if os.path.isdir(base):
            available.extend(p[:-4] for p in os.listdir(base) if p.endswith(".csv"))

    suggestion = ""
    if available:
        close = difflib.get_close_matches(dataset_name, available, n=1, cutoff=0.6)
        if close:
            suggestion = f" Did you mean '{close[0]}'?"

    raise FileNotFoundError(f"Dataset not found. Looked for: {expected}.{suggestion}")


def process_datashapes(dataset_name, data_dir: Optional[str] = None):
    """
    Registers shape info.
    
    Args:
        dataset_names (list of str): Names of datasets (without .csv extension)
        configs (dict): Configuration dictionary containing 'data' key
    """
    file_path = resolve_dataset_file(dataset_name, data_dir=data_dir)
        
    # Read CSV (datasets in this project are headerless).
    data = pd.read_csv(file_path, header=None)

    # Convert feature columns to numeric where possible, else categorical codes.
    for col in data.columns[:-1]:
        if pd.api.types.is_string_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
            try:
                data[col] = pd.to_numeric(data[col], errors="raise")
            except (ValueError, TypeError):
                data[col] = data[col].astype("category").cat.codes

    # Ensure labels are numeric for downstream tensor conversion.
    label_col = data.columns[-1]
    if pd.api.types.is_string_dtype(data[label_col]) or pd.api.types.is_object_dtype(data[label_col]):
        try:
            data[label_col] = pd.to_numeric(data[label_col], errors="raise")
        except (ValueError, TypeError):
            data[label_col] = data[label_col].astype("category").cat.codes
    
    print(f"{dataset_name} shape after trimming: {data.shape}")

    # Convert to NumPy
    data_np = data.to_numpy()
    y = data_np[:, -1]
    X = data_np[:, :-1].astype(np.float32)

    # Compute shapes
    num_features = X.shape[1]
    num_classes = len(np.unique(y))

    # Pass to add_shape
    add_shape(num_features, num_classes, dataset_name)


def ensure_dataset_ready(
    dataset_name: str,
    model_name: Optional[str] = None,
    algorithm: str = "RMIA",
    data_dir: str = "data/original",
    config_dir: str = "ml_privacy_meter/configs",
    skip_if_exists: bool = False,
) -> str:
    """
    Ensures a dataset can be used by run scripts with minimal manual setup.

    This function:
    1) Validates the dataset CSV exists.
    2) Ensures input/output shape is registered in models/utils.py.
    3) Creates a dataset YAML config if missing, or rewrites it when model_name is provided.
       Always updates the algorithm field to match the calling script.

    Args:
        dataset_name (str): Dataset name (without .csv extension).
        model_name (str, optional): If provided, force-write config using this model.
        algorithm (str): MIA algorithm ("RMIA", "LOSS", "LIRA"). Written to config.
        data_dir (str): Directory containing dataset CSV files.
        config_dir (str): Directory for YAML configs.

    Returns:
        str: Path to the dataset YAML config.
    """
    file_path = resolve_dataset_file(dataset_name, data_dir=data_dir)
    resolved_data_dir = os.path.dirname(file_path)

    process_datashapes(dataset_name, data_dir=resolved_data_dir)

    config_filename = f"{dataset_name}_{model_name}.yaml" if model_name else f"{dataset_name}.yaml"
    config_path = os.path.join(config_dir, config_filename)
    should_write_config = not (skip_if_exists and os.path.exists(config_path))

    if should_write_config:
        selected_model = model_name or "mlp"
        write_config_yaml(
            config_dir=config_dir,
            filename=config_filename,
            model_name=selected_model,
            algorithm=algorithm,
            data_config={"dataset": dataset_name, "data_dir": resolved_data_dir},
        )
        print(
            f"Prepared config for dataset '{dataset_name}' with model '{selected_model}': {config_path}"
        )
    else:
        # Keep existing config but update data_dir and algorithm.
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        changed = False
        cfg.setdefault("data", {})
        if cfg["data"].get("data_dir") != resolved_data_dir:
            cfg["data"]["data_dir"] = resolved_data_dir
            changed = True
        cfg.setdefault("audit", {})
        if cfg["audit"].get("algorithm") != algorithm:
            cfg["audit"]["algorithm"] = algorithm
            changed = True
        if changed:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print(
                f"Updated config for dataset '{dataset_name}' (algorithm={algorithm}, data_dir={resolved_data_dir})"
            )
        else:
            print(f"Using existing config for dataset '{dataset_name}': {config_path}")

    return config_path