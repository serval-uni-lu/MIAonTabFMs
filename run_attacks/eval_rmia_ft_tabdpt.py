#!/usr/bin/env python3
"""RMIA for original and fine-tuned TabDPT using the project column layout.

This script consumes artifacts produced by run_attacks/finetune_tabdpt.py.
Columns follow ml_privacy_meter.attacks.run_rmia exactly for target_model_idx=0:

  col 0: target model
  col 1: paired model, excluded from target references
  col 2: reference pair 0, model A
  col 3: reference pair 0, model B
  col 4: reference pair 1, model A
  col 5: reference pair 1, model B
  ...

This experiment is fixed to one RMIA reference pair, so there are 4 columns:
target, paired, reference_0_a, reference_0_b. Only the target column is replaced
by the fine-tuned TabDPT checkpoint. Paired/reference columns are evaluated with
the original checkpoint and their own RMIA train split as TabDPT context, matching
the standard RMIA reference construction without fine-tuning reference models.

uv run run_attacks/eval_rmia_ft_tabdpt.py --datasets purchases10,46905_Amazon_employee_access


"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
TABDPT_TRAINING = ROOT / "TabDPT-training"
TABDPT_INFERENCE_SRC = ROOT / "TabDPT-inference" / "src"
MIA_METER = ROOT / "ml_privacy_meter"

sys.path.insert(0, str(TABDPT_TRAINING))
sys.path.insert(0, str(TABDPT_INFERENCE_SRC))
sys.path.insert(0, str(MIA_METER))

from attacks import run_population_attack, run_rmia  # noqa: E402
from model import TabDPTModel, pad_x  # noqa: E402
from tabdpt.model import TabDPTModel as PublicTabDPTModel  # noqa: E402

NUM_RMIA_REFERENCE_MODELS = 1


class PublicTabDPTAdapter(torch.nn.Module):
    """Expose public TabDPT with the training-repo forward signature."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.num_features = model.num_features
        self.n_out = model.n_out

    def forward(self, x_src: torch.Tensor, y_src: torch.Tensor) -> torch.Tensor:
        return self.model(
            x_src=x_src.transpose(0, 1),
            y_src=y_src.transpose(0, 1).unsqueeze(-1),
            task="cls",
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="46905_Amazon_employee_access")
    p.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated datasets, e.g. purchases10,46905_Amazon_employee_access.",
    )
    p.add_argument("--data-dir", default="data/data_tabarena")
    p.add_argument(
        "--run-dir",
        default=None,
        help="Directory with finetune_tabdpt.py artifacts. Defaults to ml_privacy_meter/logs/ft_tabdpt/<dataset>.",
    )
    p.add_argument("--out-dir", default=None, help="Output directory. Defaults to <run-dir>/rmia.")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--context-size", type=int, default=0, help="Max context rows passed to the model. 0 (default) uses the full context split. Set to 1024 to match finetune_tabdpt.py eval memory usage.")
    p.add_argument("--eval-batch-size", type=int, default=32, help="Query records per forward pass. Does not affect RMIA scores. Reduce to 16 if OOM on larger datasets.")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument(
        "--feature-reduction",
        choices=["pca", "subsample", "error"],
        default="pca",
        help="How to reduce datasets with more features than the TabDPT checkpoint supports.",
    )
    return p.parse_args()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def resolve_dataset_path(dataset: str, data_dir: str) -> Path:
    candidates = [Path(data_dir), Path("data/original"), Path("data/data_tabarena")]
    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    for base in unique_candidates:
        path = base / f"{dataset}.csv"
        full_path = path if path.is_absolute() else ROOT / path
        if full_path.exists():
            return full_path

    expected = ", ".join(str(ROOT / base / f"{dataset}.csv") for base in unique_candidates)
    raise FileNotFoundError(f"Dataset not found. Looked for: {expected}")


def load_csv_arrays(dataset: str, data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    path = resolve_dataset_path(dataset, data_dir)
    df = pd.read_csv(path, header=None)
    X_df = df.iloc[:, :-1].copy()
    y_raw = df.iloc[:, -1].copy()
    for col in X_df.columns:
        if pd.api.types.is_object_dtype(X_df[col]) or pd.api.types.is_string_dtype(X_df[col]):
            X_df[col] = X_df[col].astype("category").cat.codes
    y = LabelEncoder().fit_transform(y_raw)
    return X_df.to_numpy(dtype=np.float32), y.astype(np.int64)


def preprocess_from_context(X: np.ndarray, context_idx: np.ndarray) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_ctx = imputer.fit_transform(X[context_idx])
    scaler.fit(X_ctx)
    return scaler.transform(imputer.transform(X)).astype(np.float32)


def reduce_features_for_checkpoint(
    X: np.ndarray,
    context_idx: np.ndarray,
    max_features: int,
    method: str,
    seed: int,
) -> np.ndarray:
    if X.shape[1] <= max_features:
        return X
    if method == "error":
        raise ValueError(
            f"Dataset has {X.shape[1]} features, but checkpoint supports {max_features}. "
            "Use --feature-reduction pca or --feature-reduction subsample."
        )

    if method == "pca":
        n_components = min(max_features, X.shape[1], len(context_idx))
        reducer = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
        reducer.fit(X[context_idx])
        X_reduced = reducer.transform(X).astype(np.float32)
        print(f"Reduced features with PCA: {X.shape[1]} -> {X_reduced.shape[1]}")
        return X_reduced

    if method == "subsample":
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(X.shape[1], size=max_features, replace=False))
        print(f"Reduced features by deterministic subsampling: {X.shape[1]} -> {len(selected)}")
        return X[:, selected].astype(np.float32)

    raise ValueError(f"Unknown feature reduction method: {method}")


def clean_state_dict(state: dict) -> dict:
    cleaned = {}
    for key, value in state.items():
        for prefix in ("_orig_mod.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def try_load_safetensors(path: Path, device: str) -> tuple[torch.nn.Module, object, dict] | None:
    try:
        with safe_open(str(path), framework="pt", device=device) as f:
            meta = f.metadata() or {}
            if "cfg" not in meta:
                return None
            cfg = OmegaConf.create(json.loads(meta["cfg"]))
            state = {k: f.get_tensor(k) for k in f.keys()}
    except Exception:
        return None

    cfg.env.device = device
    try:
        model = TabDPTModel.load(model_state=clean_state_dict(state), config=cfg)
        return model.to(device).eval(), cfg, {"source_format": "safetensors", "model_impl": "local_training"}
    except Exception:
        base_model = PublicTabDPTModel.load(
            model_state=state,
            config=cfg,
            use_flash=device.startswith("cuda"),
            clip_sigma=4.0,
        )
        model = PublicTabDPTAdapter(base_model)
        return model.to(device).eval(), cfg, {"source_format": "safetensors", "model_impl": "public_adapter"}


def load_model_and_cfg(ckpt_path: Path, device: str) -> tuple[torch.nn.Module, object, dict]:
    safetensors_loaded = try_load_safetensors(ckpt_path, device)
    if safetensors_loaded is not None:
        model, cfg, stats = safetensors_loaded
        return model.eval(), cfg, stats

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    cfg.env.device = device
    state = clean_state_dict(ckpt["model"])

    try:
        model = TabDPTModel.load(model_state=state, config=cfg)
    except Exception:
        base_model = PublicTabDPTModel.load(
            model_state=state,
            config=cfg,
            use_flash=device.startswith("cuda"),
            clip_sigma=4.0,
        )
        model = PublicTabDPTAdapter(base_model)
    return model.to(device).eval(), cfg, dict(ckpt.get("stats", {}))


@torch.no_grad()
def predict_proba(
    model: torch.nn.Module,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    *,
    num_classes: int,
    context_size: int,
    batch_size: int,
    temperature: float,
    device: str,
    desc: str,
) -> np.ndarray:
    model.eval()
    n_ctx = min(context_size, len(y_context))
    X_ctx = torch.tensor(X_context[:n_ctx], dtype=torch.float32, device=device)
    y_ctx = torch.tensor(y_context[:n_ctx], dtype=torch.float32, device=device)
    X_ctx = pad_x(X_ctx[:, None, :], model.num_features)

    probs = []
    for start in tqdm(range(0, len(X_query), batch_size), desc=desc, leave=True):
        xb = torch.tensor(X_query[start:start + batch_size], dtype=torch.float32, device=device)
        bsz = xb.shape[0]
        x_ctx_b = X_ctx.repeat(1, bsz, 1)
        y_ctx_b = y_ctx[:, None].repeat(1, bsz)
        x_qry = pad_x(xb[None, :, :], model.num_features)
        logits = model(torch.cat([x_ctx_b, x_qry], dim=0), y_ctx_b)
        logits = logits[-1, :, :num_classes] / temperature
        probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def true_label_score(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    return proba[np.arange(len(y)), y]



def load_saved_splits(splits_path: Path, num_reference_models: int) -> dict[str, np.ndarray]:
    splits_npz = np.load(splits_path)
    required = [
        "member_idx_original",
        "nonmember_idx_original",
        "context_idx_original",
        "test_idx_original",
        "paired_idx_original",
    ]
    for ref_id in range(num_reference_models):
        required.extend([
            f"reference_{ref_id}_a_idx_original",
            f"reference_{ref_id}_b_idx_original",
        ])
    missing = [key for key in required if key not in splits_npz]
    if missing:
        raise KeyError(
            "splits.npz is missing RMIA pair split arrays. "
            "Re-run run_attacks/finetune_tabdpt.py after the RMIA split update. "
            f"Missing: {missing}"
        )
    splits = {
        "member": splits_npz["member_idx_original"].astype(int),
        "nonmember": splits_npz["nonmember_idx_original"].astype(int),
        "paired": splits_npz["paired_idx_original"].astype(int),
        "context": splits_npz["context_idx_original"].astype(int),
        "test": splits_npz["test_idx_original"].astype(int),
    }
    for ref_id in range(num_reference_models):
        splits[f"reference_{ref_id}_a"] = splits_npz[f"reference_{ref_id}_a_idx_original"].astype(int)
        splits[f"reference_{ref_id}_b"] = splits_npz[f"reference_{ref_id}_b_idx_original"].astype(int)
    return splits


def safe_attack_metrics(scores: np.ndarray, memberships: np.ndarray) -> dict:
    labels = memberships.astype(int).ravel()
    values = scores.ravel()
    valid = np.isfinite(values)
    labels = labels[valid]
    values = values[valid]
    if len(np.unique(labels)) < 2:
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        roc_auc = 0.5
    else:
        fpr, tpr, _ = roc_curve(labels, values)
        roc_auc = float(auc(fpr, tpr))

    def tpr_at(max_fpr: float) -> float:
        idx = np.where(fpr <= max_fpr)[0]
        return float(tpr[idx[-1]]) if len(idx) else 0.0

    def tnr_at(max_fnr: float) -> float:
        idx = np.where((1.0 - tpr) <= max_fnr)[0]
        return float(1.0 - fpr[idx[0]]) if len(idx) else 0.0

    return {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc,
        "tpr@0.1%fpr": tpr_at(0.001),
        "tpr@0%fpr": tpr_at(0.0),
        "tnr@0.1%fnr": tnr_at(0.001),
        "tnr@0%fnr": tnr_at(0.0),
    }


def save_attack_result(report_dir: Path, mia_scores: np.ndarray, memberships: np.ndarray) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = safe_attack_metrics(mia_scores, memberships)
    np.savez(
        report_dir / "attack_result_0.npz",
        fpr=metrics["fpr"],
        tpr=metrics["tpr"],
        auc=metrics["auc"],
        one_tenth_fpr=metrics["tpr@0.1%fpr"],
        zero_fpr=metrics["tpr@0%fpr"],
        one_tenth_fnr=metrics["tnr@0.1%fnr"],
        zero_fnr=metrics["tnr@0%fnr"],
        scores=mia_scores,
        memberships=memberships,
    )
    return {k: v for k, v in metrics.items() if k not in {"fpr", "tpr"}}




def print_split_summary(
    splits: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    preview: int = 8,
) -> None:
    for name, indices in splits.items():
        labels, counts = np.unique(y[indices], return_counts=True)
        class_counts = ", ".join(f"{int(label)}:{int(count)}" for label, count in zip(labels, counts))
        print(f"  {name:<18} size={len(indices):>6} classes=[{class_counts}] head_original={indices[:preview].tolist()}")


def score_model_column(
    model_name: str,
    model: torch.nn.Module,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_audit: np.ndarray,
    y_audit: np.ndarray,
    X_pop: np.ndarray,
    y_pop: np.ndarray,
    args: argparse.Namespace,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    score_kw = dict(
        num_classes=num_classes,
        context_size=args.context_size if args.context_size > 0 else len(y_context),
        batch_size=args.eval_batch_size,
        temperature=args.temperature,
        device=args.device,
    )
    audit_proba = predict_proba(
        model,
        X_context,
        y_context,
        X_audit,
        desc=f"{model_name}/audit fixed_context",
        **score_kw,
    )
    pop_proba = predict_proba(
        model,
        X_context,
        y_context,
        X_pop,
        desc=f"{model_name}/population fixed_context",
        **score_kw,
    )
    return true_label_score(audit_proba, y_audit), true_label_score(pop_proba, y_pop)


def requested_datasets(args: argparse.Namespace) -> list[str]:
    if not args.datasets:
        return [args.dataset]
    if args.run_dir:
        raise ValueError("--run-dir can only be used with --dataset, not --datasets.")
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if not datasets:
        raise ValueError("--datasets was provided but no dataset names were parsed.")
    return datasets


def paths_for(args: argparse.Namespace, dataset: str) -> tuple[Path, Path]:
    run_dir = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else ROOT / "ml_privacy_meter" / "logs" / "ft_tabdpt" / dataset
    )
    if args.out_dir:
        out_base = Path(args.out_dir).resolve()
        out_dir = out_base / dataset if args.datasets else out_base
    else:
        out_dir = run_dir / "rmia"
    return run_dir, out_dir


def run_dataset(args: argparse.Namespace, dataset: str) -> None:
    cleanup_gpu()
    run_dir, out_dir = paths_for(args, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_ckpt = run_dir / "original_tabdpt.ckpt"
    ft_ckpt = run_dir / "finetuned_tabdpt.ckpt"
    splits_path = run_dir / "splits.npz"
    for path in (orig_ckpt, ft_ckpt, splits_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    splits = load_saved_splits(splits_path, NUM_RMIA_REFERENCE_MODELS)
    member_idx = splits["member"]
    nonmember_idx = splits["nonmember"]
    context_idx = splits["context"]
    test_idx = splits["test"]

    # Load model early only to read num_features for feature reduction.
    _probe_model, _, _ = load_model_and_cfg(orig_ckpt, args.device)
    num_features = _probe_model.num_features
    del _probe_model
    cleanup_gpu()

    X_all, y_all = load_csv_arrays(dataset, args.data_dir)
    X_all = preprocess_from_context(X_all, context_idx)
    X_all = reduce_features_for_checkpoint(
        X_all,
        context_idx,
        num_features,
        args.feature_reduction,
        seed=12345,
    )
    num_classes = int(np.max(y_all)) + 1

    audit_idx = np.concatenate([member_idx, nonmember_idx])
    membership_target = np.concatenate(
        [np.ones(len(member_idx), dtype=bool), np.zeros(len(nonmember_idx), dtype=bool)]
    )
    X_audit = X_all[audit_idx]
    y_audit = y_all[audit_idx]
    X_pop = X_all[test_idx]
    y_pop = y_all[test_idx]
    X_context = X_all[context_idx]
    y_context = y_all[context_idx]

    model_train_splits = [("target", member_idx), ("paired", splits["paired"])]
    for ref_id in range(NUM_RMIA_REFERENCE_MODELS):
        model_train_splits.extend(
            [
                (f"reference_{ref_id}_a", splits[f"reference_{ref_id}_a"]),
                (f"reference_{ref_id}_b", splits[f"reference_{ref_id}_b"]),
            ]
        )
    model_names = [name for name, _ in model_train_splits]

    all_memberships = np.zeros((len(audit_idx), len(model_names)), dtype=bool)
    audit_pos = {idx: pos for pos, idx in enumerate(audit_idx.tolist())}
    for col, (_, train_idx) in enumerate(model_train_splits):
        for idx in train_idx:
            pos = audit_pos.get(int(idx))
            if pos is not None:
                all_memberships[pos, col] = True

    print(f"Dataset     : {args.dataset}")
    print(f"Run dir     : {run_dir}")
    print("Split summary (original indices):")
    print_split_summary(splits, y_all)
    print(f"Audit rows  : {len(audit_idx)} ({len(member_idx)} members / {len(nonmember_idx)} nonmembers)")
    print(f"Population  : {len(test_idx)} rows")
    print(f"Target eval context: {len(context_idx)} rows; used for target column only")
    print("Paired/reference columns use the original checkpoint with their own RMIA train split as context")
    print(f"RMIA columns: {model_names}")
    print(f"References  : {NUM_RMIA_REFERENCE_MODELS} pair ({2 * NUM_RMIA_REFERENCE_MODELS} reference columns)")

    sig_dir = out_dir / "signals"
    models_dir = out_dir / "models"
    sig_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    np.save(models_dir / "memberships.npy", all_memberships)
    np.save(models_dir / "audit_idx_original.npy", audit_idx)
    np.save(models_dir / "test_idx_original.npy", test_idx)
    np.save(models_dir / "eval_context_idx_original.npy", context_idx)

    finetuned_sigs = np.zeros((len(audit_idx), len(model_names)), dtype=np.float64)
    finetuned_pop_sigs = np.zeros((len(test_idx), len(model_names)), dtype=np.float64)

    original_sigs = np.zeros((len(audit_idx), len(model_names)), dtype=np.float64)
    original_pop_sigs = np.zeros((len(test_idx), len(model_names)), dtype=np.float64)

    original_target, _, _ = load_model_and_cfg(orig_ckpt, args.device)
    original_sigs[:, 0], original_pop_sigs[:, 0] = score_model_column(
        "original_target", original_target, X_context, y_context, X_audit, y_audit, X_pop, y_pop, args, num_classes
    )

    for col, (model_name, _) in enumerate(model_train_splits[1:], start=1):
        X_column_context = X_all[model_train_splits[col][1]]
        y_column_context = y_all[model_train_splits[col][1]]
        original_sigs[:, col], original_pop_sigs[:, col] = score_model_column(
            f"original_{model_name}",
            original_target,
            X_column_context,
            y_column_context,
            X_audit,
            y_audit,
            X_pop,
            y_pop,
            args,
            num_classes,
        )
    del original_target
    cleanup_gpu()

    finetuned_target, _, _ = load_model_and_cfg(ft_ckpt, args.device)
    finetuned_sigs[:, 0], finetuned_pop_sigs[:, 0] = score_model_column(
        "finetuned_target", finetuned_target, X_context, y_context, X_audit, y_audit, X_pop, y_pop, args, num_classes
    )
    del finetuned_target
    cleanup_gpu()

    # original_refs variant: finetuned target (col 0), original model for refs (cols 1+)
    finetuned_original_refs_sigs = finetuned_sigs.copy()
    finetuned_original_refs_sigs[:, 1:] = original_sigs[:, 1:]
    finetuned_original_refs_pop_sigs = finetuned_pop_sigs.copy()
    finetuned_original_refs_pop_sigs[:, 1:] = original_pop_sigs[:, 1:]

    # ft_refs variant: finetuned model for all columns
    finetuned_ft_refs_sigs = finetuned_sigs.copy()
    finetuned_ft_refs_pop_sigs = finetuned_pop_sigs.copy()
    finetuned_ref_model, _, _ = load_model_and_cfg(ft_ckpt, args.device)
    for col, (model_name, col_train_idx) in enumerate(model_train_splits[1:], start=1):
        X_column_context = X_all[col_train_idx]
        y_column_context = y_all[col_train_idx]
        finetuned_ft_refs_sigs[:, col], finetuned_ft_refs_pop_sigs[:, col] = score_model_column(
            f"finetuned_{model_name}",
            finetuned_ref_model,
            X_column_context,
            y_column_context,
            X_audit,
            y_audit,
            X_pop,
            y_pop,
            args,
            num_classes,
        )
    del finetuned_ref_model
    cleanup_gpu()

    np.save(sig_dir / "rmia_signals_original.npy", original_sigs)
    np.save(sig_dir / "rmia_signals_original_pop.npy", original_pop_sigs)
    np.save(sig_dir / "rmia_signals_finetuned_original_refs.npy", finetuned_original_refs_sigs)
    np.save(sig_dir / "rmia_signals_finetuned_original_refs_pop.npy", finetuned_original_refs_pop_sigs)
    np.save(sig_dir / "rmia_signals_finetuned_ft_refs.npy", finetuned_ft_refs_sigs)
    np.save(sig_dir / "rmia_signals_finetuned_ft_refs_pop.npy", finetuned_ft_refs_pop_sigs)

    print("\nRunning project RMIA on target + paired + reference-pair signals ...")
    rmia_original = run_rmia(
        0, original_sigs, original_pop_sigs, all_memberships, NUM_RMIA_REFERENCE_MODELS,
    )
    rmia_finetuned_original_refs = run_rmia(
        0, finetuned_original_refs_sigs, finetuned_original_refs_pop_sigs, all_memberships, NUM_RMIA_REFERENCE_MODELS,
    )
    rmia_finetuned_ft_refs = run_rmia(
        0, finetuned_ft_refs_sigs, finetuned_ft_refs_pop_sigs, all_memberships, NUM_RMIA_REFERENCE_MODELS,
    )

    print("Running Attack-P population baseline on target-column signals ...")
    attack_p_original = run_population_attack(original_sigs[:, 0], original_pop_sigs[:, 0])
    # target col 0 is identical in both finetuned variants
    attack_p_finetuned = run_population_attack(finetuned_sigs[:, 0], finetuned_original_refs_pop_sigs[:, 0])

    result_original = save_attack_result(out_dir / "report_original" / "exp", rmia_original, membership_target)
    result_finetuned_original_refs = save_attack_result(
        out_dir / "report_finetuned_original_refs" / "exp", rmia_finetuned_original_refs, membership_target
    )
    result_finetuned_ft_refs = save_attack_result(
        out_dir / "report_finetuned_ft_refs" / "exp", rmia_finetuned_ft_refs, membership_target
    )
    result_attack_p_original = save_attack_result(
        out_dir / "report_attack_p_original" / "exp", attack_p_original, membership_target
    )
    result_attack_p_finetuned = save_attack_result(
        out_dir / "report_attack_p_finetuned" / "exp", attack_p_finetuned, membership_target
    )

    score_columns = {
        "idx_original": audit_idx,
        "member": membership_target.astype(int),
        "rmia_original": rmia_original,
        "rmia_finetuned_original_refs": rmia_finetuned_original_refs,
        "rmia_finetuned_ft_refs": rmia_finetuned_ft_refs,
        "attack_p_original": attack_p_original,
        "attack_p_finetuned": attack_p_finetuned,
        "target_signal_original": original_sigs[:, 0],
        "target_signal_finetuned": finetuned_sigs[:, 0],
        "paired_signal_original": original_sigs[:, 1],
        "paired_signal_finetuned_original_refs": finetuned_original_refs_sigs[:, 1],
        "paired_signal_finetuned_ft_refs": finetuned_ft_refs_sigs[:, 1],
    }
    for col, model_name in enumerate(model_names):
        score_columns[f"{model_name}_member"] = all_memberships[:, col].astype(int)
        if col >= 2:
            score_columns[f"{model_name}_signal_original"] = original_sigs[:, col]
            score_columns[f"{model_name}_signal_finetuned_original_refs"] = finetuned_original_refs_sigs[:, col]
            score_columns[f"{model_name}_signal_finetuned_ft_refs"] = finetuned_ft_refs_sigs[:, col]
    score_columns["reference_signal_original"] = original_sigs[:, 2:].mean(axis=1)
    score_columns["reference_signal_finetuned_original_refs"] = finetuned_original_refs_sigs[:, 2:].mean(axis=1)
    score_columns["reference_signal_finetuned_ft_refs"] = finetuned_ft_refs_sigs[:, 2:].mean(axis=1)
    pd.DataFrame(score_columns).to_csv(out_dir / "rmia_scores.csv", index=False)

    reference_train_files = {
        "paired_idx_original": splits["paired"].tolist(),
    }
    for ref_id in range(NUM_RMIA_REFERENCE_MODELS):
        reference_train_files[f"reference_{ref_id}_a_idx_original"] = splits[f"reference_{ref_id}_a"].tolist()
        reference_train_files[f"reference_{ref_id}_b_idx_original"] = splits[f"reference_{ref_id}_b"].tolist()
    summary = {
        "dataset": args.dataset,
        "run_dir": str(run_dir),
        "note": (
            "Columns follow ml_privacy_meter.attacks.run_rmia: target, paired, then "
            "2*num_reference_models reference columns. Members are the fine-tuning data. "
            "Both original_refs (FT target + original refs) and ft_refs (all FT) are computed."
        ),
        "n_member": int(len(member_idx)),
        "n_nonmember": int(len(nonmember_idx)),
        "n_population": int(len(test_idx)),
        "n_eval_context": int(len(context_idx)),
        "model_columns": model_names,
        "num_reference_models": NUM_RMIA_REFERENCE_MODELS,
        "reference_train_splits": reference_train_files,
        "original_model_rmia": result_original,
        "finetuned_model_rmia_original_refs": result_finetuned_original_refs,
        "finetuned_model_rmia_ft_refs": result_finetuned_ft_refs,
        "original_model_attack_p": result_attack_p_original,
        "finetuned_model_attack_p": result_attack_p_finetuned,
        "files": {
            "memberships": str(models_dir / "memberships.npy"),
            "rmia_scores": str(out_dir / "rmia_scores.csv"),
            "original_signals": str(sig_dir / "rmia_signals_original.npy"),
            "original_population_signals": str(sig_dir / "rmia_signals_original_pop.npy"),
            "finetuned_original_refs_signals": str(sig_dir / "rmia_signals_finetuned_original_refs.npy"),
            "finetuned_original_refs_population_signals": str(sig_dir / "rmia_signals_finetuned_original_refs_pop.npy"),
            "finetuned_ft_refs_signals": str(sig_dir / "rmia_signals_finetuned_ft_refs.npy"),
            "finetuned_ft_refs_population_signals": str(sig_dir / "rmia_signals_finetuned_ft_refs_pop.npy"),
            "original_attack_result": str(out_dir / "report_original" / "exp" / "attack_result_0.npz"),
            "finetuned_original_refs_attack_result": str(out_dir / "report_finetuned_original_refs" / "exp" / "attack_result_0.npz"),
            "finetuned_ft_refs_attack_result": str(out_dir / "report_finetuned_ft_refs" / "exp" / "attack_result_0.npz"),
            "original_attack_p_result": str(out_dir / "report_attack_p_original" / "exp" / "attack_result_0.npz"),
            "finetuned_attack_p_result": str(out_dir / "report_attack_p_finetuned" / "exp" / "attack_result_0.npz"),
        },
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Project Pair-Layout RMIA Results ===")
    print(json.dumps({
        "original_model_rmia": result_original,
        "finetuned_model_rmia_original_refs": result_finetuned_original_refs,
        "finetuned_model_rmia_ft_refs": result_finetuned_ft_refs,
        "original_model_attack_p": result_attack_p_original,
        "finetuned_model_attack_p": result_attack_p_finetuned,
        "saved_to": str(out_dir),
    }, indent=2))


def main() -> None:
    args = parse_args()
    seed_everything(12345)
    for dataset in requested_datasets(args):
        run_dataset(args, dataset)


if __name__ == "__main__":
    main()
