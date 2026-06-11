#!/usr/bin/env python3
"""Simple fine-tuned TabDPT membership-inference experiment.

This is intentionally smaller than the full RMIA/AMIA pipeline:

1. Load one CSV dataset.
2. Split rows into:
   - fine-tuning members
   - MIA nonmembers
   - fixed inference context
   - held-out test data
3. Fine-tune a TabDPT checkpoint on member rows only.
4. Score member and nonmember MIA rows with the same fixed context.
5. Report AUC using true-label probability as the membership score.

Example:
  uv run run_attacks/finetune_tabdpt.py \
    --dataset 46905_Amazon_employee_access 

If --checkpoint is omitted, local TabDPT-inference code is used and the public
TabDPT weights are downloaded/cached from Hugging Face:
Layer6/TabDPT / tabdpt1_1.safetensors.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from safetensors import safe_open
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
TABDPT_TRAINING = ROOT / "TabDPT-training"
TABDPT_INFERENCE_SRC = ROOT / "TabDPT-inference" / "src"
AMIA_DIR = ROOT / "run_attacks" / "amia"
# Keep TabDPT-inference before TabDPT-training so `import tabdpt...`
# resolves to the package in TabDPT-inference/src, not TabDPT-training/tabdpt.py.
sys.path.insert(0, str(TABDPT_TRAINING))
sys.path.insert(0, str(TABDPT_INFERENCE_SRC))
sys.path.insert(0, str(AMIA_DIR))
sys.path.insert(0, str(ROOT / "ml_privacy_meter"))

from model import TabDPTModel, pad_x  # noqa: E402
from amia_tabdpt import SDPACapture  # noqa: E402
from tabdpt.model import TabDPTModel as PublicTabDPTModel  # noqa: E402

# RMIA-style scoring is disabled for now; this experiment currently focuses on AMIA attention.

DEFAULT_TABDPT_REPO = "Layer6/TabDPT"
DEFAULT_TABDPT_WEIGHTS = "tabdpt1_1.safetensors"
NUM_RMIA_REFERENCE_MODELS = 1


def resolve_dataset_path(dataset: str, data_dir: str) -> Path:
    candidates = [Path(data_dir), Path("data/original"), Path("data/data_tabarena")]
    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        key = candidate
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)

    for base in unique_candidates:
        path = base / f"{dataset}.csv"
        full_path = path if path.is_absolute() else ROOT / path
        if full_path.exists():
            return full_path

    expected = ", ".join(str(ROOT / base / f"{dataset}.csv") for base in unique_candidates)
    raise FileNotFoundError(f"Dataset not found. Looked for: {expected}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="46905_Amazon_employee_access")
    parser.add_argument("--data-dir", default="data/data_tabarena")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional path to TabDPT weights. Supports TabDPT-training .ckpt "
            "or public TabDPT .safetensors. If omitted, downloads the public weights."
        ),
    )
    parser.add_argument("--out-dir", default="ml_privacy_meter/logs/ft_tabdpt")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8, help="Number of meta-batches per update.")
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--query-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--feature-reduction",
        choices=["pca", "subsample", "error"],
        default="pca",
        help="How to reduce datasets with more features than the TabDPT checkpoint supports.",
    )
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--skip-attention", action="store_true", help="Skip AMIA-style attention comparison.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_csv_arrays(dataset: str, data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    path = resolve_dataset_path(dataset, data_dir)

    df = pd.read_csv(path, header=None)
    X_df = df.iloc[:, :-1].copy()
    y_raw = df.iloc[:, -1].copy()

    for col in X_df.columns:
        if pd.api.types.is_object_dtype(X_df[col]) or pd.api.types.is_string_dtype(X_df[col]):
            X_df[col] = X_df[col].astype("category").cat.codes

    y = LabelEncoder().fit_transform(y_raw)
    X = X_df.to_numpy(dtype=np.float32)
    return X, y.astype(np.int64)


def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    idx = np.arange(len(y))

    candidate_idx, holdout_idx = train_test_split(
        idx,
        train_size=0.50,
        random_state=seed,
        stratify=y,
    )
    context_idx, population_idx = train_test_split(
        holdout_idx,
        train_size=0.50,
        random_state=seed + 1,
        stratify=y[holdout_idx],
    )
    member_idx, nonmember_idx = train_test_split(
        candidate_idx,
        train_size=0.50,
        random_state=seed + 2,
        stratify=y[candidate_idx],
    )

    splits = {
        "member": member_idx,
        "nonmember": nonmember_idx,
        "context": context_idx,
        "test": population_idx,
    }

    # RMIA model-column layout follows ml_privacy_meter.attacks.run_rmia:
    #   col 0 = target model, trained on member
    #   col 1 = paired model, trained on nonmember and excluded as a reference
    #   col 2.. = num_reference_models pairs. Each pair has complementary
    #              memberships over the audit pool, so every audited sample has
    #              one IN and one OUT reference per pair.
    audit_pool_idx = np.concatenate([member_idx, nonmember_idx])
    reference_train_size = len(member_idx)
    splits["paired"] = nonmember_idx
    for ref_id in range(NUM_RMIA_REFERENCE_MODELS):
        ref_a_idx, ref_b_idx = train_test_split(
            audit_pool_idx,
            train_size=reference_train_size,
            random_state=seed + 100 + ref_id,
            stratify=y[audit_pool_idx],
        )
        splits[f"reference_{ref_id}_a"] = ref_a_idx
        splits[f"reference_{ref_id}_b"] = ref_b_idx

    return splits




def print_split_summary(
    splits: dict[str, np.ndarray],
    y: np.ndarray,
    original_indices: np.ndarray,
    *,
    preview: int = 8,
) -> None:
    for name, indices in splits.items():
        labels, counts = np.unique(y[indices], return_counts=True)
        class_counts = ", ".join(f"{int(label)}:{int(count)}" for label, count in zip(labels, counts))
        head = original_indices[indices[:preview]].tolist()
        print(f"  {name:<18} size={len(indices):>6} classes=[{class_counts}] head_original={head}")


def preprocess(
    X: np.ndarray,
    splits: dict[str, np.ndarray],
) -> tuple[np.ndarray, SimpleImputer, StandardScaler]:
    # Fit preprocessing only on fixed context rows, so MIA rows are not used
    # to define the transform.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_context = imputer.fit_transform(X[splits["context"]])
    scaler.fit(X_context)
    X_proc = scaler.transform(imputer.transform(X)).astype(np.float32)
    return X_proc, imputer, scaler


def reduce_features_for_checkpoint(
    X: np.ndarray,
    splits: dict[str, np.ndarray],
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
        n_components = min(max_features, X.shape[1], len(splits["context"]))
        reducer = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
        reducer.fit(X[splits["context"]])
        X_reduced = reducer.transform(X).astype(np.float32)
        print(f"Reduced features with PCA: {X.shape[1]} -> {X_reduced.shape[1]}")
        return X_reduced

    if method == "subsample":
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(X.shape[1], size=max_features, replace=False))
        print(f"Reduced features by deterministic subsampling: {X.shape[1]} -> {len(selected)}")
        return X[:, selected].astype(np.float32)

    raise ValueError(f"Unknown feature reduction method: {method}")


class PublicTabDPTAdapter(torch.nn.Module):
    """Expose public TabDPT with the training-repo forward signature."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.num_features = model.num_features
        self.n_out = model.n_out

    def forward(self, x_src: torch.Tensor, y_src: torch.Tensor) -> torch.Tensor:
        # Local fine-tuning code uses [seq, batch, features] and [ctx, batch].
        # Public TabDPT uses [batch, seq, features] and [batch, ctx, 1].
        return self.model(
            x_src=x_src.transpose(0, 1),
            y_src=y_src.transpose(0, 1).unsqueeze(-1),
            task="cls",
        )


def resolve_checkpoint_path(checkpoint_path: str | None) -> str:
    if checkpoint_path is not None:
        return checkpoint_path
    print(
        "No --checkpoint provided; downloading/using cached public TabDPT weights "
        f"({DEFAULT_TABDPT_REPO}/{DEFAULT_TABDPT_WEIGHTS})."
    )
    return hf_hub_download(repo_id=DEFAULT_TABDPT_REPO, filename=DEFAULT_TABDPT_WEIGHTS)


def load_tabdpt(checkpoint_path: str | None, device: str) -> tuple[torch.nn.Module, dict, str]:
    resolved_path = resolve_checkpoint_path(checkpoint_path)
    suffix = Path(resolved_path).suffix.lower()

    if suffix == ".safetensors":
        with safe_open(resolved_path, framework="pt", device=device) as f:
            meta = f.metadata()
            cfg = OmegaConf.create(json.loads(meta["cfg"]))
            model_state = {k: f.get_tensor(k) for k in f.keys()}
        cfg.env.device = device
        base_model = PublicTabDPTModel.load(
            model_state=model_state,
            config=cfg,
            use_flash=device.startswith("cuda"),
            clip_sigma=4.0,
        )
        model = PublicTabDPTAdapter(base_model).to(device)
        checkpoint = {"model": model_state, "cfg": cfg, "stats": {"source_format": "safetensors"}}
        return model, checkpoint, resolved_path

    checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)
    cfg = checkpoint["cfg"]
    cfg.env.device = device
    model = TabDPTModel.load(model_state=checkpoint["model"], config=cfg)
    return model.to(device), checkpoint, resolved_path


def maybe_freeze_encoder(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.startswith("encoder.") or name.startswith("y_encoder."):
            param.requires_grad = False


def sample_meta_batch(
    X_member: np.ndarray,
    y_member: np.ndarray,
    *,
    context_size: int,
    query_size: int,
    batch_size: int,
    max_features: int,
    device: str,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(y_member)
    seq_len = context_size + query_size
    replace = n < seq_len

    x_cols = []
    y_ctx_cols = []
    y_query_cols = []
    for _ in range(batch_size):
        chosen = rng.choice(n, size=seq_len, replace=replace)
        ctx_idx = chosen[:context_size]
        qry_idx = chosen[context_size:]
        x_seq = np.concatenate([X_member[ctx_idx], X_member[qry_idx]], axis=0)
        x_cols.append(x_seq)
        y_ctx_cols.append(y_member[ctx_idx])
        y_query_cols.append(y_member[qry_idx])

    x = torch.tensor(np.stack(x_cols, axis=1), dtype=torch.float32, device=device)
    y_ctx = torch.tensor(np.stack(y_ctx_cols, axis=1), dtype=torch.float32, device=device)
    y_query = torch.tensor(np.stack(y_query_cols, axis=1), dtype=torch.long, device=device)
    x = pad_x(x, max_features)
    return x, y_ctx, y_query


def finetune(
    model: TabDPTModel,
    X_member: np.ndarray,
    y_member: np.ndarray,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(args.seed)
    num_classes = int(np.max(y_member)) + 1
    model.train()

    for epoch in range(1, args.epochs + 1):
        progress = tqdm(
            range(args.steps_per_epoch),
            desc=f"fine-tune epoch {epoch}/{args.epochs}",
            leave=True,
        )
        running = 0.0
        for step in progress:
            x, y_ctx, y_query = sample_meta_batch(
                X_member,
                y_member,
                context_size=args.context_size,
                query_size=args.query_size,
                batch_size=args.batch_size,
                max_features=model.num_features,
                device=args.device,
                rng=rng,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, y_ctx)[..., :num_classes].float()
            loss = F.cross_entropy(
                logits.reshape(-1, num_classes),
                y_query.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running = 0.95 * running + 0.05 * float(loss.detach().cpu()) if step else float(loss.detach().cpu())
            progress.set_postfix(loss=f"{running:.4f}")
    return optimizer


def save_split_files(
    out_dir: Path,
    splits: dict[str, np.ndarray],
    original_indices: np.ndarray,
) -> None:
    arrays = {"subsample_original_indices": original_indices}
    for name, indices in splits.items():
        arrays[f"{name}_idx_local"] = indices
        arrays[f"{name}_idx_original"] = original_indices[indices]
    np.savez(out_dir / "splits.npz", **arrays)


def save_finetuned_checkpoint(
    model: TabDPTModel,
    optimizer: torch.optim.Optimizer,
    source_checkpoint: dict,
    args: argparse.Namespace,
    out_dir: Path,
    splits: dict[str, np.ndarray],
    original_indices: np.ndarray,
) -> Path:
    cfg = source_checkpoint["cfg"]
    stats = dict(source_checkpoint.get("stats", {}))
    stats.update(
        {
            "fine_tuned_by": "run_attacks/finetune_tabdpt.py",
            "source_checkpoint": args.checkpoint or f"{DEFAULT_TABDPT_REPO}/{DEFAULT_TABDPT_WEIGHTS}",
            "dataset": args.dataset,
            "seed": args.seed,
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size,
            "context_size": args.context_size,
            "query_size": args.query_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "split_sizes": {name: int(len(indices)) for name, indices in splits.items()},
        }
    )
    ckpt = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "cfg": cfg,
        "stats": stats,
        "mia_split_indices_local": {name: indices for name, indices in splits.items()},
        "mia_split_indices_original": {
            name: original_indices[indices] for name, indices in splits.items()
        },
    }
    ckpt_path = out_dir / "finetuned_tabdpt.ckpt"
    torch.save(ckpt, ckpt_path)
    return ckpt_path



@torch.no_grad()
def predict_proba_fixed_context(
    model: TabDPTModel,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    *,
    num_classes: int,
    context_size: int,
    batch_size: int,
    temperature: float,
    device: str,
    desc: str = "queries",
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
        x_context_batch = X_ctx.repeat(1, bsz, 1)
        y_context_batch = y_ctx[:, None].repeat(1, bsz)
        x_eval = pad_x(xb[None, :, :], model.num_features)
        logits = model(torch.cat([x_context_batch, x_eval], dim=0), y_context_batch)
        logits = logits[-1, :, :num_classes] / temperature
        probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0)


@torch.no_grad()
def extract_attention_fixed_context(
    model: TabDPTModel,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    *,
    context_size: int,
    batch_size: int,
    device: str,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AMIA-style query-to-context attention signals using SDPACapture."""
    model.eval()
    n_ctx = min(context_size, len(y_context))
    X_ctx = torch.tensor(X_context[:n_ctx], dtype=torch.float32, device=device)
    y_ctx = torch.tensor(y_context[:n_ctx], dtype=torch.float32, device=device)
    X_ctx = pad_x(X_ctx[:, None, :], model.num_features)

    row_max_batches = []
    row_ent_batches = []
    row_arg_batches = []
    n_calls_ref = None

    for start in tqdm(range(0, len(X_query), batch_size), desc=desc, leave=True):
        xb = torch.tensor(X_query[start:start + batch_size], dtype=torch.float32, device=device)
        chunk = xb.shape[0]
        x_context_batch = X_ctx.repeat(1, chunk, 1)
        y_context_batch = y_ctx[:, None].repeat(1, chunk)
        x_eval = pad_x(xb[None, :, :], model.num_features)

        with SDPACapture(chunk_size=chunk, n_context=n_ctx) as capture:
            _ = model(torch.cat([x_context_batch, x_eval], dim=0), y_context_batch)

        calls = [r for r in capture.records if r["type"] == "dpt"]
        if not calls:
            raise RuntimeError("No TabDPT attention calls captured with SDPACapture.")

        if n_calls_ref is None:
            n_calls_ref = len(calls)
        elif len(calls) != n_calls_ref:
            keep = min(n_calls_ref, len(calls))
            calls = calls[:keep]
            row_max_batches = [arr[:, :keep, :] for arr in row_max_batches]
            row_ent_batches = [arr[:, :keep, :] for arr in row_ent_batches]
            row_arg_batches = [arr[:, :keep, :] for arr in row_arg_batches]
            n_calls_ref = keep

        # SDPACapture keeps the last query rows, but some kernels expose
        # extra query positions. Keep exactly this evaluation batch so AMIA
        # rows stay aligned with membership labels.
        row_max_batches.append(
            np.stack([r["max_attn"][:, -chunk:] for r in calls], axis=0).transpose(2, 0, 1)
        )
        row_ent_batches.append(
            np.stack([r["neg_entropy"][:, -chunk:] for r in calls], axis=0).transpose(2, 0, 1)
        )
        row_arg_batches.append(
            np.stack([r["argmax"][:, -chunk:] for r in calls], axis=0).transpose(2, 0, 1)
        )

    return (
        np.concatenate(row_max_batches, axis=0),
        np.concatenate(row_ent_batches, axis=0),
        np.concatenate(row_arg_batches, axis=0),
    )


def summarize_attention_change(
    original_max_all: np.ndarray,
    original_ent_all: np.ndarray,
    finetuned_max_all: np.ndarray,
    finetuned_ent_all: np.ndarray,
    membership: np.ndarray | None = None,
) -> dict:
    original_row_max = original_max_all.mean(axis=(1, 2))
    original_row_ent = original_ent_all.mean(axis=(1, 2))
    finetuned_row_max = finetuned_max_all.mean(axis=(1, 2))
    finetuned_row_ent = finetuned_ent_all.mean(axis=(1, 2))
    if membership is not None and len(membership) != len(finetuned_row_max):
        raise ValueError(
            "Attention rows do not match membership labels: "
            f"attention={len(finetuned_row_max)}, membership={len(membership)}"
        )
    row_max_delta = finetuned_row_max - original_row_max
    row_ent_delta = finetuned_row_ent - original_row_ent

    summary = {
        "original_row_max_mean": float(original_row_max.mean()),
        "finetuned_row_max_mean": float(finetuned_row_max.mean()),
        "row_max_delta_mean": float(row_max_delta.mean()),
        "row_max_abs_delta_mean": float(np.abs(row_max_delta).mean()),
        "original_row_ent_mean": float(original_row_ent.mean()),
        "finetuned_row_ent_mean": float(finetuned_row_ent.mean()),
        "row_ent_delta_mean": float(row_ent_delta.mean()),
        "row_ent_abs_delta_mean": float(np.abs(row_ent_delta).mean()),
    }
    if membership is not None:
        summary.update(
            {
                "finetuned_row_max_membership_auc": float(roc_auc_score(membership, finetuned_row_max)),
                "row_max_delta_membership_auc": float(roc_auc_score(membership, row_max_delta)),
                "finetuned_row_ent_membership_auc": float(roc_auc_score(membership, finetuned_row_ent)),
                "row_ent_delta_membership_auc": float(roc_auc_score(membership, row_ent_delta)),
                "member_row_max_abs_delta_mean": float(np.abs(row_max_delta[membership == 1]).mean()),
                "nonmember_row_max_abs_delta_mean": float(np.abs(row_max_delta[membership == 0]).mean()),
                "member_row_ent_abs_delta_mean": float(np.abs(row_ent_delta[membership == 1]).mean()),
                "nonmember_row_ent_abs_delta_mean": float(np.abs(row_ent_delta[membership == 0]).mean()),
            }
        )
    return summary


def safe_membership_auc(membership: np.ndarray, scores: np.ndarray) -> float:
    valid = np.isfinite(scores)
    if valid.sum() < 2 or len(np.unique(membership[valid])) < 2:
        return 0.5
    return float(roc_auc_score(membership[valid], scores[valid]))


def attention_layer_table(
    row_max_all: np.ndarray,
    row_ent_all: np.ndarray,
    membership: np.ndarray | None,
    *,
    model_name: str,
) -> pd.DataFrame:
    rows = []
    n_layers = row_max_all.shape[1]
    for layer in range(n_layers):
        row_max = row_max_all[:, layer, :].mean(axis=1)
        row_ent = row_ent_all[:, layer, :].mean(axis=1)
        row = {
            "model": model_name,
            "layer": layer,
            "row_max_mean": float(row_max.mean()),
            "row_ent_mean": float(row_ent.mean()),
        }
        if membership is not None:
            mem = membership.astype(bool)
            row.update(
                {
                    "row_max_auc": safe_membership_auc(membership, row_max),
                    "row_ent_auc": safe_membership_auc(membership, row_ent),
                    "row_max_member_mean": float(row_max[mem].mean()),
                    "row_max_nonmember_mean": float(row_max[~mem].mean()),
                    "row_ent_member_mean": float(row_ent[mem].mean()),
                    "row_ent_nonmember_mean": float(row_ent[~mem].mean()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def attention_layer_head_table(
    row_max_all: np.ndarray,
    row_ent_all: np.ndarray,
    membership: np.ndarray | None,
    *,
    model_name: str,
) -> pd.DataFrame:
    rows = []
    n_layers = row_max_all.shape[1]
    n_heads = row_max_all.shape[2]
    for layer in range(n_layers):
        for head in range(n_heads):
            row_max = row_max_all[:, layer, head]
            row_ent = row_ent_all[:, layer, head]
            row = {
                "model": model_name,
                "layer": layer,
                "head": head,
                "row_max_mean": float(row_max.mean()),
                "row_ent_mean": float(row_ent.mean()),
            }
            if membership is not None:
                mem = membership.astype(bool)
                row.update(
                    {
                        "row_max_auc": safe_membership_auc(membership, row_max),
                        "row_ent_auc": safe_membership_auc(membership, row_ent),
                        "row_max_member_mean": float(row_max[mem].mean()),
                        "row_max_nonmember_mean": float(row_max[~mem].mean()),
                        "row_ent_member_mean": float(row_ent[mem].mean()),
                        "row_ent_nonmember_mean": float(row_ent[~mem].mean()),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def save_attention_signal_tables(
    out_dir: Path,
    *,
    membership: np.ndarray,
    original_scores: np.ndarray,
    finetuned_scores: np.ndarray,
    score_delta: np.ndarray,
    original_mia_attn: tuple[np.ndarray, np.ndarray, np.ndarray],
    finetuned_mia_attn: tuple[np.ndarray, np.ndarray, np.ndarray],
    original_test_attn: tuple[np.ndarray, np.ndarray, np.ndarray],
    finetuned_test_attn: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, str]:
    """Save AMIA-style row, layer, and layer/head attention signals as CSV."""
    original_mia_max, original_mia_ent, _ = original_mia_attn
    finetuned_mia_max, finetuned_mia_ent, _ = finetuned_mia_attn
    original_test_max, original_test_ent, _ = original_test_attn
    finetuned_test_max, finetuned_test_ent, _ = finetuned_test_attn

    original_row_max = original_mia_max.mean(axis=(1, 2))
    original_row_ent = original_mia_ent.mean(axis=(1, 2))
    finetuned_row_max = finetuned_mia_max.mean(axis=(1, 2))
    finetuned_row_ent = finetuned_mia_ent.mean(axis=(1, 2))

    original_attention_summary_path = out_dir / "original_attention_summary.csv"
    pd.DataFrame(
        {
            "member": membership.astype(int),
            "true_label_score": original_scores,
            "row_max": original_row_max,
            "row_ent": original_row_ent,
        }
    ).to_csv(original_attention_summary_path, index=False)

    finetuned_attention_summary_path = out_dir / "finetuned_attention_summary.csv"
    pd.DataFrame(
        {
            "member": membership.astype(int),
            "true_label_score": finetuned_scores,
            "score_delta": score_delta,
            "row_max": finetuned_row_max,
            "row_ent": finetuned_row_ent,
        }
    ).to_csv(finetuned_attention_summary_path, index=False)

    membership_summary_path = out_dir / "membership_attention_summary.csv"
    pd.DataFrame(
        {
            "member": membership.astype(int),
            "original_true_label_score": original_scores,
            "finetuned_true_label_score": finetuned_scores,
            "score_delta": score_delta,
            "row_max_original": original_row_max,
            "row_ent_original": original_row_ent,
            "row_max_finetuned": finetuned_row_max,
            "row_ent_finetuned": finetuned_row_ent,
            "row_max_delta": finetuned_row_max - original_row_max,
            "row_ent_delta": finetuned_row_ent - original_row_ent,
        }
    ).to_csv(membership_summary_path, index=False)

    membership_layers_path = out_dir / "membership_attention_layers.csv"
    pd.concat(
        [
            attention_layer_table(original_mia_max, original_mia_ent, membership, model_name="original"),
            attention_layer_table(finetuned_mia_max, finetuned_mia_ent, membership, model_name="finetuned"),
        ],
        ignore_index=True,
    ).to_csv(membership_layers_path, index=False)

    membership_layer_heads_path = out_dir / "membership_attention_layer_heads.csv"
    pd.concat(
        [
            attention_layer_head_table(original_mia_max, original_mia_ent, membership, model_name="original"),
            attention_layer_head_table(finetuned_mia_max, finetuned_mia_ent, membership, model_name="finetuned"),
        ],
        ignore_index=True,
    ).to_csv(membership_layer_heads_path, index=False)

    test_layers_path = out_dir / "test_attention_layers.csv"
    pd.concat(
        [
            attention_layer_table(original_test_max, original_test_ent, None, model_name="original"),
            attention_layer_table(finetuned_test_max, finetuned_test_ent, None, model_name="finetuned"),
        ],
        ignore_index=True,
    ).to_csv(test_layers_path, index=False)

    return {
        "original_attention_summary": str(original_attention_summary_path),
        "finetuned_attention_summary": str(finetuned_attention_summary_path),
        "membership_attention_summary": str(membership_summary_path),
        "membership_attention_layers": str(membership_layers_path),
        "membership_attention_layer_heads": str(membership_layer_heads_path),
        "test_attention_layers": str(test_layers_path),
    }


def _strip_compile_prefix(key: str) -> str:
    return key.replace("_orig_mod.", "").replace("model.", "")


def weight_delta_summary(original_state: dict, finetuned_state: dict) -> dict:
    original = {_strip_compile_prefix(k): v for k, v in original_state.items()}
    total_sq_diff = 0.0
    total_sq_orig = 0.0
    max_abs_diff = 0.0
    changed_tensors = 0
    compared_tensors = 0

    for key, ft_value in finetuned_state.items():
        clean_key = _strip_compile_prefix(key)
        if clean_key not in original:
            continue
        orig_value = original[clean_key]
        if not torch.is_tensor(orig_value) or not torch.is_tensor(ft_value):
            continue
        if orig_value.shape != ft_value.shape or not torch.is_floating_point(ft_value):
            continue

        diff = ft_value.detach().float() - orig_value.to(ft_value.device).detach().float()
        sq_diff = float(torch.sum(diff * diff).cpu())
        sq_orig = float(torch.sum(orig_value.detach().float() * orig_value.detach().float()).cpu())
        max_diff = float(torch.max(torch.abs(diff)).cpu()) if diff.numel() else 0.0

        compared_tensors += 1
        total_sq_diff += sq_diff
        total_sq_orig += sq_orig
        max_abs_diff = max(max_abs_diff, max_diff)
        if sq_diff > 0.0:
            changed_tensors += 1

    l2_diff = total_sq_diff ** 0.5
    l2_orig = total_sq_orig ** 0.5
    return {
        "compared_tensors": compared_tensors,
        "changed_tensors": changed_tensors,
        "l2_diff": l2_diff,
        "l2_original": l2_orig,
        "relative_l2_diff": l2_diff / (l2_orig + 1e-12),
        "max_abs_diff": max_abs_diff,
    }


def true_label_scores(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    return proba[np.arange(len(y)), y]


def stable_logit(prob: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    prob64 = np.asarray(prob, dtype=np.float64)
    prob64 = np.clip(prob64, eps, 1.0 - eps)
    return np.log(prob64) - np.log1p(-prob64)


def load_original_prediction_cache(
    out_dir: Path,
    *,
    expected_mia_rows: int,
    expected_test_rows: int,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    scores_path = out_dir / "membership_probabilities.npz"
    test_path = out_dir / "test_probabilities.npz"
    if not scores_path.exists() or not test_path.exists():
        scores_path = out_dir / "membership_scores.npz"
        test_path = out_dir / "test_predictions.npz"
    if not scores_path.exists() or not test_path.exists():
        return None
    try:
        scores = np.load(scores_path)
        test = np.load(test_path)
        if "original_proba" not in scores or "original_proba" not in test:
            return None
        mia_proba = scores["original_proba"]
        test_proba = test["original_proba"]
        if mia_proba.shape != (expected_mia_rows, num_classes):
            return None
        if test_proba.shape != (expected_test_rows, num_classes):
            return None
        return mia_proba, test_proba
    except Exception:
        return None


def load_original_attention_cache(
    out_dir: Path,
    *,
    expected_mia_rows: int,
    expected_test_rows: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]] | None:
    mia_path = out_dir / "membership_attention_comparison.npz"
    test_path = out_dir / "test_attention_comparison.npz"
    if not mia_path.exists() or not test_path.exists():
        return None
    try:
        mia = np.load(mia_path)
        test = np.load(test_path)
        mia_keys = ("row_max_all_original", "row_ent_all_original", "row_arg_all_original")
        test_keys = ("row_max_all_original", "row_ent_all_original", "row_arg_all_original")
        if not all(k in mia for k in mia_keys) or not all(k in test for k in test_keys):
            return None
        mia_arrays = (mia[mia_keys[0]], mia[mia_keys[1]], mia[mia_keys[2]])
        test_arrays = (test[test_keys[0]], test[test_keys[1]], test[test_keys[2]])
        if any(arr.shape[0] != expected_mia_rows for arr in mia_arrays):
            return None
        if any(arr.shape[0] != expected_test_rows for arr in test_arrays):
            return None
        return mia_arrays, test_arrays
    except Exception:
        return None


# def compute_original_reference_rmia_scores(...):
#     Disabled for now. Re-enable when running single-reference/full RMIA analyses.


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    X, y = load_csv_arrays(args.dataset, args.data_dir)
    original_indices = np.arange(len(y))
    splits = make_splits(X, y, args.seed)
    X, _, _ = preprocess(X, splits)

    num_classes = int(np.max(y)) + 1
    if num_classes < 2:
        raise ValueError("Need at least two classes for classification.")

    out_dir = ROOT / args.out_dir / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    save_split_files(out_dir, splits, original_indices)
    original_ckpt_path = out_dir / "original_tabdpt.ckpt"
    model, source_checkpoint, resolved_checkpoint_path = load_tabdpt(args.checkpoint, args.device)
    if Path(resolved_checkpoint_path).resolve() != original_ckpt_path.resolve():
        shutil.copy2(resolved_checkpoint_path, original_ckpt_path)
    X = reduce_features_for_checkpoint(
        X,
        splits,
        max_features=model.num_features,
        method=args.feature_reduction,
        seed=args.seed,
    )
    if X.shape[1] > model.num_features:
        raise ValueError(
            f"Dataset has {X.shape[1]} features, but checkpoint supports {model.num_features}."
        )
    if num_classes > model.n_out:
        raise ValueError(
            f"Dataset has {num_classes} classes, but checkpoint supports {model.n_out}."
        )
    if args.freeze_encoder:
        maybe_freeze_encoder(model)

    X_member, y_member = X[splits["member"]], y[splits["member"]]
    X_nonmember, y_nonmember = X[splits["nonmember"]], y[splits["nonmember"]]
    X_context, y_context = X[splits["context"]], y[splits["context"]]
    X_test, y_test = X[splits["test"]], y[splits["test"]]

    X_mia = np.concatenate([X_member, X_nonmember], axis=0)
    y_mia = np.concatenate([y_member, y_nonmember], axis=0)
    membership = np.concatenate([
        np.ones(len(y_member), dtype=np.int64),
        np.zeros(len(y_nonmember), dtype=np.int64),
    ])

    print("Split summary (local/original indices are identical unless you later subsample):")
    print_split_summary(splits, y, original_indices)

    original_cache = load_original_prediction_cache(
        out_dir,
        expected_mia_rows=len(y_mia),
        expected_test_rows=len(y_test),
        num_classes=num_classes,
    )
    if original_cache is not None:
        print("\nReusing cached original predictions")
        original_mia_proba, original_test_proba = original_cache
    else:
        print("\nEvaluating original checkpoint with fixed context...")
        original_mia_proba = predict_proba_fixed_context(
            model,
            X_context,
            y_context,
            X_mia,
            num_classes=num_classes,
            context_size=args.context_size,
            batch_size=args.eval_batch_size,
            temperature=args.temperature,
            device=args.device,
            desc="original member/nonmember scores",
        )
        original_test_proba = predict_proba_fixed_context(
            model,
            X_context,
            y_context,
            X_test,
            num_classes=num_classes,
            context_size=args.context_size,
            batch_size=args.eval_batch_size,
            temperature=args.temperature,
            device=args.device,
            desc="original test predictions",
        )
    original_scores = true_label_scores(original_mia_proba, y_mia)
    original_member_pred = original_mia_proba[:len(y_member)].argmax(axis=1)
    original_metrics = {
        "train_accuracy": float(accuracy_score(y_member, original_member_pred)),
        "test_accuracy": float(accuracy_score(y_test, original_test_proba.argmax(axis=1))),
        "member_score_mean": float(original_scores[membership == 1].mean()),
        "nonmember_score_mean": float(original_scores[membership == 0].mean()),
        "membership_auc_true_label_probability": float(roc_auc_score(membership, original_scores)),
    }
    with (out_dir / "original_performance.json").open("w") as f:
        json.dump(original_metrics, f, indent=2)

    original_mia_attn = None
    original_test_attn = None
    if not args.skip_attention:
        original_attention_cache = load_original_attention_cache(
            out_dir,
            expected_mia_rows=len(y_mia),
            expected_test_rows=len(y_test),
        )
        if original_attention_cache is not None:
            print("\nReusing cached original attention from attention comparison files")
            original_mia_attn, original_test_attn = original_attention_cache
        else:
            print("\nCapturing original AMIA-style member/nonmember attention with fixed context...")
            original_mia_attn = extract_attention_fixed_context(
                model,
                X_context,
                y_context,
                X_mia,
                context_size=args.context_size,
                batch_size=args.eval_batch_size,
                device=args.device,
                desc="original member/nonmember attention",
            )
            original_test_attn = extract_attention_fixed_context(
                model,
                X_context,
                y_context,
                X_test,
                context_size=args.context_size,
                batch_size=args.eval_batch_size,
                device=args.device,
                desc="original test attention",
            )

    print("\nStarting fine-tuning...")
    optimizer = finetune(model, X_member, y_member, args)
    ckpt_path = save_finetuned_checkpoint(
        model, optimizer, source_checkpoint, args, out_dir, splits, original_indices
    )

    rmia_column_files = {}

    print("\nEvaluating fine-tuned checkpoint with the same fixed context...")
    finetuned_mia_proba = predict_proba_fixed_context(
        model,
        X_context,
        y_context,
        X_mia,
        num_classes=num_classes,
        context_size=args.context_size,
        batch_size=args.eval_batch_size,
        temperature=args.temperature,
        device=args.device,
        desc="fine-tuned member/nonmember scores",
    )
    finetuned_test_proba = predict_proba_fixed_context(
        model,
        X_context,
        y_context,
        X_test,
        num_classes=num_classes,
        context_size=args.context_size,
        batch_size=args.eval_batch_size,
        temperature=args.temperature,
        device=args.device,
        desc="fine-tuned test predictions",
    )

    finetuned_scores = true_label_scores(finetuned_mia_proba, y_mia)
    score_delta = stable_logit(finetuned_scores) - stable_logit(original_scores)
    finetuned_member_pred = finetuned_mia_proba[:len(y_member)].argmax(axis=1)
    finetuned_metrics = {
        "train_accuracy": float(accuracy_score(y_member, finetuned_member_pred)),
        "test_accuracy": float(accuracy_score(y_test, finetuned_test_proba.argmax(axis=1))),
        "member_score_mean": float(finetuned_scores[membership == 1].mean()),
        "nonmember_score_mean": float(finetuned_scores[membership == 0].mean()),
        "membership_auc_true_label_probability": float(roc_auc_score(membership, finetuned_scores)),
        "membership_auc_logit_delta_vs_original": float(roc_auc_score(membership, score_delta)),
    }
    with (out_dir / "original_performance.json").open("w") as f:
        json.dump(original_metrics, f, indent=2)
    with (out_dir / "finetuned_performance.json").open("w") as f:
        json.dump(finetuned_metrics, f, indent=2)

    attention_metrics = None
    attention_signal_files = {}
    if not args.skip_attention:
        print("\nCapturing fine-tuned AMIA-style member/nonmember attention with the same fixed context...")
        finetuned_mia_attn = extract_attention_fixed_context(
            model,
            X_context,
            y_context,
            X_mia,
            context_size=args.context_size,
            batch_size=args.eval_batch_size,
            device=args.device,
            desc="fine-tuned member/nonmember attention",
        )
        finetuned_test_attn = extract_attention_fixed_context(
            model,
            X_context,
            y_context,
            X_test,
            context_size=args.context_size,
            batch_size=args.eval_batch_size,
            device=args.device,
            desc="fine-tuned test attention",
        )

        attention_metrics = {
            "membership": summarize_attention_change(
                original_mia_attn[0],
                original_mia_attn[1],
                finetuned_mia_attn[0],
                finetuned_mia_attn[1],
                membership=membership,
            ),
            "test": summarize_attention_change(
                original_test_attn[0],
                original_test_attn[1],
                finetuned_test_attn[0],
                finetuned_test_attn[1],
            ),
        }
        with (out_dir / "attention_change.json").open("w") as f:
            json.dump(attention_metrics, f, indent=2)
        attention_signal_files = save_attention_signal_tables(
            out_dir,
            membership=membership,
            original_scores=original_scores,
            finetuned_scores=finetuned_scores,
            score_delta=score_delta,
            original_mia_attn=original_mia_attn,
            finetuned_mia_attn=finetuned_mia_attn,
            original_test_attn=original_test_attn,
            finetuned_test_attn=finetuned_test_attn,
        )
        np.savez(
            out_dir / "membership_attention_comparison.npz",
            membership=membership,
            row_max_all_original=original_mia_attn[0],
            row_ent_all_original=original_mia_attn[1],
            row_arg_all_original=original_mia_attn[2],
            row_max_all_finetuned=finetuned_mia_attn[0],
            row_ent_all_finetuned=finetuned_mia_attn[1],
            row_arg_all_finetuned=finetuned_mia_attn[2],
            row_max_original=original_mia_attn[0].mean(axis=(1, 2)),
            row_ent_original=original_mia_attn[1].mean(axis=(1, 2)),
            row_max_finetuned=finetuned_mia_attn[0].mean(axis=(1, 2)),
            row_ent_finetuned=finetuned_mia_attn[1].mean(axis=(1, 2)),
            member_idx_original=original_indices[splits["member"]],
            nonmember_idx_original=original_indices[splits["nonmember"]],
        )
        np.savez(
            out_dir / "test_attention_comparison.npz",
            row_max_all_original=original_test_attn[0],
            row_ent_all_original=original_test_attn[1],
            row_arg_all_original=original_test_attn[2],
            row_max_all_finetuned=finetuned_test_attn[0],
            row_ent_all_finetuned=finetuned_test_attn[1],
            row_arg_all_finetuned=finetuned_test_attn[2],
            row_max_original=original_test_attn[0].mean(axis=(1, 2)),
            row_ent_original=original_test_attn[1].mean(axis=(1, 2)),
            row_max_finetuned=finetuned_test_attn[0].mean(axis=(1, 2)),
            row_ent_finetuned=finetuned_test_attn[1].mean(axis=(1, 2)),
            test_idx_original=original_indices[splits["test"]],
            context_idx_original=original_indices[splits["context"]],
        )

    delta_metrics = {
        "train_accuracy_delta": finetuned_metrics["train_accuracy"] - original_metrics["train_accuracy"],
        "test_accuracy_delta": finetuned_metrics["test_accuracy"] - original_metrics["test_accuracy"],
        "member_score_mean_delta": finetuned_metrics["member_score_mean"] - original_metrics["member_score_mean"],
        "nonmember_score_mean_delta": finetuned_metrics["nonmember_score_mean"] - original_metrics["nonmember_score_mean"],
        "weight_delta": weight_delta_summary(source_checkpoint["model"], model.state_dict()),
    }
    with (out_dir / "model_change.json").open("w") as f:
        json.dump(delta_metrics, f, indent=2)

    result = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint or f"{DEFAULT_TABDPT_REPO}/{DEFAULT_TABDPT_WEIGHTS}",
        "resolved_checkpoint_path": str(resolved_checkpoint_path),
        "seed": args.seed,
        "num_classes": num_classes,
        "sizes": {name: int(len(indices)) for name, indices in splits.items()},
        "num_rmia_reference_models": NUM_RMIA_REFERENCE_MODELS,
        "files": {
            "original_checkpoint": str(original_ckpt_path),
            "finetuned_checkpoint": str(ckpt_path),
            **rmia_column_files,
            "splits": str(out_dir / "splits.npz"),
            "original_performance": str(out_dir / "original_performance.json"),
            "finetuned_performance": str(out_dir / "finetuned_performance.json"),
            "model_change": str(out_dir / "model_change.json"),
            "membership_scores": str(out_dir / "membership_scores.csv"),
            "membership_probabilities": str(out_dir / "membership_probabilities.npz"),
            "test_predictions": str(out_dir / "test_predictions.csv"),
            "test_probabilities": str(out_dir / "test_probabilities.npz"),
            "attention_change": str(out_dir / "attention_change.json") if attention_metrics is not None else None,
            "membership_attention": str(out_dir / "membership_attention_comparison.npz") if attention_metrics is not None else None,
            "test_attention": str(out_dir / "test_attention_comparison.npz") if attention_metrics is not None else None,
            **attention_signal_files,
        },
    }

    with (out_dir / "result.json").open("w") as f:
        json.dump(result, f, indent=2)
    mia_idx_local = np.concatenate([splits["member"], splits["nonmember"]])
    mia_idx_original = original_indices[mia_idx_local]
    original_mia_pred = original_mia_proba.argmax(axis=1)
    finetuned_mia_pred = finetuned_mia_proba.argmax(axis=1)
    original_test_pred = original_test_proba.argmax(axis=1)
    finetuned_test_pred = finetuned_test_proba.argmax(axis=1)

    # CSV keeps row-level scalar analysis values. NPZ keeps only full probability
    # matrices, which are awkward/lossy to store in CSV for multiclass datasets.
    np.savez(
        out_dir / "membership_probabilities.npz",
        idx_original=mia_idx_original,
        original_proba=original_mia_proba,
        finetuned_proba=finetuned_mia_proba,
    )
    pd.DataFrame(
        {
            "idx_local": mia_idx_local,
            "idx_original": mia_idx_original,
            "member": membership.astype(bool),
            "y": y_mia,
            "original_pred": original_mia_pred,
            "finetuned_pred": finetuned_mia_pred,
            "original_true_label_score": original_scores,
            "finetuned_true_label_score": finetuned_scores,
            "score_delta": score_delta,
            "original_correct": original_mia_pred == y_mia,
            "finetuned_correct": finetuned_mia_pred == y_mia,
        }
    ).to_csv(out_dir / "membership_scores.csv", index=False)

    np.savez(
        out_dir / "test_probabilities.npz",
        idx_original=original_indices[splits["test"]],
        original_proba=original_test_proba,
        finetuned_proba=finetuned_test_proba,
    )
    pd.DataFrame(
        {
            "idx_local": splits["test"],
            "idx_original": original_indices[splits["test"]],
            "y": y_test,
            "original_pred": original_test_pred,
            "finetuned_pred": finetuned_test_pred,
            "original_correct": original_test_pred == y_test,
            "finetuned_correct": finetuned_test_pred == y_test,
            "changed_prediction": original_test_pred != finetuned_test_pred,
        }
    ).to_csv(out_dir / "test_predictions.csv", index=False)

    for stale_name in ("membership_scores.npz", "test_predictions.npz"):
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    print(json.dumps(result, indent=2))
    print(f"\nSaved: {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
