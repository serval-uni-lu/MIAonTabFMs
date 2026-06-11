#!/usr/bin/env python3
"""Compute stored RMIA features for fine-tuned TabDPT context-member rows.

This is the companion producer for ft_tabdpt_source_attribution.py. It audits
rows from splits.npz::context_idx_original instead of the fine-tuning member /
nonmember audit pool, and saves context-specific files with different names.

Always computes both reference modes in one pass:
  - report_context_original_refs:   finetuned target col + original checkpoint refs
  - report_context_finetuned_refs:  finetuned target col + finetuned checkpoint refs

uv run run_attacks/eval_rmia_ft_tabdpt_context_members.py  \
--datasets purchases10,46905_Amazon_employee_access
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
MIA_METER = ROOT / "ml_privacy_meter"
CONTEXT_RMIA_ORIGINAL_REFS_SUBDIR = "report_context_original_refs"
CONTEXT_RMIA_FINETUNED_REFS_SUBDIR = "report_context_finetuned_refs"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MIA_METER) not in sys.path:
    sys.path.insert(0, str(MIA_METER))

import torch.nn.functional as F
from attacks import run_population_attack, run_rmia  # noqa: E402
from run_attacks.eval_rmia_ft_tabdpt import (  # noqa: E402
    NUM_RMIA_REFERENCE_MODELS,
    load_csv_arrays,
    load_model_and_cfg,
    load_saved_splits,
    predict_proba,
    preprocess_from_context,
    reduce_features_for_checkpoint,
    save_attack_result,
    true_label_score,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="purchases10")
    p.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated datasets, e.g. purchases10,46905_Amazon_employee_access.",
    )
    p.add_argument("--data-dir", default="data/data_tabarena")
    p.add_argument("--run-dir", default=None, help="Only valid for a single --dataset run.")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval-batch-size", type=int, default=32, help="Query records per forward pass. Does not affect RMIA scores. Reduce to 16 if OOM on larger datasets.")
    p.add_argument("--context-size", type=int, default=0, help="Max context rows passed to the model. 0 (default) uses the full context split.")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--feature-reduction", choices=["pca", "subsample", "error"], default="pca")
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional smoke-test cap. Default audits all context_idx_original rows.",
    )
    p.add_argument(
        "--max-pop-rows",
        type=int,
        default=None,
        help="Optional smoke-test cap for population scoring. Default uses all test_idx_original rows.",
    )
    p.add_argument(
        "--skip-amia",
        action="store_true",
        help="Skip attention-based MIA extraction (faster; RMIA results are unaffected).",
    )
    p.add_argument(
        "--amia-batch-size", type=int, default=32,
        help="Batch size for AMIA attention extraction. Reduce if OOM (default: 32).",
    )
    p.add_argument(
        "--amia-only",
        action="store_true",
        help="Run only AMIA attention extraction, skipping all RMIA scoring. "
             "Requires splits.npz and both checkpoints to exist.",
    )
    return p.parse_args()


def cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def freeze_for_eval(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


class _BatchedSDPACapture:
    """SDPA hook that captures per-sample attention for the last query position only.

    In predict_proba the input layout is (n_ctx + 1, bsz, features): the last
    sequence position is the query sample, one per batch element.  We compute
    attention only for that row so memory stays O(bsz * n_heads * k_len) instead
    of O(bsz * n_heads * q_len * k_len).
    """

    def __init__(self, n_context: int) -> None:
        self._orig = None
        self.records: list = []
        self.n_context = n_context

    def __enter__(self):
        self.records.clear()
        _orig = F.scaled_dot_product_attention
        self._orig = _orig
        records = self.records
        n_context = self.n_context
        counter = [0]

        def _hook(query, key, value,
                  attn_mask=None, dropout_p=0.0, is_causal=False,
                  scale=None, **kwargs):
            q_len = query.shape[-2]
            k_len = key.shape[-2]
            if k_len == n_context and q_len > k_len:
                with torch.no_grad():
                    s = query.shape[-1] ** -0.5 if scale is None else scale
                    # Only the last query row attends to context keys
                    q_last = query[:, :, -1:, :].float()           # (B, H, 1, d)
                    sc = torch.matmul(q_last, key.float().transpose(-2, -1)) * s
                    w = torch.softmax(sc[:, :, 0, :], dim=-1).cpu().numpy()  # (B, H, k)
                    eps = 1e-12
                    records.append({
                        "type":        "dpt",
                        "call_idx":    counter[0],
                        "max_attn":    w.max(axis=-1).astype(np.float32),          # (B, H)
                        "neg_entropy": (w * np.log(w + eps)).sum(axis=-1).astype(np.float32),
                    })
                    counter[0] += 1
            extra = {} if scale is None else {"scale": scale}
            extra.update(kwargs)
            return _orig(query, key, value,
                         attn_mask=attn_mask, dropout_p=dropout_p,
                         is_causal=is_causal, **extra)

        F.scaled_dot_product_attention = _hook
        return self

    def __exit__(self, *_):
        F.scaled_dot_product_attention = self._orig


def extract_amia_row_max(
    model: torch.nn.Module,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    *,
    context_size: int,
    device: str,
    batch_size: int = 32,
    desc: str = "AMIA",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Per-sample (row_max, row_ent) of shape (n_query, n_layers, n_heads).

    Uses _BatchedSDPACapture so the full batch is processed in one forward pass
    per chunk — ~batch_size× faster than the bsz=1 approach.
    """
    from model import pad_x  # TabDPT-training is on sys.path via eval_rmia_ft_tabdpt

    model.eval()
    n_ctx = min(context_size, len(y_context)) if context_size > 0 else len(y_context)
    X_ctx_t   = torch.tensor(X_context[:n_ctx], dtype=torch.float32, device=device)
    y_ctx_t   = torch.tensor(y_context[:n_ctx], dtype=torch.float32, device=device)
    X_ctx_pad = pad_x(X_ctx_t[:, None, :], model.num_features)  # (n_ctx, 1, feat)

    row_max_list: list[np.ndarray] = []
    row_ent_list: list[np.ndarray] = []
    n_calls_ref: int | None = None

    for start in tqdm(range(0, len(X_query), batch_size), desc=desc, leave=False):
        batch = X_query[start:start + batch_size]
        bsz   = len(batch)
        xb    = torch.tensor(batch, dtype=torch.float32, device=device)
        x_ctx_b = X_ctx_pad.repeat(1, bsz, 1)            # (n_ctx, bsz, feat)
        y_ctx_b = y_ctx_t[:, None].repeat(1, bsz)         # (n_ctx, bsz)
        x_qry   = pad_x(xb[None, :, :], model.num_features)  # (1, bsz, feat)

        cap = _BatchedSDPACapture(n_context=n_ctx)
        with torch.no_grad(), cap:
            model(torch.cat([x_ctx_b, x_qry], dim=0), y_ctx_b)

        dpt_calls = [r for r in cap.records if r["type"] == "dpt"]
        if not dpt_calls:
            return None, None
        if n_calls_ref is None:
            n_calls_ref = len(dpt_calls)

        # records[i]["max_attn"]: (bsz, n_heads)
        # stack over layers → (n_layers, bsz, n_heads) → transpose → (bsz, n_layers, n_heads)
        rm = np.stack([r["max_attn"]    for r in dpt_calls[:n_calls_ref]], axis=0).transpose(1, 0, 2)
        re = np.stack([r["neg_entropy"] for r in dpt_calls[:n_calls_ref]], axis=0).transpose(1, 0, 2)
        row_max_list.append(rm)
        row_ent_list.append(re)

    if not row_max_list:
        return None, None
    return np.concatenate(row_max_list, axis=0), np.concatenate(row_ent_list, axis=0)


def save_context_amia(
    run_dir: Path,
    ctx_idx: np.ndarray,
    nm_idx: np.ndarray,
    ctx_rm_orig: np.ndarray | None,
    ctx_re_orig: np.ndarray | None,
    nm_rm_orig: np.ndarray | None,
    nm_re_orig: np.ndarray | None,
    ctx_rm_ft: np.ndarray | None,
    ctx_re_ft: np.ndarray | None,
    nm_rm_ft: np.ndarray | None,
    nm_re_ft: np.ndarray | None,
) -> dict:
    """Save raw attention arrays and a per-sample summary CSV with AMIA AUC."""
    out_dir = run_dir / "amia_context"
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {"out_dir": str(out_dir)}

    for fname, arr in [
        ("context_member_row_max_original",  ctx_rm_orig),
        ("context_member_row_ent_original",  ctx_re_orig),
        ("nonmember_row_max_original",       nm_rm_orig),
        ("nonmember_row_ent_original",       nm_re_orig),
        ("context_member_row_max_finetuned", ctx_rm_ft),
        ("context_member_row_ent_finetuned", ctx_re_ft),
        ("nonmember_row_max_finetuned",      nm_rm_ft),
        ("nonmember_row_ent_finetuned",      nm_re_ft),
    ]:
        if arr is not None:
            np.save(out_dir / f"{fname}.npy", arr)

    def _agg(arr: np.ndarray | None) -> np.ndarray | None:
        return arr.mean(axis=(1, 2)) if arr is not None else None  # (n,)

    def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
        valid = ~np.isnan(scores.ravel())
        if valid.sum() < 2 or len(np.unique(labels.ravel()[valid])) < 2:
            return 0.5
        return float(roc_auc_score(labels.ravel()[valid], scores.ravel()[valid]))

    ctx_rm_s_orig = _agg(ctx_rm_orig)
    ctx_re_s_orig = _agg(ctx_re_orig)
    ctx_rm_s_ft   = _agg(ctx_rm_ft)
    ctx_re_s_ft   = _agg(ctx_re_ft)
    nm_rm_s_orig  = _agg(nm_rm_orig)
    nm_re_s_orig  = _agg(nm_re_orig)
    nm_rm_s_ft    = _agg(nm_rm_ft)
    nm_re_s_ft    = _agg(nm_re_ft)

    if ctx_rm_s_orig is not None or nm_rm_s_orig is not None:
        rows = []
        if ctx_rm_s_orig is not None:
            for i, ix in enumerate(ctx_idx):
                rows.append({
                    "idx_original": int(ix),
                    "context_member": 1,
                    "row_max_original": float(ctx_rm_s_orig[i]),
                    "row_ent_original": float(ctx_re_s_orig[i]) if ctx_re_s_orig is not None else np.nan,
                    "row_max_finetuned": float(ctx_rm_s_ft[i]) if ctx_rm_s_ft is not None else np.nan,
                    "row_ent_finetuned": float(ctx_re_s_ft[i]) if ctx_re_s_ft is not None else np.nan,
                })
        if nm_rm_s_orig is not None:
            for i, ix in enumerate(nm_idx):
                rows.append({
                    "idx_original": int(ix),
                    "context_member": 0,
                    "row_max_original": float(nm_rm_s_orig[i]),
                    "row_ent_original": float(nm_re_s_orig[i]) if nm_re_s_orig is not None else np.nan,
                    "row_max_finetuned": float(nm_rm_s_ft[i]) if nm_rm_s_ft is not None else np.nan,
                    "row_ent_finetuned": float(nm_re_s_ft[i]) if nm_re_s_ft is not None else np.nan,
                })
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(out_dir / "context_amia_summary.csv", index=False)

        labels = summary_df["context_member"].to_numpy(dtype=int)
        for col, key in [
            ("row_max_original",  "row_max_auc_original"),
            ("row_max_finetuned", "row_max_auc_finetuned"),
            ("row_ent_original",  "row_ent_auc_original"),
            ("row_ent_finetuned", "row_ent_auc_finetuned"),
        ]:
            scores = summary_df[col].to_numpy(dtype=float)
            result[key] = _auc(scores, labels)
        result["auc_original"] = result["row_max_auc_original"]
        result["auc_finetuned"] = result["row_max_auc_finetuned"]

    with (out_dir / "context_amia_auc.json").open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\nContext AMIA AUC  original={result.get('auc_original')}  finetuned={result.get('auc_finetuned')}")
    return result


def run_amia_only(args: argparse.Namespace, dataset: str) -> dict:
    """Load models and data, run AMIA extraction only, skip all RMIA scoring."""
    cleanup_gpu()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else ROOT / "ml_privacy_meter" / "logs" / "ft_tabdpt" / dataset

    orig_ckpt   = run_dir / "original_tabdpt.ckpt"
    ft_ckpt     = run_dir / "finetuned_tabdpt.ckpt"
    splits_path = run_dir / "splits.npz"
    for path in (orig_ckpt, ft_ckpt, splits_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    splits   = load_saved_splits(splits_path, NUM_RMIA_REFERENCE_MODELS)
    query_idx = splits["context"].astype(int)
    if args.max_rows is not None:
        query_idx = query_idx[: args.max_rows]
    validate_context_experiment_splits(splits, query_idx)
    nonmember_sample_idx = splits["nonmember"][: len(query_idx)].astype(int)

    original_model, _, _ = load_model_and_cfg(orig_ckpt, args.device)
    original_model = freeze_for_eval(original_model)
    num_features = original_model.num_features

    X_all, y_all = load_csv_arrays(dataset, args.data_dir)
    X_all = preprocess_from_context(X_all, splits["context"])
    X_all = reduce_features_for_checkpoint(
        X_all, splits["context"], num_features, args.feature_reduction, seed=12345,
    )
    context_idx      = splits["context"].astype(int)
    X_target_context = X_all[context_idx]
    y_target_context = y_all[context_idx]
    X_query          = X_all[query_idx]

    amia_ctx_rm_orig, amia_ctx_re_orig = extract_amia_row_max(
        original_model, X_target_context, y_target_context, X_query,
        context_size=args.context_size, device=args.device,
        batch_size=args.amia_batch_size,
        desc="AMIA context members / original",
    )
    amia_nm_rm_orig, amia_nm_re_orig = extract_amia_row_max(
        original_model, X_target_context, y_target_context, X_all[nonmember_sample_idx],
        context_size=args.context_size, device=args.device,
        batch_size=args.amia_batch_size,
        desc="AMIA nonmembers / original",
    )
    del original_model
    cleanup_gpu()

    finetuned_model, _, _ = load_model_and_cfg(ft_ckpt, args.device)
    finetuned_model = freeze_for_eval(finetuned_model)

    amia_ctx_rm_ft, amia_ctx_re_ft = extract_amia_row_max(
        finetuned_model, X_target_context, y_target_context, X_query,
        context_size=args.context_size, device=args.device,
        batch_size=args.amia_batch_size,
        desc="AMIA context members / finetuned",
    )
    amia_nm_rm_ft, amia_nm_re_ft = extract_amia_row_max(
        finetuned_model, X_target_context, y_target_context, X_all[nonmember_sample_idx],
        context_size=args.context_size, device=args.device,
        batch_size=args.amia_batch_size,
        desc="AMIA nonmembers / finetuned",
    )
    del finetuned_model
    cleanup_gpu()

    return save_context_amia(
        run_dir, query_idx, nonmember_sample_idx,
        amia_ctx_rm_orig, amia_ctx_re_orig,
        amia_nm_rm_orig,  amia_nm_re_orig,
        amia_ctx_rm_ft,   amia_ctx_re_ft,
        amia_nm_rm_ft,    amia_nm_re_ft,
    )


def requested_datasets(args: argparse.Namespace) -> list[str]:
    if not args.datasets:
        return [args.dataset]
    if args.run_dir:
        raise ValueError("--run-dir can only be used with --dataset, not --datasets.")
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if not datasets:
        raise ValueError("--datasets was provided but no dataset names were parsed.")
    return datasets


def model_membership_splits(splits: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    cols = [("target", splits["context"]), ("paired", splits["paired"])]
    for ref_id in range(NUM_RMIA_REFERENCE_MODELS):
        cols.extend(
            [
                (f"reference_{ref_id}_a", splits[f"reference_{ref_id}_a"]),
                (f"reference_{ref_id}_b", splits[f"reference_{ref_id}_b"]),
            ]
        )
    return cols


def membership_matrix(query_idx: np.ndarray, columns: list[tuple[str, np.ndarray]]) -> np.ndarray:
    out = np.zeros((len(query_idx), len(columns)), dtype=bool)
    query_idx = query_idx.astype(int)
    for col, (_, train_idx) in enumerate(columns):
        train_set = set(np.asarray(train_idx, dtype=int).tolist())
        out[:, col] = np.array([int(idx) in train_set for idx in query_idx], dtype=bool)
    return out


def validate_context_experiment_splits(splits: dict[str, np.ndarray], query_idx: np.ndarray) -> None:
    context_set = set(splits["context"].astype(int).tolist())
    query_set = set(query_idx.astype(int).tolist())
    if not query_set.issubset(context_set):
        unexpected = sorted(query_set - context_set)[:10]
        raise ValueError(f"Context-member audit contains rows outside context_idx_original: {unexpected}")

    for split_name in ["member", "nonmember", "test"]:
        overlap = query_set & set(splits[split_name].astype(int).tolist())
        if overlap:
            raise ValueError(
                f"Context-member audit overlaps {split_name}_idx_original; "
                f"first overlaps: {sorted(overlap)[:10]}"
            )


def array_fingerprint(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def score_column(
    model_name: str,
    model: torch.nn.Module,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    *,
    num_classes: int,
    context_size: int,
    batch_size: int,
    temperature: float,
    device: str,
) -> np.ndarray:
    cleanup_gpu()
    with torch.inference_mode():
        proba = predict_proba(
            model,
            X_context,
            y_context,
            X_query,
            num_classes=num_classes,
            context_size=context_size if context_size > 0 else len(y_context),
            batch_size=batch_size,
            temperature=temperature,
            device=device,
            desc=f"{model_name}/context-member audit",
        )
    scores = true_label_score(proba, y_query)
    del proba
    cleanup_gpu()
    return scores


def add_feature_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rmia_delta"] = out["rmia_finetuned"] - out["rmia_original"]
    out["attack_p_delta"] = out["attack_p_finetuned"] - out["attack_p_original"]
    out["target_signal_delta"] = out["target_signal_finetuned"] - out["target_signal_original"]
    out["reference_signal_delta"] = out["reference_signal_finetuned_target_eval"] - out["reference_signal_original"]
    out["target_reference_gap_original"] = out["target_signal_original"] - out["reference_signal_original"]
    out["target_reference_gap_finetuned"] = out["target_signal_finetuned"] - out["reference_signal_finetuned_target_eval"]
    out["target_reference_gap_delta"] = out["target_reference_gap_finetuned"] - out["target_reference_gap_original"]
    return out


def load_audit_nonmember_rmia_scores(run_dir: Path, original_reference_attack: bool) -> pd.DataFrame:
    path = run_dir / "rmia" / "rmia_scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing base RMIA scores for context AUC negatives: {path}")
    scores = pd.read_csv(path)
    src_col = "rmia_finetuned_original_refs" if original_reference_attack else "rmia_finetuned_ft_refs"
    required = ["idx_original", "member", "rmia_original", src_col]
    missing = [col for col in required if col not in scores.columns]
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}. Re-run eval_rmia_ft_tabdpt.py first.")
    scores = scores.rename(columns={src_col: "rmia_finetuned"})
    return scores[scores["member"].astype(int) == 0].copy()


def save_context_auc_reports(
    out_dir: Path,
    context_df: pd.DataFrame,
    nonmember_df: pd.DataFrame,
) -> dict:
    context_labels = np.ones(len(context_df), dtype=bool)
    nonmember_labels = np.zeros(len(nonmember_df), dtype=bool)
    labels = np.concatenate([context_labels, nonmember_labels])

    original_scores = np.concatenate([
        context_df["rmia_original"].to_numpy(),
        nonmember_df["rmia_original"].to_numpy(),
    ])
    finetuned_scores = np.concatenate([
        context_df["rmia_finetuned"].to_numpy(),
        nonmember_df["rmia_finetuned"].to_numpy(),
    ])

    result_original = save_attack_result(
        out_dir / "report_context_vs_nonmember_original" / "exp",
        original_scores,
        labels,
    )
    result_finetuned = save_attack_result(
        out_dir / "report_context_vs_nonmember_finetuned" / "exp",
        finetuned_scores,
        labels,
    )

    auc_table = pd.concat(
        [
            pd.DataFrame({
                "idx_original": context_df["idx_original"].astype(int),
                "source": "context_member",
                "membership": 1,
                "rmia_original": context_df["rmia_original"],
                "rmia_finetuned": context_df["rmia_finetuned"],
            }),
            pd.DataFrame({
                "idx_original": nonmember_df["idx_original"].astype(int),
                "source": "audit_nonmember",
                "membership": 0,
                "rmia_original": nonmember_df["rmia_original"],
                "rmia_finetuned": nonmember_df["rmia_finetuned"],
            }),
        ],
        ignore_index=True,
    )
    auc_table.to_csv(out_dir / "context_vs_nonmember_auc_scores.csv", index=False)

    return {
        "n_context_member_positive": int(len(context_df)),
        "n_audit_nonmember_negative": int(len(nonmember_df)),
        "original_model_rmia": result_original,
        "finetuned_model_rmia": result_finetuned,
        "files": {
            "auc_scores": str(out_dir / "context_vs_nonmember_auc_scores.csv"),
            "original_attack_result": str(out_dir / "report_context_vs_nonmember_original" / "exp" / "attack_result_0.npz"),
            "finetuned_attack_result": str(out_dir / "report_context_vs_nonmember_finetuned" / "exp" / "attack_result_0.npz"),
        },
    }


def _build_score_columns(
    query_idx: np.ndarray,
    y_query: np.ndarray,
    memberships: np.ndarray,
    model_names: list[str],
    rmia_original: np.ndarray,
    rmia_finetuned: np.ndarray,
    attack_p_original: np.ndarray,
    attack_p_finetuned: np.ndarray,
    original_sigs: np.ndarray,
    finetuned_sigs: np.ndarray,
) -> dict:
    cols: dict = {
        "idx_original": query_idx,
        "source": "context_member",
        "source_id": 1,
        "y": y_query,
        "rmia_original": rmia_original,
        "rmia_finetuned": rmia_finetuned,
        "attack_p_original": attack_p_original,
        "attack_p_finetuned": attack_p_finetuned,
        "target_signal_original": original_sigs[:, 0],
        "target_signal_finetuned": finetuned_sigs[:, 0],
        "paired_signal_original": original_sigs[:, 1],
        "paired_signal_finetuned_target_eval": finetuned_sigs[:, 1],
    }
    for col, name in enumerate(model_names):
        cols[f"{name}_member"] = memberships[:, col].astype(int)
        if col >= 2:
            cols[f"{name}_signal_original"] = original_sigs[:, col]
            cols[f"{name}_signal_finetuned_target_eval"] = finetuned_sigs[:, col]
    cols["reference_signal_original"] = original_sigs[:, 2:].mean(axis=1)
    cols["reference_signal_finetuned_target_eval"] = finetuned_sigs[:, 2:].mean(axis=1)
    return cols


def _save_variant(
    out_dir: Path,
    query_idx: np.ndarray,
    memberships: np.ndarray,
    original_sigs: np.ndarray,
    finetuned_sigs: np.ndarray,
    feature_df: pd.DataFrame,
    run_dir: Path,
    original_reference_attack: bool,
    summary_fields: dict,
) -> dict:
    sig_dir = out_dir / "signals"
    np.save(sig_dir / "context_member_rmia_signals_original.npy", original_sigs)
    np.save(sig_dir / "context_member_rmia_signals_finetuned.npy", finetuned_sigs)
    np.save(out_dir / "context_member_memberships.npy", memberships)
    np.save(out_dir / "context_member_idx_original.npy", query_idx)

    feature_path = out_dir / "context_member_rmia_features.csv"
    feature_df.to_csv(feature_path, index=False)

    nonmember_df = load_audit_nonmember_rmia_scores(run_dir, original_reference_attack)
    context_auc = save_context_auc_reports(out_dir, feature_df, nonmember_df)

    summary = {
        **summary_fields,
        "attack_reference_mode": "original_checkpoint_references" if original_reference_attack else "finetuned_checkpoint_references",
        "context_vs_nonmember_auc": context_auc,
        "files": {
            "context_member_rmia_features": str(feature_path),
            "context_member_original_signals": str(sig_dir / "context_member_rmia_signals_original.npy"),
            "context_member_finetuned_signals": str(sig_dir / "context_member_rmia_signals_finetuned.npy"),
            "context_member_memberships": str(out_dir / "context_member_memberships.npy"),
            "context_member_idx_original": str(out_dir / "context_member_idx_original.npy"),
            "context_vs_nonmember_auc_scores": str(out_dir / "context_vs_nonmember_auc_scores.csv"),
            "context_vs_nonmember_original_attack_result": str(out_dir / "report_context_vs_nonmember_original" / "exp" / "attack_result_0.npz"),
            "context_vs_nonmember_finetuned_attack_result": str(out_dir / "report_context_vs_nonmember_finetuned" / "exp" / "attack_result_0.npz"),
        },
        "score_means": {
            "rmia_original": float(np.mean(feature_df["rmia_original"])),
            "rmia_finetuned": float(np.mean(feature_df["rmia_finetuned"])),
            "attack_p_original": float(np.mean(feature_df["attack_p_original"])),
            "attack_p_finetuned": float(np.mean(feature_df["attack_p_finetuned"])),
        },
    }
    summary_path = out_dir / "context_member_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def run_dataset(args: argparse.Namespace, dataset: str) -> dict:
    cleanup_gpu()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else ROOT / "ml_privacy_meter" / "logs" / "ft_tabdpt" / dataset
    rmia_base = run_dir / "rmia"

    out_dir_original_refs = rmia_base / CONTEXT_RMIA_ORIGINAL_REFS_SUBDIR
    out_dir_finetuned_refs = rmia_base / CONTEXT_RMIA_FINETUNED_REFS_SUBDIR

    for d in (out_dir_original_refs, out_dir_finetuned_refs):
        d.mkdir(parents=True, exist_ok=True)
        (d / "signals").mkdir(parents=True, exist_ok=True)

    orig_ckpt = run_dir / "original_tabdpt.ckpt"
    ft_ckpt = run_dir / "finetuned_tabdpt.ckpt"
    splits_path = run_dir / "splits.npz"
    for path in (orig_ckpt, ft_ckpt, splits_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    splits = load_saved_splits(splits_path, NUM_RMIA_REFERENCE_MODELS)
    query_idx = splits["context"].astype(int)
    if args.max_rows is not None:
        query_idx = query_idx[: args.max_rows]
    validate_context_experiment_splits(splits, query_idx)
    nonmember_sample_idx = splits["nonmember"][:len(query_idx)].astype(int)

    source_sig_dir = rmia_base / "signals"
    original_pop = np.load(source_sig_dir / "rmia_signals_original_pop.npy")
    finetuned_original_refs_pop = np.load(source_sig_dir / "rmia_signals_finetuned_original_refs_pop.npy")
    finetuned_ft_refs_pop = np.load(source_sig_dir / "rmia_signals_finetuned_ft_refs_pop.npy")
    if args.max_pop_rows is not None:
        original_pop = original_pop[: args.max_pop_rows].copy()
        finetuned_original_refs_pop = finetuned_original_refs_pop[: args.max_pop_rows].copy()
        finetuned_ft_refs_pop = finetuned_ft_refs_pop[: args.max_pop_rows].copy()

    original_model, _, _ = load_model_and_cfg(orig_ckpt, args.device)
    original_model = freeze_for_eval(original_model)
    num_features = original_model.num_features

    X_all, y_all = load_csv_arrays(dataset, args.data_dir)
    X_all = preprocess_from_context(X_all, splits["context"])
    X_all = reduce_features_for_checkpoint(
        X_all,
        splits["context"],
        num_features,
        args.feature_reduction,
        seed=12345,
    )
    num_classes = int(np.max(y_all)) + 1
    context_idx = splits["context"].astype(int)
    if not np.array_equal(query_idx, context_idx[: len(query_idx)]):
        raise ValueError("Context-member queries must be the fixed context_idx_original order.")
    X_target_context = X_all[context_idx]
    y_target_context = y_all[context_idx]
    X_query = X_all[query_idx]
    y_query = y_all[query_idx]
    pop_idx = splits["test"].astype(int)
    if args.max_pop_rows is not None:
        pop_idx = pop_idx[: args.max_pop_rows]
    if args.max_rows is None and not np.array_equal(X_query, X_target_context):
        raise ValueError("Full context-member audit must query the exact same rows used as target context.")
    if not np.array_equal(y_query, y_target_context[: len(y_query)]):
        raise ValueError("Context query labels do not match the fixed target context labels.")
    context_idx_sha256 = array_fingerprint(context_idx)
    target_context_sha256 = array_fingerprint(X_target_context)

    columns = model_membership_splits(splits)
    model_names = [name for name, _ in columns]
    memberships = membership_matrix(query_idx, columns)

    original_sigs = np.zeros((len(query_idx), len(columns)), dtype=np.float64)
    original_sigs[:, 0] = score_column(
        "original_target",
        original_model,
        X_target_context,
        y_target_context,
        X_query,
        y_query,
        num_classes=num_classes,
        context_size=args.context_size,
        batch_size=args.eval_batch_size,
        temperature=args.temperature,
        device=args.device,
    )
    for col, (name, train_idx) in enumerate(columns[1:], start=1):
        original_sigs[:, col] = score_column(
            f"original_{name}",
            original_model,
            X_all[train_idx],
            y_all[train_idx],
            X_query,
            y_query,
            num_classes=num_classes,
            context_size=args.context_size,
            batch_size=args.eval_batch_size,
            temperature=args.temperature,
            device=args.device,
        )
    if not args.skip_amia:
        amia_ctx_rm_orig, amia_ctx_re_orig = extract_amia_row_max(
            original_model, X_target_context, y_target_context, X_query,
            context_size=args.context_size, device=args.device,
            desc="AMIA context members / original",
        )
        amia_nm_rm_orig, amia_nm_re_orig = extract_amia_row_max(
            original_model, X_target_context, y_target_context, X_all[nonmember_sample_idx],
            context_size=args.context_size, device=args.device,
            desc="AMIA nonmembers / original",
        )
    else:
        amia_ctx_rm_orig = amia_ctx_re_orig = amia_nm_rm_orig = amia_nm_re_orig = None

    del original_model
    cleanup_gpu()

    finetuned_model, _, _ = load_model_and_cfg(ft_ckpt, args.device)
    finetuned_model = freeze_for_eval(finetuned_model)

    finetuned_target_col = score_column(
        "finetuned_target",
        finetuned_model,
        X_target_context,
        y_target_context,
        X_query,
        y_query,
        num_classes=num_classes,
        context_size=args.context_size,
        batch_size=args.eval_batch_size,
        temperature=args.temperature,
        device=args.device,
    )

    # original_refs: target col finetuned, ref cols from original checkpoint
    finetuned_original_refs_sigs = np.zeros((len(query_idx), len(columns)), dtype=np.float64)
    finetuned_original_refs_sigs[:, 0] = finetuned_target_col
    finetuned_original_refs_sigs[:, 1:] = original_sigs[:, 1:]

    # finetuned_refs: target col finetuned, ref cols scored with finetuned checkpoint
    finetuned_ft_refs_sigs = np.zeros((len(query_idx), len(columns)), dtype=np.float64)
    finetuned_ft_refs_sigs[:, 0] = finetuned_target_col
    for col, (name, train_idx) in enumerate(columns[1:], start=1):
        finetuned_ft_refs_sigs[:, col] = score_column(
            f"finetuned_{name}",
            finetuned_model,
            X_all[train_idx],
            y_all[train_idx],
            X_query,
            y_query,
            num_classes=num_classes,
            context_size=args.context_size,
            batch_size=args.eval_batch_size,
            temperature=args.temperature,
            device=args.device,
        )

    if not args.skip_amia:
        amia_ctx_rm_ft, amia_ctx_re_ft = extract_amia_row_max(
            finetuned_model, X_target_context, y_target_context, X_query,
            context_size=args.context_size, device=args.device,
            desc="AMIA context members / finetuned",
        )
        amia_nm_rm_ft, amia_nm_re_ft = extract_amia_row_max(
            finetuned_model, X_target_context, y_target_context, X_all[nonmember_sample_idx],
            context_size=args.context_size, device=args.device,
            desc="AMIA nonmembers / finetuned",
        )
    else:
        amia_ctx_rm_ft = amia_ctx_re_ft = amia_nm_rm_ft = amia_nm_re_ft = None

    del finetuned_model
    cleanup_gpu()

    rmia_original = run_rmia(0, original_sigs, original_pop, memberships, NUM_RMIA_REFERENCE_MODELS)
    rmia_finetuned_original_refs = run_rmia(0, finetuned_original_refs_sigs, finetuned_original_refs_pop, memberships, NUM_RMIA_REFERENCE_MODELS)
    rmia_finetuned_ft_refs = run_rmia(0, finetuned_ft_refs_sigs, finetuned_ft_refs_pop, memberships, NUM_RMIA_REFERENCE_MODELS)
    attack_p_original = run_population_attack(original_sigs[:, 0], original_pop[:, 0])
    attack_p_finetuned = run_population_attack(finetuned_target_col, finetuned_original_refs_pop[:, 0])

    shared_summary = {
        "dataset": dataset,
        "run_dir": str(run_dir),
        "target_membership_definition": "target_member=1 means idx_original is in context_idx_original, not member_idx_original",
        "n_context_member": int(len(query_idx)),
        "n_fixed_target_context": int(len(context_idx)),
        "n_population_scored": int(len(pop_idx)),
        "context_split": str(splits_path) + "::context_idx_original",
        "context_idx_sha256": context_idx_sha256,
        "target_context_sha256_after_preprocessing": target_context_sha256,
        "same_target_context_for_original_and_finetuned": True,
        "finetuned_target_checkpoint": str(ft_ckpt),
        "model_columns": model_names,
    }

    score_cols_original_refs = _build_score_columns(
        query_idx, y_query, memberships, model_names,
        rmia_original, rmia_finetuned_original_refs,
        attack_p_original, attack_p_finetuned,
        original_sigs, finetuned_original_refs_sigs,
    )
    feature_df_original_refs = add_feature_deltas(pd.DataFrame(score_cols_original_refs))

    score_cols_ft_refs = _build_score_columns(
        query_idx, y_query, memberships, model_names,
        rmia_original, rmia_finetuned_ft_refs,
        attack_p_original, attack_p_finetuned,
        original_sigs, finetuned_ft_refs_sigs,
    )
    feature_df_ft_refs = add_feature_deltas(pd.DataFrame(score_cols_ft_refs))

    summary_original_refs = _save_variant(
        out_dir_original_refs,
        query_idx, memberships,
        original_sigs, finetuned_original_refs_sigs,
        feature_df_original_refs,
        run_dir,
        original_reference_attack=True,
        summary_fields={**shared_summary, "reference_checkpoint_for_finetuned_attack": str(orig_ckpt)},
    )

    summary_ft_refs = _save_variant(
        out_dir_finetuned_refs,
        query_idx, memberships,
        original_sigs, finetuned_ft_refs_sigs,
        feature_df_ft_refs,
        run_dir,
        original_reference_attack=False,
        summary_fields={**shared_summary, "reference_checkpoint_for_finetuned_attack": str(ft_ckpt)},
    )

    amia_summary = save_context_amia(
        run_dir,
        query_idx,
        nonmember_sample_idx,
        amia_ctx_rm_orig, amia_ctx_re_orig,
        amia_nm_rm_orig,  amia_nm_re_orig,
        amia_ctx_rm_ft,   amia_ctx_re_ft,
        amia_nm_rm_ft,    amia_nm_re_ft,
    )

    return {
        "dataset": dataset,
        "original_refs": summary_original_refs,
        "ft_refs": summary_ft_refs,
        "amia_context": amia_summary,
    }


def main() -> None:
    args = parse_args()
    datasets = requested_datasets(args)
    runner = run_amia_only if args.amia_only else run_dataset
    summaries = [runner(args, dataset) for dataset in datasets]
    if len(summaries) > 1:
        summary_base = ROOT / "ml_privacy_meter" / "logs" / "ft_tabdpt"
        summary_path = summary_base / "context_member_rmia_batch_summary.json"
        with summary_path.open("w") as f:
            json.dump({"datasets": datasets, "runs": summaries}, f, indent=2)
        print(f"Saved batch summary to {summary_path}")


if __name__ == "__main__":
    main()
