#!/usr/bin/env python3
"""
Attention-based Membership Inference Analysis (AMIA) for TabDPT
===============================================================

Research question
-----------------
Does TabDPT's cross-attention pattern expose membership through attention
concentration, similar to TabICL and TabPFN?

Hypothesis
----------
TabDPT processes tabular data through a stack of cross-attention layers where
K and V are computed from the training context only.  Training members appear
in the K/V projection; non-members do not.  This creates the same side-channel
as TabICL: members receive more concentrated attention weights (higher max_attn,
lower entropy) than non-members.

TabDPT architecture
-------------------
  TabDPTModel.forward(x_src, y_src, task):
      eval_pos = y_src.shape[0]      # = training set size
      x_src = x_src.transpose(0,1)  # (B, L, E) where L = eval_pos + query_size
      for layer in self.transformer_encoder:
          src = layer(src, eval_pos)

  TransformerEncoderLayer.forward(x, eval_pos):
      q = self.q_proj(h)                               # ALL L positions
      k, v = self.kv_proj(h[:, :eval_pos]).chunk(2,-1) # training only
      q, k shapes: (B, n_heads, L, head_dim) / (B, n_heads, eval_pos, head_dim)
      attn = F.scaled_dot_product_attention(q, k, v)

  q_len = L = eval_pos + chunk_size  (all positions)
  k_len = eval_pos = n_context       (training only)
  One SDPA call per layer per forward pass; no ensemble in predict_proba.

SDPA call classification
------------------------
  is_dpt : k_len == n_context AND q_len > k_len
      Pure cross-attention: all positions as query, only training as K/V.
      This is the only SDPA call in the model, so no false-positive risk.

Signals extracted
-----------------
  Per-call, per-head:
    max_attn    : max attention weight across training keys
    neg_entropy : sum(w * log w)  -- more peaked = less negative
    argmax      : key position with highest attention weight

  One call per layer per predict_proba -> n_dpt_calls = nlayers.
  No ensemble: calls map directly to layers (call_idx == layer_idx).

  Key positions 0..n_context-1 map directly to training samples.
  No thinking rows or special tokens in TabDPT.

Outputs  (ml_privacy_meter/logs/<dataset>/<model>/amia/report/exp/)
-------------------------------------------------------------------
  01_member_vs_nonmember_attention.png
  02_attention_vs_rmia.png
  03_roc_comparison.png
  04_layer_auc.png
  05_entropy_divergence.png
  06_layer_head_heatmap.png
  07_argmax_analysis.png
  08_entropy_slope.png
  09_head_variance.png
  attention_summary.csv
  ../log_amia.log

Prerequisites
-------------
    uv run rmia.py --dataset <name> --model tabdpt --mode train
    uv run rmia.py --dataset <name> --model tabdpt --mode load

Usage examples
--------------
    uv run run_attacks/amia/amia_tabdpt.py --dataset locations
    uv run run_attacks/amia/amia_tabdpt.py --dataset purchases10 --gpu 0
    uv run run_attacks/amia/amia_tabdpt.py --dataset locations --model-idx 2
"""

import gc
import os
import subprocess
import sys
import time
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import gaussian_kde, pearsonr
from sklearn.metrics import roc_curve, auc as sk_auc


# ─── data helpers ─────────────────────────────────────────────────────────────

def load_dataset(dataset_name: str, base_dir: str = ".") -> pd.DataFrame:
    """Load a headerless CSV dataset from *base_dir*/<dataset_name>.csv."""
    path = os.path.join(base_dir, dataset_name + ".csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, header=None)


def prepare_tabular_arrays(df: pd.DataFrame):
    """
    Convert a raw DataFrame (last column = label) to float32 numpy arrays.

    String/object columns are coerced to numeric; otherwise integer-encoded
    via category codes.  Returns (X float32, y).
    """
    data = df.copy()
    for col in data.columns[:-1]:
        if pd.api.types.is_string_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
            try:
                data[col] = pd.to_numeric(data[col], errors="raise")
            except (ValueError, TypeError):
                data[col] = data[col].astype("category").cat.codes
    lc = data.columns[-1]
    if pd.api.types.is_string_dtype(data[lc]) or pd.api.types.is_object_dtype(data[lc]):
        try:
            data[lc] = pd.to_numeric(data[lc], errors="raise")
        except (ValueError, TypeError):
            data[lc] = data[lc].astype("category").cat.codes
    arr = data.to_numpy()
    return arr[:, :-1].astype(np.float32), arr[:, -1]


def _dataset_arrays(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return NumPy arrays from a TabularDataset or torch Subset."""
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        base = dataset.dataset
        idx = np.asarray(dataset.indices)
        return base.data[idx], base.targets[idx]
    return dataset.data, dataset.targets


# ─── SDPA hook ────────────────────────────────────────────────────────────────

class SDPACapture:
    """
    Context manager that patches F.scaled_dot_product_attention to extract
    per-call, per-head attention statistics for TabDPT's cross-attention layers.

    Call classification
    -------------------
    is_dpt : k_len == n_context AND q_len > k_len
        Cross-attention: all T positions as query, only training as K/V.
        This is the only attention type in TabDPT, so no false-positive risk.

    Per-head statistics
    -------------------
    max_attn    : (n_heads, chunk_size) -- max attention weight across training keys
    neg_entropy : (n_heads, chunk_size) -- sum(w * log w) per query row
    argmax      : (n_heads, chunk_size) -- training key position with max weight

    Parameters
    ----------
    chunk_size : int -- number of test samples in the current batch
    n_context  : int -- number of training context items (= training set size)
    """

    def __init__(self, chunk_size: int, n_context: int):
        self._orig      = None
        self.records: list = []
        self.chunk_size = chunk_size
        self.n_context  = n_context

    def __enter__(self):
        self.records.clear()
        _orig      = F.scaled_dot_product_attention
        self._orig = _orig
        records    = self.records
        chunk_size = self.chunk_size
        n_context  = self.n_context
        dpt_call_counter = [0]

        def _hook(query, key, value,
                  attn_mask=None, dropout_p=0.0, is_causal=False,
                  scale=None, **kwargs):
            q_len = query.shape[-2]
            k_len = key.shape[-2]

            if k_len == n_context and q_len > k_len:
                with torch.no_grad():
                    d  = query.shape[-1]
                    s  = d ** -0.5 if scale is None else scale
                    sc = torch.matmul(query.float(),
                                      key.float().transpose(-2, -1)) * s
                    if attn_mask is not None:
                        if attn_mask.dtype == torch.bool:
                            sc = sc.masked_fill(~attn_mask, float("-inf"))
                        else:
                            sc = sc + attn_mask.float()
                    w = torch.softmax(sc, dim=-1).cpu()   # (B, n_heads, q_len, k_len)
                    del sc

                    wm = w.numpy()
                    while wm.ndim > 3:
                        wm = wm.mean(0)
                    if wm.ndim == 2:
                        wm = wm[np.newaxis]
                    # wm: (n_heads, q_len, k_len)
                    n_heads = wm.shape[0]

                    # Keep only the last chunk_size query rows (test items)
                    w_rows = wm[:, -chunk_size:, :] if q_len > chunk_size else wm

                    call_idx = dpt_call_counter[0]
                    dpt_call_counter[0] += 1

                    eps   = 1e-12
                    max_a = w_rows.max(axis=2)
                    neg_e = (w_rows * np.log(w_rows + eps)).sum(axis=2)
                    argm  = w_rows.argmax(axis=2).astype(np.int32)

                    records.append({
                        "type":        "dpt",
                        "call_idx":    call_idx,
                        "n_heads":     n_heads,
                        "q_len":       q_len,
                        "k_len":       k_len,
                        "max_attn":    max_a.astype(np.float32),
                        "neg_entropy": neg_e.astype(np.float32),
                        "argmax":      argm,
                    })
                    del w, wm

            extra = {} if scale is None else {"scale": scale}
            extra.update(kwargs)
            return _orig(query, key, value,
                         attn_mask=attn_mask, dropout_p=dropout_p,
                         is_causal=is_causal, **extra)

        F.scaled_dot_product_attention = _hook
        return self

    def __exit__(self, *_):
        F.scaled_dot_product_attention = self._orig




def cleanup_runtime_cache(logger=None):
    """Release Python and CUDA allocator caches between AMIA runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    elif logger is not None:
        logger.info("Cleaned Python GC after AMIA run.")

# ─── metrics ──────────────────────────────────────────────────────────────────

def compute_roc(scores: np.ndarray, labels: np.ndarray):
    """Return (fpr, tpr, auc). Returns 0.5 AUC on NaN/degenerate input."""
    valid = ~np.isnan(scores.ravel())
    if valid.sum() < 2 or len(np.unique(labels.ravel()[valid])) < 2:
        return np.array([0., 1.]), np.array([0., 1.]), 0.5
    fpr, tpr, _ = roc_curve(labels.ravel()[valid], scores.ravel()[valid])
    return fpr, tpr, float(sk_auc(fpr, tpr))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size between two independent groups."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_var = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
    if pooled_var < 1e-30:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled_var))


# ─── signal extraction ────────────────────────────────────────────────────────

def extract_attention_signals(model, X_pool: np.ndarray,
                              n_context: int, batch_size: int, logger,
                              context_size: int | None = None):
    """
    Run inference over all pool samples and collect per-call, per-head signals.

    Returns
    -------
    row_max_all : (n_pool, n_dpt_calls, n_heads) float32
    row_ent_all : (n_pool, n_dpt_calls, n_heads) float32
    row_arg_all : (n_pool, n_dpt_calls, n_heads) int32
    All return None on failure.
    """
    n_pool = len(X_pool)
    logger.info(
        "Effective TabDPT AMIA attention extraction batch_size=%d  context_size=%s",
        batch_size,
        "full" if context_size is None else context_size,
    )

    # Use the model as-is so defense wrappers (e.g. k-anonymity) remain active.
    # SDPACapture is entered as the outer context; any defense SDPA patch applied
    # inside predict_proba becomes the inner hook, so the capture observes
    # post-defense attention weights.  Reach the base model only for dynamo.disable.
    _inner = model
    while hasattr(_inner, "model") and not hasattr(type(_inner), "fit"):
        _inner = _inner.model

    _dynamo_orig_pp = None
    try:
        import torch._dynamo
        _dynamo_orig_pp = _inner.predict_proba
        _inner.predict_proba = torch._dynamo.disable(_inner.predict_proba)
    except Exception:
        pass

    _predict_proba = model.predict_proba
    _predict = model.predict

    row_max_batches, row_ent_batches, row_arg_batches = [], [], []

    n_dpt_calls_ref = None
    n_heads_ref     = None

    try:
        for batch_start in range(0, n_pool, batch_size):
            batch_end = min(batch_start + batch_size, n_pool)
            X_batch   = X_pool[batch_start:batch_end]
            chunk     = batch_end - batch_start

            ctx = SDPACapture(chunk_size=chunk, n_context=n_context)
            with ctx:
                try:
                    _predict_proba(X_batch, context_size=context_size)
                except Exception as exc:
                    logger.warning("predict_proba raised: %s – trying predict()", exc)
                    try:
                        _predict(X_batch)
                    except Exception as exc2:
                        logger.error("predict() failed: %s", exc2)
                        return None, None, None

            dpt_calls = [r for r in ctx.records if r["type"] == "dpt"]

            if batch_start == 0:
                n_dpt_calls_ref = len(dpt_calls)
                n_heads_ref     = dpt_calls[0]["n_heads"] if dpt_calls else 1
                logger.info(
                    "SDPA calls: %d cross-attention (n_heads=%d), %d ignored",
                    n_dpt_calls_ref, n_heads_ref,
                    len(ctx.records) - len(dpt_calls),
                )
                if n_dpt_calls_ref == 0:
                    logger.error(
                        "No TabDPT cross-attention calls captured in first batch; "
                        "expected calls with k_len == n_context(%d) and q_len > k_len.",
                        n_context,
                    )
                    return None, None, None
            elif len(dpt_calls) != n_dpt_calls_ref:
                new_count = len(dpt_calls)
                if new_count < n_dpt_calls_ref:
                    if new_count == 0:
                        logger.error(
                            "DPT attention calls dropped to zero at batch %d-%d; aborting extraction.",
                            batch_start, batch_end,
                        )
                        return None, None, None
                    logger.warning(
                        "DPT-call count dropped from %d to %d -- trimming all %d collected batches",
                        n_dpt_calls_ref, new_count, len(row_max_batches),
                    )
                    row_max_batches = [b[:, :new_count, :] for b in row_max_batches]
                    row_ent_batches = [b[:, :new_count, :] for b in row_ent_batches]
                    row_arg_batches = [b[:, :new_count, :] for b in row_arg_batches]
                    n_dpt_calls_ref = new_count
                else:
                    logger.warning(
                        "DPT-call count changed: expected %d, got %d -- truncating",
                        n_dpt_calls_ref, len(dpt_calls),
                    )
                    dpt_calls = dpt_calls[:n_dpt_calls_ref]

            if not dpt_calls:
                logger.error(
                    "No DPT attention calls captured at batch %d-%d; aborting extraction.",
                    batch_start, batch_end,
                )
                return None, None, None

            rm = np.stack([r["max_attn"][:, :chunk]    for r in dpt_calls], axis=0).transpose(2, 0, 1)
            re = np.stack([r["neg_entropy"][:, :chunk] for r in dpt_calls], axis=0).transpose(2, 0, 1)
            ra = np.stack([r["argmax"][:, :chunk]      for r in dpt_calls], axis=0).transpose(2, 0, 1)

            row_max_batches.append(rm)
            row_ent_batches.append(re)
            row_arg_batches.append(ra)

            del ctx
            gc.collect()

            if batch_end % (batch_size * 4) == 0 or batch_end == n_pool:
                logger.info("  processed %d / %d samples", batch_end, n_pool)

        return (
            np.concatenate(row_max_batches, axis=0),
            np.concatenate(row_ent_batches, axis=0),
            np.concatenate(row_arg_batches, axis=0),
        )
    finally:
        if _dynamo_orig_pp is not None:
            _inner.predict_proba = _dynamo_orig_pp


# ─── plots ────────────────────────────────────────────────────────────────────

MEM_COLOR    = "#086375"
NONMEM_COLOR = "#ee6c4d"


def plot_distributions(signals_dict: dict, mem_mask: np.ndarray,
                       rep_dir: str, dataset_name: str, model_name: str):
    """Plot 01: KDE distributions of DPT attention signals split by membership."""
    labels = ["DPT attention\n(max_attn)", "DPT attention\n(neg_entropy)"]
    keys   = ["row_max", "row_ent"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.suptitle(
        f"Attention concentration: members vs non-members\n{dataset_name} / {model_name}",
        fontsize=11, fontweight="bold",
    )
    for ax, key, label in zip(axes, keys, labels):
        sig = signals_dict[key]
        for mask, name, color in [
            (mem_mask,  "Member",     MEM_COLOR),
            (~mem_mask, "Non-member", NONMEM_COLOR),
        ]:
            vals = sig[mask]
            x    = np.linspace(vals.min(), vals.max(), 200)
            try:
                kde = gaussian_kde(vals)
                ax.plot(x, kde(x), label=name, color=color, linewidth=2)
                ax.fill_between(x, kde(x), alpha=0.15, color=color)
            except Exception:
                ax.hist(vals, bins=25, alpha=0.4, color=color, label=name, density=True)
        _, _, a = compute_roc(sig, mem_mask.astype(int))
        ax.set_title(f"{label}\n(AUC={a:.3f})", fontsize=9)
        ax.set_xlabel("Signal value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(rep_dir, "01_member_vs_nonmember_attention.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention_vs_rmia(signals_dict: dict, rmia_scores: np.ndarray,
                           mem_mask: np.ndarray, rep_dir: str,
                           dataset_name: str, model_name: str):
    """Plot 02: Scatter of DPT max attention vs RMIA score, coloured by membership."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.suptitle(
        f"DPT attention concentration vs RMIA score -- {dataset_name} / {model_name}\n"
        "Members should cluster top-right if attention drives RMIA effectiveness",
        fontsize=10, fontweight="bold",
    )
    sig = signals_dict["row_max"]
    for mask, name, color in [
        (~mem_mask, "Non-member", NONMEM_COLOR),
        (mem_mask,  "Member",     MEM_COLOR),
    ]:
        ax.scatter(sig[mask], rmia_scores[mask], c=color, alpha=0.3,
                   s=8, label=name, rasterized=True)
    try:
        r, p = pearsonr(sig, rmia_scores)
        ax.set_title(f"Pearson r = {r:.3f}  (p = {p:.2e})", fontsize=9)
    except Exception:
        pass
    ax.set_xlabel("DPT attention (max_attn, layer+head avg)", fontsize=9)
    ax.set_ylabel("RMIA score", fontsize=9)
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(rep_dir, "02_attention_vs_rmia.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_comparison(signals_dict: dict, rmia_scores: np.ndarray,
                        mem_mask: np.ndarray, rep_dir: str,
                        dataset_name: str, model_name: str):
    """Plot 03: ROC curves for DPT attention vs RMIA, full and low-FPR zoom."""
    labels_int = mem_mask.astype(int)
    curves = [
        ("Query-context attention (max)", signals_dict["row_max"], MEM_COLOR,   "-"),
        ("Query-context attention (ent)", signals_dict["row_ent"], "#fdc500",    "--"),
        ("RMIA (softmax)",      rmia_scores,             NONMEM_COLOR, "-"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, title, xlim in zip(axes,
                                ["Full ROC", "Low-FPR region (FPR <= 0.10)"],
                                [(0, 1),     (0, 0.10)]):
        for name, scores, color, ls in curves:
            fpr, tpr, a = compute_roc(scores, labels_int)
            ax.plot(fpr, tpr, label=f"{name} (AUC={a:.3f})",
                    color=color, linestyle=ls, linewidth=1.8)
        ax.plot([0, 1], [0, 1], ":", color="gray", linewidth=0.8)
        ax.set_xlim(xlim);  ax.set_ylim(0, 1)
        ax.set_xlabel("FPR", fontsize=9);  ax.set_ylabel("TPR", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7.5);  ax.grid(True, alpha=0.25)
    fig.suptitle(
        f"ROC comparison -- {dataset_name} / {model_name}",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(rep_dir, "03_roc_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layer_auc(row_max_all: np.ndarray, row_ent_all: np.ndarray,
                   mem_mask: np.ndarray, rep_dir: str,
                   dataset_name: str, model_name: str):
    """Plot 04: Per-layer AUC of DPT max_attn and neg_entropy."""
    n_layers   = row_max_all.shape[1]
    x          = np.arange(n_layers)
    labels_int = mem_mask.astype(int)

    auc_max, auc_ent = [], []
    for c in range(n_layers):
        sig_max = row_max_all[:, c, :].mean(axis=1)
        sig_ent = row_ent_all[:, c, :].mean(axis=1)
        _, _, a_max = compute_roc(sig_max, labels_int)
        _, _, a_ent = compute_roc(sig_ent, labels_int)
        auc_max.append(a_max);  auc_ent.append(a_ent)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, auc_max, "o-",  color=MEM_COLOR,    linewidth=1.8, label="row_max AUC")
    ax.plot(x, auc_ent, "s--", color=NONMEM_COLOR, linewidth=1.8, label="row_ent AUC")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_xticks(x);  ax.set_xticklabels([f"L{i}" for i in x])
    ax.set_ylabel("AUC", fontsize=9)
    ax.set_title(
        f"Membership discriminability per DPT layer -- {dataset_name} / {model_name}\n"
        f"{n_layers} transformer layers (one SDPA call per layer)",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(rep_dir, "04_layer_auc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_divergence(row_ent_all: np.ndarray, mem_mask: np.ndarray,
                            rep_dir: str, dataset_name: str, model_name: str):
    """Plot 05: Per-layer mean neg-entropy for members vs non-members (+/-1 std)."""
    n_layers = row_ent_all.shape[1]
    x = np.arange(n_layers)

    mean_mem, mean_nonmem = [], []
    std_mem,  std_nonmem  = [], []
    for c in range(n_layers):
        sig = row_ent_all[:, c, :].mean(axis=1)
        mean_mem.append(sig[mem_mask].mean());     std_mem.append(sig[mem_mask].std())
        mean_nonmem.append(sig[~mem_mask].mean()); std_nonmem.append(sig[~mem_mask].std())

    mean_mem    = np.array(mean_mem);    std_mem    = np.array(std_mem)
    mean_nonmem = np.array(mean_nonmem); std_nonmem = np.array(std_nonmem)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, mean_mem,    "o-",  color=MEM_COLOR,    linewidth=1.8, label="Member")
    ax.plot(x, mean_nonmem, "s--", color=NONMEM_COLOR, linewidth=1.8, label="Non-member")
    ax.fill_between(x, mean_mem - std_mem,       mean_mem + std_mem,       alpha=0.12, color=MEM_COLOR)
    ax.fill_between(x, mean_nonmem - std_nonmem, mean_nonmem + std_nonmem, alpha=0.12, color=NONMEM_COLOR)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_xticks(x);  ax.set_xticklabels([f"L{i}" for i in x])
    ax.set_ylabel("Mean neg-entropy  (less negative = more peaked)", fontsize=9)
    ax.set_title(
        f"Attention entropy divergence per DPT layer -- {dataset_name} / {model_name}\n"
        f"{n_layers} transformer layers",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(rep_dir, "05_entropy_divergence.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layer_head_heatmap(row_max_all: np.ndarray, mem_mask: np.ndarray,
                            rep_dir: str, dataset_name: str, model_name: str):
    """Plot 06: AUC heatmap over (DPT layer x attention head)."""
    import seaborn as sns
    n_layers   = row_max_all.shape[1]
    n_heads    = row_max_all.shape[2]
    labels_int = mem_mask.astype(int)

    auc_mat = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        for h in range(n_heads):
            _, _, a = compute_roc(row_max_all[:, l, h], labels_int)
            auc_mat[l, h] = a

    fig, ax = plt.subplots(figsize=(max(5, n_heads * 1.2 + 2), max(5, n_layers * 0.55 + 2)))
    sns.heatmap(
        auc_mat, ax=ax, cmap="Blues", vmin=0.5, vmax=1.0,
        annot=True, fmt=".3f", annot_kws={"size": 8},
        xticklabels=[f"H{h}" for h in range(n_heads)],
        yticklabels=[f"L{l}" for l in range(n_layers)],
        linewidths=0.4, cbar_kws={"label": "AUC"},
    )
    ax.set_xlabel("Attention head", fontsize=9)
    ax.set_ylabel("DPT layer", fontsize=9)
    ax.set_title(
        f"Membership AUC per (DPT layer x head) -- {dataset_name} / {model_name}",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(rep_dir, "06_layer_head_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_argmax_analysis(row_arg_all: np.ndarray, mem_mask: np.ndarray,
                         rep_dir: str, dataset_name: str, model_name: str):
    """
    Plot 07: Argmax consistency -- does the model keep attending to the same
    training position across layers and heads?

    Key positions 0..n_context-1 map directly to training samples.
    No thinking rows or special tokens in TabDPT.
    """
    n_layers = row_arg_all.shape[1]
    n_pool   = row_arg_all.shape[0]

    flat_all = row_arg_all.reshape(n_pool, -1).astype(np.int32)

    def row_mode_fraction(arr2d):
        from scipy.stats import mode as scipy_mode
        result = np.zeros(arr2d.shape[0], dtype=np.float32)
        for i, row in enumerate(arr2d):
            valid = row[row >= 0]
            if len(valid) == 0:
                result[i] = 0.0
                continue
            m = scipy_mode(valid, keepdims=True).count[0]
            result[i] = m / len(valid)
        return result

    global_consistency = row_mode_fraction(flat_all)

    layer_consistency = np.zeros((n_pool, n_layers), dtype=np.float32)
    for l in range(n_layers):
        layer_flat = row_arg_all[:, l, :].astype(np.int32)
        layer_consistency[:, l] = row_mode_fraction(layer_flat)

    labels_int = mem_mask.astype(int)
    _, _, auc_global = compute_roc(global_consistency, labels_int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        f"Argmax consistency -- {dataset_name} / {model_name}\n"
        "Fraction of (layer x head) pairs attending to the same training-sample position\n"
        "Members should lock onto their training copy -> higher consistency",
        fontsize=10, fontweight="bold",
    )

    ax = axes[0]
    for mask, name, color in [
        (mem_mask,  "Member",     MEM_COLOR),
        (~mem_mask, "Non-member", NONMEM_COLOR),
    ]:
        vals = global_consistency[mask]
        xv   = np.linspace(vals.min(), vals.max(), 200)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(vals)
            ax.plot(xv, kde(xv), label=name, color=color, linewidth=2)
            ax.fill_between(xv, kde(xv), alpha=0.15, color=color)
        except Exception:
            ax.hist(vals, bins=25, alpha=0.4, color=color, label=name, density=True)
    ax.set_title(f"Global consistency  (AUC={auc_global:.3f})", fontsize=9, fontweight="bold")
    ax.set_xlabel("Mode fraction (all DPT layers x heads)", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.25)

    ax = axes[1]
    x = np.arange(n_layers)
    mean_mem_l    = layer_consistency[mem_mask].mean(axis=0)
    mean_nonmem_l = layer_consistency[~mem_mask].mean(axis=0)
    std_mem_l     = layer_consistency[mem_mask].std(axis=0)
    std_nonmem_l  = layer_consistency[~mem_mask].std(axis=0)
    ax.plot(x, mean_mem_l,    "o-",  color=MEM_COLOR,    linewidth=1.8, label="Member")
    ax.plot(x, mean_nonmem_l, "s--", color=NONMEM_COLOR, linewidth=1.8, label="Non-member")
    ax.fill_between(x, mean_mem_l - std_mem_l,       mean_mem_l + std_mem_l,       alpha=0.12, color=MEM_COLOR)
    ax.fill_between(x, mean_nonmem_l - std_nonmem_l, mean_nonmem_l + std_nonmem_l, alpha=0.12, color=NONMEM_COLOR)
    ax.set_xlabel("DPT layer", fontsize=8)
    ax.set_ylabel("Mean consistency (mode fraction)", fontsize=8)
    ax.set_title("Per-layer argmax consistency", fontsize=9, fontweight="bold")
    ax.set_xticks(x);  ax.set_xticklabels([f"L{i}" for i in x], fontsize=7)
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(rep_dir, "07_argmax_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_slope(row_ent_all: np.ndarray, mem_mask: np.ndarray,
                       rep_dir: str, dataset_name: str, model_name: str):
    """Plot 08: Layer-to-layer entropy slope (delta entropy = entropy[L+1] - entropy[L])."""
    n_layers = row_ent_all.shape[1]
    re_mean  = row_ent_all.mean(axis=2)

    delta = np.diff(re_mean, axis=1)

    x = np.arange(n_layers - 1)
    mean_delta_mem    = delta[mem_mask].mean(axis=0)
    mean_delta_nonmem = delta[~mem_mask].mean(axis=0)
    std_delta_mem     = delta[mem_mask].std(axis=0)
    std_delta_nonmem  = delta[~mem_mask].std(axis=0)

    cum_slope  = re_mean[:, -1] - re_mean[:, 0]
    labels_int = mem_mask.astype(int)
    _, _, auc_slope = compute_roc(cum_slope, labels_int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        f"Entropy slope across DPT layers -- {dataset_name} / {model_name}\n"
        "delta entropy = entropy[L+1] - entropy[L]  .  negative = attention sharpening",
        fontsize=10, fontweight="bold",
    )

    ax = axes[0]
    ax.plot(x, mean_delta_mem,    "o-",  color=MEM_COLOR,    linewidth=1.8, label="Member")
    ax.plot(x, mean_delta_nonmem, "s--", color=NONMEM_COLOR, linewidth=1.8, label="Non-member")
    ax.fill_between(x, mean_delta_mem - std_delta_mem,       mean_delta_mem + std_delta_mem,       alpha=0.12, color=MEM_COLOR)
    ax.fill_between(x, mean_delta_nonmem - std_delta_nonmem, mean_delta_nonmem + std_delta_nonmem, alpha=0.12, color=NONMEM_COLOR)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("DPT layer transition  (L -> L+1)", fontsize=8)
    ax.set_ylabel("delta neg-entropy", fontsize=8)
    ax.set_title("Per-transition entropy change", fontsize=9, fontweight="bold")
    ax.set_xticks(x);  ax.set_xticklabels([f"L{i}->{i+1}" for i in x], fontsize=6, rotation=30)
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.25)

    ax = axes[1]
    for mask, name, color in [
        (mem_mask,  "Member",     MEM_COLOR),
        (~mem_mask, "Non-member", NONMEM_COLOR),
    ]:
        vals = cum_slope[mask]
        xv   = np.linspace(vals.min(), vals.max(), 200)
        try:
            kde = gaussian_kde(vals)
            ax.plot(xv, kde(xv), label=name, color=color, linewidth=2)
            ax.fill_between(xv, kde(xv), alpha=0.15, color=color)
        except Exception:
            ax.hist(vals, bins=25, alpha=0.4, color=color, label=name, density=True)
    ax.set_title(f"Cumulative slope  entropy[L{n_layers-1}]-entropy[L0]\n(AUC={auc_slope:.3f})",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel(f"Total entropy change (L0 -> L{n_layers-1})", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(rep_dir, "08_entropy_slope.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_head_variance(row_max_all: np.ndarray, mem_mask: np.ndarray,
                       rep_dir: str, dataset_name: str, model_name: str):
    """
    Plot 09: Head variance per layer.

    For TabDPT there is no ensemble, so we show how spread out the attention
    concentration is across attention heads within each layer.  High head variance
    means different heads focus on very different training samples.

    head_var[i, l] = variance of max_attn across n_heads for sample i at layer l.
    """
    n_layers   = row_max_all.shape[1]
    head_var   = row_max_all.var(axis=2)   # (n_pool, n_layers)
    labels_int = mem_mask.astype(int)

    x = np.arange(n_layers)
    mean_var_mem    = head_var[mem_mask].mean(axis=0)
    mean_var_nonmem = head_var[~mem_mask].mean(axis=0)
    std_var_mem     = head_var[mem_mask].std(axis=0)
    std_var_nonmem  = head_var[~mem_mask].std(axis=0)

    total_var = head_var.mean(axis=1)
    _, _, auc_var = compute_roc(total_var, labels_int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        f"Head variance per DPT layer -- {dataset_name} / {model_name}\n"
        "Variance of max_attn across attention heads within each layer",
        fontsize=10, fontweight="bold",
    )

    ax = axes[0]
    ax.plot(x, mean_var_mem,    "o-",  color=MEM_COLOR,    linewidth=1.8, label="Member")
    ax.plot(x, mean_var_nonmem, "s--", color=NONMEM_COLOR, linewidth=1.8, label="Non-member")
    ax.fill_between(x, mean_var_mem - std_var_mem,       mean_var_mem + std_var_mem,       alpha=0.12, color=MEM_COLOR)
    ax.fill_between(x, mean_var_nonmem - std_var_nonmem, mean_var_nonmem + std_var_nonmem, alpha=0.12, color=NONMEM_COLOR)
    ax.set_xlabel("DPT layer", fontsize=8)
    ax.set_ylabel("Mean intra-layer head variance", fontsize=8)
    ax.set_title("Per-layer head variance", fontsize=9, fontweight="bold")
    ax.set_xticks(x);  ax.set_xticklabels([f"L{i}" for i in x], fontsize=7)
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.25)

    ax = axes[1]
    for mask, name, color in [
        (mem_mask,  "Member",     MEM_COLOR),
        (~mem_mask, "Non-member", NONMEM_COLOR),
    ]:
        vals = total_var[mask]
        xv   = np.linspace(vals.min(), vals.max(), 200)
        try:
            kde = gaussian_kde(vals)
            ax.plot(xv, kde(xv), label=name, color=color, linewidth=2)
            ax.fill_between(xv, kde(xv), alpha=0.15, color=color)
        except Exception:
            ax.hist(vals, bins=25, alpha=0.4, color=color, label=name, density=True)
    ax.set_title(f"Total head variance  (AUC={auc_var:.3f})", fontsize=9, fontweight="bold")
    ax.set_xlabel("Mean head variance across all layers", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(rep_dir, "09_head_variance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── pipeline (re-usable by eval_defenses.py) ────────────────────────────────

def run_amia_pipeline(
    model,
    X_pool: np.ndarray,
    mem: np.ndarray,
    rmia_scores: np.ndarray,
    n_context: int,
    batch_size: int,
    logger,
    sig_dir: str,
    exp_dir: str,
    dataset_name: str,
    model_name: str,
    model_idx: int = 0,
    mode: str = "load",
    max_row_calls: int = None,
    max_col_calls: int = None,
    context_size: int | None = None,
) -> dict:
    """Extract TabDPT cross-attention signals and generate AMIA plots.

    The signature mirrors the TabPFN/TabICL reusable pipelines.  Even though
    the captured call is TabDPT cross-attention, the saved AMIA signal names are
    the shared ``row_*`` names used by all AMIA backends.
    """
    del max_row_calls, max_col_calls
    if context_size is None:
        context_size = n_context

    os.makedirs(sig_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    cache = os.path.join(sig_dir, f"attn_signals_{model_idx}.npz")
    if mode == "train" and os.path.exists(cache):
        os.remove(cache)
        logger.info("Normal AMIA run: deleted cached signals %s", cache)

    row_arg_all = None
    if os.path.exists(cache):
        npz = np.load(cache)
        row_max_all = npz["row_max_all"] if "row_max_all" in npz else npz["dpt_max_all"]
        row_ent_all = npz["row_ent_all"] if "row_ent_all" in npz else npz["dpt_ent_all"]
        row_arg_all = (
            npz["row_arg_all"] if "row_arg_all" in npz
            else npz["dpt_arg_all"] if "dpt_arg_all" in npz
            else None
        )
        if row_max_all.shape[0] != len(mem):
            msg = (
                f"Cached TabDPT AMIA signals have {row_max_all.shape[0]} rows, "
                f"but current labels have {len(mem)} rows."
            )
            if mode == "load":
                raise ValueError(msg + " Re-run AMIA without --plots-only to refresh the cache.")
            logger.warning("%s Re-extracting.", msg)
            os.remove(cache)
            row_arg_all = None
        elif row_arg_all is None:
            logger.warning("Cache missing row_arg_all -- re-extracting.")
        else:
            logger.info("Loaded cached TabDPT signals from %s  shape=%s", cache, row_max_all.shape)

    if not os.path.exists(cache) or row_arg_all is None:
        logger.info("Extracting TabDPT cross-attention signals")
        row_max_all, row_ent_all, row_arg_all = extract_attention_signals(
            model,
            X_pool,
            n_context,
            batch_size,
            logger,
            context_size=context_size,
        )
        if row_max_all is None:
            raise RuntimeError("Attention extraction failed -- see log for details.")
        np.savez_compressed(
            cache,
            row_max_all=row_max_all,
            row_ent_all=row_ent_all,
            row_arg_all=row_arg_all,
        )
        logger.info("Cached TabDPT signals to %s  shape=%s", cache, row_max_all.shape)

    logger.info("Signal shapes: row_max_all=%s  row_ent_all=%s",
                row_max_all.shape, row_ent_all.shape)

    row_max = row_max_all.mean(axis=(1, 2))
    row_ent = row_ent_all.mean(axis=(1, 2))
    if len(rmia_scores) != len(mem):
        raise ValueError(
            f"RMIA score length ({len(rmia_scores)}) does not match membership labels ({len(mem)})."
        )

    signals_dict = {"row_max": row_max, "row_ent": row_ent}
    df_out = pd.DataFrame({
        "member":     mem.astype(int),
        "rmia_score": rmia_scores,
        "row_max":    row_max,
        "row_ent":    row_ent,
    })
    df_out.to_csv(os.path.join(exp_dir, "attention_summary.csv"), index=False)

    logger.info("Per-signal AUC and Cohen's d (layer+head averaged):")
    scalars: dict = {}
    for key, name in [("row_max", "row_max"), ("row_ent", "row_ent"), ("rmia_score", "RMIA")]:
        vals = df_out[key].values
        _, _, a = compute_roc(vals, mem.astype(int))
        d = cohens_d(vals[mem], vals[~mem])
        logger.info("  AUC  %-20s  %.4f  Cohen's d=%+.4f", name, a, d)
        scalars[key + "_auc"] = a
        scalars[key + "_d"] = d

    plot_distributions(signals_dict, mem, exp_dir, dataset_name, model_name)
    plot_attention_vs_rmia(signals_dict, rmia_scores, mem, exp_dir, dataset_name, model_name)
    plot_roc_comparison(signals_dict, rmia_scores, mem, exp_dir, dataset_name, model_name)
    plot_layer_auc(row_max_all, row_ent_all, mem, exp_dir, dataset_name, model_name)
    plot_entropy_divergence(row_ent_all, mem, exp_dir, dataset_name, model_name)
    plot_layer_head_heatmap(row_max_all, mem, exp_dir, dataset_name, model_name)
    if row_arg_all is not None:
        plot_argmax_analysis(row_arg_all, mem, exp_dir, dataset_name, model_name)
    else:
        logger.warning("Skipping plot 07: row_arg_all not available in cache.")
    plot_entropy_slope(row_ent_all, mem, exp_dir, dataset_name, model_name)
    plot_head_variance(row_max_all, mem, exp_dir, dataset_name, model_name)

    return scalars


# ─── main ─────────────────────────────────────────────────────────────────────

def main(dataset_name: str, model_name: str, gpu, batch_size: int,
         model_idx: int, plots_only: bool = False, seed: int | None = None,
         skip_existing: bool = False, context_pct: float | None = None):
    sys.path.append(str(Path(__file__).parent.parent.parent / "ml_privacy_meter"))
    from models.utils import load_models
    from util import setup_log

    base_log = os.path.join("ml_privacy_meter", "logs", dataset_name, model_name.lower())
    run_root = os.path.join(base_log, f"seed{seed}") if seed is not None else base_log
    if context_pct is not None and context_pct < 100.0:
        rmia_log = os.path.join(run_root, f"rmia_ctx{int(context_pct)}")
        attn_log = os.path.join(run_root, f"amia_ctx{int(context_pct)}")
    else:
        rmia_log = os.path.join(run_root, "rmia")
        attn_log = os.path.join(run_root, "amia")
    sig_dir  = os.path.join(attn_log, "signals")
    rep_dir  = os.path.join(attn_log, "report")
    exp_dir  = os.path.join(rep_dir, "exp")
    for d in (sig_dir, rep_dir, exp_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    logger = setup_log(rep_dir, "amia", True)
    t0 = time.time()

    summary_csv = os.path.join(exp_dir, "attention_summary.csv")
    cache = os.path.join(sig_dir, f"attn_signals_{model_idx}.npz")
    if skip_existing and os.path.exists(summary_csv) and os.path.exists(cache):
        cleanup_runtime_cache(logger)
        return

    config_file = f"ml_privacy_meter/configs/{dataset_name}_{model_name}.yaml"
    with open(config_file, "r") as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    if seed is not None:
        configs.setdefault("run", {})["random_seed"] = seed
        configs.setdefault("train", {})["random_state"] = seed

    np.random.seed(configs["run"].get("random_seed", 12345))

    if plots_only:
        if not os.path.exists(cache):
            raise FileNotFoundError(
                f"No cached signals at {cache}\n"
                f"Run without --plots-only first to extract signals."
            )
        logger.info("--plots-only: loading signals from cache, skipping model inference.")
        npz         = np.load(cache)
        row_max_all = npz["row_max_all"] if "row_max_all" in npz else npz["dpt_max_all"]
        row_ent_all = npz["row_ent_all"] if "row_ent_all" in npz else npz["dpt_ent_all"]
        row_arg_all = (
            npz["row_arg_all"] if "row_arg_all" in npz
            else npz["dpt_arg_all"] if "dpt_arg_all" in npz
            else None
        )
        del row_max_all, row_ent_all, row_arg_all

        summary_csv = os.path.join(exp_dir, "attention_summary.csv")
        if not os.path.exists(summary_csv):
            raise FileNotFoundError(
                f"No summary CSV at {summary_csv} -- run without --plots-only first."
            )
        df_existing = pd.read_csv(summary_csv)
        mem         = df_existing["member"].values.astype(bool)
        rmia_scores = df_existing["rmia_score"].values
        logger.info("Loaded %d samples from %s", len(mem), summary_csv)
        run_amia_pipeline(
            model=None,
            X_pool=np.empty((0, 0)),
            mem=mem,
            rmia_scores=rmia_scores,
            n_context=0,
            batch_size=batch_size,
            logger=logger,
            sig_dir=sig_dir,
            exp_dir=exp_dir,
            dataset_name=dataset_name,
            model_name=model_name,
            model_idx=model_idx,
            mode="load",
        )
        logger.info("Done in %.1f s", time.time() - t0)
        cleanup_runtime_cache(logger)
        return

    else:
        torch.manual_seed(configs["run"].get("random_seed", 12345))

        if gpu is not None:
            configs.setdefault("train", {})["device"] = "cuda:0"
            configs.setdefault("audit", {})["device"] = "cuda:0"
        else:
            configs.setdefault("train", {})["device"] = "cpu"
            configs.setdefault("audit", {})["device"] = "cpu"

        data_dir      = configs["data"]["data_dir"]
        df_raw        = load_dataset(dataset_name, data_dir)
        X, y          = prepare_tabular_arrays(df_raw)
        if seed is not None:
            split_path = os.path.join(rmia_log, "splits", "dataset_permutation.npy")
            if not os.path.exists(split_path):
                raise FileNotFoundError(
                    f"Seeded RMIA split not found: {split_path}\n"
                    f"Run RMIA first with --seed {seed}."
                )
            order = np.load(split_path)
            X = X[order]
            y = y[order]
            logger.info(
                "Applied seeded RMIA dataset permutation for seed=%d from %s",
                seed,
                split_path,
            )
        training_size = int(len(y) * 0.75)
        logger.info("Dataset: %d rows total, candidate pool (75%%): %d", len(y), training_size)

        models_list, memberships = load_models(rmia_log, None, None, configs, logger)
        if models_list is None or model_idx >= len(models_list):
            n_found = len(models_list) if models_list is not None else 0
            raise RuntimeError(f"model_idx={model_idx} out of range (found {n_found} models)")

        model = models_list[model_idx]
        from audit import sample_auditing_dataset
        from dataset.tabular import TabularDataset

        # Match the exact RMIA audit universe. For context-size runs, memberships
        # has one column per retained context-pool row, which can be smaller than
        # the full 75% candidate training pool.
        pool_size = memberships.shape[1]
        if pool_size > training_size:
            raise ValueError(
                f"RMIA memberships have {pool_size} columns, larger than candidate pool {training_size}."
            )
        dataset = TabularDataset(X[:pool_size], y[:pool_size])
        logger.info("Using %d rows from the RMIA membership pool for TabDPT AMIA reconstruction.", pool_size)
        np.random.seed(configs["run"].get("random_seed", 12345))
        auditing_dataset, auditing_membership = sample_auditing_dataset(
            configs, dataset, logger, memberships
        )
        X_pool, _y_pool = _dataset_arrays(auditing_dataset)
        n_pool = len(X_pool)
        mem = auditing_membership[model_idx].astype(bool)
        n_members    = int(mem.sum())
        context_size = n_members   # TabDPT k_len = eval_pos = training set size
        logger.info(
            "Target model %d: %d members, %d non-members in RMIA audit pool",
            model_idx, n_members, n_pool - n_members,
        )
        logger.info(
            "Reconstructed TabDPT AMIA audit pool from seeded RMIA memberships: "
            "X_pool=%s, auditing_membership=%s",
            X_pool.shape,
            auditing_membership.shape,
        )

        for i, m in enumerate(models_list):
            if i != model_idx and hasattr(m, "to"):
                try: m.to("cpu")
                except Exception: pass
        desired_device = "cuda:0" if gpu is not None else "cpu"
        if desired_device is not None and hasattr(model, "to"):
            try:
                model.to(desired_device)
                logger.info("Moved TabDPT target model to %s for AMIA extraction.", desired_device)
            except Exception as exc:
                logger.warning("Could not move TabDPT target model to %s: %s", desired_device, exc)
        del models_list
        gc.collect()

        rmia_sig_path = os.path.join(rmia_log, "signals", "rmia_signals.npy")
        rmia_pop_path = os.path.join(rmia_log, "signals", "rmia_signals_pop.npy")
        if not os.path.exists(rmia_sig_path):
            raise FileNotFoundError(
                f"RMIA signals not found: {rmia_sig_path}\n"
                f"Run first: uv run rmia.py --dataset {dataset_name} --model {model_name} --mode load"
            )
        if not os.path.exists(rmia_pop_path):
            raise FileNotFoundError(
                f"RMIA population signals not found: {rmia_pop_path}\n"
                f"Run first: uv run rmia.py --dataset {dataset_name} --model {model_name} --mode load"
            )
        rmia_signals     = np.load(rmia_sig_path)
        rmia_signals_pop = np.load(rmia_pop_path)
        num_ref_models   = configs["audit"]["num_ref_models"]
        from attacks import run_rmia, tune_offline_a
        if rmia_signals.shape[0] != n_pool or auditing_membership.shape[1] != n_pool:
            raise ValueError(
                "RMIA signal rows do not match reconstructed TabDPT AMIA audit pool: "
                f"signals={rmia_signals.shape[0]}, pool={n_pool}, memberships={auditing_membership.shape}"
            )

        # auditing_membership is (n_models, n_audit_samples); run_rmia expects (n_samples, n_models)
        # tune_offline_a sweeps α on the paired model (same procedure as the original RMIA evaluation)
        best_offline_a, _, _ = tune_offline_a(
            target_model_idx=model_idx,
            all_signals=rmia_signals,
            population_signals=rmia_signals_pop,
            all_memberships=auditing_membership.T,
            logger=logger,
        )
        logger.info("Best offline_a for target model %d: %.2f", model_idx, best_offline_a)
        rmia_scores = run_rmia(
            target_model_idx=model_idx,
            all_signals=rmia_signals,
            population_signals=rmia_signals_pop,
            all_memberships=auditing_membership.T,
            num_reference_models=num_ref_models,
            offline_a=best_offline_a,
        )
        logger.info(
            "Loaded RMIA signals: shape=%s  pop shape=%s  num_ref=%d  using model_idx=%d",
            rmia_signals.shape, rmia_signals_pop.shape, num_ref_models, model_idx,
        )
        run_amia_pipeline(
            model,
            X_pool,
            mem,
            rmia_scores,
            context_size,
            batch_size,
            logger,
            sig_dir,
            exp_dir,
            dataset_name,
            model_name,
            model_idx=model_idx,
            mode="train",
        )
        logger.info("Done in %.1f s", time.time() - t0)

        if seed is not None:
            from run_attacks.seed_summary import update_seed_row
            _attack_label = (f"amia_ctx{int(context_pct)}"
                             if context_pct is not None and context_pct < 100.0
                             else "amia")
            update_seed_row(_attack_label, int(seed), Path(rep_dir), Path(base_log))
        cleanup_runtime_cache(logger)
        return


# ─── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Attention-based explanation of RMIA effectiveness on TabDPT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    model_name = "tabdpt"
    parser.add_argument("--dataset",     type=str, default="locations")
    parser.add_argument("--model-idx",   type=int, default=0)
    parser.add_argument("--gpu",         type=str, default=None)
    parser.add_argument("--batch-size",  type=int, default=200)
    parser.add_argument("--skip-config", action="store_true")
    parser.add_argument("--plots-only",  action="store_true")
    parser.add_argument("--context-pct", type=float, default=None,
                        help="Context-size percentage for sweep (e.g. 50 reads from rmia_ctx50/, writes to amia_ctx50/). Default: None = full context (rmia/).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a seed if cached AMIA signals and attention_summary.csv already exist.")
    parser.add_argument("--seed",        type=int, default=1,
                        help="Seed number for artifacts under logs/<dataset>/<model>/seed<seed>/{rmia,amia}/. Default: 1.")
    parser.add_argument("--seeds",       type=str, default=None,
                        help="Comma-separated seeded AMIA trials to run, e.g. 1,2,3,4,5.")
    args = parser.parse_args()

    if args.seeds is not None:
        seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
        if not seeds:
            raise ValueError("--seeds must contain at least one integer seed.")

        base_argv = sys.argv[1:]
        filtered_argv = []
        skip_next = False
        for item in base_argv:
            if skip_next:
                skip_next = False
                continue
            if item in {"--seeds", "--seed"}:
                skip_next = True
                continue
            if item.startswith("--seeds=") or item.startswith("--seed="):
                continue
            filtered_argv.append(item)

        failures = []
        for seed in seeds:
            if args.skip_existing:
                _base  = os.path.join("ml_privacy_meter", "logs",
                                      args.dataset, "tabdpt", f"seed{seed}")
                _cache = os.path.join(_base, "amia", "signals",
                                      f"attn_signals_{args.model_idx}.npz")
                _csv   = os.path.join(_base, "amia", "report", "exp",
                                      "attention_summary.csv")
                if os.path.exists(_cache) and os.path.exists(_csv):
                    print(f"[SEED] TabDPT AMIA seed={seed} already done, skipping.")
                    continue
            cmd = [sys.executable, __file__, *filtered_argv, "--seed", str(seed)]
            print(f"[SEED] Running TabDPT AMIA seed={seed}")
            code = subprocess.run(cmd, check=False).returncode
            if code != 0:
                failures.append(seed)
                print(f"[FAIL] seed={seed} exit_code={code}")
                break

        if failures:
            raise SystemExit(1)
        print(f"[SEED] Completed TabDPT AMIA for seeds: {seeds}")
        raise SystemExit(0)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from configs import ensure_dataset_ready
    ensure_dataset_ready(
        dataset_name=args.dataset,
        model_name=model_name,
        algorithm="RMIA",
        skip_if_exists=args.skip_config,
    )

    try:
        main(
            dataset_name=args.dataset,
            model_name=model_name,
            gpu=args.gpu,
            batch_size=args.batch_size,
            model_idx=args.model_idx,
            plots_only=args.plots_only,
            seed=args.seed,
            skip_existing=args.skip_existing,
            context_pct=args.context_pct,
        )
    except Exception as e:
        import traceback
        _out = Path("results_visualizations")
        _out.mkdir(parents=True, exist_ok=True)
        with (_out / "amia_failed_runs.csv").open("a") as fh:
            fh.write(f"{args.dataset},{model_name},{str(e).replace(',', ';')}\n")
        print(f"[FAILED] {args.dataset} + {model_name}: {e}")
        traceback.print_exc()
        raise SystemExit(1)
