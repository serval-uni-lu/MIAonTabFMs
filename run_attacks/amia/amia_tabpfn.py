#!/usr/bin/env python3
"""
Attention-based Membership Inference Analysis (AMIA) for TabPFN
===============================================================

Research question
-----------------
Why is RMIA disproportionately effective against TabPFN compared to classical
models such as RF or LightGBM?

Hypothesis
----------
TabPFN processes tabular data through an in-context Transformer that attends
over the *training set at inference time*.  Training members are permanently
embedded in the context window, so the model can attend to them more strongly
and sharply than to non-members.  This creates a measurable side-channel:
members produce more *concentrated* attention weights, which correlates
directly with the higher RMIA softmax scores they receive.

TabPFN attention mechanisms
----------------------------
TabPFN v2 has two structurally distinct attention patterns per layer:

  Row attention (cross-attention)
      Query  : test items      (shape q_len = chunk_size)
      Key/Val: training context (shape k_len = n_context)
      q_len != k_len  →  each test item attends over all training items.
      MIA signal: members are already *in* the context, so the model can
      pinpoint them → higher max attention, lower entropy (more peaked).

  Column attention (feature self-attention)
      Query = Key = feature representations (small square matrix).
      q_len == k_len and q_len << n_context.
      Captures feature-to-feature interactions.  These calls are batch-level
      diagnostics rather than clean per-record membership evidence because the
      same value is assigned to every sample in the captured batch.

Signals extracted
-----------------
  Per-call, per-head (row attention only):
    max_attn    : maximum attention weight across keys — concentration proxy
    neg_entropy : sum(w * log w) — more negative = more uniform distribution

  Aggregated diagnostics (col attention):
    col_max, col_ent : same statistics averaged to a scalar per batch.  Saved
                       for troubleshooting, but excluded from the main AMIA
                       evidence plots.

TabPFN v2.6 architecture  (TabPFNV2p6Config)
---------------------------------------------
  nlayers=24 · emsize=192 · nhead=3 (cross/row attention) · features_per_group=3
  num_thinking_rows is read from the loaded TabPFN checkpoint/config and
  prepended to each training context when present.

  Thinking rows explain why k_len can be n_train + n_thinking in
  cross-attention tensors.
  The base ModelConfig uses nhead=6 for self-attention; nhead=3 is specific to the
  per-column-inter-row (cross) attention — which is the MIA-relevant signal.

SDPA call structure at inference
----------------------------------
Each encoder layer processes 8 feature blocks.  Per feature block:
  8 small self-attention calls  (feature × feature)
  8 cross-attention calls       (test items → training context, nhead=3)
  8 large self-attention calls  (item × item, nhead=6)

Total cross-attention SDPA calls: 24 layers × 8 feature-blocks × 8 = 1536.
The hook groups these 1536 calls into 24 encoder-layer signals by averaging
every 64 consecutive calls (using the _TABPFN_N_LAYERS=24 module constant).

NOTE — Key layout: positions 0..63 are thinking rows; positions 64..N-1 are
actual training samples.  max_attn and neg_entropy are computed after removing
and re-normalising away the thinking rows, so they measure attention
concentration over real training-context rows.  The argmax analysis keeps the
full key range initially, then filters positions ≥ n_thinking so
only real training rows contribute to consistency.

Per-head tracking
-----------------
The hook retains the 3 cross-attention heads separately (n_heads=3).
This enables identifying which head is most discriminative for membership.

Outputs  (ml_privacy_meter/logs/<dataset>/<model>/amia/report/)
-------
  01_member_vs_nonmember_attention.png
      KDE distributions of row-attention signals split by membership.

  02_attention_vs_rmia.png
      Scatter: row-attention concentration vs RMIA score, coloured by membership.

  03_roc_comparison.png
      ROC curves: row-attn / col-attn / RMIA / (row-attn + RMIA).

  04_layer_auc.png
      Top: per-call AUC of row_max and row_ent with head/tail regions shaded.
      Bottom: KDE of row_max at head layers vs tail layers (member/non-member),
      showing how the membership signal grows from early to deep layers.

  05_entropy_divergence.png
      Top: mean neg-entropy per call for members vs non-members with head/tail
      shading and gap annotations.
      Bottom: KDE of neg-entropy at head vs tail layers — the growing gap
      confirms entropy separation builds up with depth.

  attention_summary.csv   — per-sample flat signals + RMIA score + label
  log_amia.log            — per-signal AUC summary and runtime

Prerequisites
-------------
    uv run run_attacks/rmia.py --dataset <name> --model tabpfn

Usage examples
--------------
    uv run run_attacks/amia/amia_tabpfn.py --dataset locations
    uv run run_attacks/amia/amia_tabpfn.py --dataset locations --model real-tabpfn
    uv run run_attacks/amia/amia_tabpfn.py --dataset purchases10 --gpu 0 --batch-size 100
    uv run run_attacks/amia/amia_tabpfn.py --dataset locations --model-idx 2
"""

import gc
import os
import subprocess
import sys
import time
import warnings
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


def infer_tabpfn_thinking_rows(model, default: int | None = 0) -> int:
    """Return the loaded TabPFN checkpoint's thinking-row count."""
    try:
        from run_defenses.tabfm_introspection import infer_num_thinking_rows
    except ModuleNotFoundError:
        from tabfm_introspection import infer_num_thinking_rows
    return infer_num_thinking_rows(model, default=default)


# ─── SDPA hook ────────────────────────────────────────────────────────────────

class SDPACapture:
    """
    Context manager that patches F.scaled_dot_product_attention to extract
    per-call, per-head attention statistics without retaining the full weight
    matrix (which would OOM for TabPFN's 64 feature blocks × 12+ calls).

    Call classification by tensor shape
    ------------------------------------
    "row"  — q_len != k_len  (test items attending to training context)
    "col"  — q_len == k_len and q_len < n_context // 4  (feature attention)
    ignored — large square self-attention (item × item)

    Per-head statistics stored per call
    ------------------------------------
    max_attn    : (n_heads, n_query_rows) — max attention weight across keys
    neg_entropy : (n_heads, n_query_rows) — sum(w * log w) per query row

    For row attention, n_query_rows = chunk_size (test items only — the
    context rows prepended to the query are discarded).
    For col attention, n_query_rows = n_features (broadcast to samples later).

    Parameters
    ----------
    chunk_size : int   — number of test samples in the current batch
    n_context  : int   — number of training context items
    """

    def __init__(
        self,
        chunk_size: int,
        n_context: int,
        n_thinking: int | None = None,
        max_row_calls: int | None = None,
        max_col_calls: int | None = None,
    ):
        self._orig      = None
        self.records: list = []
        self.total_calls = 0   # every SDPA call, regardless of classification
        self.filtered_row_calls = 0  # q_len==chunk_size but k_len != expected cross-attn k_len
        self.degenerate_skipped = 0  # odd-indexed ambiguous calls dropped in degenerate mode
        self.chunk_size = chunk_size
        self.n_context  = n_context
        self.n_thinking = None if n_thinking is None else max(0, int(n_thinking))
        self.max_row_calls = None if max_row_calls is None else max(0, int(max_row_calls))
        self.max_col_calls = None if max_col_calls is None else max(0, int(max_col_calls))

    def __enter__(self):
        self.records.clear()
        self.total_calls = 0
        self.filtered_row_calls = 0
        self.degenerate_skipped = 0
        _orig      = F.scaled_dot_product_attention
        self._orig = _orig
        records    = self.records
        chunk_size = self.chunk_size
        n_context  = self.n_context
        n_thinking_config = self.n_thinking
        inferred_n_thinking = [n_thinking_config]
        max_row_calls = self.max_row_calls
        max_col_calls = self.max_col_calls
        call_counter = self   # reference so the closure can mutate total_calls
        row_call_counter   = [0]   # mutable container so the closure can increment
        col_call_counter   = [0]
        ambig_call_counter = [0]   # counts every ambiguous (q==k==chunk_size) call

        def _hook(query, key, value,
                  attn_mask=None, dropout_p=0.0, is_causal=False,
                  scale=None, **kwargs):
            call_counter.total_calls += 1
            q_len = query.shape[-2]
            k_len = key.shape[-2]

            # Cross-attention (test → training context):
            #   q_len == chunk_size  (test items are the queries)
            #   k_len == n_context + n_thinking  (context + thinking rows are the keys)
            # The k_len check filters spurious calls that happen to have q_len==chunk_size
            # but attend to a different key sequence.
            #
            # Degenerate case: when n_context + n_thinking == chunk_size, BOTH cross-attention
            # (test→context) and context-internal self-attention (context+thinking→itself)
            # produce the same (q_len, k_len) shape.  In this implementation we treat this
            # as unsupported because shape-only disambiguation is unreliable.
            expected_k = (
                n_context + inferred_n_thinking[0]
                if inferred_n_thinking[0] is not None
                else None
            )
            # Track calls passing q_len gate but with wrong k_len (unexpected call type).
            if (q_len == chunk_size) and expected_k is not None and (k_len != expected_k):
                call_counter.filtered_row_calls += 1
            if (
                q_len == chunk_size
                and k_len >= n_context
                and (
                    (expected_k is None and q_len != k_len)
                    or (expected_k is not None and k_len == expected_k)
                )
            ):
                if inferred_n_thinking[0] is None:
                    inferred_n_thinking[0] = max(0, k_len - n_context)
                if q_len == k_len:
                    ambig_call_counter[0] += 1
                    is_row = False
                    call_counter.degenerate_skipped += 1
                else:
                    is_row = True  # unambiguous: q_len != k_len
            else:
                is_row = False
            is_col = (not is_row and q_len == k_len and q_len < max(4, n_context // 4))

            call_idx = -1
            if is_row:
                call_idx = row_call_counter[0]
                row_call_counter[0] += 1
                if max_row_calls is not None and call_idx >= max_row_calls:
                    is_row = False
            elif is_col:
                col_idx = col_call_counter[0]
                col_call_counter[0] += 1
                if max_col_calls is not None and col_idx >= max_col_calls:
                    is_col = False

            if is_row or is_col:
                with torch.no_grad():
                    d  = query.shape[-1]
                    s  = d ** -0.5 if scale is None else scale
                    sc = torch.matmul(query.float(),
                                      key.float().transpose(-2, -1)) * s
                    if is_causal:
                        rows = torch.arange(q_len, device=sc.device)[:, None]
                        cols = torch.arange(k_len, device=sc.device)[None, :]
                        sc = sc.masked_fill(cols > rows, float("-inf"))
                    if attn_mask is not None:
                        if attn_mask.dtype == torch.bool:
                            sc = sc.masked_fill(~attn_mask, float("-inf"))
                        else:
                            sc = sc + attn_mask.float()
                    w = torch.softmax(sc, dim=-1)
                    w = w.cpu()
                    del sc

                    eps = 1e-12
                    wm  = w.numpy()

                    if is_row:
                        # Row cross-attention: batch dim is feature groups → average to
                        # get a single (n_heads, q_len, k_len) signal, then slice test items.
                        while wm.ndim > 3:
                            wm = wm.mean(0)
                        if wm.ndim == 2:
                            wm = wm[np.newaxis]
                        n_heads  = wm.shape[0]
                        w_rows = wm
                        # argmax over ALL key positions (including thinking rows) so that
                        # positions stay in [0, n_context+n_thinking) and the thinking-row
                        # mask in plot_argmax_analysis still works correctly.
                        argm  = w_rows.argmax(axis=2).astype(np.int32)
                        # max_attn and neg_entropy: exclude the n_thinking prepended rows
                        # so the signals reflect attention to TRAINING SAMPLES only.
                        # Non-members also attend strongly to thinking rows, contaminating
                        # the signal and causing layer-level AUC to be inconsistent.
                        n_thinking = int(inferred_n_thinking[0] or 0)
                        w_train = w_rows[:, :, n_thinking:]  # (n_heads, chunk_size, n_context)
                        w_train = w_train / (w_train.sum(axis=2, keepdims=True) + eps)
                        max_a = w_train.max(axis=2)                             # (n_heads, chunk_size)
                        neg_e = (w_train * np.log(w_train + eps)).sum(axis=2)   # (n_heads, chunk_size)
                    else:
                        # Col (feature) self-attention: batch dim = all rows in context
                        # (thinking + train + test).  Averaging the whole batch produces a
                        # constant value for every sample in a batch — the cyclic artefact.
                        # Instead, extract only the last chunk_size rows (test samples) and
                        # compute per-sample stats.
                        # Col attention operates on the feature dimension and is
                        # row-invariant — every row in the context gets the same
                        # weights.  Per-sample extraction is meaningless here;
                        # always use the batch average.
                        while wm.ndim > 3:
                            wm = wm.mean(0)
                        if wm.ndim == 2:
                            wm = wm[np.newaxis]
                        n_heads = wm.shape[0]
                        max_a   = wm.max(axis=2)
                        neg_e   = (wm * np.log(wm + eps)).sum(axis=2)
                        argm    = wm.argmax(axis=2).astype(np.int32)

                    records.append({
                        "type":        "row" if is_row else "col",
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
    """Return (fpr, tpr, auc) for a binary classifier. Returns (0.5 AUC) on NaN/degenerate input."""
    valid = ~np.isnan(scores.ravel())
    if valid.sum() < 2 or len(np.unique(labels.ravel()[valid])) < 2:
        return np.array([0., 1.]), np.array([0., 1.]), 0.5
    fpr, tpr, _ = roc_curve(labels.ravel()[valid], scores.ravel()[valid])
    return fpr, tpr, float(sk_auc(fpr, tpr))



def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size between two independent groups a and b.

    d = (mean_a - mean_b) / pooled_std

    Pooled std uses the unbiased (n-1) denominator so it is well-defined for
    small groups.  Returns 0.0 when either group has fewer than 2 samples or
    zero variance.

    Interpretation: |d| < 0.2 negligible · 0.2 small · 0.5 medium · 0.8 large
    Sign: positive means group a has higher mean.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
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
                              max_row_calls: int = None,
                              max_col_calls: int = None,
                              n_thinking: int | None = None):
    """
    Run inference over all pool samples and collect per-call, per-head signals.

    Returns
    -------
    row_max_all : (n_pool, n_row_calls, n_heads) float32
    row_ent_all : (n_pool, n_row_calls, n_heads) float32
    row_arg_all : (n_pool, n_row_calls, n_heads) int32 — argmax key position
    col_max     : (n_pool,) float32
    col_ent     : (n_pool,) float32
    All return None on failure.
    """
    n_pool = len(X_pool)

    # Prevent the degenerate case where chunk_size == n_context + thinking_rows:
    # cross-attention (test→context) and context self-attention produce the same
    # (q_len, k_len) shape, making them indistinguishable by shape alone.
    # Cap batch_size to expected_k - 1 so no batch ever lands on that size.
    if n_thinking is None:
        n_thinking = infer_tabpfn_thinking_rows(model, default=None)
    expected_k = n_context + int(n_thinking or 0)
    if expected_k > 1 and batch_size >= expected_k:
        batch_size = expected_k - 1
        logger.warning(
            "batch_size capped to %d to avoid degenerate chunk_size == "
            "n_context(%d) + thinking_rows(%d)",
            batch_size, n_context, int(n_thinking or 0),
        )
    logger.info(
        "Effective AMIA attention extraction batch_size=%d  thinking_rows=%d  max_row_calls=%s  max_col_calls=%s",
        batch_size,
        int(n_thinking or 0),
        "all" if max_row_calls is None else str(max_row_calls),
        "all" if max_col_calls is None else str(max_col_calls),
    )

    # Guard against torch.compile replacing F.scaled_dot_product_attention
    # with a fused C++ kernel in the compiled graph, bypassing our hook.
    # When model is a defense wrapper (KAnonTabFMWrapper etc.), disabling dynamo
    # only on the wrapper's predict_proba is insufficient — the wrapper calls
    # self.model.predict_proba internally, which may still be compiled.
    # Walk through wrapper layers and disable dynamo on each level.
    try:
        import torch._dynamo
        _predict_proba = torch._dynamo.disable(model.predict_proba)
        _predict       = torch._dynamo.disable(model.predict)
        _inner = model
        while hasattr(_inner, "model"):
            _inner = _inner.model
            if hasattr(_inner, "predict_proba"):
                _inner.predict_proba = torch._dynamo.disable(_inner.predict_proba)
            if hasattr(_inner, "predict"):
                _inner.predict = torch._dynamo.disable(_inner.predict)
    except Exception:
        _predict_proba = model.predict_proba
        _predict       = model.predict

    row_max_batches, row_ent_batches, row_arg_batches = [], [], []
    col_max_list,    col_ent_list    = [], []

    n_row_calls_ref = None   # set on first batch; validated on subsequent ones
    n_heads_ref     = None

    for batch_start in range(0, n_pool, batch_size):
        batch_end = min(batch_start + batch_size, n_pool)
        X_batch   = X_pool[batch_start:batch_end]
        chunk     = batch_end - batch_start

        ctx = SDPACapture(
            chunk_size=chunk,
            n_context=n_context,
            n_thinking=n_thinking,
            max_row_calls=max_row_calls,
            max_col_calls=max_col_calls,
        )
        with ctx:
            try:
                _predict_proba(X_batch)
            except Exception as exc:
                logger.warning("predict_proba raised: %s – trying predict()", exc)
                try:
                    _predict(X_batch)
                except Exception as exc2:
                    logger.error("predict() failed: %s", exc2)
                    return None, None, None, None, None

        row_calls = [r for r in ctx.records if r["type"] == "row"]
        col_calls = [r for r in ctx.records if r["type"] == "col"]

        # Some wrappers can run extra internal forward passes.  Cap both row and
        # col calls to the first N from the real input pass when requested.
        if max_row_calls is not None:
            row_calls = row_calls[:max_row_calls]
        if max_col_calls is not None:
            col_calls = col_calls[:max_col_calls]

        if batch_start == 0:
            n_row_calls_ref = len(row_calls)
            n_heads_ref     = row_calls[0]["n_heads"] if row_calls else 1
            filtered  = ctx.filtered_row_calls
            degenerate = ctx.degenerate_skipped
            logger.info(
                "SDPA calls: %d row-attention (n_heads=%d), %d col-attention, "
                "%d ignored  [%d total, %d k_len-filtered, %d degenerate-skipped]",
                n_row_calls_ref, n_heads_ref, len(col_calls),
                ctx.total_calls - len(row_calls) - len(col_calls) - filtered - degenerate,
                ctx.total_calls, filtered, degenerate,
            )
            if degenerate:
                logger.error(
                    "Unsupported degenerate context: n_context(%d) + thinking_rows(%d) == chunk_size(%d). "
                    "Cross-attention is ambiguous under shape-only filtering."
                    "Reduce batch-size or use a context setting where chunk_size != n_context + thinking_rows.",
                    n_context, int(n_thinking or 0), chunk,
                )
                return None, None, None, None, None
            if n_row_calls_ref == 0:
                logger.error(
                    "No row-attention calls captured in first batch; refusing to fabricate zero signals."
                )
                return None, None, None, None, None
        elif len(row_calls) != n_row_calls_ref:
            new_count = len(row_calls)
            if new_count < n_row_calls_ref:
                if new_count == 0:
                    logger.error(
                        "Row-attention calls dropped to zero at batch %d-%d; aborting extraction.",
                        batch_start, batch_end,
                    )
                    return None, None, None, None, None
                logger.warning(
                    "Row-call count dropped from %d to %d — trimming all %d collected batches",
                    n_row_calls_ref, new_count, len(row_max_batches),
                )
                row_max_batches = [b[:, :new_count, :] for b in row_max_batches]
                row_ent_batches = [b[:, :new_count, :] for b in row_ent_batches]
                row_arg_batches = [b[:, :new_count, :] for b in row_arg_batches]
                n_row_calls_ref = new_count
            else:
                logger.warning(
                    "Row-call count changed: expected %d, got %d — truncating",
                    n_row_calls_ref, len(row_calls),
                )
                row_calls = row_calls[:n_row_calls_ref]

        if not row_calls:
            logger.error(
                "No row-attention calls captured at batch %d-%d; aborting extraction.",
                batch_start, batch_end,
            )
            return None, None, None, None, None

        # ── row signals → (chunk, n_row_calls, n_heads) ──
        rm = np.stack([r["max_attn"][:, :chunk] for r in row_calls], axis=0).transpose(2, 0, 1)
        re = np.stack([r["neg_entropy"][:, :chunk] for r in row_calls], axis=0).transpose(2, 0, 1)
        ra = np.stack([r["argmax"][:, :chunk]    for r in row_calls], axis=0).transpose(2, 0, 1)

        row_max_batches.append(rm)
        row_ent_batches.append(re)
        row_arg_batches.append(ra)

        # ── col signals → batch-level scalar broadcast to all samples ──
        if col_calls:
            cm_s = float(np.mean([r["max_attn"].mean()    for r in col_calls]))
            ce_s = float(np.mean([r["neg_entropy"].mean() for r in col_calls]))
        else:
            cm_s, ce_s = np.nan, np.nan
        cm = np.full(chunk, cm_s, dtype=np.float32)
        ce = np.full(chunk, ce_s, dtype=np.float32)
        col_max_list.append(cm.astype(np.float32))
        col_ent_list.append(ce.astype(np.float32))

        del ctx
        gc.collect()

        if batch_end % (batch_size * 4) == 0 or batch_end == n_pool:
            logger.info("  processed %d / %d samples", batch_end, n_pool)

    return (
        np.concatenate(row_max_batches, axis=0),   # (n_pool, n_row_calls, n_heads)
        np.concatenate(row_ent_batches, axis=0),
        np.concatenate(row_arg_batches, axis=0),   # (n_pool, n_row_calls, n_heads) int32
        np.concatenate(col_max_list),              # (n_pool,)
        np.concatenate(col_ent_list),
    )


# ─── plots ────────────────────────────────────────────────────────────────────

MEM_COLOR    = "#086375"
NONMEM_COLOR = "#ee6c4d"


def plot_distributions(signals_dict: dict, mem_mask: np.ndarray,
                       rep_dir: str, dataset_name: str, model_name: str):
    """
    Plot 01: KDE distributions of row-attention head-averaged signals for
    members vs non-members.  AUC annotated per panel to indicate standalone
    discriminative power.
    """
    labels = [
        "Row attention\n(max_attn)",  "Row attention\n(neg_entropy)",
    ]
    keys = ["row_max", "row_ent"]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.5))
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
    """
    Plot 02: Scatter of attention concentration (x) vs RMIA score (y),
    coloured by membership.  Pearson r annotated per panel.
    Members clustering top-right means the two signals share the same
    memorisation source.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        f"Attention concentration vs RMIA score — {dataset_name} / {model_name}\n"
        "Members should cluster top-right if attention drives RMIA effectiveness",
        fontsize=10, fontweight="bold",
    )
    for ax, key, label in [
        (axes[0], "row_max", "Row attention (max_attn)"),
        (axes[1], "row_ent", "Row attention (neg_entropy)"),
    ]:
        sig = signals_dict[key]
        for mask, name, color in [
            (~mem_mask, "Non-member", NONMEM_COLOR),
            (mem_mask,  "Member",     MEM_COLOR),
        ]:
            ax.scatter(sig[mask], rmia_scores[mask], c=color, alpha=0.3,
                       s=8, label=name, rasterized=True)
        try:
            r, p = pearsonr(sig, rmia_scores)
            corr_txt = f"Pearson r = {r:.3f}  (p = {p:.2e})"
        except Exception:
            corr_txt = ""
        ax.set_xlabel(label, fontsize=9);  ax.set_ylabel("RMIA score", fontsize=9)
        ax.set_title(corr_txt, fontsize=9)
        ax.legend(fontsize=8, markerscale=3);  ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(rep_dir, "02_attention_vs_rmia.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_comparison(signals_dict: dict, rmia_scores: np.ndarray,
                        mem_mask: np.ndarray, rep_dir: str,
                        dataset_name: str, model_name: str):
    """
    Plot 03: ROC curves — AMIA row-attention signal vs RMIA (comparison).
    Two panels: full ROC and low-FPR zoom (FPR <= 0.10).
    RMIA is shown as a reference baseline, not the focus.
    """
    labels_int = mem_mask.astype(int)
    curves = [
        ("Row attention (AMIA)", signals_dict["row_max"], "#086375", "-"),
        ("RMIA (comparison)",    rmia_scores,             "#ee6c4d", "--"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, title, xlim in zip(axes,
                                ["Full ROC", "Low-FPR region (FPR ≤ 0.10)"],
                                [(0, 1),     (0, 0.10)]):
        for name, scores, color, ls in curves:
            fpr, tpr, a = compute_roc(scores, labels_int)
            ax.plot(fpr, tpr, label=f"{name} (AUC={a:.3f})",
                    color=color, linestyle=ls, linewidth=1.8)
        ax.plot([0, 1], [0, 1], ":", color="gray", linewidth=1.2,
                label="Random guess (AUC=0.500)")
        ax.set_xlim(xlim);  ax.set_ylim(0, 1)
        ax.set_xlabel("FPR", fontsize=9);  ax.set_ylabel("TPR", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7.5);  ax.grid(True, alpha=0.25)
    fig.suptitle(
        f"ROC comparison — {dataset_name} / {model_name}\n"
        "AMIA (attention concentration) vs RMIA (output confidence) — random guess shown for reference",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(rep_dir, "03_roc_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# TabPFN v2.6: nlayers=24, nhead=3 (cross/row attention), features_per_group=3
# 1536 raw cross-SDPA calls = 24 layers × 8 feature-blocks × 8 calls/block
_TABPFN_N_LAYERS      = 24


def _group_by_layer(arr: np.ndarray, n_layers: int = _TABPFN_N_LAYERS) -> np.ndarray:
    """
    Collapse raw SDPA calls into encoder layers by averaging within each group.

    TabPFN v2.6 makes 64 cross-attention SDPA calls per encoder layer
    (8 feature-blocks × 8 calls/block).  Groups consecutive calls into
    n_layers buckets and averages → (n_pool, n_layers, n_heads).

    Parameters
    ----------
    arr      : (n_pool, n_raw_calls, n_heads)
    n_layers : encoder layers to group into (default: _TABPFN_N_LAYERS=24)

    Returns NaN array if there are fewer raw calls than layers (e.g. when
    no SDPA calls were captured for a given context-size run).
    """
    n_pool, n_raw, n_heads = arr.shape
    calls_per_layer = n_raw // n_layers
    if calls_per_layer == 0:
        return np.full((n_pool, n_layers, n_heads), np.nan, dtype=np.float32)
    if n_raw % n_layers != 0:
        warnings.warn(
            f"_group_by_layer: n_raw={n_raw} is not divisible by n_layers={n_layers}; "
            f"trimming {n_raw % n_layers} tail call(s). Layer-to-encoder alignment may be off.",
            stacklevel=2,
        )
    trimmed = arr[:, :calls_per_layer * n_layers, :]
    return trimmed.reshape(n_pool, n_layers, calls_per_layer, n_heads).mean(axis=2)


def plot_layer_auc(row_max_all: np.ndarray, row_ent_all: np.ndarray,
                   mem_mask: np.ndarray, rep_dir: str,
                   dataset_name: str, model_name: str):
    """
    Plot 04: Per-layer AUC and Cohen's d effect size.

    Two y-axes share the same x-axis (encoder layer):
      Left  — AUC of row_max and row_ent (rank-order separability).
      Right — Cohen's d for row_max (mean separation in pooled-std units).

    AUC and d together answer two different questions:
      AUC: can you rank members above non-members?
      d:   how far apart are the distributions in absolute terms?
    A layer with high AUC but small d has good rank ordering but overlapping
    distributions.  A large d confirms a strong, consistent effect.
    """
    n_layers = _TABPFN_N_LAYERS
    rm = _group_by_layer(row_max_all, n_layers)   # (n_pool, n_layers, n_heads)
    re = _group_by_layer(row_ent_all, n_layers)
    calls_per_layer = row_max_all.shape[1] // n_layers

    x          = np.arange(n_layers)
    labels_int = mem_mask.astype(int)

    auc_max, auc_ent, d_max, d_ent = [], [], [], []
    for c in range(n_layers):
        sig_max = rm[:, c, :].mean(axis=1)   # head-averaged
        sig_ent = re[:, c, :].mean(axis=1)
        _, _, a_max = compute_roc(sig_max, labels_int)
        _, _, a_ent = compute_roc(sig_ent, labels_int)
        auc_max.append(a_max)
        auc_ent.append(a_ent)
        # Cohen's d: members should have higher max_attn and higher (less-negative) entropy
        d_max.append(cohens_d(sig_max[mem_mask], sig_max[~mem_mask]))
        d_ent.append(cohens_d(sig_ent[mem_mask], sig_ent[~mem_mask]))

    auc_max = np.array(auc_max)
    auc_ent = np.array(auc_ent)
    d_max   = np.array(d_max)
    d_ent   = np.array(d_ent)

    fig, ax_auc = plt.subplots(figsize=(13, 4.5))
    ax_d = ax_auc.twinx()

    # AUC lines (left axis)
    l1, = ax_auc.plot(x, auc_max, "o-",  color=MEM_COLOR,    linewidth=1.8, label="row_max AUC")
    l2, = ax_auc.plot(x, auc_ent, "s--", color=NONMEM_COLOR, linewidth=1.8, label="row_ent AUC")
    ax_auc.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax_auc.set_ylabel("AUC", fontsize=9, color="black")
    ax_auc.set_ylim(0.45, 1.0)

    # Cohen's d bars (right axis) — semi-transparent so AUC lines stay readable
    ax_d.bar(x - 0.2, d_max, width=0.35, alpha=0.25, color=MEM_COLOR,    label="d  row_max")
    ax_d.bar(x + 0.2, d_ent, width=0.35, alpha=0.25, color=NONMEM_COLOR, label="d  row_ent")
    # Reference lines for effect-size thresholds
    for thresh, label in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        ax_d.axhline(thresh, color="silver", linestyle=":", linewidth=0.8)
        ax_d.text(n_layers - 0.3, thresh + 0.02, label, fontsize=7, color="gray", va="bottom")
    ax_d.set_ylabel("Cohen's d  (member − non-member)", fontsize=9, color="gray")
    ax_d.set_ylim(bottom=0)

    ax_auc.set_xlim(-0.5, n_layers - 0.5)
    ax_auc.set_xticks(x)
    ax_auc.set_xticklabels([f"L{i}" for i in x])
    ax_auc.set_xlabel("Encoder layer", fontsize=9)

    lines_auc  = [l1, l2]
    labels_auc = [l.get_label() for l in lines_auc]
    bar_patches = [
        plt.Rectangle((0, 0), 1, 1, fc=MEM_COLOR,    alpha=0.4),
        plt.Rectangle((0, 0), 1, 1, fc=NONMEM_COLOR, alpha=0.4),
    ]
    ax_auc.legend(lines_auc + bar_patches,
                  labels_auc + ["d  row_max", "d  row_ent"],
                  fontsize=8, loc="upper left")
    ax_auc.grid(True, alpha=0.25)
    ax_auc.set_title(
        f"Per-layer AUC and Cohen's d — {dataset_name} / {model_name}\n"
        f"{n_layers} layers × {calls_per_layer} SDPA calls averaged  ·  "
        "d thresholds: 0.2 small · 0.5 medium · 0.8 large",
        fontsize=10, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(os.path.join(rep_dir, "04_layer_auc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_divergence(row_ent_all: np.ndarray, mem_mask: np.ndarray,
                            rep_dir: str, dataset_name: str, model_name: str):
    """
    Plot 05: Per-layer entropy divergence + head/tail KDE comparison.

    Top panel — Mean neg-entropy per call for members vs non-members (±1 std
    shaded).  Head and tail call regions are shaded.  The gap between the two
    lines is annotated at each region: a growing gap from head to tail shows
    that entropy separation (i.e. members becoming more peaked) builds up with
    depth.

    Bottom panels — KDE of entropy at head calls (left) vs tail calls (right),
    split by membership.  Visually confirms whether the distribution shift is
    present already in early layers or only emerges at the end.
    """
    n_layers = _TABPFN_N_LAYERS
    re = _group_by_layer(row_ent_all, n_layers)   # (n_pool, n_layers, n_heads)
    calls_per_layer = row_ent_all.shape[1] // n_layers

    x       = np.arange(n_layers)

    mean_mem, mean_nonmem = [], []
    std_mem,  std_nonmem  = [], []
    for c in range(n_layers):
        sig = re[:, c, :].mean(axis=1)
        mean_mem.append(sig[mem_mask].mean());    std_mem.append(sig[mem_mask].std())
        mean_nonmem.append(sig[~mem_mask].mean()); std_nonmem.append(sig[~mem_mask].std())

    mean_mem    = np.array(mean_mem);    std_mem    = np.array(std_mem)
    mean_nonmem = np.array(mean_nonmem); std_nonmem = np.array(std_nonmem)

    fig, ax_line = plt.subplots(figsize=(12, 4.5))

    ax_line.plot(x, mean_mem,    "o-",  color=MEM_COLOR,    linewidth=1.8, label="Member")
    ax_line.plot(x, mean_nonmem, "s--", color=NONMEM_COLOR, linewidth=1.8, label="Non-member")
    ax_line.fill_between(x, mean_mem - std_mem,       mean_mem + std_mem,       alpha=0.12, color=MEM_COLOR)
    ax_line.fill_between(x, mean_nonmem - std_nonmem, mean_nonmem + std_nonmem, alpha=0.12, color=NONMEM_COLOR)

    ax_line.set_xlim(-0.5, n_layers - 0.5)
    ax_line.set_xticks(x);  ax_line.set_xticklabels([f"L{i}" for i in x])
    ax_line.set_ylabel("Mean neg-entropy  (less negative = more peaked)", fontsize=9)
    ax_line.set_title(
        f"Attention entropy divergence per encoder layer — {dataset_name} / {model_name}\n"
        f"{n_layers} layers × {calls_per_layer} feature-block calls averaged",
        fontsize=10, fontweight="bold",
    )
    ax_line.legend(fontsize=9);  ax_line.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(rep_dir, "05_entropy_divergence.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _amia_scalars_from_summary(summary_csv: str) -> dict:
    """Compute AMIA scalar metrics from a per-seed attention_summary.csv."""
    df = pd.read_csv(summary_csv)
    if "member" not in df.columns:
        raise ValueError(f"AMIA summary is missing member column: {summary_csv}")

    mem = df["member"].to_numpy(dtype=bool)
    scalars = {}
    for key in ["row_max", "row_ent", "col_max", "col_ent", "rmia_score"]:
        if key not in df.columns:
            continue
        vals = df[key].to_numpy(dtype=float)
        _, _, a = compute_roc(vals, mem.astype(int))
        scalars[f"{key}_auc"] = a
        scalars[f"{key}_d"] = cohens_d(vals[mem], vals[~mem])
    return scalars


def summarize_amia_seed_results(seeds: list[int], summary_dir: str) -> None:
    """Append/update AMIA seed rows in attack_result_seed_runs/summary CSVs."""
    metric_keys = [
        "row_max_auc",
        "row_ent_auc",
        "col_max_auc",
        "col_ent_auc",
        "rmia_score_auc",
        "row_max_d",
        "row_ent_d",
        "col_max_d",
        "col_ent_d",
        "rmia_score_d",
    ]
    rows = []
    for seed in seeds:
        report_dir = os.path.join(summary_dir, f"seed{seed}", "amia", "report")
        summary_csv = os.path.join(report_dir, "exp", "attention_summary.csv")
        if not os.path.exists(summary_csv):
            raise FileNotFoundError(
                f"Missing AMIA summary for seed {seed}: {summary_csv}"
            )
        scalars = _amia_scalars_from_summary(summary_csv)
        row = {"attack": "amia", "seed": seed, "report_dir": report_dir}
        for key in metric_keys:
            if key in scalars:
                row[key] = float(scalars[key])
        rows.append(row)

    Path(summary_dir).mkdir(parents=True, exist_ok=True)
    runs_path = os.path.join(summary_dir, "attack_result_seed_runs.csv")
    new_runs = pd.DataFrame(rows)
    if os.path.exists(runs_path):
        runs_df = pd.read_csv(runs_path)
        if "attack" not in runs_df.columns:
            runs_df.insert(0, "attack", "rmia")
        runs_df = runs_df[
            ~((runs_df["attack"] == "amia") & (runs_df["seed"].isin(seeds)))
        ]
        runs_df = pd.concat([runs_df, new_runs], ignore_index=True, sort=False)
    else:
        runs_df = new_runs
    runs_df.to_csv(runs_path, index=False)

    summary_rows = []
    amia_runs = runs_df[runs_df["attack"] == "amia"]
    for key in metric_keys:
        if key not in amia_runs.columns:
            continue
        values = amia_runs[key].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        summary_rows.append({
            "attack": "amia",
            "metric": key,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "num_seeds": len(values),
        })

    summary_path = os.path.join(summary_dir, "attack_result_seed_summary.csv")
    new_summary = pd.DataFrame(summary_rows)
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        if "attack" not in summary_df.columns:
            summary_df.insert(0, "attack", "rmia")
        summary_df = summary_df[summary_df["attack"] != "amia"]
        summary_df = pd.concat([summary_df, new_summary], ignore_index=True, sort=False)
    else:
        summary_df = new_summary
    summary_df.to_csv(summary_path, index=False)


def plot_layer_head_heatmap(row_max_all: np.ndarray, mem_mask: np.ndarray,
                            rep_dir: str, dataset_name: str, model_name: str):
    """
    Plot 06: Side-by-side heatmaps of AUC and Cohen's d over (layer × head).

    Left  — AUC: rank-order separability of max_attn between members and non-members.
    Right — Cohen's d: mean separation in pooled-std units.

    Together they reveal whether a head has good ranking (high AUC) but a small
    absolute gap (low d), or a large gap that is also well-ordered.
    The best heads for membership detection maximise both.
    """
    n_layers = _TABPFN_N_LAYERS
    import seaborn as sns
    rm         = _group_by_layer(row_max_all, n_layers)   # (n_pool, n_layers, n_heads)
    n_heads    = rm.shape[2]
    labels_int = mem_mask.astype(int)

    auc_mat = np.zeros((n_layers, n_heads))
    d_mat   = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        for h in range(n_heads):
            sig = rm[:, l, h]
            _, _, a = compute_roc(sig, labels_int)
            auc_mat[l, h] = a
            d_mat[l, h]   = cohens_d(sig[mem_mask], sig[~mem_mask])

    cell_w = max(5, n_heads * 1.2 + 2)
    cell_h = max(5, n_layers * 0.55 + 2)
    fig, axes = plt.subplots(1, 2, figsize=(cell_w * 2 + 1, cell_h))

    sns.heatmap(
        auc_mat, ax=axes[0], cmap="Blues", vmin=0.5, vmax=1.0,
        annot=True, fmt=".3f", annot_kws={"size": 7},
        xticklabels=[f"H{h}" for h in range(n_heads)],
        yticklabels=[f"L{l}" for l in range(n_layers)],
        linewidths=0.4, cbar_kws={"label": "AUC"},
    )
    axes[0].set_xlabel("Attention head", fontsize=9)
    axes[0].set_ylabel("Encoder layer", fontsize=9)
    axes[0].set_title("AUC  (rank separability)", fontsize=10, fontweight="bold")

    d_vmax = max(0.8, np.nanpercentile(d_mat, 95))
    sns.heatmap(
        d_mat, ax=axes[1], cmap="Oranges", vmin=0.0, vmax=d_vmax,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        xticklabels=[f"H{h}" for h in range(n_heads)],
        yticklabels=[f"L{l}" for l in range(n_layers)],
        linewidths=0.4, cbar_kws={"label": "Cohen's d"},
    )
    axes[1].set_xlabel("Attention head", fontsize=9)
    axes[1].set_ylabel("Encoder layer", fontsize=9)
    axes[1].set_title("Cohen's d  (effect size, 0.2/0.5/0.8 = small/medium/large)",
                      fontsize=10, fontweight="bold")

    fig.suptitle(
        f"Membership discriminability per (layer × head) — {dataset_name} / {model_name}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(rep_dir, "06_layer_head_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_argmax_analysis(row_arg_all: np.ndarray, mem_mask: np.ndarray,
                         rep_dir: str, dataset_name: str, model_name: str,
                         n_thinking: int = 0):
    """
    Plot 07: Argmax consistency — does the model keep attending to the same
    context position across layers and heads?

    For each sample we compute the 'argmax consistency': the fraction of
    (layer × head) pairs that share the same most-attended context position
    (the mode of all argmax values for that sample).

    Members may repeatedly lock onto a small set of training-context positions
    that support memorised predictions, producing high consistency.  Non-members
    are expected to have more diffuse nearest-context evidence, so the argmax
    scatters more.

    Two panels:
      Left  — KDE of global consistency (all layers) for members vs non-members.
      Right — mean consistency per layer for members vs non-members, showing
              at which depth the model 'locks on'.
    """
    n_layers = _TABPFN_N_LAYERS
    # Group argmax into layers: take the mode across feature blocks within each layer
    n_pool, n_raw, n_heads = row_arg_all.shape
    calls_per_layer = n_raw // n_layers
    trimmed = row_arg_all[:, :calls_per_layer * n_layers, :]
    # (n_pool, n_layers, calls_per_layer, n_heads)
    grouped = trimmed.reshape(n_pool, n_layers, calls_per_layer, n_heads)

    # Per-layer mode across (calls_per_layer × n_heads) positions → (n_pool, n_layers)
    # Flatten last two dims, then compute mode
    flat = grouped.reshape(n_pool, n_layers, -1)   # (n_pool, n_layers, cpl*n_heads)

    # Key layout: positions 0..n_thinking-1 are thinking rows prepended
    # before the training samples.  Attending to a thinking row does not indicate
    # finding a training copy, so mask those positions out before computing
    # consistency.  Positions below the threshold are set to -1 (invalid sentinel)
    # so they do not contribute to the mode calculation.
    def _mask_thinking_rows(argmax_arr: np.ndarray) -> np.ndarray:
        masked = argmax_arr.copy().astype(np.int32)
        if n_thinking > 0:
            masked[masked < n_thinking] = -1
        return masked

    def row_mode_fraction(arr2d):
        """For each row of arr2d, return fraction of valid (≥0) values == mode."""
        from scipy.stats import mode as scipy_mode
        result = np.zeros(arr2d.shape[0], dtype=np.float32)
        for i, row in enumerate(arr2d):
            valid = row[row >= 0]
            if len(valid) == 0:
                result[i] = np.nan
                continue
            m = scipy_mode(valid, keepdims=True).count[0]
            result[i] = m / len(valid)
        return result

    # Global consistency: mode fraction across ALL layers and heads (thinking rows masked)
    all_flat = _mask_thinking_rows(row_arg_all.reshape(n_pool, -1))
    global_consistency = row_mode_fraction(all_flat)

    # Per-layer consistency: mode fraction within each layer (thinking rows masked)
    flat_masked = _mask_thinking_rows(flat)
    layer_consistency = np.zeros((n_pool, n_layers), dtype=np.float32)
    for l in range(n_layers):
        layer_consistency[:, l] = row_mode_fraction(flat_masked[:, l, :])

    labels_int = mem_mask.astype(int)
    _, _, auc_global = compute_roc(global_consistency, labels_int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    thinking_note = (
        f"thinking-row positions 0–{n_thinking - 1} excluded"
        if n_thinking > 0 else
        "no thinking rows detected"
    )
    fig.suptitle(
        f"Argmax consistency — {dataset_name} / {model_name}\n"
        "Fraction of (layer × head) pairs attending to the same training-sample position\n"
        f"({thinking_note})  ·  "
        "Members should lock onto fewer training-context positions → higher consistency",
        fontsize=10, fontweight="bold",
    )

    # Left: global consistency KDE
    ax = axes[0]
    for mask, name, color in [
        (mem_mask,  "Member",     MEM_COLOR),
        (~mem_mask, "Non-member", NONMEM_COLOR),
    ]:
        vals = global_consistency[mask]
        vals = vals[~np.isnan(vals)]
        if len(vals) < 2:
            continue
        xv   = np.linspace(vals.min(), vals.max(), 200)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(vals)
            ax.plot(xv, kde(xv), label=name, color=color, linewidth=2)
            ax.fill_between(xv, kde(xv), alpha=0.15, color=color)
        except Exception:
            ax.hist(vals, bins=25, alpha=0.4, color=color, label=name, density=True)
    ax.set_title(f"Global consistency  (AUC={auc_global:.3f})", fontsize=9, fontweight="bold")
    ax.set_xlabel("Mode fraction (all layers × heads)", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.25)

    # Right: per-layer mean consistency
    ax = axes[1]
    x = np.arange(n_layers)
    mean_mem    = np.nanmean(layer_consistency[mem_mask], axis=0)
    mean_nonmem = np.nanmean(layer_consistency[~mem_mask], axis=0)
    std_mem     = np.nanstd(layer_consistency[mem_mask], axis=0)
    std_nonmem  = np.nanstd(layer_consistency[~mem_mask], axis=0)
    ax.plot(x, mean_mem,    "o-",  color=MEM_COLOR,    linewidth=1.8, label="Member")
    ax.plot(x, mean_nonmem, "s--", color=NONMEM_COLOR, linewidth=1.8, label="Non-member")
    ax.fill_between(x, mean_mem - std_mem,       mean_mem + std_mem,       alpha=0.12, color=MEM_COLOR)
    ax.fill_between(x, mean_nonmem - std_nonmem, mean_nonmem + std_nonmem, alpha=0.12, color=NONMEM_COLOR)
    ax.set_xlabel("Encoder layer", fontsize=8)
    ax.set_ylabel("Mean consistency (mode fraction)", fontsize=8)
    ax.set_title("Per-layer argmax consistency", fontsize=9, fontweight="bold")
    ax.set_xticks(x);  ax.set_xticklabels([f"L{i}" for i in x], fontsize=7)
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(rep_dir, "07_argmax_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_slope(row_ent_all: np.ndarray, mem_mask: np.ndarray,
                       rep_dir: str, dataset_name: str, model_name: str):
    """
    Plot 08: Layer-to-layer neg-entropy slope.

    row_ent is actually neg-entropy: sum(w * log w).  More peaked attention is
    less negative, so sharpening appears as a positive Δ neg-entropy as depth
    increases.  Members are expected to show a larger positive increase in later
    layers if attention concentration contributes to memorised predictions.

    Two panels:
      Left  — mean Δ neg-entropy per layer transition, members vs non-members.
      Right — KDE of per-sample cumulative slope (neg_entropy[Ln] − neg_entropy[L0]),
              showing total sharpening from first to last layer.
    """
    n_layers = _TABPFN_N_LAYERS
    re = _group_by_layer(row_ent_all, n_layers)   # (n_pool, n_layers, n_heads)
    # head-average → (n_pool, n_layers)
    re_mean = re.mean(axis=2)

    # Δ neg-entropy at each transition: L→L+1, shape (n_pool, n_layers-1)
    delta = np.diff(re_mean, axis=1)

    x = np.arange(n_layers - 1)
    mean_delta_mem    = delta[mem_mask].mean(axis=0)
    mean_delta_nonmem = delta[~mem_mask].mean(axis=0)
    std_delta_mem     = delta[mem_mask].std(axis=0)
    std_delta_nonmem  = delta[~mem_mask].std(axis=0)

    # Cumulative slope: last layer neg-entropy − first layer neg-entropy
    cum_slope = re_mean[:, -1] - re_mean[:, 0]   # (n_pool,)
    labels_int = mem_mask.astype(int)
    _, _, auc_slope = compute_roc(cum_slope, labels_int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        f"Entropy slope across layers — {dataset_name} / {model_name}\n"
        "Δ neg-entropy = neg_entropy[L+1] − neg_entropy[L]  ·  positive = attention sharpening",
        fontsize=10, fontweight="bold",
    )

    # Left: per-transition Δ entropy
    ax = axes[0]
    ax.plot(x, mean_delta_mem,    "o-",  color=MEM_COLOR,    linewidth=1.8, label="Member")
    ax.plot(x, mean_delta_nonmem, "s--", color=NONMEM_COLOR, linewidth=1.8, label="Non-member")
    ax.fill_between(x, mean_delta_mem - std_delta_mem,       mean_delta_mem + std_delta_mem,       alpha=0.12, color=MEM_COLOR)
    ax.fill_between(x, mean_delta_nonmem - std_delta_nonmem, mean_delta_nonmem + std_delta_nonmem, alpha=0.12, color=NONMEM_COLOR)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Layer transition  (L → L+1)", fontsize=8)
    ax.set_ylabel("Δ neg-entropy", fontsize=8)
    ax.set_title("Per-transition entropy change", fontsize=9, fontweight="bold")
    ax.set_xticks(x);  ax.set_xticklabels([f"L{i}→{i+1}" for i in x], fontsize=6, rotation=30)
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.25)

    # Right: KDE of cumulative slope
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
    ax.set_title(f"Cumulative slope  neg_entropy[L{n_layers-1}]−neg_entropy[L0]\n(AUC={auc_slope:.3f})",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel(f"Total neg-entropy change (L0 → L{n_layers-1})", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(rep_dir, "08_entropy_slope.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_block_variance(row_max_all: np.ndarray, mem_mask: np.ndarray,
                                rep_dir: str, dataset_name: str, model_name: str):
    """
    Plot 09: Variance of max_attn across feature blocks within each layer.

    Each encoder layer processes (calls_per_layer) feature blocks in parallel.
    The variance of max_attn across these blocks measures how unevenly the
    model's attention concentration is distributed across feature dimensions.

    Members: the model strongly attends on the feature blocks that distinguish
             their training copy → high variance (some blocks very concentrated,
             others diffuse).
    Non-members: attention is more uniform across feature blocks → lower variance.

    Two panels:
      Left  — mean intra-layer variance per encoder layer, members vs non-members.
      Right — KDE of total variance (averaged over all layers) per sample.
    """
    n_layers = _TABPFN_N_LAYERS
    n_pool, n_raw, n_heads = row_max_all.shape
    calls_per_layer = n_raw // n_layers
    # (n_pool, n_layers, calls_per_layer, n_heads) → variance across feature blocks
    grouped  = row_max_all[:, :calls_per_layer * n_layers, :].reshape(
        n_pool, n_layers, calls_per_layer, n_heads)
    # Head-average first, then variance across feature blocks
    # (n_pool, n_layers, calls_per_layer)
    grouped_hm = grouped.mean(axis=3)
    # Variance across feature blocks → (n_pool, n_layers)
    block_var  = grouped_hm.var(axis=2)

    labels_int = mem_mask.astype(int)
    x = np.arange(n_layers)
    mean_var_mem    = block_var[mem_mask].mean(axis=0)
    mean_var_nonmem = block_var[~mem_mask].mean(axis=0)
    std_var_mem     = block_var[mem_mask].std(axis=0)
    std_var_nonmem  = block_var[~mem_mask].std(axis=0)

    total_var = block_var.mean(axis=1)   # (n_pool,)
    _, _, auc_var = compute_roc(total_var, labels_int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        f"Feature-block attention variance per layer — {dataset_name} / {model_name}\n"
        "Variance of max_attn across the feature blocks within each encoder layer\n"
        "High variance = model focuses strongly on some feature dimensions, ignores others",
        fontsize=10, fontweight="bold",
    )

    # Left: per-layer variance
    ax = axes[0]
    ax.plot(x, mean_var_mem,    "o-",  color=MEM_COLOR,    linewidth=1.8, label="Member")
    ax.plot(x, mean_var_nonmem, "s--", color=NONMEM_COLOR, linewidth=1.8, label="Non-member")
    ax.fill_between(x, mean_var_mem - std_var_mem,       mean_var_mem + std_var_mem,       alpha=0.12, color=MEM_COLOR)
    ax.fill_between(x, mean_var_nonmem - std_var_nonmem, mean_var_nonmem + std_var_nonmem, alpha=0.12, color=NONMEM_COLOR)
    ax.set_xlabel("Encoder layer", fontsize=8)
    ax.set_ylabel("Mean intra-layer variance", fontsize=8)
    ax.set_title("Per-layer feature-block variance", fontsize=9, fontweight="bold")
    ax.set_xticks(x);  ax.set_xticklabels([f"L{i}" for i in x], fontsize=7)
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.25)

    # Right: KDE of total variance
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
    ax.set_title(f"Total feature-block variance  (AUC={auc_var:.3f})", fontsize=9, fontweight="bold")
    ax.set_xlabel("Mean variance across all layers", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(rep_dir, "09_feature_block_variance.png"), dpi=150, bbox_inches="tight")
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
    n_thinking: int | None = None,
) -> dict:
    """Extract attention signals (with caching) and generate all AMIA plots.

    Parameters
    ----------
    model       : wrapped or raw tabular model with predict_proba / predict.
    X_pool      : (n_pool, n_features) pool samples (members + non-members).
    mem         : (n_pool,) bool — True for training members.
    rmia_scores : (n_pool,) pre-computed RMIA scores for scatter / ROC plots.
    n_context   : number of training context items (= number of True in mem).
    batch_size  : pool samples per SDPA-capture forward pass.
    logger      : Python logger instance.
    sig_dir     : directory where the .npz signal cache is saved / loaded.
    exp_dir     : directory where plots and attention_summary.csv are saved.
    dataset_name, model_name : strings used in plot titles.
    model_idx   : shadow-model index (used only for cache file name).
    mode        : internal cache policy. Normal CLI runs use 'train' to
                  recompute signals; --plots-only uses 'load' to reuse them.
    max_row_calls, max_col_calls : optional cap for SDPA call counts when a
        wrapper runs extra internal forward passes.

    Returns
    -------
    dict with keys: row_max_auc, row_ent_auc, col_max_auc, col_ent_auc,
    row_max_d, row_ent_d (Cohen's d, member − non-member).
    """
    os.makedirs(sig_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    cache = os.path.join(sig_dir, f"attn_signals_{model_idx}.npz")

    if mode == "train" and os.path.exists(cache):
        os.remove(cache)
        logger.info("Normal AMIA run: deleted cached signals %s", cache)

    row_arg_all = None
    if os.path.exists(cache):
        try:
            npz         = np.load(cache)
            row_max_all = npz["row_max_all"]
            row_ent_all = npz["row_ent_all"]
            row_arg_all = npz["row_arg_all"] if "row_arg_all" in npz else None
            col_max     = npz["col_max"]
            col_ent     = npz["col_ent"]
            if n_thinking is None and "n_thinking" in npz:
                n_thinking = int(np.asarray(npz["n_thinking"]).reshape(-1)[0])
        except Exception as exc:
            if model is None:
                raise ValueError(f"Cached signals at {cache} are unreadable: {exc}") from exc
            logger.warning("Cached signals at %s are unreadable (%s); deleting and re-extracting.", cache, exc)
            os.remove(cache)
            row_arg_all = None

    if os.path.exists(cache):
        if row_max_all.shape[0] != len(mem) or len(col_max) != len(mem):
            msg = (
                f"Cached signals have {row_max_all.shape[0]} rows, but current labels "
                f"have {len(mem)} rows."
            )
            if mode == "load":
                raise ValueError(msg + " Re-run AMIA without --plots-only to refresh the cache.")
            logger.warning("%s Re-extracting.", msg)
            os.remove(cache)
            row_arg_all = None
        elif row_max_all.shape[1] < 2:
            logger.warning(
                "Cached signals look degenerate (n_raw=%d < 2) — re-extracting.",
                row_max_all.shape[1],
            )
            os.remove(cache)
            row_arg_all = None
        elif row_arg_all is None:
            logger.warning("Cache missing row_arg_all — re-extracting.")
        else:
            logger.info("Loaded cached signals from %s  shape=%s", cache, row_max_all.shape)

    if not os.path.exists(cache) or row_arg_all is None:
        if n_thinking is None and model is not None:
            n_thinking = infer_tabpfn_thinking_rows(model, default=None)
        logger.info("Extracting attention signals")
        row_max_all, row_ent_all, row_arg_all, col_max, col_ent = extract_attention_signals(
            model, X_pool, n_context, batch_size, logger,
            max_row_calls=max_row_calls,
            max_col_calls=max_col_calls,
            n_thinking=n_thinking,
        )
        if row_max_all is None:
            raise RuntimeError("Attention extraction failed — see log for details.")
        np.savez_compressed(cache,
                            row_max_all=row_max_all, row_ent_all=row_ent_all,
                            row_arg_all=row_arg_all, col_max=col_max, col_ent=col_ent,
                            n_thinking=np.asarray([int(n_thinking or 0)], dtype=np.int16))
        logger.info("Cached signals to %s  shape=%s", cache, row_max_all.shape)

    logger.info("Signal shapes: row_max_all=%s  col_max=%s", row_max_all.shape, col_max.shape)

    n_raw = row_max_all.shape[1]
    if n_raw < _TABPFN_N_LAYERS:
        logger.warning(
            "Only %d raw SDPA calls captured (need >= %d for layer-level plots). "
            "Layer plots (04–09) will be skipped.", n_raw, _TABPFN_N_LAYERS,
        )

    row_max = row_max_all.mean(axis=(1, 2))
    row_ent = row_ent_all.mean(axis=(1, 2))
    if len(rmia_scores) != len(mem):
        raise ValueError(
            f"RMIA score length ({len(rmia_scores)}) does not match membership labels ({len(mem)})."
        )
    signals_dict = {
        "row_max": row_max, "row_ent": row_ent,
        "col_max": col_max, "col_ent": col_ent,
    }

    df_out = pd.DataFrame({
        "member":     mem.astype(int),
        "rmia_score": rmia_scores,
        **signals_dict,
    })
    df_out.to_csv(os.path.join(exp_dir, "attention_summary.csv"), index=False)

    logger.info("Per-signal AUC and Cohen's d (head-averaged, all calls):")
    logger.info("  %-20s  %6s  %8s  %s", "Signal", "AUC", "Cohen's d", "Interpretation")
    logger.info("  " + "-" * 60)
    scalars: dict = {}
    for key, name in [("row_max", "row_max"), ("row_ent", "row_ent"),
                      ("col_max", "col_max"), ("col_ent", "col_ent"),
                      ("rmia_score", "RMIA")]:
        vals = df_out[key].values
        _, _, a = compute_roc(vals, mem.astype(int))
        d = cohens_d(vals[mem], vals[~mem])
        interp = ("negligible" if abs(d) < 0.2 else
                  "small"      if abs(d) < 0.5 else
                  "medium"     if abs(d) < 0.8 else
                  "large"      if abs(d) < 1.2 else "very large")
        logger.info("  %-20s  %.4f  %+.4f   %s", name, a, d, interp)
        scalars[key + "_auc"] = a
        scalars[key + "_d"]   = d

    plot_distributions(signals_dict, mem, exp_dir, dataset_name, model_name)
    plot_attention_vs_rmia(signals_dict, rmia_scores, mem, exp_dir, dataset_name, model_name)
    plot_roc_comparison(signals_dict, rmia_scores, mem, exp_dir, dataset_name, model_name)

    if n_raw >= _TABPFN_N_LAYERS:
        plot_layer_auc(row_max_all, row_ent_all, mem, exp_dir, dataset_name, model_name)
        plot_entropy_divergence(row_ent_all, mem, exp_dir, dataset_name, model_name)
        plot_layer_head_heatmap(row_max_all, mem, exp_dir, dataset_name, model_name)
        if row_arg_all is not None:
            plot_argmax_analysis(
                row_arg_all,
                mem,
                exp_dir,
                dataset_name,
                model_name,
                n_thinking=int(n_thinking or 0),
            )
        else:
            logger.warning("Skipping plot 07: row_arg_all not available in cache.")
        plot_entropy_slope(row_ent_all, mem, exp_dir, dataset_name, model_name)
        plot_feature_block_variance(row_max_all, mem, exp_dir, dataset_name, model_name)
    else:
        logger.warning("Skipping layer-level plots (04–09): only %d raw SDPA calls.", n_raw)

    return scalars


# ─── main ─────────────────────────────────────────────────────────────────────

def main(dataset_name: str, model_name: str, gpu, batch_size: int | None,
         model_idx: int, plots_only: bool = False,
         context_pct: float = 100.0, seed: int | None = None,
         skip_existing: bool = False,
         max_audit_samples: int | None = None,
         max_row_calls: int | None = None,
         max_col_calls: int | None = None,
         audit_nonmember_dataset: str | None = None,
         population_dataset: str | None = None,
         ood_data_dir: str = "data/ood_noise25"):
    """
    Parameters
    ----------
    plots_only : bool
        If True, skip model loading and signal extraction entirely.
        Requires a cached .npz file from a previous run.  Use this to
        quickly regenerate or update plots without re-running inference.

    Steps (normal mode)
    -------------------
    1. Load config, dataset; keep only the 75 % candidate pool.
    2. Load the single target model (model_idx) and free all others.
    3. Compute RMIA scores via run_rmia() from rmia/signals/rmia_signals{_pop}.npy.
    4. Extract (or load from cache) per-call, per-head attention signals.
    5. Derive head-averaged flat signals for overview plots.
    6. Write attention_summary.csv and log per-signal AUC.
    7. Produce all five output plots.
    """
    sys.path.append(str(Path(__file__).parent.parent.parent / "ml_privacy_meter"))
    from models.utils import load_models
    from util import setup_log
    from run_attacks.ood_eval import amia_ood_run_name, rmia_ood_run_name

    torch.backends.cudnn.benchmark = True

    base_log = os.path.join("ml_privacy_meter", "logs", dataset_name, model_name.lower())
    run_root = os.path.join(base_log, f"seed{seed}") if seed is not None else base_log
    ood_eval = audit_nonmember_dataset is not None or population_dataset is not None
    if ood_eval:
        rmia_log = os.path.join(run_root, rmia_ood_run_name(context_pct or 100.0, audit_nonmember_dataset, population_dataset))
        attn_log = os.path.join(run_root, amia_ood_run_name(context_pct or 100.0, audit_nonmember_dataset, population_dataset))
    elif context_pct is not None and context_pct < 100.0:
        rmia_log = os.path.join(run_root, f"rmia_ctx{int(context_pct)}")
        attn_log = os.path.join(run_root, f"amia_ctx{int(context_pct)}")
    else:
        rmia_log = os.path.join(run_root, "rmia")
        attn_log = os.path.join(run_root, "amia")
    sig_dir = os.path.join(attn_log, "signals")
    rep_dir = os.path.join(attn_log, "report")
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

    cfg_batch_size = int(configs.get("audit", {}).get("batch_size", 5000))
    if batch_size is None:
        batch_size = min(128, cfg_batch_size)
        logger.info(
            "AMIA batch size not provided; using %d (config audit.batch_size=%d is for generic signal inference).",
            batch_size, cfg_batch_size,
        )
    else:
        logger.info(
            "AMIA batch size override: %d (config audit.batch_size=%d).",
            batch_size, cfg_batch_size,
        )

    if plots_only:
        # ── plots-only: load mem + rmia_scores from CSV, re-generate all plots ─
        if not os.path.exists(cache):
            raise FileNotFoundError(
                f"No cached signals at {cache}\n"
                f"Run without --plots-only first to extract signals."
            )
        if not os.path.exists(summary_csv):
            raise FileNotFoundError(
                f"No summary CSV at {summary_csv} — run without --plots-only first."
            )
        logger.info("--plots-only: regenerating plots from cache (no model inference).")
        df_existing = pd.read_csv(summary_csv)
        mem         = df_existing["member"].values.astype(bool)
        rmia_scores = df_existing["rmia_score"].values
        logger.info("Loaded %d samples from %s", len(mem), summary_csv)

        run_amia_pipeline(
            model=None,
            X_pool=np.empty((0, 0)),   # not used — cache is loaded instead
            mem=mem,
            rmia_scores=rmia_scores,
            n_context=0,               # not used
            batch_size=batch_size,
            logger=logger,
            sig_dir=sig_dir,
            exp_dir=exp_dir,
            dataset_name=dataset_name,
            model_name=model_name,
            model_idx=model_idx,
            mode="load",               # never delete cache in plots-only mode
        )

    else:
        # ── full run ────────────────────────────────────────────────────────────
        torch.manual_seed(configs["run"].get("random_seed", 12345))

        if gpu is not None:
            configs.setdefault("train", {})["device"] = "cuda:0"
            configs.setdefault("audit", {})["device"] = "cuda:0"
        else:
            configs.setdefault("train", {})["device"] = "cpu"
            configs.setdefault("audit", {})["device"] = "cpu"

        # Data
        data_dir = configs["data"]["data_dir"]
        df_raw   = load_dataset(dataset_name, data_dir)
        X, y     = prepare_tabular_arrays(df_raw)
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
        
        # Model
        models_list, memberships = load_models(rmia_log, None, None, configs, logger)
        if models_list is None or model_idx >= len(models_list):
            n_found = len(models_list) if models_list is not None else 0
            raise RuntimeError(f"model_idx={model_idx} out of range (found {n_found} models)")

        raw_model    = models_list[model_idx]
        thinking_rows = infer_tabpfn_thinking_rows(raw_model, default=0)
        logger.info("Detected TabPFN thinking_rows=%d", thinking_rows)

        from audit import sample_auditing_dataset
        from dataset.tabular import TabularDataset
        from run_attacks.ood_eval import (
            amia_ood_run_name,
            load_ood_dataset,
            make_ood_auditing_dataset,
            rmia_ood_run_name,
        )

        pool_size = memberships.shape[1]
        dataset = TabularDataset(X[:pool_size], y[:pool_size])
        np.random.seed(configs["run"].get("random_seed", 12345))
        if audit_nonmember_dataset is not None:
            ood_dataset = load_ood_dataset(
                audit_nonmember_dataset, ood_data_dir, prepare_tabular_arrays, TabularDataset
            )
            if ood_dataset.data.shape[1] != dataset.data.shape[1]:
                raise ValueError(
                    f"OOD audit dataset {audit_nonmember_dataset} has {ood_dataset.data.shape[1]} features; "
                    f"expected {dataset.data.shape[1]}."
                )
            auditing_dataset, auditing_membership = make_ood_auditing_dataset(
                configs, dataset, ood_dataset, logger, memberships, TabularDataset
            )
        else:
            auditing_dataset, auditing_membership = sample_auditing_dataset(
                configs, dataset, logger, memberships
            )
        X_pool, _ = _dataset_arrays(auditing_dataset)
        n_pool = len(X_pool)
        mem = auditing_membership[model_idx].astype(bool)
        n_members    = int(mem.sum())
        context_size = n_members   # TabPFN: training set IS the context window
        logger.info("Target model %d: %d members, %d non-members",
                    model_idx, n_members, n_pool - n_members)

        model = raw_model
        # Force memory_saving_mode=True so TabPFN always uses save_peak_mem_factor=8,
        # splitting each attention call into per-feature-group chunks.  With "auto"
        # the decision is based on free GPU memory at runtime, which causes the number
        # of visible SDPA calls to flip between ~576 (no chunking, slow) and ~4032
        # (chunked, fast) across runs — making hook coverage and timing unpredictable.
        # Forcing True gives consistent chunked calls and is ~2x faster for large
        # contexts like ALOI (n_context ~18k) because smaller chunks fit GPU cache.
        _inner = model
        while _inner is not None:
            if hasattr(_inner, "memory_saving_mode"):
                _inner.memory_saving_mode = True
            _inner = getattr(_inner, "model", None)

        _max_row_calls = max_row_calls
        _max_col_calls = max_col_calls

        # Load RMIA signals from the existing rmia/ cache.
        rmia_cache_dir  = os.path.join(rmia_log, "signals")
        rmia_score_path = os.path.join(sig_dir, f"rmia_scores_{model_idx}.npy")

        logger.info("Loading RMIA signals from %s", rmia_cache_dir)
        rmia_signals     = np.load(os.path.join(rmia_cache_dir, "rmia_signals.npy"))
        rmia_signals_pop = np.load(os.path.join(rmia_cache_dir, "rmia_signals_pop.npy"))
        logger.info("RMIA signals loaded: pool=%s  pop=%s",
                    rmia_signals.shape, rmia_signals_pop.shape)
        if rmia_signals.shape[0] != n_pool or auditing_membership.shape[1] != n_pool:
            raise ValueError(
                "RMIA signal rows do not match reconstructed AMIA audit pool: "
                f"signals={rmia_signals.shape[0]}, pool={n_pool}, memberships={auditing_membership.shape}"
            )

        num_ref_models = configs["audit"]["num_ref_models"]
        from attacks import run_rmia, tune_offline_a
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
        np.save(rmia_score_path, rmia_scores)

        if max_audit_samples is not None:
            if max_audit_samples <= 0:
                raise ValueError("--max-audit-samples must be positive when provided.")
            if max_audit_samples < n_pool:
                rng = np.random.default_rng(configs["run"].get("random_seed", 12345))
                member_idx = np.flatnonzero(mem)
                nonmember_idx = np.flatnonzero(~mem)
                n_member = min(len(member_idx), max_audit_samples // 2)
                n_nonmember = min(len(nonmember_idx), max_audit_samples - n_member)
                remaining = max_audit_samples - n_member - n_nonmember
                if remaining > 0 and len(member_idx) > n_member:
                    n_member = min(len(member_idx), n_member + remaining)
                elif remaining > 0 and len(nonmember_idx) > n_nonmember:
                    n_nonmember = min(len(nonmember_idx), n_nonmember + remaining)
                chosen = np.concatenate([
                    rng.choice(member_idx, size=n_member, replace=False),
                    rng.choice(nonmember_idx, size=n_nonmember, replace=False),
                ])
                chosen.sort()
                X_pool = X_pool[chosen]
                mem = mem[chosen]
                rmia_scores = rmia_scores[chosen]
                n_pool = len(X_pool)
                logger.info(
                    "Subsampled AMIA audit pool to %d samples (%d members, %d non-members) via --max-audit-samples.",
                    n_pool, int(mem.sum()), int((~mem).sum()),
                )

        for i, m in enumerate(models_list):
            if i != model_idx and hasattr(m, "to"):
                try:
                    m.to("cpu")
                except Exception:
                    pass
        gc.collect()
        
        pipeline_mode = "load" if os.path.exists(cache) and not os.path.exists(summary_csv) else "train"
        if pipeline_mode == "load":
            logger.info("Recovering incomplete AMIA run from cached signals: %s", cache)
        run_amia_pipeline(
            model, X_pool, mem, rmia_scores, context_size, batch_size, logger,
            sig_dir, exp_dir, dataset_name, model_name, model_idx, mode=pipeline_mode,
            max_row_calls=_max_row_calls,
            max_col_calls=_max_col_calls,
            n_thinking=thinking_rows,
        )

    logger.info("Done in %.1f s", time.time() - t0)

    if seed is not None:
        from run_attacks.seed_summary import update_seed_row
        _attack_label = f"amia_ctx{int(context_pct)}" if context_pct < 100.0 else "amia"
        update_seed_row(_attack_label, int(seed), Path(rep_dir), Path(base_log))
    cleanup_runtime_cache(logger)


# ─── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Attention-based explanation of RMIA effectiveness on TabPFN / Real-TabPFN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog="""
Examples:
  uv run run_attacks/amia/amia_tabpfn.py --dataset locations
  uv run run_attacks/amia/amia_tabpfn.py --dataset locations --model real-tabpfn
  uv run run_attacks/amia/amia_real_tabpfn.py --dataset locations
  uv run run_attacks/amia/amia_tabpfn.py --dataset purchases10 --gpu 0 --batch-size 100
  uv run run_attacks/amia/amia_tabpfn.py --dataset locations --model-idx 2
        """,
    )
    parser.add_argument("--dataset",     type=str, default="locations",
                        help="Dataset name (must have a matching config YAML and RMIA log).")
    parser.add_argument("--model",       type=str, default="tabpfn",
                        choices=["tabpfn", "real-tabpfn"],
                        help="Target model name: tabpfn or real-tabpfn (default: tabpfn).")
    parser.add_argument("--model-idx",   type=int, default=0,
                        help="Index of the shadow model to use as target (default: 0).")
    parser.add_argument("--gpu",         type=str, default=None,
                        help="CUDA device index, e.g. '0'.  Omit for CPU.")
    parser.add_argument("--batch-size",  type=int, default=None,
                        help=(
                            "AMIA attention-capture batch size. Defaults to min(128, "
                            "config audit.batch_size); reduce if OOM."
                        ))
    parser.add_argument("--skip-config", action="store_true",
                        help="Skip config/dataset preparation if already exists.")
    parser.add_argument("--plots-only", action="store_true",
                        help="Regenerate plots from cached signals without re-running inference.")
    parser.add_argument("--context-pct", type=float, default=100.0,
                        help="Context-size percentage matching a prior rmia_ctx<pct>/ run (default: 100 = full context).")
    parser.add_argument("--seed", type=str, default="1",
                        help=(
                            "Seed number for artifacts under logs/<dataset>/<model>/seed<seed>/{rmia,amia}/. "
                            "Also accepts a comma list for convenience, e.g. --seed 1,2,3."
                        ))
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seed list, e.g. --seeds 1,2,3,4,5.")
    parser.add_argument("--audit-nonmember-dataset", type=str, default=None,
                        help="OOD dataset name to use as AMIA audit nonmembers. Training/context data remains the ID --dataset.")
    parser.add_argument("--population-dataset", type=str, default=None,
                        help="OOD population dataset name used to locate the matching RMIA OOD run.")
    parser.add_argument("--ood-data-dir", type=str, default="data/ood_noise25",
                        help="Directory containing generated OOD CSV files.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a seed if cached AMIA signals and attention_summary.csv already exist.")
    parser.add_argument("--max-audit-samples", type=int, default=None,
                        help=(
                            "Optional stratified cap on member/non-member audit samples before attention extraction. "
                            "Useful for large datasets such as ALOI."
                        ))
    parser.add_argument("--max-row-calls", type=int, default=None,
                        help="Optional cap on captured row-attention SDPA calls per batch for faster approximate AMIA.")
    parser.add_argument("--max-col-calls", type=int, default=None,
                        help="Optional cap on captured column-attention SDPA calls per batch; use 0 to skip column summaries.")
    args = parser.parse_args()

    seed_arg = args.seeds if args.seeds is not None else args.seed
    seeds = [int(s.strip()) for s in str(seed_arg).split(",") if s.strip()]
    if not seeds:
        raise ValueError("No seeds provided.")

    if len(seeds) > 1:
        base_argv = sys.argv[1:]
        cleaned = []
        skip_next = False
        for item in base_argv:
            if skip_next:
                skip_next = False
                continue
            if item in {"--seed", "--seeds"}:
                skip_next = True
                continue
            if item.startswith("--seed=") or item.startswith("--seeds="):
                continue
            cleaned.append(item)

        script = Path(__file__).resolve()
        for seed in seeds:
            if args.skip_existing:
                _base = os.path.join("ml_privacy_meter", "logs",
                                     args.dataset, args.model.lower(), f"seed{seed}")
                _cache = os.path.join(_base, "amia", "signals",
                                      f"attn_signals_{args.model_idx}.npz")
                _csv   = os.path.join(_base, "amia", "report", "exp",
                                      "attention_summary.csv")
                if os.path.exists(_cache) and os.path.exists(_csv):
                    print(f"[AMIA] seed={seed} already done, skipping.", flush=True)
                    continue
            cmd = [sys.executable, str(script), *cleaned, "--seed", str(seed)]
            print(f"[AMIA] Running seed {seed}: {' '.join(cmd)}", flush=True)
            completed = subprocess.run(cmd)
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)
        summary_dir = os.path.join(
            "ml_privacy_meter",
            "logs",
            args.dataset,
            args.model.lower(),
        )
        summarize_amia_seed_results(seeds, summary_dir)
        print(f"[AMIA] Appended AMIA seed summaries to {summary_dir}", flush=True)
        raise SystemExit(0)

    seed = seeds[0]

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from configs import ensure_dataset_ready
    ensure_dataset_ready(
        dataset_name=args.dataset,
        model_name=args.model,
        algorithm="RMIA",
        skip_if_exists=args.skip_config,
    )

    try:
        main(
            dataset_name=args.dataset,
            model_name=args.model,
            gpu=args.gpu,
            batch_size=args.batch_size,
            model_idx=args.model_idx,
            plots_only=args.plots_only,
            context_pct=args.context_pct,
            seed=seed,
            skip_existing=args.skip_existing,
            audit_nonmember_dataset=args.audit_nonmember_dataset,
            population_dataset=args.population_dataset,
            ood_data_dir=args.ood_data_dir,
            max_audit_samples=args.max_audit_samples,
            max_row_calls=args.max_row_calls,
            max_col_calls=args.max_col_calls,
        )
    except Exception as e:
        import traceback
        _out = Path("results_visualizations")
        _out.mkdir(parents=True, exist_ok=True)
        with (_out / "amia_failed_runs.csv").open("a") as fh:
            fh.write(f"{args.dataset},{args.model},{str(e).replace(',', ';')}\n")
        print(f"[FAILED] {args.dataset} + {args.model}: {e}")
        traceback.print_exc()
        raise SystemExit(1)
