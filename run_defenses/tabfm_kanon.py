"""
K-anonymity defense for tabular foundation models (TabPFN, TabICL, TabDPT).

Groups the cross-attention key representations into groups of k by proximity
(sorted projection onto the first principal direction), replacing every key with
its group centroid.  No individual training sample can produce a distinguishable
self-attention spike — k-1 other samples always produce the exact same key.

Operates as an SDPA hook: no model modification or re-fitting required.
"""

import numpy as np
import torch
import torch.nn.functional as F
import threading
import contextlib


_PATCH_LOCK = threading.RLock()
_PATCH_DEPTH = 0
_PATCH_ORIG = None


def _model_classes(model) -> np.ndarray:
    """Return sklearn-style class labels for TabFM wrappers."""
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    if hasattr(model, "classes"):
        return np.asarray(model.classes)
    if hasattr(model, "y_train"):
        return np.unique(np.asarray(model.y_train))
    if hasattr(model, "num_classes"):
        return np.arange(int(model.num_classes))
    raise AttributeError(
        f"{type(model).__name__} does not expose classes_, classes, y_train, or num_classes."
    )


# ─── key transform ────────────────────────────────────────────────────────────

def _build_group_plan(K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (order, inv_order) grouping plan from key geometry.

    K: (..., N, D)
    order/inv_order: (BH, N) where BH is product of leading dims.
    """
    *leading, N, D = K.shape
    K_flat = K.reshape(-1, N, D)                     # (BH, N, D)
    BH = K_flat.shape[0]

    # Mean-center keys
    mean = K_flat.mean(dim=1, keepdim=True)
    centered = K_flat - mean                         # (BH, N, D)

    # Use the farthest centered key as deterministic non-degenerate direction.
    # This avoids the zero-vector bug from summing centered vectors.
    norms = centered.norm(dim=-1)                    # (BH, N)
    anchor_idx = norms.argmax(dim=-1)                # (BH,)
    direction = centered[torch.arange(BH, device=K.device), anchor_idx]  # (BH, D)

    # Fallback for fully collapsed keys: use e0 direction.
    dir_norm = direction.norm(dim=-1, keepdim=True)
    direction = direction / dir_norm.clamp(min=1e-8)
    deg = (dir_norm.squeeze(-1) < 1e-8)
    if deg.any():
        direction[deg] = 0.0
        direction[deg, 0] = 1.0

    proj = (centered * direction.unsqueeze(1)).sum(dim=-1)   # (BH, N)
    order = proj.argsort(dim=-1)
    inv_order = order.argsort(dim=-1)
    return order, inv_order


def _kanon_apply_with_plan(X: torch.Tensor,
                           order: torch.Tensor,
                           inv_order: torch.Tensor,
                           k: int,
                           retain_alpha: float = 0.0) -> torch.Tensor:
    """Apply centroid grouping with a fixed permutation plan.

    X: (..., N, D_x)
    order/inv_order: (BH, N), computed once from K and reused for V.
    """
    *leading, N, D = X.shape
    if N < k or k <= 1:
        return X

    X_flat = X.reshape(-1, N, D)
    BH = X_flat.shape[0]
    if order.shape[0] != BH or order.shape[1] != N:
        raise ValueError("Grouping plan shape mismatch between K and V.")

    X_sorted = torch.gather(X_flat, 1, order.unsqueeze(-1).expand(-1, -1, D))

    # Build real groups only.  Padding the final group would leave fewer than k
    # real rows indistinguishable, which is not a proper k-anonymity guarantee.
    # If there is a remainder, merge it into the previous full group.
    n_full, rem = divmod(N, k)
    if rem == 0:
        group_sizes = [k] * n_full
    elif n_full <= 1:
        group_sizes = [N]
    else:
        group_sizes = [k] * (n_full - 1) + [k + rem]

    c_per_pos = torch.empty_like(X_sorted)
    start = 0
    for size in group_sizes:
        end = start + size
        centroid = X_sorted[:, start:end, :].mean(dim=1, keepdim=True)
        replacement = centroid.expand(-1, size, -1)
        c_per_pos[:, start:end, :] = replacement
        start = end

    if retain_alpha > 0.0:
        c_per_pos = retain_alpha * X_sorted + (1.0 - retain_alpha) * c_per_pos

    X_out = torch.gather(c_per_pos, 1, inv_order.unsqueeze(-1).expand(-1, -1, D))
    return X_out.reshape(*leading, N, D)


def _kanon_keys(K: torch.Tensor, k: int) -> torch.Tensor:
    """Replace each key with its group centroid.

    Groups k_len positions by sorting along a deterministic mean-centered
    projection direction, then averages contiguous groups.  If the final group
    would have fewer than k real rows, it is merged into the previous group.
    The same grouping is deterministic per call, so Q-K dot-products are
    consistent.

    K : (..., k_len, d_head) — arbitrary leading batch/head dims.
    Returns same shape with each key replaced by its centroid.
    """
    *leading, N, D = K.shape
    if N < k or k <= 1:
        return K

    order, inv_order = _build_group_plan(K)
    return _kanon_apply_with_plan(K, order, inv_order, k)


def _kanon_apply_by_labels(
    K: torch.Tensor,
    labels: np.ndarray | None,
    k: int,
    retain_alpha: float = 0.0,
) -> torch.Tensor:
    """Apply centroiding independently inside each label group.

    Labels must correspond to the real context rows, not thinking rows.  Label
    groups with fewer than k rows are left unchanged because they cannot provide
    a k-anonymous group without crossing class boundaries.
    """
    *_, N, _ = K.shape
    if labels is None or N < k or k <= 1:
        return K

    labels = np.asarray(labels)
    if labels.shape[0] != N:
        raise ValueError(
            f"label-aware k-anon expected {N} context labels, got {labels.shape[0]}."
        )

    out_K = K.clone()
    for label in np.unique(labels):
        mask = labels == label
        idx_np = np.flatnonzero(mask)
        if idx_np.size < k:
            continue
        idx = torch.as_tensor(idx_np, device=K.device, dtype=torch.long)
        K_label = K.index_select(dim=-2, index=idx)
        order, inv_order = _build_group_plan(K_label)
        K_label = _kanon_apply_with_plan(K_label, order, inv_order, k, retain_alpha=retain_alpha)
        out_K.index_copy_(dim=-2, index=idx, source=K_label)
    return out_K


# ─── SDPA hook ────────────────────────────────────────────────────────────────

class _KAnonSDPA:
    """Context manager that patches F.scaled_dot_product_attention.

    Applies k-anonymity to K for cross-attention calls that match the
    test-to-context pattern used by TabPFN (q_len < k_len).
    """

    def __init__(self, k: int, thinking_rows: int = 0,
                 context_labels: np.ndarray | None = None,
                 retain_alpha: float = 0.0,
                 anonymize_values: bool = False,
                 context_size: int | None = None,
                 attention_mode: str = "tabpfn"):
        self.k = k
        self.thinking_rows = max(0, int(thinking_rows))
        self.context_labels = None if context_labels is None else np.asarray(context_labels)
        self.retain_alpha = float(retain_alpha)
        self.anonymize_values = bool(anonymize_values)
        self.context_size = None if context_size is None else int(context_size)
        self.attention_mode = str(attention_mode).lower()
        self._orig = None

    def __enter__(self):
        global _PATCH_DEPTH, _PATCH_ORIG
        with _PATCH_LOCK:
            if _PATCH_DEPTH == 0:
                _PATCH_ORIG = F.scaled_dot_product_attention
                k = self.k
                thinking_rows = self.thinking_rows
                context_labels = self.context_labels
                retain_alpha = self.retain_alpha
                anonymize_values = self.anonymize_values
                expected_context_size = self.context_size
                attention_mode = self.attention_mode
                orig = _PATCH_ORIG

                def _labels_for_context(n_context_keys: int) -> np.ndarray | None:
                    """Return labels aligned to this key block when possible.

                    Assumes context_labels is ordered identically to the model's
                    internal training-key order (i.e. the order passed to fit()).
                    Count equality is the only runtime check available here; if
                    the model internally reorders its context the labels will be
                    silently misaligned and class grouping will be incorrect.
                    """
                    if context_labels is None:
                        return None
                    labels = np.asarray(context_labels)
                    if labels.shape[0] == n_context_keys:
                        return labels
                    return None

                def _patched(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
                    key_len = K.shape[-2]
                    if thinking_rows > 0 and key_len > thinking_rows:
                        context_key_len = key_len - thinking_rows
                    else:
                        context_key_len = key_len
                    context_size_matched = (
                        expected_context_size is None
                        or expected_context_size == context_key_len
                    )

                    if attention_mode in {"tabicl", "tabdpt", "context_only"}:
                        should_anonymize = (
                            Q.shape[-2] > K.shape[-2]
                            and context_size_matched
                        )
                    else:
                        should_anonymize = (
                            Q.shape[-2] < K.shape[-2]
                            and context_size_matched
                        )

                    if should_anonymize:
                        # TabPFN prepends synthetic thinking rows before training keys.
                        # Exclude those fixed rows from anonymization to avoid utility collapse.
                        if thinking_rows > 0 and K.shape[-2] > thinking_rows:
                            K_keep = K[..., :thinking_rows, :]
                            K_ctx = K[..., thinking_rows:, :]
                            V_keep = V[..., :thinking_rows, :]
                            V_ctx = V[..., thinking_rows:, :]
                            labels = _labels_for_context(K_ctx.shape[-2])
                            if labels is not None:
                                K_ctx = _kanon_apply_by_labels(K_ctx, labels, k, retain_alpha=retain_alpha)
                                if anonymize_values:
                                    V_ctx = _kanon_apply_by_labels(V_ctx, labels, k, retain_alpha=retain_alpha)
                            else:
                                order, inv_order = _build_group_plan(K_ctx)
                                K_ctx = _kanon_apply_with_plan(K_ctx, order, inv_order, k)
                                if anonymize_values:
                                    V_ctx = _kanon_apply_with_plan(V_ctx, order, inv_order, k)
                            K = torch.cat([K_keep, K_ctx], dim=-2)
                            if anonymize_values:
                                V = torch.cat([V_keep, V_ctx], dim=-2)
                        else:
                            labels = _labels_for_context(K.shape[-2])
                            if labels is not None:
                                K = _kanon_apply_by_labels(K, labels, k, retain_alpha=retain_alpha)
                                if anonymize_values:
                                    V = _kanon_apply_by_labels(V, labels, k, retain_alpha=retain_alpha)
                            else:
                                order, inv_order = _build_group_plan(K)
                                K = _kanon_apply_with_plan(K, order, inv_order, k)
                                if anonymize_values:
                                    V = _kanon_apply_with_plan(V, order, inv_order, k)
                    return orig(Q, K, V, attn_mask=attn_mask, dropout_p=dropout_p,
                                is_causal=is_causal, **kw)

                F.scaled_dot_product_attention = _patched
            _PATCH_DEPTH += 1
        return self

    def __exit__(self, *args):
        global _PATCH_DEPTH, _PATCH_ORIG
        with _PATCH_LOCK:
            _PATCH_DEPTH = max(0, _PATCH_DEPTH - 1)
            if _PATCH_DEPTH == 0 and _PATCH_ORIG is not None:
                F.scaled_dot_product_attention = _PATCH_ORIG
                _PATCH_ORIG = None


# ─── wrapper ──────────────────────────────────────────────────────────────────

class KAnonTabFMWrapper:
    """K-anonymity defense wrapper for ICL tabular foundation models.

    Each call to predict_proba installs the _KAnonSDPA hook for the duration
    of the forward pass, so cross-attention keys are replaced by group
    centroids before scores are computed.
    """

    def __init__(self, model, k: int = 5, thinking_rows: int = 0,
                 context_labels: np.ndarray | None = None,
                 retain_alpha: float = 0.0,
                 anonymize_values: bool = False,
                 context_size: int | None = None,
                 attention_mode: str = "tabpfn"):
        self.model    = model
        self.k        = k
        self.thinking_rows = max(0, int(thinking_rows))
        self.context_labels = None if context_labels is None else np.asarray(context_labels)
        self.retain_alpha = float(retain_alpha)
        self.anonymize_values = bool(anonymize_values)
        if context_size is None and self.context_labels is not None:
            context_size = int(self.context_labels.shape[0])
        self.context_size = None if context_size is None else int(context_size)
        self.attention_mode = str(attention_mode).lower()
        self.classes_ = _model_classes(model)

    def predict_proba(self, X: np.ndarray, **predict_kwargs) -> np.ndarray:
        if self.attention_mode == "tabdpt" and self.context_size is not None:
            predict_kwargs.setdefault("context_size", self.context_size)
        with contextlib.ExitStack() as stack:
            if self.attention_mode == "tabicl":
                try:
                    from run_attacks.amia.amia_tabicl import tabicl_capture_runtime
                except ModuleNotFoundError:
                    from amia_tabicl import tabicl_capture_runtime
                stack.enter_context(tabicl_capture_runtime(self.model))
                try:
                    from tabicl.model.attention import flash_attn3_toggle
                except Exception:
                    try:
                        from tabicl._model.attention import flash_attn3_toggle
                    except Exception:
                        flash_attn3_toggle = None
                if flash_attn3_toggle is not None:
                    stack.enter_context(flash_attn3_toggle(False))
            stack.enter_context(_KAnonSDPA(
                self.k,
                thinking_rows=self.thinking_rows,
                context_labels=self.context_labels,
                retain_alpha=self.retain_alpha,
                anonymize_values=self.anonymize_values,
                context_size=self.context_size,
                attention_mode=self.attention_mode,
            ))
            return self.model.predict_proba(X, **predict_kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def __getattr__(self, name: str):
        if name == "predict_logits" and hasattr(self.model, "predict_logits"):
            return lambda X: np.log(self.predict_proba(X) + 1e-12)
        if name == "decision_function" and hasattr(self.model, "decision_function"):
            return lambda X: np.log(self.predict_proba(X) + 1e-12)
        return getattr(self.model, name)


