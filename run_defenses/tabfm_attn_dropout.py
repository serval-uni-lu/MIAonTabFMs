"""
Inference-time dropout defenses for tabular foundation models.

Two independent defenses:

AttnDropoutWrapper — attention weight dropout
    Drops individual entries of softmax(QK^T/√d) inside each attention head.
    Randomises which training samples each query attends to.
    Target: AMIA (attention signal).  Caveat: stochastic, averages out over
    many queries.

LayerDropoutWrapper — hidden state (layer) dropout
    Applies F.dropout to the hidden state tensor *after* each transformer
    sublayer output (before it is added to the residual).  Corrupts the full
    representation at every layer, making the model's predictions noisier.
    Target: RMIA (output confidence signal).  More aggressive than attention
    dropout; trades more accuracy for stronger stochasticity.

Both are STOCHASTIC defences — a strong attacker who can average over multiple
queries will recover the undefended signal.  For a DETERMINISTIC defence use
k-anon (index-based) from tabfm_kanon.py.

Usage
-----
    from run_defenses.tabfm_attn_dropout import (
        AttnDropoutWrapper, KNNKeySmoothWrapper, LayerDropoutWrapper,
    )

    wrapped_attn  = AttnDropoutWrapper(model, p=0.3)
    wrapped_knn = KNNKeySmoothWrapper(model, knn_k=5, alpha=0.7)
    wrapped_guard = HighRiskQueryDropoutWrapper(
        model, n_context=500, threshold=0.03, p=0.3, layer_indices="4,5,7,8,10,11"
    )
    wrapped_knn_guard = HighRiskQueryKNNWrapper(
        model, n_context=500, threshold=0.03, knn_k=5, alpha=0.7
    )
    wrapped_layer = LayerDropoutWrapper(model, p=0.2)
    proba = wrapped_attn.predict_proba(X_test)
"""

import numpy as np
import torch
import torch.nn.functional as F
import threading
import contextlib


_KNN_PATCH_LOCK = threading.RLock()
_KNN_PATCH_DEPTH = 0
_KNN_PATCH_ORIG = None

_ADROP_SDPA_LOCK = threading.RLock()
_ADROP_SDPA_DEPTH = 0
_ADROP_SDPA_ORIG = None


@contextlib.contextmanager
def _tabicl_predict_context(model):
    """Enter TabICL-specific runtime contexts required for SDPA-based defenses.

    TabICL uses an internal KV cache (model_kv_cache_) that stores the undefended
    training keys after the first predict_proba call.  Without clearing it, any
    SDPA-hook defense is bypassed — predictions use the cached undefended K/V
    instead of recomputing through the patched SDPA.  TabICL also has an FA3
    toggle (use_fa3) that must be disabled so attention flows through Python's
    F.scaled_dot_product_attention where the hooks live.

    KAnonTabFMWrapper enters tabicl_capture_runtime natively; this helper makes
    the same setup available to all other SDPA-based defense wrappers.
    """
    with contextlib.ExitStack() as stack:
        try:
            try:
                from run_attacks.amia.amia_tabicl import tabicl_capture_runtime
            except ModuleNotFoundError:
                from amia_tabicl import tabicl_capture_runtime
            stack.enter_context(tabicl_capture_runtime(model))
        except Exception:
            pass
        try:
            try:
                from tabicl.model.attention import flash_attn3_toggle
            except Exception:
                try:
                    from tabicl._model.attention import flash_attn3_toggle
                except Exception:
                    flash_attn3_toggle = None
            if flash_attn3_toggle is not None:
                stack.enter_context(flash_attn3_toggle(False))
        except Exception:
            pass
        yield


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


def _parse_layer_indices(layer_indices, n_layers: int) -> set[int] | None:
    """Return selected 0-based layer indices, or None for all layers."""
    if layer_indices is None:
        return None
    if isinstance(layer_indices, str):
        spec = layer_indices.strip()
        if spec == "all":
            return None
        if spec in {"late", "tail"}:
            start = max(0, n_layers - 6)
            return set(range(start, n_layers))
        raise ValueError(
            "Unsupported attention-dropout layer selection "
            f"{layer_indices!r}. Use all, late, or --auto-top-dropout."
        )
    return {int(i) for i in layer_indices if 0 <= int(i) < n_layers}


# ── attention weight dropout ──────────────────────────────────────────────────

class _AttnDropoutCtx:
    """Applies attention-weight dropout.

    Primary path (TabPFN): sets _attn_defense_dropout_p on self_attn_between_items modules.
    Fallback path (TabDPT/TabICL): patches F.scaled_dot_product_attention to pass dropout_p
    for cross-attention calls (q_len != k_len), covering both TabPFN-direction (q < k) and
    TabDPT/TabICL-direction (q > k) patterns.
    """

    def __init__(self, model, p: float, layer_indices=None):
        self.p = p
        self._mods: list = []
        for arch in getattr(model, "models_", []):
            if not hasattr(arch, "transformer_encoder"):
                continue
            layers = arch.transformer_encoder.layers
            selected = _parse_layer_indices(layer_indices, len(layers))
            for idx, layer in enumerate(layers):
                if selected is not None and idx not in selected:
                    continue
                if hasattr(layer, "self_attn_between_items"):
                    self._mods.append(layer.self_attn_between_items)
        self._use_sdpa_fallback = not self._mods

    def __enter__(self):
        if self._mods:
            for mod in self._mods:
                mod._attn_defense_dropout_p = self.p
        else:
            global _ADROP_SDPA_DEPTH, _ADROP_SDPA_ORIG
            with _ADROP_SDPA_LOCK:
                if _ADROP_SDPA_DEPTH == 0:
                    _ADROP_SDPA_ORIG = F.scaled_dot_product_attention
                    p = self.p
                    orig = _ADROP_SDPA_ORIG

                    def _patched(Q, K, V, attn_mask=None, dropout_p=0.0,
                                 is_causal=False, **kw):
                        if Q.shape[-2] != K.shape[-2] and not is_causal:
                            dropout_p = p
                        return orig(Q, K, V, attn_mask=attn_mask,
                                    dropout_p=dropout_p, is_causal=is_causal, **kw)

                    F.scaled_dot_product_attention = _patched
                _ADROP_SDPA_DEPTH += 1
        return self

    def __exit__(self, *_):
        if self._mods:
            for mod in self._mods:
                mod._attn_defense_dropout_p = None
        else:
            global _ADROP_SDPA_DEPTH, _ADROP_SDPA_ORIG
            with _ADROP_SDPA_LOCK:
                _ADROP_SDPA_DEPTH = max(0, _ADROP_SDPA_DEPTH - 1)
                if _ADROP_SDPA_DEPTH == 0 and _ADROP_SDPA_ORIG is not None:
                    F.scaled_dot_product_attention = _ADROP_SDPA_ORIG
                    _ADROP_SDPA_ORIG = None


class AttnDropoutWrapper:
    """Inference-time attention weight dropout.

    Drops entries of softmax(QK^T/√d) in every cross-attention call.
    Directly disrupts the AMIA signal but averages out over many queries.

    For TabICL (attention_mode="tabicl") the wrapper also enters
    _tabicl_predict_context to clear the KV cache and disable FA3, which is
    required for SDPA-hook defenses to actually modify the model's computation.
    """

    def __init__(self, model, p: float = 0.2, layer_indices=None,
                 attention_mode: str = "tabpfn"):
        if not (0.0 <= p < 1.0):
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.model    = model
        self.p        = p
        self.layer_indices = layer_indices
        self.attention_mode = str(attention_mode).lower()
        self.classes_ = _model_classes(model)
        self._ctx     = _AttnDropoutCtx(model, p, layer_indices=layer_indices)

    def predict_proba(self, X: np.ndarray, **predict_kwargs) -> np.ndarray:
        with contextlib.ExitStack() as stack:
            if self.attention_mode == "tabicl":
                stack.enter_context(_tabicl_predict_context(self.model))
            stack.enter_context(self._ctx)
            return self.model.predict_proba(X, **predict_kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def __getattr__(self, name: str):
        return getattr(self.model, name)


# ── deterministic kNN key smoothing ─────────────────────────────────────────

def _knn_smooth_keys(K: torch.Tensor, knn_k: int, alpha: float) -> torch.Tensor:
    """Move each key toward the mean of its k nearest neighbors in key space."""
    *leading, N, D = K.shape
    if knn_k <= 1 or N <= 1 or alpha >= 1.0:
        return K

    k_eff = min(int(knn_k), N - 1)
    K_flat = K.reshape(-1, N, D)
    BH = K_flat.shape[0]
    # Compute neighbors in float32 for numerical stability, then cast the
    # smoothed keys back to the original dtype.
    K_metric = K_flat.float()
    dist = torch.cdist(K_metric, K_metric, p=2)
    diag = torch.arange(N, device=K_flat.device)
    dist[:, diag, diag] = float("inf")
    nn_idx = dist.topk(k=k_eff, dim=-1, largest=False).indices  # (BH, N, k)
    # Use advanced indexing instead of expand+gather to avoid the O(N²·D)
    # temporary tensor that would blow up GPU memory for large contexts.
    b_idx = torch.arange(BH, device=K_flat.device)[:, None, None].expand(BH, N, k_eff)
    neigh = K_flat[b_idx, nn_idx]          # (BH, N, k, D)
    neigh_mean = neigh.mean(dim=2)
    K_out = alpha * K_flat + (1.0 - alpha) * neigh_mean
    return K_out.reshape(*leading, N, D)


class _KNNKeySmoothSDPA:
    """kNN key smoothing for real context rows in row cross-attention.

    For each real context key K_i, find its nearest neighbors among the other
    context keys and replace it with alpha*K_i + (1-alpha)*mean(kNN_i).  This is
    softer than hard k-anonymity and should preserve utility better.
    """

    def __init__(self, knn_k: int, alpha: float = 0.7, thinking_rows: int = 0):
        if knn_k <= 1:
            raise ValueError(f"knn_k must be > 1, got {knn_k}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.knn_k = int(knn_k)
        self.alpha = float(alpha)
        self.thinking_rows = max(0, int(thinking_rows))

    def __enter__(self):
        global _KNN_PATCH_DEPTH, _KNN_PATCH_ORIG
        with _KNN_PATCH_LOCK:
            if _KNN_PATCH_DEPTH == 0:
                _KNN_PATCH_ORIG = F.scaled_dot_product_attention
                knn_k = self.knn_k
                alpha = self.alpha
                thinking_rows = self.thinking_rows
                orig = _KNN_PATCH_ORIG

                def _patched(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
                    # Fire for both TabPFN (q < k) and TabDPT/TabICL (q > k) cross-attention.
                    if Q.shape[-2] != K.shape[-2] and not is_causal:
                        if thinking_rows > 0 and K.shape[-2] > thinking_rows + 1:
                            K_keep = K[..., :thinking_rows, :]
                            K_ctx = K[..., thinking_rows:, :]
                            K_ctx = _knn_smooth_keys(K_ctx, knn_k=knn_k, alpha=alpha)
                            K = torch.cat([K_keep, K_ctx], dim=-2)
                        else:
                            K = _knn_smooth_keys(K, knn_k=knn_k, alpha=alpha)
                    return orig(Q, K, V, attn_mask=attn_mask, dropout_p=dropout_p,
                                is_causal=is_causal, **kw)

                F.scaled_dot_product_attention = _patched
            _KNN_PATCH_DEPTH += 1
        return self

    def __exit__(self, *_):
        global _KNN_PATCH_DEPTH, _KNN_PATCH_ORIG
        with _KNN_PATCH_LOCK:
            _KNN_PATCH_DEPTH = max(0, _KNN_PATCH_DEPTH - 1)
            if _KNN_PATCH_DEPTH == 0 and _KNN_PATCH_ORIG is not None:
                F.scaled_dot_product_attention = _KNN_PATCH_ORIG
                _KNN_PATCH_ORIG = None


class KNNKeySmoothWrapper:
    """Deterministic kNN key smoothing for row attention.

    knn_k controls the local neighborhood size; alpha controls how much of the
    original key is retained.  alpha=1 is no-op, alpha=0 fully replaces each key
    by its neighbor mean.
    """

    def __init__(self, model, knn_k: int = 5, alpha: float = 0.7,
                 thinking_rows: int = 0, attention_mode: str = "tabpfn"):
        self.model = model
        self.knn_k = int(knn_k)
        self.alpha = float(alpha)
        self.thinking_rows = max(0, int(thinking_rows))
        self.attention_mode = str(attention_mode).lower()
        self.classes_ = _model_classes(model)

    def predict_proba(self, X: np.ndarray, **predict_kwargs) -> np.ndarray:
        with contextlib.ExitStack() as stack:
            if self.attention_mode == "tabicl":
                stack.enter_context(_tabicl_predict_context(self.model))
            stack.enter_context(_KNNKeySmoothSDPA(
                self.knn_k,
                alpha=self.alpha,
                thinking_rows=self.thinking_rows,
            ))
            return self.model.predict_proba(X, **predict_kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def __getattr__(self, name: str):
        return getattr(self.model, name)


# ── adaptive high-risk query attention dropout ──────────────────────────────

def _row_max_risk_from_records(records: list, chunk: int) -> np.ndarray:
    """AMIA-compatible row_max risk for a single query batch."""
    row_calls = [r for r in records if r.get("type") == "row"]
    if not row_calls:
        row_calls = [r for r in records if r.get("type") == "icl"]
    if not row_calls:
        row_calls = [r for r in records if r.get("type") == "dpt"]
    if not row_calls:
        return np.zeros(chunk, dtype=np.float32)
    rm = np.stack(
        [r["max_attn"][:, :chunk] for r in row_calls],
        axis=0,
    ).transpose(2, 0, 1)
    return rm.mean(axis=(1, 2)).astype(np.float32)


class HighRiskQueryFallbackWrapper:
    """Apply a fallback wrapper only to queries whose AMIA probe risk is high.

    The wrapper runs a normal probe forward pass while collecting AMIA-style
    row attention.  Queries whose row_max risk exceeds ``threshold`` are rerun
    with ``fallback_model`` and replace the original probabilities.

    Query risk answers "should this query be defended?"  The fallback wrapper
    answers "how should this query be protected?"
    """

    def __init__(
        self,
        model,
        fallback_model,
        n_context: int,
        threshold: float,
        probe_batch_size: int = 64,
        capture_backend: str = "tabpfn",
        thinking_rows: int | None = None,
    ):
        self.model = model
        self.fallback_model = fallback_model
        self.n_context = int(n_context)
        self.threshold = float(threshold)
        self.probe_batch_size = max(1, int(probe_batch_size))
        self.capture_backend = str(capture_backend).lower()
        self.thinking_rows = None if thinking_rows is None else max(0, int(thinking_rows))
        self.classes_ = _model_classes(model)
        self.last_risk_scores_: np.ndarray | None = None
        self.last_high_risk_mask_: np.ndarray | None = None

    @contextlib.contextmanager
    def _backend_runtime(self):
        """Expose Python SDPA calls for backends that otherwise use cache/flash paths."""
        with contextlib.ExitStack() as stack:
            if self.capture_backend == "tabicl":
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
            yield

    def _probe_chunk(
        self,
        X: np.ndarray,
        predict_kwargs: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        predict_kwargs = dict(predict_kwargs or {})
        chunk = len(X)
        if self.capture_backend in {"tabicl", "tabdpt"}:
            try:
                if self.capture_backend == "tabdpt":
                    from run_attacks.amia.amia_tabdpt import SDPACapture
                else:
                    from run_attacks.amia.amia_tabicl import SDPACapture
            except ModuleNotFoundError:
                if self.capture_backend == "tabdpt":
                    from amia_tabdpt import SDPACapture
                else:
                    from amia_tabicl import SDPACapture
            if self.capture_backend == "tabdpt":
                predict_kwargs.setdefault("context_size", self.n_context)
        else:
            try:
                from run_attacks.amia.amia_tabpfn import SDPACapture, infer_tabpfn_thinking_rows
            except ModuleNotFoundError:
                from amia_tabpfn import SDPACapture, infer_tabpfn_thinking_rows

            n_thinking = (
                self.thinking_rows
                if self.thinking_rows is not None
                else infer_tabpfn_thinking_rows(self.model, default=0)
            )
            expected_k = self.n_context + int(n_thinking or 0)
            if chunk == expected_k and chunk > 1:
                raise ValueError(
                    "HighRiskQueryFallbackWrapper got a probe chunk with "
                    f"chunk_size == n_context + thinking_rows ({chunk}); "
                    "reduce probe_batch_size to avoid ambiguous row attention."
                )

        if self.capture_backend in {"tabicl", "tabdpt"}:
            ctx = SDPACapture(chunk_size=chunk, n_context=self.n_context)
        else:
            ctx = SDPACapture(
                chunk_size=chunk,
                n_context=self.n_context,
                n_thinking=n_thinking,
            )
        # Guard against torch.compile/fused paths that bypass the Python SDPA hook
        # (same guard used in extract_attention_signals in the AMIA backends).
        predict_fn = self.model.predict_proba
        try:
            import torch._dynamo
            predict_fn = torch._dynamo.disable(predict_fn)
        except Exception:
            pass
        with self._backend_runtime():
            with ctx:
                probs = predict_fn(X, **predict_kwargs)
        risk = _row_max_risk_from_records(ctx.records, chunk)
        return probs, risk

    def predict_proba(self, X: np.ndarray, **predict_kwargs) -> np.ndarray:
        X = np.asarray(X)
        probs_batches = []
        risk_batches = []
        mask_batches = []

        # kwargs for the clean compiled model call — use only what the caller provided,
        # without the probe-internal additions (e.g. context_size for tabdpt) so that
        # non-high-risk predictions match the undefended baseline exactly.
        clean_kwargs = dict(predict_kwargs)

        try:
            import torch
            _torch_available = True
        except ImportError:
            _torch_available = False

        for start in range(0, len(X), self.probe_batch_size):
            xb = X[start: start + self.probe_batch_size]

            # Snapshot RNG state before the probe so the clean-model call sees the
            # same random state as if the probe never ran (the probe consumes GPU
            # random numbers, e.g. from inference-time stochasticity, which would
            # otherwise shift predictions away from the undefended baseline).
            if _torch_available:
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

            _, risk = self._probe_chunk(xb, predict_kwargs=predict_kwargs)

            high_risk = risk >= self.threshold

            if high_risk.all():
                # All queries need the fallback — skip the clean model call entirely.
                probs = self.fallback_model.predict_proba(xb, **predict_kwargs)
            else:
                # Restore RNG so non-high-risk predictions match the undefended baseline.
                if _torch_available:
                    torch.set_rng_state(cpu_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state_all(cuda_rng)
                probs = self.model.predict_proba(xb, **clean_kwargs)
                if high_risk.any():
                    probs = probs.copy()
                    # fallback_model manages its own backend runtime (e.g. KAnonTabFMWrapper
                    # calls tabicl_capture_runtime internally).
                    probs[high_risk] = self.fallback_model.predict_proba(
                        xb[high_risk],
                        **predict_kwargs,
                    )

            probs_batches.append(probs)
            risk_batches.append(risk)
            mask_batches.append(high_risk)

        self.last_risk_scores_ = np.concatenate(risk_batches)
        self.last_high_risk_mask_ = np.concatenate(mask_batches)
        return np.concatenate(probs_batches)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def __getattr__(self, name: str):
        return getattr(self.model, name)


class HighRiskQueryDropoutWrapper(HighRiskQueryFallbackWrapper):
    """Rerun only high-risk queries with attention dropout."""

    def __init__(
        self,
        model,
        n_context: int,
        threshold: float,
        p: float = 0.3,
        layer_indices=None,
        probe_batch_size: int = 64,
        capture_backend: str = "tabpfn",
        thinking_rows: int | None = None,
    ):
        if not (0.0 <= p < 1.0):
            raise ValueError(f"p must be in [0, 1), got {p}")
        super().__init__(
            model=model,
            fallback_model=AttnDropoutWrapper(model, p=p, layer_indices=layer_indices),
            n_context=n_context,
            threshold=threshold,
            probe_batch_size=probe_batch_size,
            capture_backend=capture_backend,
            thinking_rows=thinking_rows,
        )
        self.p = float(p)
        self.layer_indices = layer_indices


class HighRiskQueryKNNWrapper(HighRiskQueryFallbackWrapper):
    """Rerun only high-risk queries with kNN key smoothing."""

    def __init__(
        self,
        model,
        n_context: int,
        threshold: float,
        knn_k: int = 5,
        alpha: float = 0.7,
        thinking_rows: int = 0,
        probe_batch_size: int = 64,
        capture_backend: str = "tabpfn",
        attention_mode: str = "tabpfn",
    ):
        super().__init__(
            model=model,
            fallback_model=KNNKeySmoothWrapper(
                model,
                knn_k=knn_k,
                alpha=alpha,
                thinking_rows=thinking_rows,
                attention_mode=attention_mode,
            ),
            n_context=n_context,
            threshold=threshold,
            probe_batch_size=probe_batch_size,
            capture_backend=capture_backend,
            thinking_rows=thinking_rows,
        )
        self.knn_k = int(knn_k)
        self.alpha = float(alpha)
        self.thinking_rows = max(0, int(thinking_rows))


# ── layer (hidden state) dropout ─────────────────────────────────────────────

class _LayerDropoutCtx:
    """Registers a forward hook on every transformer layer that applies
    F.dropout to the output hidden state tensor.

    The hook fires after the full sublayer stack (attn + MLP + LayerNorm) and
    drops random elements of the (batch, n_items, n_blocks, d_model) tensor.
    This corrupts the representations flowing between layers, making the final
    output probabilities stochastic regardless of how the attention behaves.
    """

    def __init__(self, model, p: float):
        self.p = p
        self._hooks: list = []
        self._layers: list = []
        for arch in getattr(model, "models_", []):
            if not hasattr(arch, "transformer_encoder"):
                continue
            for layer in arch.transformer_encoder.layers:
                self._layers.append(layer)
        if not self._layers:
            import warnings
            warnings.warn(
                "LayerDropoutWrapper could not find transformer layers to hook "
                f"in {type(model).__name__}. "
                "This defense supports TabPFN-style transformer_encoder layers; "
                "it will be a no-op for TabDPT/TabICL.",
                UserWarning,
                stacklevel=3,
            )

    def __enter__(self):
        p = self.p

        def _hook(module, inputs, output):
            # output is the hidden state tensor after the full sublayer stack
            if isinstance(output, torch.Tensor):
                return F.dropout(output, p=p, training=True)
            return output

        for layer in self._layers:
            self._hooks.append(layer.register_forward_hook(_hook))
        return self

    def __exit__(self, *_):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class LayerDropoutWrapper:
    """Inference-time hidden-state (layer) dropout.

    After each transformer layer's full sublayer stack the hidden state is
    randomly zeroed at rate p.  This injects noise into the representations
    propagating through all subsequent layers, making the final output
    probabilities stochastic.

    More aggressive than attention dropout — affects both attention and MLP
    paths and compounds across layers.  Stronger RMIA defence at the cost of
    larger accuracy degradation.
    """

    def __init__(self, model, p: float = 0.1):
        if not (0.0 <= p < 1.0):
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.model    = model
        self.p        = p
        self.classes_ = _model_classes(model)
        self._ctx     = _LayerDropoutCtx(model, p)

    def predict_proba(self, X: np.ndarray, **predict_kwargs) -> np.ndarray:
        with self._ctx:
            return self.model.predict_proba(X, **predict_kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def __getattr__(self, name: str):
        return getattr(self.model, name)
