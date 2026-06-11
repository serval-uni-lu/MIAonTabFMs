"""
HAMP test-time defense.

Reference: "Overconfidence is a Dangerous Thing: Mitigating Membership
Inference Attacks by Enforcing Less Confident Prediction".

Import this module in attack scripts and call wrap_models() after load_models().
"""

import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def _random_samples(n: int, X_ref: np.ndarray) -> np.ndarray:
    """Draw n uniform random tabular samples within the observed feature range.

    Binary columns (only 0/1 observed) are sampled from {0, 1};
    continuous columns are sampled uniformly from [min, max].
    """
    lo  = X_ref.min(axis=0)
    hi  = X_ref.max(axis=0)
    out = np.random.uniform(lo, hi, size=(n, X_ref.shape[1])).astype(X_ref.dtype)
    for j in range(X_ref.shape[1]):
        unique = np.unique(X_ref[:, j])
        if len(unique) <= 2 and set(unique.tolist()).issubset({0.0, 1.0}):
            out[:, j] = np.random.randint(0, 2, size=n)
    return out


def _model_classes(model) -> np.ndarray:
    """Return sklearn-style class labels when a model does not expose classes_."""
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


def hamp_defense(proba: np.ndarray, rand_proba: np.ndarray) -> np.ndarray:
    """Replace proba values with rand_proba values, preserving rank order.

    Example (3 classes):
      proba[i]     = [0.85, 0.05, 0.10]   argsort -> [1, 2, 0]
      rand_proba[i]= [0.20, 0.30, 0.50]   sorted  -> [0.20, 0.30, 0.50]
      result[i]    = [0.50, 0.20, 0.30]   argmax unchanged (class 0)
    """
    rank_idx    = np.argsort(proba, axis=1)
    rand_sorted = np.sort(rand_proba, axis=1)
    defended    = np.empty_like(proba)
    for i in range(len(proba)):
        defended[i, rank_idx[i]] = rand_sorted[i]
    return defended


class HAMPWrapper:
    """Wraps sklearn-style (predict_proba) models with the HAMP defense.

    Used by get_probs_nontorch_models, which checks interfaces in priority
    order: predict_logits > decision_function > predict_proba. HAMPWrapper
    mirrors whichever interface(s) the original model exposes so the
    signal-extraction code path is identical for baseline and defended runs.
    """

    def __init__(self, model, X_ref: np.ndarray):
        self.model    = model
        self.X_ref    = X_ref
        self.classes_ = _model_classes(model)

    def _defended_proba(self, X: np.ndarray) -> np.ndarray:
        proba      = self.model.predict_proba(X)
        rand_proba = self.model.predict_proba(_random_samples(len(X), self.X_ref))
        return hamp_defense(proba, rand_proba)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._defended_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self._defended_proba(X).argmax(axis=1)]

    def __getattr__(self, name: str):
        if name == "predict_logits" and hasattr(self.model, "predict_logits"):
            return lambda X: np.log(self._defended_proba(X) + 1e-12)
        if name == "decision_function" and hasattr(self.model, "decision_function"):
            return lambda X: np.log(self._defended_proba(X) + 1e-12)
        raise AttributeError(name)


class HAMPTorchWrapper(object):
    """Wraps a PyTorch nn.Module with the HAMP defense.

    get_softmax calls model(x) and applies manual softmax to the result.
    forward() returns log(defended_proba) so that the manual softmax in
    get_softmax recovers exactly defended_proba:
      softmax(log(p))_i = exp(log(p_i)) / Σ exp(log(p_j)) = p_i / 1 = p_i
    """

    def __init__(self, model, X_ref: np.ndarray):
        import torch.nn as nn
        if not isinstance(model, nn.Module):
            raise TypeError("HAMPTorchWrapper requires a torch.nn.Module")
        self._wrapped = model
        self.X_ref    = X_ref

    def __call__(self, X):
        import torch
        import torch.nn.functional as F

        device = X.device
        dtype  = X.dtype

        with torch.no_grad():
            proba = F.softmax(self._wrapped(X), dim=-1).cpu().numpy()

        rand_np = _random_samples(len(X), self.X_ref)
        rand_t  = torch.tensor(rand_np, dtype=dtype, device=device)
        with torch.no_grad():
            rand_proba = F.softmax(self._wrapped(rand_t), dim=-1).cpu().numpy()

        defended = hamp_defense(proba, rand_proba)
        return torch.tensor(np.log(defended + 1e-12), dtype=dtype, device=device)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        device = next(self._wrapped.parameters()).device
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        log_proba = self(X_t)
        return torch.exp(log_proba).detach().cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def to(self, device):
        self._wrapped.to(device)
        return self

    def eval(self):
        self._wrapped.eval()
        return self


_TABFM_MODELS = {"tabpfn", "real-tabpfn", "tabicl", "tabdpt"}


class HAMPTabFMWrapper:
    """Output-level HAMP defense for ICL tabular foundation models (TabPFN, TabICL, TabDPT).

    Identical to HAMPWrapper: calls predict_proba directly (no attention patching),
    then rank-calibrates the output probabilities against random-input probabilities.
    """

    def __init__(self, model, X_ref: np.ndarray):
        self.model    = model
        self.X_ref    = X_ref
        self.classes_ = _model_classes(model)

    def _defended_proba(self, X: np.ndarray) -> np.ndarray:
        proba      = self.model.predict_proba(X)
        rand_proba = self.model.predict_proba(_random_samples(len(X), self.X_ref))
        return hamp_defense(proba, rand_proba)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._defended_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self._defended_proba(X).argmax(axis=1)]

    def __getattr__(self, name: str):
        if name == "predict_logits" and hasattr(self.model, "predict_logits"):
            return lambda X: np.log(self._defended_proba(X) + 1e-12)
        if name == "decision_function" and hasattr(self.model, "decision_function"):
            return lambda X: np.log(self._defended_proba(X) + 1e-12)
        raise AttributeError(name)


def make_hamp_wrapper(model, X_ref: np.ndarray, model_name: str = ""):
    """Return the right HAMP wrapper for the given model type.

    TabFM ICL models -> HAMPTabFMWrapper.
    PyTorch nn.Module -> HAMPTorchWrapper.
    Everything else   -> HAMPWrapper.
    """
    import torch.nn as nn
    if model_name.lower() in _TABFM_MODELS:
        return HAMPTabFMWrapper(model, X_ref)
    if isinstance(model, nn.Module):
        return HAMPTorchWrapper(model, X_ref)
    return HAMPWrapper(model, X_ref)


def _get_proba(model, X: np.ndarray) -> np.ndarray:
    """Return probability array (n_samples, n_classes) for any model type."""
    import torch.nn as nn
    if isinstance(model, nn.Module):
        import torch
        import torch.nn.functional as F
        device = next(model.parameters()).device
        with torch.no_grad():
            out = F.softmax(
                model(torch.tensor(X, dtype=torch.float32, device=device)).float(),
                dim=-1,
            )
        return out.cpu().numpy()
    return model.predict_proba(X)


def _get_preds(model, X: np.ndarray) -> np.ndarray:
    """Return predicted class labels for any model type."""
    idx = _get_proba(model, X).argmax(axis=1)
    try:
        return _model_classes(model)[idx]
    except AttributeError:
        return idx


def log_defense_accuracy(orig_model, defended_model, X: np.ndarray, y: np.ndarray,
                          report_dir: str, logger) -> tuple:
    """Log accuracy and average confidence before/after HAMP defense.

    Saves report_dir/defense_accuracy.csv.
    Returns (acc_orig, acc_defended).
    """
    import os
    import csv

    pred_orig = _get_preds(orig_model, X)
    pred_def  = _get_preds(defended_model, X)
    acc_orig  = float((pred_orig == y).mean())
    acc_def   = float((pred_def  == y).mean())

    prob_orig  = _get_proba(orig_model, X)
    prob_def   = _get_proba(defended_model, X)
    conf_orig  = float(prob_orig.max(axis=1).mean())
    conf_def   = float(prob_def.max(axis=1).mean())

    logger.info(
        "Defense accuracy  — no_defense: %.4f  defended: %.4f  drop: %.4f",
        acc_orig, acc_def, acc_orig - acc_def,
    )
    logger.info(
        "Defense confidence — no_defense: %.4f  defended: %.4f  drop: %.4f",
        conf_orig, conf_def, conf_orig - conf_def,
    )

    os.makedirs(report_dir, exist_ok=True)
    csv_path = os.path.join(report_dir, "defense_accuracy.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "no_defense", "defended", "drop"])
        w.writerow(["accuracy",       f"{acc_orig:.6f}",  f"{acc_def:.6f}",  f"{acc_orig - acc_def:.6f}"])
        w.writerow(["avg_confidence", f"{conf_orig:.6f}", f"{conf_def:.6f}", f"{conf_orig - conf_def:.6f}"])

    return acc_orig, acc_def


def wrap_models(models_list: list, X_ref: np.ndarray, defense: str, model_name: str = "") -> list:
    """Wrap every model in models_list with the requested defense.

    defense="none" (default) is a no-op — returns models_list unchanged.
    Dispatches to make_hamp_wrapper which selects the right wrapper class
    based on model_name: TabFM models get output-level HAMP; torch
    nn.Module models get HAMPTorchWrapper; others get HAMPWrapper.
    """
    if not defense or defense == "none":
        return models_list
    if defense == "hamp":
        return [make_hamp_wrapper(m, X_ref, model_name) for m in models_list]
    raise ValueError(f"Unknown defense: {defense!r}")
