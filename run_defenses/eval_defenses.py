#!/usr/bin/env python3
"""
General defense evaluator: tests k-anonymity, attention dropout, and layer
dropout defenses individually and (optionally) in combination against both
RMIA and AMIA.

Each defense is evaluated by:
  - Recomputing RMIA signals for ALL models under the defense (so the attacker
    sees only defended outputs) and reporting the resulting RMIA AUC.
  - Running the full AMIA pipeline on the target model under the defense and
    reporting AMIA row_max AUC + Cohen's d.  All AMIA plots are saved per
    defense configuration.

Defense-specific outputs go to:
    ml_privacy_meter/logs/<dataset>/<model>/defense/<name>/

The summary table and results CSV go to:
    ml_privacy_meter/logs/<dataset>/<model>/defense/

Usage
-----
  uv run run_defenses/eval_defenses.py --dataset locations --model tabpfn
  uv run run_defenses/eval_defenses.py --dataset locations --model tabpfn \\
      --kanon-ks 5 10 20 \\
      --attn-dropout-ps 0.1 0.3 0.5 \\
      --layer-dropout-ps 0.1 0.3 0.5 \\
      --attacks rmia amia --combine
"""

import csv
import gc
import os
import re
import subprocess
import sys
import time
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "run_attacks" / "amia"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ml_privacy_meter"))


def _cleanup_runtime(logger, scope_name: str) -> None:
    """Best-effort host/GPU memory cleanup between defense runs."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("[%s] Cleared CUDA cache", scope_name)
    except Exception as exc:
        logger.warning("[%s] CUDA cleanup skipped: %s", scope_name, exc)


# ── signal helpers ────────────────────────────────────────────────────────────

def _model_classes(model) -> np.ndarray:
    """Return sklearn-style class labels for estimators that may not expose classes_."""
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


def _compute_signal(model, X: np.ndarray, y: np.ndarray,
                    batch_size: int = 200) -> np.ndarray:
    """True-class probability for each sample."""
    classes = _model_classes(model)
    cls_idx = {c: i for i, c in enumerate(classes)}
    out = []
    for start in range(0, len(X), batch_size):
        xb = X[start: start + batch_size]
        yb = y[start: start + batch_size]
        probs = model.predict_proba(xb)
        idx   = np.array([cls_idx.get(yi, -1) for yi in yb])
        valid = (idx >= 0) & (idx < probs.shape[1])
        tp    = np.full(len(yb), 1e-12, dtype=np.float64)
        tp[valid] = probs[np.arange(len(yb))[valid], idx[valid]]
        out.append(tp)
    return np.concatenate(out)


def _atomic_save_npy(path: str, arr: np.ndarray) -> None:
    """Atomically save a NumPy array so interrupted runs leave valid checkpoints."""
    tmp_path = f"{path}.tmp.npy"
    np.save(tmp_path, arr)
    os.replace(tmp_path, path)


def _load_signal_checkpoint(checkpoint_dir: str | None,
                            rmia_signals: np.ndarray,
                            rmia_signals_pop: np.ndarray,
                            logger,
                            defense_name: str):
    """Load partial defended signal recomputation state if it matches this run."""
    signals = rmia_signals.copy()
    signals_pop = rmia_signals_pop.copy()
    done = np.zeros(signals.shape[1], dtype=bool)
    done_pop = np.zeros(signals_pop.shape[1], dtype=bool)

    if checkpoint_dir is None:
        return signals, signals_pop, done, done_pop

    final_paths = (
        os.path.join(checkpoint_dir, "rmia_signals.npy"),
        os.path.join(checkpoint_dir, "rmia_signals_pop.npy"),
    )
    if all(os.path.exists(path) for path in final_paths):
        return signals, signals_pop, done, done_pop

    paths = {
        "signals": os.path.join(checkpoint_dir, "rmia_signals.partial.npy"),
        "signals_pop": os.path.join(checkpoint_dir, "rmia_signals_pop.partial.npy"),
        "done": os.path.join(checkpoint_dir, "rmia_signals.partial_done.npy"),
        "done_pop": os.path.join(checkpoint_dir, "rmia_signals_pop.partial_done.npy"),
    }
    try:
        if os.path.exists(paths["signals"]) and os.path.exists(paths["done"]):
            partial = np.load(paths["signals"])
            partial_done = np.load(paths["done"]).astype(bool)
            if partial.shape == signals.shape and partial_done.shape == done.shape:
                signals = partial
                done = partial_done
                logger.info(
                    "[%s] Resuming RMIA audit-signal checkpoint: %d / %d model columns done",
                    defense_name, int(done.sum()), len(done),
                )
        if os.path.exists(paths["signals_pop"]) and os.path.exists(paths["done_pop"]):
            partial_pop = np.load(paths["signals_pop"])
            partial_done_pop = np.load(paths["done_pop"]).astype(bool)
            if (
                partial_pop.shape == signals_pop.shape
                and partial_done_pop.shape == done_pop.shape
            ):
                signals_pop = partial_pop
                done_pop = partial_done_pop
                logger.info(
                    "[%s] Resuming RMIA population-signal checkpoint: %d / %d model columns done",
                    defense_name, int(done_pop.sum()), len(done_pop),
                )
    except Exception as exc:
        logger.warning("[%s] Ignoring invalid signal checkpoint in %s: %s",
                       defense_name, checkpoint_dir, exc)
        signals = rmia_signals.copy()
        signals_pop = rmia_signals_pop.copy()
        done[:] = False
        done_pop[:] = False

    return signals, signals_pop, done, done_pop


def _save_signal_checkpoint(checkpoint_dir: str,
                            signals: np.ndarray,
                            signals_pop: np.ndarray,
                            done: np.ndarray,
                            done_pop: np.ndarray) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    _atomic_save_npy(os.path.join(checkpoint_dir, "rmia_signals.partial.npy"), signals)
    _atomic_save_npy(os.path.join(checkpoint_dir, "rmia_signals_pop.partial.npy"), signals_pop)
    _atomic_save_npy(os.path.join(checkpoint_dir, "rmia_signals.partial_done.npy"), done)
    _atomic_save_npy(os.path.join(checkpoint_dir, "rmia_signals_pop.partial_done.npy"), done_pop)


def _accuracy(model, X: np.ndarray, y: np.ndarray,
              batch_size: int = 200) -> float:
    preds = []
    for start in range(0, len(X), batch_size):
        preds.append(model.predict(X[start: start + batch_size]))
    return (np.concatenate(preds) == y).mean()


def _move_tabdpt_model(model, device: str, logger=None):
    """Move a TabDPT estimator/wrapper stack to a consistent inference device."""
    if hasattr(model, "to"):
        model.to(device)
    inner = getattr(model, "model", None)
    while inner is not None and inner is not model:
        if hasattr(inner, "to"):
            inner.to(device)
        next_inner = getattr(inner, "model", None)
        if next_inner is inner:
            break
        inner = next_inner
    if logger is not None:
        logger.info("Moved TabDPT model stack to %s", device)
    return model


def _dataset_arrays(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return NumPy arrays from a TabularDataset or torch Subset."""
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        base = dataset.dataset
        idx = np.asarray(dataset.indices)
        return base.data[idx], base.targets[idx]
    return dataset.data, dataset.targets


def _build_eval_arrays(X: np.ndarray, y: np.ndarray, configs: dict,
                       memberships: np.ndarray,
                       n_pool: int, n_pop: int, logger=None):
    """Reconstruct the exact RMIA audit/population arrays used for signals."""
    from audit import sample_auditing_dataset
    from dataset.tabular import TabularDataset

    audit_pool_size = memberships.shape[1]
    population_start = int(len(y) * 0.75)
    dataset = TabularDataset(X[:audit_pool_size], y[:audit_pool_size])

    np.random.seed(configs["run"].get("random_seed", 12345))
    auditing_dataset, auditing_membership = sample_auditing_dataset(
        configs, dataset, logger, memberships
    )
    X_pool, y_pool = _dataset_arrays(auditing_dataset)

    if len(X_pool) != n_pool or auditing_membership.shape[1] != n_pool:
        raise ValueError(
            "RMIA signal rows do not match reconstructed audit dataset: "
            f"signals={n_pool}, audit={len(X_pool)}, memberships={auditing_membership.shape}"
        )

    X_pop = X[population_start: population_start + n_pop]
    y_pop = y[population_start: population_start + n_pop]
    if len(X_pop) != n_pop:
        raise ValueError(
            f"RMIA population signals expect {n_pop} rows, but only {len(X_pop)} are available."
        )

    return X_pool, y_pool, X_pop, y_pop, auditing_membership


def _compute_defended_signals(wrap_fn, models_list, X_pool, y_pool, X_pop, y_pop,
                              rmia_signals, rmia_signals_pop,
                              batch_size, logger, defense_name,
                              checkpoint_dir: str | None = None):
    """Recompute signals for all models wrapped under the defense."""
    if wrap_fn is None:
        return rmia_signals, rmia_signals_pop

    signals, signals_pop, done, done_pop = _load_signal_checkpoint(
        checkpoint_dir,
        rmia_signals,
        rmia_signals_pop,
        logger,
        defense_name,
    )
    n = len(models_list)
    for i, m in enumerate(models_list):
        logger.info("[%s] Signal recomputation: model %d / %d", defense_name, i + 1, n)
        dm = wrap_fn(m, i)
        if done[i]:
            logger.info("[%s] Reusing checkpointed audit signals for model %d / %d",
                        defense_name, i + 1, n)
        else:
            signals[:, i] = _compute_signal(dm, X_pool, y_pool, batch_size)
            done[i] = True
            if checkpoint_dir is not None:
                _save_signal_checkpoint(checkpoint_dir, signals, signals_pop, done, done_pop)
        if done_pop[i]:
            logger.info("[%s] Reusing checkpointed population signals for model %d / %d",
                        defense_name, i + 1, n)
        else:
            signals_pop[:, i] = _compute_signal(dm, X_pop, y_pop, batch_size)
            done_pop[i] = True
            if checkpoint_dir is not None:
                _save_signal_checkpoint(checkpoint_dir, signals, signals_pop, done, done_pop)
    return signals, signals_pop


def _save_defended_rmia_artifacts(signal_dir: str,
                                  signals: np.ndarray,
                                  signals_pop: np.ndarray):
    """Persist defended RMIA artifacts for reproducibility."""
    os.makedirs(signal_dir, exist_ok=True)
    np.save(os.path.join(signal_dir, "rmia_signals.npy"), signals)
    np.save(os.path.join(signal_dir, "rmia_signals_pop.npy"), signals_pop)
    for name in (
        "rmia_signals.partial.npy",
        "rmia_signals_pop.partial.npy",
        "rmia_signals.partial_done.npy",
        "rmia_signals_pop.partial_done.npy",
    ):
        try:
            os.remove(os.path.join(signal_dir, name))
        except FileNotFoundError:
            pass


def _safe_defense_name(defense_name: str) -> str:
    return defense_name.replace(" ", "_").replace("=", "").replace(".", "p")


def _artifacts_exist_for_safe_name(safe_name: str,
                                   out_dir: str,
                                   run_rmia_flag: bool,
                                   run_amia_flag: bool,
                                   seed: int | None = None) -> bool:
    """Fast file-existence check for a defense directory that is already safe-named."""
    def_root = os.path.join(out_dir, safe_name)
    if not os.path.isdir(def_root):
        return False

    seed_part = f"seed{seed}" if seed is not None else None
    rmia_root = os.path.join(def_root, "rmia", seed_part) if seed_part else os.path.join(def_root, "rmia")
    amia_root = os.path.join(def_root, "amia", seed_part) if seed_part else os.path.join(def_root, "amia")

    rmia_needed = run_rmia_flag or run_amia_flag
    if rmia_needed:
        sig_dir = os.path.join(rmia_root, "signals")
        sig_path = os.path.join(sig_dir, "rmia_signals.npy")
        pop_path = os.path.join(sig_dir, "rmia_signals_pop.npy")
        if not (os.path.exists(sig_path) and os.path.exists(pop_path)):
            return False

        if run_rmia_flag:
            log_dir = os.path.join(rmia_root, "report")
            result_files = sorted(Path(log_dir, "exp").glob("attack_result_*.npz"))
            if not result_files:
                return False

    if run_amia_flag:
        summary_csv = os.path.join(amia_root, "report", "exp", "attention_summary.csv")
        if not os.path.exists(summary_csv):
            return False

    return True


def _fast_skip_phase6_highrisk_label_kanon(args,
                                           out_dir: str,
                                           run_rmia_flag: bool,
                                           run_amia_flag: bool) -> bool:
    """Return True when all requested Phase-6 label-k-anon runs already exist.

    This preflight is intentionally narrow so we can skip before model loading.
    """
    high_risk_enabled = args.high_risk_guardrail or args.high_risk_dropout
    if not high_risk_enabled:
        return False
    if args.high_risk_fallback != "label_kanon":
        return False
    if args.combine:
        return False
    if args.auto_top_dropout:
        return False
    if any(k > 1 for k in args.kanon_ks):
        return False
    if any(p > 0 for p in args.attn_dropout_ps):
        return False
    if any(p > 0 for p in args.layer_dropout_ps):
        return False

    if not os.path.isdir(out_dir):
        return False
    safe_dirs = [p.name for p in Path(out_dir).iterdir() if p.is_dir()]

    for k in args.label_kanon_ks:
        if k <= 1:
            continue
        for alpha in args.label_kanon_alphas:
            if alpha < 0.0 or alpha >= 1.0:
                continue

            if alpha > 0.0:
                pct = int(round(alpha * 100))
                pattern = re.compile(
                    rf"^highrisk_label_kanon_k{k}(?:_eff\\d+)?_a{pct}$"
                )
            else:
                pattern = re.compile(
                    rf"^highrisk_label_kanon_k{k}(?:_eff\\d+)?$"
                )

            matches = [name for name in safe_dirs if pattern.match(name)]
            if not matches:
                return False

            if not any(
                _artifacts_exist_for_safe_name(
                    safe_name=name,
                    out_dir=out_dir,
                    run_rmia_flag=run_rmia_flag,
                    run_amia_flag=run_amia_flag,
                    seed=args.seed,
                )
                for name in matches
            ):
                return False

    return True


def _has_matching_artifact(safe_dirs: list[str],
                           pattern: re.Pattern,
                           out_dir: str,
                           run_rmia_flag: bool,
                           run_amia_flag: bool,
                           seed: int | None) -> bool:
    """Return whether any matching defense directory has complete artifacts."""
    matches = [name for name in safe_dirs if pattern.match(name)]
    if not matches:
        return False
    return any(
        _artifacts_exist_for_safe_name(
            safe_name=name,
            out_dir=out_dir,
            run_rmia_flag=run_rmia_flag,
            run_amia_flag=run_amia_flag,
            seed=seed,
        )
        for name in matches
    )


def _fast_skip_phase5_label_and_attn(args,
                                     out_dir: str,
                                     run_rmia_flag: bool,
                                     run_amia_flag: bool) -> bool:
    """Return True when all requested Phase-5 label-k-anon + attn-dropout runs exist."""
    high_risk_enabled = args.high_risk_guardrail or args.high_risk_dropout
    if high_risk_enabled:
        return False
    if args.combine:
        return False
    if not args.auto_top_dropout:
        return False
    if any(k > 1 for k in args.kanon_ks):
        return False
    if any(p > 0 for p in args.layer_dropout_ps):
        return False

    if not os.path.isdir(out_dir):
        return False
    safe_dirs = [p.name for p in Path(out_dir).iterdir() if p.is_dir()]

    for k in args.label_kanon_ks:
        if k <= 1:
            continue
        for alpha in args.label_kanon_alphas:
            if alpha < 0.0 or alpha >= 1.0:
                continue

            if alpha > 0.0:
                pct = int(round(alpha * 100))
                pat = re.compile(rf"^label_kanon_k{k}(?:_eff\\d+)?_a{pct}$")
            else:
                pat = re.compile(rf"^label_kanon_k{k}(?:_eff\\d+)?$")

            if not _has_matching_artifact(
                safe_dirs,
                pat,
                out_dir,
                run_rmia_flag,
                run_amia_flag,
                args.seed,
            ):
                return False

    for p in args.attn_dropout_ps:
        if p <= 0:
            continue
        pct = int(round(p * 100))
        # Auto-top-dropout names are like attn_drop_p10_l18,19,20...
        pat = re.compile(rf"^attn_drop_p{pct}(?:_l[0-9,]+)?$")
        if not _has_matching_artifact(
            safe_dirs,
            pat,
            out_dir,
            run_rmia_flag,
            run_amia_flag,
            args.seed,
        ):
            return False

    return True


def _amia_backend(model_name: str):
    """Return AMIA helpers for the selected tabular model family."""
    name = model_name.lower()
    if name == "tabicl":
        import amia_tabicl as backend
    elif name == "tabdpt":
        import amia_tabdpt as backend
    elif name in {"tabpfn", "real-tabpfn"}:
        import amia_tabpfn as backend
    else:
        raise ValueError(
            f"AMIA defense evaluation is not wired for model {model_name!r}. "
            "Supported AMIA backends: tabpfn, real-tabpfn, tabicl, tabdpt."
        )
    return backend


def _amia_summary_col(df, preferred: str) -> str | None:
    """Return a common AMIA row signal column, with legacy-cache fallback."""
    if preferred in df.columns:
        return preferred
    aliases = {
        "row_max": ("icl_max", "dpt_max"),
        "row_ent": ("icl_ent", "dpt_ent"),
    }
    for alt in aliases.get(preferred, ()):
        if alt in df.columns:
            return alt
    return None


def _load_existing_defense_result(defense_name: str, out_dir: str,
                                  run_rmia_flag: bool,
                                  run_amia_flag: bool,
                                  seed: int | None = None) -> dict | None:
    """Load metrics for a completed defense run, or return None if incomplete."""
    safe_name = _safe_defense_name(defense_name)
    def_root = os.path.join(out_dir, safe_name)
    seed_part = f"seed{seed}" if seed is not None else None
    rmia_root = os.path.join(def_root, "rmia", seed_part) if seed_part else os.path.join(def_root, "rmia")
    amia_root = os.path.join(def_root, "amia", seed_part) if seed_part else os.path.join(def_root, "amia")
    result = {
        "defense": defense_name,
        "seed": seed,
        "time_s": 0,
        "rmia_auc": None,
        "amia_row_max_auc": None,
        "amia_row_ent_auc": None,
        "amia_row_max_d": None,
        "accuracy": None,
    }

    rmia_needed = run_rmia_flag or run_amia_flag
    if rmia_needed:
        sig_dir = os.path.join(rmia_root, "signals")
        sig_path = os.path.join(sig_dir, "rmia_signals.npy")
        pop_path = os.path.join(sig_dir, "rmia_signals_pop.npy")
        if not (os.path.exists(sig_path) and os.path.exists(pop_path)):
            return None

        log_dir = os.path.join(rmia_root, "report")
        result_files = sorted(
            Path(log_dir, "exp").glob("attack_result_*.npz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if run_rmia_flag and not result_files:
            return None
        if result_files:
            with np.load(result_files[0]) as npz:
                if "auc" in npz.files and run_rmia_flag:
                    result["rmia_auc"] = float(npz["auc"])

        log_files = sorted(
            Path(log_dir).glob("log_rmia_*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if log_files:
            text = log_files[0].read_text(errors="ignore")
            acc_matches = re.findall(r"Test accuracy\s+([\d.]+)", text)
            auc_matches = re.findall(r"Target Model \d+: AUC\s+([\d.]+)", text)
            if acc_matches:
                result["accuracy"] = float(acc_matches[-1])
            if auc_matches and run_rmia_flag and result["rmia_auc"] is None:
                result["rmia_auc"] = float(auc_matches[-1])

    if run_amia_flag:
        summary_csv = os.path.join(amia_root, "report", "exp", "attention_summary.csv")
        if not os.path.exists(summary_csv):
            return None
        from amia_tabpfn import cohens_d, compute_roc
        import pandas as pd

        df = pd.read_csv(summary_csv)
        score_col = _amia_summary_col(df, "row_max")
        if "member" not in df or score_col is None:
            return None
        mem = df["member"].values.astype(bool)
        _, _, result["amia_row_max_auc"] = compute_roc(df[score_col].values, mem.astype(int))
        result["amia_row_max_d"] = cohens_d(df[score_col].values[mem], df[score_col].values[~mem])
        ent_col = _amia_summary_col(df, "row_ent")
        if ent_col is not None:
            _, _, result["amia_row_ent_auc"] = compute_roc(df[ent_col].values, mem.astype(int))

    return result


def _effective_k(k: int, context_size: int, max_k_ratio: float) -> int:
    """Cap k relative to context size to avoid utility collapse on small audits."""
    if k <= 1:
        return 1
    k_cap = max(2, int(context_size * max_k_ratio))
    return min(int(k), k_cap)


def _auto_dropout_layers(amia_log: str, target_idx: int, mem: np.ndarray,
                         top_k: int, min_auc: float, logger) -> list[int]:
    """Select layers with the strongest member/non-member row_max separation."""
    from amia_tabpfn import compute_roc

    sig_path = os.path.join(amia_log, "signals", f"attn_signals_{target_idx}.npz")
    if not os.path.exists(sig_path):
        raise FileNotFoundError(
            f"Cannot auto-select dropout layers because AMIA signals are missing: {sig_path}. "
            "Run baseline AMIA first, or pass --attn-dropout-layers late/18-23."
        )

    npz = np.load(sig_path)
    row_max_all = npz["row_max_all"]  # (n_pool, n_raw_calls, n_heads)
    if row_max_all.shape[0] != len(mem):
        raise ValueError(
            "AMIA signal rows do not match current membership labels: "
            f"{row_max_all.shape[0]} vs {len(mem)}."
        )

    n_raw = row_max_all.shape[1]
    n_layers = 24 if n_raw % 24 == 0 else None
    if n_layers is None:
        # Fall back to a conservative divisor if a future architecture differs.
        divisors = [d for d in range(2, min(n_raw, 64) + 1) if n_raw % d == 0]
        n_layers = max(divisors) if divisors else n_raw
    calls_per_layer = n_raw // n_layers
    per_layer = row_max_all.reshape(
        row_max_all.shape[0], n_layers, calls_per_layer, row_max_all.shape[2]
    ).mean(axis=(2, 3))

    labels = mem.astype(int)
    aucs = []
    for layer_idx in range(n_layers):
        _, _, auc = compute_roc(per_layer[:, layer_idx], labels)
        aucs.append(float(auc))

    ranked = sorted(range(n_layers), key=lambda i: aucs[i], reverse=True)
    selected = [i for i in ranked if aucs[i] >= min_auc][:top_k]
    if not selected:
        selected = ranked[:top_k]

    selected = sorted(selected)
    logger.info(
        "Auto-selected attention-dropout layers: %s  (top AUCs: %s)",
        ",".join(str(i) for i in selected),
        ", ".join(f"L{i}={aucs[i]:.3f}" for i in ranked[:max(top_k, 6)]),
    )
    return selected


def _num_tabfm_layers(model) -> int | None:
    for arch in getattr(model, "models_", []):
        if hasattr(arch, "transformer_encoder"):
            return len(arch.transformer_encoder.layers)
    return None


def _describe_attn_dropout_layers(layer_spec, n_layers: int | None) -> tuple[object, str, str]:
    """Return (wrapper layer_spec, name suffix, human-readable description)."""
    if isinstance(layer_spec, (list, tuple, set)):
        selected = sorted({int(i) for i in layer_spec})
        suffix = "_l" + ",".join(str(i) for i in selected)
        return selected, suffix, f"auto-selected {len(selected)} layers: {selected}"

    spec = str(layer_spec).strip()
    if spec == "all":
        count = n_layers if n_layers is not None else "all"
        return "1", "", f"all layers ({count})"
    if spec in {"late", "tail"}:
        count = min(6, n_layers) if n_layers is not None else 6
        start = max(0, n_layers - count) if n_layers is not None else None
        if start is None:
            desc = f"late layers (last {count})"
        else:
            desc = f"late layers: last {count} layers L{start}-L{n_layers - 1}"
        return "late", "_llate", desc
    raise ValueError(
        f"Unsupported --attn-dropout-layers {spec!r}. "
        "Use all, late, or --auto-top-dropout."
    )


def _calibrate_high_risk_threshold(summary_csv: str,
                                   mem: np.ndarray,
                                   score_name: str,
                                   threshold: float | None,
                                   logger) -> tuple[float, str]:
    """Calibrate a high-risk query threshold from baseline AMIA summary."""
    import pandas as pd

    if not os.path.exists(summary_csv):
        raise FileNotFoundError(
            f"Cannot calibrate high-risk dropout because baseline AMIA summary is missing: {summary_csv}"
        )
    df = pd.read_csv(summary_csv)
    score_col = _amia_summary_col(df, score_name)
    if score_col is None:
        raise ValueError(f"Baseline AMIA summary is missing score column {score_name!r}.")
    scores = df[score_col].to_numpy(dtype=float)
    finite = np.isfinite(scores)
    if len(scores) != len(mem):
        raise ValueError(
            "Baseline AMIA summary row count does not match membership labels: "
            f"{len(scores)} vs {len(mem)}."
        )
    if threshold is not None:
        return float(threshold), "explicit"

    # Default policy: catch all known members in the baseline calibration set.
    # At runtime true membership is unknown; the guardrail only compares a
    # query's pre-defense AMIA risk score against this fixed threshold.
    cal = scores[mem & finite]
    note = "member_min"
    if len(cal) == 0:
        raise ValueError("No finite member calibration scores for high-risk threshold.")
    value = float(np.min(cal))
    logger.info(
        "High-risk threshold calibrated on %s: %s %.6g",
        score_col,
        note,
        value,
    )
    return value, note


def _write_adaptive_amia_summary(
    baseline_summary: str,
    fallback_summary: str,
    out_summary: str,
    rmia_scores: np.ndarray,
    mem: np.ndarray,
    score_name: str,
    threshold: float,
):
    """Compose AMIA summary for baseline-low-risk + fallback-high-risk policy."""
    from amia_tabpfn import cohens_d, compute_roc
    import pandas as pd

    base_df = pd.read_csv(baseline_summary)
    fallback_df = pd.read_csv(fallback_summary)
    if len(base_df) != len(fallback_df) or len(base_df) != len(mem):
        raise ValueError(
            "Adaptive AMIA summaries must have the same row count as membership labels: "
            f"base={len(base_df)}, fallback={len(fallback_df)}, mem={len(mem)}."
        )
    score_col = _amia_summary_col(base_df, score_name)
    if score_col is None:
        raise ValueError(f"Baseline AMIA summary is missing score column {score_name!r}.")

    risk = base_df[score_col].to_numpy(dtype=float)
    high_risk = np.isfinite(risk) & (risk >= threshold)

    adaptive = base_df.copy()
    for legacy, canonical in [("icl_max", "row_max"), ("dpt_max", "row_max")]:
        if canonical not in adaptive.columns and legacy in adaptive.columns:
            adaptive[canonical] = adaptive[legacy]
        if canonical not in fallback_df.columns and legacy in fallback_df.columns:
            fallback_df = fallback_df.copy()
            fallback_df[canonical] = fallback_df[legacy]
    for legacy, canonical in [("icl_ent", "row_ent"), ("dpt_ent", "row_ent")]:
        if canonical not in adaptive.columns and legacy in adaptive.columns:
            adaptive[canonical] = adaptive[legacy]
        if canonical not in fallback_df.columns and legacy in fallback_df.columns:
            fallback_df = fallback_df.copy()
            fallback_df[canonical] = fallback_df[legacy]

    for col in ["row_max", "row_ent", "col_max", "col_ent"]:
        if col in adaptive.columns and col in fallback_df.columns:
            adaptive.loc[high_risk, col] = fallback_df.loc[high_risk, col].to_numpy()
    adaptive = adaptive.drop(columns=[c for c in ["icl_max", "icl_ent", "dpt_max", "dpt_ent"] if c in adaptive.columns])
    adaptive["rmia_score"] = rmia_scores
    # Pre-defense score used only to decide whether the fallback is applied.
    # The AMIA columns above are final/adaptive values; for high-risk rows they
    # have already been replaced by the fallback defense summary.
    adaptive["probe_risk_score"] = risk
    adaptive["risk_threshold"] = threshold
    adaptive["fallback_applied"] = high_risk
    adaptive.to_csv(out_summary, index=False)

    result = {}
    for col in ["row_max", "row_ent"]:
        actual_col = _amia_summary_col(adaptive, col)
        if actual_col is not None:
            vals = adaptive[actual_col].to_numpy(dtype=float)
            _, _, result[f"amia_{col}_auc"] = compute_roc(vals, mem.astype(int))
            result[f"amia_{col}_d"] = cohens_d(vals[mem], vals[~mem])
    result["fallback_rate"] = float(high_risk.mean())
    result["fallback_count"] = int(high_risk.sum())
    result["fallback_mask"] = high_risk
    return result


def _write_adaptive_amia_signals(
    baseline_cache: str,
    fallback_cache: str,
    out_cache: str,
    high_risk: np.ndarray,
) -> None:
    """Compose adaptive AMIA raw signal cache to match adaptive summary rows."""
    if not (os.path.exists(baseline_cache) and os.path.exists(fallback_cache)):
        return

    with np.load(baseline_cache) as base_npz, np.load(fallback_cache) as fallback_npz:
        composed = {}
        for key in base_npz.files:
            arr = base_npz[key]
            if key in fallback_npz.files:
                fallback_arr = fallback_npz[key]
                if (
                    hasattr(arr, "shape")
                    and hasattr(fallback_arr, "shape")
                    and arr.shape == fallback_arr.shape
                    and arr.shape[0] == len(high_risk)
                ):
                    arr = arr.copy()
                    arr[high_risk] = fallback_arr[high_risk]
            composed[key] = arr

    os.makedirs(os.path.dirname(out_cache), exist_ok=True)
    np.savez_compressed(out_cache, **composed)


# ── baseline loader (pure disk reads, no inference) ──────────────────────────

def _load_baseline(amia_log: str, rmia_log: str, target_idx: int,
                   mem: np.ndarray) -> dict:
    """Load undefended baseline AUCs and accuracy from attack script artifacts."""
    from amia_tabpfn import compute_roc
    import pandas as pd
    import re

    rmia_auc = None
    rmia_path = os.path.join(amia_log, "signals", f"rmia_scores_{target_idx}.npy")
    if os.path.exists(rmia_path):
        rmia_scores = np.load(rmia_path)
        _, _, rmia_auc = compute_roc(rmia_scores, mem.astype(int))
    else:
        rmia_result_paths = [
            os.path.join(rmia_log, "report", "attack_result_average.npz"),
            os.path.join(rmia_log, "report", "exp", f"attack_result_{target_idx}.npz"),
        ]
        for rmia_result_path in rmia_result_paths:
            if not os.path.exists(rmia_result_path):
                continue
            with np.load(rmia_result_path) as result:
                if "auc" in result.files:
                    rmia_auc = float(result["auc"])
                    break

    amia_auc = None
    summary_csv = os.path.join(amia_log, "report", "exp", "attention_summary.csv")
    if os.path.exists(summary_csv):
        df = pd.read_csv(summary_csv)
        score_col = _amia_summary_col(df, "row_max")
        if score_col is not None and "member" in df.columns:
            _, _, amia_auc = compute_roc(df[score_col].values, df["member"].values)

    # Load test accuracy for target_idx from the RMIA time-analysis log.
    # Each model logs two consecutive lines: Train accuracy … / Test accuracy …
    accuracy = None
    log_path = os.path.join(rmia_log, "report", "log_time_analysis.log")
    if os.path.exists(log_path):
        with open(log_path) as fh:
            lines = fh.readlines()
        test_lines = [l for l in lines if "Test accuracy" in l]
        if target_idx < len(test_lines):
            m = re.search(r"Test accuracy\s+([\d.]+)", test_lines[target_idx])
            if m:
                accuracy = float(m.group(1))

    return {
        "defense": "no_defense",
        "time_s": 0,
        "rmia_auc": rmia_auc,
        "amia_row_max_auc": amia_auc,
        "amia_row_ent_auc": None,
        "amia_row_max_d": None,
        "accuracy": accuracy,
    }


# ── per-defense runner ────────────────────────────────────────────────────────

def _run_one_defense(
    defense_name,
    wrap_fn,
    models_list,
    target_model,
    target_idx,
    mem,
    X_pool, y_pool,
    X_pop,  y_pop,
    X_test, y_test,   # non-member pool samples — same test set as RMIA baseline
    rmia_signals,
    rmia_signals_pop,
    memberships,
    num_ref,
    n_context,
    batch_size,
    out_dir,
    amia_log,
    dataset_name,
    model_name_str,
    run_rmia_flag,
    run_amia_flag,
    logger,
    seed: int | None = None,
    skip_existing: bool = False,
) -> dict:
    """Evaluate one defense configuration for RMIA and AMIA."""
    from attacks import run_rmia, tune_offline_a
    from audit import get_audit_results
    from run_defenses.tabfm_introspection import infer_num_thinking_rows
    from util import setup_log
    run_amia_pipeline = _amia_backend(model_name_str).run_amia_pipeline
    tabpfn_thinking_rows = (
        infer_num_thinking_rows(target_model, default=0)
        if model_name_str.lower() in {"tabpfn", "real-tabpfn"}
        else 0
    )

    safe_name = _safe_defense_name(defense_name)
    def_root = os.path.join(out_dir, safe_name)
    seed_part = f"seed{seed}" if seed is not None else None
    rmia_root = os.path.join(def_root, "rmia", seed_part) if seed_part else os.path.join(def_root, "rmia")
    amia_root = os.path.join(def_root, "amia", seed_part) if seed_part else os.path.join(def_root, "amia")
    t0 = time.time()
    result = {"defense": defense_name, "seed": seed}
    wrapped_target = wrap_fn(target_model, target_idx) if wrap_fn else target_model
    result["accuracy"] = _accuracy(wrapped_target, X_test, y_test, batch_size)

    # ── RMIA ──────────────────────────────────────────────────────────────────
    need_rmia = run_rmia_flag or run_amia_flag
    if need_rmia:
        rmia_log_dir = os.path.join(rmia_root, "report")
        rmia_exp_dir = os.path.join(rmia_log_dir, "exp")
        rmia_sig_dir = os.path.join(rmia_root, "signals")
        os.makedirs(rmia_log_dir, exist_ok=True)
        os.makedirs(rmia_exp_dir, exist_ok=True)
        rmia_logger = setup_log(rmia_log_dir, f"rmia_{safe_name}", True)

        rmia_logger.info("Test accuracy %.4f (on non-member pool, same as RMIA baseline)",
                         result["accuracy"])

        logger.info("[%s] Recomputing RMIA signals for all models…", defense_name)
        signals, signals_pop = _compute_defended_signals(
            wrap_fn, models_list,
            X_pool, y_pool, X_pop, y_pop,
            rmia_signals, rmia_signals_pop,
            batch_size, logger, defense_name,
            checkpoint_dir=rmia_sig_dir,
        )

        best_a, _, _ = tune_offline_a(
            target_model_idx=target_idx,
            all_signals=signals,
            population_signals=signals_pop,
            all_memberships=memberships.T,
            logger=rmia_logger,
        )
        rmia_logger.info("Running RMIA (offline) on target model %d with offline_a=%.1f",
                         target_idx, best_a)

        rmia = run_rmia(
            target_model_idx=target_idx,
            all_signals=signals,
            population_signals=signals_pop,
            all_memberships=memberships.T,
            num_reference_models=num_ref,
            offline_a=best_a,
        )

        _save_defended_rmia_artifacts(
            rmia_sig_dir,
            signals,
            signals_pop,
        )
        rmia_logger.info("Saved defended RMIA artifacts to %s", rmia_sig_dir)

        attack_result = get_audit_results(
            rmia_exp_dir,
            target_idx,
            rmia,
            mem.astype(int),
            rmia_logger,
        )
        rmia_auc = attack_result["auc"]
        if run_rmia_flag:
            result["rmia_auc"] = rmia_auc
            logger.info("[%s] RMIA AUC=%.4f  Accuracy=%.4f", defense_name, rmia_auc, result["accuracy"])
            rmia_logger.info("RMIA AUC=%.4f  Accuracy=%.4f", rmia_auc, result["accuracy"])
        else:
            result["rmia_auc"] = None
            logger.info("[%s] RMIA recomputed for AMIA context; Accuracy=%.4f", defense_name, result["accuracy"])
            rmia_logger.info("RMIA recomputed for AMIA context; Accuracy=%.4f", result["accuracy"])
    else:
        rmia     = None
        result["rmia_auc"] = None

    # ── AMIA ──────────────────────────────────────────────────────────────────
    if run_amia_flag:
        def_sig_dir = os.path.join(amia_root, "signals")
        def_exp_dir = os.path.join(amia_root, "report", "exp")
        def_log_dir = os.path.join(amia_root, "report")
        os.makedirs(def_sig_dir, exist_ok=True)
        os.makedirs(def_exp_dir, exist_ok=True)
        def_logger = setup_log(def_log_dir, f"amia_{safe_name}", True)

        if rmia is None:
            raise RuntimeError("RMIA scores are required for defended AMIA evaluation.")

        adaptive_meta = getattr(wrap_fn, "_adaptive_amia", None)
        if adaptive_meta is not None:
            fallback_name = adaptive_meta["fallback_name"]
            fallback_wrap = adaptive_meta["fallback_wrap"]
            fallback_safe = _safe_defense_name(fallback_name)
            fallback_root = os.path.join(out_dir, fallback_safe)
            fallback_amia_root = (
                os.path.join(fallback_root, "amia", seed_part)
                if seed_part else os.path.join(fallback_root, "amia")
            )
            fallback_sig_dir = os.path.join(fallback_amia_root, "signals")
            fallback_exp_dir = os.path.join(fallback_amia_root, "report", "exp")
            fallback_log_dir = os.path.join(fallback_amia_root, "report")
            os.makedirs(fallback_sig_dir, exist_ok=True)
            os.makedirs(fallback_exp_dir, exist_ok=True)
            os.makedirs(fallback_log_dir, exist_ok=True)
            fallback_summary = os.path.join(fallback_exp_dir, "attention_summary.csv")

            if (not skip_existing) or (not os.path.exists(fallback_summary)):
                fallback_logger = setup_log(fallback_log_dir, f"amia_{fallback_safe}", True)
                logger.info(
                    "[%s] Running fallback AMIA for %s…",
                    defense_name,
                    fallback_name,
                )
                amia_kwargs = {"model_idx": target_idx, "mode": "train"}
                if model_name_str.lower() in {"tabpfn", "real-tabpfn"}:
                    amia_kwargs["n_thinking"] = tabpfn_thinking_rows
                run_amia_pipeline(
                    fallback_wrap(target_model, target_idx),
                    X_pool,
                    mem,
                    rmia,
                    n_context,
                    batch_size,
                    fallback_logger,
                    fallback_sig_dir,
                    fallback_exp_dir,
                    dataset_name,
                    model_name_str,
                    **amia_kwargs,
                )

            baseline_summary = os.path.join(
                amia_log,
                "report",
                "exp",
                "attention_summary.csv",
            )
            out_summary = os.path.join(def_exp_dir, "attention_summary.csv")
            scalars = _write_adaptive_amia_summary(
                baseline_summary=baseline_summary,
                fallback_summary=fallback_summary,
                out_summary=out_summary,
                rmia_scores=rmia,
                mem=mem,
                score_name=adaptive_meta["score_name"],
                threshold=adaptive_meta["threshold"],
            )
            _write_adaptive_amia_signals(
                baseline_cache=os.path.join(amia_log, "signals", f"attn_signals_{target_idx}.npz"),
                fallback_cache=os.path.join(fallback_sig_dir, f"attn_signals_{target_idx}.npz"),
                out_cache=os.path.join(def_sig_dir, f"attn_signals_{target_idx}.npz"),
                high_risk=scalars["fallback_mask"],
            )
            def_logger.info(
                "Adaptive high-risk AMIA: fallback=%s threshold=%.6g applied=%d/%d (%.4f)",
                fallback_name,
                adaptive_meta["threshold"],
                scalars["fallback_count"],
                len(mem),
                scalars["fallback_rate"],
            )
            logger.info("[%s] Generating AMIA plots from adaptive signals…", defense_name)
            plot_kwargs = {"model_idx": target_idx, "mode": "load"}
            if model_name_str.lower() in {"tabpfn", "real-tabpfn"}:
                plot_kwargs["n_thinking"] = tabpfn_thinking_rows
            run_amia_pipeline(
                wrapped_target, X_pool, mem, rmia,
                n_context, batch_size, def_logger,
                def_sig_dir, def_exp_dir,
                dataset_name, model_name_str,
                **plot_kwargs,
            )
        else:
            amia_mode = "load" if skip_existing else "train"

            logger.info("[%s] Running AMIA pipeline…", defense_name)
            amia_kwargs = {"model_idx": target_idx, "mode": amia_mode}
            if model_name_str.lower() in {"tabpfn", "real-tabpfn"}:
                amia_kwargs["n_thinking"] = tabpfn_thinking_rows
            scalars = run_amia_pipeline(
                wrapped_target, X_pool, mem, rmia,
                n_context, batch_size, def_logger,
                def_sig_dir, def_exp_dir,
                dataset_name, model_name_str,
                **amia_kwargs,
            )
        result["amia_row_max_auc"] = scalars.get("row_max_auc")
        result["amia_row_ent_auc"] = scalars.get("row_ent_auc")
        result["amia_row_max_d"]   = scalars.get("row_max_d")
        if result["amia_row_max_auc"] is None:
            result["amia_row_max_auc"] = scalars.get("amia_row_max_auc")
            result["amia_row_ent_auc"] = scalars.get("amia_row_ent_auc")
            result["amia_row_max_d"] = scalars.get("amia_row_max_d")
        logger.info(
            "[%s] AMIA row_max AUC=%.4f  Cohen's d=%.4f  Accuracy=%.4f",
            defense_name,
            result["amia_row_max_auc"],
            result["amia_row_max_d"],
            result["accuracy"],
        )
        def_logger.info(
            "AMIA row_max AUC=%.4f  Cohen's d=%.4f  Accuracy=%.4f",
            result["amia_row_max_auc"],
            result["amia_row_max_d"],
            result["accuracy"],
        )
    else:
        result["amia_row_max_auc"] = None
        result["amia_row_ent_auc"] = None
        result["amia_row_max_d"]   = None

    result["time_s"] = time.time() - t0
    _cleanup_runtime(logger, defense_name)
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    from configs import ensure_dataset_ready

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate k-anonymity, attention dropout, and layer dropout defenses "
            "against RMIA and AMIA — individually and (optionally) combined."
        )
    )
    parser.add_argument("--dataset",          type=str,   default="locations")
    parser.add_argument("--model",            type=str,   default="tabpfn")
    parser.add_argument("--model-idx",        type=int,   default=0)
    parser.add_argument("--seed",             type=int,   default=1,
                        help="Seed number for the trial under logs/<dataset>/<model>/seed<seed>/. Default: 1.")
    parser.add_argument("--seeds",            type=str,   default=None,
                        help="Comma-separated seeded trials to evaluate, e.g. 1,2,3,4,5.")
    parser.add_argument("--gpu",              type=str,   default=None,
                        help="CUDA device index to expose, e.g. '0'. Omit for CPU.")
    parser.add_argument("--gpus",             type=str,   default=None,
                        help="Compatibility alias for --gpu. If multiple IDs are provided, only the first is used.")
    parser.add_argument("--batch-size",       type=int,   default=200)
    parser.add_argument("--kanon-ks",         type=int,   nargs="+", default=[2, 5, 10],
                        help="k values for k-anonymity defense.")
    parser.add_argument(
        "--kanon-max-k-ratio",
        type=float,
        default=1.0 / 16.0,
        help=(
            "Cap effective k as floor(context_size * ratio) to avoid utility collapse "
            "on small contexts. Set 1.0 to disable capping."
        ),
    )
    parser.add_argument("--label-kanon-ks",   type=int,   nargs="+", default=[2, 5],
                        help="k values for label-aware k-anonymity defense.")
    parser.add_argument("--label-kanon-alphas", type=float, nargs="+", default=[0.0],
                        help=(
                            "Optional original-key retention for soft label-aware k-anon. "
                            "Default 0.0 means pure label k-anon centroiding; "
                            "0.8/0.9 are softer utility-preserving variants."
                        ))
    parser.add_argument("--knn-ks",             type=int,   nargs="+", default=[],
                        help="k values for kNN key smoothing; empty disables this defense.")
    parser.add_argument("--knn-alphas",         type=float, nargs="+", default=[0.7],
                        help=(
                            "Original-key retention for kNN smoothing in [0, 1]. "
                            "alpha=1.0 is an explicit no-smoothing identity setting."
                        ))
    parser.add_argument("--attn-dropout-ps",  type=float, nargs="+", default=[0.1, 0.3, 0.5],
                        help="Dropout probabilities for attention weight dropout.")
    parser.add_argument("--attn-dropout-layers", type=str, default="all",
                        help=(
                            "Attention-dropout layer preset: all or late. "
                            "Use --auto-top-dropout for AMIA-peak layers."
                        ))
    parser.add_argument("--auto-top-dropout", action="store_true",
                        help=(
                            "Apply attention dropout "
                            "only to the layers with the strongest baseline AMIA row_max AUC."
                        ))
    parser.add_argument("--high-risk-guardrail", action="store_true",
                        help=(
                            "Add adaptive guardrail rows: probe query AMIA row_max, then rerun "
                            "only high-risk queries with the selected fallback defense."
                        ))
    parser.add_argument("--high-risk-dropout", action="store_true",
                        help=(
                            "Deprecated alias for --high-risk-guardrail. The fallback can be "
                            "dropout, knn, or label_kanon."
                        ))
    parser.add_argument("--high-risk-fallback", type=str, default="dropout",
                        choices=["dropout", "knn", "label_kanon"],
                        help="Fallback defense used only for high-risk queries.")
    parser.add_argument("--high-risk-score", type=str, default="row_max",
                        choices=["row_max", "row_ent", "rmia_score"],
                        help="Baseline AMIA summary column used to flag high-risk queries.")
    parser.add_argument("--high-risk-threshold", type=float, default=None,
                        help="Explicit high-risk threshold. Default is the minimum known-member AMIA risk in the calibration set.")
    parser.add_argument("--high-risk-probe-batch-size", type=int, default=None,
                        help="Probe batch size for live high-risk dropout; defaults to --batch-size.")
    parser.add_argument("--auto-dropout-top-layers", type=int, default=6,
                        help="When --auto-top-dropout is set, select this many peak-AMIA layers.")
    parser.add_argument("--auto-dropout-min-auc", type=float, default=0.0,
                        help="When auto-selecting layers, only keep layers with row_max AUC at least this value.")
    parser.add_argument("--layer-dropout-ps", type=float, nargs="+", default=[0.1, 0.3, 0.5],
                        help="Dropout probabilities for hidden-state (layer) dropout.")
    parser.add_argument("--attacks",          type=str,   nargs="+",
                        default=["rmia", "amia"], choices=["rmia", "amia"],
                        help="Which attacks to evaluate against each defense.")
    parser.add_argument("--combine",          action="store_true",
                        help=(
                            "Add a grid of stacked defense combinations."
                        ))
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip defense rows if the requested RMIA/AMIA artifacts already exist.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Deprecated compatibility flag; recomputing existing defense rows is now the default.",
    )
    args = parser.parse_args()
    if args.gpu is not None and args.gpus is not None:
        raise ValueError("Use only one of --gpu or --gpus.")
    if args.gpus is not None:
        gpu_items = [item.strip() for item in args.gpus.split(",") if item.strip()]
        if not gpu_items:
            raise ValueError("--gpus must contain at least one GPU id.")
        args.gpu = gpu_items[0]

    skip_existing = bool(args.skip_existing and not args.rerun_existing)

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
            cmd = [
                sys.executable,
                __file__,
                *filtered_argv,
                "--seed",
                str(seed),
            ]
            print(f"[SEED] Running defense evaluation seed={seed}")
            code = subprocess.run(cmd, check=False).returncode
            if code != 0:
                failures.append(seed)
                print(f"[FAIL] seed={seed} exit_code={code}")
                break

        if failures:
            raise SystemExit(1)
        print(f"[SEED] Completed defense evaluation for seeds: {seeds}")
        return
    high_risk_enabled = args.high_risk_guardrail or args.high_risk_dropout

    _DATASET_BATCH_SIZE = {"locations": 64}
    if args.dataset in _DATASET_BATCH_SIZE and args.batch_size == 200:
        args.batch_size = _DATASET_BATCH_SIZE[args.dataset]

    ensure_dataset_ready(
        dataset_name=args.dataset,
        model_name=args.model,
        algorithm="RMIA",
        skip_if_exists=True,
    )

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    run_rmia_flag = "rmia" in args.attacks
    run_amia_flag = "amia" in args.attacks

    # ── paths ─────────────────────────────────────────────────────────────────
    base_log = os.path.join("ml_privacy_meter", "logs",
                            args.dataset, args.model.lower())
    run_log = os.path.join(base_log, f"seed{args.seed}") if args.seed is not None else base_log
    rmia_log = os.path.join(run_log, "rmia") if args.seed is not None else os.path.join(base_log, "rmia")
    sig_dir  = os.path.join(rmia_log, "signals")
    amia_log = os.path.join(run_log, "amia") if args.seed is not None else os.path.join(base_log, "amia")
    out_dir  = os.path.join(base_log, "defense")

    if skip_existing and _fast_skip_phase6_highrisk_label_kanon(
        args,
        out_dir,
        run_rmia_flag,
        run_amia_flag,
    ):
        print(
            "[SKIP] All requested high-risk label k-anon defense artifacts already "
            "exist; exiting before loading models."
        )
        return

    if skip_existing and _fast_skip_phase5_label_and_attn(
        args,
        out_dir,
        run_rmia_flag,
        run_amia_flag,
    ):
        print(
            "[SKIP] All requested Phase-5 label-k-anon/attention-dropout defense artifacts "
            "already exist; exiting before loading models."
        )
        return

    os.makedirs(out_dir, exist_ok=True)

    config_path = f"ml_privacy_meter/configs/{args.dataset}_{args.model}.yaml"
    with open(config_path) as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    if args.seed is not None:
        configs.setdefault("run", {})["random_seed"] = args.seed
        configs.setdefault("train", {})["random_state"] = args.seed
    if args.gpu is not None:
        configs.setdefault("train", {})["device"] = "cuda:0"
        configs.setdefault("audit", {})["device"] = "cuda:0"
    elif args.model.lower() == "tabdpt":
        configs.setdefault("train", {})["device"] = "cpu"
        configs.setdefault("audit", {})["device"] = "cpu"

    # ── data ──────────────────────────────────────────────────────────────────
    amia_backend = _amia_backend(args.model)
    load_dataset = amia_backend.load_dataset
    prepare_tabular_arrays = amia_backend.prepare_tabular_arrays

    df   = load_dataset(args.dataset, configs["data"]["data_dir"])
    X, y = prepare_tabular_arrays(df)
    split_path = None
    if args.seed is not None:
        split_path = os.path.join(rmia_log, "splits", "dataset_permutation.npy")
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"Seeded RMIA split not found: {split_path}\n"
                f"Run RMIA first with --seed {args.seed}."
            )
        order = np.load(split_path)
        X = X[order]
        y = y[order]

    rmia_signals     = np.load(os.path.join(sig_dir, "rmia_signals.npy"))
    rmia_signals_pop = np.load(os.path.join(sig_dir, "rmia_signals_pop.npy"))
    n_pool, _        = rmia_signals.shape
    n_pop            = rmia_signals_pop.shape[0]

    # ── models ────────────────────────────────────────────────────────────────
    from models.utils import load_models
    from util import setup_log

    eval_log_dir = out_dir
    logger = setup_log(eval_log_dir, "eval_defenses", True)
    if split_path is not None:
        logger.info("Using seeded RMIA split permutation: %s", split_path)
        logger.info("Using seeded RMIA artifacts from: %s", rmia_log)
    models_list, memberships = load_models(rmia_log, None, None, configs, logger)
    if models_list is None:
        raise RuntimeError("Failed to load models.")
    if args.model.lower() == "tabdpt":
        tabdpt_device = "cuda:0" if args.gpu is not None else "cpu"
        for model in models_list:
            _move_tabdpt_model(model, tabdpt_device, logger)

    X_pool, y_pool, X_pop, y_pop, auditing_membership = _build_eval_arrays(
        X, y, configs, memberships, n_pool, n_pop, logger
    )

    target_model = models_list[args.model_idx]
    mem          = auditing_membership[args.model_idx].astype(bool)
    num_ref      = configs["audit"]["num_ref_models"]
    n_context    = int(mem.sum())   # TabPFN: training set IS the context
    n_transformer_layers = _num_tabfm_layers(target_model)
    attn_dropout_layers = args.attn_dropout_layers
    if attn_dropout_layers == "auto":
        raise ValueError("Use --auto-top-dropout instead of --attn-dropout-layers auto.")
    if args.auto_top_dropout:
        attn_dropout_layers = _auto_dropout_layers(
            amia_log,
            args.model_idx,
            mem,
            args.auto_dropout_top_layers,
            args.auto_dropout_min_auc,
            logger,
        )
    attn_dropout_layers, attn_dropout_layer_suffix, attn_dropout_layer_desc = (
        _describe_attn_dropout_layers(attn_dropout_layers, n_transformer_layers)
    )
    logger.info(
        "Attention-dropout layer preset %r -> %s",
        args.attn_dropout_layers if not args.auto_top_dropout else "auto-top",
        attn_dropout_layer_desc,
    )

    # ── defense configurations ────────────────────────────────────────────────
    from run_defenses.tabfm_kanon import KAnonTabFMWrapper
    from run_defenses.tabfm_attn_dropout import (
        AttnDropoutWrapper,
        HighRiskQueryDropoutWrapper,
        HighRiskQueryFallbackWrapper,
        HighRiskQueryKNNWrapper,
        KNNKeySmoothWrapper,
        LayerDropoutWrapper,
    )
    from run_defenses.tabfm_introspection import infer_num_thinking_rows

    thinking_rows = (
        infer_num_thinking_rows(target_model, default=0)
        if args.model.lower() in {"tabpfn", "real-tabpfn"}
        else 0
    )
    logger.info("Detected thinking_rows=%d for model=%s", thinking_rows, args.model.lower())
    high_risk_threshold = None
    high_risk_note = None
    if high_risk_enabled:
        high_risk_threshold, high_risk_note = _calibrate_high_risk_threshold(
            os.path.join(amia_log, "report", "exp", "attention_summary.csv"),
            mem,
            args.high_risk_score,
            args.high_risk_threshold,
            logger,
        )

    defense_configs = []
    kanon_attention_mode = (
        args.model.lower() if args.model.lower() in {"tabicl", "tabdpt"} else "tabpfn"
    )

    def _model_context_size(idx: int) -> int:
        return int(auditing_membership[idx].astype(bool).sum())

    def _plain_kanon_component(k: int):
        k_eff = _effective_k(k, n_context, args.kanon_max_k_ratio)
        name = f"kanon_k{k}" if k_eff == k else f"kanon_k{k}_eff{k_eff}"

        def _wrap(m, idx, _k=k_eff, _tr=thinking_rows):
            return KAnonTabFMWrapper(
                m,
                k=_k,
                thinking_rows=_tr,
                context_size=_model_context_size(idx),
                attention_mode=kanon_attention_mode,
            )

        return name, _wrap

    def _label_kanon_component(
        k: int,
        alpha: float = 0.0,
    ):
        k_eff = _effective_k(k, n_context, args.kanon_max_k_ratio)
        base = f"label_kanon_k{k}" if k_eff == k else f"label_kanon_k{k}_eff{k_eff}"
        parts = [base]
        if alpha > 0.0:
            parts.append(f"a{int(round(alpha * 100))}")
        name = "_".join(parts)

        def _wrap(
            m,
            idx,
            _k=k_eff,
            _tr=thinking_rows,
            _alpha=alpha,
        ):
            membership_mask = auditing_membership[idx].astype(bool)
            return KAnonTabFMWrapper(
                m,
                k=_k,
                thinking_rows=_tr,
                context_labels=y_pool[membership_mask],
                retain_alpha=_alpha,
                context_size=_model_context_size(idx),
                attention_mode=kanon_attention_mode,
            )

        return name, _wrap

    def _knn_component(knn_k: int, alpha: float):
        pct = int(round(alpha * 100))
        name = f"knn_k{knn_k}_a{pct}"

        def _wrap(m, _idx, _k=knn_k, _alpha=alpha, _tr=thinking_rows):
            return KNNKeySmoothWrapper(
                m,
                knn_k=_k,
                alpha=_alpha,
                thinking_rows=_tr,
                attention_mode=kanon_attention_mode,
            )

        return name, _wrap

    def _attn_dropout_component(p: float):
        name = f"attn_drop_p{round(p * 100)}{attn_dropout_layer_suffix}"
        return name, lambda m, _idx, _p=p: AttnDropoutWrapper(
            m,
            p=_p,
            layer_indices=attn_dropout_layers,
            attention_mode=kanon_attention_mode,
        )

    def _high_risk_dropout_component(p: float):
        if high_risk_threshold is None or high_risk_note is None:
            raise RuntimeError("High-risk threshold was not calibrated.")
        fallback_name = f"attn_drop_p{round(p * 100)}{attn_dropout_layer_suffix}"
        name = f"highrisk_{fallback_name}"

        def _fallback_wrap(m, _idx, _p=p):
            return AttnDropoutWrapper(
                m,
                p=_p,
                layer_indices=attn_dropout_layers,
                attention_mode=kanon_attention_mode,
            )

        def _wrap(m, _idx, _p=p):
            return HighRiskQueryDropoutWrapper(
                m,
                n_context=_model_context_size(_idx),
                threshold=high_risk_threshold,
                p=_p,
                layer_indices=attn_dropout_layers,
                probe_batch_size=args.high_risk_probe_batch_size or args.batch_size,
                capture_backend=args.model.lower(),
                thinking_rows=thinking_rows,
            )

        _wrap._adaptive_amia = {
            "fallback_name": fallback_name,
            "fallback_wrap": _fallback_wrap,
            "score_name": args.high_risk_score,
            "threshold": high_risk_threshold,
        }
        return name, _wrap

    def _high_risk_knn_component(knn_k: int, alpha: float):
        if high_risk_threshold is None or high_risk_note is None:
            raise RuntimeError("High-risk threshold was not calibrated.")
        pct = int(round(alpha * 100))
        fallback_name = f"knn_k{knn_k}_a{pct}"
        name = f"highrisk_{fallback_name}"

        def _fallback_wrap(m, _idx, _k=knn_k, _alpha=alpha):
            return KNNKeySmoothWrapper(
                m,
                knn_k=_k,
                alpha=_alpha,
                thinking_rows=thinking_rows,
                attention_mode=kanon_attention_mode,
            )

        def _wrap(m, _idx, _k=knn_k, _alpha=alpha):
            return HighRiskQueryKNNWrapper(
                m,
                n_context=_model_context_size(_idx),
                threshold=high_risk_threshold,
                knn_k=_k,
                alpha=_alpha,
                thinking_rows=thinking_rows,
                probe_batch_size=args.high_risk_probe_batch_size or args.batch_size,
                capture_backend=args.model.lower(),
                attention_mode=kanon_attention_mode,
            )

        _wrap._adaptive_amia = {
            "fallback_name": fallback_name,
            "fallback_wrap": _fallback_wrap,
            "score_name": args.high_risk_score,
            "threshold": high_risk_threshold,
        }
        return name, _wrap

    def _high_risk_label_kanon_component(
        k: int,
        alpha: float = 0.0,
    ):
        if high_risk_threshold is None or high_risk_note is None:
            raise RuntimeError("High-risk threshold was not calibrated.")
        fallback_name, fallback_wrap = _label_kanon_component(k, alpha)
        name = f"highrisk_{fallback_name}"

        def _wrap(m, idx):
            return HighRiskQueryFallbackWrapper(
                model=m,
                fallback_model=fallback_wrap(m, idx),
                n_context=_model_context_size(idx),
                threshold=high_risk_threshold,
                probe_batch_size=args.high_risk_probe_batch_size or args.batch_size,
                capture_backend=args.model.lower(),
                thinking_rows=thinking_rows,
            )

        _wrap._adaptive_amia = {
            "fallback_name": fallback_name,
            "fallback_wrap": fallback_wrap,
            "score_name": args.high_risk_score,
            "threshold": high_risk_threshold,
        }
        return name, _wrap

    def _layer_dropout_component(p: float):
        name = f"layer_drop_p{round(p * 100)}"
        return name, lambda m, _idx, _p=p: LayerDropoutWrapper(m, p=_p)

    def _compose_components(components):
        name = "+".join(c[0] for c in components)

        def _wrap(m, idx, _components=components):
            wrapped = m
            for _name, fn in _components:
                wrapped = fn(wrapped, idx)
            return wrapped

        return name, _wrap

    for k in args.kanon_ks:
        if k <= 1:
            continue
        defense_configs.append(_plain_kanon_component(k))
    for k in args.label_kanon_ks:
        if k <= 1:
            continue
        for alpha in args.label_kanon_alphas:
            if alpha < 0.0 or alpha >= 1.0:
                continue
            if high_risk_enabled and args.high_risk_fallback == "label_kanon":
                defense_configs.append(
                    _high_risk_label_kanon_component(k, alpha)
                )
            else:
                defense_configs.append(_label_kanon_component(k, alpha))
    for knn_k in args.knn_ks:
        if knn_k <= 1:
            continue
        for alpha in args.knn_alphas:
            if alpha < 0.0 or alpha > 1.0:
                continue
            defense_configs.append(_knn_component(knn_k, alpha))
            if high_risk_enabled and args.high_risk_fallback == "knn" and alpha < 1.0:
                defense_configs.append(_high_risk_knn_component(knn_k, alpha))
    for p in args.attn_dropout_ps:
        if p <= 0:
            continue
        defense_configs.append(_attn_dropout_component(p))
        if high_risk_enabled and args.high_risk_fallback == "dropout":
            defense_configs.append(_high_risk_dropout_component(p))
    for p in args.layer_dropout_ps:
        if p <= 0:
            continue
        defense_configs.append(_layer_dropout_component(p))

    if args.combine:
        kanon_choices = [None]
        kanon_choices += [_plain_kanon_component(k) for k in args.kanon_ks if k > 1]
        kanon_choices += [
            _label_kanon_component(k, alpha)
            for k in args.label_kanon_ks
            if k > 1
            for alpha in args.label_kanon_alphas
            if 0.0 <= alpha < 1.0
        ]
        knn_choices = [None] + [
            _knn_component(knn_k, alpha)
            for knn_k in args.knn_ks
            if knn_k > 1
            for alpha in args.knn_alphas
            if 0.0 <= alpha < 1.0
        ]
        attn_choices = [None] + [
            _attn_dropout_component(p) for p in args.attn_dropout_ps
            if p > 0.0
        ]
        layer_choices = [None] + [
            _layer_dropout_component(p) for p in args.layer_dropout_ps
            if p > 0.0
        ]

        existing_names = {name for name, _ in defense_configs}
        for kanon_comp in kanon_choices:
            for knn_comp in knn_choices:
                for attn_comp in attn_choices:
                    for layer_comp in layer_choices:
                        components = [
                            c for c in (
                                kanon_comp,
                                knn_comp,
                                attn_comp,
                                layer_comp,
                            )
                            if c is not None
                        ]
                        if len(components) < 2:
                            continue
                        name, wrap = _compose_components(components)
                        if name in existing_names:
                            continue
                        defense_configs.append((name, wrap))
                        existing_names.add(name)

    # ── load undefended baseline from attack artifacts ────────────────────────
    base = _load_baseline(amia_log, rmia_log, args.model_idx, mem)
    base["seed"] = args.seed
    logger.info(
        "Baseline (no defense): RMIA AUC=%s  AMIA row_max AUC=%s",
        f"{base['rmia_auc']:.4f}" if base["rmia_auc"] is not None else "N/A",
        f"{base['amia_row_max_auc']:.4f}" if base["amia_row_max_auc"] is not None else "N/A",
    )

    # Require baseline attack artifacts before running defended evaluations.
    missing_baselines = []
    if run_rmia_flag and base["rmia_auc"] is None:
        missing_baselines.append("RMIA")
    if run_amia_flag and base["amia_row_max_auc"] is None:
        missing_baselines.append("AMIA")
    if missing_baselines:
        needed = " and ".join(missing_baselines)
        raise RuntimeError(
            f"Missing baseline {needed} artifacts before defense evaluation. "
            "Run baseline attacks first (e.g. run_attacks/rmia.py and "
            "run_attacks/amia/amia_tabpfn.py), then rerun eval_defenses.py."
        )

    # ── run each configuration ────────────────────────────────────────────────
    results = []
    # Non-member pool samples — same test set the RMIA baseline accuracy was logged on
    X_test = X_pool[~mem]
    y_test = y_pool[~mem]
    base["accuracy"] = _accuracy(target_model, X_test, y_test, args.batch_size)
    logger.info("Baseline recomputed accuracy on target non-member audit split: %.4f", base["accuracy"])

    for defense_name, wrap_fn in defense_configs:
        safe_name = _safe_defense_name(defense_name)
        if args.seed is not None:
            defense_eval_log_dir = os.path.join(out_dir, safe_name, f"seed{args.seed}")
        else:
            defense_eval_log_dir = os.path.join(out_dir, safe_name)
        os.makedirs(defense_eval_log_dir, exist_ok=True)
        defense_logger = setup_log(
            defense_eval_log_dir,
            "eval_defenses",
            True,
            logger_name=f"eval_defenses.{safe_name}.seed{args.seed}",
        )

        logger.info("\n" + "=" * 70)
        logger.info("Defense: %s", defense_name)
        defense_logger.info("Defense: %s", defense_name)
        if skip_existing:
            existing = _load_existing_defense_result(
                defense_name,
                out_dir,
                run_rmia_flag,
                run_amia_flag,
                seed=args.seed,
            )
            if existing is not None:
                logger.info("[%s] Existing artifacts found; skipping because --skip-existing was set.",
                            defense_name)
                defense_logger.info(
                    "[%s] Existing artifacts found; skipping because --skip-existing was set.",
                    defense_name,
                )
                results.append(existing)
                continue

        row = _run_one_defense(
            defense_name, wrap_fn,
            models_list, target_model, args.model_idx, mem,
            X_pool, y_pool, X_pop, y_pop, X_test, y_test,
            rmia_signals, rmia_signals_pop,
            auditing_membership, num_ref,
            n_context, args.batch_size,
            out_dir, amia_log, args.dataset, args.model.lower(),
            run_rmia_flag, run_amia_flag, defense_logger,
            seed=args.seed,
            skip_existing=skip_existing,
        )
        results.append(row)

    # ── summary table ─────────────────────────────────────────────────────────
    base_rmia = base.get("rmia_auc") or 0.5
    base_amia = base.get("amia_row_max_auc") or 0.5
    base_acc  = base.get("accuracy") or 1.0

    def _fmt_row(r, base_rmia, base_amia, base_acc, col_w):
        t  = f"{r['time_s']:.0f}s"
        ra = (f"{r['rmia_auc']:.4f}  (Δ {r['rmia_auc'] - base_rmia:+.4f})"
              if r["rmia_auc"] is not None else "-")
        aa = (f"{r['amia_row_max_auc']:.4f}  (Δ {r['amia_row_max_auc'] - base_amia:+.4f})"
              if r["amia_row_max_auc"] is not None else "-")
        ac = (f"{r['accuracy']:.4f}  (Δ {r['accuracy'] - base_acc:+.4f})"
              if r["accuracy"] is not None else "-")
        return f"{r['defense']:<{col_w}} {t:>6}  {ra:>24}  {aa:>24}  {ac:>18}"

    col_w = 32
    sep  = "=" * 100
    sep2 = "-" * 100
    hdr  = (
        f"{'Defense':<{col_w}} {'Time':>6}  "
        f"{'RMIA AUC':>24}  {'AMIA row_max AUC':>24}  {'Accuracy':>18}"
    )
    bra = f"{base['rmia_auc']:.4f}" if base["rmia_auc"] is not None else "-"
    baa = f"{base['amia_row_max_auc']:.4f}" if base["amia_row_max_auc"] is not None else "-"
    bac = f"{base['accuracy']:.4f}" if base["accuracy"] is not None else "-"
    base_row = (
        f"{'no_defense (baseline)':<{col_w}} {'':>6}  {bra:>24}  {baa:>24}  {bac:>18}"
    )
    result_rows = [_fmt_row(r, base_rmia, base_amia, base_acc, col_w) for r in results]

    table_lines = [sep, hdr, sep2, base_row, sep2, *result_rows, sep]
    table_str = "\n".join(table_lines)
    logger.info("Final defense summary:\n%s", table_str)

    # ── save CSV (merge: preserve existing rows, update/add new ones) ─────────
    csv_path = os.path.join(out_dir, "defense_eval_results.csv")
    fieldnames = [
        "seed", "defense", "time_s",
        "rmia_auc", "amia_row_max_auc", "amia_row_ent_auc", "amia_row_max_d",
        "accuracy",
    ]
    # Load existing rows keyed by seed + defense name.
    existing: dict[tuple[str, str], dict] = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                existing[(str(row.get("seed", "")), row["defense"])] = row
    # Update with current run (baseline + new results)
    for row in [base, *results]:
        key = row.get("defense", "no_defense (baseline)")
        # DictWriter extrasaction="ignore" handles extra keys; ensure key field present
        if "defense" not in row:
            row = {**row, "defense": "no_defense (baseline)"}
        row_seed = row.get("seed", args.seed)
        existing[(str(row_seed), key)] = {f: row.get(f) for f in fieldnames}
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing.values())
    logger.info("Results saved to %s", csv_path)


if __name__ == "__main__":
    main()
