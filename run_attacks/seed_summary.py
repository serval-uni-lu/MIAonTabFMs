"""Shared helpers for updating attack_result_seed_runs.csv and _summary.csv.

Called by individual attack scripts (lira, population, ...) and by
rebuild_seed_summaries.py for backfilling existing results.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

METRIC_KEYS = ["auc", "one_tenth_fpr", "zero_fpr", "one_tenth_fnr", "zero_fnr"]
AMIA_SCORE_KEYS = ["row_max", "row_ent", "col_max", "col_ent", "rmia_score"]


def _npz_for_report(report_dir: Path):
    avg = report_dir / "attack_result_average.npz"
    if avg.exists():
        return avg
    single = report_dir / "exp" / "attack_result_0.npz"
    if single.exists():
        return single
    return None


def _test_acc(report_dir: Path):
    for candidate in (report_dir, *report_dir.parents):
        meta = candidate / "models" / "models_metadata.json"
        if not meta.exists():
            continue
        with open(meta) as f:
            md = json.load(f)
        m0 = md.get("0")
        if m0 and "test_acc" in m0:
            return float(m0["test_acc"])
        break
    return None


def _npz_metrics(report_dir: Path) -> dict | None:
    npz = _npz_for_report(report_dir)
    if npz is None:
        return None
    with np.load(npz) as data:
        row = {}
        for key in METRIC_KEYS:
            if key in data.files:
                row[key] = float(data[key])
    acc = _test_acc(report_dir)
    if acc is not None:
        row["target_model_0_test_acc"] = acc
    return row


def _amia_metrics(report_dir: Path) -> dict | None:
    summary_csv = report_dir / "exp" / "attention_summary.csv"
    if not summary_csv.exists():
        return None
    df = pd.read_csv(summary_csv)
    if "member" not in df.columns:
        return None
    mem = df["member"].to_numpy(dtype=bool)
    if mem.sum() == 0 or (~mem).sum() == 0:
        return None
    row = {}
    for key in AMIA_SCORE_KEYS:
        if key not in df.columns:
            continue
        vals = df[key].to_numpy(dtype=float)
        try:
            row[f"{key}_auc"] = float(roc_auc_score(mem.astype(int), vals))
        except Exception:
            pass
    return row or None


def update_seed_row(
    attack: str,
    seed: int,
    report_dir: Path,
    summary_dir: Path,
    metrics: dict | None = None,
) -> None:
    """Upsert one seed's row in attack_result_seed_runs.csv and recompute summary."""
    if metrics is None:
        if attack == "amia":
            metrics = _amia_metrics(report_dir)
        else:
            metrics = _npz_metrics(report_dir)
    if not metrics:
        return

    row = {"attack": attack, "seed": seed, "report_dir": str(report_dir)}
    row.update(metrics)

    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    runs_path = summary_dir / "attack_result_seed_runs.csv"

    if runs_path.exists():
        runs_df = pd.read_csv(runs_path)
        if "attack" not in runs_df.columns:
            runs_df.insert(0, "attack", "rmia")
        runs_df = runs_df[~((runs_df["attack"] == attack) & (runs_df["seed"] == seed))]
        runs_df = pd.concat([runs_df, pd.DataFrame([row])], ignore_index=True, sort=False)
    else:
        runs_df = pd.DataFrame([row])

    runs_df = runs_df.sort_values(["attack", "seed"]).reset_index(drop=True)
    runs_df.to_csv(runs_path, index=False)
    _recompute_summary(attack, runs_df, summary_dir)


def _recompute_summary(attack: str, runs_df: pd.DataFrame, summary_dir: Path) -> None:
    attack_runs = runs_df[runs_df["attack"] == attack]
    skip_cols = {"attack", "seed", "report_dir"}
    numeric_cols = [c for c in attack_runs.columns if c not in skip_cols]
    new_rows = []
    for col in numeric_cols:
        values = attack_runs[col].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        new_rows.append({
            "attack": attack,
            "metric": col,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "num_seeds": int(len(values)),
        })

    summary_path = summary_dir / "attack_result_seed_summary.csv"
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        if "attack" not in summary_df.columns:
            summary_df.insert(0, "attack", "rmia")
        summary_df = summary_df[summary_df["attack"] != attack]
        summary_df = pd.concat([summary_df, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    else:
        summary_df = pd.DataFrame(new_rows)

    summary_df.to_csv(summary_path, index=False)
