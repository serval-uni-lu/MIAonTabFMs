#!/usr/bin/env python3
"""
Detect high-risk audit queries from an AMIA attention summary.

The detector is intentionally lightweight: it does not rerun TabPFN or AMIA.
It reads attention_summary.csv and flags queries whose attention-based risk
score is at least the minimum known-member risk score when membership labels
are available. This matches the adaptive guardrail policy that tries to catch
as many member-like queries as possible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _default_summary_path(dataset: str, model: str, defense: str | None) -> Path:
    base = Path("ml_privacy_meter") / "logs" / dataset / model.lower()
    if defense:
        return base / "defense" / defense / "amia" / "report" / "exp" / "attention_summary.csv"
    return base / "amia" / "report" / "exp" / "attention_summary.csv"


def _zscore(values: np.ndarray) -> np.ndarray:
    mu = np.nanmean(values)
    sigma = np.nanstd(values)
    if not np.isfinite(sigma) or sigma == 0:
        return values - mu
    return (values - mu) / sigma


def _score(df: pd.DataFrame, score_name: str) -> np.ndarray:
    if score_name == "row_max":
        return df["row_max"].to_numpy(dtype=float)
    if score_name == "row_ent":
        return df["row_ent"].to_numpy(dtype=float)
    if score_name == "rmia_score":
        return df["rmia_score"].to_numpy(dtype=float)
    if score_name == "combined":
        parts = [_zscore(df["row_max"].to_numpy(dtype=float))]
        if "row_ent" in df:
            parts.append(_zscore(df["row_ent"].to_numpy(dtype=float)))
        if "rmia_score" in df:
            parts.append(_zscore(df["rmia_score"].to_numpy(dtype=float)))
        return np.mean(np.vstack(parts), axis=0)
    raise ValueError(f"Unknown score: {score_name}")


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC without sklearn/scipy."""
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j

    pos = labels.astype(bool)
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _format_rate(num: int, den: int) -> str:
    if den == 0:
        return "n/a"
    return f"{num / den:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Flag high-risk queries from AMIA attention_summary.csv. "
            "By default, the threshold is the minimum known-member risk score."
        )
    )
    parser.add_argument("--dataset", type=str, default="credit_rating")
    parser.add_argument("--model", type=str, default="tabpfn")
    parser.add_argument(
        "--defense",
        type=str,
        default=None,
        help="Defense folder name, e.g. attn_drop_p30. Omit for baseline AMIA.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Explicit attention_summary.csv path. Overrides --dataset/--model/--defense.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults beside the input summary.",
    )
    parser.add_argument(
        "--score",
        choices=["row_max", "row_ent", "rmia_score", "combined"],
        default="row_max",
        help=(
            "Risk score. row_max is the AMIA attention-peak signal; row_ent is "
            "negative entropy, so larger means sharper attention."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Explicit risk threshold. If omitted, use the minimum known-member "
            "risk score from the AMIA summary."
        ),
    )
    parser.add_argument(
        "--only-flagged",
        action="store_true",
        help="Write only high-risk rows instead of the full summary plus flags.",
    )
    parser.add_argument(
        "--fallback-defense",
        type=str,
        default=None,
        help=(
            "Optional defended run to apply to high-risk rows in a simulated "
            "adaptive AMIA summary, e.g. attn_drop_p30_l4,5,7,8,10,11."
        ),
    )
    parser.add_argument(
        "--fallback-summary-csv",
        type=Path,
        default=None,
        help="Explicit fallback defense attention_summary.csv path.",
    )
    parser.add_argument(
        "--adaptive-summary-csv",
        type=Path,
        default=None,
        help="Where to write the simulated adaptive AMIA summary.",
    )
    args = parser.parse_args()

    summary_path = args.summary_csv or _default_summary_path(args.dataset, args.model, args.defense)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing AMIA summary: {summary_path}")

    df = pd.read_csv(summary_path)
    required = {args.score} if args.score != "combined" else {"row_max"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{summary_path} is missing required columns: {missing}")

    risk = _score(df, args.score)
    finite = np.isfinite(risk)
    if not finite.any():
        raise ValueError(f"No finite values found for score {args.score}.")

    labels = None
    has_member = "member" in df.columns
    if has_member:
        labels = df["member"].to_numpy(dtype=bool)

    if args.threshold is not None:
        threshold = float(args.threshold)
        calibration_note = "explicit threshold"
    else:
        if labels is None:
            raise ValueError(
                "Default high-risk detection requires a member column in the "
                "AMIA summary. Pass --threshold to use an explicit threshold."
            )
        if not (labels & finite).any():
            raise ValueError("Cannot calibrate because no finite member scores exist.")
        calibration_scores = risk[labels & finite]
        threshold = float(np.min(calibration_scores))
        calibration_note = "minimum known-member risk"

    high_risk = finite & (risk >= threshold)

    out = df.copy()
    out.insert(0, "query_idx", np.arange(len(out)))
    out["risk_score"] = risk
    out["risk_threshold"] = threshold
    out["high_risk"] = high_risk

    if args.only_flagged:
        out = out[out["high_risk"]].sort_values("risk_score", ascending=False)

    out_path = args.out_csv or summary_path.with_name(
        f"high_risk_queries_{args.score}.csv"
    )
    out.to_csv(out_path, index=False)

    print(f"Input: {summary_path}")
    print(f"Output: {out_path}")
    print(f"Score: {args.score}")
    print(f"Threshold: {threshold:.6g} ({calibration_note})")
    print(f"Flagged: {int(high_risk.sum())}/{len(high_risk)} ({high_risk.mean():.4f})")

    if labels is not None:
        tp = int((high_risk & labels).sum())
        fp = int((high_risk & ~labels).sum())
        fn = int((~high_risk & labels & finite).sum())
        tn = int((~high_risk & ~labels & finite).sum())
        print(f"Precision among flagged: {_format_rate(tp, tp + fp)}")
        print(f"Member recall: {_format_rate(tp, tp + fn)}")
        print(f"Non-member FPR: {_format_rate(fp, fp + tn)}")
        print(f"Risk-score AUC: {_auc(risk[finite], labels[finite]):.4f}")

    fallback_path = args.fallback_summary_csv
    if fallback_path is None and args.fallback_defense:
        fallback_path = _default_summary_path(args.dataset, args.model, args.fallback_defense)
    if fallback_path is not None:
        if not fallback_path.exists():
            raise FileNotFoundError(f"Missing fallback AMIA summary: {fallback_path}")
        fallback = pd.read_csv(fallback_path)
        if len(fallback) != len(df):
            raise ValueError(
                "Fallback summary must have the same row count as the probe summary: "
                f"{len(fallback)} vs {len(df)}."
            )

        adaptive = df.copy()
        for col in ["rmia_score", "row_max", "row_ent", "col_max", "col_ent"]:
            if col in adaptive.columns and col in fallback.columns:
                adaptive.loc[high_risk, col] = fallback.loc[high_risk, col].to_numpy()
        adaptive.insert(0, "query_idx", np.arange(len(adaptive)))
        adaptive["probe_risk_score"] = risk
        adaptive["risk_threshold"] = threshold
        adaptive["fallback_applied"] = high_risk

        adaptive_path = args.adaptive_summary_csv or summary_path.with_name(
            f"attention_summary_adaptive_{args.score}.csv"
        )
        adaptive.to_csv(adaptive_path, index=False)

        print(f"Fallback summary: {fallback_path}")
        print(f"Adaptive summary: {adaptive_path}")
        if labels is not None and "row_max" in adaptive:
            adaptive_auc = _auc(
                adaptive["row_max"].to_numpy(dtype=float),
                labels,
            )
            base_auc = _auc(df["row_max"].to_numpy(dtype=float), labels)
            fallback_auc = (
                _auc(fallback["row_max"].to_numpy(dtype=float), labels)
                if "row_max" in fallback
                else float("nan")
            )
            print(f"Baseline row_max AUC: {base_auc:.4f}")
            print(f"Fallback row_max AUC: {fallback_auc:.4f}")
            print(f"Adaptive row_max AUC: {adaptive_auc:.4f}")


if __name__ == "__main__":
    main()
