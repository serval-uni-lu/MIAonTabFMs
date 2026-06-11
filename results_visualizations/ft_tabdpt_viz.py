"""Visualize original vs fine-tuned TabDPT ft_tabdpt runs.

Recommended use from the repository root:
    .venv/bin/python results_visualizations/ft_tabdpt_viz.py

The script expects one experiment directory like:
    ml_privacy_meter/logs/ft_tabdpt/46905_Amazon_employee_access

It creates several plots that compare original and fine-tuned behavior:
accuracy/performance deltas, membership score shifts, ROC curves, attention
summary changes, layer/head sensitivity, and prediction transition counts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_curve


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_LOGS_BASE = Path("ml_privacy_meter/logs/ft_tabdpt")
DEFAULT_OUTPUT_BASE = Path("results_visualizations/ft_tabdpt_viz")
DATASETS = ["purchases10", "46905_Amazon_employee_access"]
CONTEXT_RMIA_SUBDIRS = ("report_context_finetuned_refs", "report_context_original_refs")
CONTEXT_REFERENCE_MODES = {
    "report_context_finetuned_refs": "finetuned_checkpoint_references",
    "report_context_original_refs": "original_checkpoint_references",
}
# Maps context subdir → matching finetuned variant key in summary.json and report dir
CONTEXT_TO_FT_VARIANT = {
    "report_context_finetuned_refs": ("finetuned_model_rmia_ft_refs", "report_finetuned_ft_refs"),
    "report_context_original_refs": ("finetuned_model_rmia_original_refs", "report_finetuned_original_refs"),
}
CONTEXT_RMIA_LABELS = {
    "report_context_finetuned_refs": "context / finetuned refs",
    "report_context_original_refs": "context / original refs",
}

PALETTE = {
    "original": "#086375",
    "finetuned": "#fb8b24",
    "member": "#7a5195",
    "nonmember": "#2a9d8f",
    "delta": "#d9a441",
    "neutral": "#4f5d75",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def safe_filename(name: str) -> str:
    keep = [c if c.isalnum() or c in ("-", "_") else "_" for c in str(name)]
    return "".join(keep).strip("_") or "ft_tabdpt_run"


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def savefig(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path



def plot_finetuning_composite(run_dir: Path, rmia_df: pd.DataFrame, rmia_dir: Path) -> None:
    """Create a clean single-row summary without cropping pre-rendered figures."""
    if rmia_df.empty:
        return

    original_perf = load_json(run_dir / "original_performance.json")
    finetuned_perf = load_json(run_dir / "finetuned_performance.json")
    metric_labels = {
        "train_accuracy": "Train accuracy",
        "test_accuracy": "Test accuracy",
        "member_score_mean": "Member true-label prob.",
        "nonmember_score_mean": "Non-member true-label prob.",
        "membership_auc_true_label_probability": "MIA AUC",
    }
    perf_rows = []
    for metric, label in metric_labels.items():
        perf_rows.append({"metric": label, "model": "Original", "value": original_perf.get(metric, np.nan)})
        perf_rows.append({"metric": label, "model": "Fine-tuned", "value": finetuned_perf.get(metric, np.nan)})
    perf_df = pd.DataFrame(perf_rows).dropna(subset=["value"])

    df = add_rmia_features(rmia_df).copy()
    if "membership_group" not in df.columns:
        return
    df["target_signal_delta"] = df["target_signal_finetuned"] - df["target_signal_original"]

    with plt.rc_context({
        "font.size": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    }):
        fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
        sns.barplot(
            data=perf_df,
            x="metric",
            y="value",
            hue="model",
            palette={"Original": PALETTE["original"], "Fine-tuned": PALETTE["finetuned"]},
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Value")
        ax.set_title("")
        ax.tick_params(axis="x", labelrotation=35)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        ax.legend(title="", loc="lower center", bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False)
        fig.savefig(rmia_dir / "15_a_finetuning_utility_bars.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    with plt.rc_context({
        "font.size": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.02, wspace=0.08, hspace=0.02)

        for score_col, color, label in [
            ("rmia_finetuned", PALETTE["finetuned"], "Fine-tuned"),
            ("rmia_original", PALETTE["original"], "Original"),
        ]:
            if score_col not in df.columns:
                continue
            for grp, ls in [("Member", "-"), ("Non-member", "--")]:
                subset = df.loc[df["membership_group"] == grp, score_col].dropna()
                if len(subset) > 2:
                    sns.kdeplot(
                        x=subset,
                        ax=axes[0],
                        color=color,
                        linestyle=ls,
                        linewidth=2.2,
                        common_norm=False,
                        fill=False,
                        clip=(0, 1),
                        label=f"{label}, {grp}",
                    )
        axes[0].axvline(0.5, color="#333333", linestyle=":", linewidth=1)
        axes[0].set_xlabel("RMIA score")
        axes[0].set_ylabel("Density")
        axes[0].set_title("")
        axes[0].set_xlim(0, 1)
        axes[0].legend(fontsize=11, ncol=1, frameon=True)

        for grp, color, ls in [
            ("Non-member", PALETTE["nonmember"], "--"),
            ("Member", PALETTE["member"], "-"),
        ]:
            subset = df.loc[df["membership_group"] == grp, "target_signal_delta"].dropna()
            if len(subset) > 10:
                sns.kdeplot(
                    x=subset,
                    ax=axes[1],
                    color=color,
                    linestyle=ls,
                    linewidth=2.4,
                    fill=False,
                    common_norm=False,
                    label=grp,
                )
        axes[1].axvline(0, color="#333333", linestyle=":", linewidth=1)
        axes[1].set_xlabel(r"$\Delta P = P_{\theta_{\mathrm{ft}}} - P_{\theta_0}$")
        axes[1].set_ylabel("Density")
        axes[1].set_title("")
        axes[1].legend(title="", frameon=True)

        fig.savefig(rmia_dir / "15_finetuning_mia_composite.png", dpi=300, bbox_inches="tight")
        plt.close(fig)



def signed_logit(prob: pd.Series, eps: float = 1e-8) -> pd.Series:
    p = prob.clip(eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def add_membership_features(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    out["member_bool"] = bool_series(out["member"])
    out["membership_group"] = np.where(out["member_bool"], "Member", "Non-member")
    out["original_logit"] = signed_logit(out["original_true_label_score"])
    out["finetuned_logit"] = signed_logit(out["finetuned_true_label_score"])
    if "score_delta" not in out:
        out["score_delta"] = out["finetuned_logit"] - out["original_logit"]
    out["prob_delta"] = out["finetuned_true_label_score"] - out["original_true_label_score"]
    out["correct_transition"] = (
        out["original_correct"].astype(str)
        + " -> "
        + out["finetuned_correct"].astype(str)
    )
    return out


def plot_performance_summary(
    original_perf: dict[str, Any],
    finetuned_perf: dict[str, Any],
    model_change: dict[str, Any],
    attention_change: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    perf_rows = []
    metric_labels = {
        "train_accuracy": "Train accuracy",
        "test_accuracy": "Test accuracy",
        "member_score_mean": "Member true-label prob.",
        "nonmember_score_mean": "Non-member true-label prob.",
        "membership_auc_true_label_probability": "MIA AUC",
    }
    if "membership_auc_logit_delta_vs_original" in finetuned_perf:
        metric_labels["membership_auc_logit_delta_vs_original"] = "MIA AUC: logit delta"

    for metric, label in metric_labels.items():
        if metric == "membership_auc_logit_delta_vs_original":
            perf_rows.append(
                {
                    "metric": label,
                    "model": "finetuned",
                    "value": finetuned_perf.get(metric, np.nan),
                    "delta": np.nan,
                }
            )
            continue
        original_value = original_perf.get(metric, np.nan)
        finetuned_value = finetuned_perf.get(metric, np.nan)
        perf_rows.append({"metric": label, "model": "original", "value": original_value, "delta": 0.0})
        perf_rows.append(
            {
                "metric": label,
                "model": "finetuned",
                "value": finetuned_value,
                "delta": finetuned_value - original_value,
            }
        )

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(output_dir / "performance_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(
        data=perf_df.dropna(subset=["value"]),
        y="metric",
        x="value",
        hue="model",
        palette={"original": PALETTE["original"], "finetuned": PALETTE["finetuned"]},
        ax=axes[0],
    )
    axes[0].set_title("Original vs fine-tuned summary metrics")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("")
    axes[0].legend(title="")

    delta_df = perf_df[(perf_df["model"] == "finetuned") & perf_df["delta"].notna()].copy()
    sns.barplot(data=delta_df, y="metric", x="delta", color=PALETTE["delta"], ax=axes[1])
    axes[1].axvline(0, color="#333333", linewidth=1)
    axes[1].set_title("Fine-tuning delta")
    axes[1].set_xlabel("Fine-tuned minus original")
    axes[1].set_ylabel("")

    savefig(fig, output_dir, "01_performance_summary.png")

    compact = {
        "model_change": model_change,
        "attention_change": attention_change,
        "performance_summary_csv": str(output_dir / "performance_summary.csv"),
    }
    with (output_dir / "run_summary.json").open("w") as f:
        json.dump(compact, f, indent=2)
    return perf_df


def plot_membership_distributions(scores: pd.DataFrame, output_dir: Path) -> None:
    long_prob = scores.melt(
        id_vars=["member_bool", "membership_group"],
        value_vars=["original_true_label_score", "finetuned_true_label_score"],
        var_name="model",
        value_name="true_label_probability",
    )
    long_prob["model"] = long_prob["model"].str.replace("_true_label_score", "", regex=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for model, color in [("original", PALETTE["original"]), ("finetuned", PALETTE["finetuned"])]:
        for group, linestyle in [("Member", "-"), ("Non-member", "--")]:
            subset = long_prob[(long_prob["model"] == model) & (long_prob["membership_group"] == group)]
            sns.kdeplot(
                data=subset,
                x="true_label_probability",
                common_norm=False,
                fill=False,
                color=color,
                linestyle=linestyle,
                label=f"{model}, {group}",
                ax=axes[0, 0],
            )
    axes[0, 0].set_title("True-label probability distributions")
    axes[0, 0].set_xlabel("True-label probability")

    sns.violinplot(
        data=scores,
        x="membership_group",
        y="score_delta",
        hue="membership_group",
        palette={"Member": PALETTE["member"], "Non-member": PALETTE["nonmember"]},
        legend=False,
        cut=0,
        inner="quartile",
        ax=axes[0, 1],
    )
    axes[0, 1].axhline(0, color="#333333", linewidth=1)
    axes[0, 1].set_title("Logit score delta by membership")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Fine-tuned logit minus original logit")

    sample = scores.sample(min(len(scores), 6000), random_state=7)
    sns.scatterplot(
        data=sample,
        x="original_true_label_score",
        y="finetuned_true_label_score",
        hue="membership_group",
        palette={"Member": PALETTE["member"], "Non-member": PALETTE["nonmember"]},
        alpha=0.35,
        linewidth=0,
        s=12,
        ax=axes[1, 0],
    )
    axes[1, 0].plot([0, 1], [0, 1], color="#333333", linewidth=1, linestyle="--")
    axes[1, 0].set_title("Per-record confidence shift")
    axes[1, 0].set_xlabel("Original true-label probability")
    axes[1, 0].set_ylabel("Fine-tuned true-label probability")

    delta_summary = (
        scores.groupby("membership_group")
        .agg(
            prob_delta_mean=("prob_delta", "mean"),
            prob_delta_median=("prob_delta", "median"),
            logit_delta_mean=("score_delta", "mean"),
            logit_delta_median=("score_delta", "median"),
        )
        .reset_index()
    )
    delta_summary.to_csv(output_dir / "membership_delta_summary.csv", index=False)
    sns.barplot(
        data=delta_summary.melt("membership_group", value_vars=["prob_delta_mean", "prob_delta_median"]),
        x="membership_group",
        y="value",
        hue="variable",
        palette=[PALETTE["delta"], PALETTE["neutral"]],
        ax=axes[1, 1],
    )
    axes[1, 1].axhline(0, color="#333333", linewidth=1)
    axes[1, 1].set_title("Mean/median probability change")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Fine-tuned probability minus original")
    axes[1, 1].legend(title="")

    savefig(fig, output_dir, "02_membership_score_distributions.png")


def plot_membership_roc(scores: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    y_true = scores["member_bool"].astype(int).to_numpy()
    roc_specs = [
        ("Original true-label prob.", scores["original_true_label_score"].to_numpy(), PALETTE["original"]),
        ("Fine-tuned true-label prob.", scores["finetuned_true_label_score"].to_numpy(), PALETTE["finetuned"]),
        ("Logit delta", scores["score_delta"].to_numpy(), PALETTE["delta"]),
        ("Probability delta", scores["prob_delta"].to_numpy(), PALETTE["neutral"]),
    ]

    rows = []
    fig, ax = plt.subplots(figsize=(8, 7))
    for label, values, color in roc_specs:
        fpr, tpr, _ = roc_curve(y_true, values)
        curve_auc = sk_auc(fpr, tpr)
        rows.append({"score": label, "auc": curve_auc})
        ax.plot(fpr, tpr, label=f"{label} (AUC={curve_auc:.4f})", linewidth=2, color=color)

    ax.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1)
    ax.set_title("Membership inference ROC from confidence signals")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    savefig(fig, output_dir, "03_membership_roc.png")

    auc_df = pd.DataFrame(rows)
    auc_df.to_csv(output_dir / "membership_auc_summary.csv", index=False)
    return auc_df


def plot_attention_summary(attention_df: pd.DataFrame, output_dir: Path) -> None:
    if attention_df.empty:
        return
    df = attention_df.copy()
    df["member_bool"] = bool_series(df["member"])
    df["membership_group"] = np.where(df["member_bool"], "Member", "Non-member")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for metric, ax, title in [
        ("row_max_delta", axes[0, 0], "Attention row-max delta"),
        ("row_ent_delta", axes[0, 1], "Attention row-entropy delta"),
    ]:
        sns.kdeplot(
            data=df,
            x=metric,
            hue="membership_group",
            common_norm=False,
            fill=False,
            palette={"Member": PALETTE["member"], "Non-member": PALETTE["nonmember"]},
            ax=ax,
        )
        ax.axvline(0, color="#333333", linewidth=1)
        ax.set_title(title)

    sample = df.sample(min(len(df), 6000), random_state=11)
    sns.scatterplot(
        data=sample,
        x="score_delta",
        y="row_max_delta",
        hue="membership_group",
        palette={"Member": PALETTE["member"], "Non-member": PALETTE["nonmember"]},
        alpha=0.35,
        linewidth=0,
        s=12,
        ax=axes[1, 0],
    )
    axes[1, 0].axhline(0, color="#333333", linewidth=1)
    axes[1, 0].axvline(0, color="#333333", linewidth=1)
    axes[1, 0].set_title("Confidence delta vs attention concentration delta")
    axes[1, 0].set_xlabel("Logit score delta")
    axes[1, 0].set_ylabel("Row-max delta")

    summary = (
        df.groupby("membership_group")
        .agg(
            row_max_delta_mean=("row_max_delta", "mean"),
            row_max_delta_median=("row_max_delta", "median"),
            row_ent_delta_mean=("row_ent_delta", "mean"),
            row_ent_delta_median=("row_ent_delta", "median"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "attention_delta_summary.csv", index=False)
    sns.barplot(
        data=summary.melt("membership_group"),
        x="membership_group",
        y="value",
        hue="variable",
        palette="Set2",
        ax=axes[1, 1],
    )
    axes[1, 1].axhline(0, color="#333333", linewidth=1)
    axes[1, 1].set_title("Attention deltas by membership")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Delta")
    axes[1, 1].legend(title="", fontsize=8)

    savefig(fig, output_dir, "04_attention_summary_deltas.png")


def plot_layer_attention(layers: pd.DataFrame, output_dir: Path) -> None:
    if layers.empty:
        return
    wide = layers.pivot(index="layer", columns="model", values=["row_max_mean", "row_ent_mean"])
    layer_df = pd.DataFrame(index=wide.index)
    layer_df["row_max_delta"] = wide[("row_max_mean", "finetuned")] - wide[("row_max_mean", "original")]
    layer_df["row_ent_delta"] = wide[("row_ent_mean", "finetuned")] - wide[("row_ent_mean", "original")]
    layer_df = layer_df.reset_index()
    layer_df.to_csv(output_dir / "layer_attention_deltas.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    sns.lineplot(data=layers, x="layer", y="row_max_mean", hue="model", marker="o", ax=axes[0])
    axes[0].set_title("Attention concentration by layer")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean row max")

    sns.barplot(
        data=layer_df.melt("layer", value_vars=["row_max_delta", "row_ent_delta"]),
        x="layer",
        y="value",
        hue="variable",
        palette=[PALETTE["delta"], PALETTE["neutral"]],
        ax=axes[1],
    )
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set_title("Fine-tuned attention deltas by layer")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Delta")
    axes[1].legend(title="")
    savefig(fig, output_dir, "05_attention_layer_deltas.png")


def plot_head_attention(heads: pd.DataFrame, output_dir: Path) -> None:
    if heads.empty:
        return
    required_models = set(heads["model"].dropna())
    if not {"original", "finetuned"}.issubset(required_models):
        return

    idx_cols = ["layer", "head"]
    original = heads[heads["model"] == "original"].set_index(idx_cols)
    finetuned = heads[heads["model"] == "finetuned"].set_index(idx_cols)
    delta = pd.DataFrame(index=original.index)
    delta["row_max_delta"] = finetuned["row_max_mean"] - original["row_max_mean"]
    delta["row_ent_delta"] = finetuned["row_ent_mean"] - original["row_ent_mean"]
    delta["row_max_auc_delta"] = finetuned["row_max_auc"] - original["row_max_auc"]
    delta = delta.reset_index()
    delta.to_csv(output_dir / "head_attention_deltas.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for metric, ax, title in [
        ("row_max_delta", axes[0], "Head row-max delta"),
        ("row_ent_delta", axes[1], "Head row-entropy delta"),
        ("row_max_auc_delta", axes[2], "Head MIA AUC delta"),
    ]:
        heat = delta.pivot(index="layer", columns="head", values=metric)
        limit = float(np.nanmax(np.abs(heat.to_numpy()))) if not heat.empty else 0.0
        limit = limit if math.isfinite(limit) and limit > 0 else 1e-6
        sns.heatmap(
            heat,
            cmap="vlag",
            center=0,
            vmin=-limit,
            vmax=limit,
            linewidths=0.2,
            linecolor="white",
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
    savefig(fig, output_dir, "06_attention_head_heatmaps.png")


def plot_prediction_transitions(scores: pd.DataFrame, test_predictions: pd.DataFrame, output_dir: Path) -> None:
    score_transitions = scores["correct_transition"].value_counts().rename_axis("transition").reset_index(name="count")
    score_transitions["split"] = "membership"

    frames = [score_transitions]
    if not test_predictions.empty and {"original_correct", "finetuned_correct"}.issubset(test_predictions.columns):
        test = test_predictions.copy()
        test["correct_transition"] = (
            test["original_correct"].astype(str) + " -> " + test["finetuned_correct"].astype(str)
        )
        test_counts = test["correct_transition"].value_counts().rename_axis("transition").reset_index(name="count")
        test_counts["split"] = "test"
        frames.append(test_counts)

    transitions = pd.concat(frames, ignore_index=True)
    transitions.to_csv(output_dir / "prediction_transition_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=transitions, x="transition", y="count", hue="split", palette="Set2", ax=ax)
    ax.set_title("Correctness transitions after fine-tuning")
    ax.set_xlabel("Original correctness -> fine-tuned correctness")
    ax.set_ylabel("Records")
    ax.tick_params(axis="x", rotation=20)
    savefig(fig, output_dir, "07_prediction_transitions.png")



def load_attack_curve(path: Path) -> dict[str, np.ndarray | float] | None:
    if not path.exists():
        return None
    with np.load(path) as data:
        return {
            "fpr": data["fpr"],
            "tpr": data["tpr"],
            "auc": float(data["auc"]),
            "scores": data["scores"] if "scores" in data.files else np.array([]),
            "memberships": data["memberships"] if "memberships" in data.files else np.array([]),
        }


def normalize_rmia_scores(df: pd.DataFrame, context_rmia_subdir: str) -> pd.DataFrame:
    """Rename variant-specific columns to generic names so all plots work unchanged."""
    _, ft_report_dir = CONTEXT_TO_FT_VARIANT.get(
        context_rmia_subdir, ("finetuned_model_rmia_original_refs", "report_finetuned_original_refs")
    )
    suffix = "ft_refs" if "ft_refs" in ft_report_dir else "original_refs"
    renames = {
        f"rmia_finetuned_{suffix}": "rmia_finetuned",
        f"paired_signal_finetuned_{suffix}": "paired_signal_finetuned_target_eval",
        f"reference_signal_finetuned_{suffix}": "reference_signal_finetuned_target_eval",
    }
    for col in df.columns:
        if col.endswith(f"_finetuned_{suffix}") and col not in renames:
            renames[col] = col.replace(f"_finetuned_{suffix}", "_finetuned_target_eval")
    return df.rename(columns={k: v for k, v in renames.items() if k in df.columns})


def add_rmia_features(rmia_scores: pd.DataFrame) -> pd.DataFrame:
    out = rmia_scores.copy()
    member_col = "member" if "member" in out.columns else "target_member"
    if member_col in out.columns:
        out["member_bool"] = bool_series(out[member_col])
        out["membership_group"] = np.where(out["member_bool"], "Member", "Non-member")
    ref_ft_col = next(
        (c for c in ["reference_signal_finetuned_target_eval", "reference_signal_finetuned"] if c in out.columns),
        None,
    )
    if {"target_signal_original", "reference_signal_original"}.issubset(out.columns):
        out["original_target_minus_reference"] = out["target_signal_original"] - out["reference_signal_original"]
        out["original_target_reference_ratio"] = out["target_signal_original"] / out["reference_signal_original"].clip(1e-8)
    if ref_ft_col and "target_signal_finetuned" in out.columns:
        out["finetuned_target_minus_reference"] = out["target_signal_finetuned"] - out[ref_ft_col]
        out["finetuned_target_reference_ratio"] = out["target_signal_finetuned"] / out[ref_ft_col].clip(1e-8)
    if {"rmia_original", "rmia_finetuned"}.issubset(out.columns) and "rmia_delta" not in out.columns:
        out["rmia_delta"] = out["rmia_finetuned"] - out["rmia_original"]
    return out


def _load_context_eval(run_dir: Path, context_rmia_subdir: str) -> dict:
    """Load context eval results for the requested reference mode."""
    rmia_dir = run_dir / "rmia"
    ctx_dir = rmia_dir / context_rmia_subdir
    features_path = ctx_dir / "context_member_rmia_features.csv"
    summary_path = ctx_dir / "context_member_summary.json"
    if not features_path.exists():
        return {}
    summary = load_json(summary_path) if summary_path.exists() else {}
    ctx_auc = summary.get("context_vs_nonmember_auc", {})
    return {
        "subdir": context_rmia_subdir,
        "df": pd.read_csv(features_path),
        "original_curve": load_attack_curve(
            ctx_dir / "report_context_vs_nonmember_original" / "exp" / "attack_result_0.npz"
        ),
        "finetuned_curve": load_attack_curve(
            ctx_dir / "report_context_vs_nonmember_finetuned" / "exp" / "attack_result_0.npz"
        ),
        "original_auc": ctx_auc.get("original_model_rmia", {}).get("auc", np.nan),
        "finetuned_auc": ctx_auc.get("finetuned_model_rmia", {}).get("auc", np.nan),
        "attack_reference_mode": summary.get(
            "attack_reference_mode",
            CONTEXT_REFERENCE_MODES.get(context_rmia_subdir, ""),
        ),
    }


def plot_rmia_summary(
    run_dir: Path,
    rmia_df: pd.DataFrame,
    output_dir: Path,
    context_rmia_subdir: str,
) -> pd.DataFrame:
    rmia_dir = run_dir / "rmia"
    summary_path = rmia_dir / "summary.json"
    if rmia_df.empty or not summary_path.exists():
        return pd.DataFrame()

    rmia_summary = load_json(summary_path)
    ft_summary_key, ft_report_dir = CONTEXT_TO_FT_VARIANT.get(
        context_rmia_subdir, ("finetuned_model_rmia_original_refs", "report_finetuned_original_refs")
    )
    original_curve = load_attack_curve(rmia_dir / "report_original" / "exp" / "attack_result_0.npz")
    finetuned_curve = load_attack_curve(rmia_dir / ft_report_dir / "exp" / "attack_result_0.npz")
    ctx = _load_context_eval(run_dir, context_rmia_subdir)

    # Build metrics table: FT eval + context eval
    rows = []
    for exp, model, result in [
        ("ft_eval", "original", rmia_summary.get("original_model_rmia", {})),
        ("ft_eval", "finetuned", rmia_summary.get(ft_summary_key, {})),
    ]:
        for metric in ["auc", "tpr@0.1%fpr", "tpr@0%fpr", "tnr@0.1%fnr", "tnr@0%fnr"]:
            rows.append({"experiment": exp, "model": model, "metric": metric,
                         "value": result.get(metric, np.nan)})
    if ctx:
        ctx_auc = ctx.get("original_auc", np.nan)
        ctx_ft_auc = ctx.get("finetuned_auc", np.nan)
        for model, auc_val in [("original", ctx_auc), ("finetuned", ctx_ft_auc)]:
            rows.append({"experiment": CONTEXT_RMIA_LABELS.get(context_rmia_subdir, context_rmia_subdir), "model": model, "metric": "auc", "value": auc_val})

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "rmia_metric_summary.csv", index=False)

    ft_ref_mode = CONTEXT_REFERENCE_MODES.get(context_rmia_subdir, "")
    context_ref_mode = ctx.get("attack_reference_mode", ft_ref_mode)
    context_label = CONTEXT_RMIA_LABELS.get(context_rmia_subdir, context_rmia_subdir.replace("_", " "))
    CURVE_SPECS = [
        ("FT eval — original model",  original_curve,  PALETTE["original"],  "-"),
        ("FT eval — FT model",         finetuned_curve, PALETTE["finetuned"], "-"),
    ]
    if ctx:
        CURVE_SPECS += [
            (f"{context_label} — original model", ctx.get("original_curve"),  PALETTE["original"],  "--"),
            (f"{context_label} — FT model",        ctx.get("finetuned_curve"), PALETTE["finetuned"], "--"),
        ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # AUC bar chart
    auc_df = metrics_df[metrics_df["metric"] == "auc"].copy()
    auc_df["label"] = auc_df["experiment"].str.replace("_", " ") + " / " + auc_df["model"]
    sns.barplot(
        data=auc_df,
        x="label",
        y="value",
        hue="model",
        palette={"original": PALETTE["original"], "finetuned": PALETTE["finetuned"]},
        legend=False,
        ax=axes[0],
    )
    axes[0].axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("RMIA AUC by experiment and model")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("AUC")
    axes[0].tick_params(axis="x", rotation=25)

    # ROC curves
    for label, curve, color, ls in CURVE_SPECS:
        if curve is None:
            continue
        axes[1].plot(curve["fpr"], curve["tpr"], color=color, linestyle=ls, linewidth=2,
                     label=f"{label} (AUC={curve['auc']:.3f})")
    axes[1].plot([0, 1], [0, 1], color="#aaaaaa", linestyle=":", linewidth=1)
    axes[1].set_title("RMIA ROC — all experiments")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f"FT eval: {ft_ref_mode}  |  Context eval: {context_ref_mode}",
        fontsize=9,
        y=1.01,
    )
    savefig(fig, output_dir, "08_rmia_auc_roc.png")

    return metrics_df


def plot_rmia_score_distributions(rmia_df: pd.DataFrame, output_dir: Path) -> None:
    if rmia_df.empty or not {"rmia_original", "rmia_finetuned"}.issubset(rmia_df.columns):
        return

    df = add_rmia_features(rmia_df)
    long_scores = df.melt(
        id_vars=["member_bool", "membership_group"],
        value_vars=["rmia_original", "rmia_finetuned"],
        var_name="model",
        value_name="rmia_score",
    )
    long_scores["model"] = long_scores["model"].str.replace("rmia_", "", regex=False)

    rmia_delta_summary = (
        df.groupby("membership_group")
        .agg(
            rmia_original_mean=("rmia_original", "mean"),
            rmia_finetuned_mean=("rmia_finetuned", "mean"),
            rmia_delta_mean=("rmia_delta", "mean"),
            rmia_delta_median=("rmia_delta", "median"),
        )
        .reset_index()
    )
    rmia_delta_summary.to_csv(output_dir / "rmia_score_delta_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for model, color in [("original", PALETTE["original"]), ("finetuned", PALETTE["finetuned"])]:
        for group, linestyle in [("Member", "-"), ("Non-member", "--")]:
            subset = long_scores[(long_scores["model"] == model) & (long_scores["membership_group"] == group)]
            sns.kdeplot(
                data=subset,
                x="rmia_score",
                common_norm=False,
                fill=False,
                color=color,
                linestyle=linestyle,
                label=f"{model}, {group}",
                ax=axes[0, 0],
            )
    axes[0, 0].set_title("RMIA score distributions")
    axes[0, 0].set_xlabel("RMIA score")

    sns.violinplot(
        data=df,
        x="membership_group",
        y="rmia_delta",
        hue="membership_group",
        palette={"Member": PALETTE["member"], "Non-member": PALETTE["nonmember"]},
        legend=False,
        cut=0,
        inner="quartile",
        ax=axes[0, 1],
    )
    axes[0, 1].axhline(0, color="#333333", linewidth=1)
    axes[0, 1].set_title("RMIA score delta by membership")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Fine-tuned RMIA score minus original")

    sample = df.sample(min(len(df), 6000), random_state=19)
    sns.scatterplot(
        data=sample,
        x="rmia_original",
        y="rmia_finetuned",
        hue="membership_group",
        palette={"Member": PALETTE["member"], "Non-member": PALETTE["nonmember"]},
        alpha=0.35,
        linewidth=0,
        s=12,
        ax=axes[1, 0],
    )
    axes[1, 0].plot([0, 1], [0, 1], color="#333333", linewidth=1, linestyle="--")
    axes[1, 0].set_title("Per-record RMIA score shift")
    axes[1, 0].set_xlabel("Original RMIA score")
    axes[1, 0].set_ylabel("Fine-tuned RMIA score")

    sns.barplot(
        data=rmia_delta_summary.melt(
            "membership_group",
            value_vars=["rmia_original_mean", "rmia_finetuned_mean", "rmia_delta_mean"],
        ),
        x="membership_group",
        y="value",
        hue="variable",
        palette=[PALETTE["original"], PALETTE["finetuned"], PALETTE["delta"]],
        ax=axes[1, 1],
    )
    axes[1, 1].axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Mean RMIA scores")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].legend(title="", fontsize=8)

    savefig(fig, output_dir, "09_rmia_score_distributions.png")


def plot_experiment_comparison(
    ft_df: pd.DataFrame,
    ctx: dict,
    output_dir: Path,
    context_rmia_subdir: str,
) -> None:
    """Compare FT eval and context eval RMIA scores: who leaks what."""
    if ft_df.empty or not ctx:
        return

    ft = add_rmia_features(ft_df)
    ctx_df = add_rmia_features(ctx["df"])

    # Build a combined frame: FT members, context members, shared non-members
    ft_members = ft[ft["member_bool"] == True].copy()
    ft_members["group"] = "FT member"
    ft_nonmembers = ft[ft["member_bool"] == False].copy()
    ft_nonmembers["group"] = "Non-member"
    ctx_members = ctx_df.copy()
    ctx_members["group"] = "Context member"

    combined = pd.concat(
        [ft_members[["group", "rmia_original", "rmia_finetuned", "rmia_delta"]],
         ft_nonmembers[["group", "rmia_original", "rmia_finetuned", "rmia_delta"]],
         ctx_members[["group", "rmia_original", "rmia_finetuned", "rmia_delta"]]],
        ignore_index=True,
    )
    GROUP_ORDER = ["FT member", "Context member", "Non-member"]
    GROUP_PALETTE = {
        "FT member": PALETTE["finetuned"],
        "Context member": "#2a9d8f",
        "Non-member": PALETTE["neutral"],
    }
    combined["group"] = pd.Categorical(combined["group"], categories=GROUP_ORDER, ordered=True)
    combined.to_csv(output_dir / "experiment_comparison_scores.csv", index=False)

    long = combined.melt(
        id_vars=["group"],
        value_vars=["rmia_original", "rmia_finetuned"],
        var_name="model",
        value_name="rmia_score",
    )
    long["model"] = long["model"].map({"rmia_original": "Original model", "rmia_finetuned": "FT model"})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: RMIA score distributions by group (both models, violin)
    sns.violinplot(
        data=long,
        x="group",
        y="rmia_score",
        hue="model",
        order=GROUP_ORDER,
        palette={"Original model": PALETTE["original"], "FT model": PALETTE["finetuned"]},
        cut=0,
        inner="quartile",
        linewidth=0.8,
        ax=axes[0],
    )
    axes[0].axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    axes[0].set_title("RMIA score by group and model")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("RMIA score")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend(title="", fontsize=8)

    # Panel 2: rmia_delta (FT model - original model) by group
    sns.boxplot(
        data=combined,
        x="group",
        y="rmia_delta",
        hue="group",
        order=GROUP_ORDER,
        palette=GROUP_PALETTE,
        showfliers=False,
        linewidth=0.8,
        legend=False,
        ax=axes[1],
    )
    axes[1].axhline(0, color="#333333", linestyle="--", linewidth=1)
    axes[1].set_title("RMIA delta: FT model − original model")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("rmia_finetuned − rmia_original")
    axes[1].tick_params(axis="x", rotation=15)

    # Panel 3: mean RMIA scores summary bar chart
    summary = (
        combined.groupby("group", observed=True)[["rmia_original", "rmia_finetuned"]]
        .mean()
        .reset_index()
        .melt("group", var_name="model", value_name="mean_rmia")
    )
    summary["model"] = summary["model"].map(
        {"rmia_original": "Original model", "rmia_finetuned": "FT model"}
    )
    sns.barplot(
        data=summary,
        x="group",
        y="mean_rmia",
        hue="model",
        order=GROUP_ORDER,
        palette={"Original model": PALETTE["original"], "FT model": PALETTE["finetuned"]},
        ax=axes[2],
    )
    axes[2].axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Mean RMIA score by group")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Mean RMIA score")
    axes[2].tick_params(axis="x", rotation=15)
    axes[2].legend(title="", fontsize=8)

    fig.suptitle(
        f"FT eval: members = fine-tuning data  |  {CONTEXT_RMIA_LABELS.get(context_rmia_subdir, context_rmia_subdir)}: members = context window  |  Non-members: shared",
        fontsize=9, y=1.01,
    )
    savefig(fig, output_dir, "11_experiment_comparison.png")



def plot_finetuned_target_audit_comparison(
    run_dir: Path,
    rmia_df: pd.DataFrame,
    output_dir: Path,
    context_rmia_subdir: str,
) -> None:
    """Compare finetuned-target RMIA when auditing fine-tune data vs context data."""
    ctx = _load_context_eval(run_dir, context_rmia_subdir)
    if rmia_df.empty or not ctx:
        return
    ctx_path = run_dir / "rmia" / ctx["subdir"] / "context_member_rmia_features.csv"

    ft = add_rmia_features(rmia_df)
    if "member_bool" not in ft.columns or "rmia_finetuned" not in ft.columns:
        return
    ctx_df = ctx["df"].copy()
    if "rmia_finetuned" not in ctx_df.columns:
        return

    ft_members = ft.loc[ft["member_bool"], ["rmia_finetuned"]].copy()
    ft_members["audit_group"] = "Fine-tune audit members"
    ft_members["source_id"] = 1

    context_members = ctx_df[["rmia_finetuned"]].copy()
    context_members["audit_group"] = "Context audit members"
    context_members["source_id"] = 2

    nonmembers = ft.loc[~ft["member_bool"], ["rmia_finetuned"]].copy()
    nonmembers["audit_group"] = "Held-out nonmembers"
    nonmembers["source_id"] = 0

    combined = pd.concat([ft_members, context_members, nonmembers], ignore_index=True)
    group_order = ["Fine-tune audit members", "Context audit members", "Held-out nonmembers"]
    group_palette = {
        "Fine-tune audit members": "#0072b2",
        "Context audit members": "#cc79a7",
        "Held-out nonmembers": "#4f5d75",
    }
    combined["audit_group"] = pd.Categorical(combined["audit_group"], categories=group_order, ordered=True)
    combined["context_rmia_subdir"] = ctx["subdir"]
    combined["context_reference_mode"] = ctx.get(
        "attack_reference_mode",
        CONTEXT_REFERENCE_MODES.get(context_rmia_subdir, ""),
    )
    combined.to_csv(output_dir / "finetuned_target_audit_comparison_scores.csv", index=False)

    def _roc_for_positive(pos_group: str) -> tuple[np.ndarray, np.ndarray, float, float, str]:
        subset = combined[combined["audit_group"].isin([pos_group, "Held-out nonmembers"])]
        y = (subset["audit_group"] == pos_group).astype(int).to_numpy()
        scores = subset["rmia_finetuned"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y, scores)
        raw_auc = sk_auc(fpr, tpr)
        if raw_auc >= 0.5:
            return fpr, tpr, raw_auc, raw_auc, "higher = audit member"
        inv_fpr, inv_tpr, _ = roc_curve(y, -scores)
        return inv_fpr, inv_tpr, raw_auc, sk_auc(inv_fpr, inv_tpr), "lower = audit member"

    ft_fpr, ft_tpr, ft_raw_auc, ft_best_auc, ft_direction = _roc_for_positive("Fine-tune audit members")
    ctx_fpr, ctx_tpr, ctx_raw_auc, ctx_best_auc, ctx_direction = _roc_for_positive("Context audit members")

    source_subset = combined[combined["audit_group"].isin(["Fine-tune audit members", "Context audit members"])]
    y_source = (source_subset["audit_group"] == "Fine-tune audit members").astype(int).to_numpy()
    source_scores = source_subset["rmia_finetuned"].to_numpy(dtype=float)
    src_fpr, src_tpr, _ = roc_curve(y_source, source_scores)
    source_auc = sk_auc(src_fpr, src_tpr)

    summary = (
        combined.groupby("audit_group", observed=True)["rmia_finetuned"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reindex(group_order)
        .reset_index()
    )
    extra = pd.DataFrame([
        {
            "comparison": "fine_tune_members_vs_nonmembers",
            "raw_auc": ft_raw_auc,
            "best_auc": ft_best_auc,
            "direction": ft_direction,
        },
        {
            "comparison": "context_members_vs_nonmembers",
            "raw_auc": ctx_raw_auc,
            "best_auc": ctx_best_auc,
            "direction": ctx_direction,
        },
        {
            "comparison": "fine_tune_members_vs_context_members",
            "raw_auc": source_auc,
            "best_auc": max(source_auc, 1.0 - source_auc),
            "direction": "higher = fine-tune" if source_auc >= 0.5 else "lower = fine-tune",
        },
    ])
    summary.to_csv(output_dir / "finetuned_target_audit_comparison_summary.csv", index=False)
    extra.to_csv(output_dir / "finetuned_target_audit_comparison_auc.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    for group in group_order:
        values = combined.loc[combined["audit_group"] == group, "rmia_finetuned"].dropna()
        sns.kdeplot(
            x=values,
            ax=ax,
            color=group_palette[group],
            linewidth=2,
            fill=False,
            clip=(0, 1),
            label=f"{group} (median={values.median():.3f})",
        )
    ax.axvline(0.5, color="#333333", linestyle=":", linewidth=1)
    ax.set_title("Finetuned target RMIA distributions")
    ax.set_xlabel("rmia_finetuned")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.legend(title="")

    ax = axes[0, 1]
    sns.boxplot(
        data=combined,
        x="audit_group",
        y="rmia_finetuned",
        hue="audit_group",
        order=group_order,
        palette=group_palette,
        showfliers=False,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=combined.sample(min(len(combined), 5000), random_state=23),
        x="audit_group",
        y="rmia_finetuned",
        order=group_order,
        color="#222222",
        alpha=0.12,
        size=1.2,
        ax=ax,
    )
    ax.axhline(0.5, color="#333333", linestyle=":", linewidth=1)
    ax.set_title("Score spread by audit source")
    ax.set_xlabel("")
    ax.set_ylabel("rmia_finetuned")
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1, 0]
    ax.plot(ft_fpr, ft_tpr, color=group_palette["Fine-tune audit members"], linewidth=2,
            label=f"Fine-tune audit vs nonmembers: AUC={ft_raw_auc:.3f}")
    ax.plot(ctx_fpr, ctx_tpr, color=group_palette["Context audit members"], linewidth=2,
            label=f"Context audit vs nonmembers: AUC={ctx_raw_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#aaaaaa", linestyle=":", linewidth=1)
    ax.set_title("Membership ROC using finetuned target RMIA")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1, 1]
    mean_df = summary[["audit_group", "mean", "median"]].melt(
        id_vars="audit_group",
        var_name="stat",
        value_name="rmia_finetuned",
    )
    sns.barplot(
        data=mean_df,
        x="audit_group",
        y="rmia_finetuned",
        hue="stat",
        order=group_order,
        palette={"mean": "#8ecae6", "median": "#ffb703"},
        ax=ax,
    )
    ax.axhline(0.5, color="#333333", linestyle=":", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Audit-source separation: FT vs context AUC={source_auc:.3f}")
    ax.set_xlabel("")
    ax.set_ylabel("rmia_finetuned")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="")

    fig.suptitle(
        f"Finetuned target RMIA: fine-tune-data audit vs context-data audit ({ctx['subdir']})",
        fontsize=12,
        y=1.01,
    )
    savefig(fig, output_dir, "13_finetuned_target_audit_comparison.png")


def plot_rmia_signal_diagnostics(rmia_df: pd.DataFrame, output_dir: Path) -> None:
    if rmia_df.empty:
        return

    df = add_rmia_features(rmia_df)
    ref_ft_col = next(
        (c for c in ["reference_signal_finetuned_target_eval", "reference_signal_finetuned"] if c in df.columns),
        None,
    )
    base_needed = {"target_signal_original", "reference_signal_original", "target_signal_finetuned"}
    if not base_needed.issubset(df.columns) or ref_ft_col is None:
        return

    # Collect signal columns: original and finetuned variants (handles _target_eval suffix)
    ft_suffixes = ("_finetuned_target_eval", "_finetuned")
    orig_suffixes = ("_original",)
    signal_cols = [c for c in df.columns if any(c.endswith(s) for s in orig_suffixes + ft_suffixes)
                   and any(c.startswith(p) for p in ("target_", "paired_", "reference_"))]
    signal_cols = list(dict.fromkeys(signal_cols))

    def _is_finetuned(col: str) -> bool:
        return any(col.endswith(s) for s in ft_suffixes)

    def _col_role(col: str) -> str:
        for s in ft_suffixes + orig_suffixes:
            col = col.replace(s, "")
        return col.replace("_signal", "").strip("_")

    signal_long = df.melt(
        id_vars=["membership_group"],
        value_vars=signal_cols,
        var_name="signal",
        value_name="true_label_probability",
    )
    signal_long["model"] = np.where(signal_long["signal"].apply(_is_finetuned), "finetuned", "original")
    signal_long["column"] = signal_long["signal"].apply(_col_role)

    ratio_cols = [
        "original_target_minus_reference",
        "finetuned_target_minus_reference",
        "original_target_reference_ratio",
        "finetuned_target_reference_ratio",
    ]
    ratio_summary = df.groupby("membership_group")[ratio_cols].agg(["mean", "median"]).reset_index()
    ratio_summary.columns = [
        "_".join([str(part) for part in col if part]).strip("_")
        if isinstance(col, tuple)
        else col
        for col in ratio_summary.columns
    ]
    ratio_summary.to_csv(output_dir / "rmia_signal_ratio_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    sns.boxplot(
        data=signal_long,
        x="column",
        y="true_label_probability",
        hue="model",
        palette={"original": PALETTE["original"], "finetuned": PALETTE["finetuned"]},
        showfliers=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("RMIA signal columns")
    axes[0, 0].set_xlabel("Column role")
    axes[0, 0].set_ylabel("True-label probability")
    axes[0, 0].tick_params(axis="x", rotation=20)

    sns.boxplot(
        data=signal_long,
        x="column",
        y="true_label_probability",
        hue="membership_group",
        palette={"Member": PALETTE["member"], "Non-member": PALETTE["nonmember"]},
        showfliers=False,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Signal columns by membership")
    axes[0, 1].set_xlabel("Column role")
    axes[0, 1].set_ylabel("True-label probability")
    axes[0, 1].tick_params(axis="x", rotation=20)

    ratio_long = df.melt(
        id_vars=["membership_group"],
        value_vars=["original_target_reference_ratio", "finetuned_target_reference_ratio"],
        var_name="model",
        value_name="target_reference_ratio",
    )
    ratio_long["model"] = ratio_long["model"].str.replace("_target_reference_ratio", "", regex=False)
    sns.kdeplot(
        data=ratio_long,
        x="target_reference_ratio",
        hue="model",
        common_norm=False,
        fill=False,
        palette={"original": PALETTE["original"], "finetuned": PALETTE["finetuned"]},
        ax=axes[1, 0],
    )
    axes[1, 0].axvline(1.0, color="#333333", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Target/reference signal ratio")
    axes[1, 0].set_xlabel("Target signal / mean reference signal")

    diff_long = df.melt(
        id_vars=["membership_group"],
        value_vars=["original_target_minus_reference", "finetuned_target_minus_reference"],
        var_name="model",
        value_name="target_minus_reference",
    )
    diff_long["model"] = diff_long["model"].str.replace("_target_minus_reference", "", regex=False)
    sns.violinplot(
        data=diff_long,
        x="membership_group",
        y="target_minus_reference",
        hue="model",
        palette={"original": PALETTE["original"], "finetuned": PALETTE["finetuned"]},
        cut=0,
        inner="quartile",
        ax=axes[1, 1],
    )
    axes[1, 1].axhline(0, color="#333333", linewidth=1)
    axes[1, 1].set_title("Target minus mean reference signal")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Probability difference")

    savefig(fig, output_dir, "10_rmia_signal_diagnostics.png")


def plot_ft_data_rmia(run_dir: Path, rmia_df: pd.DataFrame, output_dir: Path, context_rmia_subdir: str) -> None:
    """RMIA results for the fine-tuning data audit (members = fine-tuned data).

    Shows the original-model attack and the fine-tuned-model attack side by side:
    ROC curves, score distributions by membership, and key metrics table.
    Data comes from eval_rmia_ft_tabdpt.py (rmia_scores.csv).
    """
    if rmia_df.empty:
        return

    df = add_rmia_features(rmia_df)
    if "membership_group" not in df.columns:
        return

    rmia_dir = run_dir / "rmia"
    ft_summary_key, ft_report_dir = CONTEXT_TO_FT_VARIANT.get(
        context_rmia_subdir, ("finetuned_model_rmia_original_refs", "report_finetuned_original_refs")
    )
    original_curve = load_attack_curve(rmia_dir / "report_original" / "exp" / "attack_result_0.npz")
    finetuned_curve = load_attack_curve(rmia_dir / ft_report_dir / "exp" / "attack_result_0.npz")
    summary = load_json(rmia_dir / "summary.json") if (rmia_dir / "summary.json").exists() else {}

    # Key metrics
    metrics = {}
    for label, curve, skey in [
        ("original", original_curve, "original_model_rmia"),
        ("finetuned", finetuned_curve, ft_summary_key),
    ]:
        if curve is not None:
            metrics[label] = {
                "auc": curve["auc"],
                "tpr@0.1%fpr": float(summary.get(skey, {}).get("tpr@0.1%fpr", np.nan)),
            }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel (0,0): ROC curves
    ax = axes[0, 0]
    for label, curve, color in [
        ("Original model (orig. refs)", original_curve, PALETTE["original"]),
        ("FT model (FT refs)", finetuned_curve, PALETTE["finetuned"]),
    ]:
        if curve is None:
            continue
        ax.plot(curve["fpr"], curve["tpr"], color=color, linewidth=2,
                label=f"{label}  AUC={curve['auc']:.4f}")
    ax.plot([0, 1], [0, 1], color="#aaaaaa", linestyle=":", linewidth=1)
    ax.set_title("RMIA ROC — fine-tuning data members")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=9)

    # Panel (0,1): Score distributions — members vs non-members, both attacks
    ax = axes[0, 1]
    for score_col, color, label in [
        ("rmia_original", PALETTE["original"], "Original model"),
        ("rmia_finetuned", PALETTE["finetuned"], "FT model"),
    ]:
        if score_col not in df.columns:
            continue
        for grp, ls in [("Member", "-"), ("Non-member", "--")]:
            subset = df.loc[df["membership_group"] == grp, score_col].dropna()
            sns.kdeplot(x=subset, ax=ax, color=color, linestyle=ls, linewidth=1.6,
                        common_norm=False, fill=False, clip=(0, 1),
                        label=f"{label}, {grp}")
    ax.axvline(0.5, color="#333333", linestyle=":", linewidth=1)
    ax.set_title("RMIA score distributions")
    ax.set_xlabel("RMIA score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(0, 1)

    # Panel (1,0): Mean RMIA scores — original vs FT model, by membership
    mean_data = (
        df.groupby("membership_group")[["rmia_original", "rmia_finetuned"]]
        .mean()
        .reset_index()
        .melt("membership_group", var_name="attack", value_name="mean_rmia_score")
    )
    mean_data["attack"] = mean_data["attack"].map(
        {"rmia_original": "Original model", "rmia_finetuned": "FT model"}
    )
    sns.barplot(
        data=mean_data,
        x="membership_group",
        y="mean_rmia_score",
        hue="attack",
        palette={"Original model": PALETTE["original"], "FT model": PALETTE["finetuned"]},
        order=["Member", "Non-member"],
        ax=axes[1, 0],
    )
    axes[1, 0].axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Mean RMIA score by membership")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Mean RMIA score")
    axes[1, 0].legend(title="", fontsize=9)

    # Panel (1,1): Scatter — original RMIA score vs FT RMIA score, coloured by membership
    sample = df.sample(min(len(df), 6000), random_state=42)
    sns.scatterplot(
        data=sample,
        x="rmia_original",
        y="rmia_finetuned",
        hue="membership_group",
        palette={"Member": PALETTE["member"], "Non-member": PALETTE["nonmember"]},
        alpha=0.45,
        linewidth=0,
        s=10,
        ax=axes[1, 1],
    )
    axes[1, 1].plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Original vs FT model RMIA score per record")
    axes[1, 1].set_xlabel("RMIA score — original model")
    axes[1, 1].set_ylabel("RMIA score — FT model")
    axes[1, 1].legend(title="", fontsize=9)

    ref_mode = CONTEXT_REFERENCE_MODES.get(context_rmia_subdir, "")
    fig.suptitle(
        f"Fine-tuning data RMIA  |  members = fine-tuned data, non-members = held-out  |  ref mode: {ref_mode}",
        fontsize=10, y=1.01,
    )
    savefig(fig, output_dir, "12_ft_data_rmia.png")

    # Save metrics CSV
    rows = []
    for label, result in [
        ("original", summary.get("original_model_rmia", {})),
        ("finetuned", summary.get(ft_summary_key, {})),
    ]:
        for metric in ["auc", "tpr@0.1%fpr", "tpr@0%fpr", "tnr@0.1%fnr", "tnr@0%fnr"]:
            rows.append({"model": label, "metric": metric, "value": result.get(metric, np.nan)})
    pd.DataFrame(rows).to_csv(output_dir / "ft_data_rmia_metrics.csv", index=False)


def plot_tail_saturation(rmia_df: pd.DataFrame, output_dir: Path) -> dict:
    """Highlight the confidence-shift mechanism behind fine-tune membership leakage.

    Four panels:
      A — finetuned confidence distribution (log scale) to expose the non-saturated tail.
      B — per-sample confidence shift (finetuned − original) by membership.
      C — original vs finetuned confidence scatter, showing which records escape saturation.
      D — RMIA AUC stratified by saturation status: all / saturated / tail.
    """
    from sklearn.metrics import roc_auc_score

    if rmia_df.empty:
        return {}

    df = add_rmia_features(rmia_df).copy()
    required = {"membership_group", "member_bool", "target_signal_finetuned", "target_signal_original", "rmia_finetuned"}
    if not required.issubset(df.columns):
        return {}

    SAT_THRESH = 1.0 - 1e-6
    df["target_signal_delta"] = df["target_signal_finetuned"] - df["target_signal_original"]
    df["is_saturated"] = df["target_signal_finetuned"] >= SAT_THRESH

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: finetuned confidence histogram, log y-scale
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 51)
    for grp, color, alpha in [("Non-member", PALETTE["nonmember"], 0.6), ("Member", PALETTE["member"], 0.7)]:
        subset = df.loc[df["membership_group"] == grp, "target_signal_finetuned"]
        ax.hist(subset, bins=bins, color=color, alpha=alpha, label=grp, density=True)
    ax.set_yscale("log")
    ax.axvline(SAT_THRESH, color="#333333", linestyle="--", linewidth=1.2, label="Saturation (P=1)")
    ax.set_xlabel("P(y | x)  —  finetuned model")
    ax.set_ylabel("Density (log scale)")
    ax.set_title("A  —  Confidence distribution after fine-tuning")
    ax.legend(fontsize=8)

    # Panel B: KDE of confidence shift (finetuned - original)
    ax = axes[0, 1]
    for grp, color, ls in [("Non-member", PALETTE["nonmember"], "--"), ("Member", PALETTE["member"], "-")]:
        subset = df.loc[df["membership_group"] == grp, "target_signal_delta"].dropna()
        if len(subset) > 10:
            sns.kdeplot(x=subset, ax=ax, color=color, linestyle=ls, linewidth=2,
                        fill=False, common_norm=False, label=grp)
    ax.axvline(0, color="#333333", linestyle=":", linewidth=1)
    ax.set_xlabel("ΔP(y | x)  =  finetuned − original")
    ax.set_ylabel("Density")
    ax.set_title("B  —  Per-sample confidence shift by membership")
    ax.legend(fontsize=8)

    # Panel C: scatter original vs finetuned confidence, colored by membership
    ax = axes[1, 0]
    sample = df.sample(min(len(df), 5000), random_state=42)
    for grp, color in [("Non-member", PALETTE["nonmember"]), ("Member", PALETTE["member"])]:
        sub = sample[sample["membership_group"] == grp]
        ax.scatter(sub["target_signal_original"], sub["target_signal_finetuned"],
                   color=color, alpha=0.25, s=7, linewidths=0, label=grp)
    ax.axhline(SAT_THRESH, color="#333333", linestyle="--", linewidth=1.2, label="Saturation threshold")
    ax.plot([0, 1], [0, 1], color="#aaaaaa", linestyle=":", linewidth=1)
    ax.set_xlabel("P(y | x)  —  original model")
    ax.set_ylabel("P(y | x)  —  finetuned model")
    ax.set_title("C  —  Original vs finetuned confidence per record")
    ax.legend(fontsize=8)

    # Panel D: RMIA AUC stratified by saturation status
    ax = axes[1, 1]
    strata = [
        ("All records", pd.Series(True, index=df.index)),
        ("Saturated  (P=1.0)", df["is_saturated"]),
        ("Tail  (P<1.0)", ~df["is_saturated"]),
    ]
    rows_auc = []
    for label, mask in strata:
        sub = df[mask]
        n_pos = int(sub["member_bool"].sum())
        n_neg = int((~sub["member_bool"]).sum())
        try:
            auc_val = roc_auc_score(sub["member_bool"].astype(int), sub["rmia_finetuned"]) if n_pos > 0 and n_neg > 0 else np.nan
        except Exception:
            auc_val = np.nan
        rows_auc.append({"label": label, "auc": auc_val, "n": len(sub), "n_member": n_pos, "n_nonmember": n_neg})

    auc_df = pd.DataFrame(rows_auc)
    bar_colors = [PALETTE["original"], "#888888", PALETTE["finetuned"]]
    bars = ax.bar(auc_df["label"], auc_df["auc"], color=bar_colors, alpha=0.85, width=0.5)
    ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    for bar, row in zip(bars, rows_auc):
        if not np.isnan(row["auc"]):
            ax.text(bar.get_x() + bar.get_width() / 2, row["auc"] + 0.02,
                    f"AUC={row['auc']:.3f}\nn={row['n']}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("RMIA AUC (finetuned model)")
    ax.set_title("D  —  RMIA AUC by saturation group")
    ax.tick_params(axis="x", labelsize=8)

    sat_frac = df["is_saturated"].mean()
    fig.suptitle(
        f"Tail saturation  |  {sat_frac:.1%} of records saturated to P≈1.0  |  "
        f"leakage driven by non-saturated tail ({1 - sat_frac:.1%} of records)",
        fontsize=10, y=1.01,
    )
    savefig(fig, output_dir, "14_tail_saturation.png")

    summary_rows = []
    for grp in ["Member", "Non-member"]:
        sub = df[df["membership_group"] == grp]
        summary_rows.append({
            "group": grp,
            "n": len(sub),
            "frac_saturated": float(sub["is_saturated"].mean()),
            "mean_signal_finetuned": float(sub["target_signal_finetuned"].mean()),
            "mean_signal_original": float(sub["target_signal_original"].mean()),
            "mean_signal_delta": float(sub["target_signal_delta"].mean()),
            "std_signal_delta": float(sub["target_signal_delta"].std()),
        })
    pd.DataFrame(summary_rows).to_csv(output_dir / "tail_saturation_summary.csv", index=False)

    return {
        "saturation_fraction": float(sat_frac),
        "tail_fraction": float(1 - sat_frac),
        **{f"auc_{r['label'].split()[0].lower()}": float(r["auc"]) if not np.isnan(r["auc"]) else None for r in rows_auc},
    }


def write_extreme_examples(scores: pd.DataFrame, output_dir: Path, n: int) -> None:
    cols = [
        c
        for c in [
            "idx_local",
            "idx_original",
            "member",
            "y",
            "original_pred",
            "finetuned_pred",
            "original_true_label_score",
            "finetuned_true_label_score",
            "prob_delta",
            "score_delta",
            "original_correct",
            "finetuned_correct",
        ]
        if c in scores.columns
    ]
    top_up = scores.nlargest(n, "score_delta")[cols]
    top_down = scores.nsmallest(n, "score_delta")[cols]
    top_abs = scores.reindex(scores["score_delta"].abs().sort_values(ascending=False).head(n).index)[cols]
    top_up.to_csv(output_dir / "top_positive_score_delta_examples.csv", index=False)
    top_down.to_csv(output_dir / "top_negative_score_delta_examples.csv", index=False)
    top_abs.to_csv(output_dir / "top_absolute_score_delta_examples.csv", index=False)


def visualize_run(run_dir: Path, output_base: Path, top_n: int, context_rmia_subdir: str, generate_shared: bool = True) -> Path:
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    result = load_json(run_dir / "result.json")
    dataset = result.get("dataset", run_dir.name)
    shared_dir = (output_base / safe_filename(dataset)).resolve()
    rmia_dir = (shared_dir / context_rmia_subdir).resolve()
    shared_dir.mkdir(parents=True, exist_ok=True)
    rmia_dir.mkdir(parents=True, exist_ok=True)

    if generate_shared:
        original_perf = load_json(run_dir / "original_performance.json")
        finetuned_perf = load_json(run_dir / "finetuned_performance.json")
        model_change = load_json(run_dir / "model_change.json")
        attention_change = load_json(run_dir / "attention_change.json")
        scores = add_membership_features(read_csv_if_exists(run_dir / "membership_scores.csv"))
        attention_summary = read_csv_if_exists(run_dir / "membership_attention_summary.csv")
        layers = read_csv_if_exists(run_dir / "membership_attention_layers.csv")
        heads = read_csv_if_exists(run_dir / "membership_attention_layer_heads.csv")
        test_predictions = read_csv_if_exists(run_dir / "test_predictions.csv")

        plot_performance_summary(original_perf, finetuned_perf, model_change, attention_change, shared_dir)
        if not scores.empty:
            plot_membership_distributions(scores, shared_dir)
            plot_membership_roc(scores, shared_dir)
            plot_prediction_transitions(scores, test_predictions, shared_dir)
            write_extreme_examples(scores, shared_dir, top_n)
        plot_attention_summary(attention_summary, shared_dir)
        plot_layer_attention(layers, shared_dir)
        plot_head_attention(heads, shared_dir)

    _rmia_raw = read_csv_if_exists(run_dir / "rmia" / "rmia_scores.csv")
    rmia_scores = normalize_rmia_scores(_rmia_raw, context_rmia_subdir) if not _rmia_raw.empty else _rmia_raw
    ctx = _load_context_eval(run_dir, context_rmia_subdir)
    if not rmia_scores.empty:
        plot_rmia_summary(run_dir, rmia_scores, rmia_dir, context_rmia_subdir)
        plot_rmia_score_distributions(rmia_scores, rmia_dir)
        plot_rmia_signal_diagnostics(rmia_scores, rmia_dir)
        plot_experiment_comparison(rmia_scores, ctx, rmia_dir, context_rmia_subdir)
        plot_ft_data_rmia(run_dir, rmia_scores, rmia_dir, context_rmia_subdir)
        plot_tail_saturation(rmia_scores, rmia_dir)
        plot_finetuning_composite(run_dir, rmia_scores, rmia_dir)
        plot_finetuned_target_audit_comparison(run_dir, rmia_scores, rmia_dir, context_rmia_subdir)

    return rmia_dir


def plot_dataset_context_attack_comparison(
    dataset: str,
    logs_base: Path,
    output_dir: Path,
) -> None:
    """Per-dataset grouped bar: RMIA original/finetuned and AMIA original/finetuned on context members."""
    import json as _json
    from sklearn.metrics import roc_auc_score as _roc_auc

    run_dir = logs_base / dataset
    scores_path = run_dir / "rmia" / "report_context_original_refs" / "context_vs_nonmember_auc_scores.csv"
    if not scores_path.exists():
        print(f"  [context attack plot] missing RMIA context scores for {dataset}, skipping.")
        return

    scores = pd.read_csv(scores_path)
    rmia_orig = _roc_auc(scores["membership"], scores["rmia_original"])
    rmia_ft   = _roc_auc(scores["membership"], scores["rmia_finetuned"])

    amia_orig = amia_ft = float("nan")
    amia_json = run_dir / "amia_context" / "context_amia_auc.json"
    if amia_json.exists():
        with amia_json.open() as f:
            amia_data = _json.load(f)
        amia_orig = amia_data.get("auc_original", float("nan"))
        amia_ft   = amia_data.get("auc_finetuned", float("nan"))

    attacks = [
        ("RMIA (original)",   rmia_orig, PALETTE["original"], ""),
        ("RMIA (fine-tuned)", rmia_ft,   PALETTE["original"], "///"),
        ("AMIA (original)",   amia_orig, "#b23a48",            ""),
        ("AMIA (fine-tuned)", amia_ft,   "#b23a48",            "///"),
    ]

    visible = [(label, val, color, hatch) for (label, val, color, hatch) in attacks if not np.isnan(val)]
    n = len(visible)
    bar_w = 0.055
    gap = 0.02
    x = np.array([0.0])
    offsets = np.linspace(-(n - 1) * (bar_w + gap) / 2, (n - 1) * (bar_w + gap) / 2, n)

    with plt.rc_context({"font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12,
                          "ytick.labelsize": 12, "legend.fontsize": 11}):
        fig, ax = plt.subplots(figsize=(5, 4.5))
        for (label, val, color, hatch), offset in zip(visible, offsets):
            ax.bar(x + offset, [val], width=bar_w,
                   color=color, alpha=0.80, hatch=hatch,
                   edgecolor="black", linewidth=0.5, label=label)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, alpha=0.75)
        half_span = (n - 1) * (bar_w + gap) / 2 + bar_w / 2
        ax.set_xlim(-half_span - 0.12, half_span + 0.12)
        ax.set_xticks([])
        ax.set_ylabel("MIA AUC", fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(frameon=True, loc="upper left")
        fig.tight_layout()
        out = output_dir / "context_attack_comparison.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  [{dataset}] Saved context attack comparison to: {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset names to visualize (default: all in DATASETS).",
    )
    parser.add_argument(
        "--logs-base",
        type=Path,
        default=DEFAULT_LOGS_BASE,
        help="Base directory containing per-dataset ft_tabdpt experiment folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="Base directory where plots and summaries will be saved.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of largest shifted examples to save in each diagnostic CSV.",
    )
    parser.add_argument(
        "--context-rmia-subdir",
        choices=[*CONTEXT_RMIA_SUBDIRS, "both"],
        default="both",
        help="Context RMIA folder to visualize. Default creates separate folders for both modes.",
    )
    return parser.parse_args()


def main() -> None:
    setup_style()
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",")] if args.datasets else DATASETS
    context_subdirs = CONTEXT_RMIA_SUBDIRS if args.context_rmia_subdir == "both" else (args.context_rmia_subdir,)
    for dataset in datasets:
        run_dir = args.logs_base / dataset
        for i, context_rmia_subdir in enumerate(context_subdirs):
            output_dir = visualize_run(run_dir, args.output_dir, args.top_n, context_rmia_subdir, generate_shared=(i == 0))
            print(f"[{dataset}] Saved ft_tabdpt visualizations ({context_rmia_subdir}) to: {output_dir}")

    for dataset in datasets:
        per_ds_out = args.output_dir / dataset
        per_ds_out.mkdir(parents=True, exist_ok=True)
        plot_dataset_context_attack_comparison(dataset, args.logs_base, per_ds_out)


if __name__ == "__main__":
    main()
