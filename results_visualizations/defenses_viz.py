#!/usr/bin/env python3
"""Visualize TabFM defense results.

The comparison set is intentionally defined by the defenses present on the MIC
(46980_MIC) dataset. Other datasets may contain additional exploratory defenses;
those are filtered out so every plot stays comparable.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from textwrap import shorten

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

MIC_DATASET = "46980_MIC"
DEFAULT_DATASETS = (
    "46980_MIC",
    "credit_rating",
    "dropout_success",
    "locations",
    "url",
    "us_stocks_financial",
)
DEFAULT_MODELS = ("tabpfn", "real-tabpfn", "tabicl", "tabdpt")
METRIC_COLUMNS = {
    "rmia_auc": "RMIA AUC",
    "amia_row_max_auc": "AMIA row-max AUC",
    "accuracy": "Accuracy",
}
MODEL_LABELS = {
    "tabpfn": "TabPFN",
    "real-tabpfn": "Real TabPFN",
    "tabicl": "TabICL",
    "tabdpt": "TabDPT",
}
MODEL_ORDER = list(DEFAULT_MODELS)
MODEL_LABEL_ORDER = [MODEL_LABELS[m] for m in MODEL_ORDER]
FAMILY_ORDER = [
    "No defense",
    "Label k-anon",
    "Attention dropout",
    "Label k-anon + dropout",
    "High-risk label k-anon",
    "Other",
]
PALETTE = {
    "No defense": "#4d4d4d",
    "Label k-anon": "#0072b2",
    "Attention dropout": "#d55e00",
    "Label k-anon + dropout": "#cc79a7",
    "High-risk label k-anon": "#009e73",
    "Other": "#999999",
}
PARAM_EVOLUTION_DATASETS = ("locations", "dropout_success")


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 240,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def canonical_defense(name: str) -> str:
    """Normalize model-specific layer lists while preserving defense strength."""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    if name in {"", "none", "no_defense"}:
        return "no_defense"
    # Layer ids differ by model, so compare by dropout probability only.
    name = re.sub(r"attn_drop_p(\d+)_l[0-9,]+", r"attn_drop_p\1", name)
    return name


def defense_family(defense_key: str) -> str:
    if defense_key == "no_defense":
        return "No defense"
    has_label = defense_key.startswith("label_kanon")
    has_attn = "attn_drop" in defense_key
    if defense_key.startswith("highrisk_label_kanon"):
        return "High-risk label k-anon"
    if has_label and has_attn:
        return "Label k-anon + dropout"
    if has_label:
        return "Label k-anon"
    if has_attn:
        return "Attention dropout"
    return "Other"


def pretty_defense(defense_key: str) -> str:
    if defense_key == "no_defense":
        return "No defense"
    label = defense_key
    label = label.replace("highrisk_label_kanon", "high-risk label k-anon")
    label = label.replace("label_kanon", "label k-anon")
    label = label.replace("attn_drop", "attn drop")
    label = label.replace("_", " ")
    label = label.replace("+", " + ")
    return label


def short_defense(defense_key: str, width: int = 34) -> str:
    return shorten(pretty_defense(defense_key), width=width, placeholder="...")


def set_family_legend(ax) -> None:
    """Keep only defense-family legend entries and drop seaborn size legend rows."""
    handles, labels = ax.get_legend_handles_labels()
    by_label = {label: handle for handle, label in zip(handles, labels) if label in FAMILY_ORDER}
    ordered_labels = [label for label in FAMILY_ORDER if label in by_label]
    ax.legend(
        [by_label[label] for label in ordered_labels],
        ordered_labels,
        title="Family",
        loc="best",
        frameon=True,
    )


def dataset_from_path(path: Path, logs_dir: Path) -> str:
    return path.relative_to(logs_dir).parts[0]


def model_from_path(path: Path, logs_dir: Path) -> str:
    return path.relative_to(logs_dir).parts[1]


def load_defense_results(logs_dir: Path, models: tuple[str, ...]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for csv_path in sorted(logs_dir.glob("*/*/defense/defense_eval_results.csv")):
        model = model_from_path(csv_path, logs_dir)
        if model not in models:
            continue
        dataset = dataset_from_path(csv_path, logs_dir)
        try:
            frame = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - defensive logging path
            print(f"Skipping {csv_path}: {exc}")
            continue
        frame["dataset"] = dataset
        frame["model"] = model
        frame["source_csv"] = str(csv_path)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    for col in METRIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["seed"] = pd.to_numeric(df.get("seed", 1), errors="coerce").fillna(1).astype(int)
    df["defense_raw"] = df["defense"].astype(str)
    df["defense_key"] = df["defense_raw"].map(canonical_defense)
    df["defense_label"] = df["defense_key"].map(pretty_defense)
    df["defense_short"] = df["defense_key"].map(short_defense)
    df["family"] = df["defense_key"].map(defense_family)
    df["model_label"] = df["model"].map(lambda x: MODEL_LABELS.get(x, x))
    return df


def mic_defense_set(df: pd.DataFrame, mic_dataset: str) -> list[str]:
    mic = df[df["dataset"] == mic_dataset]
    if mic.empty:
        raise SystemExit(f"No defense_eval_results.csv rows found for {mic_dataset}.")
    allowed = sorted(mic["defense_key"].dropna().unique())
    if "no_defense" in allowed:
        allowed.remove("no_defense")
        allowed.insert(0, "no_defense")
    return allowed


def add_baseline_deltas(df: pd.DataFrame) -> pd.DataFrame:
    base = (
        df[df["defense_key"] == "no_defense"]
        .groupby(["dataset", "model", "seed"], as_index=False)[list(METRIC_COLUMNS)]
        .mean()
        .rename(columns={
            "rmia_auc": "baseline_rmia_auc",
            "amia_row_max_auc": "baseline_amia_row_max_auc",
            "accuracy": "baseline_accuracy",
        })
    )
    out = df.merge(base, on=["dataset", "model", "seed"], how="left")
    out["rmia_auc_reduction"] = out["baseline_rmia_auc"] - out["rmia_auc"]
    out["amia_auc_reduction"] = out["baseline_amia_row_max_auc"] - out["amia_row_max_auc"]
    out["accuracy_drop"] = out["baseline_accuracy"] - out["accuracy"]
    out["privacy_gain"] = out[["rmia_auc_reduction", "amia_auc_reduction"]].mean(axis=1, skipna=True)
    out["tradeoff_score"] = out["privacy_gain"] - out["accuracy_drop"].fillna(0)
    return out


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["defense_key", "defense_label", "defense_short", "family", "model", "model_label"], as_index=False)
        .agg(
            rmia_auc=("rmia_auc", "mean"),
            amia_row_max_auc=("amia_row_max_auc", "mean"),
            accuracy=("accuracy", "mean"),
            rmia_auc_reduction=("rmia_auc_reduction", "mean"),
            amia_auc_reduction=("amia_auc_reduction", "mean"),
            accuracy_drop=("accuracy_drop", "mean"),
            privacy_gain=("privacy_gain", "mean"),
            tradeoff_score=("tradeoff_score", "mean"),
            n_results=("tradeoff_score", "size"),
            n_datasets=("dataset", "nunique"),
        )
    )
    return agg.sort_values(["tradeoff_score", "privacy_gain"], ascending=False)


def ordered_defenses(summary: pd.DataFrame) -> list[str]:
    order = (
        summary.groupby("defense_key", as_index=False)
        .agg(tradeoff_score=("tradeoff_score", "mean"), privacy_gain=("privacy_gain", "mean"))
        .sort_values(["tradeoff_score", "privacy_gain"], ascending=False)
        ["defense_key"].tolist()
    )
    if "no_defense" in order:
        order.remove("no_defense")
        order.insert(0, "no_defense")
    return order


def plot_privacy_utility(df: pd.DataFrame, out_dir: Path, datasets: tuple[str, ...]) -> None:
    """Per-defense privacy-utility tradeoff averaged across dataset-level means."""
    dataset_means = (
        df[df["dataset"].isin(datasets)]
        .groupby(["dataset", "defense_key", "defense_label", "defense_short", "family"], as_index=False)
        .agg(
            rmia_auc=("rmia_auc", "mean"),
            amia_row_max_auc=("amia_row_max_auc", "mean"),
            accuracy=("accuracy", "mean"),
            tradeoff_score=("tradeoff_score", "mean"),
            n_models=("model", "nunique"),
        )
    )
    if dataset_means.empty:
        return

    plot_df = (
        dataset_means.groupby(["defense_key", "defense_label", "defense_short", "family"], as_index=False)
        .agg(
            rmia_auc=("rmia_auc", "mean"),
            amia_row_max_auc=("amia_row_max_auc", "mean"),
            accuracy=("accuracy", "mean"),
            tradeoff_score=("tradeoff_score", "mean"),
            rmia_auc_std=("rmia_auc", "std"),
            amia_row_max_auc_std=("amia_row_max_auc", "std"),
            accuracy_std=("accuracy", "std"),
            n_datasets=("dataset", "nunique"),
            n_dataset_defense_points=("dataset", "size"),
        )
        .sort_values("tradeoff_score", ascending=False)
    )
    dataset_means.to_csv(out_dir / "defenses_01_dataset_means.csv", index=False)
    plot_df.to_csv(out_dir / "defenses_01_privacy_utility_tradeoff.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "amia_row_max_auc", "AMIA leakage vs accuracy"),
        (axes[1], "rmia_auc", "RMIA leakage vs accuracy"),
    ]:
        sns.scatterplot(
            data=plot_df,
            x="accuracy",
            y=metric,
            hue="family",
            hue_order=FAMILY_ORDER,
            palette=PALETTE,
            s=92,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
            ax=ax,
        )
        ax.axhline(0.5, color="#777777", lw=0.9, ls="--", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Average accuracy across datasets/models")
        ax.set_ylabel(METRIC_COLUMNS[metric])
        ax.grid(True, alpha=0.25)
        set_family_legend(ax)

    fig.suptitle("Defense privacy-utility tradeoff, averaged per defense across datasets", y=1.02, fontsize=13)
    fig.savefig(out_dir / "defenses_01_privacy_utility_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def plot_delta_tradeoff(summary_all_models: pd.DataFrame, out_dir: Path) -> None:
    plot_df = (
        summary_all_models.groupby(["defense_key", "defense_label", "defense_short", "family"], as_index=False)
        .agg(
            rmia_auc_reduction=("rmia_auc_reduction", "mean"),
            amia_auc_reduction=("amia_auc_reduction", "mean"),
            accuracy_drop=("accuracy_drop", "mean"),
            tradeoff_score=("tradeoff_score", "mean"),
            coverage=("n_results", "sum"),
        )
    )
    plot_df = plot_df[plot_df["defense_key"] != "no_defense"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "amia_auc_reduction", "AMIA Δ AUC vs Δ ACC"),
        (axes[1], "rmia_auc_reduction", "RMIA Δ AUC vs Δ ACC"),
    ]:
        sns.scatterplot(
            data=plot_df,
            x="accuracy_drop",
            y=metric,
            hue="family",
            palette=PALETTE,
            size="coverage",
            sizes=(45, 150),
            alpha=0.86,
            edgecolor="white",
            linewidth=0.6,
            ax=ax,
        )
        ax.axhline(0, color="#777777", lw=0.9, ls="--", alpha=0.7)
        ax.axvline(0, color="#777777", lw=0.9, ls="--", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Δ ACC")
        ax.set_ylabel("Δ AUC")
        ax.grid(True, alpha=0.25)
        set_family_legend(ax)
    fig.suptitle("Defense Δ tradeoff relative to no defense", y=1.02, fontsize=13)
    fig.savefig(out_dir / "defenses_01b_delta_tradeoff.png", bbox_inches="tight")
    plt.close(fig)



def plot_delta_auc_vs_target_accuracy(summary_all_models: pd.DataFrame, out_dir: Path) -> None:
    """Defense variants summarized across models."""
    families = {"Attention dropout", "Label k-anon", "High-risk label k-anon"}
    detail_df = summary_all_models[
        (summary_all_models["defense_key"] != "no_defense")
        & (summary_all_models["family"].isin(families))
    ].copy()
    detail_df = detail_df.dropna(subset=["accuracy_drop", "amia_auc_reduction", "rmia_auc_reduction"])
    if detail_df.empty:
        return

    label_parsed = detail_df["defense_key"].str.extract(r"(?:highrisk_)?label_kanon_k(?P<label_k>\d+)(?:_a(?P<label_a>\d+))?")
    drop_parsed = detail_df["defense_key"].str.extract(r"attn_drop_p(?P<dropout_p>\d+)")
    detail_df["k"] = pd.to_numeric(label_parsed["label_k"], errors="coerce")
    detail_df["alpha"] = label_parsed["label_a"].fillna("base").map(lambda x: "base" if x == "base" else f"a{x}")
    detail_df["dropout_p"] = pd.to_numeric(drop_parsed["dropout_p"], errors="coerce")
    detail_df["variant_order"] = detail_df["k"].fillna(detail_df["dropout_p"])
    detail_df["variant"] = np.where(
        detail_df["family"].eq("Attention dropout"),
        "p" + detail_df["dropout_p"].astype("Int64").astype(str),
        "k" + detail_df["k"].astype("Int64").astype(str) + " " + detail_df["alpha"],
    )

    plot_df = (
        detail_df.groupby(["defense_key", "defense_label", "family", "variant", "variant_order", "k", "alpha", "dropout_p"], as_index=False, dropna=False)
        .agg(
            accuracy_drop=("accuracy_drop", "mean"),
            amia_auc_reduction=("amia_auc_reduction", "mean"),
            rmia_auc_reduction=("rmia_auc_reduction", "mean"),
            tradeoff_score=("tradeoff_score", "mean"),
            n_models=("model", "nunique"),
        )
        .sort_values(["family", "variant_order", "alpha"])
    )
    plot_df.to_csv(out_dir / "defenses_01d_delta_auc_vs_target_accuracy.csv", index=False)

    family_styles = {
        "Attention dropout": {"color": PALETTE["Attention dropout"], "label": "Attention dropout"},
        "Label k-anon": {"color": PALETTE["Label k-anon"], "label": "Label k-anonymity"},
        "High-risk label k-anon": {"color": PALETTE["High-risk label k-anon"], "label": "Target label k-anonymity"},
    }
    family_order = [family for family in FAMILY_ORDER if family in set(plot_df["family"])]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.9), sharex=False, sharey=True)
    fig.subplots_adjust(bottom=0.24, wspace=0.08)
    panels = [
        (axes[0], "amia_auc_reduction", "AMIA"),
        (axes[1], "rmia_auc_reduction", "RMIA"),
    ]

    for ax, metric, title in panels:
        for family in family_order:
            series = plot_df[plot_df["family"] == family].sort_values(["variant_order", "alpha"])
            if series.empty:
                continue
            style = family_styles[family]
            ax.scatter(
                -series["accuracy_drop"],
                series[metric],
                s=135,
                marker="o",
                color=style["color"],
                edgecolor="white",
                linewidth=0.75,
                alpha=0.88,
                label=style["label"],
            )
        ax.axhline(0, color="#777777", lw=1.0, ls="--", alpha=0.72)
        ax.axvline(0, color="#777777", lw=1.0, ls="--", alpha=0.72)
        ax.set_title(title, fontsize=17, pad=10, fontweight="bold")
        ax.set_xlabel("Δ Target Accuracy", fontsize=15, fontweight="bold")
        ax.set_ylabel("Δ Attack AUC" if ax is axes[0] else "", fontsize=15, fontweight="bold")
        ax.tick_params(axis="both", labelsize=13)
        ax.tick_params(axis="x", labelbottom=True)
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.22)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#cfcfcf")
            spine.set_linewidth(1.0)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=family_styles[family]["color"],
            markeredgecolor="white",
            markersize=8,
            label=family_styles[family]["label"],
        )
        for family in family_order
    ]
    legend = fig.legend(
        handles=handles,
        title="Defense family",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(3, len(handles)),
        frameon=True,
        fontsize=13,
        title_fontsize=15,
    )
    legend.get_title().set_color("#222222")
    for text in legend.get_texts():
        text.set_color("#222222")
    fig.savefig(out_dir / "defenses_01d_delta_auc_vs_target_accuracy.png", bbox_inches="tight")
    plt.close(fig)

def plot_delta_tradeoff_by_dataset(df: pd.DataFrame, out_dir: Path, datasets: tuple[str, ...]) -> None:
    """One 3x3 figure of per-dataset defense deltas, with AMIA/RMIA marker symbols."""
    dataset_means = (
        df[df["defense_key"] != "no_defense"]
        .groupby(["dataset", "defense_key", "defense_label", "defense_short", "family"], as_index=False)
        .agg(
            rmia_auc_reduction=("rmia_auc_reduction", "mean"),
            amia_auc_reduction=("amia_auc_reduction", "mean"),
            accuracy_drop=("accuracy_drop", "mean"),
            n_models=("model", "nunique"),
        )
    )
    dataset_means = dataset_means[dataset_means["dataset"].isin(datasets)].copy()
    if dataset_means.empty:
        return

    plot_df = dataset_means.melt(
        id_vars=["dataset", "defense_key", "defense_label", "defense_short", "family", "accuracy_drop", "n_models"],
        value_vars=["amia_auc_reduction", "rmia_auc_reduction"],
        var_name="attack_metric",
        value_name="auc_reduction",
    )
    plot_df["attack"] = plot_df["attack_metric"].map({
        "amia_auc_reduction": "AMIA",
        "rmia_auc_reduction": "RMIA",
    })
    plot_df = plot_df.dropna(subset=["accuracy_drop", "auc_reduction"])
    if plot_df.empty:
        return

    plot_df.to_csv(out_dir / "defenses_01c_delta_tradeoff_by_dataset.csv", index=False)

    present = set(plot_df["dataset"])
    present_datasets = [dataset for dataset in datasets if dataset in present]
    n_panels = min(9, max(1, len(present_datasets)))
    fig, axes = plt.subplots(3, 3, figsize=(15.2, 12.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.13, hspace=0.34, wspace=0.18)
    axes_flat = axes.ravel()
    attack_markers = {"AMIA": "o", "RMIA": "X"}

    x_min, x_max = plot_df["accuracy_drop"].min(), plot_df["accuracy_drop"].max()
    y_min, y_max = plot_df["auc_reduction"].min(), plot_df["auc_reduction"].max()
    x_pad = max((x_max - x_min) * 0.08, 0.005)
    y_pad = max((y_max - y_min) * 0.08, 0.005)

    for ax_idx, ax in enumerate(axes_flat):
        if ax_idx >= n_panels:
            ax.set_visible(False)
            continue

        dataset = present_datasets[ax_idx]
        panel = plot_df[plot_df["dataset"] == dataset]
        for family in FAMILY_ORDER:
            family_df = panel[panel["family"] == family]
            if family_df.empty:
                continue
            for attack, marker_shape in attack_markers.items():
                attack_df = family_df[family_df["attack"] == attack]
                if attack_df.empty:
                    continue
                ax.scatter(
                    attack_df["accuracy_drop"],
                    attack_df["auc_reduction"],
                    c=PALETTE.get(family, "#999999"),
                    marker=marker_shape,
                    s=38,
                    alpha=0.84,
                    edgecolors="white",
                    linewidths=0.45,
                )

        ax.axhline(0, color="#777777", lw=0.8, ls="--", alpha=0.7)
        ax.axvline(0, color="#777777", lw=0.8, ls="--", alpha=0.7)
        ax.set_title(f"{dataset} ({panel['defense_key'].nunique()} defenses)")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        if ax_idx % 3 == 0:
            ax.set_ylabel("Δ AUC")
        if ax_idx >= 6:
            ax.set_xlabel("Δ ACC")

    family_handles = [
        Line2D([0], [0], marker="o", color="none", label=family,
               markerfacecolor=PALETTE[family], markeredgecolor="white", markersize=7)
        for family in FAMILY_ORDER
        if family in set(plot_df["family"])
    ]
    attack_handles = [
        Line2D([0], [0], marker=marker_shape, color="#4d4d4d", label=attack,
               linestyle="none", markerfacecolor="#4d4d4d", markersize=7)
        for attack, marker_shape in attack_markers.items()
    ]
    fig.legend(handles=family_handles, title="Defense family", loc="lower center",
               bbox_to_anchor=(0.38, 0.035), ncol=3, frameon=True)
    fig.legend(handles=attack_handles, title="Attack metric", loc="lower center",
               bbox_to_anchor=(0.82, 0.035), ncol=2, frameon=True)
    fig.suptitle("Defense Δ tradeoff by dataset, relative to no defense", y=1.02, fontsize=13)
    fig.savefig(out_dir / "defenses_01c_delta_tradeoff_by_dataset_3x3.png", bbox_inches="tight")
    plt.close(fig)




def plot_locations_highrisk_delta(df: pd.DataFrame, out_dir: Path) -> None:
    """High-risk k-anon averaged across locations + dropout_success: Δ accuracy vs Δ AUC per model.

    Color encodes alpha variant; line style (solid/dashed) encodes AMIA vs RMIA.
    """
    DATASETS = ("locations", "dropout_success")
    plot_df = df[
        df["dataset"].isin(DATASETS)
        & (df["family"] == "High-risk label k-anon")
    ].copy()
    plot_df = plot_df.dropna(subset=["accuracy_drop", "amia_auc_reduction", "rmia_auc_reduction"])
    if plot_df.empty:
        return

    parsed = plot_df["defense_key"].str.extract(r"highrisk_label_kanon_k(?P<k>\d+)(?:_a(?P<a>\d+))?")
    plot_df["k"] = pd.to_numeric(parsed["k"], errors="coerce").astype(int)
    plot_df["alpha"] = parsed["a"].fillna("base").map(lambda x: "base" if x == "base" else f"a{x}")

    plot_df = (
        plot_df.groupby(["model_label", "defense_key", "k", "alpha"], as_index=False)
        .agg(
            accuracy_drop=("accuracy_drop", "mean"),
            amia_auc_reduction=("amia_auc_reduction", "mean"),
            rmia_auc_reduction=("rmia_auc_reduction", "mean"),
            n_datasets=("dataset", "nunique"),
        )
    )
    plot_df.to_csv(out_dir / "defenses_01e_locs_ds_highrisk_delta.csv", index=False)

    present_models = [m for m in MODEL_LABEL_ORDER if m in set(plot_df["model_label"])]

    alpha_styles = {
        "base": dict(color="#d7191c", label="base (α = 0)"),
        "a30":  dict(color="#fdae61", label="α = 30%"),
        "a50":  dict(color="#1a9641", label="α = 50%"),
    }
    attack_styles = {
        "AMIA": dict(ls="-",  lw=2.0),
        "RMIA": dict(ls="--", lw=2.0),
    }
    k_markers = {3: "o", 5: "s", 10: "^"}

    attack_pairs = [
        ("amia_auc_reduction", "AMIA"),
        ("rmia_auc_reduction", "RMIA"),
    ]

    tabdpt_label = "TabDPT"

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 10.4),
                             sharex=False, sharey=False,
                             constrained_layout=True)
    axes_flat = list(axes.flatten())

    for idx, (ax, model) in enumerate(zip(axes_flat, present_models)):
        sub = plot_df[plot_df["model_label"] == model]

        for alpha_val, a_style in alpha_styles.items():
            path = sub[sub["alpha"] == alpha_val].sort_values("k")
            if path.empty:
                continue
            for auc_col, atk_label in attack_pairs:
                atk_style = attack_styles[atk_label]
                ax.plot(
                    -path["accuracy_drop"], path[auc_col],
                    color=a_style["color"], ls=atk_style["ls"], lw=atk_style["lw"],
                    alpha=0.85, zorder=3,
                )
                for _, r in path.iterrows():
                    ax.scatter(
                        [-r["accuracy_drop"]], [r[auc_col]],
                        s=80, color=a_style["color"], marker=k_markers[int(r["k"])],
                        edgecolors="white", linewidths=0.6, zorder=4, alpha=0.9,
                    )

        if idx >= 2:
            ax.set_xlabel("Δ Target Accuracy", fontsize=15, fontweight="bold")
        if idx % 2 == 0:
            ax.set_ylabel("Δ Attack AUC", fontsize=15, fontweight="bold")
        ax.set_title(model, fontsize=17, pad=8, fontweight="bold")
        ax.axhline(0, color="#bbbbbb", lw=0.9, ls="--", alpha=0.6)
        ax.axvline(0, color="#bbbbbb", lw=0.9, ls="--", alpha=0.6)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=13)
        if model == tabdpt_label:
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))

    # shared y-axis limits across all panels
    all_y = [v for auc_col, _ in attack_pairs
             for v in plot_df[auc_col].dropna().tolist()]
    y_pad = (max(all_y) - min(all_y)) * 0.05
    y_lim = (min(all_y) - y_pad, max(all_y) + y_pad)

    # shared x-axis limits for TabPFN / Real TabPFN only;
    # TabICL and TabDPT have very different ranges and auto-scale independently
    own_xlim = {tabdpt_label, "TabICL"}
    shared_x_df = plot_df[~plot_df["model_label"].isin(own_xlim)]
    all_x = (-shared_x_df["accuracy_drop"]).dropna().tolist()
    x_pad = (max(all_x) - min(all_x)) * 0.05
    x_lim = (min(all_x) - x_pad, max(all_x) + x_pad)

    for ax, model in zip(axes_flat, present_models):
        ax.set_ylim(y_lim)
        if model not in own_xlim:
            ax.set_xlim(x_lim)

    for ax in axes_flat[len(present_models):]:
        ax.set_visible(False)

    alpha_handles = [
        Line2D([0], [0], color=s["color"], ls="-", lw=2, marker="o",
               markerfacecolor=s["color"], markeredgecolor="white", markersize=7, label=s["label"])
        for s in alpha_styles.values()
    ]
    attack_handles = [
        Line2D([0], [0], color="#555555", ls="-",  lw=2, label="AMIA"),
        Line2D([0], [0], color="#555555", ls="--", lw=2, label="RMIA"),
    ]
    k_handles = [
        Line2D([0], [0], color="#888888", marker=marker, ls="", markersize=8, label=f"k={k}")
        for k, marker in k_markers.items()
    ]
    fig.legend(
        handles=alpha_handles + attack_handles + k_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=len(alpha_handles) + len(attack_handles) + len(k_handles),
        frameon=True,
        fontsize=13,
    )
    fig.savefig(out_dir / "defenses_01e_locs_ds_highrisk_delta.png", bbox_inches="tight")
    plt.close(fig)


def plot_locations_highrisk_k_sweep(df: pd.DataFrame, out_dir: Path) -> None:
    """Locations high-risk label k-anon effect by k, model, and attack."""
    plot_df = df[
        (df["dataset"] == "locations")
        & (df["family"] == "High-risk label k-anon")
    ].copy()
    plot_df = plot_df.dropna(subset=["accuracy_drop", "amia_auc_reduction", "rmia_auc_reduction"])
    if plot_df.empty:
        return

    parsed = plot_df["defense_key"].str.extract(r"highrisk_label_kanon_k(?P<k>\d+)(?:_a(?P<a>\d+))?")
    plot_df["k"] = pd.to_numeric(parsed["k"], errors="coerce")
    plot_df["alpha"] = parsed["a"].fillna("base").map(lambda x: "base" if x == "base" else f"a{x}")
    plot_df = plot_df.dropna(subset=["k"]).copy()
    plot_df["k"] = plot_df["k"].astype(int)
    plot_df.to_csv(out_dir / "defenses_01f_locations_highrisk_k_sweep.csv", index=False)

    present_models = [model for model in MODEL_LABEL_ORDER if model in set(plot_df["model_label"])]
    alpha_order = [alpha for alpha in ["base", "a30", "a50"] if alpha in set(plot_df["alpha"])]
    alpha_styles = {
        "base": {"color": "#0072b2", "marker": "o", "label": "base"},
        "a30": {"color": "#d55e00", "marker": "s", "label": "a30"},
        "a50": {"color": "#009e73", "marker": "^", "label": "a50"},
    }

    fig, axes = plt.subplots(
        len(present_models),
        2,
        figsize=(12.8, max(8.0, 2.45 * len(present_models))),
        sharex=True,
        sharey=False,
        squeeze=False,
        constrained_layout=True,
    )
    panels = [
        ("amia_auc_reduction", "AMIA"),
        ("rmia_auc_reduction", "RMIA"),
    ]

    for row_idx, model_label in enumerate(present_models):
        model_df = plot_df[plot_df["model_label"] == model_label]
        for col_idx, (metric, attack_title) in enumerate(panels):
            ax = axes[row_idx][col_idx]
            for alpha in alpha_order:
                series = model_df[model_df["alpha"] == alpha].sort_values("k")
                if series.empty:
                    continue
                style = alpha_styles.get(alpha, {"color": "#666666", "marker": "o", "label": alpha})
                ax.plot(
                    series["k"],
                    series[metric],
                    marker=style["marker"],
                    color=style["color"],
                    lw=2.2,
                    ms=7.5,
                    alpha=0.94,
                )

            ax.axhline(0, color="#777777", lw=0.9, ls="--", alpha=0.72)
            ax.set_title(attack_title if row_idx == 0 else "", fontsize=19, pad=9)
            ax.set_ylabel(f"{model_label}\nΔ attack AUC", fontsize=14.5)
            ax.set_xlabel("k" if row_idx == len(present_models) - 1 else "", fontsize=16)
            ax.tick_params(axis="both", labelsize=14)
            ax.grid(True, alpha=0.25)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    alpha_handles = [
        Line2D(
            [0],
            [0],
            marker=alpha_styles[alpha]["marker"],
            color=alpha_styles[alpha]["color"],
            lw=2.2,
            markersize=8.5,
            label=alpha_styles[alpha]["label"],
        )
        for alpha in alpha_order
    ]
    fig.legend(
        handles=alpha_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=len(alpha_handles),
        frameon=True,
        fontsize=14,
    )
    fig.savefig(out_dir / "defenses_01f_locations_highrisk_k_sweep.png", bbox_inches="tight")
    plt.close(fig)


def plot_highrisk_focus(summary_all_models: pd.DataFrame, out_dir: Path) -> None:
    plot_df = summary_all_models[summary_all_models["family"] == "High-risk label k-anon"].copy()
    if plot_df.empty:
        return

    parsed = plot_df["defense_key"].str.extract(r"highrisk_label_kanon_k(?P<k>\d+)(?:_a(?P<a>\d+))?")
    plot_df["k"] = pd.to_numeric(parsed["k"], errors="coerce")
    plot_df["alpha"] = parsed["a"].fillna("base").map(lambda x: "a=base" if x == "base" else f"a={x}")
    plot_df = plot_df.dropna(subset=["k"])
    plot_df["k"] = plot_df["k"].astype(int)

    model_palette = {
        "TabPFN": "#0072b2",
        "Real TabPFN": "#cc79a7",
        "TabICL": "#009e73",
        "TabDPT": "#d55e00",
    }
    alpha_markers = {"a=base": "o", "a=30": "s", "a=50": "^"}
    alpha_order = ["a=base", "a=30", "a=50"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    panels = [
        (axes[0], "amia_auc_reduction", "AMIA Δ AUC vs Δ ACC, by model"),
        (axes[1], "rmia_auc_reduction", "RMIA Δ AUC vs Δ ACC, by model"),
    ]

    for ax, metric, title in panels:
        for alpha in alpha_order:
            alpha_df = plot_df[plot_df["alpha"] == alpha]
            for model_label in MODEL_LABEL_ORDER:
                series = alpha_df[alpha_df["model_label"] == model_label].sort_values("k")
                if series.empty:
                    continue
                ax.scatter(
                    series["accuracy_drop"],
                    series[metric],
                    s=120,
                    marker=alpha_markers.get(alpha, "o"),
                    color=model_palette.get(model_label, "#666666"),
                    edgecolor="white",
                    linewidth=0.75,
                    alpha=0.9,
                )
                for _, point in series.iterrows():
                    ax.text(
                        point["accuracy_drop"],
                        point[metric],
                        str(int(point["k"])),
                        ha="center",
                        va="center",
                        fontsize=6.3,
                        fontweight="bold",
                        color="white",
                        zorder=6,
                    )
                ax.plot(
                    series["accuracy_drop"],
                    series[metric],
                    color=model_palette.get(model_label, "#666666"),
                    lw=0.9,
                    alpha=0.28,
                )
        ax.axhline(0, color="#777777", lw=0.9, ls="--", alpha=0.7)
        ax.axvline(0, color="#777777", lw=0.9, ls="--", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Δ ACC")
        ax.set_ylabel("Δ AUC")
        ax.grid(True, alpha=0.25)

    model_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color,
                   markeredgecolor="white", markersize=7, label=model)
        for model, color in model_palette.items()
        if model in set(plot_df["model_label"])
    ]
    alpha_handles = [
        plt.Line2D([0], [0], marker=marker, linestyle="", color="#444444",
                   markersize=7, label=alpha)
        for alpha, marker in alpha_markers.items()
        if alpha in set(plot_df["alpha"])
    ]
    legend1 = axes[1].legend(handles=model_handles, title="Model", loc="upper left", frameon=True)
    axes[1].add_artist(legend1)
    axes[1].legend(handles=alpha_handles, title="Variant", loc="lower right", frameon=True)

    fig.suptitle("High-risk label k-anon tradeoff by model, k, and a", y=1.02, fontsize=13)
    fig.savefig(out_dir / "defenses_06_highrisk_tradeoff.png", bbox_inches="tight")
    plt.close(fig)




def plot_metric_heatmaps(summary: pd.DataFrame, defense_order: list[str], out_dir: Path) -> None:
    labels = {k: short_defense(k, 44) for k in defense_order}
    fig, axes = plt.subplots(1, 3, figsize=(16, max(8, len(defense_order) * 0.28)), constrained_layout=True)
    for ax, metric, cmap, title in [
        (axes[0], "amia_row_max_auc", "viridis", "AMIA AUC (lower is better)"),
        (axes[1], "rmia_auc", "viridis", "RMIA AUC (lower is better)"),
        (axes[2], "accuracy", "viridis_r", "Accuracy (higher is better)"),
    ]:
        pivot = summary.pivot_table(index="defense_key", columns="model_label", values=metric, aggfunc="mean")
        pivot = pivot.reindex(index=defense_order, columns=MODEL_LABEL_ORDER)
        pivot.index = [labels[idx] for idx in pivot.index]
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            annot=True,
            fmt=".3f",
            linewidths=0.35,
            linecolor="white",
            cbar_kws={"shrink": 0.65},
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Mean defense metrics by model, using only MIC-defined defenses", y=1.01, fontsize=13)
    fig.savefig(out_dir / "defenses_02_metric_heatmaps_by_model.png", bbox_inches="tight")
    plt.close(fig)


def plot_family_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.copy()
    plot_df["family"] = pd.Categorical(plot_df["family"], categories=FAMILY_ORDER, ordered=True)
    family = (
        plot_df.groupby(["family", "model_label"], observed=True, as_index=False)
        .agg(
            rmia_auc_reduction=("rmia_auc_reduction", "mean"),
            amia_auc_reduction=("amia_auc_reduction", "mean"),
            accuracy_drop=("accuracy_drop", "mean"),
        )
    )
    long = family.melt(
        id_vars=["family", "model_label"],
        value_vars=["amia_auc_reduction", "rmia_auc_reduction", "accuracy_drop"],
        var_name="metric",
        value_name="value",
    )
    metric_labels = {
        "rmia_auc_reduction": "RMIA Δ AUC",
        "amia_auc_reduction": "AMIA Δ AUC",
        "accuracy_drop": "Δ ACC",
    }
    long["metric"] = long["metric"].map(metric_labels)
    grid = sns.catplot(
        data=long,
        kind="bar",
        x="family",
        y="value",
        hue="model_label",
        hue_order=MODEL_LABEL_ORDER,
        col="metric",
        col_wrap=1,
        height=3.2,
        aspect=3.0,
        palette="Set2",
        sharey=False,
    )
    for ax in grid.axes.flat:
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=18)
        ax.grid(True, axis="y", alpha=0.25)
    grid.set_titles("{col_name}")
    grid.set_ylabels("Δ vs no defense")
    grid.fig.suptitle("Defense family behavior across datasets", y=1.02, fontsize=13)
    grid.fig.savefig(out_dir / "defenses_03_family_deltas.png", bbox_inches="tight")
    plt.close(grid.fig)


def plot_best_by_dataset(df: pd.DataFrame, out_dir: Path) -> None:
    candidates = df[df["defense_key"] != "no_defense"].copy()
    if candidates.empty:
        return
    best = candidates.sort_values("tradeoff_score", ascending=False).drop_duplicates(["dataset", "model"])
    pivot = best.pivot_table(index="dataset", columns="model_label", values="tradeoff_score", aggfunc="mean")
    pivot = pivot.reindex(columns=MODEL_LABEL_ORDER)
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.reindex(row_order)
    fig_h = max(6, 0.3 * len(pivot))
    fig, ax = plt.subplots(figsize=(8.5, fig_h), constrained_layout=True)
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".3f",
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Best tradeoff score"},
    )
    ax.set_title("Best defense tradeoff score by dataset/model")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.savefig(out_dir / "defenses_04_best_tradeoff_by_dataset.png", bbox_inches="tight")
    plt.close(fig)

    table = best[[
        "dataset", "model", "model_label", "defense_key", "defense_label",
        "amia_row_max_auc", "rmia_auc", "accuracy", "privacy_gain", "accuracy_drop", "tradeoff_score",
    ]].sort_values(["dataset", "model"])
    table.to_csv(out_dir / "defenses_best_by_dataset_model.csv", index=False)


def plot_param_evolution_focus(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df[
        df["dataset"].isin(PARAM_EVOLUTION_DATASETS)
        & (df["defense_key"] != "no_defense")
        & (df["family"] != "Label k-anon + dropout")
    ].copy()
    if plot_df.empty:
        return

    plot_df["dropout_p"] = pd.to_numeric(
        plot_df["defense_key"].str.extract(r"attn_drop_p(?P<p>\d+)")["p"],
        errors="coerce",
    )
    plot_df["k"] = pd.to_numeric(
        plot_df["defense_key"].str.extract(r"(?:highrisk_)?label_kanon_k(?P<k>\d+)")["k"],
        errors="coerce",
    )
    plot_df["alpha"] = (
        plot_df["defense_key"]
        .str.extract(r"(?:^|_)a(?P<alpha>\d+)(?:$|\+)")["alpha"]
        .fillna("base")
        .map(lambda x: "base" if x == "base" else f"a{x}")
    )

    track_map = {
        "Attention dropout": ("Attention dropout p", "dropout_p"),
        "Label k-anon": ("Label k-anon k", "k"),
        "High-risk label k-anon": ("High-risk label k-anon k", "k"),
    }
    plot_df["param_track"] = plot_df["family"].map(lambda family: track_map.get(family, (None, None))[0])
    plot_df["param_column"] = plot_df["family"].map(lambda family: track_map.get(family, (None, None))[1])
    plot_df["param_value"] = np.nan
    for column in ["dropout_p", "k"]:
        mask = plot_df["param_column"] == column
        plot_df.loc[mask, "param_value"] = plot_df.loc[mask, column]

    plot_df = plot_df.dropna(subset=["param_track", "param_value"]).copy()
    plot_df = plot_df[plot_df["alpha"] == "base"].copy()
    if plot_df.empty:
        return

    summary = (
        plot_df.groupby(["dataset", "param_track", "param_value", "alpha", "model", "model_label"], as_index=False)
        .agg(
            amia_row_max_auc=("amia_row_max_auc", "mean"),
            accuracy=("accuracy", "mean"),
            amia_auc_reduction=("amia_auc_reduction", "mean"),
            accuracy_drop=("accuracy_drop", "mean"),
            tradeoff_score=("tradeoff_score", "mean"),
            n_defenses=("defense_key", "nunique"),
        )
    )
    summary.to_csv(out_dir / "defenses_07_param_evolution_locations_dropout_success.csv", index=False)

    model_palette = {
        "TabPFN": "#0072b2",
        "Real TabPFN": "#cc79a7",
        "TabICL": "#009e73",
        "TabDPT": "#d55e00",
    }
    track_order = [
        "Attention dropout p",
        "Label k-anon k",
        "High-risk label k-anon k",
    ]
    present_tracks = [track for track in track_order if track in set(summary["param_track"])]
    present_datasets = [dataset for dataset in PARAM_EVOLUTION_DATASETS if dataset in set(summary["dataset"])]
    if not present_tracks or not present_datasets:
        return

    fig, axes = plt.subplots(
        len(present_datasets),
        len(present_tracks),
        figsize=(4.3 * len(present_tracks), 3.2 * len(present_datasets)),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for row_idx, dataset in enumerate(present_datasets):
        for col_idx, track in enumerate(present_tracks):
            ax = axes[row_idx][col_idx]
            panel = summary[
                (summary["dataset"] == dataset)
                & (summary["param_track"] == track)
            ].copy()
            if panel.empty:
                ax.set_title(track)
                ax.set_xlabel("dropout p" if track.endswith("p") else "k")
                ax.set_ylabel(dataset if col_idx == 0 else "")
                ax.text(0.5, 0.5, "No matching defenses", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            for model_label in MODEL_LABEL_ORDER:
                color = model_palette.get(model_label, "#666666")
                series = panel[panel["model_label"] == model_label].sort_values("param_value")
                if series.empty:
                    continue
                x = series["param_value"].to_numpy(dtype=float)
                ax.plot(
                    x,
                    -series["amia_auc_reduction"].to_numpy(dtype=float),
                    marker="o",
                    lw=1.9,
                    ms=4.8,
                    color=color,
                    alpha=0.96,
                )
                ax.plot(
                    x,
                    -series["accuracy_drop"].to_numpy(dtype=float),
                    marker="o",
                    lw=1.15,
                    ms=3.8,
                    ls="--",
                    color=color,
                    alpha=0.32,
                )

            ax.axhline(0, color="#777777", lw=0.85, ls="--", alpha=0.75)
            ax.set_title(track)
            ax.set_xlabel("dropout p" if track.endswith("p") else "k")
            ax.set_ylabel(f"{dataset}\nΔ (defense - no defense)" if col_idx == 0 else "")
            ax.grid(True, alpha=0.25)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    model_handles = [
        Line2D([0], [0], color=model_palette[model_label], lw=1.9, marker="o", label=model_label)
        for model_label in MODEL_LABEL_ORDER
        if model_label in set(summary["model_label"])
    ]
    metric_handles = [
        Line2D([0], [0], color="#444444", lw=1.9, marker="o", label="Δ AMIA AUC"),
        Line2D([0], [0], color="#444444", lw=1.15, ls="--", alpha=0.35, marker="o", label="Accuracy drop"),
    ]
    fig.legend(
        handles=model_handles + metric_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=min(6, len(model_handles) + len(metric_handles)),
        frameon=True,
    )

    fig.savefig(out_dir / "defenses_07_param_evolution_locations_dropout_success.png", bbox_inches="tight")
    plt.close(fig)


def write_tables(df: pd.DataFrame, summary: pd.DataFrame, allowed: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "defenses_viz_filtered_runs.csv", index=False)
    summary.to_csv(out_dir / "defenses_viz_summary_by_model.csv", index=False)
    allowed_table = pd.DataFrame({
        "defense_key": allowed,
        "defense_label": [pretty_defense(x) for x in allowed],
        "family": [defense_family(x) for x in allowed],
    })
    allowed_table.to_csv(out_dir / "defenses_viz_mic_defense_set.csv", index=False)


def run(logs_dir: Path, out_dir: Path, models: tuple[str, ...], mic_dataset: str, datasets: tuple[str, ...]) -> None:
    set_style()
    raw = load_defense_results(logs_dir, models)
    if raw.empty:
        raise SystemExit(f"No defense_eval_results.csv files found under {logs_dir} for models: {', '.join(models)}")

    dataset_set = set(datasets)
    raw = raw[raw["dataset"].isin(dataset_set)].copy()
    if raw.empty:
        raise SystemExit(f"No defense rows remain after dataset filtering: {', '.join(datasets)}")

    allowed = mic_defense_set(raw, mic_dataset)
    filtered = raw[raw["defense_key"].isin(allowed)].copy()
    filtered = add_baseline_deltas(filtered)
    summary = aggregate(filtered)
    defense_order = ordered_defenses(summary)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_tables(filtered, summary, allowed, out_dir)
    plot_privacy_utility(filtered, out_dir, datasets)
    plot_delta_tradeoff(summary, out_dir)
    plot_delta_auc_vs_target_accuracy(summary, out_dir)
    plot_locations_highrisk_delta(filtered, out_dir)
    plot_locations_highrisk_k_sweep(filtered, out_dir)
    plot_delta_tradeoff_by_dataset(filtered, out_dir, datasets)
    plot_metric_heatmaps(summary, defense_order, out_dir)
    plot_family_comparison(filtered, out_dir)
    plot_highrisk_focus(summary, out_dir)
    plot_best_by_dataset(filtered, out_dir)
    plot_param_evolution_focus(filtered, out_dir)

    print(f"Loaded {len(raw):,} defense rows from selected datasets; kept {len(filtered):,} rows after MIC defense-set filtering.")
    print(f"Datasets: {filtered['dataset'].nunique()} ({', '.join(sorted(filtered['dataset'].unique()))}) | models: {', '.join(models)} | defenses: {len(allowed)}")
    print(f"Saved defense visualizations and summary CSVs to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize defense results using only the defenses present in the MIC dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Outputs:\n"
            "  defenses_01_privacy_utility_tradeoff.png\n"
            "  defenses_01b_delta_tradeoff.png\n"
            "  defenses_01c_delta_tradeoff_by_dataset_3x3.png\n"
            "  defenses_01d_delta_auc_vs_target_accuracy.png\n"
            "  defenses_02_metric_heatmaps_by_model.png\n"
            "  defenses_03_family_deltas.png\n"
            "  defenses_04_best_tradeoff_by_dataset.png\n"
            "  defenses_06_highrisk_tradeoff.png\n"
            "  defenses_07_param_evolution_locations_dropout_success.png\n"
            "  defenses_viz_*.csv summary tables\n"
            "\nExample:\n"
            "  uv run results_visualizations/defenses_viz.py\n"
        ),
    )
    parser.add_argument("--logs-dir", default="ml_privacy_meter/logs", type=Path)
    parser.add_argument("--save-dir", default="results_visualizations/defenses_viz", type=Path)
    parser.add_argument("--mic-dataset", default=MIC_DATASET)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                        help="Datasets to include (default: same six used by attacks_viz).")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                        help="Models to include (default: four TabFM models).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.logs_dir, args.save_dir, tuple(args.models), args.mic_dataset, tuple(args.datasets))
