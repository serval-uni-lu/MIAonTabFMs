"""Visualize dataset characteristics and their relationship to MIA results.

By default this script profiles the same records used to train target model 0
in the RMIA seed-1 run. It reads the saved RMIA split indices and applies them
to ``data/original/<dataset>.csv`` after the same numeric conversion used by
RMIA. Use ``--profile-scope full_saved`` to instead reuse the existing full
``dataset_profiles.csv``.

Recommended use from the repository root:
    .venv/bin/python results_visualizations/data_viz.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset_profile import compute_dataset_profile


DEFAULT_DATA_DIR = Path("data/original")
DEFAULT_PROFILE_CSV = Path("results_visualizations/attacks_viz/dataset_profiles.csv")
DEFAULT_ATTACKS_DIR = Path("results_visualizations/attacks_viz")
DEFAULT_LOGS_DIR = Path("ml_privacy_meter/logs")
DEFAULT_OUTPUT_DIR = Path("results_visualizations/data_viz")

MODEL_ORDER = [
    "rf",
    "lightgbm",
    "mlp",
    "tabnet",
    "tabpfn",
    "real-tabpfn",
    "tabicl",
    "tabdpt",
]

MODEL_DISPLAY = {
    "rf": "RF",
    "lightgbm": "LightGBM",
    "mlp": "MLP",
    "tabnet": "TabNet",
    "tabpfn": "TabPFN",
    "real-tabpfn": "Real-TabPFN",
    "tabicl": "TabICL",
    "tabdpt": "TabDPT",
}

DATASET_LABELS = {
    "46956_seismic-bumps": "seismic-bumps",
    "46980_MIC": "MIC",
}

CRITICAL_METRICS = [
    "total_rows",
    "num_features",
    "num_binary_features",
    "num_classes",
    "audit_population_rows_seed",
    "majority_class_pct",
    "outlier_pct",
    "binary_feature_pct",
    "target_accuracy_avg_pct",
    "imbalance_ratio",
    "samples_per_feature",
    "knn_label_disagreement",
]

CORRELATION_METRICS = [
    "log_num_features",
    "log_total_rows",
    "num_classes",
    "num_binary_features",
    "imbalance_ratio",
    "target_entropy",
    "train_test_gap_pct_points",
    "knn_label_disagreement",
    "pca_n_components_95pct",
    "mean_abs_skewness",
    "max_abs_corr_with_target",
]


def fmt_dataset(name: str) -> str:
    return DATASET_LABELS.get(name, re.sub(r"^\d+_", "", name))


def fmt_model(name: str) -> str:
    return MODEL_DISPLAY.get(name, name)


def model_sort_key(name: str) -> tuple[int, str]:
    try:
        return MODEL_ORDER.index(name), name
    except ValueError:
        return len(MODEL_ORDER), name


def attack_datasets(attack_dir: Path) -> set[str]:
    datasets: set[str] = set()
    for filename in ["10_attack_auc_comparison.csv", "19_amia_vs_rmia_auc.csv"]:
        path = attack_dir / filename
        if path.exists():
            df = pd.read_csv(path, usecols=["dataset"])
            datasets.update(df["dataset"].dropna().astype(str))
    return datasets


def prepare_like_rmia(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same basic numeric conversion used before RMIA splitting."""
    data = df.copy()
    for col in data.columns:
        if pd.api.types.is_string_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
            try:
                data[col] = pd.to_numeric(data[col], errors="raise")
            except (ValueError, TypeError):
                data[col] = data[col].astype("category").cat.codes
    return data


def add_derived_profile_columns(profiles: pd.DataFrame) -> pd.DataFrame:
    profiles = profiles.copy()
    profiles["duplicate_pct"] = 100.0 * profiles["duplicate_rows"] / profiles["total_rows"]
    profiles["binary_feature_pct"] = 100.0 * profiles["num_binary_features"] / profiles["num_features"]
    profiles["categorical_feature_pct"] = 100.0 * profiles["num_categorical_features"] / profiles["num_features"]
    profiles["pca_top5_pct"] = 100.0 * profiles["pca_explained_var_top5"]
    profiles["knn_label_disagreement_pct"] = 100.0 * profiles["knn_label_disagreement"]
    profiles["log_total_rows"] = np.log10(profiles["total_rows"].clip(lower=1))
    profiles["log_num_features"] = np.log10(profiles["num_features"].clip(lower=1))
    profiles["dataset_display"] = profiles["dataset_name"].map(fmt_dataset)
    return profiles


def _select_split_dir(logs_dir: Path, dataset_name: str, seed: int, preferred_model: str | None) -> Path | None:
    model_candidates = []
    if preferred_model:
        model_candidates.append(preferred_model)
    model_candidates.extend([m for m in MODEL_ORDER if m not in model_candidates])

    for model in model_candidates:
        split_dir = logs_dir / dataset_name / model / f"seed{seed}" / "rmia" / "splits"
        if (split_dir / "context_indices_original.npy").exists() and (split_dir / "model_pair_splits.npz").exists():
            return split_dir
    return None


def load_rmia_split_profiles(
    data_dir: Path,
    logs_dir: Path,
    attack_dir: Path,
    seed: int,
    split_scope: str,
    preferred_model: str | None,
) -> pd.DataFrame:
    rows = []
    for dataset_name in sorted(attack_datasets(attack_dir)):
        data_path = data_dir / f"{dataset_name}.csv"
        split_dir = _select_split_dir(logs_dir, dataset_name, seed, preferred_model)
        if not data_path.exists() or split_dir is None:
            continue

        raw = pd.read_csv(data_path, header=None)
        transformed = prepare_like_rmia(raw)
        transformed.columns = [f"feature_{i}" for i in range(transformed.shape[1] - 1)] + ["target"]

        context_original = np.load(split_dir / "context_indices_original.npy")
        population_original = np.load(split_dir / "population_indices_original.npy")
        with np.load(split_dir / "model_pair_splits.npz") as split_npz:
            train_context = split_npz["model_0_train_indices_context"]
            test_context = split_npz["model_0_test_indices_context"]

        if split_scope == "seed_target_train":
            selected_indices = context_original[train_context]
            profile_scope = f"seed{seed}_target_train"
        elif split_scope == "seed_context":
            selected_indices = context_original
            profile_scope = f"seed{seed}_rmia_context_pool"
        elif split_scope == "seed_target_train_test":
            selected_indices = np.concatenate([context_original[train_context], context_original[test_context]])
            profile_scope = f"seed{seed}_target_train_plus_target_test"
        elif split_scope == "seed_context_plus_population":
            selected_indices = np.concatenate([context_original, population_original])
            profile_scope = f"seed{seed}_context_plus_population"
        else:
            raise ValueError(f"Unknown split scope: {split_scope}")

        selected_indices = np.asarray(selected_indices, dtype=int)
        split_df = transformed.iloc[selected_indices].reset_index(drop=True)
        profile = compute_dataset_profile(split_df, dataset_name)

        # num_classes and num_binary_features are dataset-level properties, so
        # compute them from the full raw dataset (already in memory as `transformed`).
        # Other statistics (knn_label_disagreement, majority_class_pct, etc.)
        # still come from the chosen split_scope.
        full_feat_cols = list(transformed.columns[:-1])
        full_label_col = transformed.columns[-1]
        profile["num_classes"] = int(transformed[full_label_col].nunique())
        profile["num_binary_features"] = sum(
            1 for c in full_feat_cols if transformed[c].nunique() <= 2
        )

        profile.update(
            {
                "profile_scope": profile_scope,
                "profile_seed": seed,
                "split_scope": split_scope,
                "split_source": str(split_dir),
                "split_rows": len(split_df),
                "target_train_rows_seed": len(train_context),
                "target_test_rows_seed": len(test_context),
                "context_rows_seed": len(context_original),
                "auditing_rows_seed": len(context_original),
                "population_rows_seed": len(population_original),
            }
        )
        rows.append(profile)

    if not rows:
        raise ValueError("No RMIA split profiles could be built from saved split indices")

    profiles = pd.DataFrame(rows)
    profiles = add_derived_profile_columns(profiles)
    return profiles.sort_values("dataset_name").reset_index(drop=True)


def load_profiles(profile_csv: Path, attack_dir: Path) -> pd.DataFrame:
    if not profile_csv.exists():
        raise FileNotFoundError(f"Dataset profile CSV not found: {profile_csv}")

    profiles = pd.read_csv(profile_csv)
    if "dataset_name" not in profiles.columns:
        raise ValueError(f"{profile_csv} must contain a dataset_name column")

    allowed = attack_datasets(attack_dir)
    if allowed:
        profiles = profiles[profiles["dataset_name"].isin(allowed)].copy()
    if profiles.empty:
        raise ValueError("No dataset profiles match the attack result datasets")

    profiles["profile_scope"] = "full_saved_dataset_profiles"
    profiles["profile_seed"] = np.nan
    profiles["split_scope"] = "full_saved"
    profiles["split_source"] = str(profile_csv)
    profiles = add_derived_profile_columns(profiles)
    return profiles.sort_values("dataset_name").reset_index(drop=True)


def load_seed_counts(logs_dir: Path, profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    allowed = set(profiles["dataset_name"])
    metric_to_attack = {"auc": "RMIA", "row_max_auc": "AMIA"}

    for summary_path in sorted(logs_dir.glob("*/*/attack_result_seed_summary.csv")):
        parts = summary_path.relative_to(logs_dir).parts
        if len(parts) < 3:
            continue
        dataset, model = parts[0], parts[1]
        if dataset not in allowed:
            continue
        try:
            summary = pd.read_csv(summary_path)
        except Exception:
            continue

        if "attack" not in summary.columns:
            summary["attack"] = "rmia"
        if not {"metric", "num_seeds"}.issubset(summary.columns):
            continue

        for metric, attack_family in metric_to_attack.items():
            metric_rows = summary[summary["metric"] == metric]
            if attack_family == "AMIA":
                metric_rows = metric_rows[metric_rows["attack"].fillna("amia") == "amia"]
            elif "attack" in metric_rows.columns:
                metric_rows = metric_rows[metric_rows["attack"].fillna("rmia") == "rmia"]
            if metric_rows.empty:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "attack_family": attack_family,
                    "num_seeds": float(metric_rows["num_seeds"].dropna().iloc[0]),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["dataset", "model", "attack_family", "num_seeds"])
    return pd.DataFrame(rows).drop_duplicates(["dataset", "model", "attack_family"])


def load_model_metadata_metrics(logs_dir: Path, profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    allowed = set(profiles["dataset_name"])
    for metadata_path in sorted(logs_dir.glob("*/*/seed*/rmia/models/models_metadata.json")):
        parts = metadata_path.relative_to(logs_dir).parts
        if len(parts) < 5:
            continue
        dataset, model = parts[0], parts[1]
        if dataset not in allowed:
            continue
        try:
            with metadata_path.open() as f:
                metadata = json.load(f)
        except Exception:
            continue

        target = metadata.get("0", {})
        train_acc = target.get("train_acc")
        test_acc = target.get("test_acc")
        if train_acc is None and test_acc is None:
            continue
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "target_train_acc": train_acc,
                "target_test_acc": test_acc,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "target_train_acc",
                "target_test_acc",
                "target_model_accuracy_pct",
                "train_test_gap_pct_points",
            ]
        )

    metrics = pd.DataFrame(rows)
    metrics = metrics.groupby(["dataset", "model"], as_index=False).mean(numeric_only=True)
    metrics["target_model_accuracy_pct"] = 100.0 * metrics["target_test_acc"]
    metrics["train_test_gap_pct_points"] = 100.0 * (
        metrics["target_train_acc"] - metrics["target_test_acc"]
    )
    return metrics


def load_rmia(
    attack_dir: Path,
    profiles: pd.DataFrame,
    model_metrics: pd.DataFrame,
    seed_counts: pd.DataFrame,
) -> pd.DataFrame:
    path = attack_dir / "10_attack_auc_comparison.csv"
    if not path.exists():
        return pd.DataFrame()

    auc = pd.read_csv(path)
    if "rmia" not in auc.columns:
        return pd.DataFrame()

    auc = auc[["dataset", "model", "rmia"]].rename(columns={"rmia": "auc"})
    auc["attack_family"] = "RMIA"
    auc = auc[auc["dataset"].isin(set(profiles["dataset_name"]))]
    if not model_metrics.empty:
        auc = auc.merge(model_metrics, on=["dataset", "model"], how="left")
    if not seed_counts.empty:
        auc = auc.merge(seed_counts, on=["dataset", "model", "attack_family"], how="left")
    return auc.dropna(subset=["auc"])


def load_amia(
    attack_dir: Path,
    profiles: pd.DataFrame,
    model_metrics: pd.DataFrame,
    seed_counts: pd.DataFrame,
) -> pd.DataFrame:
    path = attack_dir / "19_amia_vs_rmia_auc.csv"
    if not path.exists():
        return pd.DataFrame()

    amia = pd.read_csv(path)
    if "row_max_auc" not in amia.columns:
        return pd.DataFrame()

    long = amia.melt(
        id_vars=["dataset", "model"],
        value_vars=["row_max_auc"],
        var_name="amia_metric",
        value_name="auc",
    )
    long["attack_family"] = "AMIA"
    long = long[long["dataset"].isin(set(profiles["dataset_name"]))]
    if not model_metrics.empty:
        long = long.merge(model_metrics, on=["dataset", "model"], how="left")
    if not seed_counts.empty:
        long = long.merge(seed_counts, on=["dataset", "model", "attack_family"], how="left")
    return long.dropna(subset=["auc"])


def aggregate_dataset_auc(attacks: pd.DataFrame) -> pd.DataFrame:
    """Collapse seed-averaged model rows to dataset-level medians."""
    if attacks.empty:
        return attacks

    group_cols = ["dataset", "attack_family"]
    if "amia_metric" in attacks.columns and attacks["amia_metric"].notna().any():
        group_cols.append("amia_metric")

    return (
        attacks.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            median_auc=("auc", "median"),
            mean_auc=("auc", "mean"),
            n_models=("model", "nunique"),
            min_num_seeds=("num_seeds", "min"),
            max_num_seeds=("num_seeds", "max"),
            target_model_accuracy_pct=("target_model_accuracy_pct", "median"),
            train_test_gap_pct_points=("train_test_gap_pct_points", "median"),
        )
        .rename(columns={"median_auc": "auc"})
    )


def spearman_correlations(df: pd.DataFrame, auc_col: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        tmp = df[[metric, auc_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(tmp) < 3 or tmp[metric].nunique() < 2 or tmp[auc_col].nunique() < 2:
            continue
        rho = tmp[metric].corr(tmp[auc_col], method="spearman")
        rows.append({"metric": metric, "spearman_r": rho, "n_datasets": len(tmp)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_spearman_r"] = out["spearman_r"].abs()
    return out.sort_values("abs_spearman_r", ascending=False).reset_index(drop=True)


def save_profiles_table(profiles: pd.DataFrame, output_dir: Path) -> None:
    ordered_cols = [
        "dataset_name",
        "profile_scope",
        "profile_seed",
        "split_scope",
        "split_source",
        "split_rows",
        "target_train_rows_seed",
        "target_test_rows_seed",
        "context_rows_seed",
        "auditing_rows_seed",
        "population_rows_seed",
        "total_rows",
        "num_features",
        "num_binary_features",
        "num_classes",
        "majority_class_pct",
        "minority_class_pct",
        "missing_pct",
        "outlier_pct",
        "duplicate_pct",
        "binary_feature_pct",
        "categorical_feature_pct",
        "target_accuracy_avg_pct",
        "imbalance_ratio",
        "samples_per_feature",
        "max_abs_corr_with_target",
        "knn_label_disagreement",
        "top_correlated_feature",
    ]
    available = [col for col in ordered_cols if col in profiles.columns]
    out = profiles[available]
    out.to_csv(output_dir / "data_profiles_original.csv", index=False)
    scope = str(profiles["profile_scope"].iloc[0]) if "profile_scope" in profiles.columns else "profile"
    safe_scope = re.sub(r"[^A-Za-z0-9_\-]+", "_", scope)
    out.to_csv(output_dir / f"data_profiles_{safe_scope}.csv", index=False)


def _bar_label(value: float) -> str:
    if pd.isna(value):
        return ""
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def plot_critical_values(profiles: pd.DataFrame, output_dir: Path) -> None:
    plot_df = profiles.copy()
    plot_df["dataset_display"] = pd.Categorical(
        plot_df["dataset_display"],
        categories=plot_df.sort_values("total_rows")["dataset_display"],
        ordered=True,
    )

    fig, axes = plt.subplots(3, 4, figsize=(24, 14), constrained_layout=True)
    for ax, metric in zip(axes.flat, CRITICAL_METRICS):
        if metric == "audit_population_rows_seed":
            required = {"auditing_rows_seed", "population_rows_seed"}
            if required.issubset(plot_df.columns):
                sample_df = plot_df.melt(
                    id_vars="dataset_display",
                    value_vars=["auditing_rows_seed", "population_rows_seed"],
                    var_name="sample_type",
                    value_name="rows",
                )
                sample_df["sample_type"] = sample_df["sample_type"].map(
                    {"auditing_rows_seed": "Auditing", "population_rows_seed": "Population"}
                )
                sns.barplot(data=sample_df, y="dataset_display", x="rows", hue="sample_type", ax=ax)
                ax.set_title("auditing and population rows")
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_xscale("log")
                ax.legend(title="", fontsize=7, loc="lower right")
            else:
                ax.axis("off")
                continue
        else:
            sns.barplot(data=plot_df, y="dataset_display", x=metric, ax=ax, color="#4477AA")
            ax.set_title(metric.replace("_", " "))
            ax.set_xlabel("")
            ax.set_ylabel("")
            if metric in {"total_rows", "num_features", "imbalance_ratio", "samples_per_feature"}:
                ax.set_xscale("log")

        for container in ax.containers:
            ax.bar_label(
                container,
                labels=[_bar_label(value) for value in container.datavalues],
                padding=3,
                fontsize=7,
            )
        ax.margins(x=0.16)

    for ax in axes.flat[len(CRITICAL_METRICS) :]:
        ax.axis("off")

    fig.suptitle("Saved Dataset Profiles: Critical Values", fontsize=18, y=1.02)
    fig.savefig(output_dir / "01_original_data_critical_values.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_level_correlations(
    profiles: pd.DataFrame,
    attacks: pd.DataFrame,
    output_dir: Path,
    attack_family: str,
    amia_metric: str | None = None,
) -> pd.DataFrame:
    subset = attacks[attacks["attack_family"] == attack_family].copy()
    if amia_metric is not None and "amia_metric" in subset.columns:
        subset = subset[subset["amia_metric"] == amia_metric].copy()
    if subset.empty:
        return pd.DataFrame()

    agg = aggregate_dataset_auc(subset)
    score_name = f"dataset_scores_{attack_family.lower()}{'_' + amia_metric if amia_metric else ''}.csv"
    agg.to_csv(output_dir / score_name, index=False)

    joined = profiles.merge(agg, left_on="dataset_name", right_on="dataset", how="inner")
    corr = spearman_correlations(joined, "auc", CORRELATION_METRICS)
    if corr.empty:
        return corr

    corr.to_csv(
        output_dir / f"correlations_{attack_family.lower()}{'_' + amia_metric if amia_metric else ''}.csv",
        index=False,
    )

    top = corr.head(16).sort_values("spearman_r")
    fig, ax = plt.subplots(figsize=(12, 8.5))
    colors = np.where(top["spearman_r"] >= 0, "#228833", "#CC6677")
    ax.barh(top["metric"].str.replace("_", " "), top["spearman_r"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    title_metric = f" ({amia_metric.replace('_auc', '')})" if amia_metric else ""
    ax.set_title(f"{attack_family}{title_metric}: Dataset Profiles vs Median Model AUC")
    ax.set_xlabel("Spearman correlation")
    ax.set_xlim(-1, 1)
    fig.savefig(
        output_dir / f"04_{attack_family.lower()}{'_' + amia_metric if amia_metric else ''}_dataset_correlations.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)
    return corr


def plot_model_correlation_heatmap(
    profiles: pd.DataFrame,
    attacks: pd.DataFrame,
    output_dir: Path,
    attack_family: str,
    amia_metric: str | None = None,
) -> None:
    subset = attacks[attacks["attack_family"] == attack_family].copy()
    if amia_metric is not None and "amia_metric" in subset.columns:
        subset = subset[subset["amia_metric"] == amia_metric].copy()
    if subset.empty:
        return

    rows = []
    for model, model_df in subset.groupby("model"):
        joined = profiles.merge(model_df, left_on="dataset_name", right_on="dataset", how="inner")
        corr = spearman_correlations(joined, "auc", CORRELATION_METRICS)
        for _, row in corr.iterrows():
            rows.append({"model": model, "metric": row["metric"], "spearman_r": row["spearman_r"]})
    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        return

    pivot = corr_df.pivot(index="metric", columns="model", values="spearman_r")
    pivot = pivot[[m for m in sorted(pivot.columns, key=model_sort_key)]]
    metric_strength = pivot.abs().mean(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[metric_strength.head(18).index]
    pivot.columns = [fmt_model(c) for c in pivot.columns]
    pivot.index = pivot.index.str.replace("_", " ")

    fig_height = max(7, 0.38 * len(pivot))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Spearman correlation"},
        ax=ax,
    )
    title_metric = f" ({amia_metric.replace('_auc', '')})" if amia_metric else ""
    ax.set_title(f"{attack_family}{title_metric}: Dataset Profiles vs AUC by Model")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.savefig(
        output_dir / f"05_{attack_family.lower()}{'_' + amia_metric if amia_metric else ''}_model_correlation_heatmap.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_key_scatter(
    profiles: pd.DataFrame,
    attacks: pd.DataFrame,
    output_dir: Path,
    attack_family: str,
    key_metrics: list[str],
    amia_metric: str | None = None,
) -> None:
    subset = attacks[attacks["attack_family"] == attack_family].copy()
    if amia_metric is not None and "amia_metric" in subset.columns:
        subset = subset[subset["amia_metric"] == amia_metric].copy()
    if subset.empty:
        return

    agg = aggregate_dataset_auc(subset)
    joined = profiles.merge(agg, left_on="dataset_name", right_on="dataset", how="inner")
    if joined.empty:
        return

    n_cols = 2
    n_rows = math.ceil(len(key_metrics) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for ax, metric in zip(axes, key_metrics):
        sns.regplot(data=joined, x=metric, y="auc", ax=ax, color="#4477AA", scatter_kws={"s": 55})
        for _, row in joined.iterrows():
            ax.annotate(row["dataset_display"], (row[metric], row["auc"]), fontsize=8, alpha=0.8)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(metric.replace("_", " "))
        ax.set_ylabel("Median model AUC")
        ax.set_xlabel("")

    for ax in axes[len(key_metrics) :]:
        ax.axis("off")

    title_metric = f" ({amia_metric.replace('_auc', '')})" if amia_metric else ""
    fig.suptitle(f"{attack_family}{title_metric}: Key Dataset Profile Values", fontsize=16, y=1.02)
    fig.savefig(
        output_dir / f"06_{attack_family.lower()}{'_' + amia_metric if amia_metric else ''}_key_property_scatter.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)



def plot_class_granularity_scatter(
    profiles: pd.DataFrame,
    attacks: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot dataset class count against dataset-level MIA AUC."""
    if attacks.empty or "num_classes" not in profiles.columns:
        return

    rows = []
    plot_specs = [("RMIA", None), ("AMIA", "row_max_auc")]
    for attack_family, amia_metric in plot_specs:
        subset = attacks[attacks["attack_family"] == attack_family].copy()
        if amia_metric is not None and "amia_metric" in subset.columns:
            subset = subset[subset["amia_metric"] == amia_metric].copy()
        if subset.empty:
            continue

        agg = aggregate_dataset_auc(subset)
        joined = profiles.merge(agg, left_on="dataset_name", right_on="dataset", how="inner")
        if joined.empty:
            continue
        metric_label = "" if amia_metric is None else f" ({amia_metric.replace('_auc', '')})"
        joined["plot_attack"] = f"{attack_family}{metric_label}"
        rows.append(joined)

    if not rows:
        return

    plot_df = pd.concat(rows, ignore_index=True, sort=False)
    export_cols = [
        "dataset_name",
        "dataset_display",
        "plot_attack",
        "num_classes",
        "auc",
        "mean_auc",
        "n_models",
        "total_rows",
        "target_model_accuracy_pct",
        "train_test_gap_pct_points",
    ]
    plot_df[[col for col in export_cols if col in plot_df.columns]].to_csv(
        output_dir / "07_class_granularity_vs_mia_auc.csv",
        index=False,
    )

    attacks_order = list(dict.fromkeys(plot_df["plot_attack"]))
    fig, axes = plt.subplots(
        1,
        len(attacks_order),
        figsize=(7.5 * len(attacks_order), 5.6),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    max_rows = plot_df["total_rows"].max()
    min_rows = plot_df["total_rows"].min()
    row_span = max(max_rows - min_rows, 1)
    all_classes = sorted(plot_df["num_classes"].dropna().astype(int).unique())

    scatter = None
    for ax, attack_name in zip(axes, attacks_order):
        panel = plot_df[plot_df["plot_attack"] == attack_name].copy()
        panel = panel.sort_values(["num_classes", "auc", "dataset_display"]).reset_index(drop=True)
        sizes = 130 + 520 * (panel["total_rows"] - min_rows) / row_span

        scatter = ax.scatter(
            panel["num_classes"],
            panel["auc"],
            s=sizes,
            c=panel["target_model_accuracy_pct"],
            cmap="viridis",
            edgecolor="white",
            linewidth=0.9,
            alpha=0.88,
        )

        for idx, row in panel.iterrows():
            offset_x = 5 if idx % 2 == 0 else -5
            offset_y = 7 if idx % 3 != 0 else -12
            ax.annotate(
                row["dataset_display"],
                (row["num_classes"], row["auc"]),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                ha="left" if offset_x > 0 else "right",
                va="center",
                fontsize=8,
                alpha=0.92,
            )

        if len(panel) >= 3 and panel["num_classes"].nunique() >= 2:
            sns.regplot(
                data=panel,
                x="num_classes",
                y="auc",
                ax=ax,
                scatter=False,
                color="#333333",
                line_kws={"linewidth": 1.4, "alpha": 0.65},
                ci=None,
            )

        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(attack_name)
        ax.set_xlabel("Number of classes")
        ax.set_ylabel("Median model MIA AUC")
        ax.set_xticks(all_classes)
        ax.set_ylim(max(0.45, plot_df["auc"].min() - 0.05), min(1.02, plot_df["auc"].max() + 0.05))
        ax.grid(True, axis="both", linewidth=0.5, alpha=0.35)

        rho_df = panel[["num_classes", "auc"]].dropna()
        if len(rho_df) >= 3 and rho_df["num_classes"].nunique() >= 2 and rho_df["auc"].nunique() >= 2:
            rho = rho_df["num_classes"].corr(rho_df["auc"], method="spearman")
            ax.text(
                0.02,
                0.96,
                f"Spearman r = {rho:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.85},
            )

    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes, shrink=0.82, pad=0.02)
        cbar.set_label("Target model accuracy (%)")

    legend_sizes = [min_rows, np.median(plot_df["total_rows"]), max_rows]
    handles = [
        plt.scatter([], [], s=130 + 520 * (value - min_rows) / row_span, color="#999999", alpha=0.7)
        for value in legend_sizes
    ]
    axes[-1].legend(
        handles,
        [f"{value:,.0f} rows" for value in legend_sizes],
        title="Profile rows",
        loc="lower right",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )

    fig.suptitle("Class Granularity vs Membership Inference Performance", fontsize=16)
    fig.savefig(output_dir / "07_class_granularity_vs_mia_auc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_class_granularity_scatter_07a(
    profiles: pd.DataFrame,
    attacks: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot 07a: RMIA only. Two panels on num_classes vs MIA AUC.
    Left panel: point size = target_entropy. Right panel: point size = imbalance_ratio."""
    required = {"num_classes", "imbalance_ratio", "target_entropy"}
    if attacks.empty or not required.issubset(profiles.columns):
        return

    subset = attacks[attacks["attack_family"] == "RMIA"].copy()
    if subset.empty:
        return
    agg = aggregate_dataset_auc(subset)
    plot_df = profiles.merge(agg, left_on="dataset_name", right_on="dataset", how="inner")
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(["num_classes", "auc", "dataset_display"]).reset_index(drop=True)
    all_classes = sorted(plot_df["num_classes"].dropna().astype(int).unique())

    size_vars = [
        ("target_entropy", "Target entropy (bits)"),
        ("imbalance_ratio", "Imbalance ratio"),
    ]

    def _scale_sizes(series: pd.Series, lo: float = 80, hi: float = 650) -> pd.Series:
        vmin, vmax = series.min(), series.max()
        if vmax == vmin:
            return pd.Series([((lo + hi) / 2)] * len(series), index=series.index)
        return lo + (hi - lo) * (series - vmin) / (vmax - vmin)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True, constrained_layout=True)
    dot_color = "#4878CF"

    for ax, (size_col, size_label) in zip(axes, size_vars):
        panel = plot_df.copy()
        sizes = _scale_sizes(panel[size_col].fillna(panel[size_col].median()))

        ax.scatter(
            panel["num_classes"],
            panel["auc"],
            s=sizes,
            c=dot_color,
            edgecolor="white",
            linewidth=0.9,
            alpha=0.88,
        )

        for idx, r in panel.iterrows():
            offset_x = 5 if idx % 2 == 0 else -5
            offset_y = 7 if idx % 3 != 0 else -12
            ax.annotate(
                r["dataset_display"],
                (r["num_classes"], r["auc"]),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                ha="left" if offset_x > 0 else "right",
                va="center",
                fontsize=8,
                alpha=0.92,
            )

        valid = panel[["num_classes", "auc"]].dropna()
        if len(valid) >= 3 and valid["num_classes"].nunique() >= 2:
            sns.regplot(
                data=valid,
                x="num_classes",
                y="auc",
                ax=ax,
                scatter=False,
                color="#333333",
                line_kws={"linewidth": 1.4, "alpha": 0.65},
                ci=None,
            )

        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Number of classes")
        ax.set_ylabel("Median model MIA AUC")
        ax.set_xticks(all_classes)
        ax.set_ylim(max(0.45, plot_df["auc"].min() - 0.05), min(1.02, plot_df["auc"].max() + 0.05))
        ax.grid(True, axis="both", linewidth=0.5, alpha=0.35)

        rho_df = valid
        if len(rho_df) >= 3 and rho_df["num_classes"].nunique() >= 2 and rho_df["auc"].nunique() >= 2:
            rho = rho_df["num_classes"].corr(rho_df["auc"], method="spearman")
            ax.text(
                0.02, 0.96,
                f"Spearman r = {rho:.2f}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.85},
            )

        # Size legend: show min, median, max of the size variable
        raw_vals = panel[size_col].dropna()
        legend_vals = [raw_vals.min(), raw_vals.median(), raw_vals.max()]
        legend_sizes = _scale_sizes(pd.Series(legend_vals, dtype=float))
        fmt = ".2f"
        handles = [
            plt.scatter([], [], s=s, color="#999999", alpha=0.7)
            for s in legend_sizes
        ]
        ax.legend(
            handles,
            [f"{v:{fmt}}" for v in legend_vals],
            title=size_label,
            loc="lower right",
            frameon=True,
            fontsize=8,
            title_fontsize=8,
        )

    fig.suptitle("RMIA — Number of Classes vs MIA AUC (point size encodes dataset property)", fontsize=13)
    fig.savefig(output_dir / "07a_class_binary_features_vs_mia_auc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _tpr_at_fpr_binary(member: np.ndarray, score: np.ndarray, max_fpr: float = 0.01) -> float:
    valid = np.isfinite(score)
    member = np.asarray(member)[valid].astype(int)
    score = np.asarray(score)[valid]
    if len(member) == 0 or len(np.unique(member)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(member, score)
    idx = np.where(fpr <= max_fpr)[0]
    return float(tpr[idx[-1]]) if len(idx) else 0.0


def build_per_class_tpr_rows(
    data_dir: Path,
    logs_dir: Path,
    profiles: pd.DataFrame,
    max_fpr: float = 0.01,
) -> pd.DataFrame:
    """Compute per-class RMIA TPR at a strict FPR from saved attack scores."""
    allowed = set(profiles["dataset_name"])
    profile_lookup = profiles.set_index("dataset_name")
    rows = []

    for result_path in sorted(logs_dir.glob("*/*/seed*/rmia/report/exp/attack_result_0.npz")):
        rel = result_path.relative_to(logs_dir).parts
        if len(rel) < 7:
            continue
        dataset, model, seed_part = rel[:3]
        if dataset not in allowed or not re.fullmatch(r"seed\d+", seed_part):
            continue

        data_path = data_dir / f"{dataset}.csv"
        split_dir = logs_dir / dataset / model / seed_part / "rmia" / "splits"
        context_path = split_dir / "context_indices_original.npy"
        if not data_path.exists() or not context_path.exists():
            continue

        try:
            result = np.load(result_path, allow_pickle=True)
            scores = np.asarray(result["scores"], dtype=float)
            members = np.asarray(result["memberships"]).astype(bool)
            raw = pd.read_csv(data_path, header=None)
            transformed = prepare_like_rmia(raw)
            context_original = np.load(context_path).astype(int)
        except Exception:
            continue

        if len(scores) != len(members) or len(scores) != len(context_original):
            continue

        labels = transformed.iloc[context_original, -1].reset_index(drop=True).to_numpy()
        # Derive num_classes from the actual audit labels, not the profile lookup,
        # so it reflects all classes present in context_original.
        num_classes = int(pd.Series(labels).dropna().nunique())
        for class_label in sorted(pd.Series(labels).dropna().unique()):
            class_mask = labels == class_label
            class_members = members[class_mask]
            class_scores = scores[class_mask]
            n_members = int(class_members.sum())
            n_nonmembers = int((~class_members).sum())
            if n_members == 0 or n_nonmembers == 0:
                tpr = np.nan
            else:
                tpr = _tpr_at_fpr_binary(class_members, class_scores, max_fpr=max_fpr)
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_display": fmt_dataset(dataset),
                    "model": model,
                    "seed": int(seed_part.replace("seed", "")),
                    "class_label": class_label,
                    "num_classes": num_classes,
                    "class_size": int(class_mask.sum()),
                    "n_members": n_members,
                    "n_nonmembers": n_nonmembers,
                    "tpr_at_1pct_fpr": tpr,
                }
            )

    return pd.DataFrame(rows)


def plot_per_class_tpr_dotplot(
    data_dir: Path,
    logs_dir: Path,
    profiles: pd.DataFrame,
    output_dir: Path,
) -> None:
    rows = build_per_class_tpr_rows(data_dir, logs_dir, profiles, max_fpr=0.01)
    if rows.empty:
        return

    rows.to_csv(output_dir / "08_per_class_tpr_at_1pct_fpr_runs.csv", index=False)
    summary = (
        rows.dropna(subset=["tpr_at_1pct_fpr"])
        .groupby(["dataset", "dataset_display", "class_label", "num_classes"], as_index=False)
        .agg(
            tpr_at_1pct_fpr=("tpr_at_1pct_fpr", "median"),
            tpr_mean=("tpr_at_1pct_fpr", "mean"),
            tpr_std=("tpr_at_1pct_fpr", "std"),
            class_size=("class_size", "median"),
            n_members=("n_members", "median"),
            n_nonmembers=("n_nonmembers", "median"),
            n_runs=("tpr_at_1pct_fpr", "count"),
            n_models=("model", "nunique"),
            n_seeds=("seed", "nunique"),
        )
    )
    if summary.empty:
        return

    dataset_order = (
        profiles[profiles["dataset_name"].isin(summary["dataset"])]
        .sort_values(["num_classes", "dataset_display"])["dataset_name"]
        .tolist()
    )
    label_map = {
        row.dataset_name: f"{row.dataset_display}\n({int(row.num_classes)} classes)"
        for row in profiles[profiles["dataset_name"].isin(summary["dataset"])].itertuples()
    }
    summary["dataset_plot"] = pd.Categorical(
        summary["dataset"].map(label_map),
        categories=[label_map[d] for d in dataset_order],
        ordered=True,
    )
    summary.to_csv(output_dir / "08_per_class_tpr_at_1pct_fpr_summary.csv", index=False)

    fig_width = max(10, 1.45 * len(dataset_order))
    fig, ax = plt.subplots(figsize=(fig_width, 6.4), constrained_layout=True)
    palette = sns.color_palette("viridis", n_colors=summary["num_classes"].nunique())
    sns.stripplot(
        data=summary,
        x="dataset_plot",
        y="tpr_at_1pct_fpr",
        hue="num_classes",
        palette=palette,
        jitter=0.22,
        size=8,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
        ax=ax,
    )

    medians = summary.groupby("dataset_plot", observed=True)["tpr_at_1pct_fpr"].median()
    for xpos, (_, median) in enumerate(medians.items()):
        ax.hlines(median, xpos - 0.28, xpos + 0.28, colors="#222222", linewidth=2.0, zorder=3)

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.45)
    ax.axhline(0.01, color="#666666", linestyle="--", linewidth=1, alpha=0.55)
    ax.set_title("Per-Class Membership Recovery at 1% FPR")
    ax.set_xlabel("")
    ax.set_ylabel("Class-wise TPR@1%FPR, median over RMIA models/seeds")
    ax.set_ylim(-0.02, min(1.02, summary["tpr_at_1pct_fpr"].max() + 0.08))
    ax.tick_params(axis="x", labelrotation=0)
    ax.legend(title="Classes", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.35)

    fig.savefig(output_dir / "08_per_class_tpr_at_1pct_fpr_dotplot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_min_per_class_tpr(
    data_dir: Path,
    logs_dir: Path,
    profiles: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot the hardest class per dataset using min class-wise TPR@1%FPR."""
    summary_path = output_dir / "08_per_class_tpr_at_1pct_fpr_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
    else:
        rows = build_per_class_tpr_rows(data_dir, logs_dir, profiles, max_fpr=0.01)
        if rows.empty:
            return
        summary = (
            rows.dropna(subset=["tpr_at_1pct_fpr"])
            .groupby(["dataset", "dataset_display", "class_label", "num_classes"], as_index=False)
            .agg(
                tpr_at_1pct_fpr=("tpr_at_1pct_fpr", "median"),
                class_size=("class_size", "median"),
                n_members=("n_members", "median"),
                n_nonmembers=("n_nonmembers", "median"),
                n_runs=("tpr_at_1pct_fpr", "count"),
                n_models=("model", "nunique"),
                n_seeds=("seed", "nunique"),
            )
        )
    if summary.empty:
        return

    idx = summary.groupby("dataset")["tpr_at_1pct_fpr"].idxmin()
    min_df = summary.loc[idx].copy()
    med_df = (
        summary.groupby("dataset", as_index=False)["tpr_at_1pct_fpr"]
        .median()
        .rename(columns={"tpr_at_1pct_fpr": "median_class_tpr_at_1pct_fpr"})
    )
    min_df = min_df.merge(med_df, on="dataset", how="left")

    profile_cols = ["dataset_name", "dataset_display", "num_classes", "total_rows"]
    profile_cols = [c for c in profile_cols if c in profiles.columns]
    profile_meta = profiles[profile_cols].rename(columns={"dataset_name": "dataset"})
    min_df = min_df.drop(columns=[c for c in ["dataset_display", "num_classes"] if c in min_df.columns], errors="ignore")
    min_df = min_df.merge(profile_meta, on="dataset", how="left")
    min_df = min_df.rename(columns={"tpr_at_1pct_fpr": "min_class_tpr_at_1pct_fpr"})
    min_df = min_df.sort_values(["num_classes", "min_class_tpr_at_1pct_fpr", "dataset_display"])
    min_df["dataset_plot"] = min_df.apply(
        lambda row: f"{row['dataset_display']}\n({int(row['num_classes'])} classes)", axis=1
    )
    min_df.to_csv(output_dir / "09_min_per_class_tpr_at_1pct_fpr.csv", index=False)

    fig_width = max(9, 1.35 * len(min_df))
    fig, ax = plt.subplots(figsize=(fig_width, 5.8), constrained_layout=True)
    colors = sns.color_palette("viridis", n_colors=max(int(min_df["num_classes"].nunique()), 1))
    class_values = sorted(min_df["num_classes"].dropna().astype(int).unique())
    color_map = {value: colors[i] for i, value in enumerate(class_values)}
    bar_colors = [color_map[int(value)] for value in min_df["num_classes"]]

    bars = ax.bar(
        min_df["dataset_plot"],
        min_df["min_class_tpr_at_1pct_fpr"],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.9,
        alpha=0.88,
    )
    ax.scatter(
        min_df["dataset_plot"],
        min_df["median_class_tpr_at_1pct_fpr"],
        color="#222222",
        s=46,
        zorder=4,
        label="Median class TPR",
    )

    for bar, row in zip(bars, min_df.itertuples(index=False)):
        label = f"class {row.class_label}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.45)
    ax.axhline(0.01, color="#666666", linestyle="--", linewidth=1, alpha=0.55, label="1% TPR reference")
    ax.set_title("Hardest Class Leakage at 1% FPR")
    ax.set_xlabel("")
    ax.set_ylabel("Minimum class-wise TPR@1%FPR")
    ax.set_ylim(0, min(1.05, max(0.08, min_df["median_class_tpr_at_1pct_fpr"].max() + 0.12)))
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.35)

    class_handles = [
        Patch(facecolor=color_map[value], edgecolor="white", label=f"{value} classes")
        for value in class_values
    ]
    median_handle = Line2D([0], [0], marker="o", color="none", markerfacecolor="#222222", markersize=7, label="Median class TPR")
    ax.legend(handles=class_handles + [median_handle], title="Dataset granularity", loc="upper left", frameon=True)

    fig.savefig(output_dir / "09_min_per_class_tpr_at_1pct_fpr.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_classical_locations_credit_class_tpr_heatmap(
    data_dir: Path,
    logs_dir: Path,
    profiles: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Heatmap of class-wise low-FPR recovery for classical models on two datasets."""
    target_datasets = ["locations", "credit_rating"]
    classical_models = ["rf", "lightgbm", "mlp", "tabnet"]
    profile_subset = profiles[profiles["dataset_name"].isin(target_datasets)].copy()
    if profile_subset.empty:
        return

    rows = build_per_class_tpr_rows(data_dir, logs_dir, profile_subset, max_fpr=0.01)
    if rows.empty:
        return
    rows = rows[rows["model"].isin(classical_models)].copy()
    rows = rows[rows["dataset"].isin(target_datasets)].copy()
    if rows.empty:
        return

    summary = (
        rows.dropna(subset=["tpr_at_1pct_fpr"])
        .groupby(["dataset", "dataset_display", "class_label", "num_classes"], as_index=False)
        .agg(
            tpr_at_1pct_fpr=("tpr_at_1pct_fpr", "median"),
            tpr_mean=("tpr_at_1pct_fpr", "mean"),
            tpr_std=("tpr_at_1pct_fpr", "std"),
            class_size=("class_size", "median"),
            n_members=("n_members", "median"),
            n_nonmembers=("n_nonmembers", "median"),
            n_runs=("tpr_at_1pct_fpr", "count"),
            n_models=("model", "nunique"),
            n_seeds=("seed", "nunique"),
        )
    )
    if summary.empty:
        return

    summary["class_label"] = summary["class_label"].astype(int)
    dataset_labels = {
        "locations": "locations",
        "credit_rating": "credit_rating",
    }
    summary["dataset_plot"] = summary["dataset"].map(dataset_labels)
    summary.to_csv(output_dir / "10_locations_credit_classical_class_tpr_heatmap.csv", index=False)

    max_class = int(summary["class_label"].max())
    columns = list(range(max_class + 1))
    pivot = (
        summary.pivot(index="dataset_plot", columns="class_label", values="tpr_at_1pct_fpr")
        .reindex(index=[dataset_labels[d] for d in target_datasets], columns=columns)
    )

    annot = pivot.copy()
    annot = annot.map(lambda value: "" if pd.isna(value) else f"{value:.2f}")

    fig, ax = plt.subplots(figsize=(12, 3.4), constrained_layout=True)
    sns.heatmap(
        pivot,
        annot=annot,
        fmt="",
        cmap="mako",
        vmin=0,
        vmax=max(0.25, float(np.nanmax(pivot.to_numpy()))),
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "Median class TPR@1%FPR"},
        ax=ax,
    )
    ax.set_title("Class-wise Membership Recovery for Classical ML Models")
    ax.set_xlabel("Class label")
    ax.set_ylabel("")
    ax.set_xticklabels([str(c) for c in columns], rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    footer = "Classical models: RF, LightGBM, MLP, TabNet. Cells are medians over models and seeds."
    fig.text(0.01, -0.03, footer, ha="left", va="top", fontsize=9)
    fig.savefig(output_dir / "10_locations_credit_classical_class_tpr_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_classical_density_with_class_heatmap(
    data_dir: Path,
    logs_dir: Path,
    profiles: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Density separation panels plus model-by-class TPR heatmaps."""
    target_datasets = ["locations", "us_stocks_financial"]
    classical_models = ["rf", "lightgbm", "mlp", "tabnet"]
    model_labels = {
        "rf": "RF",
        "lightgbm": "LightGBM",
        "mlp": "MLP",
        "tabnet": "TabNet",
    }
    profile_subset = profiles[profiles["dataset_name"].isin(target_datasets)].copy()
    if profile_subset.empty:
        return

    density_rows = []
    for result_path in sorted(logs_dir.glob("*/*/seed*/rmia/report/exp/attack_result_0.npz")):
        rel = result_path.relative_to(logs_dir).parts
        if len(rel) < 7:
            continue
        dataset, model, seed_part = rel[:3]
        if dataset not in target_datasets or model not in classical_models or not re.fullmatch(r"seed\d+", seed_part):
            continue
        try:
            result = np.load(result_path, allow_pickle=True)
            scores = np.asarray(result["scores"], dtype=float)
            members = np.asarray(result["memberships"]).astype(bool)
        except Exception:
            continue
        if len(scores) != len(members):
            continue
        density_rows.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "dataset_display": fmt_dataset(dataset),
                    "model": model,
                    "model_display": model_labels[model],
                    "seed": int(seed_part.replace("seed", "")),
                    "score": scores,
                    "membership": np.where(members, "Member", "Non-member"),
                }
            )
        )

    if not density_rows:
        return
    density_df = pd.concat(density_rows, ignore_index=True)
    density_df.to_csv(output_dir / "11_classical_density_scores_locations_us_stocks.csv", index=False)

    tpr_rows = build_per_class_tpr_rows(data_dir, logs_dir, profile_subset, max_fpr=0.01)
    if tpr_rows.empty:
        return
    tpr_rows = tpr_rows[tpr_rows["dataset"].isin(target_datasets) & tpr_rows["model"].isin(classical_models)].copy()
    if tpr_rows.empty:
        return
    tpr_summary = (
        tpr_rows.dropna(subset=["tpr_at_1pct_fpr"])
        .groupby(["dataset", "model", "class_label", "num_classes"], as_index=False)
        .agg(
            tpr_at_1pct_fpr=("tpr_at_1pct_fpr", "median"),
            tpr_mean=("tpr_at_1pct_fpr", "mean"),
            tpr_std=("tpr_at_1pct_fpr", "std"),
            class_size=("class_size", "median"),
            n_members=("n_members", "median"),
            n_nonmembers=("n_nonmembers", "median"),
            n_runs=("tpr_at_1pct_fpr", "count"),
            n_seeds=("seed", "nunique"),
        )
    )
    if tpr_summary.empty:
        return
    tpr_summary["model_display"] = tpr_summary["model"].map(model_labels)
    tpr_summary["class_label"] = tpr_summary["class_label"].astype(int)
    tpr_summary.to_csv(output_dir / "11_classical_model_class_tpr_heatmap_locations_us_stocks.csv", index=False)

    fig, axes = plt.subplots(
        len(target_datasets),
        len(classical_models) + 1,
        figsize=(22, 8.8),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.32]},
        constrained_layout=True,
    )

    member_palette = {"Non-member": "#4C78A8", "Member": "#E45756"}
    heatmap_mappable = None
    for row_idx, dataset in enumerate(target_datasets):
        dataset_density = density_df[density_df["dataset"] == dataset]
        dataset_tpr = tpr_summary[tpr_summary["dataset"] == dataset]
        dataset_label = fmt_dataset(dataset)

        for col_idx, model in enumerate(classical_models):
            ax = axes[row_idx, col_idx]
            panel = dataset_density[dataset_density["model"] == model]
            if panel.empty:
                ax.axis("off")
                continue
            for membership, color in member_palette.items():
                subset = panel[panel["membership"] == membership]
                if subset["score"].nunique() > 1:
                    sns.kdeplot(
                        data=subset,
                        x="score",
                        ax=ax,
                        color=color,
                        fill=True,
                        alpha=0.24,
                        linewidth=1.8,
                        warn_singular=False,
                        label=membership,
                    )
                else:
                    ax.hist(subset["score"], bins=20, density=True, color=color, alpha=0.24, label=membership)
            ax.set_title(model_labels[model] if row_idx == 0 else "")
            ax.set_xlabel("RMIA score")
            ax.set_ylabel(f"{dataset_label}\nDensity" if col_idx == 0 else "")
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=True, fontsize=8)
            elif ax.get_legend() is not None:
                ax.get_legend().remove()
            ax.grid(True, axis="y", linewidth=0.4, alpha=0.3)

        ax_hm = axes[row_idx, -1]
        max_class = int(dataset_tpr["class_label"].max()) if not dataset_tpr.empty else 0
        columns = list(range(max_class + 1))
        pivot = (
            dataset_tpr.pivot(index="model_display", columns="class_label", values="tpr_at_1pct_fpr")
            .reindex(index=[model_labels[m] for m in classical_models], columns=columns)
        )
        class_sizes = (
            dataset_tpr.groupby("class_label")["class_size"]
            .median()
            .reindex(columns)
            .astype(float)
        )
        size_strip = pd.DataFrame([class_sizes.to_numpy()], index=["class size"], columns=columns)
        size_annot = pd.DataFrame(
            [["" if pd.isna(value) else f"n={int(value):,}" for value in class_sizes]],
            index=["class size"],
            columns=columns,
        )
        ax_size = ax_hm.inset_axes([0, 1.035, 1, 0.105])
        sns.heatmap(
            pd.DataFrame(np.ones_like(size_strip.to_numpy()), index=size_strip.index, columns=size_strip.columns),
            annot=size_annot,
            fmt="",
            cmap=sns.color_palette(["#E6E6E6"], as_cmap=True),
            vmin=0,
            vmax=1,
            linewidths=0.45,
            linecolor="white",
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            annot_kws={"fontsize": 6.2, "color": "#444444"},
            ax=ax_size,
        )
        ax_size.set_title("Class size", fontsize=7, pad=1)
        ax_size.set_xlabel("")
        ax_size.set_ylabel("")

        annot = pivot.map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
        hm = sns.heatmap(
            pivot,
            annot=annot,
            fmt="",
            cmap="mako_r",
            vmin=0,
            vmax=max(0.25, float(np.nanmax(tpr_summary["tpr_at_1pct_fpr"].to_numpy()))),
            linewidths=0.6,
            linecolor="white",
            cbar=row_idx == 0,
            cbar_kws={"label": "TPR@1%FPR"},
            ax=ax_hm,
        )
        heatmap_mappable = hm
        ax_hm.set_title("Class-wise leakage" if row_idx == 0 else "", pad=28)
        ax_hm.set_xlabel("Class label")
        ax_hm.set_ylabel("")
        ax_hm.set_xticklabels([str(c) for c in columns], rotation=0)
        ax_hm.set_yticklabels(ax_hm.get_yticklabels(), rotation=0)

    fig.suptitle("Classical ML: Member/Non-member Score Separation and Class-wise Leakage", fontsize=16)
    fig.savefig(output_dir / "11_classical_density_plus_class_tpr_heatmap_locations_us_stocks.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_binary_vs_multiclass_classical_summary(
    data_dir: Path,
    logs_dir: Path,
    profiles: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Compare class-wise low-FPR recovery for binary vs multiclass datasets."""
    classical_models = ["rf", "lightgbm", "mlp", "tabnet"]
    model_labels = {
        "rf": "RF",
        "lightgbm": "LightGBM",
        "mlp": "MLP",
        "tabnet": "TabNet",
    }
    rows = build_per_class_tpr_rows(data_dir, logs_dir, profiles, max_fpr=0.01)
    if rows.empty:
        return
    rows = rows[rows["model"].isin(classical_models)].copy()
    rows = rows.dropna(subset=["tpr_at_1pct_fpr"])
    if rows.empty:
        return

    summary = (
        rows.groupby(["dataset", "dataset_display", "model", "class_label", "num_classes"], as_index=False)
        .agg(
            tpr_at_1pct_fpr=("tpr_at_1pct_fpr", "median"),
            tpr_mean=("tpr_at_1pct_fpr", "mean"),
            tpr_std=("tpr_at_1pct_fpr", "std"),
            class_size=("class_size", "median"),
            n_members=("n_members", "median"),
            n_nonmembers=("n_nonmembers", "median"),
            n_seeds=("seed", "nunique"),
        )
    )
    summary["granularity_group"] = np.where(summary["num_classes"] <= 2, "Binary", "Multiclass")
    summary["model_display"] = summary["model"].map(model_labels)
    summary["class_label"] = summary["class_label"].astype(int)
    summary.to_csv(output_dir / "12_binary_vs_multiclass_classical_per_class_tpr.csv", index=False)

    dataset_order = (
        summary.groupby(["dataset", "dataset_display", "num_classes"], as_index=False)["tpr_at_1pct_fpr"]
        .median()
        .sort_values(["num_classes", "dataset_display"])
    )
    hue_order = dataset_order["dataset_display"].tolist()
    group_order = ["Binary", "Multiclass"]
    model_order = [model_labels[m] for m in classical_models]

    fig, axes = plt.subplots(1, len(model_order), figsize=(17.5, 5.6), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    palette = dict(zip(hue_order, sns.color_palette("tab10", n_colors=max(len(hue_order), 3))))
    markers = ["o", "s", "D", "^", "v", "P", "X", "<", ">"]
    marker_map = {dataset: markers[i % len(markers)] for i, dataset in enumerate(hue_order)}

    for ax, model_name in zip(axes, model_order):
        panel = summary[summary["model_display"] == model_name].copy()
        sns.boxplot(
            data=panel,
            x="granularity_group",
            y="tpr_at_1pct_fpr",
            order=group_order,
            color="white",
            width=0.48,
            fliersize=0,
            linewidth=1.1,
            ax=ax,
        )
        for patch in ax.patches:
            patch.set_alpha(0.55)

        for dataset_name in hue_order:
            sub = panel[panel["dataset_display"] == dataset_name]
            if sub.empty:
                continue
            sns.stripplot(
                data=sub,
                x="granularity_group",
                y="tpr_at_1pct_fpr",
                order=group_order,
                color=palette[dataset_name],
                marker=marker_map[dataset_name],
                jitter=0.18,
                size=6.8,
                alpha=0.88,
                edgecolor="white",
                linewidth=0.55,
                ax=ax,
            )

        group_medians = panel.groupby("granularity_group")["tpr_at_1pct_fpr"].median()
        for xpos, group in enumerate(group_order):
            if group in group_medians.index:
                ax.hlines(group_medians[group], xpos - 0.24, xpos + 0.24, color="#111111", linewidth=2.2, zorder=4)

        ax.axhline(0.01, color="#666666", linestyle="--", linewidth=1, alpha=0.55)
        ax.set_title(model_name)
        ax.set_xlabel("")
        ax.set_ylabel("Class-wise TPR@1%FPR" if ax is axes[0] else "")
        ax.set_ylim(-0.03, min(1.05, max(0.08, summary["tpr_at_1pct_fpr"].max() + 0.08)))
        ax.grid(True, axis="y", linewidth=0.45, alpha=0.35)

    handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map[name],
            color="none",
            markerfacecolor=palette[name],
            markeredgecolor="white",
            markersize=8,
            label=name,
        )
        for name in hue_order
    ]
    fig.legend(handles=handles, title="Dataset", loc="outside right center", frameon=True)
    fig.suptitle("Binary vs Multiclass: Class-wise Membership Recovery for Classical ML", fontsize=16)
    fig.savefig(output_dir / "12_binary_vs_multiclass_classical_per_class_tpr.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    pooled = (
        summary.groupby(["granularity_group", "dataset", "dataset_display", "num_classes"], as_index=False)
        .agg(
            median_class_tpr_at_1pct_fpr=("tpr_at_1pct_fpr", "median"),
            mean_class_tpr_at_1pct_fpr=("tpr_at_1pct_fpr", "mean"),
            min_class_tpr_at_1pct_fpr=("tpr_at_1pct_fpr", "min"),
            max_class_tpr_at_1pct_fpr=("tpr_at_1pct_fpr", "max"),
            n_class_model_points=("tpr_at_1pct_fpr", "count"),
        )
    )
    pooled.to_csv(output_dir / "12_binary_vs_multiclass_classical_dataset_summary.csv", index=False)

def write_recommendation(output_dir: Path, profile_csv: Path, profiles: pd.DataFrame) -> None:
    scope = profiles["profile_scope"].iloc[0] if "profile_scope" in profiles.columns else "unknown"
    text = f"""# Data Visualization Notes

Profile scope: `{scope}`.

The correlation tables are computed from the same profile scope used by plot 01: target model 0's seed-1 RMIA training records. They may include additional derived statistics, but those statistics come from the same split, seed, and target-model profile.

By default, data properties are computed on target model 0's seed-1 RMIA training
records, using the saved RMIA split indices. This is the most direct profile for
membership-risk analysis because membership is defined by target-model training
inclusion. Use `--profile-scope full_saved` to switch back to `{profile_csv}` for
a train+test/full-dataset descriptive profile.

Correlation basis:

1. Each model/dataset AUC row is already a seed-summary value from the attack
   result tables. The script records the available seed count in
   `attack_auc_join_source.csv` and the dataset-level score files.

2. The headline correlation uses the median AUC across models for each dataset.
   The `n_datasets` column in the correlation CSV is therefore the number of
   datasets in the correlation, not the number of seeds.

3. For membership-risk interpretation, the target-training split is preferable
   to train+test because only training rows are members. Train+test/full profiles
   are better for a descriptive dataset table, not for explaining membership risk.

4. This run uses seed 1 for the profile because that was requested. The attack
   AUCs are still seed-summary values; for a fully symmetric analysis, the next
   refinement would average the split profiles across the same seeds used by the
   attack summaries.

5. AMIA uses `row_max_auc` only.
"""
    (output_dir / "README_data_viz.md").write_text(text)


def _dataset_key(name: str) -> str:
    return re.sub(r"^\d+[_-]", "", str(name)).lower().replace("_", "-")


def _filter_excluded_datasets(df: pd.DataFrame, excluded: set[str], column: str = "dataset_name") -> pd.DataFrame:
    if df.empty or not excluded or column not in df.columns:
        return df
    excluded_keys = {_dataset_key(item) for item in excluded}
    return df[~df[column].map(lambda value: _dataset_key(value) in excluded_keys or str(value) in excluded)].copy()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--profile-csv", type=Path, default=DEFAULT_PROFILE_CSV)
    parser.add_argument("--attacks-dir", type=Path, default=DEFAULT_ATTACKS_DIR)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument(
        "--profile-scope",
        choices=["seed_target_train", "seed_context", "seed_target_train_test", "seed_context_plus_population", "full_saved"],
        default="seed_target_train",
        help="Which records to profile. Default aligns with target model 0's training members for the selected seed.",
    )
    parser.add_argument("--profile-seed", type=int, default=1)
    parser.add_argument("--split-model", default=None, help="Model folder to read split indices from; default uses the first available model.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--exclude-datasets",
        default="aloi,46956_seismic-bumps,lcld,purchases10",
        help="Comma-separated datasets to omit from data/profile visualizations. Use empty string to include all.",
    )
    args = parser.parse_args()
    excluded_datasets = {item.strip() for item in args.exclude_datasets.split(",") if item.strip()}

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 120

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.profile_scope == "full_saved":
        profiles = load_profiles(args.profile_csv, args.attacks_dir)
    else:
        profiles = load_rmia_split_profiles(
            args.data_dir,
            args.logs_dir,
            args.attacks_dir,
            args.profile_seed,
            args.profile_scope,
            args.split_model,
        )

    profiles = _filter_excluded_datasets(profiles, excluded_datasets, "dataset_name")

    seed_counts = load_seed_counts(args.logs_dir, profiles)
    seed_counts = _filter_excluded_datasets(seed_counts, excluded_datasets, "dataset")
    seed_counts.to_csv(args.output_dir / "seed_counts_by_dataset_model_attack.csv", index=False)
    model_metrics = load_model_metadata_metrics(args.logs_dir, profiles)
    model_metrics = _filter_excluded_datasets(model_metrics, excluded_datasets, "dataset")
    model_metrics.to_csv(args.output_dir / "target_model_metrics.csv", index=False)
    if not model_metrics.empty:
        target_acc_avg = (
            model_metrics.groupby("dataset", as_index=False)["target_model_accuracy_pct"]
            .mean()
            .rename(columns={"dataset": "dataset_name", "target_model_accuracy_pct": "target_accuracy_avg_pct"})
        )
        profiles = profiles.merge(target_acc_avg, on="dataset_name", how="left")

    save_profiles_table(profiles, args.output_dir)
    plot_critical_values(profiles, args.output_dir)

    rmia = load_rmia(args.attacks_dir, profiles, model_metrics, seed_counts)
    amia = load_amia(args.attacks_dir, profiles, model_metrics, seed_counts)
    attacks = pd.concat([rmia, amia], ignore_index=True, sort=False)
    attacks.to_csv(args.output_dir / "attack_auc_join_source.csv", index=False)

    rmia_corr = plot_dataset_level_correlations(profiles, attacks, args.output_dir, "RMIA")
    plot_model_correlation_heatmap(profiles, attacks, args.output_dir, "RMIA")
    rmia_keys = rmia_corr.head(4)["metric"].tolist() if not rmia_corr.empty else ["majority_class_pct"]
    plot_key_scatter(profiles, attacks, args.output_dir, "RMIA", rmia_keys)

    main_amia_metric = "row_max_auc"
    amia_corr = plot_dataset_level_correlations(
        profiles, attacks, args.output_dir, "AMIA", amia_metric=main_amia_metric
    )
    plot_model_correlation_heatmap(profiles, attacks, args.output_dir, "AMIA", amia_metric=main_amia_metric)
    amia_keys = amia_corr.head(4)["metric"].tolist() if not amia_corr.empty else ["majority_class_pct"]
    plot_key_scatter(profiles, attacks, args.output_dir, "AMIA", amia_keys, amia_metric=main_amia_metric)

    plot_class_granularity_scatter(profiles, attacks, args.output_dir)
    plot_class_granularity_scatter_07a(profiles, attacks, args.output_dir)
    plot_per_class_tpr_dotplot(args.data_dir, args.logs_dir, profiles, args.output_dir)
    plot_min_per_class_tpr(args.data_dir, args.logs_dir, profiles, args.output_dir)
    plot_classical_locations_credit_class_tpr_heatmap(args.data_dir, args.logs_dir, profiles, args.output_dir)
    plot_classical_density_with_class_heatmap(args.data_dir, args.logs_dir, profiles, args.output_dir)
    plot_binary_vs_multiclass_classical_summary(args.data_dir, args.logs_dir, profiles, args.output_dir)
    write_recommendation(args.output_dir, args.profile_csv, profiles)
    print(f"Saved data visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
