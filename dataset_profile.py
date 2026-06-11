"""Compute and persist rich dataset statistics before preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

TRAIN_RATIO = 0.75
DEFAULT_OUTPUT_CSV = "results_visualizations/dataset_profiles.csv"
KNN_K = 5
KNN_MAX_SAMPLE = 2000


def _knn_label_disagreement(X: np.ndarray, y: np.ndarray, k: int = KNN_K) -> float:
    """Fraction of samples whose k-NN majority label differs from their own label."""
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    _, indices = nbrs.kneighbors(X)
    neighbor_labels = y[indices[:, 1:]]  # exclude self
    majority = np.array([
        vals[np.argmax(counts)]
        for vals, counts in (
            np.unique(row, return_counts=True) for row in neighbor_labels
        )
    ])
    return round(float(np.mean(majority != y)), 6)


def _count_outliers_iqr(series: pd.Series) -> int:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())


def _entropy(series: pd.Series, bins: int = 10) -> float:
    """Shannon entropy (bits). Numerical columns are binned first."""
    if pd.api.types.is_numeric_dtype(series):
        counts, _ = np.histogram(series.dropna(), bins=bins)
    else:
        counts = series.value_counts().values
    counts = counts[counts > 0]
    if counts.sum() == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def compute_dataset_profile(df: pd.DataFrame, dataset_name: str) -> dict:
    """
    Compute statistics from a raw (pre-preprocessing) DataFrame.
    The last column is assumed to be the target/label.
    Outliers are computed on original column types, before any encoding.
    """
    if df is None or len(df) == 0:
        return {}

    n_rows = len(df)
    feature_cols = list(df.columns[:-1])
    label_col = df.columns[-1]

    # --- Split sizes ---
    train_size = int(n_rows * TRAIN_RATIO)
    test_size = n_rows - train_size

    # --- Feature types (raw, before encoding) ---
    numerical_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [
        c for c in feature_cols
        if pd.api.types.is_string_dtype(df[c])
        or pd.api.types.is_object_dtype(df[c])
        or isinstance(df[c].dtype, pd.CategoricalDtype)
    ]
    binary_cols = [c for c in feature_cols if df[c].nunique() <= 2]

    # --- Missing values (before fill, count '?' as missing too) ---
    raw_na = int(df.isna().sum().sum())
    question_mark_na = int((df.astype(str) == "?").sum().sum())
    missing_cells = raw_na + question_mark_na
    total_cells = n_rows * len(df.columns)
    missing_pct = round(100.0 * missing_cells / total_cells, 4) if total_cells else 0.0

    # --- Duplicate rows ---
    duplicate_rows = int(df.duplicated().sum())

    # --- Target label analysis ---
    label_series = df[label_col]
    n_classes = int(label_series.nunique())
    value_counts = label_series.value_counts()
    majority_class_pct = round(100.0 * float(value_counts.iloc[0]) / n_rows, 4)
    minority_class_pct = round(100.0 * float(value_counts.iloc[-1]) / n_rows, 4)
    min_count = float(value_counts.iloc[-1])
    imbalance_ratio = round(float(value_counts.iloc[0]) / min_count, 4) if min_count > 0 else float("inf")

    # Target entropy: Shannon entropy of class distribution
    probs = value_counts / n_rows
    target_entropy = round(float(-(probs * np.log2(probs + 1e-12)).sum()), 6)

    # Task type: classification if target is non-numeric or has few unique values
    task_type = (
        "classification"
        if (not pd.api.types.is_numeric_dtype(label_series)) or (n_classes / n_rows < 0.05)
        else "regression"
    )

    # --- Mean feature entropy (target excluded) ---
    feature_entropies = [_entropy(df[c]) for c in feature_cols]
    mean_feature_entropy = round(float(np.mean(feature_entropies)), 6) if feature_entropies else 0.0
    max_feature_entropy = round(float(np.max(feature_entropies)), 6) if feature_entropies else 0.0

    # --- Outliers: IQR on raw numerical features ---
    features_with_outliers = 0
    total_outlier_cells = 0
    for col in numerical_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        n_out = _count_outliers_iqr(s)
        if n_out > 0:
            features_with_outliers += 1
            total_outlier_cells += n_out

    outlier_pct = (
        round(100.0 * total_outlier_cells / (n_rows * len(numerical_cols)), 4)
        if numerical_cols else 0.0
    )

    # --- Correlation with target (minimally encoded for non-numeric columns) ---
    encoded = df[feature_cols].copy()
    for col in categorical_cols:
        encoded[col] = encoded[col].astype("category").cat.codes
    label_enc = (
        label_series if pd.api.types.is_numeric_dtype(label_series)
        else label_series.astype("category").cat.codes
    )

    correlations = {}
    for col in encoded.columns:
        try:
            r = float(encoded[col].corr(label_enc))
            if not np.isnan(r):
                correlations[str(col)] = round(abs(r), 6)
        except Exception:
            pass

    vals = list(correlations.values())
    mean_abs_corr = round(float(np.mean(vals)), 6) if vals else 0.0
    max_abs_corr = round(float(max(vals)), 6) if vals else 0.0
    top_correlated_feature = max(correlations, key=correlations.get) if correlations else ""

    # --- Numerical variance ---
    near_zero_var_count = 0
    mean_var = 0.0
    if numerical_cols:
        variances = df[numerical_cols].apply(pd.to_numeric, errors="coerce").var(ddof=0)
        near_zero_var_count = int((variances < 1e-6).sum())
        mean_var = round(float(variances.mean()), 6)


    # Samples per feature: low ratio (higher memorization risk)
    samples_per_feature = round(train_size / len(feature_cols), 4) if feature_cols else 0.0

    # Mean pairwise inter-feature correlation: high redundancy (simpler effective model)
    mean_pairwise_feature_corr = 0.0
    if len(numerical_cols) > 1:
        corr_matrix = encoded[numerical_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
        vals_corr = upper.stack().values
        mean_pairwise_feature_corr = round(float(np.nanmean(vals_corr)), 6) if len(vals_corr) else 0.0

    # Skewness / kurtosis of numerical features (abs mean skewness)
    mean_abs_skewness = 0.0
    mean_kurtosis = 0.0
    if numerical_cols:
        skews = [df[c].dropna().skew() for c in numerical_cols]
        kurts = [df[c].dropna().kurt() for c in numerical_cols]
        mean_abs_skewness = round(float(np.nanmean(np.abs(skews))), 6)
        mean_kurtosis = round(float(np.nanmean(kurts)), 6)

    # PCA effective dimensionality on all encoded features (standardised)
    pca_n_components_95pct = len(feature_cols)
    pca_explained_var_top5 = 1.0
    if len(feature_cols) >= 2 and n_rows > 1:
        try:
            X_all = encoded.fillna(0).values.astype(np.float32)
            X_scaled = StandardScaler().fit_transform(X_all)
            pca = PCA(n_components=min(len(feature_cols), n_rows - 1)).fit(X_scaled)
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            pca_n_components_95pct = int(np.searchsorted(cum_var, 0.95) + 1)
            pca_explained_var_top5 = round(float(np.sum(pca.explained_variance_ratio_[:5])), 6)
        except Exception:
            pass

    # k-NN label disagreement: proxy for label noise / class overlap (forces memorization)
    knn_label_disagreement = 0.0
    if len(feature_cols) >= 1 and n_rows > KNN_K:
        try:
            X_all = encoded.fillna(0).values.astype(np.float32)
            X_scaled = StandardScaler().fit_transform(X_all)
            y_arr = label_enc.values
            if n_rows > KNN_MAX_SAMPLE:
                rng = np.random.default_rng(42)
                idx = rng.choice(n_rows, KNN_MAX_SAMPLE, replace=False)
                X_scaled, y_arr = X_scaled[idx], y_arr[idx]
            knn_label_disagreement = _knn_label_disagreement(X_scaled, y_arr)
        except Exception:
            pass

    return {
        "dataset_name": dataset_name,
        "total_rows": n_rows,
        "train_rows": train_size,
        "test_rows": test_size,
        "num_features": len(feature_cols),
        "num_numerical_features": len(numerical_cols),
        "num_categorical_features": len(categorical_cols),
        "num_binary_features": len(binary_cols),
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "task_type": task_type,
        "num_classes": n_classes,
        "majority_class_pct": majority_class_pct,
        "minority_class_pct": minority_class_pct,
        "imbalance_ratio": imbalance_ratio,
        "target_entropy": target_entropy,
        "mean_feature_entropy": mean_feature_entropy,
        "max_feature_entropy": max_feature_entropy,
        "num_features_with_outliers": features_with_outliers,
        "total_outlier_cells": total_outlier_cells,
        "outlier_pct": outlier_pct,
        "mean_abs_corr_with_target": mean_abs_corr,
        "max_abs_corr_with_target": max_abs_corr,
        "top_correlated_feature": top_correlated_feature,
        "near_zero_variance_features": near_zero_var_count,
        "mean_numerical_variance": mean_var,
        # MIA-specific
        "samples_per_feature": samples_per_feature,
        "mean_pairwise_feature_corr": mean_pairwise_feature_corr,
        "mean_abs_skewness": mean_abs_skewness,
        "mean_kurtosis": mean_kurtosis,
        "pca_n_components_95pct": pca_n_components_95pct,
        "pca_explained_var_top5": pca_explained_var_top5,
        "knn_label_disagreement": knn_label_disagreement,
    }


def save_dataset_profile(
    df: pd.DataFrame,
    dataset_name: str,
    output_csv: str = DEFAULT_OUTPUT_CSV,
) -> None:
    """Compute and persist dataset profile, one row per dataset (upsert)."""
    profile = compute_dataset_profile(df, dataset_name)
    if not profile:
        return

    row = pd.DataFrame([profile])
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing = pd.read_csv(output_path)
        if "dataset_name" in existing.columns:
            existing = existing[existing["dataset_name"] != dataset_name]
        output_df = pd.concat([existing, row], ignore_index=True)
    else:
        output_df = row

    output_df.to_csv(output_path, index=False)
