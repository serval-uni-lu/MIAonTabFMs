from pathlib import Path
import argparse
import csv
import shutil

import numpy as np
import pandas as pd
import openml

from dataset_profile import save_dataset_profile


def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def classify_task_type(task) -> str:
    task_type_obj = getattr(task, "task_type", None)
    task_type_name = str(task_type_obj).lower()

    # OpenML may expose task type as enum/object with id/value attributes.
    task_type_id = getattr(task, "task_type_id", None)
    if task_type_id is None and task_type_obj is not None:
        task_type_id = getattr(task_type_obj, "id", None)
    if task_type_id is None and task_type_obj is not None:
        task_type_id = getattr(task_type_obj, "value", None)

    if "regression" in task_type_name or task_type_id == 2:
        return "regression"
    return "classification"


def preprocess_like_get_data(df: pd.DataFrame, task_kind: str) -> pd.DataFrame:
    """Apply data cleaning steps inspired by get_data.py::preprocess_data."""
    data = df.copy()

    # Remove NAs similarly: '?' to NaN then fill with mode.
    data.replace("?", np.nan, inplace=True)
    if data.isna().sum().sum() > 0:
        data = data.fillna(data.mode().iloc[0])

    # Remove no-variance columns.
    drop_cols = []
    for col in data.columns:
        unique_values = np.unique(data[col].dropna())
        if unique_values.size <= 1:
            drop_cols.append(col)
    if drop_cols:
        data.drop(columns=drop_cols, inplace=True)

    # Remove duplicate rows.
    data.drop_duplicates(inplace=True)

    # Convert string/object columns to numeric when possible, else category codes.
    for col in data.columns:
        if pd.api.types.is_string_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
            try:
                data[col] = pd.to_numeric(data[col], errors="raise")
            except (ValueError, TypeError):
                data[col] = data[col].astype("category").cat.codes

    # Classification-only label filtering and remapping.
    if task_kind == "classification":
        label_counts = data.iloc[:, -1].value_counts()
        rare_labels = label_counts[label_counts <= 24].index
        if len(rare_labels) > 0:
            data = data[~data.iloc[:, -1].isin(rare_labels)].copy()

        if not data.empty:
            unique_labels = np.sort(data.iloc[:, -1].unique())
            label_mapping = {
                old_label: new_label for new_label, old_label in enumerate(unique_labels)
            }
            data.iloc[:, -1] = data.iloc[:, -1].map(label_mapping).astype("int32")

    return data


def build_frame_with_target(dataset, target_name: str) -> pd.DataFrame:
    X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=target_name)
    frame = X.copy()
    if y is not None:
        frame[target_name] = y
    return frame


def compute_stats(df: pd.DataFrame, task_kind: str) -> str:
    if task_kind == "classification":
        return "classification"
    return "regression"


def main(study_id: int, output_dir: str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep OpenML cache scoped under output dir.
    openml.config.cache_directory = str(out_dir / ".openml_cache")

    suite = openml.study.get_suite(study_id)
    rows = []
    dataset_stats = []
    seen_dataset_ids = set()

    print(f"Study {study_id}: {len(suite.tasks)} tasks")

    for task_id in suite.tasks:
        try:
            task = openml.tasks.get_task(task_id)
            dataset_id = int(task.dataset_id)

            if dataset_id in seen_dataset_ids:
                rows.append(
                    {
                        "task_id": task_id,
                        "dataset_id": dataset_id,
                        "dataset_name": "",
                        "target_attribute": task.target_name,
                        "rows": "",
                        "cols": "",
                        "file_path": "",
                        "status": "skipped_duplicate_dataset_id",
                        "error": "",
                    }
                )
                continue

            dataset = openml.datasets.get_dataset(
                dataset_id,
                download_data=True,
                download_qualities=False,
                download_features_meta_data=False,
            )

            task_kind = classify_task_type(task)
            raw_df = build_frame_with_target(dataset, task.target_name)
            safe_name = sanitize_name(f"{dataset_id}_{dataset.name}")
            save_dataset_profile(raw_df, safe_name)
            processed_df = preprocess_like_get_data(raw_df, task_kind)
            file_path = out_dir / f"{safe_name}.csv"
            # Match get_data.py output style (headerless CSV).
            processed_df.to_csv(file_path, index=False, header=False)

            task_type_label = compute_stats(processed_df, task_kind)
            dataset_stats.append(
                {
                    "dataset_name": safe_name,
                    "dataset_id": dataset_id,
                    "openml_name": dataset.name,
                    "task_type": task_type_label,
                }
            )

            rows.append(
                {
                    "task_id": task_id,
                    "dataset_id": dataset_id,
                    "dataset_name": dataset.name,
                    "target_attribute": task.target_name,
                    "rows": len(processed_df),
                    "cols": processed_df.shape[1],
                    "file_path": str(file_path),
                    "status": "downloaded",
                    "error": "",
                }
            )
            seen_dataset_ids.add(dataset_id)

        except Exception as exc:
            rows.append(
                {
                    "task_id": task_id,
                    "dataset_id": "",
                    "dataset_name": "",
                    "target_attribute": "",
                    "rows": "",
                    "cols": "",
                    "file_path": "",
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"[FAIL] task={task_id}: {exc}")

    stats_path = out_dir / "dataset_overview.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_name",
                "dataset_id",
                "openml_name",
                "task_type",
            ],
        )
        writer.writeheader()
        writer.writerows(dataset_stats)

    print(f"Done. Unique datasets downloaded: {len(seen_dataset_ids)}")
    print(f"Dataset overview: {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download all datasets referenced by an OpenML study task suite."
    )
    parser.add_argument("--study-id", type=int, default=457)
    parser.add_argument("--output-dir", type=str, default="data/data_tabarena")
    args = parser.parse_args()

    main(study_id=args.study_id, output_dir=args.output_dir)
