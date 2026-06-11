"""
Comprehensive Results Visualization and Analysis Script

This script analyzes all RMIA attack results from ml_privacy_meter logs,
generating visualizations for model accuracy, attack AUC, and computational costs
across different models and datasets.
"""

import os
import json
import re
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from scipy.stats import wilcoxon, friedmanchisquare
from scipy.stats import bootstrap as scipy_bootstrap
from sklearn.metrics import roc_curve, auc as sk_auc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class ResultsAnalyzer:
    """Analyzes and visualizes RMIA attack results from ml_privacy_meter logs."""

    RMIA_COLOR = "#086375"
    AMIA_COLOR = "#b23a48"
    LIRA_COLOR = "#ee6c4d"
    ATTACK_P_COLOR = "#fdc500"
    DEFAULT_EXCLUDED_DATASETS = {"aloi", "46956_seismic-bumps", "seismic_bumps", "seismic-bumps", "lcld", "purchases10"}
    DATASET_DISPLAY_LABELS = {"locations": "Locations", "dropout_success": "Dropout Success"}
    DATASET_PLOT_ORDER = ["locations", "dropout_success"]

    def __init__(
        self,
        logs_base_dir: str = "ml_privacy_meter/logs",
        output_dir: str = "results_visualizations/attacks_viz",
        excluded_datasets: Optional[set[str]] = None,
    ):
        """
        Initialize the analyzer.

        Args:
            logs_base_dir (str): Base directory containing all experiment logs
            output_dir (str): Directory to save all generated visualizations
        """
        self.logs_base_dir = logs_base_dir
        self.output_dir = output_dir
        self.excluded_datasets = set(self.DEFAULT_EXCLUDED_DATASETS if excluded_datasets is None else excluded_datasets)
        self.results = []
        self.datasets = set()
        self.models = set()
        self.model_plot_order = [
            "rf",
            "lightgbm",
            "mlp",
            "tabnet",
            "tabpfn",
            "real-tabpfn",
            "tabicl",
            "tabdpt",
        ]
        self._model_order_map = {m: i for i, m in enumerate(self.model_plot_order)}
        self._model_display = {
            "rf": "RF",
            "lightgbm": "LightGBM",
            "mlp": "MLP",
            "tabnet": "Tabnet",
            "tabpfn": "TabPFN",
            "real-tabpfn": "Real-TabPFN",
            "tabicl": "TabICL",
            "tabdpt": "TabDPT",
        }
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory for visualizations: {self.output_dir}")

    @classmethod
    def _fmt_dataset(cls, name: str) -> str:
        """Return display label for a dataset, stripping leading numeric OpenML IDs."""
        raw = str(name)
        stripped = re.sub(r'^\d+_', '', raw)
        return cls.DATASET_DISPLAY_LABELS.get(raw, cls.DATASET_DISPLAY_LABELS.get(stripped, stripped))

    @staticmethod
    def _dataset_key(name: str) -> str:
        return re.sub(r"^\d+[_-]", "", str(name)).lower().replace("_", "-")

    def _is_excluded_dataset(self, dataset: str) -> bool:
        key = self._dataset_key(dataset)
        excluded = {self._dataset_key(d) for d in self.excluded_datasets}
        return key in excluded or str(dataset) in self.excluded_datasets

    def _apply_dataset_exclusions(self) -> None:
        if not self.excluded_datasets:
            return
        self.results = [r for r in self.results if not self._is_excluded_dataset(r.get("dataset", ""))]
        self.datasets = {r["dataset"] for r in self.results if r.get("dataset") is not None}
        self.models = {r["model"] for r in self.results if r.get("model") is not None}

    def _filter_excluded_df(self, df: pd.DataFrame, dataset_col: str = "dataset") -> pd.DataFrame:
        if df.empty or not self.excluded_datasets or dataset_col not in df.columns:
            return df
        return df[~df[dataset_col].map(self._is_excluded_dataset)].copy()

    def _fmt_model(self, name: str) -> str:
        """Return display name for a model key."""
        return self._model_display.get(name, name)

    def _sort_models(self, models: List[str]) -> List[str]:
        """Sort model names using configured plot order, unknowns last."""
        return sorted(models, key=lambda m: (self._model_order_map.get(m, len(self.model_plot_order)), m))

    def _sort_datasets(self, datasets: List[str]) -> List[str]:
        """Sort datasets with common paired plots in a stable narrative order."""
        order = {name: i for i, name in enumerate(self.DATASET_PLOT_ORDER)}
        return sorted(datasets, key=lambda d: (order.get(str(d), len(order)), self._fmt_dataset(str(d)).lower()))

    def _sort_df_by_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort a DataFrame by model using configured plot order."""
        if "model" not in df.columns or df.empty:
            return df
        return (
            df.assign(_model_order=df["model"].map(lambda m: self._model_order_map.get(m, len(self.model_plot_order))))
            .sort_values(["_model_order", "model"])
            .drop(columns=["_model_order"])
        )

    def _aggregate_seed_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse per-seed logs to one mean row per dataset/model/attack."""
        if df.empty or "seed" not in df.columns or not df["seed"].notna().any():
            return df

        group_cols = [c for c in ["dataset", "model", "attack"] if c in df.columns]
        if not group_cols:
            return df

        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != "seed"
        ]
        non_numeric_cols = [
            c for c in df.columns
            if c not in group_cols and c not in numeric_cols and c != "seed"
        ]

        grouped = df.groupby(group_cols, as_index=False)
        mean_df = grouped[numeric_cols].mean()
        std_df = grouped[numeric_cols].std().rename(columns={c: f"{c}_std" for c in numeric_cols})
        # Avoid duplicating std columns that are already present in the raw data.
        duplicate_std_cols = [c for c in std_df.columns if c in mean_df.columns and c not in group_cols]
        std_df = std_df.drop(columns=duplicate_std_cols)
        seed_counts = (
            df.groupby(group_cols, as_index=False)["seed"]
            .nunique()
            .rename(columns={"seed": "num_seeds"})
        )
        out = mean_df.merge(std_df, on=group_cols, how="left").merge(seed_counts, on=group_cols, how="left")

        for col in non_numeric_cols:
            first_vals = df.groupby(group_cols, as_index=False)[col].first()
            out = out.merge(first_vals, on=group_cols, how="left")

        return out

    def load_target_model_metadata_means(self) -> pd.DataFrame:
        """Load target model train/test accuracy means from models_metadata.json."""
        base_path = Path(self.logs_base_dir)
        rows = []
        for metadata_path in sorted(base_path.glob("*/*/seed*/rmia/models/models_metadata.json")):
            rel_parts = metadata_path.relative_to(base_path).parts
            if len(rel_parts) < 5:
                continue
            dataset, model, seed_part = rel_parts[:3]
            if not re.fullmatch(r"seed\d+", seed_part):
                continue
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning("Error reading model metadata %s: %s", metadata_path, e)
                continue
            target = metadata.get("0", {})
            rows.append({
                "dataset": dataset,
                "model": model,
                "target_train_acc_metadata": target.get("train_acc", np.nan),
                "target_test_acc_metadata": target.get("test_acc", np.nan),
                "metadata_seed": int(seed_part.replace("seed", "")),
            })

        df = self._filter_excluded_df(pd.DataFrame(rows))
        if df.empty:
            return df
        return (
            df.groupby(["dataset", "model"], as_index=False)
            .agg(
                target_train_acc_metadata=("target_train_acc_metadata", "mean"),
                target_train_acc_metadata_std=("target_train_acc_metadata", lambda s: float(np.nanstd(s, ddof=1)) if s.notna().sum() > 1 else 0.0),
                target_test_acc_metadata=("target_test_acc_metadata", "mean"),
                target_test_acc_metadata_std=("target_test_acc_metadata", lambda s: float(np.nanstd(s, ddof=1)) if s.notna().sum() > 1 else 0.0),
                metadata_num_seeds=("metadata_seed", "nunique"),
            )
        )

    def load_seed_summary_means(self, attack: Optional[str] = None) -> pd.DataFrame:
        """Load mean metrics from attack_result_seed_summary.csv files.

        The new seed layout stores one summary CSV per dataset/model directory:
        logs/<dataset>/<model>/attack_result_seed_summary.csv
        """
        base_path = Path(self.logs_base_dir)
        rows = []
        for summary_path in sorted(base_path.glob("*/*/attack_result_seed_summary.csv")):
            rel_parts = summary_path.relative_to(base_path).parts
            if len(rel_parts) < 3:
                continue
            dataset, model = rel_parts[0], rel_parts[1]

            try:
                summary = pd.read_csv(summary_path)
            except Exception as e:
                logger.warning("Error reading seed summary %s: %s", summary_path, e)
                continue

            if not {"metric", "mean"}.issubset(summary.columns):
                logger.warning("Seed summary %s is missing required columns", summary_path)
                continue

            if "attack" not in summary.columns:
                summary["attack"] = "rmia"
            summary["attack"] = summary["attack"].fillna("rmia")
            if attack is not None:
                summary = summary[summary["attack"] == attack]
            if summary.empty:
                continue

            for attack_name, attack_summary in summary.groupby("attack", sort=True):
                by_metric = attack_summary.set_index("metric")

                def _mean(metric: str):
                    return float(by_metric.loc[metric, "mean"]) if metric in by_metric.index else np.nan

                def _std(metric: str):
                    if "std" not in by_metric.columns or metric not in by_metric.index:
                        return np.nan
                    return float(by_metric.loc[metric, "std"])

                def _num_seeds(metric: str = "auc"):
                    if "num_seeds" not in by_metric.columns or metric not in by_metric.index:
                        return np.nan
                    return int(by_metric.loc[metric, "num_seeds"])

                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "attack": attack_name,
                    "attack_auc": _mean("auc"),
                    "attack_auc_std": _std("auc"),
                    "tpr_at_fpr_01": _mean("one_tenth_fpr"),
                    "tpr_at_fpr_01_std": _std("one_tenth_fpr"),
                    "tpr_at_fpr_0": _mean("zero_fpr"),
                    "tpr_at_fpr_0_std": _std("zero_fpr"),
                    "tnr_at_fnr_01": _mean("one_tenth_fnr"),
                    "tnr_at_fnr_01_std": _std("one_tenth_fnr"),
                    "tnr_at_fnr_0": _mean("zero_fnr"),
                    "tnr_at_fnr_0_std": _std("zero_fnr"),
                    "target_test_acc": _mean("target_model_0_test_acc"),
                    "target_test_acc_std": _std("target_model_0_test_acc"),
                    "num_seeds": _num_seeds(),
                    "source": str(summary_path),
                })

        summary_df = self._filter_excluded_df(pd.DataFrame(rows))
        if not summary_df.empty:
            attack_label = attack if attack is not None else "all attacks"
            logger.info("Loaded %d seed-summary mean rows for %s", len(summary_df), attack_label)
        return summary_df

    def _results_df_with_seed_summary_means(self) -> pd.DataFrame:
        """Return plotting rows with attack metrics overlaid from summary CSV means.

        The overlay is limited to attacks already present in self.results, so RMIA-only
        figures do not accidentally pull LiRA/RMIA-online rows back in.
        """
        df = self._aggregate_seed_rows(pd.DataFrame(self.results))
        key_cols = ["dataset", "model", "attack"]
        if df.empty:
            return df

        summary_df = self.load_seed_summary_means()
        if summary_df.empty or not set(key_cols).issubset(df.columns):
            return df

        allowed_attacks = set(df["attack"].dropna().unique())
        if allowed_attacks:
            summary_df = summary_df[summary_df["attack"].isin(allowed_attacks)].copy()
        if summary_df.empty:
            return df

        summary_metric_cols = [
            c for c in summary_df.columns
            if c not in key_cols and c != "source"
        ]
        df = df.copy()
        for col in summary_metric_cols:
            if col not in df.columns:
                df[col] = np.nan
        if "summary_source" not in df.columns:
            df["summary_source"] = pd.Series([None] * len(df), index=df.index, dtype=object)

        df_indexed = df.set_index(key_cols, drop=False)
        appended_rows = []
        for _, summary_row in summary_df.iterrows():
            key = tuple(summary_row[c] for c in key_cols)
            if key in df_indexed.index:
                for col in summary_metric_cols:
                    value = summary_row[col]
                    if not pd.isna(value):
                        df_indexed.loc[key, col] = value
                df_indexed.loc[key, "summary_source"] = summary_row.get("source", np.nan)
            else:
                new_row = {col: np.nan for col in df.columns}
                new_row["summary_source"] = None
                for col in key_cols + summary_metric_cols:
                    if col in summary_row.index:
                        new_row[col] = summary_row[col]
                new_row["summary_source"] = summary_row.get("source", np.nan)
                appended_rows.append(new_row)

        out = df_indexed.reset_index(drop=True)
        if appended_rows:
            out = pd.concat([out, pd.DataFrame(appended_rows)], ignore_index=True, sort=False)

        metadata_df = self.load_target_model_metadata_means()
        if not metadata_df.empty and {"dataset", "model"}.issubset(out.columns):
            out = out.merge(metadata_df, on=["dataset", "model"], how="left")
            for acc_col, meta_col in [
                ("target_train_acc", "target_train_acc_metadata"),
                ("target_test_acc", "target_test_acc_metadata"),
            ]:
                if acc_col not in out.columns:
                    out[acc_col] = np.nan
                out[acc_col] = pd.to_numeric(out[acc_col], errors="coerce")
                out[meta_col] = pd.to_numeric(out[meta_col], errors="coerce")
                out[acc_col] = out[acc_col].fillna(out[meta_col])
            for std_col, meta_std_col in [
                ("target_train_acc_std", "target_train_acc_metadata_std"),
                ("target_test_acc_std", "target_test_acc_metadata_std"),
            ]:
                if std_col not in out.columns:
                    out[std_col] = np.nan
                out[std_col] = pd.to_numeric(out[std_col], errors="coerce")
                out[meta_std_col] = pd.to_numeric(out[meta_std_col], errors="coerce")
                out[std_col] = out[std_col].fillna(out[meta_std_col])
        return out

    def _create_dataset_grid(self, num_datasets: int, max_cols: int = 4, fixed_shape: Optional[Tuple[int, int]] = None):
        """Create a readable subplot grid for dataset-wise charts."""
        if fixed_shape is not None:
            nrows, ncols = fixed_shape
        else:
            ncols = min(max_cols, max(1, num_datasets))
            nrows = math.ceil(num_datasets / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.2 * ncols, 4.4 * nrows),
            squeeze=False,
        )
        return fig, axes, nrows, ncols
        
    def parse_log_file(self, log_file_path: str) -> Dict:
        """
        Parse a log file and extract key metrics.

        Args:
            log_file_path (str): Path to the log file

        Returns:
            Dict: Extracted metrics
        """
        metrics = {
            'train_accuracies': [],
            'test_accuracies': [],
            'target_train_acc': None,
            'target_test_acc': None,
            'ref_train_accs': [],
            'ref_test_accs': [],
            'train_losses': [],
            'test_losses': [],
            'model_training_time': 0,
            'signal_preparation_time': 0,
            'attack_time': 0,
            'total_runtime': 0,
            'attack_auc': None,
            'tpr_at_fpr_01': None,
            'tpr_at_fpr_0': None,
            'tnr_at_fnr_01': None,
            'tnr_at_fnr_0': None,
            'predict_type': None,
        }

        try:
            with open(log_file_path, 'r') as f:
                content = f.read()

            # Logs may contain multiple runs concatenated. If every run block has
            # its own terminal runtime line the log is incremental (different seeds
            # appended) and the first block is this seed's data. If only the last
            # block is complete it is a rerun after failure — use the last block.
            _terminal_patterns = [
                r"Done in [\d.]+ s",
                r"Total runtime: [\d.]+ seconds",
                r"Population attack runtime: [\d.]+ seconds",
                r"LOSS runtime: [\d.]+ seconds",
                r"Script finished in [\d.]+ seconds",
                r"Auditing the privacy risks of target model \d+ costs [\d.]+ seconds",
            ]
            run_starts = [m.start() for m in re.finditer(r'Training \d+ models', content)]
            if len(run_starts) > 1:
                blocks = [content[run_starts[i]:run_starts[i + 1]] for i in range(len(run_starts) - 1)]
                blocks.append(content[run_starts[-1]:])
                all_complete = all(
                    any(re.search(p, b) for p in _terminal_patterns) for b in blocks
                )
                content = blocks[0] if all_complete else content[run_starts[-1]:]

            # Extract per-model train/test accuracies and losses
            # Track model index from "Training model N:" lines
            model_accs: Dict[int, Dict[str, float]] = {}
            current_idx = None
            for line in content.splitlines():
                m = re.search(r'Training model (\d+):', line)
                if m:
                    current_idx = int(m.group(1))
                    model_accs.setdefault(current_idx, {})
                    continue
                if current_idx is not None:
                    m = re.search(r'Train accuracy ([\d.]+)', line)
                    if m:
                        model_accs[current_idx]['train'] = float(m.group(1))
                    m = re.search(r'Test accuracy ([\d.]+)', line)
                    if m:
                        model_accs[current_idx]['test'] = float(m.group(1))

            # Target model = index 0; reference models = all others
            if 0 in model_accs:
                metrics['target_train_acc'] = model_accs[0].get('train')
                metrics['target_test_acc']  = model_accs[0].get('test')
            ref_idxs = sorted(k for k in model_accs if k > 0)
            metrics['ref_train_accs'] = [model_accs[i]['train'] for i in ref_idxs if 'train' in model_accs[i]]
            metrics['ref_test_accs']  = [model_accs[i]['test']  for i in ref_idxs if 'test'  in model_accs[i]]

            # Keep flat lists for backward-compat consumers
            metrics['train_accuracies'] = [model_accs[i]['train'] for i in sorted(model_accs) if 'train' in model_accs[i]]
            metrics['test_accuracies']  = [model_accs[i]['test']  for i in sorted(model_accs) if 'test'  in model_accs[i]]

            train_loss_pattern = r'Train Loss ([\d.]+)'
            test_loss_pattern = r'Test Loss ([\d.]+)'
            metrics['train_losses'] = [float(x) for x in re.findall(train_loss_pattern, content)]
            metrics['test_losses'] = [float(x) for x in re.findall(test_loss_pattern, content)]

            # Extract timing information
            model_training_pattern = r'Model training took ([\d.]+) seconds'
            signal_prep_pattern = r'Preparing signals took ([\d.]+) seconds'
            attack_time_pattern = r'Auditing the privacy risks of target model \d+ costs ([\d.]+) seconds'
            total_runtime_pattern = r'Total runtime: ([\d.]+) seconds'

            model_training_match = re.search(model_training_pattern, content)
            if model_training_match:
                metrics['model_training_time'] = float(model_training_match.group(1))

            signal_prep_match = re.search(signal_prep_pattern, content)
            if signal_prep_match:
                metrics['signal_preparation_time'] = float(signal_prep_match.group(1))

            attack_time_matches = re.findall(attack_time_pattern, content)
            if attack_time_matches:
                metrics['attack_time'] = sum(float(x) for x in attack_time_matches)

            total_runtime_match = re.search(total_runtime_pattern, content)
            if total_runtime_match:
                metrics['total_runtime'] = float(total_runtime_match.group(1))
            else:
                # LOSS attack uses a different pattern
                loss_runtime_match = re.search(r'LOSS runtime: ([\d.]+) seconds', content)
                if loss_runtime_match:
                    metrics['total_runtime'] = float(loss_runtime_match.group(1))

            # Extract attack metrics — use the LAST reported value so re-runs take precedence.
            auc_pattern = r'Target Model 0: AUC ([\d.]+).*TPR@0.1%FPR of ([\d.]+).*TPR@0.0%FPR of ([\d.]+)'
            auc_matches = re.findall(auc_pattern, content)
            if auc_matches:
                last = auc_matches[-1]
                metrics['attack_auc'] = float(last[0])
                metrics['tpr_at_fpr_01'] = float(last[1])
                metrics['tpr_at_fpr_0'] = float(last[2])

            tnr_pattern = r'Target Model 0: AUC [\d.]+.*TNR@0.1%FNR of ([\d.]+).*TNR@0.0%FNR of ([\d.]+)'
            tnr_matches = re.findall(tnr_pattern, content)
            if tnr_matches:
                last_tnr = tnr_matches[-1]
                metrics['tnr_at_fnr_01'] = float(last_tnr[0])
                metrics['tnr_at_fnr_0'] = float(last_tnr[1])

            # Extract predict type used during signal computation
            predict_type_match = re.search(r'Model exposing via (\w+)\.', content)
            if predict_type_match:
                metrics['predict_type'] = predict_type_match.group(1)

        except Exception as e:
            logger.warning(f"Error parsing log file {log_file_path}: {e}")

        return metrics

    def load_npz_results(self, npz_file_path: str) -> Dict:
        """
        Load attack results from NPZ file.

        Args:
            npz_file_path (str): Path to the attack_result_0.npz file

        Returns:
            Dict: Attack results
        """
        try:
            data = np.load(npz_file_path)
            result = {
                'auc': float(data['auc']),
                'scores': data['scores'],
                'memberships': data['memberships'],
            }
            if 'one_tenth_fpr' in data:
                result['tpr_at_fpr_01'] = float(data['one_tenth_fpr'])
            if 'zero_fpr' in data:
                result['tpr_at_fpr_0'] = float(data['zero_fpr'])
            if 'one_tenth_fnr' in data:
                result['tnr_at_fnr_01'] = float(data['one_tenth_fnr'])
            if 'zero_fnr' in data:
                result['tnr_at_fnr_0'] = float(data['zero_fnr'])
            return result
        except Exception as e:
            logger.warning(f"Error loading NPZ file {npz_file_path}: {e}")
            return {}

    def collect_all_results(self):
        """Collect results from nested log directories.

        Expected layouts:
        - logs/<dataset>/<model>/report/log_time_analysis.log
        - logs/<dataset>/<model>/<attack>/report/log_time_analysis.log
        """
        base_path = Path(self.logs_base_dir)
        if not base_path.exists():
            logger.error(f"Logs base directory not found: {self.logs_base_dir}")
            return

        for log_path in sorted(base_path.rglob("log_time_analysis.log")):
            rel_parts = log_path.relative_to(base_path).parts
            if len(rel_parts) < 4:
                # Must at least contain dataset/model/report/log_time_analysis.log
                continue

            dataset = rel_parts[0]
            if self._is_excluded_dataset(dataset):
                continue
            model = rel_parts[1]
            seed = None
            if len(rel_parts) >= 5 and re.fullmatch(r"seed\d+", rel_parts[2]):
                seed = int(rel_parts[2].replace("seed", ""))
                attack_subdir = rel_parts[3]
                is_online = len(rel_parts) > 4 and rel_parts[4] == "report_online"
            else:
                attack_subdir = rel_parts[2]
                is_online = len(rel_parts) > 3 and rel_parts[3] == "report_online"

            if attack_subdir == "report":
                attack = "rmia"  # legacy layout without attack subdirectory
            else:
                attack = attack_subdir + ("_online" if is_online else "")

            self.datasets.add(dataset)
            self.models.add(model)

            report_dir = log_path.parent
            npz_file = report_dir / "exp" / "attack_result_0.npz"

            metrics = self.parse_log_file(str(log_path))
            npz_results = self.load_npz_results(str(npz_file)) if npz_file.exists() else {}

            result = {
                'dataset': dataset,
                'model': model,
                'attack': attack,
                'seed': seed,
                'exp_dir': str(log_path.parent.parent),
                **metrics,
                **npz_results,
            }

            self.results.append(result)

        if not self.results:
            summary_df = self.load_seed_summary_means()
            if not summary_df.empty:
                summary_df = self._filter_excluded_df(summary_df)
                self.results = summary_df.to_dict("records")
                self.datasets = set(summary_df["dataset"])
                self.models = set(summary_df["model"])

        self._apply_dataset_exclusions()
        logger.info("Excluded datasets from visualizations: %s", sorted(self.excluded_datasets))
        logger.info(f"Loaded {len(self.results)} experiment results")
        logger.info(f"Datasets: {sorted(self.datasets)}")
        logger.info(f"Models: {sorted(self.models)}")

    def create_accuracy_comparison(self, save_dir: str = "results_visualizations/attacks_viz"):
        """Create model accuracy comparison visualizations."""
        os.makedirs(save_dir, exist_ok=True)
        
        plot_df = self._results_df_with_seed_summary_means()
        if "attack" in plot_df.columns:
            plot_df = plot_df[plot_df["attack"] == "rmia"].copy()
        datasets = sorted(plot_df["dataset"].dropna().unique()) if not plot_df.empty else sorted(self.datasets)
        plot_results = plot_df.to_dict("records")
        
        # 1. Training vs Test Accuracy by Model for each Dataset
        fig, axes, nrows, ncols = self._create_dataset_grid(len(datasets), fixed_shape=(3, 3))
        axes_flat = axes.flatten()

        for idx, dataset in enumerate(datasets):
            ax = axes_flat[idx]
            dataset_results = [r for r in plot_results if r['dataset'] == dataset]
            dataset_results = sorted(
                dataset_results,
                key=lambda r: (self._model_order_map.get(r['model'], len(self.model_plot_order)), r['model']),
            )
            
            models_list = [r['model'] for r in dataset_results]
            train_accs = pd.to_numeric(
                pd.Series([r.get('target_train_acc') for r in dataset_results]),
                errors='coerce',
            ).fillna(0.0).to_numpy(dtype=float)
            test_accs = pd.to_numeric(
                pd.Series([r.get('target_test_acc') for r in dataset_results]),
                errors='coerce',
            ).fillna(0.0).to_numpy(dtype=float)
            train_stds = pd.to_numeric(
                pd.Series([r.get('target_train_acc_std') for r in dataset_results]),
                errors='coerce',
            ).fillna(0.0).to_numpy(dtype=float)
            test_stds = pd.to_numeric(
                pd.Series([r.get('target_test_acc_std') for r in dataset_results]),
                errors='coerce',
            ).fillna(0.0).to_numpy(dtype=float)

            x = np.arange(len(models_list))
            width = 0.35

            ax.bar(
                x - width/2, train_accs, width, yerr=train_stds, capsize=3,
                label='Train', color='#3b6064', edgecolor='black', linewidth=0.35,
            )
            ax.bar(
                x + width/2, test_accs, width, yerr=test_stds, capsize=3,
                label='Test', color='#87bba2', edgecolor='black', linewidth=0.35,
            )
            
            ax.set_xlabel('')
            ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            ax.set_title(f'{self._fmt_dataset(dataset)} - Model Accuracy', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([self._fmt_model(m) for m in models_list], rotation=35, ha='right', fontsize=9)
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.4)

        # Remove unused axes so fixed-grid plots do not keep a blank row.
        for i in range(len(datasets), len(axes_flat)):
            fig.delaxes(axes_flat[i])

        # One shared legend at row 4, column 3 when that slot is free.
        shared_handles = [
            Patch(facecolor='#3b6064', edgecolor='none', label='Train'),
            Patch(facecolor='#87bba2', edgecolor='none', label='Test'),
        ]
        legend_row = 3  # 4th row (0-based)
        legend_col = 2  # 3rd column (0-based)
        if nrows > legend_row and ncols > legend_col:
            legend_idx = legend_row * ncols + legend_col
            if legend_idx >= len(datasets):
                legend_ax = axes_flat[legend_idx]
                legend_ax.axis('off')
                legend_ax.legend(
                    handles=shared_handles,
                    loc='center',
                    frameon=True,
                    title='Split',
                )
            else:
                fig.legend(
                    handles=shared_handles,
                    loc='center left',
                    bbox_to_anchor=(1.01, 0.5),
                    frameon=True,
                    title='Split',
                )
        else:
            fig.legend(
                handles=shared_handles,
                loc='center left',
                bbox_to_anchor=(1.01, 0.5),
                frameon=True,
                title='Split',
            )

        plt.tight_layout(pad=1.0)
        plt.savefig(os.path.join(save_dir, '01_accuracy_comparison_rmia.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_target_vs_reference_accuracy(self, save_dir: str = "results_visualizations/attacks_viz"):
        """01b — Target model accuracy vs reference models (online) and target-only (offline).

        Online setting: attacker has shadow/reference models → can compare target vs reference.
        Offline setting: attacker only has the target model → no reference comparison available.

        Each subplot = one dataset.  Per model-type:
          - Bar   = target model (model 0) train/test accuracy
          - Box   = distribution of reference models' train/test accuracy (online only)
        """
        os.makedirs(save_dir, exist_ok=True)

        datasets = sorted(self.datasets)
        COLOR_TARGET_TRAIN = '#3b6064'
        COLOR_TARGET_TEST  = '#87bba2'
        COLOR_REF_TRAIN    = '#c9a84c'
        COLOR_REF_TEST     = '#e8d5a3'

        for variant, fname in [("online", "01b_accuracy_target_vs_reference_online.png"),
                                ("offline", "01c_accuracy_target_only_offline.png")]:
            fig, axes, nrows, ncols = self._create_dataset_grid(len(datasets))
            axes_flat = axes.flatten()

            for idx, dataset in enumerate(datasets):
                ax = axes_flat[idx]
                dataset_results = sorted(
                    [r for r in self.results if r['dataset'] == dataset],
                    key=lambda r: (self._model_order_map.get(r['model'], len(self.model_plot_order)), r['model']),
                )
                models_list = [r['model'] for r in dataset_results]
                x = np.arange(len(models_list))
                width = 0.35 if variant == "offline" else 0.2

                for i, r in enumerate(dataset_results):
                    t_tr = r['target_train_acc']
                    t_te = r['target_test_acc']
                    if t_tr is not None:
                        ax.bar(i - width / 2, t_tr, width, color=COLOR_TARGET_TRAIN, zorder=3)
                    if t_te is not None:
                        ax.bar(i + width / 2, t_te, width, color=COLOR_TARGET_TEST,  zorder=3)

                    if variant == "online":
                        ref_tr = r.get('ref_train_accs') or []
                        ref_te = r.get('ref_test_accs')  or []
                        if ref_tr:
                            ax.boxplot(ref_tr, positions=[i - width * 1.6], widths=width * 0.8,
                                       patch_artist=True,
                                       boxprops=dict(facecolor=COLOR_REF_TRAIN, alpha=0.7),
                                       medianprops=dict(color='black', linewidth=1.5),
                                       whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
                                       flierprops=dict(markersize=3), zorder=2)
                        if ref_te:
                            ax.boxplot(ref_te, positions=[i + width * 1.6], widths=width * 0.8,
                                       patch_artist=True,
                                       boxprops=dict(facecolor=COLOR_REF_TEST, alpha=0.7),
                                       medianprops=dict(color='black', linewidth=1.5),
                                       whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
                                       flierprops=dict(markersize=3), zorder=2)

                ax.set_xticks(x)
                ax.set_xticklabels([self._fmt_model(m) for m in models_list], rotation=35, ha='right')
                ax.set_ylim([0, 1.05])
                ax.set_ylabel('Accuracy', fontsize=9)
                ax.set_title(self._fmt_dataset(dataset), fontsize=10, fontweight='bold')
                ax.grid(True, axis='y', alpha=0.3)

            for i in range(len(datasets), len(axes_flat)):
                axes_flat[i].axis('off')

            if variant == "online":
                legend_handles = [
                    Patch(facecolor=COLOR_TARGET_TRAIN, label='Target train'),
                    Patch(facecolor=COLOR_TARGET_TEST,  label='Target test'),
                    Patch(facecolor=COLOR_REF_TRAIN, alpha=0.7, label='Reference train (box)'),
                    Patch(facecolor=COLOR_REF_TEST,  alpha=0.7, label='Reference test (box)'),
                ]
                title_suffix = "Online (target + reference models)"
            else:
                legend_handles = [
                    Patch(facecolor=COLOR_TARGET_TRAIN, label='Target train'),
                    Patch(facecolor=COLOR_TARGET_TEST,  label='Target test'),
                ]
                title_suffix = "Offline (target model only)"

            fig.suptitle(f"Model Accuracy — {title_suffix}", fontsize=13, fontweight='bold')
            fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles),
                       bbox_to_anchor=(0.5, -0.02), frameon=True)
            plt.tight_layout(pad=1.0, rect=[0, 0.04, 1, 1])
            plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
            plt.close()

    def create_attack_auc_comparison(self, save_dir: str = "results_visualizations/attacks_viz"):
        """Create attack AUC comparison visualizations."""
        os.makedirs(save_dir, exist_ok=True)
        
        df = self._results_df_with_seed_summary_means()
        if "attack" in df.columns:
            df = df[df["attack"] == "rmia"].copy()
        df = df[df['attack_auc'].notna()]

        # 1. AUC by Model for each Dataset (bar plot)
        datasets = self._sort_datasets(list(df["dataset"].dropna().unique())) if not df.empty else self._sort_datasets(list(self.datasets))
        ncols = min(3, max(1, len(datasets)))
        nrows = math.ceil(len(datasets) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.0 * ncols, 3.9 * nrows),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        for idx, dataset in enumerate(datasets):
            ax = axes_flat[idx]
            dataset_df = self._sort_df_by_model(df[df['dataset'] == dataset]).copy()
            x = np.arange(len(dataset_df))
            auc_vals = pd.to_numeric(dataset_df['attack_auc'], errors='coerce').to_numpy(dtype=float)
            auc_stds = pd.to_numeric(
                dataset_df.get('attack_auc_std', pd.Series(index=dataset_df.index)),
                errors='coerce',
            ).fillna(0.0).to_numpy(dtype=float)

            bars = ax.bar(
                x, auc_vals, yerr=auc_stds,
                color='#086375', alpha=0.78, edgecolor='black', linewidth=0.4, capsize=3,
                label='RMIA AUC',
            )

            # Add AUC value labels on bars.
            for bar, val in zip(bars, auc_vals):
                if np.isfinite(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        min(1.03, val + 0.018),
                        f'{val:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        fontweight='bold',
                    )

            # Overlay target model test accuracy as a diamond with seed std.
            test_vals = pd.to_numeric(
                dataset_df.get('target_test_acc', pd.Series(index=dataset_df.index)),
                errors='coerce',
            ).to_numpy(dtype=float)
            test_stds = pd.to_numeric(
                dataset_df.get('target_test_acc_std', pd.Series(index=dataset_df.index)),
                errors='coerce',
            ).fillna(0.0).to_numpy(dtype=float)
            for i, (acc, std) in enumerate(zip(test_vals, test_stds)):
                if np.isfinite(acc):
                    ax.errorbar(
                        i, acc, yerr=std, fmt='D', color='black', ecolor='black',
                        markersize=5.5, capsize=3, zorder=5, markeredgecolor='white', markeredgewidth=0.8,
                    )

            ax.set_xlabel('')
            ax.set_ylabel('Seed mean RMIA AUC / test accuracy', fontsize=11, fontweight='bold')
            ax.set_title(f'{self._fmt_dataset(dataset)}', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([self._fmt_model(m) for m in dataset_df['model']], rotation=35, ha='right', fontsize=9)
            ax.set_ylim([0, 1.06])
            ax.axhline(0.5, color='red', linestyle='--', linewidth=1.8, alpha=0.95)
            ax.grid(True, axis='y', alpha=0.4)

        for i in range(len(datasets), len(axes_flat)):
            fig.delaxes(axes_flat[i])

        legend_handles = [
            Line2D([0], [0], color='red', linestyle='--', linewidth=1.8, alpha=0.95, label='Random guess (AUC=0.5)'),
            Line2D([0], [0], marker='D', color='black', linestyle='None', markersize=6,
                   markeredgecolor='white', markeredgewidth=0.8, label='Test accuracy (seed mean ± std)'),
        ]
        if len(axes_flat):
            axes_flat[0].legend(handles=legend_handles, loc='lower left', frameon=True, fontsize=8)
        plt.tight_layout(pad=0.8)
        plt.savefig(os.path.join(save_dir, '02_attack_auc_comparison_rmia.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. AUC Heatmap (Dataset vs Model). Prefer the new 5-seed summary means.
        heatmap_df = self.load_seed_summary_means(attack="rmia")
        if heatmap_df.empty:
            heatmap_df = df
            heatmap_title = "RMIA AUC Heatmap"
            heatmap_agg = "mean"
        else:
            heatmap_title = "RMIA AUC Heatmap (seed mean)"
            heatmap_agg = "first"

        pivot_auc = heatmap_df.pivot_table(
            values='attack_auc', index='dataset', columns='model', aggfunc=heatmap_agg
        )
        ordered_cols = [m for m in self.model_plot_order if m in pivot_auc.columns]
        ordered_cols += [m for m in pivot_auc.columns if m not in ordered_cols]
        pivot_auc = pivot_auc[ordered_cols]
        pivot_auc.index = pivot_auc.index.map(self._fmt_dataset)
        pivot_auc.columns = pivot_auc.columns.map(self._fmt_model)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_auc, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                   cbar_kws={'label': 'MIA AUC'}, ax=ax, linewidths=0.5)
        ax.set_title(heatmap_title, fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('', fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '03_attack_auc_heatmap_rmia.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_comprehensive_dashboard(self, save_dir: str = "results_visualizations/attacks_viz"):
        """Create a comprehensive dashboard with all key metrics."""
        os.makedirs(save_dir, exist_ok=True)
        
        df = self._results_df_with_seed_summary_means()
        df = df[df['attack_auc'].notna()]

        # Create a 1x2 dashboard (top row only)
        fig = plt.figure(figsize=(16, 6.5))
        gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

        # 1. Average AUC by Model
        ax1 = fig.add_subplot(gs[0, 0])
        model_auc = df.groupby('model')['attack_auc'].agg(['mean', 'std'])
        ordered_models = [m for m in self.model_plot_order if m in model_auc.index]
        extra_models = [m for m in model_auc.index if m not in ordered_models]
        model_auc = model_auc.reindex(ordered_models + sorted(extra_models))
        bars = ax1.barh(range(len(model_auc)), model_auc['mean'], xerr=model_auc['std'],
                       alpha=0.7, color='tab:green', capsize=5, edgecolor='black')
        ax1.set_yticks(range(len(model_auc)))
        ax1.set_yticklabels([self._fmt_model(m) for m in model_auc.index])
        ax1.set_xlabel('Average AUC', fontsize=11, fontweight='bold')
        ax1.set_title('Average Attack AUC by Model', fontsize=12, fontweight='bold')
        ax1.set_xlim(left=0)
        ax1.grid(True, axis='x', alpha=0.4)

        # 2. Runtime by Model (boxplot)
        ax2 = fig.add_subplot(gs[0, 1])
        all_models_rt = [m for m in self.model_plot_order if m in df['model'].unique()]
        all_models_rt += sorted(m for m in df['model'].unique() if m not in all_models_rt)
        rt_data = [(df[df['model'] == m]['total_runtime'].dropna() / 60.0).values for m in all_models_rt]
        bp = ax2.boxplot(rt_data, positions=range(len(all_models_rt)), vert=False,
                         patch_artist=True, notch=False,
                         boxprops=dict(facecolor='tab:orange', alpha=0.7),
                         medianprops=dict(color='black', linewidth=2),
                         whiskerprops=dict(linewidth=1.2),
                         flierprops=dict(marker='o', markersize=4, alpha=0.5))
        ax2.set_yticks(range(len(all_models_rt)))
        ax2.set_yticklabels([self._fmt_model(m) for m in all_models_rt])
        ax2.set_xlabel('Runtime (minutes)', fontsize=11, fontweight='bold')
        ax2.set_title('Runtime Distribution by Model', fontsize=12, fontweight='bold')
        ax2.grid(True, axis='x', alpha=0.4)

        plt.suptitle('RMIA Attack Results', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(os.path.join(save_dir, '04_comprehensive_dashboard_rmia.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_member_nonmember_signal_distributions(
        self,
        save_dir: str = "results_visualizations/attacks_viz",
        generate_plots: bool = True,
    ):
        """Compare member vs non-member RMIA attack score distributions.

        Uses RMIA artifacts per model:
        - report/exp/attack_result_0.npz (scores + memberships from target model 0)
        """
        os.makedirs(save_dir, exist_ok=True)

        models_of_interest = ["tabpfn", "real-tabpfn", "tabicl", "tabdpt", "mlp", "lightgbm", "rf"]

        base_path = Path(self.logs_base_dir)
        if not base_path.exists():
            logger.warning("Logs base directory does not exist, skipping signal distributions.")
            return

        dataset_candidates = sorted(self.datasets)
        if not dataset_candidates:
            logger.warning("No datasets found, skipping signal distributions.")
            return

        # credit_rating has the sharpest member/non-member contrast across models.
        preferred = ["credit_rating", "locations"]
        focus_dataset = next((d for d in preferred if d in dataset_candidates), dataset_candidates[0])

        # --- Load RMIA attack scores (scores + memberships) per model ---
        scores_by_model: Dict[str, Dict[str, np.ndarray]] = {
            m: {"member": np.array([]), "nonmember": np.array([])} for m in models_of_interest
        }

        for model in models_of_interest:
            attack_path = base_path / focus_dataset / model / "rmia" / "report" / "exp" / "attack_result_0.npz"
            if not attack_path.exists():
                # Some folder layouts omit the attack sub-directory.
                attack_path = base_path / focus_dataset / model / "report" / "exp" / "attack_result_0.npz"
            if not attack_path.exists():
                continue
            try:
                npz = np.load(attack_path)
                scores = np.asarray(npz["scores"]).reshape(-1)
                memberships = np.asarray(npz["memberships"]).astype(bool).reshape(-1)
                n = min(len(scores), len(memberships))
                scores = scores[:n]
                memberships = memberships[:n]
                scores_by_model[model]["member"] = scores[memberships]
                scores_by_model[model]["nonmember"] = scores[~memberships]
            except Exception as exc:
                logger.warning("Failed loading attack scores for %s/%s: %s", focus_dataset, model, exc)

        available_models = [m for m in models_of_interest if len(scores_by_model[m]["member"]) > 0]

        if generate_plots and available_models:
            # --- Plot 1: KDE density overlay per model (RMIA attack scores) ---
            n_models = len(available_models)
            ncols = min(3, n_models)
            nrows = math.ceil(n_models / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
            fig.suptitle(
                f"RMIA Attack Score Distributions — {focus_dataset}\n"
                "(higher score = model thinks sample was a member)",
                fontsize=13, fontweight="bold", y=1.01,
            )

            for idx, model in enumerate(available_models):
                ax = axes[idx // ncols][idx % ncols]
                mem = scores_by_model[model]["member"]
                non = scores_by_model[model]["nonmember"]

                # KDE via seaborn for clean density curves.
                if len(mem) > 1:
                    sns.kdeplot(mem, ax=ax, fill=True, color="tab:blue", alpha=0.45, label="Member", clip=(0, 1))
                if len(non) > 1:
                    sns.kdeplot(non, ax=ax, fill=True, color="tab:orange", alpha=0.45, label="Non-member", clip=(0, 1))

                # Vertical mean lines.
                if len(mem):
                    ax.axvline(np.mean(mem), color="tab:blue", linestyle="--", linewidth=1.2, alpha=0.9)
                if len(non):
                    ax.axvline(np.mean(non), color="tab:orange", linestyle="--", linewidth=1.2, alpha=0.9)

                # Fetch AUC for this model/dataset from cached results.
                auc_val = None
                for r in self.results:
                    if r.get("dataset") == focus_dataset and r.get("model") == model and r.get("attack", "").lower() == "rmia":
                        auc_val = r.get("attack_auc")
                        break
                auc_str = f"  AUC={auc_val:.3f}" if auc_val is not None else ""

                ax.set_title(f"{self._fmt_model(model)}{auc_str}", fontweight="bold")
                ax.set_xlabel("RMIA attack score")
                ax.set_ylabel("Density")
                ax.set_xlim(0.0, 1.0)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

            # Hide any unused axes.
            for idx in range(len(available_models), nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)

            plt.tight_layout()
            out_path = os.path.join(save_dir, f"06a_member_nonmember_kde_{focus_dataset}_rmia.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()

            # --- Plot 2: Attack effectiveness per model (TPR and TNR at score=0.5 threshold) ---
            # TPR = fraction of members correctly identified (score > 0.5)
            # TNR = fraction of non-members correctly rejected (score <= 0.5)
            # TPR@FPR10 = TPR when only 10% of non-members are falsely flagged (strict threshold)
            eff_rows = []
            for model in available_models:
                mem = scores_by_model[model]["member"]
                non = scores_by_model[model]["nonmember"]
                tpr_50 = float(np.mean(mem > 0.5)) if len(mem) else float("nan")
                tnr_50 = float(np.mean(non <= 0.5)) if len(non) else float("nan")
                # TPR at FPR=10%: find score threshold where 10% of non-members are above it.
                if len(non) > 0 and len(mem) > 0:
                    thresh_fpr10 = float(np.percentile(non, 90))  # top 10% of non-member scores
                    tpr_fpr10 = float(np.mean(mem > thresh_fpr10))
                else:
                    tpr_fpr10 = float("nan")
                eff_rows.append({"model": model, "TPR@0.5": tpr_50, "TNR@0.5": tnr_50, "TPR@FPR=10%": tpr_fpr10})

            eff_df = pd.DataFrame(eff_rows).set_index("model")

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
            fig.suptitle(
                f"Attack Effectiveness on Members vs Non-members — {focus_dataset}\n"
                "(threshold: RMIA score = 0.5)",
                fontsize=13, fontweight="bold",
            )

            x = np.arange(len(eff_df))
            width = 0.38
            model_labels = [self._fmt_model(m) for m in eff_df.index]

            # Left panel: TPR vs TNR at threshold 0.5.
            ax_left = axes[0, 0]
            ax_left.bar(x - width / 2, eff_df["TPR@0.5"], width=width,
                        color="tab:blue", alpha=0.85, label="TPR — members detected")
            ax_left.bar(x + width / 2, eff_df["TNR@0.5"], width=width,
                        color="tab:orange", alpha=0.85, label="TNR — non-members rejected")
            ax_left.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
            ax_left.set_xticks(x)
            ax_left.set_xticklabels(model_labels, rotation=20, ha="right")
            ax_left.set_ylim(0.0, 1.05)
            ax_left.set_ylabel("Rate")
            ax_left.set_title("TPR vs TNR at score threshold = 0.5", fontweight="bold")
            ax_left.legend(fontsize=9)
            ax_left.grid(True, alpha=0.3, axis="y")
            for i, (tpr, tnr) in enumerate(zip(eff_df["TPR@0.5"], eff_df["TNR@0.5"])):
                if not np.isnan(tpr):
                    ax_left.text(i - width / 2, tpr + 0.01, f"{tpr:.0%}", ha="center", va="bottom", fontsize=7)
                if not np.isnan(tnr):
                    ax_left.text(i + width / 2, tnr + 0.01, f"{tnr:.0%}", ha="center", va="bottom", fontsize=7)

            # Right panel: TPR at FPR=10% (strict realistic threshold).
            ax_right = axes[0, 1]
            colors = ["tab:red" if v > 0.3 else "tab:gray" for v in eff_df["TPR@FPR=10%"].fillna(0)]
            bars = ax_right.bar(x, eff_df["TPR@FPR=10%"], color=colors, alpha=0.85, width=0.55)
            ax_right.axhline(0.1, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="FPR=10% baseline")
            ax_right.set_xticks(x)
            ax_right.set_xticklabels(model_labels, rotation=20, ha="right")
            ax_right.set_ylim(0.0, 1.05)
            ax_right.set_ylabel("True Positive Rate")
            ax_right.set_title("Members detected at FPR=10%\n(only 1 in 10 non-members falsely flagged)", fontweight="bold")
            ax_right.legend(fontsize=9)
            ax_right.grid(True, alpha=0.3, axis="y")
            for i, v in enumerate(eff_df["TPR@FPR=10%"]):
                if not np.isnan(v):
                    ax_right.text(i, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=8, fontweight="bold")

            plt.tight_layout()
            out_path = os.path.join(save_dir, f"06b_attack_effectiveness_{focus_dataset}_rmia.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()

        # --- Summary CSV ---
        def _safe_mean(arr: np.ndarray) -> float:
            return float(np.mean(arr)) if len(arr) else float("nan")

        def _safe_std(arr: np.ndarray) -> float:
            return float(np.std(arr)) if len(arr) else float("nan")

        def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
            if len(a) == 0 or len(b) == 0:
                return float("nan")
            a_sorted = np.sort(a)
            b_sorted = np.sort(b)
            vals = np.sort(np.unique(np.concatenate([a_sorted, b_sorted])))
            cdf_a = np.searchsorted(a_sorted, vals, side="right") / len(a_sorted)
            cdf_b = np.searchsorted(b_sorted, vals, side="right") / len(b_sorted)
            return float(np.max(np.abs(cdf_a - cdf_b)))

        summary_rows = []
        for model in models_of_interest:
            mem = scores_by_model[model]["member"]
            non = scores_by_model[model]["nonmember"]
            tpr_50 = float(np.mean(mem > 0.5)) if len(mem) else float("nan")
            tnr_50 = float(np.mean(non <= 0.5)) if len(non) else float("nan")
            if len(non) > 0 and len(mem) > 0:
                thresh_fpr10 = float(np.percentile(non, 90))
                tpr_fpr10 = float(np.mean(mem > thresh_fpr10))
            else:
                tpr_fpr10 = float("nan")
            summary_rows.append(
                {
                    "dataset": focus_dataset,
                    "model": model,
                    "member_count": int(len(mem)),
                    "nonmember_count": int(len(non)),
                    "member_score_mean": _safe_mean(mem),
                    "nonmember_score_mean": _safe_mean(non),
                    "score_mean_gap_member_minus_nonmember": _safe_mean(mem) - _safe_mean(non),
                    "member_score_std": _safe_std(mem),
                    "nonmember_score_std": _safe_std(non),
                    "ks_score": _ks_statistic(mem, non),
                    "tpr_at_threshold_0.5": tpr_50,
                    "tnr_at_threshold_0.5": tnr_50,
                    "tpr_at_fpr_10pct": tpr_fpr10,
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(save_dir, f"06_member_nonmember_summary_{focus_dataset}.csv")
        summary_df.to_csv(summary_csv, index=False)

    def create_dataset_stats_auc_correlation(self, save_dir: str = "results_visualizations/attacks_viz"):
        """Plot per-model correlation between dataset statistics and MIA AUC."""
        os.makedirs(save_dir, exist_ok=True)

        results_df = self._results_df_with_seed_summary_means()
        if results_df.empty or "attack_auc" not in results_df.columns:
            logger.warning("No results available for dataset-stats/AUC correlation plot.")
            return

        results_df = results_df[results_df["attack_auc"].notna()].copy()
        if results_df.empty:
            logger.warning("No valid attack_auc values available for correlation plot.")
            return

        # Use target model (model 0) accuracy only
        results_df["train_acc_mean"] = pd.to_numeric(results_df["target_train_acc"], errors="coerce")
        results_df["test_acc_mean"]  = pd.to_numeric(results_df["target_test_acc"],  errors="coerce")
        results_df["target_model_accuracy_pct"] = results_df["test_acc_mean"] * 100.0
        results_df["train_test_gap_pct_points"] = (
            (results_df["train_acc_mean"] - results_df["test_acc_mean"]) * 100.0
        )

        stats_path = Path(save_dir) / "dataset_profiles.csv"
        if not stats_path.exists():
            # Also check the default output dir
            stats_path = Path("results_visualizations") / "dataset_profiles.csv"
        if not stats_path.exists():
            logger.warning("dataset_profiles.csv not found; skipping correlation plot.")
            return

        stats_df = pd.read_csv(stats_path)
        if "dataset_name" not in stats_df.columns:
            logger.warning("dataset_profiles.csv missing 'dataset_name' column; skipping.")
            return

        stats_df = (
            stats_df.sort_values("dataset_name")
            .drop_duplicates(subset=["dataset_name"], keep="last")
            .rename(columns={"dataset_name": "dataset"})
        )

        merged = pd.merge(results_df, stats_df, on="dataset", how="inner")
        if merged.empty:
            logger.warning("No overlap between results datasets and dataset_profiles.csv entries.")
            return

        # Build log-scaled size columns for better correlation stability.
        merged["log_total_rows"] = np.log10(np.clip(pd.to_numeric(merged["total_rows"], errors="coerce"), 1.0, None))
        merged["log_num_features"] = np.log10(np.clip(pd.to_numeric(merged["num_features"], errors="coerce"), 1.0, None))

        # Columns from dataset_profiles.csv to correlate against MIA AUC.
        profile_stat_map = {
            "log_total_rows":               "Dataset Size (log10 rows)",
            "log_num_features":             "Feature Count (log10)",
            "num_numerical_features":       "# Numerical Features",
            "num_categorical_features":     "# Categorical Features",
            "num_binary_features":          "# Binary Features",
            "missing_pct":                  "Missing Cells (%)",
            "duplicate_rows":               "Duplicate Rows",
            "num_classes":                  "# Classes",
            "majority_class_pct":           "Majority Class (%)",
            "minority_class_pct":           "Minority Class (%)",
            "imbalance_ratio":              "Class Imbalance Ratio",
            "target_entropy":               "Target Entropy",
            "mean_feature_entropy":         "Mean Feature Entropy",
            "max_feature_entropy":          "Max Feature Entropy",
            "outlier_pct":                  "Outlier Cell (%)",
            "near_zero_variance_features":  "Near-Zero Variance Features",
            "mean_numerical_variance":      "Mean Numerical Variance",
            "mean_abs_corr_with_target":    "Mean |Corr| w/ Target",
            "max_abs_corr_with_target":     "Max |Corr| w/ Target",
            "samples_per_feature":          "Samples per Feature",
            "mean_pairwise_feature_corr":   "Mean Pairwise Feature Corr",
            "mean_abs_skewness":            "Mean Feature Skewness",
            "mean_kurtosis":                "Mean Feature Kurtosis",
            "pca_n_components_95pct":       "PCA Components (95% var)",
            "pca_explained_var_top5":       "PCA Top-5 Explained Var",
            "knn_label_disagreement":       "KNN Label Disagreement",
            "train_test_gap_pct_points":    "Train-Test Gap (pp)",
            "target_model_accuracy_pct":    "Target Model Accuracy (%)",
        }

        stat_cols = []
        stat_labels = {}
        for col, label in profile_stat_map.items():
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")
                stat_cols.append(col)
                stat_labels[col] = label

        corr_rows = []
        for model_name, group in merged.groupby("model"):
            row = {"model": model_name, "n_datasets": int(group["dataset"].nunique())}
            for col in stat_cols:
                # Need at least 2 points and non-constant vectors for a defined correlation.
                if len(group) < 2 or group[col].nunique() < 2 or group["attack_auc"].nunique() < 2:
                    row[col] = np.nan
                else:
                    row[col] = float(group[col].corr(group["attack_auc"], method="pearson"))
            corr_rows.append(row)

        corr_df = pd.DataFrame(corr_rows)

        # Sort models by plot order.
        model_order = [m for m in self.model_plot_order if m in corr_df["model"].values]
        model_order += [m for m in corr_df["model"].values if m not in model_order]
        corr_df = corr_df.set_index("model").reindex(model_order).reset_index()

        corr_csv_path = os.path.join(save_dir, "08_dataset_stats_auc_correlation.csv")
        corr_df.to_csv(corr_csv_path, index=False)

        if corr_df.empty:
            logger.warning("No model-wise correlations computed.")
            return

        # Rows = models, columns = dataset stats.
        heatmap_df = corr_df.set_index("model")[stat_cols].rename(columns=stat_labels)
        # Drop columns that are fully NaN.
        heatmap_df = heatmap_df.loc[:, heatmap_df.notna().any(axis=0)]
        # Apply display names to model rows.
        heatmap_df.index = [self._fmt_model(m) for m in heatmap_df.index]

        fig_width = max(10.0, 0.9 * heatmap_df.shape[1] + 2.0)
        fig_height = max(4.0, 0.55 * len(heatmap_df) + 2.0)
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0.0,
            vmin=-1.0,
            vmax=1.0,
            linewidths=0.5,
            cbar_kws={"label": "Pearson r  (red = more→higher AUC, blue = more→lower AUC)"},
        )
        plt.title("Per-Model Correlation: Dataset Stats vs MIA AUC — RMIA", fontsize=13, fontweight="bold")
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(rotation=60, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "08_dataset_stats_auc_correlation_rmia.png"), dpi=300, bbox_inches="tight")
        plt.close()


    def create_feature_target_correlation_summary(
        self,
        save_dir: str = "results_visualizations/attacks_viz",
        corr_threshold: float = 0.1,
    ):
        """Save count of target-correlated features for each dataset present in logs."""
        os.makedirs(save_dir, exist_ok=True)

        dataset_candidates = sorted(self.datasets)
        if not dataset_candidates:
            logger.warning("No datasets found in logs for feature-target correlation summary.")
            return

        data_dir_candidates = [
            Path("data/original"),
            Path("data/data_tabarena")
        ]

        def _resolve_dataset_file(dataset_name: str) -> Path | None:
            for base in data_dir_candidates:
                path = base / f"{dataset_name}.csv"
                if path.exists():
                    return path
            return None

        def _to_numeric_codes(series: pd.Series) -> np.ndarray:
            if pd.api.types.is_numeric_dtype(series):
                return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            casted = series.astype("category").cat.codes.to_numpy(dtype=float)
            casted[casted < 0] = np.nan
            return casted

        # Prefer using precomputed split sizes from dataset_profiles.csv for consistency.
        dataset_rows_map: Dict[str, float] = {}
        dataset_features_map: Dict[str, float] = {}
        stats_path = Path(save_dir) / "dataset_profiles.csv"
        if not stats_path.exists():
            stats_path = Path("results_visualizations") / "dataset_profiles.csv"
        if stats_path.exists():
            try:
                stats_df = pd.read_csv(stats_path)
                if {"dataset_name", "total_rows", "train_rows", "test_rows"}.issubset(set(stats_df.columns)):
                    stats_df = (
                        stats_df.sort_values("dataset_name")
                        .drop_duplicates(subset=["dataset_name"], keep="last")
                    )
                    dataset_rows_map = {
                        str(row["dataset_name"]): float(row["total_rows"])
                        for _, row in stats_df.iterrows()
                        if pd.notna(row["dataset_name"]) and pd.notna(row["total_rows"])
                    }
                    if "num_features" in stats_df.columns:
                        dataset_features_map = {
                            str(row["dataset_name"]): float(row["num_features"])
                            for _, row in stats_df.iterrows()
                            if pd.notna(row["dataset_name"]) and pd.notna(row["num_features"])
                        }

            except Exception as exc:
                logger.warning("Failed loading dataset_profiles.csv for row counts: %s", exc)

        summary_rows = []
        for dataset_name in dataset_candidates:
            dataset_path = _resolve_dataset_file(dataset_name)
            if dataset_path is None:
                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "dataset_path": "",
                        "num_features": np.nan,
                        "num_correlated_features": np.nan,
                        "correlated_ratio": np.nan,
                        "correlation_threshold_abs": corr_threshold,
                        "status": "missing_dataset_file",
                    }
                )
                continue

            try:
                df = pd.read_csv(dataset_path, header=None)
                if df.shape[1] < 2:
                    summary_rows.append(
                        {
                            "dataset": dataset_name,
                            "dataset_path": str(dataset_path),
                            "num_features": max(df.shape[1] - 1, 0),
                            "num_correlated_features": 0,
                            "correlated_ratio": 0.0,
                            "correlation_threshold_abs": corr_threshold,
                            "status": "insufficient_columns",
                        }
                    )
                    continue

                y = _to_numeric_codes(df.iloc[:, -1])
                y_valid = np.isfinite(y)
                y_std = np.nanstd(y)
                y_for_counts = pd.Series(y).dropna()

                # Class-distribution characteristics.
                if len(y_for_counts) > 0:
                    class_probs = y_for_counts.value_counts(normalize=True)
                    majority_frac = float(class_probs.max())
                    minority_frac = float(class_probs.min())
                    class_imbalance_ratio = (
                        float(majority_frac / minority_frac) if minority_frac > 0 else np.nan
                    )
                    label_entropy = float(-(class_probs * np.log2(np.clip(class_probs, 1e-12, 1.0))).sum())
                else:
                    majority_frac = np.nan
                    minority_frac = np.nan
                    class_imbalance_ratio = np.nan
                    label_entropy = np.nan

                mapped_num_features = dataset_features_map.get(dataset_name, np.nan)
                actual_features = df.shape[1] - 1
                num_features = min(int(mapped_num_features), actual_features) if pd.notna(mapped_num_features) and mapped_num_features > 0 else actual_features
                correlated_count = 0
                total_rows = dataset_rows_map.get(dataset_name, np.nan)
                p_over_n_ratio = float(num_features / total_rows) if total_rows > 0 else np.nan
                
                numeric_outlier_flags = 0
                numeric_total_cells = 0
                high_cardinality_features = 0

                for col_idx in range(num_features):
                    raw_col = df.iloc[:, col_idx]
                    x = _to_numeric_codes(raw_col)
                    valid = np.isfinite(x) & y_valid
                    if np.sum(valid) < 2:
                        continue

                    x_valid = x[valid]
                    y_sub = y[valid]
                    if np.nanstd(x_valid) == 0.0 or y_std == 0.0:
                        continue

                    corr = np.corrcoef(x_valid, y_sub)[0, 1]
                    if np.isfinite(corr) and abs(float(corr)) >= corr_threshold:
                        correlated_count += 1

                    # Numeric/categorical characterization per feature.
                    numeric_cast = pd.to_numeric(raw_col, errors="coerce")
                    valid_numeric = numeric_cast.notna().sum()
                    numeric_ratio = (valid_numeric / len(raw_col)) if len(raw_col) > 0 else 0.0
                    if numeric_ratio >= 0.9:
                        values = numeric_cast.to_numpy(dtype=float)
                        valid_vals = values[np.isfinite(values)]
                        if len(valid_vals) >= 4:
                            q1 = np.percentile(valid_vals, 25)
                            q3 = np.percentile(valid_vals, 75)
                            iqr = q3 - q1
                            if iqr > 0:
                                low = q1 - 1.5 * iqr
                                high = q3 + 1.5 * iqr
                                outlier_mask = (valid_vals < low) | (valid_vals > high)
                                numeric_outlier_flags += int(np.sum(outlier_mask))
                                numeric_total_cells += len(valid_vals)
                    else:
                        non_null = raw_col.dropna()
                        n_unique = int(non_null.nunique())
                        ratio = (n_unique / len(non_null)) if len(non_null) > 0 else 0.0
                        if ratio > 0.1:
                            high_cardinality_features += 1

                numeric_outlier_cell_ratio = (
                    float(numeric_outlier_flags / numeric_total_cells)
                    if numeric_total_cells > 0
                    else np.nan
                )
                high_cardinality_feature_ratio = (
                    float(high_cardinality_features / num_features) if num_features > 0 else np.nan
                )

                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "dataset_path": str(dataset_path),
                        "num_features": int(num_features),
                        "num_correlated_features": int(correlated_count),
                        "correlated_ratio": float(correlated_count / num_features) if num_features > 0 else 0.0,
                        "p_over_n_ratio": p_over_n_ratio,
                        "class_imbalance_ratio": class_imbalance_ratio,
                        "minority_class_fraction": minority_frac,
                        "label_entropy": label_entropy,
                        "numeric_outlier_cell_ratio": numeric_outlier_cell_ratio,
                        "high_cardinality_feature_ratio": high_cardinality_feature_ratio,
                        "correlation_threshold_abs": corr_threshold,
                        "status": "ok",
                    }
                )
            except Exception as exc:
                logger.warning("Failed feature-target correlation for %s: %s", dataset_name, exc)
                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "dataset_path": str(dataset_path),
                        "num_features": np.nan,
                        "num_correlated_features": np.nan,
                        "correlated_ratio": np.nan,
                        "correlation_threshold_abs": corr_threshold,
                        "status": f"error: {exc}",
                    }
                )

        out_df = pd.DataFrame(summary_rows).sort_values("dataset")
        out_path = os.path.join(save_dir, "09_feature_target_correlation_summary_rmia.csv")
        out_df.to_csv(out_path, index=False)

    def create_attack_type_comparison(self, save_dir: str = "results_visualizations/attacks_viz"):
        """Compare RMIA, LiRA, LOSS, Attack-P and online variants across models and datasets."""
        os.makedirs(save_dir, exist_ok=True)

        _all_attacks = [
            "rmia",
            "rmia_online",
            "lira",
            "lira_online",
            "attack_p",
        ]

        df = self._results_df_with_seed_summary_means()
        df = df[df["attack_auc"].notna() & df["attack"].isin(_all_attacks)].copy()

        if df.empty:
            logger.warning("No multi-attack results to compare.")
            return

        attack_colors = {
            "rmia":        self.RMIA_COLOR,
            "rmia_online": "#0fa8c4",
            "lira":        self.LIRA_COLOR,
            "lira_online": "#f4a67a",
            "attack_p":    self.ATTACK_P_COLOR,
        }
        attack_labels = {
            "rmia":        "RMIA",
            "rmia_online": "RMIA-online",
            "lira":        "LiRA",
            "lira_online": "LiRA-online",
            "attack_p":    "Attack-P",
        }
        present_attacks = [a for a in _all_attacks if a in df["attack"].unique()]
        models = self._sort_models(list(df["model"].unique()))

        # --- Plot 10: Average AUC per model per attack (offline only, grouped bar) ---
        _offline_attacks = ["rmia", "lira", "attack_p"]
        offline_df = df[df["attack"].isin(_offline_attacks)].copy()
        offline_present = [a for a in _offline_attacks if a in offline_df["attack"].unique()]
        offline_colors = {k: v for k, v in attack_colors.items() if k in _offline_attacks}
        offline_labels = {k: v for k, v in attack_labels.items() if k in _offline_attacks}

        if offline_present:
            offline_models = self._sort_models(list(offline_df["model"].unique()))
            x = np.arange(len(offline_models))
            n = len(offline_present)
            width = 0.22
            offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)

            fig, ax = plt.subplots(figsize=(12, 5.85))
            for i, attack in enumerate(offline_present):
                adf = offline_df[offline_df["attack"] == attack]
                means = [adf[adf["model"] == m]["attack_auc"].mean() for m in offline_models]
                stds = np.asarray([adf[adf["model"] == m]["attack_auc"].std() for m in offline_models], dtype=float)
                yerr = None if np.isnan(stds).all() else np.nan_to_num(stds, nan=0.0)
                ax.bar(
                    x + offsets[i], means, width, yerr=yerr,
                    label=offline_labels[attack], color=offline_colors[attack],
                    alpha=0.8, capsize=4, edgecolor="black",
                )

            ax.set_xticks(x)
            ax.set_xticklabels([self._fmt_model(m) for m in offline_models], rotation=35, ha="right", fontsize=13)
            ax.set_xlabel("")
            ax.set_ylabel("MIA AUC", fontsize=15, fontweight="bold")
            ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
            ax.set_ylim(bottom=0)
            handles, labels = ax.get_legend_handles_labels()
            handles.append(Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Random guess"))
            ax.legend(handles=handles, ncol=len(offline_present) + 1, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=True, fontsize=13)
            ax.tick_params(axis="y", labelsize=13)
            ax.grid(True, axis="y", alpha=0.4)
            plt.tight_layout(rect=(0, 0, 1, 0.90))
            plt.savefig(os.path.join(save_dir, "09_attack_comparison_by_model.png"), dpi=300, bbox_inches="tight")
            plt.close()

            # --- Plot 12: Per-dataset grouped bar comparing attacks per model ---
            datasets = sorted(offline_df["dataset"].unique())
            fig, axes, _, _ = self._create_dataset_grid(len(datasets), fixed_shape=(3, 3))
            axes_flat = axes.flatten()

            for idx, dataset in enumerate(datasets):
                ax = axes_flat[idx]
                ds_df = offline_df[offline_df["dataset"] == dataset]
                ds_models = self._sort_models(list(ds_df["model"].unique()))
                x_ds = np.arange(len(ds_models))
                offsets_ds = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)

                for i, attack in enumerate(offline_present):
                    heights = []
                    seed_stds = []
                    for m in ds_models:
                        row = ds_df[(ds_df["attack"] == attack) & (ds_df["model"] == m)]
                        if row.empty:
                            heights.append(np.nan)
                            seed_stds.append(np.nan)
                            continue
                        heights.append(row["attack_auc"].iloc[0])
                        if "attack_auc_std" in row.columns:
                            seed_stds.append(row["attack_auc_std"].iloc[0])
                        else:
                            seed_stds.append(np.nan)
                    yerr = np.asarray(seed_stds, dtype=float)
                    yerr = None if np.isnan(yerr).all() else np.nan_to_num(yerr, nan=0.0)
                    ax.bar(
                        x_ds + offsets_ds[i], heights, width, yerr=yerr, capsize=3,
                        label=offline_labels[attack], color=offline_colors[attack], alpha=0.8, edgecolor="black",
                    )

                ax.set_xticks(x_ds)
                ax.set_xticklabels([self._fmt_model(m) for m in ds_models], rotation=35, ha="right", fontsize=9)
                ax.set_title(self._fmt_dataset(dataset), fontsize=10, fontweight="bold")
                ax.set_ylim(0, 1)
                ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
                ax.set_ylabel("MIA AUC", fontsize=10, fontweight="bold")
                ax.grid(True, axis="y", alpha=0.4)

            for i in range(len(datasets), len(axes_flat)):
                fig.delaxes(axes_flat[i])

            shared_handles = [
                Patch(facecolor=offline_colors[a], edgecolor="black", alpha=0.8, label=offline_labels[a])
                for a in offline_present
            ]
            shared_handles.append(Line2D([0], [0], color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="Random guess"))
            fig.legend(handles=shared_handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(offline_present) + 1, fontsize=10, frameon=True)
            plt.tight_layout(pad=1.0)
            plt.savefig(os.path.join(save_dir, "11_attack_auc_per_dataset.png"), dpi=300, bbox_inches="tight")
            plt.close()

            # --- Plot 13: TPR@0.1%FPR comparison (offline only) ---
            tpr_df = offline_df[offline_df["tpr_at_fpr_01"].notna()]
            if not tpr_df.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                max_top = 0.0
                for i, attack in enumerate(offline_present):
                    attack_tpr = tpr_df[tpr_df["attack"] == attack]
                    means = []
                    stds = []
                    for m in offline_models:
                        vals = pd.to_numeric(
                            attack_tpr[attack_tpr["model"] == m]["tpr_at_fpr_01"],
                            errors="coerce",
                        ).dropna().to_numpy(dtype=float)
                        if len(vals) == 0:
                            means.append(np.nan)
                            stds.append(np.nan)
                            continue
                        means.append(float(np.mean(vals)))
                        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
                    means_arr = np.asarray(means, dtype=float)
                    stds_arr = np.asarray(stds, dtype=float)
                    finite = np.isfinite(means_arr)
                    if finite.any():
                        max_top = max(max_top, float(np.nanmax(means_arr[finite] + np.nan_to_num(stds_arr[finite], nan=0.0))))
                    yerr = None if np.isnan(stds_arr).all() else np.nan_to_num(stds_arr, nan=0.0)
                    ax.bar(
                        x + offsets[i], means_arr, width, yerr=yerr, capsize=4,
                        label=offline_labels[attack], color=offline_colors[attack], alpha=0.8,
                        edgecolor="black",
                    )
                ax.set_xticks(x)
                ax.set_xticklabels([self._fmt_model(m) for m in offline_models], rotation=35, ha="right")
                ax.set_xlabel("")
                ax.set_ylabel("Mean TPR @ 0.1% FPR (std across datasets)", fontweight="bold")
                ax.set_ylim(0, min(1.0, max(0.01, max_top * 1.25)))
                ax.legend(ncol=len(offline_present), loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=True)
                ax.grid(True, axis="y", alpha=0.4)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "12_tpr_comparison_by_model.png"), dpi=300, bbox_inches="tight")
                plt.close()

        # --- Plot 2: Side-by-side AUC heatmaps per attack type ---
        n_attacks = len(present_attacks)
        fig, axes = plt.subplots(1, n_attacks, figsize=(7 * n_attacks, max(5, len(df["dataset"].unique()) * 0.55 + 2)))
        if n_attacks == 1:
            axes = [axes]

        ordered_models = [m for m in self.model_plot_order if m in df["model"].unique()]
        ordered_models += sorted(m for m in df["model"].unique() if m not in ordered_models)
        rmia_summary_df = self.load_seed_summary_means(attack="rmia")

        for ax, attack in zip(axes, present_attacks):
            if attack == "rmia" and not rmia_summary_df.empty:
                attack_df = rmia_summary_df
                aggfunc = "first"
                title_suffix = " (seed mean)"
            else:
                attack_df = df[df["attack"] == attack]
                aggfunc = "mean"
                title_suffix = ""

            pivot = attack_df.pivot_table(
                values="attack_auc", index="dataset", columns="model", aggfunc=aggfunc
            )
            cols = [m for m in ordered_models if m in pivot.columns]
            pivot = pivot[cols]
            pivot.index = pivot.index.map(self._fmt_dataset)
            pivot.columns = pivot.columns.map(self._fmt_model)
            sns.heatmap(
                pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0.5,
                vmin=0, vmax=1, ax=ax, linewidths=0.5,
                cbar_kws={"label": "MIA AUC"},
            )
            ax.set_title(f"{attack_labels[attack]}{title_suffix}", fontsize=12, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")

        # plt.suptitle("Attack AUC Heatmaps", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "10_attack_auc_heatmaps.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # --- Save CSV: wide format AUC comparison per (dataset, model, attack) ---
        csv_df = df.copy()
        if not rmia_summary_df.empty:
            csv_df = pd.concat(
                [df[df["attack"] != "rmia"], rmia_summary_df[df.columns.intersection(rmia_summary_df.columns)]],
                ignore_index=True,
                sort=False,
            )
        pivot_csv = csv_df.pivot_table(
            values="attack_auc", index=["dataset", "model"], columns="attack", aggfunc="mean"
        ).reset_index()
        pivot_csv.to_csv(os.path.join(save_dir, "10_attack_auc_comparison.csv"), index=False)


    def create_dataset_attack_roc_curves(self, save_dir: str = "results_visualizations/attacks_viz"):
        """Average ROC curves for selected datasets, averaged over seeds then models."""
        os.makedirs(save_dir, exist_ok=True)
        datasets = ["locations", "dropout_success"]
        attacks = ["rmia", "lira", "attack_p"]
        attack_labels = {
            "rmia": "RMIA",
            "lira": "LiRA",
            "attack_p": "Attack-P",
        }
        attack_colors = {
            "rmia": self.RMIA_COLOR,
            "lira": self.LIRA_COLOR,
            "attack_p": self.ATTACK_P_COLOR,
        }
        fpr_grid = np.linspace(0.0, 1.0, 401)
        rows = []
        summary_rows = []

        def _load_curve(npz_path: Path):
            data = np.load(npz_path)
            if "fpr" not in data or "tpr" not in data:
                return None
            fpr = np.asarray(data["fpr"], dtype=float)
            tpr = np.asarray(data["tpr"], dtype=float)
            valid = np.isfinite(fpr) & np.isfinite(tpr)
            fpr, tpr = fpr[valid], tpr[valid]
            if len(fpr) < 2:
                return None
            order = np.argsort(fpr)
            fpr, tpr = fpr[order], tpr[order]
            fpr, unique_idx = np.unique(fpr, return_index=True)
            tpr = tpr[unique_idx]
            if fpr[0] > 0.0:
                fpr = np.r_[0.0, fpr]
                tpr = np.r_[0.0, tpr]
            if fpr[-1] < 1.0:
                fpr = np.r_[fpr, 1.0]
                tpr = np.r_[tpr, 1.0]
            auc = float(data["auc"]) if "auc" in data else float(np.trapz(tpr, fpr))
            return np.interp(fpr_grid, fpr, tpr), auc

        base_path = Path(self.logs_base_dir)
        for dataset in datasets:
            for attack in attacks:
                model_curves = []
                model_aucs = []
                model_seed_counts = []
                for model in self.model_plot_order:
                    curves = []
                    aucs = []
                    model_base = base_path / dataset / model
                    for npz_path in sorted(model_base.glob(f"seed*/{attack}/report/exp/attack_result_0.npz")):
                        loaded = _load_curve(npz_path)
                        if loaded is None:
                            continue
                        curve, auc = loaded
                        curves.append(curve)
                        aucs.append(auc)
                    if not curves:
                        continue
                    model_curves.append(np.vstack(curves).mean(axis=0))
                    model_aucs.append(float(np.mean(aucs)))
                    model_seed_counts.append(len(curves))

                if not model_curves:
                    logger.warning("No ROC curves found for dataset=%s attack=%s", dataset, attack)
                    continue

                curve_stack = np.vstack(model_curves)
                mean_tpr = curve_stack.mean(axis=0)
                std_tpr = curve_stack.std(axis=0, ddof=1) if len(model_curves) > 1 else np.zeros_like(mean_tpr)
                mean_auc = float(np.mean(model_aucs))
                std_auc = float(np.std(model_aucs, ddof=1)) if len(model_aucs) > 1 else 0.0
                total_seed_curves = int(sum(model_seed_counts))

                summary_rows.append({
                    "dataset": dataset,
                    "attack": attack,
                    "attack_label": attack_labels[attack],
                    "mean_auc": mean_auc,
                    "std_auc_across_models": std_auc,
                    "n_models": len(model_curves),
                    "n_seed_curves": total_seed_curves,
                })
                for fpr_value, mean_value, std_value in zip(fpr_grid, mean_tpr, std_tpr):
                    rows.append({
                        "dataset": dataset,
                        "attack": attack,
                        "attack_label": attack_labels[attack],
                        "fpr": fpr_value,
                        "mean_tpr": mean_value,
                        "std_tpr_across_models": std_value,
                        "mean_auc": mean_auc,
                        "std_auc_across_models": std_auc,
                        "n_models": len(model_curves),
                        "n_seed_curves": total_seed_curves,
                    })

        roc_df = pd.DataFrame(rows)
        if roc_df.empty:
            logger.warning("No dataset-level ROC curves could be created.")
            return
        roc_df.to_csv(os.path.join(save_dir, "22_dataset_attack_roc_curves.csv"), index=False)
        pd.DataFrame(summary_rows).to_csv(os.path.join(save_dir, "22_dataset_attack_roc_summary.csv"), index=False)

        fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharex=True, sharey=True)
        for ax, dataset in zip(axes, datasets):
            panel = roc_df[roc_df["dataset"] == dataset]
            for attack in attacks:
                curve = panel[panel["attack"] == attack]
                if curve.empty:
                    continue
                mean_auc = curve["mean_auc"].iloc[0]
                std_auc = curve["std_auc_across_models"].iloc[0]
                n_models = int(curve["n_models"].iloc[0])
                ax.plot(
                    curve["fpr"],
                    curve["mean_tpr"],
                    color=attack_colors[attack],
                    linewidth=2.1,
                    label=f"{attack_labels[attack]} (AUC = {mean_auc:.3f} $\\pm$ {std_auc:.3f})",
                )
            ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=2.4, alpha=0.9, label="Random guess")
            ax.set_title(self._fmt_dataset(dataset), fontsize=12, fontweight="bold")
            ax.set_xlabel("False positive rate", fontweight="bold")
            ax.grid(True, alpha=0.35)
            ax.legend(loc="lower right", frameon=True, fontsize=8)
        axes[0].set_ylabel("True positive rate", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "22_dataset_attack_roc_curves.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved dataset attack ROC curves to %s", os.path.join(save_dir, "22_dataset_attack_roc_curves.png"))


    def create_locations_tabpfn_attack_roc_curves(self, save_dir: str = "results_visualizations/attacks_viz"):
        """ROC curves for locations + TabPFN, averaged across seeds for each attack."""
        os.makedirs(save_dir, exist_ok=True)
        dataset = "locations"
        model = "tabpfn"
        attacks = ["rmia", "lira", "attack_p"]
        attack_labels = {
            "rmia": "RMIA",
            "lira": "LiRA",
            "attack_p": "Attack-P",
        }
        attack_colors = {
            "rmia": self.RMIA_COLOR,
            "lira": self.LIRA_COLOR,
            "attack_p": self.ATTACK_P_COLOR,
        }
        fpr_grid = np.linspace(0.0, 1.0, 401)
        rows = []
        summary_rows = []

        def _load_curve(npz_path: Path):
            data = np.load(npz_path)
            if "fpr" not in data or "tpr" not in data:
                return None
            fpr = np.asarray(data["fpr"], dtype=float)
            tpr = np.asarray(data["tpr"], dtype=float)
            valid = np.isfinite(fpr) & np.isfinite(tpr)
            fpr, tpr = fpr[valid], tpr[valid]
            if len(fpr) < 2:
                return None
            order = np.argsort(fpr)
            fpr, tpr = fpr[order], tpr[order]
            fpr, unique_idx = np.unique(fpr, return_index=True)
            tpr = tpr[unique_idx]
            if fpr[0] > 0.0:
                fpr = np.r_[0.0, fpr]
                tpr = np.r_[0.0, tpr]
            if fpr[-1] < 1.0:
                fpr = np.r_[fpr, 1.0]
                tpr = np.r_[tpr, 1.0]
            auc = float(data["auc"]) if "auc" in data else float(np.trapz(tpr, fpr))
            return np.interp(fpr_grid, fpr, tpr), auc

        base_path = Path(self.logs_base_dir) / dataset / model
        for attack in attacks:
            curves = []
            aucs = []
            for npz_path in sorted(base_path.glob(f"seed*/{attack}/report/exp/attack_result_0.npz")):
                loaded = _load_curve(npz_path)
                if loaded is None:
                    continue
                curve, auc = loaded
                curves.append(curve)
                aucs.append(auc)

            if not curves:
                logger.warning("No ROC curves found for dataset=%s model=%s attack=%s", dataset, model, attack)
                continue

            curve_stack = np.vstack(curves)
            mean_tpr = curve_stack.mean(axis=0)
            std_tpr = curve_stack.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean_tpr)
            mean_auc = float(np.mean(aucs))
            std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0

            summary_rows.append({
                "dataset": dataset,
                "model": model,
                "attack": attack,
                "attack_label": attack_labels[attack],
                "mean_auc": mean_auc,
                "std_auc_across_seeds": std_auc,
                "n_seeds": len(curves),
            })
            for fpr_value, mean_value, std_value in zip(fpr_grid, mean_tpr, std_tpr):
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "attack": attack,
                    "attack_label": attack_labels[attack],
                    "fpr": fpr_value,
                    "mean_tpr": mean_value,
                    "std_tpr_across_seeds": std_value,
                    "mean_auc": mean_auc,
                    "std_auc_across_seeds": std_auc,
                    "n_seeds": len(curves),
                })

        roc_df = pd.DataFrame(rows)
        if roc_df.empty:
            logger.warning("No locations TabPFN ROC curves could be created.")
            return
        roc_df.to_csv(os.path.join(save_dir, "23_locations_tabpfn_attack_roc_curves.csv"), index=False)
        pd.DataFrame(summary_rows).to_csv(os.path.join(save_dir, "23_locations_tabpfn_attack_roc_summary.csv"), index=False)

        fig, ax = plt.subplots(figsize=(8.2, 6.4))
        for attack in attacks:
            curve = roc_df[roc_df["attack"] == attack]
            if curve.empty:
                continue
            mean_auc = curve["mean_auc"].iloc[0]
            std_auc = curve["std_auc_across_seeds"].iloc[0]
            n_seeds = int(curve["n_seeds"].iloc[0])
            ax.plot(
                curve["fpr"],
                curve["mean_tpr"],
                color=attack_colors[attack],
                linewidth=3.6,
                label=f"{attack_labels[attack]} (AUC = {mean_auc:.3f} $\\pm$ {std_auc:.3f})",
            )
        ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=2.4, alpha=0.9, label="Random guess")
        ax.set_title(self._fmt_dataset(dataset), fontsize=22, fontweight="bold")
        ax.set_xlabel("False positive rate", fontsize=19, fontweight="bold")
        ax.set_ylabel("True positive rate", fontsize=19, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.35)
        ax.tick_params(axis="both", labelsize=17, width=1.6, length=6)
        ax.legend(loc="lower right", frameon=True, fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "23_locations_tabpfn_attack_roc_curves.png"), dpi=300, bbox_inches="tight")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(1e-4, 1)
        ax.set_xlabel("False positive rate", fontsize=19, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "23b_locations_tabpfn_attack_roc_curves_logx.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved locations TabPFN ROC curves to %s", os.path.join(save_dir, "23_locations_tabpfn_attack_roc_curves.png"))

    def create_dropout_success_tabpfn_attack_roc_curves(self, save_dir: str = "results_visualizations/attacks_viz"):
        """ROC curves for dropout_success + TabPFN, averaged across seeds for each attack."""
        os.makedirs(save_dir, exist_ok=True)
        dataset = "dropout_success"
        model = "tabpfn"
        attacks = ["rmia", "lira", "attack_p"]
        attack_labels = {
            "rmia": "RMIA",
            "lira": "LiRA",
            "attack_p": "Attack-P",
        }
        attack_colors = {
            "rmia": self.RMIA_COLOR,
            "lira": self.LIRA_COLOR,
            "attack_p": self.ATTACK_P_COLOR,
        }
        fpr_grid = np.linspace(0.0, 1.0, 401)
        rows = []
        summary_rows = []

        def _load_curve(npz_path: Path):
            data = np.load(npz_path)
            if "fpr" not in data or "tpr" not in data:
                return None
            fpr = np.asarray(data["fpr"], dtype=float)
            tpr = np.asarray(data["tpr"], dtype=float)
            valid = np.isfinite(fpr) & np.isfinite(tpr)
            fpr, tpr = fpr[valid], tpr[valid]
            if len(fpr) < 2:
                return None
            order = np.argsort(fpr)
            fpr, tpr = fpr[order], tpr[order]
            fpr, unique_idx = np.unique(fpr, return_index=True)
            tpr = tpr[unique_idx]
            if fpr[0] > 0.0:
                fpr = np.r_[0.0, fpr]
                tpr = np.r_[0.0, tpr]
            if fpr[-1] < 1.0:
                fpr = np.r_[fpr, 1.0]
                tpr = np.r_[tpr, 1.0]
            auc = float(data["auc"]) if "auc" in data else float(np.trapz(tpr, fpr))
            return np.interp(fpr_grid, fpr, tpr), auc

        base_path = Path(self.logs_base_dir) / dataset / model
        for attack in attacks:
            curves = []
            aucs = []
            for npz_path in sorted(base_path.glob(f"seed*/{attack}/report/exp/attack_result_0.npz")):
                loaded = _load_curve(npz_path)
                if loaded is None:
                    continue
                curve, auc = loaded
                curves.append(curve)
                aucs.append(auc)

            if not curves:
                logger.warning("No ROC curves found for dataset=%s model=%s attack=%s", dataset, model, attack)
                continue

            curve_stack = np.vstack(curves)
            mean_tpr = curve_stack.mean(axis=0)
            std_tpr = curve_stack.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean_tpr)
            mean_auc = float(np.mean(aucs))
            std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0

            summary_rows.append({
                "dataset": dataset,
                "model": model,
                "attack": attack,
                "attack_label": attack_labels[attack],
                "mean_auc": mean_auc,
                "std_auc_across_seeds": std_auc,
                "n_seeds": len(curves),
            })
            for fpr_value, mean_value, std_value in zip(fpr_grid, mean_tpr, std_tpr):
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "attack": attack,
                    "attack_label": attack_labels[attack],
                    "fpr": fpr_value,
                    "mean_tpr": mean_value,
                    "std_tpr_across_seeds": std_value,
                    "mean_auc": mean_auc,
                    "std_auc_across_seeds": std_auc,
                    "n_seeds": len(curves),
                })

        roc_df = pd.DataFrame(rows)
        if roc_df.empty:
            logger.warning("No dropout_success TabPFN ROC curves could be created.")
            return
        roc_df.to_csv(os.path.join(save_dir, "24_dropout_success_tabpfn_attack_roc_curves.csv"), index=False)
        pd.DataFrame(summary_rows).to_csv(os.path.join(save_dir, "24_dropout_success_tabpfn_attack_roc_summary.csv"), index=False)

        fig, ax = plt.subplots(figsize=(8.2, 6.4))
        for attack in attacks:
            curve = roc_df[roc_df["attack"] == attack]
            if curve.empty:
                continue
            mean_auc = curve["mean_auc"].iloc[0]
            std_auc = curve["std_auc_across_seeds"].iloc[0]
            n_seeds = int(curve["n_seeds"].iloc[0])
            ax.plot(
                curve["fpr"],
                curve["mean_tpr"],
                color=attack_colors[attack],
                linewidth=3.6,
                label=f"{attack_labels[attack]} (AUC = {mean_auc:.3f} $\\pm$ {std_auc:.3f})",
            )
        ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=2.4, alpha=0.9, label="Random guess")
        ax.set_title(self._fmt_dataset(dataset), fontsize=22, fontweight="bold")
        ax.set_xlabel("False positive rate", fontsize=19, fontweight="bold")
        ax.set_ylabel("True positive rate", fontsize=19, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.35)
        ax.tick_params(axis="both", labelsize=17, width=1.6, length=6)
        ax.legend(loc="lower right", frameon=True, fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "24_dropout_success_tabpfn_attack_roc_curves.png"), dpi=300, bbox_inches="tight")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(1e-4, 1)
        ax.set_xlabel("False positive rate", fontsize=19, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "24b_dropout_success_tabpfn_attack_roc_curves_logx.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved dropout_success TabPFN ROC curves to %s", os.path.join(save_dir, "24_dropout_success_tabpfn_attack_roc_curves.png"))


    @staticmethod
    def _roc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute ROC exactly like AMIA plot 03: sklearn roc_curve + auc."""
        labels = np.asarray(y_true).astype(int).ravel()
        scores = np.asarray(scores, dtype=float).ravel()
        valid = np.isfinite(scores)
        if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5
        fpr, tpr, _ = roc_curve(labels[valid], scores[valid])
        return fpr, tpr, float(sk_auc(fpr, tpr))

    def _collect_tabpfn_rmia_amia_roc_rows(self, dataset: str, model: str = "tabpfn") -> tuple[pd.DataFrame, pd.DataFrame]:
        """Collect RMIA and AMIA ROC curves from AMIA attention summaries."""
        base_path = Path(self.logs_base_dir) / dataset / model
        score_specs = {
            "rmia": ("RMIA", "rmia_score"),
            "amia": ("AMIA", "row_max"),
        }
        fpr_grid = np.r_[np.linspace(0.0, 0.10, 501), np.linspace(0.105, 1.0, 180)]
        rows = []
        summary_rows = []

        for attack, (attack_label, score_col) in score_specs.items():
            curves = []
            aucs = []
            seed_meta = []
            for summary_path in sorted(base_path.glob("seed*/amia/report/exp/attention_summary.csv")):
                seed_part = summary_path.relative_to(base_path).parts[0]
                if not re.fullmatch(r"seed\d+", seed_part):
                    continue
                seed = int(seed_part.replace("seed", ""))
                try:
                    summary = pd.read_csv(summary_path)
                except Exception as exc:
                    logger.warning("Could not read %s: %s", summary_path, exc)
                    continue
                if "member" not in summary.columns or score_col not in summary.columns:
                    continue
                y_true = pd.to_numeric(summary["member"], errors="coerce").fillna(0).astype(int).to_numpy(dtype=bool)
                scores = pd.to_numeric(summary[score_col], errors="coerce").to_numpy(dtype=float)
                fpr, tpr, auc = self._roc_from_scores(y_true, scores)
                if len(fpr) < 2 or pd.isna(auc):
                    continue
                curves.append(np.interp(fpr_grid, fpr, tpr))
                aucs.append(auc)
                seed_meta.append({
                    "seed": seed,
                    "auc": auc,
                    "n_samples": len(summary),
                    "n_members": int(y_true.sum()),
                    "n_nonmembers": int((~y_true).sum()),
                    "source": str(summary_path),
                })

            if not curves:
                logger.warning("No ROC curves found for dataset=%s model=%s attack=%s", dataset, model, attack)
                continue

            curve_stack = np.vstack(curves)
            mean_tpr = curve_stack.mean(axis=0)
            std_tpr = curve_stack.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean_tpr)
            mean_auc = float(np.mean(aucs))
            std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
            n_seeds = len(curves)

            summary_rows.append({
                "dataset": dataset,
                "model": model,
                "attack": attack,
                "attack_label": attack_label,
                "score_col": score_col,
                "mean_auc": mean_auc,
                "std_auc_across_seeds": std_auc,
                "n_seeds": n_seeds,
                "n_samples_mean": float(np.mean([m["n_samples"] for m in seed_meta])),
                "n_members_mean": float(np.mean([m["n_members"] for m in seed_meta])),
                "n_nonmembers_mean": float(np.mean([m["n_nonmembers"] for m in seed_meta])),
            })
            for fpr_value, mean_value, std_value in zip(fpr_grid, mean_tpr, std_tpr):
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "attack": attack,
                    "attack_label": attack_label,
                    "score_col": score_col,
                    "fpr": fpr_value,
                    "mean_tpr": mean_value,
                    "std_tpr_across_seeds": std_value,
                    "mean_auc": mean_auc,
                    "std_auc_across_seeds": std_auc,
                    "n_seeds": n_seeds,
                })

        return pd.DataFrame(rows), pd.DataFrame(summary_rows)

    def create_tabpfn_rmia_amia_roc_curves(
        self,
        dataset: str,
        plot_prefix: str,
        save_dir: str = "results_visualizations/attacks_viz",
        model: str = "tabpfn",
    ):
        """ROC curves for RMIA and AMIA row-max on the same audited AMIA samples."""
        os.makedirs(save_dir, exist_ok=True)
        roc_df, summary_df = self._collect_tabpfn_rmia_amia_roc_rows(dataset=dataset, model=model)
        if roc_df.empty:
            logger.warning("No RMIA/AMIA ROC rows found for dataset=%s model=%s", dataset, model)
            return

        curves_csv = os.path.join(save_dir, f"{plot_prefix}_{dataset}_{model}_rmia_amia_roc_curves.csv")
        summary_csv = os.path.join(save_dir, f"{plot_prefix}_{dataset}_{model}_rmia_amia_roc_summary.csv")
        roc_df.to_csv(curves_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)

        attacks = ["rmia", "amia"]
        attack_colors = {"rmia": self.RMIA_COLOR, "amia": self.AMIA_COLOR}
        attack_styles = {"rmia": "-", "amia": "-"}
        fig, ax = plt.subplots(figsize=(5.4, 4.2), constrained_layout=True)
        for attack in attacks:
            curve = roc_df[roc_df["attack"] == attack]
            if curve.empty:
                continue
            mean_auc = float(curve["mean_auc"].iloc[0])
            std_auc = float(curve["std_auc_across_seeds"].iloc[0])
            attack_label = str(curve["attack_label"].iloc[0])
            ax.plot(
                curve["fpr"],
                curve["mean_tpr"],
                color=attack_colors[attack],
                linestyle=attack_styles[attack],
                linewidth=3.0,
                label=f"{attack_label} (AUC = {mean_auc:.3f} $\\pm$ {std_auc:.3f})",
            )
        ax.plot([0, 1], [0, 1], color="#777777", linestyle=":", linewidth=1.8, alpha=0.9, label="Random guess")
        ax.set_title(
            self._fmt_dataset(dataset),
            fontsize=17,
            fontweight="bold",
        )
        ax.set_xlabel("False positive rate", fontsize=16, fontweight="bold")
        ax.set_ylabel("True positive rate", fontsize=16, fontweight="bold")
        ax.set_xlim(0.0, 0.10)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.28)
        ax.tick_params(axis="both", labelsize=14, width=1.3, length=5)
        ax.legend(loc="upper left", frameon=True, fontsize=12)
        out = os.path.join(save_dir, f"{plot_prefix}_{dataset}_{model}_rmia_amia_roc_curves.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved RMIA/AMIA low-FPR ROC curves to %s", out)

    def create_locations_tabpfn_rmia_amia_roc_curves(self, save_dir: str = "results_visualizations/attacks_viz"):
        return self.create_tabpfn_rmia_amia_roc_curves("locations", "25", save_dir=save_dir, model="tabpfn")

    def create_dropout_success_tabpfn_rmia_amia_roc_curves(self, save_dir: str = "results_visualizations/attacks_viz"):
        return self.create_tabpfn_rmia_amia_roc_curves("dropout_success", "26", save_dir=save_dir, model="tabpfn")

    def create_all_models_rmia_amia_roc_curves(self, save_dir: str = "results_visualizations/attacks_viz"):
        """2×2 grid: one panel per FM, AMIA and RMIA curves for locations (solid) and dropout_success (dashed)."""
        os.makedirs(save_dir, exist_ok=True)

        models = ["tabpfn", "real-tabpfn", "tabicl", "tabdpt"]
        datasets = ["locations", "dropout_success"]
        dataset_styles = {"locations": "-", "dropout_success": "--"}
        dataset_labels = {"locations": "Locations", "dropout_success": "Dropout Success"}
        attack_colors = {"rmia": self.RMIA_COLOR, "amia": self.AMIA_COLOR}
        attack_display = {"rmia": "RMIA", "amia": "AMIA"}

        # collect ROC data for all combinations
        data: dict[tuple[str, str], pd.DataFrame] = {}
        for model in models:
            for dataset in datasets:
                df, _ = self._collect_tabpfn_rmia_amia_roc_rows(dataset=dataset, model=model)
                if not df.empty:
                    data[(model, dataset)] = df

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
        panel_order = [("tabpfn", axes[0, 0], True), ("real-tabpfn", axes[0, 1], True),
                       ("tabicl", axes[1, 0], False), ("tabdpt", axes[1, 1], False)]

        for model, ax, is_top_row in panel_order:
            for attack in ["rmia", "amia"]:
                for dataset in datasets:
                    df = data.get((model, dataset))
                    if df is None:
                        continue
                    curve = df[df["attack"] == attack]
                    if curve.empty:
                        continue
                    mean_auc = float(curve["mean_auc"].iloc[0])
                    std_auc = float(curve["std_auc_across_seeds"].iloc[0])
                    fpr_vals = curve["fpr"].to_numpy()
                    tpr_vals = curve["mean_tpr"].to_numpy()
                    mask = fpr_vals <= 0.10
                    label = (
                        f"{attack_display[attack]} — {dataset_labels[dataset]}"
                        f" (AUC = {mean_auc:.3f} $\\pm$ {std_auc:.3f})"
                    )
                    ax.plot(
                        fpr_vals[mask], tpr_vals[mask],
                        color=attack_colors[attack],
                        linestyle=dataset_styles[dataset],
                        linewidth=2.4,
                        label=label,
                    )
            ax.plot([0, 0.10], [0, 0.10], color="#777777", linestyle=":", linewidth=1.5, alpha=0.85)
            ax.set_title(self._fmt_model(model), fontsize=17, fontweight="bold")
            if not is_top_row:
                ax.set_xlabel("False positive rate", fontsize=15, fontweight="bold")
            ax.set_ylabel("True positive rate", fontsize=15, fontweight="bold")
            ax.set_xlim(0.0, 0.10)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=13)
            legend_loc = "lower right" if not is_top_row else "upper left"
            ax.legend(loc=legend_loc, frameon=True, fontsize=10.5, handlelength=2.2)

        out = os.path.join(save_dir, "27b_all_models_rmia_amia_roc_curves.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved all-models RMIA/AMIA ROC grid to %s", out)

    def create_online_offline_comparison(self, save_dir: str = "results_visualizations/attacks_viz"):
        """Compare online vs offline variants for RMIA and LiRA.
        """
        os.makedirs(save_dir, exist_ok=True)

        df = self._results_df_with_seed_summary_means()
        df = df[df["attack_auc"].notna() & df["attack"].isin(
            ["rmia", "rmia_online", "lira", "lira_online"]
        )].copy()

        if df.empty:
            logger.warning("No online/offline results found; skipping online vs offline comparison.")
            return

        colors = {
            "rmia":        "#086375",
            "rmia_online": "#0fa8c4",
            "lira":        "#ee6c4d",
            "lira_online": "#f4a67a",
        }
        labels = {
            "rmia":        "RMIA offline",
            "rmia_online": "RMIA online",
            "lira":        "LiRA offline",
            "lira_online": "LiRA online",
        }

        # Save CSV
        df[["dataset", "model", "attack", "attack_auc", "tpr_at_fpr_01", "tpr_at_fpr_0", "tnr_at_fnr_01", "tnr_at_fnr_0"]].to_csv(
            os.path.join(save_dir, "online_offline_comparison.csv"), index=False
        )

        models = self._sort_models(list(df["model"].unique()))
        x = np.arange(len(models))

        # --- Plot 15a: RMIA offline vs online (mean AUC per model) ---
        rmia_df = df[df["attack"].isin(["rmia", "rmia_online"])]
        if not rmia_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            width = 0.35
            for i, attack in enumerate(["rmia", "rmia_online"]):
                sub = rmia_df[rmia_df["attack"] == attack]
                means = [sub[sub["model"] == m]["attack_auc"].mean() for m in models]
                stds  = [sub[sub["model"] == m]["attack_auc"].std() for m in models]
                ax.bar(x + (i - 0.5) * width, means, width, yerr=stds,
                       label=labels[attack], color=colors[attack], alpha=0.85,
                       capsize=4, edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels([self._fmt_model(m) for m in models], rotation=35, ha="right")
            ax.set_ylabel("MIA AUC", fontweight="bold")
            ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Random guess")
            ax.set_ylim(bottom=0)
            ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.1), frameon=True)
            ax.grid(True, axis="y", alpha=0.4)
            # ax.set_title("RMIA: Online vs Offline — Mean AUC by Model", fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "14a_rmia_online_vs_offline_by_model.png"), dpi=300, bbox_inches="tight")
            plt.close()

        # --- Plot 15b: LiRA offline vs online (mean AUC per model) ---
        lira_df = df[df["attack"].isin(["lira", "lira_online"])]
        if not lira_df.empty:
            lira_models = self._sort_models(list(lira_df["model"].unique()))
            x_lira = np.arange(len(lira_models))
            fig, ax = plt.subplots(figsize=(10, 5))
            width = 0.35
            for i, attack in enumerate(["lira", "lira_online"]):
                sub = lira_df[lira_df["attack"] == attack]
                means = [sub[sub["model"] == m]["attack_auc"].mean() for m in lira_models]
                stds  = [sub[sub["model"] == m]["attack_auc"].std() for m in lira_models]
                ax.bar(x_lira + (i - 0.5) * width, means, width, yerr=stds,
                       label=labels[attack], color=colors[attack], alpha=0.85,
                       capsize=4, edgecolor="black")
            ax.set_xticks(x_lira)
            ax.set_xticklabels([self._fmt_model(m) for m in lira_models], rotation=35, ha="right")
            ax.set_ylabel("MIA AUC", fontweight="bold")
            ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Random guess")
            ax.set_ylim(bottom=0)
            ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.1), frameon=True)
            ax.grid(True, axis="y", alpha=0.4)
            # ax.set_title("LiRA: Online vs Offline — Mean AUC by Model", fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "14b_lira_online_vs_offline_by_model.png"), dpi=300, bbox_inches="tight")
            plt.close()

        # --- Plot 15c: Heatmap of AUC_online − AUC_offline per (dataset, model) ---
        gain_rows = []
        for base_attack, online_attack in [("rmia", "rmia_online"), ("lira", "lira_online")]:
            offline_df = df[df["attack"] == base_attack][["dataset", "model", "attack_auc"]].rename(columns={"attack_auc": "auc_offline"})
            online_sub = df[df["attack"] == online_attack][["dataset", "model", "attack_auc"]].rename(columns={"attack_auc": "auc_online"})
            merged = pd.merge(offline_df, online_sub, on=["dataset", "model"], how="inner")
            merged["gain"] = merged["auc_online"] - merged["auc_offline"]
            merged["attack_type"] = base_attack.upper()
            gain_rows.append(merged)

        if gain_rows:
            gain_df = pd.concat(gain_rows, ignore_index=True)
            attack_types = ["RMIA", "LIRA"]  # fixed order
            attack_types = [a for a in attack_types if a in gain_df["attack_type"].unique()]
            n_attacks = len(attack_types)
            fig, axes = plt.subplots(1, n_attacks, figsize=(8 * n_attacks, max(4, gain_df["dataset"].nunique() * 0.5 + 2)))
            if n_attacks == 1:
                axes = [axes]

            ordered_models = [m for m in self.model_plot_order if m in gain_df["model"].unique()]
            ordered_models += sorted(m for m in gain_df["model"].unique() if m not in ordered_models)

            # Compute a single global abs_max so both panels share the same colour scale.
            global_abs_max = gain_df["gain"].abs().max()
            if global_abs_max == 0 or np.isnan(global_abs_max):
                global_abs_max = 0.05

            for ax, atype in zip(axes, attack_types):
                sub = gain_df[gain_df["attack_type"] == atype]
                pivot = sub.pivot_table(values="gain", index="dataset", columns="model", aggfunc="mean")
                cols = [m for m in ordered_models if m in pivot.columns]
                pivot = pivot[cols]
                pivot.index = pivot.index.map(self._fmt_dataset)
                pivot.columns = pivot.columns.map(self._fmt_model)
                sns.heatmap(
                    pivot, ax=ax, annot=True, fmt="+.3f",
                    cmap="RdBu_r", center=0, vmin=-global_abs_max, vmax=global_abs_max,
                    linewidths=0.5, mask=pivot.isna(),
                    cbar_kws={"label": "MIA AUC online − offline"},
                )
                ax.set_title(f"{atype}: online gain over offline", fontsize=12, fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(axis="x", rotation=35)

            plt.suptitle("MIA AUC gain from online vs offline setting\n(positive = online is stronger)", fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "14c_online_gain_heatmap.png"), dpi=300, bbox_inches="tight")
            plt.close()

    def create_dataset_properties_summary(self, save_dir: str = "results_visualizations/attacks_viz"):
        """Heatmap of normalized dataset characteristics for classification datasets."""
        os.makedirs(save_dir, exist_ok=True)

        profiles_path = Path(save_dir) / "dataset_profiles.csv"
        if not profiles_path.exists():
            profiles_path = Path("results_visualizations") / "dataset_profiles.csv"
        if not profiles_path.exists():
            logger.warning("dataset_profiles.csv not found; skipping dataset properties summary.")
            return

        df = pd.read_csv(profiles_path)
        df = self._filter_excluded_df(df, dataset_col="dataset_name")
        if profiles_path == Path(save_dir) / "dataset_profiles.csv":
            df.to_csv(profiles_path, index=False)

        # Filter to classification tasks only.
        if "task_type" in df.columns:
            df = df[df["task_type"] == "classification"]
        if df.empty:
            logger.warning("No classification datasets found in dataset_profiles.csv.")
            return

        # Filter to datasets with fewer than 50k rows and 500 features.
        if "total_rows" in df.columns:
            df = df[pd.to_numeric(df["total_rows"], errors="coerce") < 50_000]
        if "num_features" in df.columns:
            df = df[pd.to_numeric(df["num_features"], errors="coerce") < 500]
        if df.empty:
            logger.warning("No datasets remain after size filters (<50k rows, <500 features).")
            return

        # Exclude specific tabarena datasets.
        _tabarena_exclude = {"students_dropout_and_academic_success"}
        if "dataset_name" in df.columns:
            df = df[~df["dataset_name"].str.contains("|".join(_tabarena_exclude), case=False, na=False)]

        # Strip leading OpenML ID prefix from dataset names.
        if "dataset_name" in df.columns:
            df["dataset_name"] = df["dataset_name"].apply(self._fmt_dataset)
            df = df.set_index("dataset_name")

        # Properties to display and their display labels.
        prop_map = {
            "total_rows":                   "Rows",
            "num_features":                 "Features",
            "num_numerical_features":       "Numerical Features",
            "num_categorical_features":     "Categorical Features",
            "num_binary_features":          "Binary Features",
            "num_classes":                  "Classes",
            "imbalance_ratio":              "Imbalance Ratio",
            "majority_class_pct":           "Majority Class (%)",
            "target_entropy":               "Target Entropy",
            "num_features_with_outliers":   "Features w/ Outliers",
            "outlier_pct":                  "Outliers (%)",
            "samples_per_feature":          "Samples / Feature",
            "mean_abs_corr_with_target":    "Mean |Corr| w/ Target",
            "max_abs_corr_with_target":     "Max |Corr| w/ Target",
            "knn_label_disagreement":       "KNN Label Disagreement",
            "mean_pairwise_feature_corr":   "Mean Pairwise Corr",
            "mean_abs_skewness":            "Mean Skewness",
            "mean_kurtosis":                "Mean Kurtosis",
            "pca_n_components_95pct":       "PCA Components (95%)",
            "pca_explained_var_top5":       "PCA Var Top-5",
            "near_zero_variance_features":  "Near-Zero Var Features",
        }

        available = {k: v for k, v in prop_map.items() if k in df.columns}
        plot_df = df[[*available.keys()]].copy()
        plot_df = plot_df.rename(columns=available)
        plot_df = plot_df.apply(pd.to_numeric, errors="coerce")

        # Min-max normalise each column for the colour scale.
        norm_df = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min() + 1e-12)

        fig_width = max(12.0, 0.9 * len(available) + 2.0)
        fig_height = max(4.0, 0.55 * len(plot_df) + 2.0)
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(
            norm_df,
            annot=plot_df.round(2),
            fmt="",
            cmap="YlOrRd",
            linewidths=0.5,
            annot_kws={"size": 13},
            cbar_kws={"label": "Min-max normalised value", "shrink": 0.4},
        )
        plt.title("Dataset Properties — Classification Tasks", fontsize=16, fontweight="bold")
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(rotation=90, fontsize=13)
        plt.yticks(rotation=0, fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "00_dataset_properties_summary.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def create_context_size_analysis(self, save_dir: str = "results_visualizations/attacks_viz"):
        """Analyse how RMIA attack AUC and test accuracy change with context size.

        Collects results from rmia_ctx<pct>/ runs and the full 100% rmia/ run,
        then plots AUC (left axis) and mean test accuracy (right axis) against
        context percentage, with one subplot per dataset.
        """
        os.makedirs(save_dir, exist_ok=True)

        foundation_models = {"tabpfn", "tabicl"}
        ctx_pattern = re.compile(r'^rmia_ctx(\d+(?:\.\d+)?)$')
        rows = []
        for r in self.results:
            if r.get("model", "").lower() not in foundation_models:
                continue
            attack = r.get("attack", "")
            m = ctx_pattern.match(attack)
            if m:
                pct = float(m.group(1))
            elif attack == "rmia":
                pct = 100.0
            else:
                continue

            rows.append({
                "dataset": r["dataset"],
                "model": r["model"],
                "context_pct": pct,
                "attack_auc": r.get("attack_auc"),
                "mean_test_acc": r["target_test_acc"] if r.get("target_test_acc") is not None else float("nan"),
                "model_training_time": r.get("model_training_time") or float("nan"),
            })

        if not rows:
            logger.warning("No context-size results found, skipping context size analysis.")
            return

        df = pd.DataFrame(rows).dropna(subset=["attack_auc"])
        df = df.sort_values(["dataset", "model", "context_pct"])

        summary_csv = os.path.join(save_dir, "07_context_size_summary.csv")
        df.to_csv(summary_csv, index=False)

        model_colors = {
            "tabpfn": "#4C72B0", "real-tabpfn": "#4C72B0",
            "tabicl": "#55A868", "tabdpt": "#8172B2",
            "tabnet": "#CCB974", "mlp": "#C44E52",
            "lightgbm": "#64B5CD", "rf": "#DD8452",
        }

        datasets_with_ctx = sorted(df["dataset"].unique())

        # --- Plot 07a: one subplot per dataset ---
        ncols = min(3, len(datasets_with_ctx))
        nrows = math.ceil(len(datasets_with_ctx) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
        fig.suptitle("Effect of Context Size on RMIA Attack AUC", fontsize=14, fontweight="bold")

        for idx, dataset in enumerate(datasets_with_ctx):
            ax = axes[idx // ncols][idx % ncols]
            ax2 = ax.twinx()

            dset_df = df[df["dataset"] == dataset]
            models_present = self._sort_models(dset_df["model"].unique().tolist())

            for model in models_present:
                mdf = dset_df[dset_df["model"] == model].sort_values("context_pct")
                color = model_colors.get(model, "gray")
                label = self._fmt_model(model)

                ax.plot(mdf["context_pct"], mdf["attack_auc"],
                        marker="o", linewidth=2, color=color, label=label, zorder=3)

                if mdf["mean_test_acc"].notna().any():
                    ax2.plot(mdf["context_pct"], mdf["mean_test_acc"],
                             marker="s", linewidth=1.5, linestyle="--",
                             color=color, alpha=0.55, zorder=2)

            ax.axhline(0.5, color="red", linestyle=":", linewidth=1.2, alpha=0.8)
            ax.set_title(self._fmt_dataset(dataset), fontweight="bold")
            ax.set_xlabel("Context size (% of training pool)")
            ax.set_ylabel("RMIA AUC", color="black")
            ax.set_ylim(0.45, 1.02)
            ax.set_xlim(0, 105)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="lower right")

            ax2.set_ylabel("Mean test accuracy", color="gray", fontsize=9)
            ax2.set_ylim(0.45, 1.02)
            ax2.tick_params(axis="y", labelcolor="gray")

        for idx in range(len(datasets_with_ctx), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        # Shared legend note for line styles
        fig.text(0.5, -0.01,
                 "Solid = RMIA AUC (left)   |   Dashed = Test accuracy (right)",
                 ha="center", fontsize=9, color="gray")

        plt.tight_layout()
        out = os.path.join(save_dir, "07a_context_size_auc.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

        # --- Plot 07b: summary across datasets (mean AUC ± std per model) ---
        if len(datasets_with_ctx) > 1:
            summary = (
                df.groupby(["model", "context_pct"])["attack_auc"]
                .agg(["mean", "std"])
                .reset_index()
            )
            models_present = self._sort_models(summary["model"].unique().tolist())

            fig, ax = plt.subplots(figsize=(10, 5))
            for model in models_present:
                mdf = summary[summary["model"] == model].sort_values("context_pct")
                color = model_colors.get(model, "gray")
                ax.plot(mdf["context_pct"], mdf["mean"], marker="o", linewidth=2,
                        color=color, label=self._fmt_model(model))
                ax.fill_between(mdf["context_pct"],
                                mdf["mean"] - mdf["std"].fillna(0),
                                mdf["mean"] + mdf["std"].fillna(0),
                                color=color, alpha=0.15)

            ax.axhline(0.5, color="red", linestyle=":", linewidth=1.2, alpha=0.8, label="Random (AUC=0.5)")
            ax.set_title("Mean RMIA AUC vs Context Size (across all datasets)", fontweight="bold", fontsize=13)
            ax.set_xlabel("Context size (% of training pool)")
            ax.set_ylabel("Mean RMIA AUC")
            ax.set_ylim(0.45, 1.02)
            ax.set_xlim(0, 105)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            out = os.path.join(save_dir, "07b_context_size_summary.png")
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()

    def create_ctx_amia_sweep(self, save_dir: str = "results_visualizations/attacks_viz",
                              dataset: str = "locations", model: str = "tabpfn"):
        """Line plot: RMIA AUC, AMIA row_max AUC, and test accuracy vs context %.

        RMIA and accuracy come from self.results (already collected).
        AMIA AUC is parsed directly from amia_ctx{pct}/report/log_amia.log files,
        since AMIA results are not part of the standard collect_all_results pass.

        Output: results_visualizations/attacks_viz/ctx_amia_sweep_{dataset}_{model}.png
        """
        os.makedirs(save_dir, exist_ok=True)

        # ── collect RMIA AUC + accuracy directly from log files ────────────
        # (self.results may be filtered to attack=="rmia" at this point, so
        # rmia_ctx* entries would be missing — parse log files instead)
        base_log = os.path.join(self.logs_base_dir, dataset, model.lower())
        rmia_rows = {}

        # baseline 100 %
        for candidate in [
            os.path.join(base_log, "rmia", "report", "log_time_analysis.log"),
            os.path.join(base_log, "report", "log_time_analysis.log"),  # legacy layout
        ]:
            if os.path.exists(candidate):
                m = self.parse_log_file(candidate)
                rmia_rows[100.0] = {"rmia_auc": m.get("attack_auc"), "accuracy": m.get("target_test_acc")}
                break

        # ctx sweep percentages
        if os.path.isdir(base_log):
            for entry in os.listdir(base_log):
                match = re.fullmatch(r"rmia_ctx(\d+(?:\.\d+)?)", entry)
                if not match:
                    continue
                pct = float(match.group(1))
                log_path = os.path.join(base_log, entry, "report", "log_time_analysis.log")
                if os.path.exists(log_path):
                    m = self.parse_log_file(log_path)
                    rmia_rows[pct] = {"rmia_auc": m.get("attack_auc"), "accuracy": m.get("target_test_acc")}

        if not rmia_rows:
            logger.warning(
                "No context-sweep RMIA results for %s/%s — skipping ctx_amia_sweep.", dataset, model
            )
            return

        # ── collect AMIA (SDPA hook) AUC from log files ────────────────────
        amia_auc_by_pct = {}

        # baseline (100 %)
        baseline_log = os.path.join(base_log, "amia", "report", "log_amia.log")
        if os.path.exists(baseline_log):
            val = self._parse_amia_log_auc(baseline_log, "row_max")
            if val is not None:
                amia_auc_by_pct[100.0] = val

        # ctx sweep
        if os.path.isdir(base_log):
            for entry in os.listdir(base_log):
                m = re.fullmatch(r"amia_ctx(\d+)", entry)
                if not m:
                    continue
                pct = float(m.group(1))
                log_path = os.path.join(base_log, entry, "report", "log_amia.log")
                val = self._parse_amia_log_auc(log_path, "row_max")
                if val is not None:
                    amia_auc_by_pct[pct] = val

        # ── merge into one sorted series ───────────────────────────────────
        all_pcts = sorted(set(rmia_rows) | set(amia_auc_by_pct))

        pcts          = all_pcts
        rmia_aucs     = [rmia_rows.get(p, {}).get("rmia_auc")  for p in pcts]
        amia_aucs     = [amia_auc_by_pct.get(p)                for p in pcts]
        accs          = [rmia_rows.get(p, {}).get("accuracy")   for p in pcts]

        def _xy(xs, ys):
            pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
            return (list(zip(*pairs)) if pairs else ([], []))

        rmia_x,     rmia_y     = _xy(pcts, rmia_aucs)
        amia_x,     amia_y     = _xy(pcts, amia_aucs)
        acc_x,      acc_y      = _xy(pcts, accs)

        # ── plot ───────────────────────────────────────────────────────────
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()

        RMIA_C     = "#086375"
        AMIA_C     = self.AMIA_COLOR
        ACC_C      = "#6a0572"

        if rmia_x:
            ax1.plot(list(rmia_x), list(rmia_y), "o-",  color=RMIA_C,     linewidth=2,
                     markersize=6, label="RMIA AUC")
        if amia_x:
            ax1.plot(list(amia_x), list(amia_y), "s--", color=AMIA_C,     linewidth=2,
                     markersize=6, label="AMIA row_max AUC")
        if acc_x:
            ax2.plot(list(acc_x),  list(acc_y),  "^:",  color=ACC_C,      linewidth=2,
                     markersize=6, label="Test accuracy")

        ax1.axhline(0.5, color="gray", linestyle=":", linewidth=0.9)
        ax1.set_xlabel("Context size (% of training set)", fontsize=10)
        ax1.set_ylabel("MIA AUC", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Test accuracy", fontsize=10, color=ACC_C)
        ax2.tick_params(axis="y", labelcolor=ACC_C)
        ax1.set_ylim(0.45, 1.02)
        ax2.set_ylim(0.0,  1.05)
        ax1.set_xticks(pcts)
        ax1.set_xticklabels([f"{int(p)}%" for p in pcts], fontsize=8, rotation=45)
        ax1.grid(True, alpha=0.25)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right")

        ax1.set_title(
            f"MIA vulnerability vs context size — {self._fmt_dataset(dataset)} / "
            f"{self._fmt_model(model)}\n"
            "AMIA plotted only where amia_ctx* logs exist",
            fontsize=10, fontweight="bold",
        )
        fig.tight_layout()
        out = os.path.join(save_dir, f"ctx_amia_sweep_{dataset}_{model}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved ctx_amia_sweep to %s", out)

    @staticmethod
    def _parse_amia_log_auc(log_path: str, signal: str = "row_max"):
        """Return the last AUC value for *signal* from a log_amia.log file."""
        if not os.path.exists(log_path):
            return None
        pattern = re.compile(rf"AUC\s+{re.escape(signal)}\s+([0-9.]+)")
        result = None
        with open(log_path) as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    result = float(m.group(1))
        return result

    # Proxy model analysis
    # ------------------------------------------------------------------

    def collect_proxy_results(self) -> pd.DataFrame:
        """Walk rmia_proxy/<proxy>/report/exp/attack_result_0.npz and return a DataFrame.

        Columns: dataset, target_model, proxy_model, auc, tpr_at_fpr_01, tpr_at_fpr_0
        """
        base_path = Path(self.logs_base_dir)
        rows = []
        for npz_path in sorted(base_path.rglob("attack_result_0.npz")):
            # Expected: <base>/<dataset>/<target>/rmia_proxy/<proxy>/report/exp/attack_result_0.npz
            try:
                rel = npz_path.relative_to(base_path)
                parts = rel.parts  # 7 parts
                if len(parts) != 7 or parts[2] != "rmia_proxy" or parts[5] != "exp":
                    continue
                dataset      = parts[0]
                target_model = parts[1]
                proxy_model  = parts[3]
            except Exception:
                continue

            try:
                data = np.load(str(npz_path))
                rows.append({
                    "dataset":       dataset,
                    "target_model":  target_model,
                    "proxy_model":   proxy_model,
                    "auc":           float(data["auc"]),
                    "tpr_at_fpr_01": float(data["one_tenth_fpr"]) if "one_tenth_fpr" in data else np.nan,
                    "tpr_at_fpr_0":  float(data["zero_fpr"])      if "zero_fpr"       in data else np.nan,
                    "tnr_at_fnr_01": float(data["one_tenth_fnr"]) if "one_tenth_fnr" in data else np.nan,
                    "tnr_at_fnr_0":  float(data["zero_fnr"])      if "zero_fnr"       in data else np.nan,
                })
            except Exception as exc:
                logger.warning("Could not load proxy npz %s: %s", npz_path, exc)

        df = pd.DataFrame(rows)
        logger.info("Loaded %d proxy experiment results", len(df))
        return df

    def _proxy_statistical_tests(self, df: pd.DataFrame, save_dir: str, all_models: list) -> pd.DataFrame:
        """
        Statistical tests for each (target_model, proxy_model) pair vs Attack-P.

        Unit of observation: one dataset (N=19).
        For each pair, the 19 per-dataset gains (proxy_AUC - attackP_AUC) are tested.

        Tests (per pair):
          - Wilcoxon signed-rank, H0: median gain = 0  (one-sided: proxy > Attack-P)
          - Bootstrap 95% CI on mean gain

        Multiple-comparison correction:
          - Holm-Bonferroni across all (target, proxy) pairs

        Omnibus per target:
          - Friedman test across all proxies + Attack-P column

        """
        _FOUNDATION = {"tabpfn", "real-tabpfn", "tabicl", "tabdpt", "tarte"}

        # Build Attack-P and same-arch RMIA lookup per (dataset, model)
        attackp_lut: dict[tuple, float] = {}
        sameArch_lut: dict[tuple, float] = {}
        for r in self.results:
            key = (r["dataset"], r["model"])
            if r.get("attack") == "attack_p":
                attackp_lut[key] = r.get("attack_auc")
            elif r.get("attack") == "rmia":
                sameArch_lut[key] = r.get("attack_auc")

        if not attackp_lut:
            logger.warning("No Attack-P results found; skipping proxy statistical tests.")
            return pd.DataFrame()

        # Build flat table: one row per (dataset, target_model, proxy_model)
        rows = []
        for _, row in df.iterrows():
            ap = attackp_lut.get((row["dataset"], row["target_model"]))
            if ap is None:
                continue
            sa = sameArch_lut.get((row["dataset"], row["target_model"]))
            rows.append({
                "dataset":         row["dataset"],
                "target_model":    row["target_model"],
                "proxy_model":     row["proxy_model"],
                "proxy_auc":       row["auc"],
                "attackp_auc":     ap,
                "same_arch_auc":   sa,
                "gain_vs_attackp": row["auc"] - ap,
            })

        if not rows:
            logger.warning("No (proxy, Attack-P) pairs found; skipping.")
            return pd.DataFrame()

        stat_df = pd.DataFrame(rows)

        target_models = [m for m in all_models if m in stat_df["target_model"].values]
        proxy_models  = [m for m in all_models if m in stat_df["proxy_model"].values]
        datasets_all  = stat_df["dataset"].unique()
        n_ds = len(datasets_all)
        rng = np.random.default_rng(42)

        # ── Per-(target, proxy) Wilcoxon + bootstrap CI ──────────────────────
        pair_rows = []
        for target in target_models:
            for proxy in proxy_models:
                sub = stat_df[
                    (stat_df["target_model"] == target) &
                    (stat_df["proxy_model"]  == proxy)
                ]
                gains = sub["gain_vs_attackp"].dropna().values
                if len(gains) < 4:
                    continue

                bs = scipy_bootstrap(
                    (gains,), np.mean, n_resamples=2000, random_state=rng,
                    confidence_level=0.95, method="percentile",
                )
                try:
                    _, p_w = wilcoxon(gains, alternative="greater")
                except Exception:
                    p_w = np.nan

                pair_rows.append({
                    "target_model":  target,
                    "proxy_model":   proxy,
                    "target_type":   "foundation" if target in _FOUNDATION else "classical",
                    "proxy_type":    "foundation" if proxy  in _FOUNDATION else "classical",
                    "n_datasets":    len(gains),
                    "mean_gain":     float(np.mean(gains)),
                    "median_gain":   float(np.median(gains)),
                    "ci_95_lo":      float(bs.confidence_interval.low),
                    "ci_95_hi":      float(bs.confidence_interval.high),
                    "p_value_raw":   p_w,
                })

        pair_df = pd.DataFrame(pair_rows)
        if pair_df.empty:
            logger.warning("No (target, proxy) pairs with enough data; skipping.")
            return pair_df

        # Holm-Bonferroni across all pairs
        valid_mask = pair_df["p_value_raw"].notna()
        p_raw = pair_df.loc[valid_mask, "p_value_raw"].values
        n_tests = len(p_raw)
        order = np.argsort(p_raw)
        p_adj = p_raw.copy()
        for i, idx in enumerate(order):
            p_adj[idx] = min(1.0, p_raw[idx] * (n_tests - i))
        for i in range(len(order) - 2, -1, -1):
            p_adj[order[i]] = min(p_adj[order[i]], p_adj[order[i + 1]])
        pair_df.loc[valid_mask, "p_value_holm"] = p_adj
        pair_df["sig_05"] = pair_df["p_value_holm"] < 0.05
        pair_df["sig_10"] = pair_df["p_value_holm"] < 0.10

        pair_df.to_csv(os.path.join(save_dir, "P03_proxy_stats.csv"), index=False)

        # ── Friedman per target model ─────────────────────────────────────────
        friedman_rows = []
        for target in target_models:
            sub = stat_df[stat_df["target_model"] == target]
            # Exclude same-arch proxy (no cross-arch data for that cell)
            valid_proxies = [p for p in proxy_models if p != target]
            cols = []
            for proxy in valid_proxies:
                gains = sub[sub["proxy_model"] == proxy].set_index("dataset")["gain_vs_attackp"]
                cols.append(gains.reindex(datasets_all).values)
            cols.append(np.zeros(n_ds))  # Attack-P column (gain = 0 by definition)
            mat = np.column_stack(cols)
            ok = ~np.isnan(mat).any(axis=1)
            mat_c = mat[ok]
            fstat, fp = np.nan, np.nan
            if mat_c.shape[0] >= 3:
                try:
                    fstat, fp = friedmanchisquare(*[mat_c[:, i] for i in range(mat_c.shape[1])])
                except Exception:
                    pass
            friedman_rows.append({
                "target_model": target,
                "n_datasets":   int(ok.sum()),
                "friedman_chi2": fstat,
                "p_value":       fp,
                "significant":   fp < 0.05 if not np.isnan(fp) else False,
            })

        friedman_df = pd.DataFrame(friedman_rows)
        friedman_df.to_csv(os.path.join(save_dir, "P03_proxy_friedman.csv"), index=False)

        # ── Significance heatmap: mean gain with * / ** annotations ──────────
        piv_gain = pair_df.pivot_table(
            index="target_model", columns="proxy_model", values="mean_gain",
        ).reindex(
            index=[m for m in target_models if m in pair_df["target_model"].values],
            columns=[m for m in proxy_models  if m in pair_df["proxy_model"].values],
        )
        piv_sig05 = pair_df.pivot_table(
            index="target_model", columns="proxy_model", values="sig_05", aggfunc="first",
        ).reindex(index=piv_gain.index, columns=piv_gain.columns).fillna(False)
        piv_sig10 = pair_df.pivot_table(
            index="target_model", columns="proxy_model", values="sig_10", aggfunc="first",
        ).reindex(index=piv_gain.index, columns=piv_gain.columns).fillna(False)

        sig05_arr = piv_sig05.values.astype(bool)
        sig10_arr = piv_sig10.values.astype(bool)
        piv_gain.index   = [self._fmt_model(m) for m in piv_gain.index]
        piv_gain.columns = [self._fmt_model(m) for m in piv_gain.columns]

        abs_max = max(piv_gain.abs().max().max(), 1e-6)
        nrows_h, ncols_h = piv_gain.shape
        hw = max(5.0, ncols_h * 0.95)
        hh = max(4.0, nrows_h * 0.85)

        fig, ax = plt.subplots(figsize=(hw + 2, hh + 1.5))
        im = sns.heatmap(
            piv_gain, ax=ax, annot=False,
            cmap="RdBu_r", center=0, vmin=-abs_max, vmax=abs_max,
            linewidths=0.5, mask=piv_gain.isna(),
            cbar_kws={"label": "Mean AUC gain over Attack-P", "shrink": 0.8},
        )

        # Overlay significance markers
        for ri in range(nrows_h):
            for ci in range(ncols_h):
                val = piv_gain.iloc[ri, ci]
                if np.isnan(val):
                    continue
                marker = "*" if sig05_arr[ri, ci] else ("†" if sig10_arr[ri, ci] else "")
                label = f"{val:+.3f}\n{marker}" if marker else f"{val:+.3f}"
                txt_color = "white" if abs(val) > 0.6 * abs_max else "black"
                ax.text(ci + 0.5, ri + 0.5, label, ha="center", va="center",
                        fontsize=8, color=txt_color)

        ax.set_title(
            f"RMIA with cross-architecture proxy vs. Attack-P baseline\n"
            f"Each cell: mean AUC gain over Attack-P across {n_ds} datasets\n"
            f"Significance (Wilcoxon signed-rank, Holm-corrected): † p<0.10 · * p<0.05",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel("Proxy model (used as reference for RMIA)", fontweight="bold")
        ax.set_ylabel("Target model (victim)", fontweight="bold")
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, "P03_proxy_significance_heatmap.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close()

        return pair_df

    def create_proxy_heatmaps(self, save_dir: str = "results_visualizations/attacks_viz"):
        """Proxy model analysis: gain and consistency over same-architecture reference.
        """
        os.makedirs(save_dir, exist_ok=True)

        df = self.collect_proxy_results()
        if df.empty:
            logger.warning("No proxy results found; skipping proxy analysis.")
            return

        df.to_csv(os.path.join(save_dir, "proxy_results.csv"), index=False)

        # Canonical model order for both axes
        all_models = sorted(
            set(df["target_model"].unique()) | set(df["proxy_model"].unique()),
            key=lambda m: (self._model_order_map.get(m, len(self.model_plot_order)), m),
        )

        n = len(all_models)
        hw = max(5.0, n * 0.85)

        # --- Gain over self-proxy (regular RMIA AUC / TPR for same target model) ---
        # Build baseline from already-loaded self.results
        baseline: dict[tuple, dict] = {}
        for r in self.results:
            if r.get("attack") == "rmia":
                key = (r["dataset"], r["model"])
                baseline[key] = {
                    "auc":           r.get("attack_auc"),
                    "tpr_at_fpr_01": r.get("tpr_at_fpr_01"),
                    "tnr_at_fnr_01": r.get("tnr_at_fnr_01"),
                }

        if baseline:
            gain_rows = []
            for _, row in df.iterrows():
                b = baseline.get((row["dataset"], row["target_model"]), {})
                b_auc  = b.get("auc")
                b_tpr  = b.get("tpr_at_fpr_01")
                b_tnr  = b.get("tnr_at_fnr_01")
                gain_rows.append({
                    "dataset":       row["dataset"],
                    "target_model":  row["target_model"],
                    "proxy_model":   row["proxy_model"],
                    "auc_gain":      row["auc"]           - b_auc  if b_auc  is not None else np.nan,
                    "tpr01_gain":    row["tpr_at_fpr_01"] - b_tpr  if b_tpr  is not None else np.nan,
                    "tnr01_gain":    row["tnr_at_fnr_01"] - b_tnr  if b_tnr  is not None else np.nan,
                })

            gain_df = pd.DataFrame(gain_rows)
            n_ds = gain_df["dataset"].nunique()
            gain_df.to_csv(os.path.join(save_dir, "P01_proxy_auc_gain.csv"), index=False)

            gain_metrics = [
                ("auc_gain",   "AUC gain",         "auc"),
                ("tpr01_gain", "TPR@0.1%FPR gain", "tpr01"),
                ("tnr01_gain", "TNR@0.1%FNR gain", "tnr01"),
            ]
            for col, label, fname_tag in gain_metrics:
                if gain_df[col].isna().all():
                    continue
                piv_gain = gain_df.pivot_table(
                    index="target_model", columns="proxy_model",
                    values=col, aggfunc="mean",
                )
                piv_gain = piv_gain.reindex(
                    index=[m for m in all_models if m in piv_gain.index],
                    columns=[m for m in all_models if m in piv_gain.columns],
                )
                piv_gain.index   = [self._fmt_model(m) for m in piv_gain.index]
                piv_gain.columns = [self._fmt_model(m) for m in piv_gain.columns]

                abs_max = piv_gain.abs().max().max()
                fig, ax = plt.subplots(figsize=(hw + 2, hw))
                sns.heatmap(
                    piv_gain, ax=ax, annot=True, fmt="+.3f",
                    cmap="RdBu_r", center=0,
                    vmin=-abs_max, vmax=abs_max,
                    linewidths=0.5, mask=piv_gain.isna(),
                    cbar_kws={"label": label, "shrink": 0.8},
                )
                ax.set_title(
                    f"Proxy RMIA {label} over self-proxy\n"
                    f"(mean across {n_ds} dataset(s); positive = proxy helps)",
                    fontsize=11, fontweight="bold",
                )
                ax.set_xlabel("Proxy model (reference)", fontweight="bold")
                ax.set_ylabel("Target model", fontweight="bold")
                ax.tick_params(axis="x", rotation=35)
                ax.tick_params(axis="y", rotation=0)
                plt.tight_layout()
                out = os.path.join(save_dir, f"P01_proxy_{fname_tag}_gain.png")
                plt.savefig(out, dpi=300, bbox_inches="tight")
                plt.close()

            # --- Consistency analysis: win rate across datasets ---
            # win_rate = fraction of datasets where proxy AUC > self-proxy AUC (gain > 0)
            n_datasets_per_pair = gain_df.groupby(["target_model", "proxy_model"])["dataset"].nunique()
            wins = (gain_df[gain_df["auc_gain"] > 0]
                    .groupby(["target_model", "proxy_model"])["dataset"].nunique()
                    .rename("wins"))
            consistency = (
                gain_df.groupby(["target_model", "proxy_model"])
                .agg(
                    mean_gain=("auc_gain", "mean"),
                    std_gain=("auc_gain", "std"),
                    n_datasets=("dataset", "nunique"),
                )
                .join(wins)
                .fillna({"wins": 0})
                .assign(win_rate=lambda d: d["wins"] / d["n_datasets"])
                .reset_index()
            )
            consistency.to_csv(os.path.join(save_dir, "P02_proxy_consistency.csv"), index=False)

            # P02a: win-rate heatmap
            piv_wr = consistency.pivot_table(
                index="target_model", columns="proxy_model",
                values="win_rate", aggfunc="mean",
            )
            piv_wr = piv_wr.reindex(
                index=[m for m in all_models if m in piv_wr.index],
                columns=[m for m in all_models if m in piv_wr.columns],
            )
            piv_wr.index   = [self._fmt_model(m) for m in piv_wr.index]
            piv_wr.columns = [self._fmt_model(m) for m in piv_wr.columns]

            fig, ax = plt.subplots(figsize=(hw + 2, hw))
            sns.heatmap(
                piv_wr, ax=ax, annot=True, fmt=".0%",
                cmap="Blues", vmin=0, vmax=1,
                linewidths=0.5, mask=piv_wr.isna(),
                cbar=False,
            )
            ax.set_title(
                f"Proxy consistency — win rate across {n_ds} dataset(s)\n"
                "(% of datasets where using this proxy beats the same-architecture reference)",
                fontsize=11, fontweight="bold",
            )
            ax.set_xlabel("Proxy model (reference)", fontweight="bold")
            ax.set_ylabel("Target model", fontweight="bold")
            ax.tick_params(axis="x", rotation=35)
            ax.tick_params(axis="y", rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "P02a_proxy_winrate_heatmap.png"), dpi=300, bbox_inches="tight")
            plt.close()

        self._proxy_statistical_tests(df, save_dir, all_models)

    def create_predict_type_analysis(self, save_dir: str = "results_visualizations/attacks_viz"):
        """Analyse which predict interface each model exposes and how it relates to attack AUC.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Models that use the PyTorch get_softmax path (forward() → raw logits).
        # They never emit a "Model exposing via" log line, so we fill them in here.
        PYTORCH_LOGIT_MODELS = {"mlp"}

        df = pd.DataFrame(self.results)
        if df.empty or "predict_type" not in df.columns:
            logger.warning("No predict_type data available; skipping predict_type analysis.")
            return

        df = df.copy()
        mask = df["predict_type"].isna() & df["model"].isin(PYTORCH_LOGIT_MODELS)
        df.loc[mask, "predict_type"] = "predict_logits"

        df_typed = df[df["predict_type"].notna()].copy()
        if df_typed.empty:
            logger.warning("No rows with predict_type found in results; skipping.")
            return

        # --- CSV ---
        csv_cols = [c for c in ["dataset", "model", "attack", "predict_type", "attack_auc"] if c in df_typed.columns]
        df_typed[csv_cols].sort_values(["model", "dataset"]).to_csv(
            os.path.join(save_dir, "predict_type_summary.csv"), index=False
        )

        # Colour palette: one colour per predict_type
        all_types = sorted(df_typed["predict_type"].unique())
        type_palette = {
            "predict_logits": "#086375",
            "predict_proba": "#87bba2",
            "decision_function": "#ee6c4d",
        }
        # fall-back for any unexpected type
        fallback_colors = ["#fdc500", "#8172B2", "#C44E52"]
        for i, t in enumerate(all_types):
            if t not in type_palette:
                type_palette[t] = fallback_colors[i % len(fallback_colors)]

        models_sorted = self._sort_models(df_typed["model"].unique().tolist())

        # Build model→predict_type map (majority vote across datasets/attacks)
        model_type_map = (
            df_typed.groupby("model")["predict_type"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown")
            .to_dict()
        )

        # --- Plot 14b: mean AUC per model, coloured by predict_type ---
        df_auc = df_typed[df_typed["attack_auc"].notna()].copy()
        if df_auc.empty:
            logger.warning("No AUC values to compare by predict_type; skipping 14b.")
            return

        fig, ax_model = plt.subplots(1, 1, figsize=(7, 5))
        mean_auc = (
            df_auc.groupby("model")["attack_auc"].mean().reset_index()
        )
        mean_auc["predict_type"] = mean_auc["model"].map(model_type_map)
        mean_auc = mean_auc[mean_auc["predict_type"].notna()]
        mean_auc = mean_auc.assign(
            _order=mean_auc["model"].map(lambda m: self._model_order_map.get(m, len(self.model_plot_order)))
        ).sort_values("_order")

        bar_colors = [type_palette.get(t, "gray") for t in mean_auc["predict_type"]]
        x_pos = np.arange(len(mean_auc))
        ax_model.bar(x_pos, mean_auc["attack_auc"], color=bar_colors, edgecolor="black", alpha=0.85)
        ax_model.set_xticks(x_pos)
        ax_model.set_xticklabels([self._fmt_model(m) for m in mean_auc["model"]], rotation=35, ha="right")
        ax_model.axhline(0.5, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
        ax_model.set_ylabel("MIA AUC", fontweight="bold")
        ax_model.set_ylim(0, 1.05)
        ax_model.grid(True, axis="y", alpha=0.3)

        legend_patches = [Patch(facecolor=type_palette[t], edgecolor="black", alpha=0.85, label=t) for t in all_types]
        legend_patches.append(
            Line2D([0], [0], color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="Random guess")
        )
        ax_model.legend(handles=legend_patches, fontsize=8, loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "13_predict_type_auc_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _binary_auc_from_scores(scores, labels) -> float:
        """Compute AUC where larger scores indicate membership."""
        scores = pd.to_numeric(pd.Series(scores), errors="coerce").to_numpy(dtype=float)
        labels = pd.to_numeric(pd.Series(labels), errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(scores) & np.isfinite(labels)
        scores = scores[valid]
        labels = labels[valid].astype(int)
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return np.nan
        ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=float)
        pos_rank_sum = ranks[labels == 1].sum()
        return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

    def load_amia_seed_results(self) -> pd.DataFrame:
        """Load one AMIA metric row per dataset/model/seed.

        Preferred source is attack_result_seed_runs.csv because AMIA scripts append
        row_max_auc/row_ent_auc there. If a run row is absent, recompute the same
        AUCs from seed*/amia/report/exp/attention_summary.csv.
        """
        base_path = Path(self.logs_base_dir)
        metric_cols = [
            "row_max_auc",
            "row_ent_auc",
            "col_max_auc",
            "col_ent_auc",
            "rmia_score_auc",
        ]
        rows = []
        seen = set()

        for runs_path in sorted(base_path.glob("*/*/attack_result_seed_runs.csv")):
            rel_parts = runs_path.relative_to(base_path).parts
            if len(rel_parts) < 3:
                continue
            dataset, model = rel_parts[0], rel_parts[1]
            try:
                runs = pd.read_csv(runs_path)
            except Exception as exc:
                logger.warning("Error reading AMIA seed runs %s: %s", runs_path, exc)
                continue
            if "attack" not in runs.columns or "seed" not in runs.columns:
                continue
            amia = runs[runs["attack"] == "amia"].copy()
            for _, row in amia.iterrows():
                try:
                    seed = int(row["seed"])
                except Exception:
                    continue
                out = {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "source": str(runs_path),
                }
                has_metric = False
                for col in metric_cols:
                    value = pd.to_numeric(row.get(col, np.nan), errors="coerce")
                    out[col] = float(value) if pd.notna(value) else np.nan
                    has_metric = has_metric or pd.notna(value)
                if has_metric:
                    rows.append(out)
                    seen.add((dataset, model, seed))

        score_to_metric = {
            "row_max": "row_max_auc",
            "row_ent": "row_ent_auc",
            "col_max": "col_max_auc",
            "col_ent": "col_ent_auc",
            "rmia_score": "rmia_score_auc",
        }
        for summary_path in sorted(base_path.glob("*/*/seed*/amia/report/exp/attention_summary.csv")):
            rel_parts = summary_path.relative_to(base_path).parts
            if len(rel_parts) < 6:
                continue
            dataset, model, seed_part = rel_parts[0], rel_parts[1], rel_parts[2]
            if not re.fullmatch(r"seed\d+", seed_part):
                continue
            seed = int(seed_part.replace("seed", ""))
            if (dataset, model, seed) in seen:
                continue
            try:
                summary = pd.read_csv(summary_path)
            except Exception as exc:
                logger.warning("Error reading AMIA attention summary %s: %s", summary_path, exc)
                continue
            if "member" not in summary.columns:
                continue
            out = {
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "source": str(summary_path),
            }
            has_metric = False
            for score_col, metric_col in score_to_metric.items():
                if score_col in summary.columns:
                    out[metric_col] = self._binary_auc_from_scores(summary[score_col], summary["member"])
                    has_metric = has_metric or pd.notna(out[metric_col])
                else:
                    out[metric_col] = np.nan
            if has_metric:
                rows.append(out)
                seen.add((dataset, model, seed))

        df = self._filter_excluded_df(pd.DataFrame(rows))
        if not df.empty:
            logger.info("Loaded %d AMIA seed rows", len(df))
        return df

    def summarize_amia_seed_results(self) -> pd.DataFrame:
        """Summarize AMIA seed rows to dataset/model means and seed stds."""
        seed_df = self.load_amia_seed_results()
        if seed_df.empty:
            return seed_df
        metric_cols = [c for c in seed_df.columns if c.endswith("_auc")]
        grouped = seed_df.groupby(["dataset", "model"], as_index=False)
        mean_df = grouped[metric_cols].mean()
        std_df = grouped[metric_cols].std().rename(columns={c: f"{c}_std" for c in metric_cols})
        count_df = grouped["seed"].nunique().rename(columns={"seed": "num_seeds"})
        out = mean_df.merge(std_df, on=["dataset", "model"], how="left").merge(
            count_df, on=["dataset", "model"], how="left"
        )
        return out

    def create_amia_results_plots(self, save_dir: str = "results_visualizations/attacks_viz"):
        """Create AMIA result diagnostics from per-seed attention summaries."""
        os.makedirs(save_dir, exist_ok=True)
        seed_df = self.load_amia_seed_results()
        summary_df = self.summarize_amia_seed_results()
        if seed_df.empty or summary_df.empty:
            logger.warning("No AMIA seed results found; skipping AMIA plots.")
            return

        seed_csv = os.path.join(save_dir, "16_amia_seed_runs.csv")
        summary_csv = os.path.join(save_dir, "16_amia_seed_summary.csv")
        seed_df.sort_values(["dataset", "model", "seed"]).to_csv(seed_csv, index=False)
        summary_df.sort_values(["dataset", "model"]).to_csv(summary_csv, index=False)

        rmia_summary_df = self.load_seed_summary_means(attack="rmia")
        rmia_online_summary_df = self.load_seed_summary_means(attack="rmia_online")
        if rmia_online_summary_df.empty:
            collected_df = self._aggregate_seed_rows(pd.DataFrame(self.results))
            if not collected_df.empty and "attack" in collected_df.columns:
                rmia_online_summary_df = collected_df[
                    collected_df["attack"].eq("rmia_online")
                    & collected_df.get("attack_auc", pd.Series(dtype=float)).notna()
                ].copy()
        foundation_models = {"tabpfn", "real-tabpfn", "tabicl", "tabdpt"}

        def _models_from(df: pd.DataFrame) -> set[str]:
            if df.empty or "model" not in df.columns:
                return set()
            return set(df["model"].dropna())

        amia_models = self._sort_models(
            sorted(
                (_models_from(summary_df) | _models_from(rmia_summary_df) | _models_from(rmia_online_summary_df))
                & foundation_models
            )
        )

        # 16: Average attack AUC per foundation model. Rows are dataset-level seed means;
        # error bars are std across datasets, matching plot 09 semantics.
        if "row_max_auc" in summary_df.columns and summary_df["row_max_auc"].notna().any():
            series = [
                ("RMIA", rmia_summary_df, "attack_auc", "#086375"),
                ("AMIA", summary_df, "row_max_auc", self.AMIA_COLOR),
            ]
            x = np.arange(len(amia_models)) * 0.48
            width = min(0.10, 0.28 / max(1, len(series)))
            offsets = np.linspace(-(len(series) - 1) * width / 2, (len(series) - 1) * width / 2, len(series))
            fig, ax = plt.subplots(figsize=(7.4, 3.6))
            for i, (label, plot_df, metric, color) in enumerate(series):
                means = []
                stds = []
                for model in amia_models:
                    vals = plot_df[plot_df["model"] == model][metric]
                    means.append(vals.mean())
                    stds.append(vals.std())
                stds = np.asarray(stds, dtype=float)
                yerr = None if np.isnan(stds).all() else np.nan_to_num(stds, nan=0.0)
                ax.bar(
                    x + offsets[i], means, width, yerr=yerr, capsize=4,
                    label=label, color=color, alpha=0.85, edgecolor="black",
                )
            ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([self._fmt_model(m) for m in amia_models], rotation=30, ha="right", fontsize=10)
            ax.set_ylabel("MIA AUC", fontsize=12, fontweight="bold")
            ax.set_ylim(0, 1.05)
            handles, labels = ax.get_legend_handles_labels()
            handles.append(Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Random guess"))
            ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=len(series) + 1, frameon=True, fontsize=10)
            ax.tick_params(axis="y", labelsize=10)
            ax.grid(True, axis="y", alpha=0.35)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "16_amia_auc_by_model.png"), dpi=300, bbox_inches="tight")
            plt.close()

            # 16a: Same as plot 16, with RMIA-online included when available.
            series_16a = [
                ("RMIA", rmia_summary_df, "attack_auc", self.RMIA_COLOR),
                ("RMIA-online", rmia_online_summary_df, "attack_auc", "#0fa8c4"),
                ("AMIA", summary_df, "row_max_auc", self.AMIA_COLOR),
            ]
            series_16a = [
                (label, plot_df, metric, color)
                for label, plot_df, metric, color in series_16a
                if not plot_df.empty and metric in plot_df.columns and plot_df[metric].notna().any()
            ]
            if series_16a:
                csv_rows = []
                for label, plot_df, metric, _ in series_16a:
                    for model in amia_models:
                        vals = pd.to_numeric(plot_df[plot_df["model"] == model][metric], errors="coerce").dropna()
                        if vals.empty:
                            continue
                        csv_rows.append({
                            "attack": label,
                            "model": model,
                            "model_label": self._fmt_model(model),
                            "mean_auc_across_datasets": float(vals.mean()),
                            "std_auc_across_datasets": float(vals.std()),
                            "num_datasets": int(vals.count()),
                        })
                if csv_rows:
                    pd.DataFrame(csv_rows).to_csv(
                        os.path.join(save_dir, "16a_amia_rmia_online_auc_by_model.csv"),
                        index=False,
                    )

                x = np.arange(len(amia_models)) * 0.54
                width = min(0.10, 0.34 / max(1, len(series_16a)))
                offsets = np.linspace(
                    -(len(series_16a) - 1) * width / 2,
                    (len(series_16a) - 1) * width / 2,
                    len(series_16a),
                )
                fig, ax = plt.subplots(figsize=(7.8, 3.8))
                for i, (label, plot_df, metric, color) in enumerate(series_16a):
                    means = []
                    stds = []
                    for model in amia_models:
                        vals = pd.to_numeric(plot_df[plot_df["model"] == model][metric], errors="coerce")
                        means.append(vals.mean())
                        stds.append(vals.std())
                    stds = np.asarray(stds, dtype=float)
                    yerr = None if np.isnan(stds).all() else np.nan_to_num(stds, nan=0.0)
                    ax.bar(
                        x + offsets[i], means, width, yerr=yerr, capsize=4,
                        label=label, color=color, alpha=0.85, edgecolor="black",
                    )
                ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels([self._fmt_model(m) for m in amia_models], rotation=30, ha="right", fontsize=10)
                ax.set_ylabel("MIA AUC", fontsize=12, fontweight="bold")
                ax.set_ylim(0, 1.05)
                handles, labels = ax.get_legend_handles_labels()
                handles.append(Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Random guess"))
                ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=len(series_16a) + 1, frameon=True, fontsize=10)
                ax.tick_params(axis="y", labelsize=10)
                ax.grid(True, axis="y", alpha=0.35)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "16a_amia_rmia_online_auc_by_model.png"), dpi=300, bbox_inches="tight")
                plt.close()

        # 17: Heatmap of seed-mean row_max AMIA AUC.
        if "row_max_auc" in summary_df.columns and summary_df["row_max_auc"].notna().any():
            pivot = summary_df.pivot_table(values="row_max_auc", index="dataset", columns="model", aggfunc="first")
            ordered_cols = [m for m in self.model_plot_order if m in pivot.columns]
            ordered_cols += [m for m in pivot.columns if m not in ordered_cols]
            pivot = pivot[ordered_cols]
            pivot.index = pivot.index.map(self._fmt_dataset)
            pivot.columns = pivot.columns.map(self._fmt_model)
            fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(pivot) + 2)))
            sns.heatmap(
                pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0.5,
                vmin=0, vmax=1, linewidths=0.5, cbar_kws={"label": "AMIA row_max MIA AUC"}, ax=ax,
            )
            ax.set_title("AMIA row_max AUC heatmap (seed mean)", fontsize=14, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "17_amia_row_max_heatmap.png"), dpi=300, bbox_inches="tight")
            plt.close()

            # 18: Per-dataset RMIA vs AMIA row_max AUC with seed std on each bar.
            rmia_plot_df = rmia_summary_df[rmia_summary_df["model"].isin(foundation_models)].copy()
            datasets = self._sort_datasets(list(set(summary_df["dataset"].dropna()) | set(rmia_plot_df["dataset"].dropna())))
            ncols = min(3, max(1, len(datasets)))
            nrows = math.ceil(len(datasets) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.9 * nrows), squeeze=False)
            axes_flat = axes.flatten()
            width = 0.36
            for idx, dataset in enumerate(datasets):
                ax = axes_flat[idx]
                amia_ds = summary_df[summary_df["dataset"] == dataset].copy()
                rmia_ds = rmia_plot_df[rmia_plot_df["dataset"] == dataset].copy()
                ds_models = self._sort_models(
                    sorted(set(amia_ds["model"].dropna()) | set(rmia_ds["model"].dropna()))
                )
                x_ds = np.arange(len(ds_models))

                amia_vals = []
                amia_errs = []
                rmia_vals = []
                rmia_errs = []
                for model in ds_models:
                    amia_row = amia_ds[amia_ds["model"] == model]
                    rmia_row = rmia_ds[rmia_ds["model"] == model]
                    amia_vals.append(amia_row["row_max_auc"].iloc[0] if not amia_row.empty else np.nan)
                    if not amia_row.empty and "row_max_auc_std" in amia_row.columns:
                        amia_errs.append(pd.to_numeric(amia_row["row_max_auc_std"].iloc[0], errors="coerce"))
                    else:
                        amia_errs.append(np.nan)
                    rmia_vals.append(rmia_row["attack_auc"].iloc[0] if not rmia_row.empty else np.nan)
                    if not rmia_row.empty and "attack_auc_std" in rmia_row.columns:
                        rmia_errs.append(pd.to_numeric(rmia_row["attack_auc_std"].iloc[0], errors="coerce"))
                    else:
                        rmia_errs.append(np.nan)

                amia_yerr = np.asarray(amia_errs, dtype=float)
                amia_yerr = None if np.isnan(amia_yerr).all() else np.nan_to_num(amia_yerr, nan=0.0)
                rmia_yerr = np.asarray(rmia_errs, dtype=float)
                rmia_yerr = None if np.isnan(rmia_yerr).all() else np.nan_to_num(rmia_yerr, nan=0.0)

                ax.bar(
                    x_ds - width / 2,
                    rmia_vals,
                    width,
                    yerr=rmia_yerr,
                    capsize=3,
                    color=self.RMIA_COLOR,
                    alpha=0.82,
                    edgecolor="black",
                    label="RMIA",
                )
                ax.bar(
                    x_ds + width / 2,
                    amia_vals,
                    width,
                    yerr=amia_yerr,
                    capsize=3,
                    color=self.AMIA_COLOR,
                    alpha=0.82,
                    edgecolor="black",
                    label="AMIA",
                )
                ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
                ax.set_xticks(x_ds)
                ax.set_xticklabels([self._fmt_model(m) for m in ds_models], rotation=35, ha="right", fontsize=8)
                ax.set_title(self._fmt_dataset(dataset), fontsize=10, fontweight="bold")
                ax.set_ylabel("MIA AUC", fontsize=9, fontweight="bold")
                ax.set_ylim(0, 1.05)
                ax.grid(True, axis="y", alpha=0.35)
            for i in range(len(datasets), len(axes_flat)):
                axes_flat[i].axis("off")
            fig.legend(
                handles=[
                    Patch(facecolor=self.RMIA_COLOR, edgecolor="black", alpha=0.82, label="RMIA"),
                    Patch(facecolor=self.AMIA_COLOR, edgecolor="black", alpha=0.82, label="AMIA"),
                    Line2D([0], [0], color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="Random guess"),
                ],
                loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True,
            )
            plt.tight_layout(pad=1.0)
            plt.savefig(os.path.join(save_dir, "18_amia_row_max_per_dataset.png"), dpi=300, bbox_inches="tight")
            plt.close()

        # 19: Compare AMIA row_max against RMIA AUC for the same dataset/model.
        rmia_df = self.load_seed_summary_means(attack="rmia")
        if not rmia_df.empty and "row_max_auc" in summary_df.columns:
            merged = summary_df.merge(
                rmia_df[["dataset", "model", "attack_auc"]].rename(columns={"attack_auc": "rmia_auc"}),
                on=["dataset", "model"], how="inner",
            )
            merged = merged[merged["row_max_auc"].notna() & merged["rmia_auc"].notna()].copy()
            if not merged.empty:
                merged.to_csv(os.path.join(save_dir, "19_amia_vs_rmia_auc.csv"), index=False)
                fig, ax = plt.subplots(figsize=(7, 6))
                for model in self._sort_models(merged["model"].unique().tolist()):
                    mdf = merged[merged["model"] == model]
                    ax.scatter(mdf["rmia_auc"], mdf["row_max_auc"], s=55, alpha=0.82, label=self._fmt_model(model), edgecolor="black", linewidth=0.4)
                ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0)
                ax.axvline(0.5, color="gray", linestyle=":", linewidth=1.0)
                ax.set_xlabel("RMIA AUC (seed mean)", fontweight="bold")
                ax.set_ylabel("AMIA row_max AUC (seed mean)", fontweight="bold")
                ax.set_xlim(0, 1.02)
                ax.set_ylim(0, 1.02)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc="best", frameon=True)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "19_amia_vs_rmia_auc.png"), dpi=300, bbox_inches="tight")
                plt.close()

                # 20: Paired AMIA-vs-RMIA significance table for foundation models.
                sig_df = merged[merged["model"].isin(foundation_models)].copy()
                sig_df = sig_df[sig_df["row_max_auc"].notna() & sig_df["rmia_auc"].notna()].copy()
                if not sig_df.empty:
                    sig_df["amia_minus_rmia_auc"] = sig_df["row_max_auc"] - sig_df["rmia_auc"]
                    sig_df = sig_df.sort_values(["model", "dataset"])
                    paired_cols = ["dataset", "model", "rmia_auc", "row_max_auc", "amia_minus_rmia_auc"]
                    sig_df[paired_cols].to_csv(os.path.join(save_dir, "20_amia_rmia_paired_differences.csv"), index=False)

                    diffs = sig_df["amia_minus_rmia_auc"].to_numpy(dtype=float)
                    summary = {
                        "comparison": "AMIA row_max - RMIA",
                        "scope": "foundation models",
                        "n_pairs": int(len(sig_df)),
                        "n_datasets": int(sig_df["dataset"].nunique()),
                        "n_models": int(sig_df["model"].nunique()),
                        "mean_rmia_auc": float(sig_df["rmia_auc"].mean()),
                        "mean_amia_row_max_auc": float(sig_df["row_max_auc"].mean()),
                        "mean_diff": float(np.mean(diffs)),
                        "median_diff": float(np.median(diffs)),
                        "std_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else np.nan,
                        "positive_pairs": int(np.sum(diffs > 0)),
                        "negative_pairs": int(np.sum(diffs < 0)),
                        "zero_pairs": int(np.sum(diffs == 0)),
                        "wilcoxon_alternative": "greater",
                        "wilcoxon_h1": "AMIA row_max AUC > RMIA AUC",
                    }
                    if len(diffs) >= 2 and np.any(diffs != 0):
                        w_one = wilcoxon(
                            sig_df["row_max_auc"], sig_df["rmia_auc"],
                            alternative="greater", zero_method="wilcox", mode="auto",
                        )
                        w_two = wilcoxon(
                            sig_df["row_max_auc"], sig_df["rmia_auc"],
                            alternative="two-sided", zero_method="wilcox", mode="auto",
                        )
                        rng = np.random.default_rng(12345)
                        boots = np.array([
                            rng.choice(diffs, size=len(diffs), replace=True).mean()
                            for _ in range(20000)
                        ])
                        ci_low, ci_high = np.quantile(boots, [0.025, 0.975])
                        summary.update({
                            "wilcoxon_statistic_one_sided": float(w_one.statistic),
                            "wilcoxon_p_one_sided": float(w_one.pvalue),
                            "wilcoxon_statistic_two_sided": float(w_two.statistic),
                            "wilcoxon_p_two_sided": float(w_two.pvalue),
                            "bootstrap_mean_diff_ci_low": float(ci_low),
                            "bootstrap_mean_diff_ci_high": float(ci_high),
                        })
                    else:
                        summary.update({
                            "wilcoxon_statistic_one_sided": np.nan,
                            "wilcoxon_p_one_sided": np.nan,
                            "wilcoxon_statistic_two_sided": np.nan,
                            "wilcoxon_p_two_sided": np.nan,
                            "bootstrap_mean_diff_ci_low": np.nan,
                            "bootstrap_mean_diff_ci_high": np.nan,
                        })
                    pd.DataFrame([summary]).to_csv(os.path.join(save_dir, "20_amia_rmia_significance_summary.csv"), index=False)

                    labels = [
                        f"{self._fmt_model(row.model)} / {self._fmt_dataset(row.dataset)}"
                        for row in sig_df.itertuples(index=False)
                    ]
                    colors = ["#2a9d8f" if v >= 0 else "#b23a48" for v in sig_df["amia_minus_rmia_auc"]]
                    fig_h = max(5, 0.32 * len(sig_df) + 1.8)
                    fig, ax = plt.subplots(figsize=(9, fig_h))
                    y = np.arange(len(sig_df))
                    ax.barh(y, sig_df["amia_minus_rmia_auc"], color=colors, alpha=0.86, edgecolor="black", linewidth=0.4)
                    ax.axvline(0, color="black", linewidth=1.0)
                    ax.set_yticks(y)
                    ax.set_yticklabels(labels, fontsize=8)
                    ax.set_xlabel("AMIA row_max AUC - RMIA AUC", fontweight="bold")
                    ax.set_title("Paired AMIA vs RMIA differences (foundation models)", fontsize=13, fontweight="bold")
                    ax.grid(True, axis="x", alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, "20_amia_rmia_paired_differences.png"), dpi=300, bbox_inches="tight")
                    plt.close()

    def create_purchases10_amia_rmia_comparison(self, save_dir: str = "results_visualizations/attacks_viz"):
        """Special purchases10 RMIA/AMIA comparison, even when purchases10 is globally excluded."""
        os.makedirs(save_dir, exist_ok=True)
        dataset = "purchases10"
        foundation_models = {"tabpfn", "real-tabpfn", "tabicl", "tabdpt"}
        rows = []
        base_path = Path(self.logs_base_dir) / dataset
        for model_dir in sorted(base_path.glob("*")):
            if not model_dir.is_dir() or model_dir.name not in foundation_models:
                continue
            summary_path = model_dir / "attack_result_seed_summary.csv"
            if not summary_path.exists():
                continue
            try:
                summary = pd.read_csv(summary_path)
            except Exception as exc:
                logger.warning("Could not read %s: %s", summary_path, exc)
                continue
            if "attack" not in summary.columns:
                summary["attack"] = "rmia"
            by_attack_metric = summary.set_index(["attack", "metric"])

            def _metric(attack: str, metric: str, col: str):
                key = (attack, metric)
                if key not in by_attack_metric.index or col not in by_attack_metric.columns:
                    return np.nan
                return pd.to_numeric(by_attack_metric.loc[key, col], errors="coerce")

            rows.append({
                "dataset": dataset,
                "model": model_dir.name,
                "rmia_auc": _metric("rmia", "auc", "mean"),
                "rmia_auc_std": _metric("rmia", "auc", "std"),
                "rmia_num_seeds": _metric("rmia", "auc", "num_seeds"),
                "amia_row_max_auc": _metric("amia", "row_max_auc", "mean"),
                "amia_row_max_auc_std": _metric("amia", "row_max_auc", "std"),
                "amia_num_seeds": _metric("amia", "row_max_auc", "num_seeds"),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("No purchases10 RMIA/AMIA rows found.")
            return
        df["_order"] = df["model"].map(lambda m: self._model_order_map.get(m, len(self.model_plot_order)))
        df = df.sort_values(["_order", "model"]).drop(columns="_order")
        out_csv = os.path.join(save_dir, "18b_purchases10_amia_rmia_auc.csv")
        df.to_csv(out_csv, index=False)

        x = np.arange(len(df))
        width = 0.36
        rmia_vals = pd.to_numeric(df["rmia_auc"], errors="coerce").to_numpy(dtype=float)
        rmia_errs = pd.to_numeric(df["rmia_auc_std"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        amia_vals = pd.to_numeric(df["amia_row_max_auc"], errors="coerce").to_numpy(dtype=float)
        amia_errs = pd.to_numeric(df["amia_row_max_auc_std"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.bar(x - width / 2, rmia_vals, width, yerr=rmia_errs, capsize=4,
               color=self.RMIA_COLOR, alpha=0.82, edgecolor="black", label="RMIA")
        ax.bar(x + width / 2, amia_vals, width, yerr=amia_errs, capsize=4,
               color=self.AMIA_COLOR, alpha=0.82, edgecolor="black", label="AMIA row max")
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([self._fmt_model(m) for m in df["model"]], rotation=25, ha="right")
        ax.set_ylabel("MIA AUC", fontsize=10, fontweight="bold")
        ax.set_title("purchases10: RMIA vs AMIA row_max", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.35)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "18b_purchases10_amia_rmia_auc.png"), dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _parse_runtime_from_log(log_path: Path) -> Optional[float]:
        """Parse the runtime value from an attack log.

        Logs may contain multiple runs concatenated. If every run block (delimited
        by 'Training N models') has its own terminal runtime line, the log is
        incremental (different seeds appended) and the first block's value is
        this seed's runtime. If only the last block is complete (a rerun after
        failure), the last value is the correct one.
        """
        try:
            content = log_path.read_text(errors="replace")
        except OSError:
            return None

        patterns = [
            r"Done in ([\d.]+) s",
            r"Total runtime: ([\d.]+) seconds",
            r"Population attack runtime: ([\d.]+) seconds",
            r"LOSS runtime: ([\d.]+) seconds",
            r"Script finished in ([\d.]+) seconds",
            r"Auditing the privacy risks of target model \d+ costs ([\d.]+) seconds",
        ]

        run_starts = [m.start() for m in re.finditer(r"Training \d+ models", content)]
        if len(run_starts) > 1:
            blocks = [content[run_starts[i]:run_starts[i + 1]] for i in range(len(run_starts) - 1)]
            blocks.append(content[run_starts[-1]:])
            all_complete = all(
                any(re.search(p, block) for p in patterns)
                for block in blocks
            )
            if all_complete:
                content = blocks[0]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                return float(matches[-1])
        return None

    @staticmethod
    def _normalize_attack_name(attack: str, is_online: bool = False) -> str:
        attack = {
            "attack_p": "population",
            "population": "population",
            "qmia": "quantile",
        }.get(attack, attack)
        if is_online and attack in {"rmia", "lira"}:
            return f"{attack}_online"
        return attack

    def _collect_attack_runtime_rows(self) -> pd.DataFrame:
        """Collect per-seed attack runtime rows from log_time_analysis/log_amia logs."""
        base_path = Path(self.logs_base_dir)
        rows = []

        for log_path in sorted(base_path.rglob("log_time_analysis.log")):
            rel_parts = log_path.relative_to(base_path).parts
            if len(rel_parts) < 5 or not re.fullmatch(r"seed\d+", rel_parts[2]):
                continue
            dataset, model, seed_part, attack_part = rel_parts[:4]
            report_part = rel_parts[4]
            rows.append({
                "dataset": dataset,
                "model": model,
                "seed": int(seed_part.replace("seed", "")),
                "attack": self._normalize_attack_name(attack_part, report_part == "report_online"),
                "runtime_s": self._parse_runtime_from_log(log_path),
                "runtime_source": str(log_path),
            })

        for log_path in sorted(base_path.rglob("log_amia.log")):
            rel_parts = log_path.relative_to(base_path).parts
            if len(rel_parts) < 5 or not re.fullmatch(r"seed\d+", rel_parts[2]):
                continue
            dataset, model, seed_part = rel_parts[:3]
            rows.append({
                "dataset": dataset,
                "model": model,
                "seed": int(seed_part.replace("seed", "")),
                "attack": "amia",
                "runtime_s": self._parse_runtime_from_log(log_path),
                "runtime_source": str(log_path),
            })

        return self._filter_excluded_df(pd.DataFrame(rows))

    def _collect_attack_auc_rows(self) -> pd.DataFrame:
        """Collect per-seed AUC rows from attack_result_seed_runs.csv files."""
        base_path = Path(self.logs_base_dir)
        rows = []

        for runs_path in sorted(base_path.glob("*/*/attack_result_seed_runs.csv")):
            dataset, model = runs_path.relative_to(base_path).parts[:2]
            try:
                runs = pd.read_csv(runs_path)
            except Exception as exc:
                logger.warning("Could not read %s: %s", runs_path, exc)
                continue
            if "attack" not in runs.columns:
                runs.insert(0, "attack", "rmia")
            if "seed" not in runs.columns:
                continue

            for _, row in runs.iterrows():
                attack = self._normalize_attack_name(str(row["attack"]))
                auc = np.nan
                auc_metric = "auc"
                if attack == "amia":
                    for candidate in ("row_max_auc", "rmia_score_auc", "row_ent_auc"):
                        if candidate in runs.columns and pd.notna(row.get(candidate)):
                            auc = float(row[candidate])
                            auc_metric = candidate
                            break
                elif "auc" in runs.columns and pd.notna(row.get("auc")):
                    auc = float(row["auc"])

                if pd.isna(auc):
                    continue
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "seed": int(row["seed"]),
                    "attack": attack,
                    "auc": auc,
                    "auc_metric": auc_metric,
                    "auc_source": str(runs_path),
                })

        return self._filter_excluded_df(pd.DataFrame(rows))


    def _collect_row_max_layer_auc(
        self,
        datasets: list[str],
        model: str,
        n_layers: int,
    ) -> pd.DataFrame:
        """Collect per-seed row-max AUC by encoder layer for an AMIA model."""
        rows = []
        base_path = Path(self.logs_base_dir)
        for dataset in datasets:
            model_path = base_path / dataset / model
            for signals_path in sorted(model_path.glob("seed*/amia/signals/attn_signals_0.npz")):
                seed_part = signals_path.relative_to(model_path).parts[0]
                if not re.fullmatch(r"seed\d+", seed_part):
                    continue
                seed = int(seed_part.replace("seed", ""))
                summary_path = model_path / seed_part / "amia" / "report" / "exp" / "attention_summary.csv"
                if not summary_path.exists():
                    logger.warning("Missing attention_summary.csv for %s", signals_path)
                    continue
                try:
                    signals = np.load(signals_path)
                    row_max_all = np.asarray(signals["row_max_all"], dtype=float)
                    summary = pd.read_csv(summary_path)
                except Exception as exc:
                    logger.warning("Could not load row-max layer data for %s: %s", signals_path, exc)
                    continue
                if "member" not in summary.columns or row_max_all.ndim != 3:
                    continue
                n_samples = min(row_max_all.shape[0], len(summary))
                if n_samples == 0:
                    continue
                y_true = pd.to_numeric(summary["member"].iloc[:n_samples], errors="coerce").fillna(0).astype(int).to_numpy()
                row_max_all = row_max_all[:n_samples]

                if row_max_all.shape[1] == n_layers:
                    calls_per_layer = 1
                    layer_scores = row_max_all.mean(axis=2)  # average heads
                elif row_max_all.shape[1] % n_layers == 0:
                    calls_per_layer = row_max_all.shape[1] // n_layers
                    grouped = row_max_all.reshape(n_samples, n_layers, calls_per_layer, row_max_all.shape[2])
                    layer_scores = grouped.mean(axis=(2, 3))  # average calls and heads
                else:
                    logger.warning(
                        "Skipping %s: row_max_all calls=%s cannot map to %s layers",
                        signals_path,
                        row_max_all.shape[1],
                        n_layers,
                    )
                    continue

                for layer_idx in range(n_layers):
                    _, _, auc_val = self._roc_from_scores(y_true, layer_scores[:, layer_idx])
                    rows.append({
                        "dataset": dataset,
                        "model": model,
                        "seed": seed,
                        "layer": layer_idx,
                        "row_max_auc": auc_val,
                        "n_samples": n_samples,
                        "n_members": int(np.sum(y_true == 1)),
                        "n_nonmembers": int(np.sum(y_true == 0)),
                        "calls_per_layer": calls_per_layer,
                        "n_heads": row_max_all.shape[2],
                        "source": str(signals_path),
                    })
        return pd.DataFrame(rows)

    def _plot_row_max_layer_auc_for_model(
        self,
        model: str,
        n_layers: int,
        save_dir: str,
        datasets: list[str],
        dataset_labels: dict[str, str],
        dataset_colors: dict[str, str],
    ) -> None:
        runs = self._collect_row_max_layer_auc(datasets=datasets, model=model, n_layers=n_layers)
        if runs.empty:
            logger.warning("No row-max layer AUC rows found for %s.", model)
            return

        prefix = f"28_{model}_row_max_layer_auc"
        runs.to_csv(os.path.join(save_dir, f"{prefix}_runs.csv"), index=False)
        summary = (
            runs.groupby(["dataset", "model", "layer"], as_index=False)
            .agg(
                mean_auc=("row_max_auc", "mean"),
                std_auc=("row_max_auc", lambda s: float(np.nanstd(pd.to_numeric(s, errors="coerce"), ddof=1)) if pd.to_numeric(s, errors="coerce").notna().sum() > 1 else 0.0),
                n_seeds=("row_max_auc", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
                calls_per_layer=("calls_per_layer", "first"),
                n_heads=("n_heads", "first"),
            )
        )
        summary.to_csv(os.path.join(save_dir, f"{prefix}_summary.csv"), index=False)

        fig, ax = plt.subplots(figsize=(8.6, 2.8), constrained_layout=True)
        for dataset in datasets:
            ds = summary[summary["dataset"] == dataset].sort_values("layer")
            if ds.empty:
                continue
            x = ds["layer"].to_numpy(dtype=int)
            y = ds["mean_auc"].to_numpy(dtype=float)
            std = ds["std_auc"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.2,
                markersize=4.8,
                color=dataset_colors[dataset],
                label=dataset_labels[dataset],
            )
            ax.fill_between(
                x,
                np.maximum(0, y - std),
                np.minimum(1, y + std),
                color=dataset_colors[dataset],
                alpha=0.16,
                linewidth=0,
            )
        ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1.4, alpha=0.9)
        ax.set_xlabel("Encoder layer", fontsize=14, fontweight="bold")
        ax.set_ylabel("AMIA AUC", fontsize=14, fontweight="bold")
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_ylim(0.45, 1.0)
        ax.set_xticks(np.arange(n_layers))
        ax.set_xticklabels([f"L{i}" for i in range(n_layers)], rotation=45, ha="right")
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower left", frameon=True, fontsize=12)
        out = os.path.join(save_dir, f"{prefix}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s row-max layer AUC plot to %s", model, out)

    def create_row_max_layer_study(self, save_dir: str = "results_visualizations/attacks_viz"):
        """Layer-wise AMIA row-max AUC across seeds for locations and dropout_success."""
        os.makedirs(save_dir, exist_ok=True)
        datasets = ["locations", "dropout_success"]
        dataset_labels = {"locations": "Locations", "dropout_success": "Dropout Success"}
        dataset_colors = {"locations": "#4e79a7", "dropout_success": "#f28e2b"}
        model_layers = {
            "tabpfn": 24,
            "tabicl": 12,
            "tabdpt": 16,
        }
        for model, n_layers in model_layers.items():
            self._plot_row_max_layer_auc_for_model(
                model=model,
                n_layers=n_layers,
                save_dir=save_dir,
                datasets=datasets,
                dataset_labels=dataset_labels,
                dataset_colors=dataset_colors,
            )

    def create_amia_attack_time(self, save_dir: str = "results_visualizations/attacks_viz"):
        """Create an AMIA/RMIA AUC and attack-time dashboard for tabular FMs."""
        os.makedirs(save_dir, exist_ok=True)

        runtime_df = self._collect_attack_runtime_rows()
        if runtime_df.empty:
            logger.warning("No runtime rows found for attack-time plot.")
            return

        foundation_models = {"tabpfn", "real-tabpfn", "tabicl", "tabdpt"}
        attack_order = ["amia", "rmia"]
        attack_labels = {"amia": "AMIA", "rmia": "RMIA"}
        attack_colors = {"amia": self.AMIA_COLOR, "rmia": self.RMIA_COLOR}

        df = runtime_df[
            runtime_df["model"].isin(foundation_models)
            & runtime_df["attack"].isin(attack_order)
            & runtime_df["runtime_s"].notna()
        ].copy()
        if df.empty:
            logger.warning("No AMIA/RMIA runtime rows found for tabular foundation models.")
            return

        auc_df = self._collect_attack_auc_rows()
        if not auc_df.empty:
            auc_df = auc_df[
                auc_df["model"].isin(foundation_models)
                & auc_df["attack"].isin(attack_order)
                & auc_df["auc"].notna()
            ].copy()
            df = pd.merge(
                df,
                auc_df[["dataset", "model", "seed", "attack", "auc", "auc_metric"]],
                on=["dataset", "model", "seed", "attack"],
                how="left",
            )
        else:
            df["auc"] = np.nan
            df["auc_metric"] = np.nan

        dataset_summary = (
            df.groupby(["dataset", "model", "attack"], as_index=False)
            .agg(
                mean_runtime_s=("runtime_s", "mean"),
                mean_auc=("auc", "mean"),
                n_seed_rows=("runtime_s", "size"),
                n_auc_seed_rows=("auc", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            )
        )
        summary = (
            dataset_summary.groupby(["model", "attack"], as_index=False)
            .agg(
                mean_runtime_s=("mean_runtime_s", "mean"),
                std_runtime_s=("mean_runtime_s", lambda s: float(np.nanstd(s, ddof=1)) if s.notna().sum() > 1 else 0.0),
                median_runtime_s=("mean_runtime_s", "median"),
                mean_auc=("mean_auc", "mean"),
                std_auc=("mean_auc", lambda s: float(np.nanstd(s, ddof=1)) if s.notna().sum() > 1 else 0.0),
                n_runs=("n_seed_rows", "sum"),
                n_auc_runs=("n_auc_seed_rows", "sum"),
                n_datasets=("dataset", "nunique"),
            )
        )
        fm_order = ["tabdpt", "tabicl", "real-tabpfn", "tabpfn"]
        ordered_models = [m for m in fm_order if m in summary["model"].unique()]
        ordered_models += sorted(m for m in summary["model"].unique() if m not in ordered_models)
        present_attacks = [a for a in attack_order if a in set(summary["attack"])]
        summary["_model_order"] = summary["model"].map({m: i for i, m in enumerate(ordered_models)})
        summary["_attack_order"] = summary["attack"].map({a: i for i, a in enumerate(present_attacks)})
        summary = summary.sort_values(["_model_order", "_attack_order"]).drop(columns=["_model_order", "_attack_order"])
        summary["attack_label"] = summary["attack"].map(attack_labels)

        df["attack_label"] = df["attack"].map(attack_labels)
        dataset_summary["attack_label"] = dataset_summary["attack"].map(attack_labels)
        df.to_csv(os.path.join(save_dir, "21_attack_time_runs.csv"), index=False)
        dataset_summary.to_csv(os.path.join(save_dir, "21_attack_time_dataset_means.csv"), index=False)
        summary.to_csv(os.path.join(save_dir, "21_attack_time_summary.csv"), index=False)

        out = os.path.join(save_dir, "21_attack_time.png")
        with plt.rc_context({"axes.grid": False}):
            fig = plt.figure(figsize=(16, 6.5))
            gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.15)
            y_base = np.arange(len(ordered_models))
            bar_h = 0.34 if len(present_attacks) > 1 else 0.55
            offsets = np.linspace(-(len(present_attacks) - 1) * bar_h / 2, (len(present_attacks) - 1) * bar_h / 2, len(present_attacks))

            # 1. Attack AUC by model and attack.
            ax1 = fig.add_subplot(gs[0, 0])
            for attack, offset in zip(present_attacks, offsets):
                attack_summary = summary[summary["attack"] == attack].set_index("model")
                means = []
                stds = []
                for model in ordered_models:
                    if model not in attack_summary.index:
                        means.append(np.nan)
                        stds.append(np.nan)
                        continue
                    row = attack_summary.loc[model]
                    means.append(float(row["mean_auc"]) if not pd.isna(row["mean_auc"]) else np.nan)
                    stds.append(float(row["std_auc"]) if not pd.isna(row["std_auc"]) else 0.0)
                means_arr = np.asarray(means, dtype=float)
                stds_arr = np.nan_to_num(np.asarray(stds, dtype=float), nan=0.0)
                ax1.barh(
                    y_base + offset, means_arr, height=bar_h * 0.9, xerr=stds_arr,
                    alpha=0.78, color=attack_colors[attack], edgecolor="black", linewidth=0.4,
                    capsize=4, label=attack_labels[attack],
                )
            ax1.axvline(0.5, color="red", linestyle="--", linewidth=1.2, alpha=0.75)
            ax1.set_yticks(y_base)
            ax1.set_yticklabels([self._fmt_model(m) for m in ordered_models], fontsize=15, rotation=30, ha="right")
            ax1.set_xlabel("MIA AUC", fontsize=15, fontweight="bold")
            ax1.tick_params(axis="x", labelsize=14)
            ax1.set_xlim(0, 1.05)
            ax1.xaxis.grid(True, alpha=0.4)
            ax1.yaxis.grid(True, color=".85", linewidth=0.8)

            # 2. Runtime distribution by model and attack.
            ax2 = fig.add_subplot(gs[0, 1])
            runtime_minutes = dataset_summary.copy()
            runtime_minutes["runtime_min"] = pd.to_numeric(runtime_minutes["mean_runtime_s"], errors="coerce") / 60.0
            # RMIA ran on 2 GPUs, AMIA on 1 — normalize to GPU·min for a fair comparison
            runtime_minutes.loc[runtime_minutes["attack"] == "rmia", "runtime_min"] *= 2
            box_data = []
            positions = []
            box_attacks = []
            for attack, offset in zip(present_attacks, offsets):
                for model_idx, model in enumerate(ordered_models):
                    vals = runtime_minutes[
                        (runtime_minutes["attack"] == attack)
                        & (runtime_minutes["model"] == model)
                    ]["runtime_min"].dropna().to_numpy(dtype=float)
                    if len(vals) == 0:
                        continue
                    box_data.append(vals)
                    positions.append(y_base[model_idx] + offset)
                    box_attacks.append(attack)

            if box_data:
                bp = ax2.boxplot(
                    box_data,
                    positions=positions,
                    vert=False,
                    widths=bar_h * 0.78,
                    patch_artist=True,
                    showmeans=False,
                    boxprops=dict(linewidth=0.9),
                    medianprops=dict(color="black", linewidth=1.4),
                    whiskerprops=dict(linewidth=0.9),
                    capprops=dict(linewidth=0.9),
                    flierprops=dict(marker="o", markersize=6, alpha=0.85, markeredgewidth=0.8, markeredgecolor="black"),
                )
                for patch, attack in zip(bp["boxes"], box_attacks):
                    patch.set_facecolor(attack_colors[attack])
                    patch.set_alpha(0.72)
                    patch.set_edgecolor("black")

            ax2.set_yticks([])
            ax2.set_xlabel("Runtime (min)", fontsize=15, fontweight="bold")
            ax2.tick_params(axis="x", labelsize=14)
            ax2.set_xscale("log")
            ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax2.xaxis.grid(True, which="both", alpha=0.4)
            legend_order = [a for a in ["rmia", "amia"] if a in present_attacks]
            legend_handles = [
                Patch(facecolor=attack_colors[a], edgecolor="black", alpha=0.72, label=attack_labels[a])
                for a in legend_order
            ]
            ax2.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=13)
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()
        logger.info("Saved AMIA/RMIA AUC and attack-time dashboard to %s", out)

    def create_attack_time_auc_comparison(self, save_dir: str = "results_visualizations/attacks_viz"):
        """Backward-compatible alias for plot 21."""
        return self.create_amia_attack_time(save_dir)


    def create_highrisk_selection_rows(self, save_dir: str = "results_visualizations/attacks_viz"):
        """Plot 31: attack-side high-risk selection from AMIA row_max.

        The high-risk rule is the same conservative calibration used by the
        guardrail: threshold = minimum member row_max in the baseline AMIA
        attention_summary.csv.  The figure shows member/non-member AMIA score
        distributions for locations and dropout_success, plus the selected-row
        membership composition.
        """
        from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

        os.makedirs(save_dir, exist_ok=True)
        datasets = ["locations", "dropout_success"]
        dataset_labels = {"locations": "Locations", "dropout_success": "Dropout Success"}
        models = ["tabpfn", "real-tabpfn", "tabicl", "tabdpt"]
        model_labels = {
            "tabpfn": "TabPFN",
            "real-tabpfn": "Real-TabPFN",
            "tabicl": "TabICL",
            "tabdpt": "TabDPT",
        }
        model_colors = {
            "tabpfn": "#0072b2",
            "real-tabpfn": "#cc79a7",
            "tabicl": "#009e73",
            "tabdpt": "#d55e00",
        }
        mem_color = "#7a5195"
        non_color = "#2a9d8f"
        threshold_color = "#4d4d4d"

        rows = []
        summary_rows = []
        base_path = Path(self.logs_base_dir)
        for dataset in datasets:
            for model in models:
                model_root = base_path / dataset / model
                for summary_path in sorted(model_root.glob("seed*/amia/report/exp/attention_summary.csv")):
                    seed_part = summary_path.relative_to(model_root).parts[0]
                    if not re.fullmatch(r"seed\d+", seed_part):
                        continue
                    try:
                        df = pd.read_csv(summary_path)
                    except Exception as exc:
                        logger.warning("Could not read %s: %s", summary_path, exc)
                        continue
                    if "member" not in df.columns or "row_max" not in df.columns:
                        continue
                    member = pd.to_numeric(df["member"], errors="coerce").fillna(0).astype(int).astype(bool)
                    row_max = pd.to_numeric(df["row_max"], errors="coerce")
                    valid = row_max.notna()
                    member = member[valid].to_numpy(bool)
                    row_max = row_max[valid].to_numpy(float)
                    if len(row_max) == 0 or member.sum() == 0:
                        continue
                    threshold = float(np.min(row_max[member]))
                    selected = row_max >= threshold
                    summary_rows.append({
                        "dataset": dataset,
                        "model": model,
                        "model_label": model_labels[model],
                        "threshold": threshold,
                        "selected_members": int((selected & member).sum()),
                        "selected_nonmembers": int((selected & ~member).sum()),
                        "selected_rate": float(selected.mean()),
                        "selected_member_share": float((selected & member).sum() / selected.sum()),
                    })
                    for value, is_member in zip(row_max, member):
                        rows.append({
                            "dataset": dataset,
                            "model": model,
                            "model_label": model_labels[model],
                            "row_max": float(value),
                            "member": "Member" if is_member else "Non-member",
                        })

        points = pd.DataFrame(rows)
        summary = pd.DataFrame(summary_rows)
        if points.empty or summary.empty:
            logger.warning("No AMIA baseline attention summaries found for high-risk selection plot.")
            return

        points.to_csv(os.path.join(save_dir, "31_fm_highrisk_selection_rows_points.csv"), index=False)
        summary.to_csv(os.path.join(save_dir, "31_fm_highrisk_selection_rows_summary.csv"), index=False)
        agg = (
            summary.groupby(["dataset", "model", "model_label"], as_index=False)
            .agg(
                selected_members=("selected_members", "sum"),
                selected_nonmembers=("selected_nonmembers", "sum"),
                selected_rate=("selected_rate", "mean"),
                selected_member_share=("selected_member_share", "mean"),
                threshold_mean=("threshold", "mean"),
            )
        )
        agg["selected_total"] = agg["selected_members"] + agg["selected_nonmembers"]
        agg["model"] = pd.Categorical(agg["model"], categories=models, ordered=True)
        agg = agg.sort_values(["dataset", "model"])
        agg.to_csv(os.path.join(save_dir, "31_fm_highrisk_selection_rows_by_model.csv"), index=False)

        def hist_density(values, bins):
            counts, edges = np.histogram(values, bins=bins, density=True)
            return (edges[:-1] + edges[1:]) / 2, counts

        def padded_limits(values, pad_frac=0.24):
            lo, hi = np.nanquantile(values, [0.002, 0.998])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
            return lo, hi + max(hi - lo, 1e-9) * pad_frac

        def draw_density(ax, panel, base_y, height, downward=False, bins=None):
            densities = []
            for label in ["Non-member", "Member"]:
                subset = panel[panel["member"] == label]["row_max"].to_numpy(float)
                x, y = hist_density(subset, bins)
                densities.append((label, x, y))
            max_y = max(float(np.nanmax(y)) for _, _, y in densities if len(y))
            max_y = max(max_y, 1e-12)
            for label, x, y in densities:
                color = mem_color if label == "Member" else non_color
                alpha = 0.42 if label == "Member" else 0.34
                scaled = y / max_y * height
                yy = base_y - scaled if downward else base_y + scaled
                ax.fill_between(x, base_y, yy, step="mid", color=color, alpha=alpha, linewidth=0)
                ax.plot(x, yy, color=color, linewidth=1.1, alpha=0.92)

        def annotate_stats(ax, dataset, model, y, va="top"):
            row = agg[(agg["dataset"] == dataset) & (agg["model"] == model)].iloc[0]
            ax.text(
                0.985, y,
                f"{model_labels[model]}\nthr={row.threshold_mean:.4g}\nsel={row.selected_rate:.0%}",
                transform=ax.transAxes,
                ha="right",
                va=va,
                fontsize=12,
                color="black",
                bbox=dict(boxstyle="round,pad=0.26", facecolor="white", edgecolor="#cccccc", alpha=0.94),
            )

        def colored_split_title(ax):
            ax.set_title("TabPFN / Real-TabPFN", color="black", fontweight="bold")

        def model_panel(ax, dataset, model, show_xlabel):
            panel = points[(points["dataset"] == dataset) & (points["model"] == model)]
            row = agg[(agg["dataset"] == dataset) & (agg["model"] == model)].iloc[0]
            values = panel["row_max"].to_numpy(float)
            lo, hi = padded_limits(values)
            bins = np.linspace(lo, hi - (hi - lo) * 0.24 / 1.24, 70)
            draw_density(ax, panel, 0, 0.84, False, bins)
            ax.axvline(float(row["threshold_mean"]), color=threshold_color, linestyle="--", linewidth=1.65)
            ax.set_xlim(lo, hi)
            ax.set_ylim(0, 1.0)
            ax.set_title(model_labels[model], color="black", fontweight="bold")
            ax.set_xlabel("AMIA" if show_xlabel else "")
            ax.set_yticks([])
            ax.grid(True, axis="x", alpha=0.14)
            annotate_stats(ax, dataset, model, 0.92, "top")

        def split_tabpfn_panel(ax, dataset, show_xlabel):
            p_tab = points[(points["dataset"] == dataset) & (points["model"] == "tabpfn")]
            p_real = points[(points["dataset"] == dataset) & (points["model"] == "real-tabpfn")]
            values = pd.concat([p_tab["row_max"], p_real["row_max"]]).to_numpy(float)
            lo, hi = padded_limits(values, 0.30)
            bins = np.linspace(lo, hi - (hi - lo) * 0.30 / 1.30, 70)
            draw_density(ax, p_real, 1.0, 0.40, True, bins)
            draw_density(ax, p_tab, 0.0, 0.40, False, bins)
            for model in ["tabpfn", "real-tabpfn"]:
                threshold = float(agg[(agg["dataset"] == dataset) & (agg["model"] == model)]["threshold_mean"].iloc[0])
                ax.axvline(threshold, color=model_colors[model], linestyle="--", linewidth=1.65)
            ax.axhline(0.5, color="#bbbbbb", linewidth=0.8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel("AMIA" if show_xlabel else "")
            ax.grid(True, axis="x", alpha=0.14)
            colored_split_title(ax)
            annotate_stats(ax, dataset, "real-tabpfn", 0.94, "top")
            annotate_stats(ax, dataset, "tabpfn", 0.06, "bottom")

        def caught_panel(ax, dataset):
            panel = agg[agg["dataset"] == dataset].copy()
            panel["model"] = pd.Categorical(panel["model"], categories=models, ordered=True)
            panel = panel.sort_values("model")
            x = np.arange(len(panel))
            selected_members = panel["selected_members"].to_numpy(float)
            selected_nonmembers = panel["selected_nonmembers"].to_numpy(float)
            totals = np.maximum(selected_members + selected_nonmembers, 1)
            member_share = selected_members / totals
            nonmember_share = selected_nonmembers / totals
            ax.bar(x, member_share, color=mem_color)
            ax.bar(x, nonmember_share, bottom=member_share, color=non_color)
            for i, row in enumerate(panel.itertuples(index=False)):
                member_pct = row.selected_member_share
                nonmember_pct = 1 - member_pct
                if member_share[i] >= 0.14:
                    ax.text(i, member_share[i] / 2, f"{member_pct:.0%} M", ha="center", va="center", fontsize=11, color="white", fontweight="bold")
                else:
                    ax.text(i, member_share[i] + 0.025, f"{member_pct:.0%} M", ha="center", va="bottom", fontsize=11, color=mem_color, fontweight="bold")
                if nonmember_share[i] >= 0.14:
                    ax.text(i, member_share[i] + nonmember_share[i] / 2, f"{nonmember_pct:.0%} NM", ha="center", va="center", fontsize=11, color="white", fontweight="bold")
                else:
                    ax.text(i, min(1.08, member_share[i] + nonmember_share[i] + 0.025), f"{nonmember_pct:.0%} NM", ha="center", va="bottom", fontsize=11, color=non_color, fontweight="bold")
            ax.set_ylim(0, 1.18)
            ax.set_xticks(x)
            if dataset == datasets[-1]:
                ax.set_xticklabels(panel["model_label"], rotation=22, ha="right")
                ax.tick_params(axis="x", length=3)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)
            ax.set_yticks([])
            ax.set_title("Selected High-Risk Samples", fontweight="bold")
            ax.grid(True, axis="y", alpha=0.14)

        with plt.rc_context({
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 13,
        }):
            fig, axes = plt.subplots(2, 4, figsize=(17.2, 8.3), constrained_layout=True)
            fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.04, wspace=0.025, hspace=0.05)
            for row_idx, dataset in enumerate(datasets):
                show_xlabel = row_idx == 1
                split_tabpfn_panel(axes[row_idx, 0], dataset, show_xlabel)
                model_panel(axes[row_idx, 1], dataset, "tabicl", show_xlabel)
                model_panel(axes[row_idx, 2], dataset, "tabdpt", show_xlabel)
                caught_panel(axes[row_idx, 3], dataset)
                axes[row_idx, 0].text(
                    -0.105, 0.5, dataset_labels[dataset],
                    transform=axes[row_idx, 0].transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=17,
                    fontweight="bold",
                )
            legend_handles = [
                Line2D([0], [0], color=mem_color, lw=5, alpha=0.65, label="Member"),
                Line2D([0], [0], color=non_color, lw=5, alpha=0.65, label="Non-member"),
                Line2D([0], [0], color=threshold_color, lw=1.65, linestyle="--", label="Threshold"),
            ]
            fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.075))
            out = os.path.join(save_dir, "31_fm_highrisk_selection_rows.png")
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
        logger.info("Saved high-risk selection rows plot to %s", out)


    def _collect_context_rmia_amia_evolution(self, dataset: str, models: list[str]) -> pd.DataFrame:
        """Collect context-size RMIA and AMIA AUC rows for Plot 29."""
        rows = []
        base_path = Path(self.logs_base_dir) / dataset
        ctx_re = re.compile(r"^(rmia|amia)_ctx(\d+(?:\.\d+)?)$")

        for model in models:
            model_root = base_path / model
            if not model_root.is_dir():
                continue
            for seed_dir in sorted(model_root.glob("seed*")):
                if not seed_dir.is_dir() or not re.fullmatch(r"seed\d+", seed_dir.name):
                    continue
                seed = int(seed_dir.name.replace("seed", ""))
                for run_dir in sorted(seed_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    match = ctx_re.fullmatch(run_dir.name)
                    if match:
                        attack_kind = match.group(1)
                        context_pct = float(match.group(2))
                    elif run_dir.name in {"rmia", "amia"}:
                        attack_kind = run_dir.name
                        context_pct = 100.0
                    else:
                        continue

                    if attack_kind == "rmia":
                        result_files = sorted((run_dir / "report" / "exp").glob("attack_result_*.npz"))
                        if not result_files:
                            continue
                        result_path = result_files[0]
                        try:
                            auc_val = float(np.load(result_path, allow_pickle=True)["auc"])
                        except Exception as exc:
                            logger.warning("Could not read RMIA context result %s: %s", result_path, exc)
                            continue
                        rows.append({
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "context_pct": context_pct,
                            "attack": "RMIA",
                            "auc": auc_val,
                            "source": str(result_path),
                        })
                    else:
                        summary_path = run_dir / "report" / "exp" / "attention_summary.csv"
                        if not summary_path.exists():
                            continue
                        try:
                            summary = pd.read_csv(summary_path)
                        except Exception as exc:
                            logger.warning("Could not read AMIA context result %s: %s", summary_path, exc)
                            continue
                        if "member" not in summary.columns or "row_max" not in summary.columns:
                            continue
                        y_true = pd.to_numeric(summary["member"], errors="coerce").fillna(0).astype(int).to_numpy()
                        scores = pd.to_numeric(summary["row_max"], errors="coerce").to_numpy(dtype=float)
                        _, _, auc_val = self._roc_from_scores(y_true, scores)
                        rows.append({
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "context_pct": context_pct,
                            "attack": "AMIA row_max",
                            "auc": auc_val,
                            "source": str(summary_path),
                        })
        return pd.DataFrame(rows)

    def create_context_rmia_amia_evolution(self, save_dir: str = "results_visualizations/attacks_viz"):
        """Plot 29: RMIA/AMIA AUC evolution over context size for tabular FMs."""
        os.makedirs(save_dir, exist_ok=True)
        datasets = ["locations", "dropout_success"]
        dataset_labels = {"locations": "Locations", "dropout_success": "Dropout Success"}
        models = ["tabpfn", "real-tabpfn", "tabicl", "tabdpt"]
        model_labels = {
            "tabpfn": "TabPFN",
            "real-tabpfn": "Real-TabPFN",
            "tabicl": "TabICL",
            "tabdpt": "TabDPT",
        }
        attack_styles = {
            "RMIA": {"color": self.RMIA_COLOR, "marker": "o", "linestyle": "-"},
            "AMIA row_max": {"color": self.AMIA_COLOR, "marker": "s", "linestyle": "--"},
        }

        combined_points = []
        combined_summaries = []

        for dataset in datasets:
            points = self._collect_context_rmia_amia_evolution(dataset, models)
            if points.empty:
                logger.warning("No context RMIA/AMIA rows found for %s.", dataset)
                continue
            points["context_pct"] = pd.to_numeric(points["context_pct"], errors="coerce")
            points["auc"] = pd.to_numeric(points["auc"], errors="coerce")
            points = points.dropna(subset=["context_pct", "auc"])
            points["context_pct"] = points["context_pct"].round(6)
            points = points.sort_values(["model", "attack", "context_pct", "seed"])

            prefix = f"29_{dataset}_context_rmia_amia_evolution"
            points.to_csv(os.path.join(save_dir, f"{prefix}_points.csv"), index=False)
            summary = (
                points.groupby(["model", "context_pct", "attack"], as_index=False)
                .agg(
                    auc_mean=("auc", "mean"),
                    auc_std=("auc", lambda s: float(np.nanstd(pd.to_numeric(s, errors="coerce"), ddof=1)) if pd.to_numeric(s, errors="coerce").notna().sum() > 1 else 0.0),
                    num_seeds=("seed", "nunique"),
                )
            )
            summary["dataset"] = dataset
            summary["model"] = pd.Categorical(summary["model"], categories=models, ordered=True)
            summary = summary.sort_values(["model", "context_pct", "attack"])
            combined_points.append(points.copy())
            combined_summaries.append(summary.copy())
            summary.to_csv(os.path.join(save_dir, f"{prefix}_summary.csv"), index=False)

            present_models = [m for m in models if m in set(points["model"])]
            if not present_models:
                continue
            ncols = 2
            nrows = math.ceil(len(present_models) / ncols)
            with plt.rc_context({
                "font.size": 13,
                "axes.titlesize": 15,
                "axes.labelsize": 13,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 12,
            }):
                fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 4.0 * nrows), squeeze=False, sharex=True, sharey=True)
                for idx, model in enumerate(present_models):
                    ax = axes[idx // ncols][idx % ncols]
                    model_df = summary[summary["model"] == model]
                    for attack, style in attack_styles.items():
                        attack_df = model_df[model_df["attack"] == attack].sort_values("context_pct")
                        if attack_df.empty:
                            continue
                        x = attack_df["context_pct"].to_numpy(dtype=float)
                        y = attack_df["auc_mean"].to_numpy(dtype=float)
                        std = attack_df["auc_std"].fillna(0.0).to_numpy(dtype=float)
                        ax.plot(
                            x,
                            y,
                            marker=style["marker"],
                            linestyle=style["linestyle"],
                            linewidth=2.3,
                            markersize=5.8,
                            color=style["color"],
                            label=attack.replace(" row_max", ""),
                        )
                        if np.nanmax(std) > 0:
                            ax.fill_between(x, np.maximum(0, y - std), np.minimum(1, y + std), color=style["color"], alpha=0.14, linewidth=0)
                    ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1.1)
                    ax.set_title(model_labels.get(model, model), fontweight="bold")
                    ax.set_xlim(0, 105)
                    ax.set_ylim(0.45, 1.02)
                    ax.grid(True, alpha=0.25)
                    if idx // ncols == nrows - 1:
                        ax.set_xlabel("Context size (% of training pool)")
                    if idx % ncols == 0:
                        ax.set_ylabel("MIA AUC")
                    ax.set_xticks([5, 10, 20, 40, 60, 80, 100])
                    ax.set_xticklabels(["5", "10", "20", "40", "60", "80", "100"])

                for idx in range(len(present_models), nrows * ncols):
                    axes[idx // ncols][idx % ncols].set_visible(False)
                handles = [
                    Line2D([0], [0], color=style["color"], marker=style["marker"], linestyle=style["linestyle"], linewidth=2.3, markersize=5.8, label=attack.replace(" row_max", ""))
                    for attack, style in attack_styles.items()
                ]
                fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.02))
                fig.tight_layout(rect=(0, 0.04, 1, 1.0))
                out = os.path.join(save_dir, f"{prefix}.png")
                fig.savefig(out, dpi=300, bbox_inches="tight")
                plt.close(fig)
            logger.info("Saved context RMIA/AMIA evolution plot to %s", out)


        if not combined_summaries:
            logger.warning("No context RMIA/AMIA rows found for combined plot.")
            return

        all_points = pd.concat(combined_points, ignore_index=True, sort=False)
        all_summary = pd.concat(combined_summaries, ignore_index=True, sort=False)
        all_points.to_csv(os.path.join(save_dir, "29_context_rmia_amia_evolution_combined_points.csv"), index=False)
        all_summary.to_csv(os.path.join(save_dir, "29_context_rmia_amia_evolution_combined_summary.csv"), index=False)

        with plt.rc_context({
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 12,
        }):
            fig, axes = plt.subplots(
                len(datasets),
                len(models),
                figsize=(17.0, 7.2),
                squeeze=False,
                sharex=True,
                sharey=True,
            )
            for row_idx, dataset in enumerate(datasets):
                for col_idx, model in enumerate(models):
                    ax = axes[row_idx, col_idx]
                    model_df = all_summary[
                        (all_summary["dataset"] == dataset)
                        & (all_summary["model"].astype(str) == model)
                    ]
                    for attack, style in attack_styles.items():
                        attack_df = model_df[model_df["attack"] == attack].sort_values("context_pct")
                        if attack_df.empty:
                            continue
                        x = attack_df["context_pct"].to_numpy(dtype=float)
                        y = attack_df["auc_mean"].to_numpy(dtype=float)
                        std = attack_df["auc_std"].fillna(0.0).to_numpy(dtype=float)
                        ax.plot(
                            x,
                            y,
                            marker=style["marker"],
                            linestyle=style["linestyle"],
                            linewidth=2.1,
                            markersize=5.2,
                            color=style["color"],
                            label=attack.replace(" row_max", ""),
                        )
                        if len(std) and np.nanmax(std) > 0:
                            ax.fill_between(
                                x,
                                np.maximum(0, y - std),
                                np.minimum(1, y + std),
                                color=style["color"],
                                alpha=0.12,
                                linewidth=0,
                            )
                    ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1.0)
                    if row_idx == 0:
                        ax.set_title(model_labels.get(model, model), fontweight="bold")
                    if col_idx == 0:
                        ax.set_ylabel("MIA AUC")
                        ax.text(
                            -0.30,
                            0.5,
                            dataset_labels.get(dataset, dataset),
                            transform=ax.transAxes,
                            rotation=90,
                            va="center",
                            ha="center",
                            fontsize=14,
                            fontweight="bold",
                        )
                    if row_idx == len(datasets) - 1:
                        ax.set_xlabel("Context size (%)")
                    ax.set_xlim(0, 105)
                    ax.set_ylim(0.45, 1.02)
                    ax.set_xticks([5, 10, 20, 40, 60, 80, 100])
                    ax.set_xticklabels(["5", "10", "20", "40", "60", "80", "100"])
                    ax.grid(True, alpha=0.25)

            handles = [
                Line2D([0], [0], color=style["color"], marker=style["marker"], linestyle=style["linestyle"], linewidth=2.1, markersize=5.2, label=attack.replace(" row_max", ""))
                for attack, style in attack_styles.items()
            ]
            fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.02))
            fig.tight_layout(rect=(0.02, 0.05, 1, 1))
            combined_out = os.path.join(save_dir, "29_context_rmia_amia_evolution_combined.png")
            fig.savefig(combined_out, dpi=300, bbox_inches="tight")
            plt.close(fig)
        logger.info("Saved combined context RMIA/AMIA evolution plot to %s", combined_out)

    # ── plot registry ─────────────────────────────────────────────────────────
    # Maps CLI key: (method_name, needs_rmia_filter).
    # Methods that produce multiple numbered outputs are grouped under one key.
    _PLOT_REGISTRY = {
        "0":            ("create_dataset_properties_summary", False),
        "1":            ("create_accuracy_comparison",        True),
        "2":            ("create_attack_auc_comparison",      True),   # 02 + 03
        "4":            ("create_comprehensive_dashboard",    True),
        "5":            ("create_member_nonmember_signal_distributions", True),  # 05–07
        "8":            ("create_dataset_stats_auc_correlation", True),
        "9":            ("create_attack_type_comparison",     False),  # 09–12
        "13":           ("create_predict_type_analysis",      True),
        "14":           ("create_online_offline_comparison",  False),  # 14a–14c
        "22":           ("create_dataset_attack_roc_curves", False),
        "23":           ("create_locations_tabpfn_attack_roc_curves", False),
        "24":           ("create_dropout_success_tabpfn_attack_roc_curves", False),
        "25":           ("create_locations_tabpfn_rmia_amia_roc_curves", False),
        "26":           ("create_dropout_success_tabpfn_rmia_amia_roc_curves", False),
        "27b":          ("create_all_models_rmia_amia_roc_curves", False),
        "28":           ("create_row_max_layer_study", False),
        "29":           ("create_context_rmia_amia_evolution", False),
        "31":           ("create_highrisk_selection_rows", False),
        "row_max_layers": ("create_row_max_layer_study", False),
        "highrisk_selection": ("create_highrisk_selection_rows", False),
        "roc_rmia_amia": ("create_locations_tabpfn_rmia_amia_roc_curves", False),
        "roc_rmia_amia_dropout": ("create_dropout_success_tabpfn_rmia_amia_roc_curves", False),
        "roc":          ("create_dataset_attack_roc_curves", False),
        "roc_tabpfn":   ("create_locations_tabpfn_attack_roc_curves", False),
        "roc_dropout_tabpfn": ("create_dropout_success_tabpfn_attack_roc_curves", False),
        "16":           ("create_amia_results_plots",         False),  # 16–19
        "16a":          ("create_amia_results_plots",         False),
        "18b":          ("create_purchases10_amia_rmia_comparison", False),
        "purchases10":  ("create_purchases10_amia_rmia_comparison", False),
        "21":           ("create_amia_attack_time",        False),
        "amia":         ("create_amia_results_plots",         False),
        "amia_attack_time": ("create_amia_attack_time",     False),
        "time_auc":     ("create_amia_attack_time",        False),
        "feature_corr": ("create_feature_target_correlation_summary", True),
        "ctx_size":     ("create_context_size_analysis",      False),
        "proxy":        ("create_proxy_heatmaps",             False),
        # ctx_amia is handled separately (takes dataset/model kwargs)
    }

    def run_plots(self, keys: list, save_dir: str = None,
                  ctx_amia_dataset: str = "locations",
                  ctx_amia_model: str = "tabpfn"):
        """Run only the requested subset of plots.

        Keys are CLI identifiers: numbers like '1', '9', or names like 'ctx_size',
        'ctx_amia', 'proxy'.  Multiple keys can be given.
        """
        if save_dir is None:
            save_dir = self.output_dir
        os.makedirs(save_dir, exist_ok=True)

        self.create_dataset_properties_summary(save_dir)
        self.collect_all_results()
        if not self.results:
            logger.error("No results found!")
            return

        all_results = self.results
        all_datasets = set(self.datasets)
        all_models   = set(self.models)
        rmia_results = [r for r in self.results if r.get("attack") == "rmia"]

        def _use_rmia():
            self.results  = rmia_results
            self.datasets = {r["dataset"] for r in rmia_results}
            self.models   = {r["model"]   for r in rmia_results}

        def _use_all():
            self.results  = all_results
            self.datasets = all_datasets
            self.models   = all_models

        unknown = [k for k in keys if k not in self._PLOT_REGISTRY and k != "ctx_amia"]
        if unknown:
            logger.warning("Unknown plot key(s): %s. Available: %s, ctx_amia",
                           unknown, sorted(self._PLOT_REGISTRY))

        for key in keys:
            if key == "ctx_amia":
                _use_all()
                self.create_ctx_amia_sweep(save_dir,
                                           dataset=ctx_amia_dataset,
                                           model=ctx_amia_model)
                continue
            if key not in self._PLOT_REGISTRY:
                continue
            method_name, needs_rmia = self._PLOT_REGISTRY[key]
            _use_rmia() if needs_rmia else _use_all()
            getattr(self, method_name)(save_dir)

        _use_all()
        logger.info("Done. Plots saved to %s/", save_dir)

    def run_all_analysis(self, save_dir: str = None):
        """Run all analysis and create all visualizations."""
        if save_dir is None:
            save_dir = self.output_dir

        logger.info("Starting comprehensive results analysis...")

        self.create_dataset_properties_summary(save_dir)
        self.collect_all_results()

        if not self.results:
            logger.error("No results found!")
            return

        # Multi-attack comparison before filtering to RMIA-only
        logger.info("Creating multi-attack comparison visualizations...")
        self.create_attack_type_comparison(save_dir)

        # Online vs offline comparison for RMIA and LiRA
        logger.info("Creating online vs offline comparison...")
        self.create_online_offline_comparison(save_dir)

        logger.info("Creating dataset-level attack ROC curves...")
        self.create_dataset_attack_roc_curves(save_dir)

        logger.info("Creating locations TabPFN attack ROC curves...")
        self.create_locations_tabpfn_attack_roc_curves(save_dir)

        logger.info("Creating dropout_success TabPFN attack ROC curves...")
        self.create_dropout_success_tabpfn_attack_roc_curves(save_dir)

        # AMIA result diagnostics from seed-level attention summaries.
        logger.info("Creating AMIA result visualizations...")
        self.create_amia_results_plots(save_dir)

        logger.info("Creating purchases10 AMIA/RMIA comparison...")
        self.create_purchases10_amia_rmia_comparison(save_dir)

        # AMIA attack-time dashboard.
        logger.info("Creating AMIA attack-time dashboard...")
        self.create_amia_attack_time(save_dir)

        # Context-size analysis — uses rmia_ctx<pct> runs, must run before RMIA filter.
        logger.info("Creating context-size analysis...")
        self.create_context_size_analysis(save_dir)

        # Filter to RMIA offline-only for remaining analyses (avoids double-counting when
        # lira/loss/attack_p subfolders share the same model directory).
        all_results = self.results  # keep full results for proxy stats (needs Attack-P)
        rmia_results = [r for r in self.results if r.get("attack") == "rmia"]
        if rmia_results:
            self.results = rmia_results
            self.datasets = {r["dataset"] for r in self.results}
            self.models = {r["model"] for r in self.results}

        self.create_accuracy_comparison(save_dir)
        self.create_attack_auc_comparison(save_dir)
        self.create_comprehensive_dashboard(save_dir)
        self.create_member_nonmember_signal_distributions(save_dir)
        self.create_dataset_stats_auc_correlation(save_dir)
        self.create_feature_target_correlation_summary(save_dir)
        self.create_predict_type_analysis(save_dir)
        self.create_ctx_amia_sweep(save_dir, dataset="locations", model="tabpfn")

        # Proxy model analysis — restore full results so Attack-P is available for stats
        self.results = all_results
        self.create_proxy_heatmaps(save_dir)

        logger.info(f"All visualizations saved to: {save_dir}/")
        logger.info("Analysis complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize MIA results.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Plot keys:\n"
            "  Numbers : 0, 1, 2, 4, 5, 8, 9, 13, 14, 21, 22, 23, 24, 25, 26, 28, 29, 31\n"
            "  Named   : amia, amia_attack_time, time_auc, roc, roc_tabpfn, roc_dropout_tabpfn, ctx_size, ctx_amia, proxy, feature_corr, highrisk_selection\n"
            "  Omit    : run all plots\n"
            "\nExamples:\n"
            "  uv run results_visualizations/attacks_viz.py                         # all plots\n"
            "  uv run results_visualizations/attacks_viz.py ctx_amia                # ctx sweep only\n"
            "  uv run results_visualizations/attacks_viz.py ctx_amia --dataset adult --model tabicl\n"
            "  uv run results_visualizations/attacks_viz.py 1 2 9                   # selected plots\n"
        ),
    )
    parser.add_argument(
        "plots", nargs="*",
        help="Plot key(s) to generate (see below). Omit to run all.",
    )
    parser.add_argument("--dataset", default="locations",
                        help="Dataset for ctx_amia (default: locations)")
    parser.add_argument("--model", default="tabpfn",
                        help="Model for ctx_amia (default: tabpfn)")
    parser.add_argument("--save-dir", default="results_visualizations/attacks_viz",
                        help="Output directory (default: results_visualizations/attacks_viz)")
    parser.add_argument(
        "--exclude-datasets",
        default="aloi,46956_seismic-bumps,lcld,purchases10",
        help="Comma-separated datasets to omit from plots/CSVs. Use empty string to include all.",
    )
    args = parser.parse_args()

    excluded = {item.strip() for item in args.exclude_datasets.split(",") if item.strip()}
    analyzer = ResultsAnalyzer(
        logs_base_dir="ml_privacy_meter/logs",
        output_dir=args.save_dir,
        excluded_datasets=excluded,
    )

    if args.plots:
        analyzer.run_plots(
            args.plots,
            save_dir=args.save_dir,
            ctx_amia_dataset=args.dataset,
            ctx_amia_model=args.model,
        )
    else:
        analyzer.run_all_analysis(args.save_dir)
