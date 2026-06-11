# MIA on Tabular Foundation Models

Membership Inference Attacks (MIA) on tabular models using five attack methods: **RMIA**, **LiRA**, **Attack-P**, **LOSS**, and **QMIA**. A new attack **AMIA** and **High-risk (target) label k-anonymity** defence are proposed.

---

## Setup

**Clone TabPFN first** (required because `pyproject.toml` installs it from the local editable path `./TabPFN`):
```bash
git clone https://github.com/PriorLabs/TabPFN.git
cd TabPFN
git checkout 6.3.2
cd ..
```

**Create the environment and install dependencies:**
```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

**Login to Hugging Face** (required for real-TabPFN):
```bash
uv run huggingface-cli login
```

---

## Data

**Download datasets:**
```bash
uv run get_data.py
```

**Download TabArena datasets** (OpenML):
```bash
uv run download_tabarena_openml.py
```

For any new dataset, place `<dataset>.csv` in `data/original/` with the label in the **last column** (no header). Shape registration and YAML config are created automatically on first run.

---

## Clone ml_privacy_meter

Necessary for RMIA. This project expects the `ml_privacy_meter` fork at
[`tmcarvalho/ml_privacy_meter`](https://github.com/tmcarvalho/ml_privacy_meter.git).

```bash
uv run setup_repo.py
```
Original [`ml_privacy_meter`](https://github.com/privacytrustlab/ml_privacy_meter). 

---

## Supported Models

| Model | Type | Hyperparameter tuning | Notes |
|---|---|---|---|
| `mlp` | Neural network | Optuna | |
| `rf` | Random forest | Optuna | |
| `lightgbm` | Gradient boosting | Optuna | |
| `tabpfn` | Foundation model | — | Max 50k samples, 2k features |
| `real-tabpfn` | Foundation model | — | Max 50k samples, 2k features |
| `tabicl` | Foundation model | — | Max 100k samples, 2k features |
| `tabdpt` | Foundation model | — | |
| `tabnet` | Neural network | Optuna | |

Optuna runs 30 trials with 3-fold CV per shadow model (configurable via `tuning_n_trials` / `tuning_cv` in `configs.py`). Cached per run in `best_params.json` under the model log directory.

---

## Attack Pipeline

LiRA, LOSS, Attack-P (population attack), and QMIA (quantile regression) **reuse the models trained by RMIA** — always run RMIA first.


### Single dataset + model

**RMIA** (trains shadow models + audits):
```bash
uv run run_attacks/rmia.py --dataset locations --model mlp --mode train
```

**RMIA online** (reuses existing signals, no retraining):
```bash
uv run run_attacks/rmia.py --dataset locations --model mlp --mode load --online
```
Results go to `report_online/` so offline results are not overwritten.

**LiRA** (reuses RMIA models; loads cached signals):
```bash
uv run run_attacks/lira.py --dataset locations --model mlp
```

**LiRA — recompute signals** (deletes cached LiRA signals; re-copies from RMIA signals if available):
```bash
uv run run_attacks/lira.py --dataset locations --model mlp --mode signal
```

**LiRA online** (uses both IN and OUT shadow models):
```bash
uv run run_attacks/lira.py --dataset locations --model mlp --online
```

**LOSS** (reuses RMIA models):
```bash
uv run run_attacks/loss.py --dataset locations --model mlp
```

**QMIA** (quantile-regression MIA; reuses RMIA models):
```bash
uv run run_attacks/quantile_regression.py --dataset locations --model mlp
```

**Population attack / attack-p** (reuses RMIA models + population signals):
```bash
uv run run_attacks/population_attack.py --dataset locations --model mlp
```

**Population attack — recompute signals:**
```bash
uv run run_attacks/population_attack.py --dataset locations --model mlp --mode signal
```


### Batch runs (all datasets × all models)

All attacks share a single launcher. Replace `<attack>` with `rmia`, `lira`, `loss`, `population`, or `qmia`.

```bash
uv run run_attacks/run_batch.py --attack <attack> --datasets all --models all --skip-existing --continue-on-error
```


### Full experiment launcher

Run the end-to-end launcher from the beginning:

```bash
./run_all_experiments_batch.sh
```

Resume directly at Phase 5 and continue with following phases:

```bash
START_PHASE=5 ./run_all_experiments_batch.sh
```

Run only Phase 5:

```bash
START_PHASE=5 END_PHASE=5 ./run_all_experiments_batch.sh
```

Useful overrides compose with the phase range, for example:

```bash
START_PHASE=5 DATASETS=credit_rating TABFM_MODELS=tabpfn GPUS=0 ./run_all_experiments_batch.sh
```

If executable permissions complain, use `bash run_all_experiments_batch.sh` with the same environment variables.


### Including TabArena datasets in batch

```bash
uv run run_attacks/run_batch.py --attack rmia \
  --datasets all,46905_Amazon_employee_access,46908_APSFailure \
  --models all --mode train --skip-existing --continue-on-error
```

Add `--online` to run the online version (reuses existing models and signals).

### Run a single model across all datasets

```bash
uv run run_attacks/run_batch.py --attack lira \
  --datasets all,46905_Amazon_employee_access,46908_APSFailure \
  --model tabicl --skip-existing --continue-on-error
```

### Useful batch flags

| Flag | Applies to | Description |
|---|---|---|
| `--mode train` | rmia | Delete existing models and retrain from scratch, then recompute signals. Add `--skip-existing` to skip retraining if models are already on disk |
| `--mode signal` | rmia, lira, population, qmia | Load existing RMIA models and recompute signals (no retraining). For rmia this is explicit; for others it is the only recompute option |
| `--mode load` | rmia, lira, population, qmia | Reuse existing models and cached signals (default for all except rmia) |
| `--online` | rmia, lira | **Online** variant: scores each sample using both IN and OUT shadow models. Results go to `report_online/`. Use with `--mode load` to reuse existing signals |
| `--defense hamp` | rmia, lira, loss, population, qmia | Apply HAMP test-time defense before computing signals |
| `--skip-existing` | all | Skip combinations that already have output |
| `--continue-on-error` | all | Keep going if one combination fails |
| `--dry-run` | all | Print planned commands without executing |
| `--dataset <name>` | all | Run a single dataset (overrides `--datasets`) |
| `--model <name>` | all | Run a single model (overrides `--models`) |
| `--gpus 0,1` | rmia | Distribute jobs across GPUs in parallel |
| `--seeds 1,2,3,4,5` | rmia | Run repeated default RMIA trials for the listed seeds and write mean/std summaries |

### Seeded RMIA robustness runs

Use `--seeds` to evaluate whether RMIA results are stable across different random trials. For each seed, RMIA shuffles the dataset before the 75/25 train-population split, varies target/reference memberships, controls model `random_state`, and reuses the same seed for audit downsampling.

```bash
uv run run_attacks/run_batch.py --attack rmia \
  --datasets all --models all --mode train \
  --seeds 1,2,3,4,5 --continue-on-error
```

Single dataset/model:
```bash
uv run run_attacks/rmia.py --dataset locations --model rf --mode train --seed 1
```

`dataset_permutation.npy` is the source of truth for the seeded 75/25 split once it exists. RMIA, AMIA, and defense evaluation all use it to reconstruct the same seeded data order.


---

## Context-size sweep (foundation models only)

Studies how the number of training samples seen at inference time (context size) affects MIA vulnerability. Supported models: `tabpfn`, `real-tabpfn`, `tabicl`, `tabdpt`.

Each run uses a fraction of the training pool as context. 100% is excluded — use regular batch runs for that baseline.

**Single dataset, default percentages (5, 10, 20, 30, 40, 50, 60, 70, 80, 90):**
```bash
uv run run_attacks/run_context_size_batch.py \
  --datasets all --model tabicl --mode train --skip-existing --continue-on-error
```

**Custom percentages:**
```bash
uv run run_attacks/run_context_size_batch.py \
  --dataset locations --model tabdpt \
  --context-pcts 10,25,50,75,90 --mode train --continue-on-error
```

---

## Reference model analysis

Studies how the number of RMIA reference models affects attack performance.

`run_ref_analysis.py` is a convenience wrapper around `rmia.py --num-ref`. It:
1. Calls `rmia.py --mode train --num-ref <max>` **once** — trains all models (target pair + `max` reference pairs) and caches their signals
2. Loops `rmia.py --mode load --num-ref k` for each k — reruns only the **audit** using the cached signals, slicing the first 2×k reference columns

You can also call `rmia.py --num-ref` directly for a single k (see below).

Results go to `report_ref{k}/` so the baseline report is never overwritten.

**All datasets × all models, default counts (1, 2, 5, 10, 25, 50):**
```bash
uv run run_attacks/run_ref_analysis.py --datasets all --models all --max-ref 50 --skip-existing --continue-on-error --gpus 0,1,2
```
`--datasets all` discovers datasets from `ml_privacy_meter/logs/` (i.e., what RMIA has already trained).

**Single dataset + model:**
```bash
uv run run_attacks/run_ref_analysis.py --dataset locations --model rf
```

**Custom reference counts:**
```bash
uv run run_attacks/run_ref_analysis.py --datasets all --models all --ref-counts 1,2,5,10,25,50
```

**All models already trained — audit only for each k:**
```bash
uv run run_attacks/run_ref_analysis.py --datasets all --models all --skip-train
```

**Single k directly via `rmia.py`:**
```bash
# Step 1 — train all models (target + 50 reference pairs)
uv run run_attacks/rmia.py --dataset locations --model rf --mode train --num-ref 50

# Step 2 — audit only for k=5, reuses cached signals → report_ref5/
uv run run_attacks/rmia.py --dataset locations --model rf --mode load --num-ref 5
```


---
## Proxy models
Use other models as reference models for a specific target model (RMIA and QMIA).

All target × proxy combinations (offline by default):
```bash
uv run run_attacks/run_batch.py --attack rmia \
  --datasets all --models all --proxy-models all --skip-existing --continue-on-error
```

Single target model, all proxy models:
```bash
uv run run_attacks/run_batch.py --attack rmia \
  --dataset 46980_MIC --model lightgbm --proxy-models all --gpus 0,1
```

---

## Attention-based MIA

Exploits the ICL transformer's attention weights to distinguish members from non-members. A sample present in the training context can attend to itself in the key set, producing a sharp attention spike. Non-members have no matching key, so their attention is more diffuse.

**Pre-requisite:** RMIA must have been run in train mode first — it produces the shadow models and `memberships.npy` that this script reads.

The MIA score for a sample is its `max_attn` over the training context under the model that was trained on it.

```bash
uv run run_attacks/amia/amia_tabpfn.py --dataset locations
uv run run_attacks/amia/amia_tabpfn.py --dataset locations --model real-tabpfn
```

To regenerate plots, skip model loading and signal extraction.
```bash
uv run run_attacks/amia/amia_tabpfn.py --dataset locations --plots-only --skip-config
```

**Seeded AMIA** (uses the matching seeded RMIA split, models, signals, and config):
```bash
# Requires RMIA seed run first:
uv run run_attacks/rmia.py --dataset locations --model tabpfn --mode train --seed 1

# AMIA baseline for that exact seeded trial:
uv run run_attacks/amia/amia_tabpfn.py --dataset locations --seed 1
```

**With a context-size percentage** (reads from `rmia_ctx<pct>/`, writes to `amia_ctx<pct>/`):
```bash
uv run run_attacks/amia/amia_tabpfn.py --dataset locations --context-pct 50
```

**Batch sweep over all context percentages:**
```bash
uv run run_attacks/amia/run_amia_context_size_batch.py --datasets all --models all \
  --context-pcts 10,25,50,75,90 --continue-on-error
```

Requires the corresponding `rmia_ctx<pct>/` runs to exist (produced by `run_context_size_batch.py`).

### Options

| Flag | Default | Description |
|---|---|---|
| `--dataset` | required | Dataset name (e.g. `locations`) |
| `--model` | `tabpfn` | Only for `amia_tabpfn.py`: `tabpfn` or `real-tabpfn` |
| `--gpu` | auto | GPU ID, e.g. `0` |
| `--batch-size` | 200 | Inference batch size |
| `--plots-only` | off | Regenerate plots from cached signals without re-running inference |
| `--defense` | `none` | `hamp` applies the output-probability HAMP defense before signal extraction. Other TabFM defenses are run through `run_defenses/eval_defenses.py` |
| `--context-pct` | `100` | Context-size percentage matching a prior `rmia_ctx<pct>/` run |
| `--seed` | `1` | Use seeded artifacts under `logs/<dataset>/<model>/seed<seed>/{rmia,amia}/` |

`amia_tabicl.py` and `amia_tabdpt.py` infer the model from the script name and do not accept `--model`.



---

## Defenses

The **HAMP** (High-confidence Adversarial Membership Privacy) defense is a test-time output perturbation. It replaces each sample's predicted probability magnitudes with those obtained on a random input, while preserving the rank order (argmax is unchanged). This reduces confidence without affecting classification accuracy significantly.

Results are written to `{attack}/defense_hamp/report/` alongside a `defense_accuracy.csv` that reports accuracy and average confidence before and after the defense.

For **TabFM / TabPFN defenses beyond HAMP**, see
[run_defenses/README.md](run_defenses/README.md). It covers k-anon, label
k-anon, kNN smoothing, attention dropout/clipping, auto top-layer dropout, and
high-risk guardrails, including the seeded defense workflow.


## Visualize Results

```bash
uv run results_visualizations/attacks_viz.py                          # all attack plots
uv run results_visualizations/attacks_viz.py 1 2 9                   # selected attack plots by key
uv run results_visualizations/attacks_viz.py amia                    # AMIA summary plots only
uv run results_visualizations/attacks_viz.py ctx_amia --dataset adult --model tabicl
uv run results_visualizations/defenses_viz.py                        # defense plots, filtered to MIC defense set
```

`attacks_viz.py` writes attack figures and CSVs to `results_visualizations/attacks_viz/`. `defenses_viz.py` reads `ml_privacy_meter/logs/*/{tabpfn,real-tabpfn,tabicl,tabdpt}/defense/defense_eval_results.csv`, keeps only defenses whose canonical names appear in `46980_MIC`, and writes raw privacy-utility plots, delta tradeoff plots, metric heatmaps, family-level deltas, focused high-risk plots, best-defense tables, and parameter-evolution plots for `locations` and `dropout_success` to `results_visualizations/defenses_viz/`.

Failed RMIA runs are recorded to `results_visualizations/attacks_viz/rmia_failed_runs.csv`.

AMIA plots read seeded AMIA outputs from `attack_result_seed_runs.csv` when
available, and otherwise recompute AUCs from
`seed*/amia/report/exp/attention_summary.csv`. The generated AMIA CSVs use one
row per seed and one summary row per dataset/model with seed mean and seed std.

### Plot index

| Key | Output file(s) | Description | Data scope |
|---|---|---|---|
| `0` | `00_dataset_properties_summary.png` | Dataset size, features, class balance | All |
| `1` | `01_accuracy_comparison_rmia.png` | Target model accuracy vs reference models | RMIA |
| `1b` | `01b_accuracy_target_vs_reference_online.png`<br>`01c_accuracy_target_only_offline.png` | Target vs reference accuracy (online / offline split) | RMIA |
| `2` | `02_attack_auc_comparison_rmia.png`<br>`03_attack_auc_heatmap_rmia.png` | Attack AUC bar chart and heatmap by model | RMIA |
| `4` | `04_comprehensive_dashboard_rmia.png` | Combined accuracy + AUC dashboard | RMIA |
| `5` | `06a_member_nonmember_kde_{dataset}.png`<br>`06b_attack_effectiveness_{dataset}.png` | Member / non-member score distributions and ROC | RMIA |
| `8` | `08_dataset_stats_auc_correlation_rmia.png` | Dataset statistics vs attack AUC correlations | RMIA |
| `9` | `09_attack_comparison_by_model.png`<br>`10_attack_auc_heatmaps.png`<br>`11_attack_auc_per_dataset.png`<br>`12_tpr_comparison_by_model.png` | Cross-attack AUC comparison (all attacks) | All |
| `13` | `13_predict_type_auc_comparison.png` | Predict-type influence on AUC | RMIA |
| `14` | `14a_rmia_online_vs_offline_by_model.png`<br>`14b_lira_online_vs_offline_by_model.png`<br>`14c_online_gain_heatmap.png` | Online vs offline attack gain for RMIA and LiRA | All |
| `15` | `15_swarmplot_summary.png` | Summary swarmplot across all attacks × models | All |
| `16` / `amia` | `16_amia_auc_by_model.png`<br>`16_amia_seed_runs.csv`<br>`16_amia_seed_summary.csv`<br>`17_amia_row_max_heatmap.png`<br>`18_amia_row_max_per_dataset.png`<br>`19_amia_vs_rmia_auc.png`<br>`19_amia_vs_rmia_auc.csv`<br>`20_amia_rmia_paired_differences.png`<br>`20_amia_rmia_paired_differences.csv`<br>`20_amia_rmia_significance_summary.csv` | AMIA seed summaries, row-max heatmap, per-dataset seed-std bars, AMIA-vs-RMIA comparison, and paired Wilcoxon/bootstrap significance table | AMIA + RMIA summary |
| `ctx_size` | `07a_context_size_auc.png`<br>`07b_context_size_summary.png` | Context size vs attack AUC (foundation models) | All |
| `ctx_amia` | `ctx_amia_sweep_{dataset}_{model}.png` | AMIA context sweep — requires `--dataset` / `--model` | All |
| `proxy` | `P01_proxy_*_gain.png`<br>`P02a_proxy_winrate_heatmap.png`<br>`P03_proxy_significance_heatmap.png` | Proxy model attack gain and win-rate | All |
| `feature_corr` | `09_feature_target_correlation_summary_rmia.csv` | Feature–target correlation table (CSV, no plot) | RMIA |
