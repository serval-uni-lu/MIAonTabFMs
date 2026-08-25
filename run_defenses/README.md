# Defense Evaluation

Use `eval_defenses.py` for the current defense experiments.  

## Outputs

Defense artifacts are written to:

```text
ml_privacy_meter/logs/<dataset>/<model>/defense/<defense_name>/
```

The summary CSV is:

```text
ml_privacy_meter/logs/<dataset>/<model>/defense/defense_eval_results.csv
```

Existing complete defense rows are recomputed by default.  Use
`--skip-existing` only when you explicitly want to reuse complete RMIA/AMIA
artifacts.

## HAMP

HAMP is an output-probability defense, so it is evaluated with output-based
attacks such as RMIA using `--defense hamp`. It recomputes attack signals after
wrapping the target and reference models.

The other TabFM defenses are applied to the attention mechanisms and evaluated through `run_defenses/eval_defenses.py`.

Single seeded RMIA HAMP run:

```bash
uv run run_attacks/rmia.py \
  --dataset locations --model rf --mode load --seed 1 --defense hamp
```

Batch seeded RMIA HAMP runs:

```bash
uv run run_attacks/run_batch.py --attack rmia \
  --datasets all --models all --mode load \
  --seeds 1,2,3,4,5 --defense hamp --continue-on-error
```

## Seeded Robustness Trials

Seeded defense evaluation uses the same seeded RMIA split, models, signals, and
AMIA artifacts. If `--seed` is omitted, `eval_defenses.py` defaults to seed `1`.

Run seeded RMIA first:

```bash
uv run run_attacks/rmia.py \
  --dataset locations --model tabpfn --mode train --seed 1
```

The seeded RMIA source artifacts are read from:

```text
ml_privacy_meter/logs/<dataset>/<model>/seed<seed>/rmia/
```

Standalone defended RMIA runs such as:

```bash
uv run run_attacks/rmia.py \
  --dataset credit_rating \
  --model rf \
  --mode load \
  --seed 1 \
  --defense hamp
```

## No-op Values

These values disable a defense and are skipped:

```text
--kanon-ks 1
--label-kanon-ks 1
--knn-ks 1
--attn-dropout-ps 0
--layer-dropout-ps 0
```

## Useful Commands

Attention dropout, all layers:

```bash
uv run run_defenses/eval_defenses.py \
  --dataset credit_rating \
  --model tabpfn \
  --kanon-ks 1 \
  --label-kanon-ks 1 \
  --knn-ks 1 \
  --attn-dropout-ps 0.2 0.3 0.4 \
  --attn-dropout-layers all \
  --layer-dropout-ps 0 \
  --attacks rmia amia
```

Attention dropout layer presets:

- `--attn-dropout-layers all`: all transformer layers.
- `--attn-dropout-layers late`: last 6 layers; for TabPFN this is L18-L23.
- `--auto-top-dropout`: AMIA-peak layers selected from cached baseline AMIA signals.


Adaptive high-risk guardrail with auto-selected peak-layer dropout:

```bash
uv run run_defenses/eval_defenses.py \
  --dataset credit_rating \
  --model tabpfn \
  --kanon-ks 1 \
  --label-kanon-ks 1 \
  --knn-ks 1 \
  --attn-dropout-ps 0.3 \
  --auto-top-dropout \
  --auto-dropout-top-layers 6 \
  --high-risk-guardrail \
  --layer-dropout-ps 0 \
  --attacks rmia amia
```

Adaptive high-risk guardrail with label k-anon fallback:

```bash
uv run run_defenses/eval_defenses.py \
  --dataset credit_rating \
  --model tabpfn \
  --seed 1 \
  --kanon-ks 1 \
  --label-kanon-ks 2 3 5\
  --label-kanon-alphas 0.5 0.6 0.7 0.9 \
  --knn-ks 1 \
  --attn-dropout-ps 0 \
  --high-risk-guardrail \
  --high-risk-fallback label_kanon \
  --layer-dropout-ps 0 \
  --attacks rmia amia
```

By default, `--high-risk-guardrail` sets the threshold to the minimum
known-member AMIA risk in the baseline calibration set. At runtime, any query
with `probe_risk_score >= risk_threshold` is defended. Use
`--high-risk-threshold` only when you want to override that default.

Adaptive high-risk guardrail with a non-member margin push:

```bash
uv run run_defenses/eval_defenses.py \
  --dataset credit_rating \
  --model tabpfn \
  --seed 1 \
  --kanon-ks 1 \
  --label-kanon-ks 2 3 5 \
  --label-kanon-alphas 0.5 0.6 0.7 0.9 \
  --knn-ks 1 \
  --attn-dropout-ps 0 \
  --high-risk-guardrail \
  --high-risk-fallback label_kanon \
  --high-risk-nonmember-margin 0.75 \
  --high-risk-near-perfect-auc 0.85 \
  --high-risk-kfold-folds 5 \
  --layer-dropout-ps 0 \
  --attacks rmia amia
```

`--high-risk-nonmember-margin` only fires when the out-of-fold achievable AUC
of the default (min-known-member) threshold's raw risk score reaches
`--high-risk-near-perfect-auc` — i.e. members and non-members are already
near-perfectly separated. When it fires, the threshold is pushed down so it
additionally catches at least that margin fraction of non-members (e.g. 0.75
= 75%). Threshold calibration is always leave-fold-out
(`--high-risk-kfold-folds`, default 5): for each fold the threshold is
computed from the other folds only, so no row calibrates its own bar.
Leaving the margin at its default (`0.0`) disables the push entirely — this
is opt-in. Affected defense names get a `_m<margin*100>` suffix (e.g.
`highrisk_label_kanon_k3_m75`) so margin and non-margin runs never overwrite
each other.

In adaptive high-risk `attention_summary.csv` files, `probe_risk_score` is the
baseline, pre-defense score used for the high-risk decision:
`fallback_applied = probe_risk_score >= risk_threshold`. The AMIA columns such
as `row_max` are final/adaptive values; for flagged rows they come from the
fallback defense, so they can be lower than `probe_risk_score`.

Label-aware k-anon:

```bash
uv run run_defenses/eval_defenses.py \
  --dataset credit_rating \
  --model tabicl \
  --kanon-ks 1 \
  --label-kanon-ks 2 4 \
  --label-kanon-alphas 0.7 0.9 \
  --knn-ks 1 \
  --attn-dropout-ps 0 \
  --layer-dropout-ps 0 \
  --attacks rmia amia
```

To run softer utility-preserving variants, add `--label-kanon-alphas`, e.g.
`--label-kanon-alphas 0.8 0.9`. Without this argument, the default is
`alpha = 0.0`, i.e. pure label k-anon centroiding.
