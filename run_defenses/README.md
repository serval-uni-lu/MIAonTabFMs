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

TabFM defenses are applied to the attention mechanisms and evaluated through `run_defenses/eval_defenses.py`.

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
