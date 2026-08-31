#!/usr/bin/env bash
set -u

# Full experiment launcher.
#
# Defaults:
#   - all datasets discovered by run_attacks/run_batch.py
#   - all supported attack models for RMIA/LiRA/population
#   - TabFM models only for AMIA and TabFM defenses
#   - seed 1 for seeded RMIA/AMIA/defense evaluation
#   - GPU 0 by default; set GPUS="" to force CPU
#
# Useful overrides:
#   DATASETS=credit_rating MODELS=tabicl TABFM_MODELS=tabicl SEEDS=1 DRY_RUN=1 ./run_all_experiments_batch.sh
#   START_PHASE=5 ./run_all_experiments_batch.sh  # resume directly at Phase 5
#   START_PHASE=5 END_PHASE=5 ./run_all_experiments_batch.sh  # run only Phase 5
#   GPUS=0,1,2 CONTINUE_ON_ERROR=1 ./run_all_experiments_batch.sh
#   GPUS="" ./run_all_experiments_batch.sh  # force CPU

DATASETS="${DATASETS:-all}"
SKIP_DATASETS="${SKIP_DATASETS:-aloi,purchases10,46956_seismic-bumps,lcld}"
MODELS="${MODELS:-all}"
TABFM_MODELS="${TABFM_MODELS:-tabpfn,real-tabpfn,tabicl,tabdpt}"
SEEDS="${SEEDS:-1,2,3,4,5}"
GPUS="${GPUS:-0}"
DRY_RUN="${DRY_RUN:-0}"
START_PHASE="${START_PHASE:-1}"
END_PHASE="${END_PHASE:-5}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
MAX_JOBS="${MAX_JOBS:-1}"

LABEL_KS="${LABEL_KS:-3 5 10}"
LABEL_ALPHAS="${LABEL_ALPHAS:-0.0 0.3 0.5}"
ATTN_DROPOUT_PS="${ATTN_DROPOUT_PS:-0.1 0.3 0.5 0.7 0.9}"
# LAYER_DROPOUT_PS="${LAYER_DROPOUT_PS:-0.1 0.3 0.5}"
DEFENSE_SEEDS="${DEFENSE_SEEDS:-1}"

failures=()
_interrupted=0
active_jobs=0

# Ctrl+C kills only the running child; the script continues to the next phase.
trap '_interrupted=1; echo; echo "[SIGINT] Skipping current step — continuing..." >&2' INT

validate_phase_range() {
  case "${START_PHASE}:${END_PHASE}" in
    *[!0-9:]*|:*|*:)
      echo "START_PHASE and END_PHASE must be integers in 1..5" >&2
      exit 2
      ;;
  esac
  if (( START_PHASE < 1 || END_PHASE > 5 || START_PHASE > END_PHASE )); then
    echo "Invalid phase range: START_PHASE=${START_PHASE} END_PHASE=${END_PHASE}; expected 1 <= START_PHASE <= END_PHASE <= 5" >&2
    exit 2
  fi
}

should_run_phase() {
  local phase="$1"
  (( phase >= START_PHASE && phase <= END_PHASE ))
}

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

run_cmd() {
  echo
  echo "[$(timestamp)] $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi

  _interrupted=0
  "$@"
  local code=$?
  if [[ $_interrupted -eq 1 ]]; then
    echo "[SKIP] Step interrupted by user: $*" >&2
    failures+=("interrupted :: $*")
    _interrupted=0
    return 0
  fi
  if [[ $code -ne 0 ]]; then
    echo "[FAIL] exit_code=$code :: $*" >&2
    failures+=("$code :: $*")
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit "$code"
    fi
  fi
  return 0
}

run_cmd_job() {
  echo
  echo "[$(timestamp)] $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi

  "$@"
  local code=$?
  if [[ $code -ne 0 ]]; then
    echo "[FAIL] exit_code=$code :: $*" >&2
  fi
  return "$code"
}

wait_for_one_job() {
  local code=0
  if wait -n; then
    code=0
  else
    code=$?
  fi
  active_jobs=$((active_jobs - 1))
  if [[ $code -ne 0 ]]; then
    failures+=("parallel job failed with exit_code=$code")
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      wait || true
      exit "$code"
    fi
  fi
}

wait_for_available_slot() {
  while (( active_jobs >= MAX_JOBS )); do
    wait_for_one_job
  done
}

wait_for_all_jobs() {
  while (( active_jobs > 0 )); do
    wait_for_one_job
  done
}

run_cmd_async() {
  if (( MAX_JOBS <= 1 )); then
    run_cmd "$@"
    return 0
  fi

  wait_for_available_slot
  run_cmd_job "$@" &
  active_jobs=$((active_jobs + 1))
}

batch_common_args() {
  local -n out_ref=$1
  out_ref=(--datasets "$DATASETS_FILTERED" --models "$MODELS" --skip-existing)
  if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
    out_ref+=(--continue-on-error)
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    out_ref+=(--dry-run)
  fi
}

batch_gpu_args() {
  local -n out_ref=$1
  if [[ -n "$GPUS" ]]; then
    out_ref+=(--gpus "$GPUS")
  fi
}

amia_gpu_args() {
  local -n out_ref=$1
  local job_index="${2:-0}"
  out_ref=()
  if [[ -n "$GPUS" ]]; then
    local gpu_ids=()
    mapfile -t gpu_ids < <(csv_to_lines "$GPUS")
    local gpu="${gpu_ids[$((job_index % ${#gpu_ids[@]}))]}"
    out_ref+=(--gpu "$gpu")
  fi
}

batch_gpu_args_for_job() {
  local -n out_ref=$1
  local job_index="${2:-0}"
  out_ref=()
  if [[ -n "$GPUS" ]]; then
    local gpu_ids=()
    mapfile -t gpu_ids < <(csv_to_lines "$GPUS")
    local gpu="${gpu_ids[$((job_index % ${#gpu_ids[@]}))]}"
    out_ref+=(--gpus "$gpu")
  fi
}

resolve_datasets() {
  DATASETS_RAW="$DATASETS" SKIP_DATASETS_RAW="${SKIP_DATASETS:-}" uv run python - <<'PY'
import os
from run_attacks.run_batch import discover_original_datasets, parse_csv_arg

raw = os.environ["DATASETS_RAW"]
skip_raw = os.environ.get("SKIP_DATASETS_RAW", "")
skip_set = {s.strip().lower() for s in skip_raw.split(",") if s.strip()}

items = parse_csv_arg(raw)
if any(item.lower() == "all" for item in items):
    explicit = [item for item in items if item.lower() != "all"]
    datasets = sorted(set(discover_original_datasets() + explicit))
else:
    datasets = items
for dataset in datasets:
    if dataset.lower() not in skip_set:
        print(dataset)
PY
}

csv_to_lines() {
  local raw="$1"
  raw="${raw//,/ }"
  for item in $raw; do
    [[ -n "$item" ]] && echo "$item"
  done
}

amia_command_for_model() {
  local model="$1"
  case "$model" in
    tabpfn)
      echo "run_attacks/amia/amia_tabpfn.py --model tabpfn"
      ;;
    real-tabpfn)
      echo "run_attacks/amia/amia_tabpfn.py --model real-tabpfn"
      ;;
    tabicl)
      echo "run_attacks/amia/amia_tabicl.py"
      ;;
    tabdpt)
      echo "run_attacks/amia/amia_tabdpt.py"
      ;;
    *)
      return 1
      ;;
  esac
}

seeded_rmia_exists() {
  local dataset="$1"
  local model="$2"
  local seed="$3"
  [[ -d "ml_privacy_meter/logs/${dataset}/${model}/seed${seed}/rmia/models" ]]
}

amia_extra_args_for_dataset_model() {
  local -n out_ref=$1
  local dataset="$2"
  local model="$3"
  out_ref=()

  # Large audit pools make TabPFN attention extraction very expensive. Keep the
  # same audit samples, but use a larger batch and half of the row-attention
  # calls for the TabPFN backends on datasets with >~5k audit queries.
  case "$model" in
    tabpfn|real-tabpfn)
      case "$dataset" in
        aloi|purchases10|url|lcld)
          out_ref+=(--batch-size 512 --max-col-calls 0) # --max-row-calls 256 
          ;;
      esac
      ;;
  esac
}

_filtered_list=()
mapfile -t _filtered_list < <(resolve_datasets)
validate_phase_range
DATASETS_FILTERED=$(IFS=','; echo "${_filtered_list[*]}")

echo "[$(timestamp)] Starting full experiment batch"
echo "DATASETS=${DATASETS}"
echo "SKIP_DATASETS=${SKIP_DATASETS:-<none>}"
echo "DATASETS_FILTERED=${DATASETS_FILTERED}"
echo "MODELS=${MODELS}"
echo "TABFM_MODELS=${TABFM_MODELS}"
echo "SEEDS=${SEEDS}"
echo "GPUS=${GPUS:-<none>}"
echo "DRY_RUN=${DRY_RUN}"
echo "CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}"
echo "START_PHASE=${START_PHASE}"
echo "END_PHASE=${END_PHASE}"
echo "MAX_JOBS=${MAX_JOBS}"

common=()
batch_common_args common
gpu_args=()
batch_gpu_args gpu_args
amia_device_args=()
amia_gpu_args amia_device_args
if [[ ${#amia_device_args[@]} -gt 0 ]]; then
  echo "AMIA_GPU=${amia_device_args[1]}"
else
  echo "AMIA_GPU=<cpu>"
fi

mapfile -t datasets < <(resolve_datasets)
mapfile -t tabfm_models < <(csv_to_lines "$TABFM_MODELS")
mapfile -t seeds < <(csv_to_lines "$SEEDS")
mapfile -t defense_seeds < <(csv_to_lines "$DEFENSE_SEEDS")
echo
if should_run_phase 1; then
echo "========== Phase 1: seeded RMIA training + all attacks =========="
seeded_train_args=(--datasets "$DATASETS_FILTERED" --models "$MODELS" --mode load --seeds "$SEEDS" --skip-existing)
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  seeded_train_args+=(--continue-on-error)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  seeded_train_args+=(--dry-run)
fi
if [[ -n "$GPUS" ]]; then
  seeded_train_args+=(--gpus "$GPUS")
fi
run_cmd uv run run_attacks/run_batch.py --attack rmia "${seeded_train_args[@]}"

echo
fi

if should_run_phase 2; then
echo "========== Phase 2: all attacks =========="
seeded_load_args=(--datasets "$DATASETS_FILTERED" --models "$MODELS" --seeds "$SEEDS" --skip-existing)
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  seeded_load_args+=(--continue-on-error)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  seeded_load_args+=(--dry-run)
fi
rmia_online_args=(--datasets "$DATASETS_FILTERED" --models "$MODELS" --seeds "$SEEDS" --mode load --online --skip-existing)
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  rmia_online_args+=(--continue-on-error)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  rmia_online_args+=(--dry-run)
fi
run_cmd uv run run_attacks/run_batch.py --attack rmia "${rmia_online_args[@]}" "${gpu_args[@]}"
run_cmd uv run run_attacks/run_batch.py --attack lira "${seeded_load_args[@]}" --mode load
run_cmd uv run run_attacks/run_batch.py --attack lira "${seeded_load_args[@]}" --mode load --online
#run_cmd uv run run_attacks/run_batch.py --attack loss "${seeded_load_args[@]}"
run_cmd uv run run_attacks/run_batch.py --attack population "${seeded_load_args[@]}" --mode load

echo
fi

if should_run_phase 3; then
echo "========== Phase 3: AMIA for TabFM models =========="

amia_job_index=0
for dataset in "${datasets[@]}"; do
  for model in "${tabfm_models[@]}"; do
    amia_cmd_string="$(amia_command_for_model "$model" || true)"
    if [[ -z "$amia_cmd_string" ]]; then
      echo "[SKIP] AMIA backend not configured for model=${model}"
      continue
    fi
    read -r -a amia_cmd <<< "$amia_cmd_string"
    amia_extra_args=()
    amia_extra_args_for_dataset_model amia_extra_args "$dataset" "$model"
    for seed in "${seeds[@]}"; do
      if [[ "$DRY_RUN" != "1" ]] && ! seeded_rmia_exists "$dataset" "$model" "$seed"; then
        echo "[SKIP] AMIA missing seeded RMIA prerequisite: dataset=${dataset} model=${model} seed=${seed}"
        continue
      fi
      amia_device_args=()
      amia_gpu_args amia_device_args "$amia_job_index"
      run_cmd_async uv run "${amia_cmd[@]}" \
        --dataset "$dataset" \
        --seed "$seed" \
        --skip-config \
        --skip-existing \
        "${amia_extra_args[@]}" \
        "${amia_device_args[@]}"
      amia_job_index=$((amia_job_index + 1))
    done
  done
done
wait_for_all_jobs

echo
fi

if should_run_phase 4; then
echo "========== Phase 4: label k-anon + attention dropout defenses =========="
defense_job_index=0
for dataset in "${datasets[@]}"; do
  for model in "${tabfm_models[@]}"; do
    for seed in "${defense_seeds[@]}"; do
      if [[ "$DRY_RUN" != "1" ]] && ! seeded_rmia_exists "$dataset" "$model" "$seed"; then
        echo "[SKIP] Defense missing seeded RMIA prerequisite: dataset=${dataset} model=${model} seed=${seed}"
        continue
      fi
      job_gpu_args=()
      batch_gpu_args_for_job job_gpu_args "$defense_job_index"
      # --combine can be used in the next command
      run_cmd_async uv run run_defenses/eval_defenses.py \
        --dataset "$dataset" \
        --model "$model" \
        --seed "$seed" \
        --kanon-ks 1 \
        --label-kanon-ks $LABEL_KS \
        --label-kanon-alphas $LABEL_ALPHAS \
        --attn-dropout-ps $ATTN_DROPOUT_PS \
        --layer-dropout-ps 0 \
        --auto-top-dropout \
        --skip-existing \
        --attacks rmia amia \
        "${job_gpu_args[@]}"
      defense_job_index=$((defense_job_index + 1))
    done
  done
done
wait_for_all_jobs

echo
fi

if should_run_phase 5; then
echo "========== Phase 5: high-risk label k-anon defenses =========="
high_risk_job_index=0
for dataset in "${datasets[@]}"; do
  for model in "${tabfm_models[@]}"; do
    for seed in "${defense_seeds[@]}"; do
      if [[ "$DRY_RUN" != "1" ]] && ! seeded_rmia_exists "$dataset" "$model" "$seed"; then
        echo "[SKIP] High-risk defense missing seeded RMIA prerequisite: dataset=${dataset} model=${model} seed=${seed}"
        continue
      fi
      job_gpu_args=()
      batch_gpu_args_for_job job_gpu_args "$high_risk_job_index"
      run_cmd_async uv run run_defenses/eval_defenses.py \
        --dataset "$dataset" \
        --model "$model" \
        --seed "$seed" \
        --kanon-ks 1 \
        --label-kanon-ks $LABEL_KS \
        --label-kanon-alphas $LABEL_ALPHAS \
        --attn-dropout-ps 0 \
        --layer-dropout-ps 0 \
        --high-risk-guardrail \
        --high-risk-fallback label_kanon \
        --skip-existing \
        --attacks rmia amia \
        "${job_gpu_args[@]}"
      high_risk_job_index=$((high_risk_job_index + 1))
    done
  done
done
wait_for_all_jobs

echo
fi
echo "[$(timestamp)] Full experiment batch finished"
if [[ ${#failures[@]} -gt 0 ]]; then
  echo
  echo "Failures (${#failures[@]}):"
  for failure in "${failures[@]}"; do
    echo "  - ${failure}"
  done
  exit 1
fi
