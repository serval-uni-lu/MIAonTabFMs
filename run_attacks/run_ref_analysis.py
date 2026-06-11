"""
Reference model analysis: vary the number of RMIA reference models,
rerunning only the audit step for each k (models and signals are trained once).

Strategy
--------
1. Train ALL models once at the maximum reference count (default: 10).
   With max_ref=10 this trains 2*(10+1) = 22 models in total:
     - 1 target pair  (models 0 and 1)
     - 10 reference pairs (models 2–21)
   Predictions for all 22 models are cached in rmia_signals.npy.

2. For every k in --ref-counts, re-run only the audit step (--mode load)
   with --num-ref k.  The full signal cache is reused as-is;
   _get_reference_columns slices the first 2*k reference columns, so
   no extra training or inference is needed for smaller k values.
   Results go to report_ref{k}/ so runs for different k never overwrite
   each other.

Usage
-----
  python run_ref_analysis.py --datasets all --models all
  python run_ref_analysis.py --datasets locations --models rf
  python run_ref_analysis.py --datasets all --models all --max-ref 10 --ref-counts 1,2,4,10
  python run_ref_analysis.py --datasets all --models all --skip-train   # audit only, models already trained
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

SUPPORTED_MODELS = ["rf", "lightgbm", "tabpfn", "real-tabpfn", "tabicl", "tabdpt", "tabnet", "mlp"]
DEFAULT_REF_COUNTS = [1, 2, 4, 10]


BASE_MODELS = 4  # standard RMIA models (num_ref=1: 2*(1+1)=4) always preserved


def clean_extra_models(dataset: str, model: str, dry_run: bool = False) -> None:
    """Delete models with index >= BASE_MODELS from metadata and disk, forcing retrain."""
    import json as _json
    models_dir = (
        Path("ml_privacy_meter") / "logs" / dataset / model / "rmia" / "models"
    )
    meta_path = models_dir / "models_metadata.json"
    if not meta_path.exists():
        return
    with open(meta_path) as f:
        d = _json.load(f)
    to_remove = {k: v for k, v in d.items() if int(k) >= BASE_MODELS}
    if not to_remove:
        return
    print(f"[REF] Cleaning models {BASE_MODELS}–{max(int(k) for k in to_remove)} "
          f"for {dataset}/{model} (keeping 0–{BASE_MODELS - 1}).")
    if not dry_run:
        for entry in to_remove.values():
            pkl = entry.get("model_path", "")
            if pkl and Path(pkl).exists():
                Path(pkl).unlink()
        trimmed = {k: v for k, v in d.items() if int(k) < BASE_MODELS}
        with open(meta_path, "w") as f:
            _json.dump(trimmed, f, indent=4)


def ref_audit_done(dataset: str, model: str, k: int) -> bool:
    exp_dir = (
        Path("ml_privacy_meter") / "logs" / dataset / model / "rmia"
        / f"report_ref{k}" / "exp"
    )
    return exp_dir.exists() and any(exp_dir.iterdir())


def models_exist(dataset: str, model: str, min_models: int) -> bool:
    import json
    meta = Path("ml_privacy_meter") / "logs" / dataset / model / "rmia" / "models" / "models_metadata.json"
    if not meta.exists():
        return False
    with open(meta) as f:
        return len(json.load(f)) >= min_models


def discover_datasets() -> list:
    logs_dir = Path("ml_privacy_meter") / "logs"
    if not logs_dir.exists():
        return []
    return sorted(p.name for p in logs_dir.iterdir() if p.is_dir())


def parse_list_arg(raw: str, all_values: list) -> list:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if items == ["all"]:
        return all_values
    resolved = []
    for item in items:
        if item == "all":
            resolved.extend(all_values)
        else:
            resolved.append(item)
    return sorted(set(resolved))


def run_cmd(cmd: list, label: str, dry_run: bool) -> int:
    print(f"\n{'='*60}")
    print(f"[REF] {label}")
    print("  " + " ".join(cmd))
    print("=" * 60)
    if dry_run:
        return 0
    result = subprocess.run(cmd, check=False)
    return result.returncode


def run_one(dataset: str, model: str, ref_counts: list, max_ref: int,
            skip_train: bool, skip_existing: bool, gpu: str, dry_run: bool,
            continue_on_error: bool) -> list:
    """Run the full reference analysis for one dataset × model. Returns list of failed k values."""
    base_cmd = [sys.executable, "rmia.py", "--dataset", dataset, "--model", model]
    if gpu:
        base_cmd += ["--gpu", gpu]

    # If all audits are already done, skip everything.
    # If some audits are done, models/signals are already on disk — skip training too.
    if skip_existing and not dry_run:
        pending_ks = [k for k in ref_counts if not ref_audit_done(dataset, model, k)]
        if not pending_ks:
            print(f"[REF] All {len(ref_counts)} audits already done for {dataset}/{model} — skipping.")
            return []
        if len(pending_ks) < len(ref_counts):
            # Only skip training if the full model set is actually on disk.
            skip_train = models_exist(dataset, model, 2 * (max_ref + 1))
            if not skip_train:
                print(f"[REF] Some audits done but models are incomplete for {dataset}/{model} — will retrain.")

    # ── Step 1: train with the maximum reference count ──────────────────────
    n_expected = 2 * (max_ref + 1)
    if skip_train or (skip_existing and models_exist(dataset, model, n_expected)):
        print(f"[REF] Skipping training for {dataset}/{model} "
              f"({n_expected} models already exist).")
    else:
        if not skip_existing:
            clean_extra_models(dataset, model, dry_run)
        train_cmd = base_cmd + ["--mode", "train", "--num-ref", str(max_ref)]
        label = f"TRAIN  dataset={dataset} model={model} num_ref={max_ref}"
        rc = run_cmd(train_cmd, label, dry_run)
        if rc != 0 and not dry_run:
            print(f"[REF] Training failed (exit {rc}) for {dataset}/{model}.")
            if not continue_on_error:
                sys.exit(rc)
            return list(ref_counts)  # mark all k as failed

    # ── Step 2: audit-only for each k ───────────────────────────────────────
    failed = []
    for k in ref_counts:
        if skip_existing and ref_audit_done(dataset, model, k) and not dry_run:
            print(f"[REF] Skipping audit num_ref={k} for {dataset}/{model} (already done).")
            continue
        audit_cmd = base_cmd + ["--mode", "load", "--num-ref", str(k)]
        label = f"AUDIT  dataset={dataset} model={model} num_ref={k}  → report_ref{k}/"
        rc = run_cmd(audit_cmd, label, dry_run)
        if rc != 0 and not dry_run:
            print(f"[REF] Audit for num_ref={k} failed (exit {rc}). Continuing.")
            failed.append(k)

    return failed


def main():
    parser = argparse.ArgumentParser(
        description="Reference model analysis: vary RMIA reference-model count across datasets and models."
    )

    parser.add_argument(
        "--datasets", required=True,
        help="Dataset name, comma-separated list, or 'all' (e.g. locations  or  locations,purchases10  or  all)",
    )
    parser.add_argument(
        "--models", required=True,
        help=f"Model name, comma-separated list, or 'all' (e.g. rf  or  rf,lightgbm  or  all). Supported: {SUPPORTED_MODELS}",
    )

    parser.add_argument(
        "--max-ref", type=int, default=10,
        help="Maximum number of reference models to train (default: 10). "
             "Trains 2*(max_ref+1) models in total. "
             "All --ref-counts values must be ≤ this.",
    )
    parser.add_argument(
        "--ref-counts", type=str, default=None,
        help=f"Comma-separated reference counts to audit (default: {','.join(map(str, DEFAULT_REF_COUNTS))}).",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training; assume all models for --max-ref are already trained.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip training if enough models already exist; skip audit for k values that already have results.",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Keep going if one dataset/model combination fails.",
    )
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU ids, e.g. '0' or '0,1'. "
             "With multiple ids, dataset×model combinations run in parallel (one per GPU).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing.",
    )
    args = parser.parse_args()

    # Resolve datasets and models
    datasets = parse_list_arg(args.datasets, discover_datasets())
    models = parse_list_arg(args.models, SUPPORTED_MODELS)

    if not datasets:
        parser.error("No datasets found. Check ml_privacy_meter/logs/ or ml_privacy_meter/configs/.")

    ref_counts = (
        sorted({int(x.strip()) for x in args.ref_counts.split(",")})
        if args.ref_counts
        else DEFAULT_REF_COUNTS
    )

    if max(ref_counts) > args.max_ref:
        parser.error(f"All --ref-counts values must be ≤ --max-ref ({args.max_ref}). "
                     f"Got: {max(ref_counts)}")

    gpu_list = [g.strip() for g in args.gpus.split(",")] if args.gpus else None
    combos = [(ds, m) for ds in datasets for m in models]
    print(f"[REF] {len(combos)} combination(s): {len(datasets)} dataset(s) × {len(models)} model(s)")
    print(f"[REF] ref counts: {ref_counts}  (train up to max_ref={args.max_ref})")
    if gpu_list:
        print(f"[REF] GPUs: {gpu_list} — running {len(gpu_list)} combination(s) in parallel")

    all_failed = {}

    if gpu_list and not args.dry_run:
        gpu_queue: Queue = Queue()
        for g in gpu_list:
            gpu_queue.put(g)

        def run_job(combo):
            dataset, model = combo
            gpu = gpu_queue.get()
            try:
                failed_ks = run_one(
                    dataset, model, ref_counts, args.max_ref,
                    args.skip_train, args.skip_existing, gpu, args.dry_run, args.continue_on_error,
                )
                return (dataset, model), failed_ks
            finally:
                gpu_queue.put(gpu)

        with ThreadPoolExecutor(max_workers=len(gpu_list)) as executor:
            futures = {executor.submit(run_job, combo): combo for combo in combos}
            for future in as_completed(futures):
                (dataset, model), failed_ks = future.result()
                if failed_ks:
                    all_failed[(dataset, model)] = failed_ks
                    if not args.continue_on_error:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
    else:
        for dataset, model in combos:
            failed_ks = run_one(
                dataset, model, ref_counts, args.max_ref,
                args.skip_train, args.skip_existing,
                gpu_list[0] if gpu_list else None,
                args.dry_run, args.continue_on_error,
            )
            if failed_ks:
                all_failed[(dataset, model)] = failed_ks

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if args.dry_run:
        print("[REF] Dry run complete — no commands were executed.")
    elif all_failed:
        print("[REF] Done with errors:")
        for (ds, m), ks in all_failed.items():
            print(f"  {ds}/{m}: failed ref counts {ks}")
    else:
        print("[REF] All done. Results in:")
        for dataset, model in combos:
            log_base = Path("ml_privacy_meter") / "logs" / dataset / model / "rmia"
            for k in ref_counts:
                print(f"  {log_base}/report_ref{k}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
