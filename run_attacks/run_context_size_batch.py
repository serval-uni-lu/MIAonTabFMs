"""
Batch runner for RMIA context-size sweep on foundation models.

Calls rmia.py repeatedly with --context-pct, sweeping over percentages of the
training pool.  Both data/original and data/data_tabarena datasets are supported.

100% is excluded: those results already exist in the regular rmia/ runs.
Context sweep results are saved under rmia_ctx<pct>/ (e.g. rmia_ctx50/).

Usage:
    uv run run_context_size_batch.py --datasets locations --models tabpfn
    uv run run_context_size_batch.py --datasets all --models all \\
        --context-pcts 5,10,20,30,40,50,60,70,80,90 --mode train --continue-on-error
"""

import argparse
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import List

FOUNDATION_MODELS = ["tabpfn", "real-tabpfn", "tabicl", "tabdpt"]
DEFAULT_CONTEXT_PCTS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
DATA_DIR_CANDIDATES = [
    Path("data/original"),
    Path("data/data_tabarena"),
]
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_pcts(raw: str) -> List[float]:
    return [float(s.strip()) for s in raw.split(",") if s.strip()]


def is_allowed_dataset(dataset_name: str) -> bool:
    if re.match(r"^purchases\d+$", dataset_name):
        return dataset_name == "purchases10"
    return True


def resolve_dataset_path(dataset_name: str) -> Path | None:
    for candidate_dir in DATA_DIR_CANDIDATES:
        candidate = candidate_dir / f"{dataset_name}.csv"
        if candidate.exists():
            return candidate
    return None


def discover_datasets(data_dir: Path) -> List[str]:
    if not data_dir.exists():
        return []
    return sorted(p.stem for p in data_dir.glob("*.csv"))


def resolve_datasets(raw: str) -> List[str]:
    """Expand 'all' and comma-separated dataset names, mirroring run_rmia_batch.py."""
    items = [s.strip() for s in raw.split(",") if s.strip()]
    items_lower = [s.lower() for s in items]
    if "all" in items_lower:
        explicit = [s for s in items if s.lower() != "all"]
        base = discover_datasets(Path("data/original"))
        return sorted(set(base + explicit))
    return items


def log_dir_for(dataset: str, model: str, pct: float, seed: int = None) -> Path:
    base = Path("ml_privacy_meter") / "logs" / dataset / model.lower()
    subdir = f"rmia_ctx{int(pct)}" if pct < 100.0 else "rmia"
    if seed is not None:
        return base / f"seed{seed}" / subdir
    return base / subdir


def run_already_completed(dataset: str, model: str, pct: float, seed: int = None) -> bool:
    report_dir = log_dir_for(dataset, model, pct, seed) / "report"
    exp_dir = report_dir / "exp"
    if not report_dir.exists():
        return False
    if (report_dir / "attack_result_average.csv").exists():
        return True
    if exp_dir.exists() and any(exp_dir.iterdir()):
        return True
    return False


def seed_best_params(dataset: str, model: str, pct: float, dry_run: bool, seed: int = None) -> None:
    """Copy best_params.json from the 100% (rmia/) run into rmia_ctx<pct>/.

    This ensures Optuna-tunable models (rf, lightgbm, mlp, tabnet) reuse the
    hyperparameters found at full context rather than re-tuning per percentage,
    keeping context size as the only variable across the sweep.
    """
    src = Path("ml_privacy_meter") / "logs" / dataset / model.lower() / "rmia" / "best_params.json"
    dst_dir = log_dir_for(dataset, model, pct, seed)
    dst = dst_dir / "best_params.json"

    if not src.exists():
        # Foundation models don't produce best_params.json — nothing to do.
        return

    if dst.exists():
        return  # Already seeded from a previous run.

    seed_label = f"_seed{seed}" if seed is not None else ""
    print(f"[SEED] copying best_params.json from rmia/ → rmia_ctx{int(pct)}{seed_label}/")
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def run_one(dataset: str, model: str, pct: float, mode: str, dry_run: bool,
            gpu: str = None, seed: int = None) -> int:
    cmd = [
        sys.executable, str(SCRIPT_DIR / "rmia.py"),
        "--dataset", dataset,
        "--model", model,
        "--mode", mode,
        "--context-pct", str(pct),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if gpu is not None:
        cmd += ["--gpu", gpu]
    label = f"[RUN] dataset={dataset} model={model} context_pct={pct}%"
    if seed is not None:
        label += f" seed={seed}"
    if gpu:
        label += f" gpu={gpu}"
    print(label)
    print("      " + " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def main():
    parser = argparse.ArgumentParser(
        description="Sweep RMIA over context-size percentages for foundation models."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help=(
            "Comma-separated datasets. Use 'all' to auto-discover from data/original. "
            "Use 'all,<dataset1>,<dataset2>' to run all original datasets plus extras."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Single dataset name (overrides --datasets).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=f"Comma-separated foundation models or 'all' ({', '.join(FOUNDATION_MODELS)}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=FOUNDATION_MODELS,
        help="Single foundation model (overrides --models).",
    )
    parser.add_argument(
        "--context-pcts",
        type=str,
        default=",".join(str(p) for p in DEFAULT_CONTEXT_PCTS),
        help="Comma-separated percentages of the training pool (e.g. 5,10,25,50,100).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "load", "signal"],
        help="Mode passed through to rmia.py.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip percentages that already have RMIA outputs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next percentage if one run fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs for parallel execution (e.g. '0,1,2'). One job per GPU at a time.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed integers (e.g. '1,2,3,4,5'). Default: None (unseeded run, writes to rmia_ctx<pct>/).",
    )
    args = parser.parse_args()

    pcts = parse_pcts(args.context_pcts)
    invalid = [p for p in pcts if not (0 < p <= 100)]
    if invalid:
        raise ValueError(f"Context percentages must be in (0, 100]: {invalid}")
    skipped_100 = [p for p in pcts if p == 100.0]
    pcts = [p for p in pcts if p != 100.0]
    for _ in skipped_100:
        print("[SKIP] context_pct=100 — use run_rmia_batch.py for full-context runs (results in rmia/).")

    # Resolve datasets: --dataset overrides --datasets.
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = resolve_datasets(args.datasets)
    datasets = [d for d in datasets if is_allowed_dataset(d)]
    if not datasets:
        raise ValueError("No datasets found/provided.")

    missing = [d for d in datasets if resolve_dataset_path(d) is None]
    if missing:
        print(f"[WARN] dataset files not found, skipping: {missing}")
        datasets = [d for d in datasets if d not in missing]
    if not datasets:
        raise FileNotFoundError("No dataset files found.")

    # Resolve models: --model overrides --models.
    if args.model:
        models = [args.model]
    else:
        raw = args.models.strip().lower()
        models = FOUNDATION_MODELS if raw == "all" else [m.strip() for m in raw.split(",") if m.strip()]
        invalid_models = [m for m in models if m not in FOUNDATION_MODELS]
        if invalid_models:
            raise ValueError(f"Unsupported model(s): {invalid_models}. Supported: {FOUNDATION_MODELS}")

    seeds_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else [None]

    print(f"Datasets ({len(datasets)}): {datasets}")
    print(f"Models ({len(models)}): {models}")
    print(f"Context percentages: {pcts}")
    print(f"Seeds: {seeds_list}")

    failures = []
    gpu_list = [g.strip() for g in args.gpus.split(",")] if args.gpus else None

    # Build job list up front
    jobs = []
    for dataset in datasets:
        for model in models:
            for seed in seeds_list:
                for pct in pcts:
                    if args.skip_existing and run_already_completed(dataset, model, pct, seed):
                        seed_label = f" seed={seed}" if seed is not None else ""
                        print(f"[SKIP] already completed: {dataset} + {model} @ {pct}%{seed_label}")
                        continue
                    seed_best_params(dataset, model, pct, args.dry_run, seed)
                    jobs.append((dataset, model, pct, seed))

    if gpu_list:
        gpu_queue: Queue = Queue()
        for g in gpu_list:
            gpu_queue.put(g)

        def run_job(job):
            dataset, model, pct, seed = job
            gpu = gpu_queue.get()
            try:
                code = run_one(dataset, model, pct, args.mode, args.dry_run, gpu, seed)
                return dataset, model, pct, seed, code
            finally:
                gpu_queue.put(gpu)

        with ThreadPoolExecutor(max_workers=len(gpu_list)) as executor:
            futures = {executor.submit(run_job, job): job for job in jobs}
            for future in as_completed(futures):
                dataset, model, pct, seed, code = future.result()
                if code != 0:
                    failures.append((dataset, model, pct, seed))
                    print(f"[FAIL] dataset={dataset} model={model} context_pct={pct}% seed={seed} exit_code={code}")
                    if not args.continue_on_error:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
    else:
        for dataset, model, pct, seed in jobs:
            code = run_one(dataset, model, pct, args.mode, args.dry_run, seed=seed)
            if code != 0:
                failures.append((dataset, model, pct, seed))
                print(f"[FAIL] dataset={dataset} model={model} context_pct={pct}% seed={seed} exit_code={code}")
                if not args.continue_on_error:
                    break

    if failures:
        print("\nFailed runs:")
        for dataset, model, pct, seed in failures:
            print(f"  - {dataset} + {model} @ {pct}% seed={seed}")
        raise SystemExit(1)

    print("\nAll context-size runs completed successfully.")


if __name__ == "__main__":
    main()
