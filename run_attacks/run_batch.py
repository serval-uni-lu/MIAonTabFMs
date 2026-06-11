"""Unified batch launcher for all MIA attacks.

Usage examples:
  python run_batch.py --attack rmia --datasets all --models all --mode train
  python run_batch.py --attack lira --datasets all --models lightgbm,rf
  python run_batch.py --attack qmia --dataset purchases10 --model mlp
  python run_batch.py --attack rmia --models lightgbm --proxy-models tabpfn,rf
  python run_batch.py --attack rmia --gpus 0,1 --mode train
"""

import argparse
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from queue import Queue
from typing import List, Optional

import pandas as pd

SUPPORTED_MODELS = ["rf", "lightgbm", "tabpfn", "real-tabpfn", "tabicl", "tabdpt", "tabnet", "mlp"]
SUPPORTED_ATTACKS = ["rmia", "lira", "loss", "population", "qmia"]
SCRIPT_DIR = Path(__file__).resolve().parent
TABPFN_MAX_SAMPLES = 50_000
TABPFN_MAX_FEATURES = 2_000
TABICL_MAX_SAMPLES = 100_000
TABICL_MAX_FEATURES = 2_000
DATA_DIR_CANDIDATES = [
    Path("data/original"),
    Path("data/data_tabarena"),
]
DEFAULT_MODES = {"rmia": "train", "lira": "load", "loss": None, "population": "load", "qmia": "load"}


def parse_csv_arg(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def is_allowed_dataset(dataset_name: str) -> bool:
    if re.match(r"^purchases\d+$", dataset_name):
        return dataset_name == "purchases10"
    return True


def _discover_datasets(data_dir: Path) -> List[str]:
    if not data_dir.exists():
        return []
    return sorted(p.stem for p in data_dir.glob("*.csv"))


def discover_original_datasets() -> List[str]:
    return _discover_datasets(Path("data/original"))


def discover_datasets_from_logs() -> List[str]:
    logs_dir = Path("ml_privacy_meter") / "logs"
    if not logs_dir.exists():
        return []
    return sorted(p.name for p in logs_dir.iterdir() if p.is_dir())


def resolve_dataset_path(dataset_name: str) -> Optional[Path]:
    for candidate_dir in DATA_DIR_CANDIDATES:
        candidate_path = candidate_dir / f"{dataset_name}.csv"
        if candidate_path.exists():
            return candidate_path
    return None


def get_num_features(dataset_path: Path) -> int:
    if not dataset_path.exists():
        return -1
    sample = pd.read_csv(dataset_path, header=None, nrows=1)
    return max(sample.shape[1] - 1, 0)


@lru_cache(maxsize=None)
def get_num_samples(dataset_path_str: str) -> int:
    dataset_path = Path(dataset_path_str)
    if not dataset_path.exists():
        return -1
    with dataset_path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def should_skip_combo(dataset_name: str, model_name: str, dataset_path: Path) -> bool:
    if model_name not in {"tabpfn", "real-tabpfn", "tabicl"}:
        return False

    num_features = get_num_features(dataset_path)
    num_samples = get_num_samples(str(dataset_path))

    if model_name in {"tabpfn", "real-tabpfn"}:
        max_samples, max_features = TABPFN_MAX_SAMPLES, TABPFN_MAX_FEATURES
    else:
        max_samples, max_features = TABICL_MAX_SAMPLES, TABICL_MAX_FEATURES

    if num_samples > max_samples:
        print(f"[SKIP] {dataset_name} + {model_name}: {num_samples} samples exceeds {model_name.upper()} limit ({max_samples}).")
        return True
    if num_features > max_features:
        print(f"[SKIP] {dataset_name} + {model_name}: {num_features} features exceeds {model_name.upper()} limit ({max_features}).")
        return True
    return False



def _all_exist(directory: Path, *filenames: str) -> bool:
    return all((directory / f).exists() for f in filenames)


def _report_done(report_dir: Path) -> bool:
    exp_dir = report_dir / "exp"
    return (report_dir / "attack_result_average.csv").exists() or (exp_dir.exists() and any(exp_dir.iterdir()))


def rmia_signals_exist(dataset_name: str, model_name: str) -> bool:
    sig_dir = Path("ml_privacy_meter") / "logs" / dataset_name / model_name / "rmia" / "signals"
    return _all_exist(sig_dir, "rmia_signals.npy", "rmia_signals_pop.npy")


def rmia_models_exist(dataset_name: str, model_name: str, seed: Optional[str] = None) -> bool:
    run_subdir = Path(f"seed{seed}") / "rmia" if seed is not None else Path("rmia")
    models_dir = Path("ml_privacy_meter") / "logs" / dataset_name / model_name / run_subdir / "models"
    return models_dir.exists() and any(models_dir.glob("*.pkl"))


def rmia_seed_summary_exists(dataset_name: str, model_name: str) -> bool:
    summary_path = Path("ml_privacy_meter") / "logs" / dataset_name / model_name / "attack_result_seed_summary.csv"
    return summary_path.exists()


def rmia_any_seeded_models_exist(dataset_name: str, model_name: str) -> bool:
    base = Path("ml_privacy_meter") / "logs" / dataset_name / model_name
    if not base.exists():
        return False
    for models_dir in base.glob("seed*/rmia/models"):
        if any(models_dir.glob("*.pkl")):
            return True
    return False


def rmia_seed_online_done(dataset_name: str, model_name: str, seeds: str) -> bool:
    return all(
        _report_done(
            Path("ml_privacy_meter") / "logs" / dataset_name / model_name / f"seed{s}" / "rmia" / "report_online"
        )
        for s in parse_csv_arg(seeds)
    )


def resolve_rmia_mode(dataset_name: str, model_name: str, requested_mode: str, seeds: Optional[str] = None) -> str:
    if requested_mode in ("load", "signal") and seeds:
        missing = [seed for seed in parse_csv_arg(seeds) if not rmia_models_exist(dataset_name, model_name, seed)]
        if missing:
            print(f"[AUTO] {dataset_name} + {model_name}: no seeded models found for seeds {missing} — switching mode={requested_mode} to mode=train")
            return "train"
    elif requested_mode in ("load", "signal") and not rmia_models_exist(dataset_name, model_name):
        print(f"[AUTO] {dataset_name} + {model_name}: no models found — switching mode={requested_mode} to mode=train")
        return "train"
    return requested_mode


def run_already_completed(
    attack: str,
    dataset_name: str,
    model_name: str,
    online: bool = False,
    proxy_model: Optional[str] = None,
    seed: Optional[int] = None,
) -> bool:
    base = Path("ml_privacy_meter") / "logs" / dataset_name / model_name
    if seed is not None:
        base = base / f"seed{seed}"

    if attack == "rmia":
        sig_dir = base / "rmia" / "signals"
        if not _all_exist(sig_dir, "rmia_signals.npy", "rmia_signals_pop.npy"):
            return False
        report_subdir = "report_online" if online else "report"
        if proxy_model is not None:
            return _report_done(base / "rmia_proxy" / proxy_model / "report")
        return _report_done(base / "rmia" / report_subdir)

    if attack == "lira":
        if not _all_exist(base / "lira" / "signals", "lira_signals.npy"):
            return False
        report_subdir = "report_online" if online else "report"
        return _report_done(base / "lira" / report_subdir)

    if attack == "loss":
        sig_dir = base / "loss" / "signals"
        if not _all_exist(sig_dir, "loss_signals.npy"):
            return False
        return _report_done(base / "loss" / "report")

    if attack == "population":
        sig_dir = base / "attack_p" / "signals"
        if not _all_exist(sig_dir, "population_signals.npy", "population_signals_pop.npy"):
            return False
        return _report_done(base / "attack_p" / "report")

    if attack == "qmia":
        sig_dir = base / "quantile_reg" / "signals"
        if _all_exist(sig_dir, "rmia_signals.npy", "rmia_signals_pop.npy"):
            return True
        return _report_done(base / "quantile_reg" / "report")

    return False


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

ATTACK_SCRIPTS = {
    "rmia": "rmia.py",
    "lira": "lira.py",
    "loss": "loss.py",
    "population": "population_attack.py",
    "qmia": "quantile_regression.py",
}


def run_one(
    attack: str,
    dataset_name: str,
    model_name: str,
    dry_run: bool,
    mode: Optional[str] = None,
    skip_config: bool = False,
    skip_existing: bool = False,
    online: bool = False,
    gpu: Optional[str] = None,
    proxy_model: Optional[str] = None,
    defense: str = "none",
    max_audit_samples: Optional[int] = None,
    seeds: Optional[str] = None,
    seed: Optional[int] = None,
) -> int:
    cmd = [sys.executable, str(SCRIPT_DIR / ATTACK_SCRIPTS[attack]), "--dataset", dataset_name, "--model", model_name]

    if mode is not None and attack != "loss":
        # Non-training attacks use "signal" instead of "train" (they reuse RMIA models).
        effective_mode = "signal" if (mode == "train" and attack != "rmia") else mode
        cmd += ["--mode", effective_mode]
    if skip_config:
        cmd += ["--skip-config"]
    if skip_existing and attack == "rmia":
        cmd += ["--skip-existing"]
    if online and attack in ("rmia", "lira"):
        cmd += ["--online"]
    if gpu and attack == "rmia":
        cmd += ["--gpu", gpu]
    if proxy_model and attack == "rmia":
        cmd += ["--proxy-model", proxy_model]
    if seeds and attack == "rmia":
        cmd += ["--seeds", seeds]
    if seed is not None and attack != "rmia":
        cmd += ["--seed", str(seed)]
    if defense != "none":
        cmd += ["--defense", defense]
    if max_audit_samples is not None and attack == "qmia":
        cmd += ["--max-audit-samples", str(max_audit_samples)]

    label = f"[RUN] attack={attack} dataset={dataset_name} model={model_name}"
    if mode:
        label += f" mode={mode}"
    if online:
        label += " (online)"
    if proxy_model:
        label += f" proxy={proxy_model}"
    if seeds:
        label += f" seeds={seeds}"
    if seed is not None:
        label += f" seed={seed}"
    if gpu:
        label += f" gpu={gpu}"
    print(label)
    print("      " + " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified batch launcher for all MIA attacks.")
    parser.add_argument("--attack", type=str, required=True, choices=SUPPORTED_ATTACKS,
                        help=f"Attack to run: {', '.join(SUPPORTED_ATTACKS)}.")
    parser.add_argument("--datasets", type=str, default="all",
                        help="Comma-separated datasets or 'all' to auto-discover.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset (overrides --datasets).")
    parser.add_argument("--models", type=str, default="all",
                        help=f"Comma-separated models or 'all' ({', '.join(SUPPORTED_MODELS)}).")
    parser.add_argument("--model", type=str, default=None,
                        help="Single model (overrides --models).")
    parser.add_argument("--mode", type=str, default=None, choices=["train", "load", "signal"],
                        help="'train' trains models + computes signals; 'signal' loads models + recomputes signals; 'load' reuses existing. Default varies by attack.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip already completed combinations.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue if a run fails.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--online", action="store_true",
                        help="Online mode (rmia/lira only).")
    parser.add_argument("--defense", type=str, default="none", choices=["none", "hamp"],
                        help="Test-time defense (rmia/lira/loss/population/qmia).")
    parser.add_argument("--proxy-models", type=str, default=None,
                        help="Comma-separated proxy model names or 'all' (rmia only).")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs for parallel execution (rmia only).")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for repeated default RMIA runs, e.g. 1,2,3,4,5.")
    parser.add_argument("--max-audit-samples", type=int, default=None,
                        help="Cap audit dataset size (qmia only).")
    parser.add_argument("--skip-config", action="store_true",
                        help="Skip rewriting the YAML config if it already exists.")

    args = parser.parse_args()
    attack = args.attack
    mode = args.mode if args.mode is not None else DEFAULT_MODES[attack]
    if attack == "rmia" and args.defense != "none" and args.seeds is None:
        args.seeds = "1"

    # --- Resolve datasets ---
    dataset_items = parse_csv_arg(args.datasets)
    dataset_items_lower = [item.lower() for item in dataset_items]
    # RMIA trains from scratch -> discover from data files; others need existing RMIA runs -> logs
    discover_fn = discover_original_datasets if attack == "rmia" else discover_datasets_from_logs

    if args.dataset:
        datasets = [args.dataset]
    elif "all" in dataset_items_lower:
        explicit_items = [item for item in dataset_items if item.lower() != "all"]
        discovered = discover_fn()
        datasets = sorted(set(discovered + explicit_items))
    else:
        datasets = dataset_items

    # --- Resolve models ---
    if args.model:
        models = [args.model.lower()]
    elif args.models.lower() == "all":
        models = SUPPORTED_MODELS
    else:
        models = [m.lower() for m in parse_csv_arg(args.models)]

    invalid_models = [m for m in models if m not in SUPPORTED_MODELS]
    if invalid_models:
        raise ValueError(f"Unsupported model(s): {invalid_models}. Supported: {SUPPORTED_MODELS}")

    if not datasets:
        raise ValueError("No datasets found/provided.")

    filtered = [d for d in datasets if is_allowed_dataset(d)]
    for d in datasets:
        if d not in filtered:
            print(f"[SKIP] {d}: only purchases10 is enabled for purchases variations.")
    datasets = filtered

    if not datasets:
        raise ValueError("No datasets left after applying purchases filter.")

    # --- Resolve proxy models ---
    proxy_models = None
    if args.proxy_models is not None and attack == "rmia":
        raw = args.proxy_models.strip().lower()
        proxy_models = SUPPORTED_MODELS if raw == "all" else [m.strip().lower() for m in raw.split(",") if m.strip()]
        invalid_proxy = [m for m in proxy_models if m not in SUPPORTED_MODELS]
        if invalid_proxy:
            raise ValueError(f"Unsupported proxy model(s): {invalid_proxy}.")

    gpu_list = [g.strip() for g in args.gpus.split(",")] if args.gpus and attack == "rmia" else None
    if args.seeds is not None and args.proxy_models is not None:
        raise ValueError("--seeds is not supported together with --proxy-models.")

    print(f"Attack:   {attack}")
    print(f"Datasets: ({len(datasets)}): {datasets}")
    print(f"Models:   ({len(models)}): {models}")
    if proxy_models:
        print(f"Proxy models ({len(proxy_models)}): {proxy_models}")
    if args.seeds:
        print(f"Seeds:    {args.seeds}")

    # --- Build job list ---
    jobs = []  # (dataset_name, model_name, effective_mode, proxy_model_or_None)
    planned = 0

    for dataset_name in datasets:
        dataset_path = resolve_dataset_path(dataset_name)
        if dataset_path is None:
            print(f"[SKIP] dataset file missing for: {dataset_name}")
            continue

        for model_name in models:
            planned += 1
            if should_skip_combo(dataset_name, model_name, dataset_path):
                continue

            if proxy_models:
                for proxy_model in proxy_models:
                    if proxy_model == model_name:
                        continue
                    if attack == "rmia":
                        if not rmia_signals_exist(dataset_name, model_name):
                            print(f"[SKIP] proxy: target signals missing for {dataset_name} + {model_name}")
                            continue
                        if not rmia_signals_exist(dataset_name, proxy_model):
                            print(f"[SKIP] proxy: proxy signals missing for {dataset_name} + {proxy_model}")
                            continue
                    if args.skip_existing and run_already_completed(attack, dataset_name, model_name, proxy_model=proxy_model):
                        print(f"[SKIP] already completed: {dataset_name} + {model_name} -> proxy {proxy_model}")
                        continue
                    jobs.append((dataset_name, model_name, "load", proxy_model, None))
            else:
                if args.skip_existing and args.seeds and attack == "rmia":
                    if args.online and rmia_seed_online_done(dataset_name, model_name, args.seeds):
                        print(f"[SKIP] seed online already completed: {dataset_name} + {model_name}")
                        continue
                    elif not args.online and rmia_seed_summary_exists(dataset_name, model_name):
                        print(f"[SKIP] seed summary already completed: {dataset_name} + {model_name}")
                        continue
                if args.skip_existing and not args.seeds and run_already_completed(attack, dataset_name, model_name, online=args.online):
                    print(f"[SKIP] already completed: {dataset_name} + {model_name}")
                    continue
                if attack == "rmia" and not args.skip_existing and mode == "train" and not args.online:
                    base_dir = Path("ml_privacy_meter") / "logs" / dataset_name / model_name
                    if args.seeds:
                        for seed in parse_csv_arg(args.seeds):
                            seed_dir = base_dir / f"seed{seed}"
                            if seed_dir.exists():
                                print(f"[CLEAN] removing {seed_dir}")
                                if not args.dry_run:
                                    shutil.rmtree(seed_dir)
                        for summary_name in ("attack_result_seed_runs.csv", "attack_result_seed_summary.csv"):
                            summary_path = base_dir / summary_name
                            if summary_path.exists():
                                print(f"[CLEAN] removing {summary_path}")
                                if not args.dry_run:
                                    summary_path.unlink()
                    else:
                        log_dir = base_dir / "rmia"
                        if log_dir.exists():
                            print(f"[CLEAN] removing {log_dir}")
                            if not args.dry_run:
                                shutil.rmtree(log_dir)
                effective_mode = resolve_rmia_mode(dataset_name, model_name, mode, args.seeds) if attack == "rmia" else mode
                if args.seeds and attack != "rmia":
                    for seed_str in parse_csv_arg(args.seeds):
                        seed_int = int(seed_str)
                        if args.skip_existing and run_already_completed(attack, dataset_name, model_name, online=args.online, seed=seed_int):
                            print(f"[SKIP] already completed: {dataset_name} + {model_name} seed={seed_int}")
                            continue
                        jobs.append((dataset_name, model_name, effective_mode, None, seed_int))
                else:
                    jobs.append((dataset_name, model_name, effective_mode, None, None))

    # --- Execute ---
    failures = []
    executed = 0

    def _run_job(job, gpu=None):
        ds, mdl, eff_mode, proxy, job_seed = job
        return run_one(
            attack, ds, mdl, args.dry_run,
            mode=eff_mode,
            skip_config=args.skip_config,
            skip_existing=args.skip_existing,
            online=args.online,
            gpu=gpu,
            proxy_model=proxy,
            defense=args.defense,
            max_audit_samples=args.max_audit_samples,
            seeds=args.seeds if attack == "rmia" else None,
            seed=job_seed,
        )

    if gpu_list:
        gpu_queue: Queue = Queue()
        for g in gpu_list:
            gpu_queue.put(g)

        def run_gpu_job(job):
            gpu = gpu_queue.get()
            try:
                ds, mdl, _, proxy, _ = job
                code = _run_job(job, gpu=gpu)
                return ds, mdl, proxy, code
            finally:
                gpu_queue.put(gpu)

        with ThreadPoolExecutor(max_workers=len(gpu_list)) as executor:
            futures = {executor.submit(run_gpu_job, job): job for job in jobs}
            for future in as_completed(futures):
                ds, mdl, proxy, code = future.result()
                executed += 1
                if code != 0:
                    failures.append((ds, mdl, proxy, code))
                    label = f"dataset={ds} model={mdl}" + (f" proxy={proxy}" if proxy else "")
                    print(f"[FAIL] {label} exit_code={code}")
                    if not args.continue_on_error:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
    else:
        for job in jobs:
            ds, mdl, _, proxy, _ = job
            code = _run_job(job)
            executed += 1
            if code != 0:
                failures.append((ds, mdl, proxy, code))
                label = f"dataset={ds} model={mdl}" + (f" proxy={proxy}" if proxy else "")
                print(f"[FAIL] {label} exit_code={code}")
                if not args.continue_on_error:
                    break

    print(f"Planned combinations: {planned}")
    print(f"Executed combinations: {executed}")

    if failures:
        print("Failures:")
        for ds, mdl, proxy, code in failures:
            label = f"{ds} + {mdl}" + (f" -> proxy {proxy}" if proxy else "")
            print(f"  - {label} (exit {code})")
        raise SystemExit(1)

    print(f"All requested {attack.upper()} runs completed successfully.")


if __name__ == "__main__":
    main()
