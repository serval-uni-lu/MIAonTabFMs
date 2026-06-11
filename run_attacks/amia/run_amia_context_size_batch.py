"""
Batch runner for AMIA-internal context-size sweep on TabPFN / Real-TabPFN.

Calls amia_tabpfn.py repeatedly with --context-pct, sweeping over
percentages from 5 to 100.  Unlike the SDPA-hook variant (run_amia_context_size_batch.py),
this script allows 100% as a valid percentage (reads from rmia/, writes to amia_internal/).

Pre-requisites
--------------
For pct < 100 : rmia_ctx<pct>/ must exist (produced by run_context_size_batch.py).
For pct == 100: rmia/ must exist (produced by a regular rmia.py run).

Usage
-----
    uv run run_attacks/amia/run_amia_context_size_batch.py --dataset locations --model tabpfn

    uv run run_attacks/amia/run_amia_context_size_batch.py \\
        --datasets all --models all --context-pcts 5,10,20,30,40,50,60,70,80,90,100 \\
        --continue-on-error
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

SUPPORTED_MODELS    = ["tabpfn", "real-tabpfn", "tabicl", "tabdpt"]
DEFAULT_CONTEXT_PCTS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

_SCRIPT_DIR = Path(__file__).resolve().parent
AMIA_SCRIPTS = {
    "tabpfn":      _SCRIPT_DIR / "amia_tabpfn.py",
    "real-tabpfn": _SCRIPT_DIR / "amia_tabpfn.py",
    "tabicl":      _SCRIPT_DIR / "amia_tabicl.py",
    "tabdpt":      _SCRIPT_DIR / "amia_tabdpt.py",
}


def parse_pcts(raw: str) -> List[float]:
    return [float(s.strip()) for s in raw.split(",") if s.strip()]


def discover_datasets() -> List[str]:
    log_root = Path("ml_privacy_meter") / "logs"
    if not log_root.exists():
        return []
    return sorted(p.name for p in log_root.iterdir() if p.is_dir())


def resolve_datasets(raw: str) -> List[str]:
    items = [s.strip() for s in raw.split(",") if s.strip()]
    if "all" in [s.lower() for s in items]:
        explicit = [s for s in items if s.lower() != "all"]
        return sorted(set(discover_datasets() + explicit))
    return items


def rmia_dir_for(dataset: str, model: str, pct: float, seed: int = None) -> Path:
    base = Path("ml_privacy_meter") / "logs" / dataset / model.lower()
    if pct >= 100.0:
        if seed is not None:
            return base / f"seed{seed}" / "rmia"
        return base / "rmia"
    subdir = f"rmia_ctx{int(pct)}"
    if seed is not None:
        return base / f"seed{seed}" / subdir
    return base / subdir


def run_already_completed(dataset: str, model: str, pct: float, seed: int = None) -> bool:
    base = Path("ml_privacy_meter") / "logs" / dataset / model.lower()
    seed_root = base / f"seed{seed}" if seed is not None else base
    if pct >= 100.0:
        subdir = "amia"
    else:
        subdir = f"amia_ctx{int(pct)}"
    exp_dir = seed_root / subdir / "report" / "exp"
    return (exp_dir / "attention_summary.csv").exists()


def run_one(dataset: str, model: str, pct: float,
            dry_run: bool, gpu: str = None, batch_size: int = 200,
            seed: int = None) -> int:
    script = AMIA_SCRIPTS[model]
    cmd = [
        sys.executable, str(script),
        "--dataset",     dataset,
        "--context-pct", str(pct),
        "--batch-size",  str(batch_size),
    ]
    if model in ("tabpfn", "real-tabpfn"):
        cmd += ["--model", model]
        if seed is not None:
            cmd += ["--seed", str(seed)]
    else:
        cmd += ["--skip-config"]
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
        description="Sweep AMIA-internal over context-size percentages (5–100) "
                    "for TabPFN / Real-TabPFN.",
    )
    parser.add_argument("--datasets", type=str, default="all",
                        help="Comma-separated datasets or 'all' to auto-discover from logs/.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset (overrides --datasets).")
    parser.add_argument("--models", type=str, default="all",
                        help=f"Comma-separated models or 'all' ({', '.join(SUPPORTED_MODELS)}).")
    parser.add_argument("--model", type=str, default=None, choices=SUPPORTED_MODELS,
                        help="Single model (overrides --models).")
    parser.add_argument("--context-pcts", type=str,
                        default=",".join(str(p) for p in DEFAULT_CONTEXT_PCTS),
                        help="Comma-separated percentages in (0, 100]. "
                             "100 reads from rmia/ (full context baseline).")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Pool samples per forward pass (default: 200).")
    parser.add_argument("--gpu", type=str, default=None,
                        help="CUDA device index, e.g. '0'.")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seed integers (e.g. '1,2,3,4,5'). Default: None (unseeded).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip percentages that already have outputs.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue to the next run if one fails.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned commands without executing.")
    args = parser.parse_args()

    pcts = parse_pcts(args.context_pcts)
    invalid = [p for p in pcts if not (0 < p <= 100)]
    if invalid:
        raise ValueError(f"Context percentages must be in (0, 100]: {invalid}")

    datasets = [args.dataset] if args.dataset else resolve_datasets(args.datasets)
    if not datasets:
        raise ValueError("No datasets found.")

    if args.model:
        models = [args.model]
    else:
        raw    = args.models.strip().lower()
        models = SUPPORTED_MODELS if raw == "all" else [m.strip() for m in raw.split(",")]
        bad    = [m for m in models if m not in SUPPORTED_MODELS]
        if bad:
            raise ValueError(f"Unsupported model(s): {bad}. Supported: {SUPPORTED_MODELS}")

    seeds_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else [None]

    print(f"Datasets ({len(datasets)}): {datasets}")
    print(f"Models   ({len(models)}): {models}")
    print(f"Context percentages: {pcts}")
    print(f"Seeds: {seeds_list}")

    failures = []
    for dataset in datasets:
        for model in models:
            model_seeds = seeds_list
            for seed in model_seeds:
                for pct in pcts:
                    rdir = rmia_dir_for(dataset, model, pct, seed)
                    if not rdir.exists():
                        src = "rmia/" if pct >= 100.0 else f"rmia_ctx{int(pct)}/"
                        seed_label = f" seed={seed}" if seed is not None else ""
                        print(f"[SKIP] no {src} for {dataset}/{model}{seed_label} — run RMIA first")
                        continue
                    if args.skip_existing and run_already_completed(dataset, model, pct, seed):
                        seed_label = f" seed={seed}" if seed is not None else ""
                        print(f"[SKIP] already completed: {dataset} + {model} @ {pct}%{seed_label}")
                        continue
                    code = run_one(dataset, model, pct,
                                   args.dry_run, args.gpu, args.batch_size, seed)
                    if code != 0:
                        failures.append((dataset, model, pct, seed))
                        seed_label = f" seed={seed}" if seed is not None else ""
                        print(f"[FAIL] {dataset} + {model} @ {pct}%{seed_label} exit_code={code}")
                        if not args.continue_on_error:
                            raise SystemExit(1)

    if failures:
        print("\nFailed runs:")
        for dataset, model, pct, seed in failures:
            seed_label = f" seed={seed}" if seed is not None else ""
            print(f"  - {dataset} + {model} @ {pct}%{seed_label}")
        raise SystemExit(1)

    print("\nAll AMIA-internal context-size runs completed successfully.")


if __name__ == "__main__":
    main()
