import os
import re
import subprocess
import sys


_SENTINEL = ".setup_done"


def remove_cuda_packages(requirements_file: str):
    """Removes CUDA-specific packages from a requirements file in-place."""
    if not os.path.exists(requirements_file):
        print(f"{requirements_file} not found, skipping CUDA cleanup.")
        return

    cuda_pattern = re.compile(
        r"(?i)(cupy|cuda|torch.*cu\d|torchvision.*cu\d|torchaudio.*cu\d|nvidia-)",
    )

    with open(requirements_file, "r") as f:
        lines = f.readlines()

    cleaned = [line for line in lines if not cuda_pattern.search(line)]

    if len(cleaned) < len(lines):
        removed = len(lines) - len(cleaned)
        with open(requirements_file, "w") as f:
            f.writelines(cleaned)
        print(f"Removed {removed} CUDA package(s) from {requirements_file}.")
    else:
        print("No CUDA packages found in requirements.txt.")


def clone_and_setup_repo(user_name: str, repo_name: str = "ml_privacy_meter"):
    """
    Clones a forked GitHub repo and sets up the environment path.
    The full setup (clone + install) only runs once; subsequent calls
    only ensure sys.path is correct.

    Args:
        user_name (str): Your GitHub username containing the fork.
        repo_name (str): The repository name (default: ml_privacy_meter)
    """
    clone_dir = os.path.join(os.getcwd(), repo_name)
    sentinel = os.path.join(clone_dir, _SENTINEL)

    # Always ensure the repo is on sys.path, regardless of setup state.
    if clone_dir not in sys.path:
        sys.path.append(clone_dir)

    # If setup was already completed, skip everything.
    if os.path.exists(sentinel):
        print(f"Repo already set up ({repo_name}). Skipping.")
        os.chdir(clone_dir)
        return

    # --- First-time setup ---

    repo_url = f"https://github.com/{user_name}/{repo_name}.git"

    if not os.path.exists(clone_dir):
        subprocess.run(["git", "clone", repo_url], check=True)
    else:
        print("Repo directory exists but setup not complete. Continuing setup.")

    print(f"Setting up in: {clone_dir}")

    requirements_file = os.path.join(clone_dir, "requirements.txt")

    # Clean CUDA packages from requirements.txt
    remove_cuda_packages(requirements_file)

    # Install dependencies
    try:
        subprocess.run(
            [
                "uv", "pip", "install", "-r", requirements_file,
                "--index-strategy", "unsafe-best-match",
            ],
            check=True,
            cwd=clone_dir,
        )
    except subprocess.CalledProcessError as e:
        print("uv pip install failed.")
        print(e)
        return  # Don't write sentinel if install failed

    # Mark setup as complete
    open(sentinel, "w").close()
    print(f"Setup complete. Sentinel written to {sentinel}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clone and set up ml_privacy_meter.")
    parser.add_argument("user_name", help="GitHub username with the forked repo")
    parser.add_argument("--repo", default="ml_privacy_meter", help="Repo name (default: ml_privacy_meter)")
    args = parser.parse_args()

    clone_and_setup_repo(args.user_name, args.repo)
