"""Run the GitHub tests workflow (.github/workflows/tests.yml) locally, for every OS in its matrix.

Each (os, python-version) job of the matrix is run the way it can be run on this machine:

    ubuntu-*   in an Ubuntu Docker container (one image, a venv per Python version)
    windows-*  natively, in a fresh venv per Python version  (only when this machine runs Windows)
    macos-*    natively, in a fresh venv per Python version  (only when this machine runs macOS)

Jobs for an OS that cannot run here (e.g. macOS from a Windows PC: Docker has no macOS images) are
reported as SKIPPED. Native venvs live in .test-envs/ and are reused between runs.

    python scripts/docker_test.py                       # the whole matrix of tests.yml
    python scripts/docker_test.py --os ubuntu windows   # only some OSes (prefix match)
    python scripts/docker_test.py --python 3.11         # only some Python versions
    python scripts/docker_test.py --fresh               # recreate the native venvs
    python scripts/docker_test.py -- -k mixed -x        # extra arguments after -- go to pytest

The Linux jobs need Docker running (Docker Desktop on Windows / macOS).
Exit code is 0 only if no job failed (skipped jobs do not count as failures).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
IMAGE = "fitfleet-tests"
ENVS = ROOT / ".test-envs"

PASSED, FAILED, SKIPPED = "PASSED", "FAILED", "SKIPPED"
HOST = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")


def workflow_matrix_list(key: str) -> list[str]:
    """A list from the workflow's matrix, e.g. workflow_matrix_list("python-version") -> ["3.11", "3.13"]."""
    match = re.search(rf"^\s*{re.escape(key)}:\s*\[([^\]]*)\]", WORKFLOW.read_text(encoding="utf-8"), re.M)
    if not match:
        sys.exit(f"could not find '{key}: [...]' in {WORKFLOW}")
    return [v.strip().strip("\"'") for v in match.group(1).split(",") if v.strip()]


def os_family(runner: str) -> str:
    """'ubuntu-latest' -> 'linux', 'windows-2022' -> 'windows', 'macos-14' -> 'macos'."""
    name = runner.lower()
    if name.startswith("windows"):
        return "windows"
    if name.startswith("macos"):
        return "macos"
    return "linux"


def banner(text: str) -> None:
    print(f"\n========== {text} ==========", flush=True)


# ---------------------------------------------------------------- Linux: Docker

def docker_ready() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def build_docker_image(versions: list[str]) -> bool:
    """One Ubuntu image holding a venv per Python version (see Dockerfile)."""
    banner(f"ubuntu: docker build (Python {', '.join(versions)})")
    build = ["docker", "build", "--build-arg", f"PYTHON_VERSIONS={' '.join(versions)}", "-t", IMAGE, str(ROOT)]
    return subprocess.run(build).returncode == 0


def run_docker(label: str, version: str, pytest_args: list[str]) -> tuple[str, str]:
    banner(f"{label}: pytest")
    # --shm-size: Docker's default 64MB /dev/shm is too small for the shared-memory tests
    run = ["docker", "run", "--rm", "--shm-size=2g", "-e", f"PYTHON_VERSIONS={version}", IMAGE, *pytest_args]
    ok = subprocess.run(run).returncode == 0
    return (PASSED, "docker") if ok else (FAILED, "docker")


# ---------------------------------------------------------------- Windows / macOS: native venv

def find_interpreter(version: str) -> list[str] | None:
    """A command that starts Python <version> on this machine, or None if it is not installed."""
    candidates = []
    if HOST == "windows" and shutil.which("py"):
        candidates.append(["py", f"-{version}"])
    if exe := shutil.which(f"python{version}"):
        candidates.append([exe])
    for cmd in candidates:
        probe = subprocess.run([*cmd, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                               capture_output=True, text=True)
        if probe.returncode == 0 and probe.stdout.strip() == version:
            return cmd
    return None


def run_native(label: str, version: str, pytest_args: list[str], fresh: bool) -> tuple[str, str]:
    interpreter = find_interpreter(version)
    if interpreter is None:
        return SKIPPED, f"Python {version} is not installed on this machine"

    venv = ENVS / f"{HOST}-py{version}"
    python = venv / ("Scripts/python.exe" if HOST == "windows" else "bin/python")
    if fresh and venv.exists():
        shutil.rmtree(venv)
    banner(f"{label}: venv + pip install")
    if not python.exists():
        if subprocess.run([*interpreter, "-m", "venv", str(venv)]).returncode != 0:
            return FAILED, "venv creation failed"
    env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1", "MPLBACKEND": "Agg"}
    env.pop("VIRTUAL_ENV", None)  # don't let an activated venv leak into the test venv
    install = [str(python), "-m", "pip", "install", "-q", "-e", ".[dev]"]
    if subprocess.run(install, cwd=ROOT, env=env).returncode != 0:
        return FAILED, "pip install failed"
    banner(f"{label}: pytest")
    ok = subprocess.run([str(python), "-m", "pytest", "-q", *pytest_args], cwd=ROOT, env=env).returncode == 0
    return (PASSED, "native venv") if ok else (FAILED, "native venv")


# ---------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--os", nargs="+", metavar="OS", dest="oses",
                        help="OSes to test, matched by prefix: ubuntu windows macos (default: all in tests.yml)")
    parser.add_argument("--python", nargs="+", metavar="VERSION",
                        help="Python versions to test (default: the ones in tests.yml)")
    parser.add_argument("--fresh", action="store_true", help="recreate the native venvs in .test-envs/")
    args, pytest_args = parser.parse_known_args()
    pytest_args = [a for a in pytest_args if a != "--"]

    runners = workflow_matrix_list("os")
    if args.oses:
        runners = [r for r in runners if any(r.lower().startswith(o.lower()) for o in args.oses)]
        if not runners:
            sys.exit(f"--os {' '.join(args.oses)} matches nothing in tests.yml")
    versions = args.python or workflow_matrix_list("python-version")

    need_docker = any(os_family(r) == "linux" for r in runners)
    have_docker = need_docker and docker_ready()
    if need_docker and not have_docker:
        print("WARNING: Docker is not running (or not installed): the Linux jobs will be skipped.")

    print(f"host: {HOST}   matrix: {', '.join(runners)} x Python {', '.join(versions)}")
    image_ok = have_docker and build_docker_image(versions)
    results = []
    for runner in runners:
        family = os_family(runner)
        for version in versions:
            label = f"{runner} / Python {version}"
            start = time.perf_counter()
            if family == "linux":
                if not have_docker:
                    status, note = SKIPPED, "Docker is not running"
                elif not image_ok:
                    status, note = FAILED, "image build failed"
                else:
                    status, note = run_docker(label, version, pytest_args)
            elif family == HOST:
                status, note = run_native(label, version, pytest_args, args.fresh)
            else:
                status, note = SKIPPED, f"{family} jobs can't run on a {HOST} machine (only on GitHub)"
            results.append((runner, version, status, note, time.perf_counter() - start))

    banner("summary")
    width = max(len(r) for r, *_ in results)
    for runner, version, status, note, seconds in results:
        took = f"{seconds:4.0f}s" if status != SKIPPED else "     "
        print(f"  {runner:<{width}}  Python {version:<6} {status:<8} {took}   {note}")
    return 1 if any(status == FAILED for _, _, status, _, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
