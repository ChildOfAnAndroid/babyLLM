#!/usr/bin/env python3
"""Guarded deployment of the merged BabyLLM unified bot on its own Mac.

This is deliberately a LOCAL operator command, not a GitHub Actions deploy:
hosted CI cannot access the Mac's model, optimizer, or live process.

Check only:
    python3 scripts/deploy_unified_local.py --check

Apply (after checking):
    python3 scripts/deploy_unified_local.py --apply

It never force-pulls, deletes files, creates an arbitrary bot, or edits a model
checkpoint directly. A verified SIGHUP to the existing unified *supervisor*
causes wakeup.py to finish training, save, and restart its worker.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

REPOSITORY = "ChildOfAnAndroid/babyLLM"
MINIMUM_SOURCE_COMMIT = "6bec23f4c2f2e361c4a635d5b3a1d5e96bcfe1aa"
MINIMUM_FREE_BYTES = 40 * 1024 ** 3
HEALTH_URL = "http://127.0.0.1:4420/api/state"
REQUIRED_CHECKPOINTS = (
    "babyllm_S_4200.pth",
    "babyllm_S_4200.pth.optim",
    "babyllm_S_4200.pth.membuff",
)
OPTIONAL_STATE = (
    "stepCheckpoint.txt",
    "lossCheckpoint.txt",
    "lossCheckpointAppend.txt",
    "trainingBuffer.json",
    "chatBuffer.json",
    "bbyUserData.json",
    "optInUsers.txt",
)


class DeploymentBlocked(RuntimeError):
    """Stop without making unreviewed or destructive changes."""


def command(argv: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(argv, cwd=str(cwd), text=True, capture_output=True)
    if check and result.returncode:
        raise DeploymentBlocked(
            f"{' '.join(argv[:3])} failed (exit={result.returncode}): "
            f"{result.stderr.strip()[:300]}"
        )
    return result


def git(root: Path, *args: str, check: bool = True) -> str:
    return command(["git", *args], cwd=root, check=check).stdout.strip()


def verify_repo(root: Path) -> str:
    if not (root / "wakeup.py").is_file() or not (root / "babyLLM.py").is_file():
        raise DeploymentBlocked("Not a BabyLLM source checkout.")
    actual_root = Path(git(root, "rev-parse", "--show-toplevel")).resolve()
    if actual_root != root.resolve():
        raise DeploymentBlocked(f"Unexpected Git checkout root: {actual_root}")
    remote = git(root, "remote", "get-url", "origin").strip().lower().removesuffix(".git")
    if not re.search(r"(?:github\.com[:/])childofanandroid/babyllm$", remote):
        raise DeploymentBlocked("The origin remote is not ChildOfAnAndroid/babyLLM.")
    branch = git(root, "branch", "--show-current")
    if branch != "main":
        raise DeploymentBlocked(f"Switch to main first; refusing to change branch {branch!r}.")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise DeploymentBlocked("Tracked local changes exist; refusing to overwrite them.")
    return git(root, "rev-parse", "HEAD")


def snapshot_checkpoints(root: Path) -> list[Path]:
    folder = root / "SHKAIRA" / "soul"
    paths: list[Path] = []
    for name in REQUIRED_CHECKPOINTS:
        path = folder / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size < 1024:
            raise DeploymentBlocked(f"Missing, unreadable, or suspect checkpoint: {path}")
        paths.append(path)
    for name in OPTIONAL_STATE:
        path = folder / name
        if path.exists():
            if not path.is_file() or path.is_symlink():
                raise DeploymentBlocked(f"Unexpected state file type: {path}")
            paths.append(path)
    for name in ("babyllm_S_4200.pth.tmp", "babyllm_S_4200.pth.optim.tmp",
                 "babyllm_S_4200.pth.membuff.tmp"):
        if (folder / name).exists():
            raise DeploymentBlocked(f"Checkpoint save appears unfinished: {name}")
    return paths


def available_space(root: Path) -> int:
    free = shutil.disk_usage(root).free
    if free < MINIMUM_FREE_BYTES:
        raise DeploymentBlocked(
            f"Only {free / 1024 ** 3:.2f} GiB free; require at least "
            f"{MINIMUM_FREE_BYTES / 1024 ** 3:.0f} GiB before checkpoint-saving restart."
        )
    return free


def parse_processes(output: str) -> list[tuple[int, int, list[str]]]:
    """Parse macOS ps's pid/ppid/command columns without matching grep itself."""
    import shlex

    found = []
    for line in output.splitlines():
        match = re.match(r"^\s*(\d+)\s+(\d+)\s+(.+)$", line)
        if not match:
            continue
        try:
            tokens = shlex.split(match.group(3))
        except ValueError:
            continue
        # A grep/search process that mentions wakeup.py is not Baby's
        # Python supervisor. Require a Python interpreter in the command.
        if not any("python" in Path(token).name.lower() for token in tokens[:2]):
            continue
        if not any(Path(token).name == "wakeup.py" for token in tokens):
            continue
        if not any(token in ("unified", "u", "all") for token in tokens):
            continue
        found.append((int(match.group(1)), int(match.group(2)), tokens))
    return found


def validate_process_cwd(root: Path, pid: int, tokens: list[str]) -> None:
    script_tokens = [token for token in tokens if Path(token).name == "wakeup.py"]
    if len(script_tokens) != 1:
        raise DeploymentBlocked("Could not uniquely identify the running wakeup.py.")
    script = Path(script_tokens[0])
    if script.is_absolute():
        if script.resolve() != (root / "wakeup.py").resolve():
            raise DeploymentBlocked("Running bot belongs to a different checkout.")
        return
    lsof = shutil.which("lsof")
    if not lsof:
        raise DeploymentBlocked("Cannot verify the running bot's working directory (lsof unavailable).")
    process = command([lsof, "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
                      cwd=root, check=False)
    observed = [
        Path(line[1:]).resolve()
        for line in process.stdout.splitlines() if line.startswith("n/")
    ]
    if process.returncode or observed != [root.resolve()]:
        raise DeploymentBlocked("Cannot prove that the running unified supervisor uses this checkout.")


def find_supervisor(root: Path) -> tuple[int, int]:
    ps = command(["ps", "-ww", "-axo", "pid=,ppid=,command="], cwd=root)
    rows = parse_processes(ps.stdout)
    parents = [(pid, tokens) for pid, _, tokens in rows
               if "--unified-worker" not in tokens]
    if len(parents) != 1:
        raise DeploymentBlocked(
            f"Expected exactly one unified supervisor, found {len(parents)}."
        )
    parent_pid, parent_tokens = parents[0]
    children = [pid for pid, ppid, tokens in rows
                if ppid == parent_pid and "--unified-worker" in tokens]
    if len(children) != 1:
        raise DeploymentBlocked(
            f"Expected exactly one unified worker for PID {parent_pid}, found {len(children)}."
        )
    validate_process_cwd(root, parent_pid, parent_tokens)
    return parent_pid, children[0]


def health_check() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=3) as response:
            if response.status != 200:
                return False
            return isinstance(json.load(response), dict)
    except (OSError, ValueError, TimeoutError):
        return False


def backup_checkpoint_set(root: Path, paths: list[Path], previous_commit: str) -> Path:
    """Preserve existing data as APFS clonefiles, never as hard links.

    Refuse if cloning is unsupported rather than falling back to multi-GB
    non-atomic copies on a constrained volume. Each original is checked for
    mtime/size changes during its clone.
    """
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = root / "SHKAIRA" / "soul" / "deploy_backups" / f"{now}-{previous_commit[:12]}"
    folder.mkdir(parents=True, exist_ok=False, mode=0o700)
    inventory = []
    try:
        for source in paths:
            before = source.stat()
            target = folder / source.name
            command(["/bin/cp", "-c", str(source), str(target)], cwd=root)
            after = source.stat()
            copied = target.stat()
            if ((before.st_size, before.st_mtime_ns) !=
                (after.st_size, after.st_mtime_ns) or
                copied.st_size != before.st_size or
                copied.st_ino == before.st_ino):
                raise DeploymentBlocked(f"Checkpoint changed during snapshot: {source.name}")
            inventory.append({
                "name": source.name, "bytes": before.st_size,
                "mtime_ns": before.st_mtime_ns,
            })
        (folder / "manifest.json").write_text(
            json.dumps({
                "source_commit": previous_commit,
                "created_utc": now,
                "note": "APFS pre-deploy clones; model files remain owned by BabyLLM.",
                "files": inventory,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        # Preserve partial recovery evidence rather than silently deleting.
        raise
    return folder


def wait_for_new_worker(root: Path, parent: int, old_worker: int,
                        *, timeout_seconds: float = 240) -> int:
    """A restart is not verified until a different child and API both work."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            current_parent, current_worker = find_supervisor(root)
            if current_parent == parent and current_worker != old_worker and health_check():
                return current_worker
        except DeploymentBlocked:
            pass
        time.sleep(3)
    raise DeploymentBlocked(
        "Restart was requested but new worker/API health could not be verified. "
        "DO NOT send another signal blindly; inspect Baby's logs and recovery clones."
    )


def deploy(root: Path, *, apply: bool) -> None:
    if sys.platform != "darwin":
        raise DeploymentBlocked("This deployment command must run on Baby's Mac.")
    root = root.expanduser().resolve()
    previous_commit = verify_repo(root)
    sources = snapshot_checkpoints(root)
    free = available_space(root)
    parent, old_worker = find_supervisor(root)
    if not health_check():
        raise DeploymentBlocked("The existing BabyLLM loopback API is unhealthy; do not blind-restart.")

    print(f"BabyLLM checkout: {root}")
    print(f"Current main: {previous_commit}")
    print(f"Checkpoint: {len(sources)} files present; {free / 1024**3:.2f} GiB free")
    print(f"Unified supervisor: {parent}; current worker: {old_worker}")
    if not apply:
        print("READ-ONLY CHECK PASSED. No Git updates, backup, signals, or model writes.")
        return

    # Convergence must be fast-forward-only. Local tracked edits cannot be
    # stashed, discarded, rebased or force-overwritten by this helper.
    git(root, "fetch", "--quiet", "origin", "main")
    remote_commit = git(root, "rev-parse", "refs/remotes/origin/main")
    if command(["git", "merge-base", "--is-ancestor", previous_commit, remote_commit],
               cwd=root, check=False).returncode:
        raise DeploymentBlocked("Local main has diverged from origin/main.")
    if command(["git", "merge-base", "--is-ancestor", MINIMUM_SOURCE_COMMIT, remote_commit],
               cwd=root, check=False).returncode:
        raise DeploymentBlocked("Remote main lacks the verified Self-Voice Trust merge.")
    backup = backup_checkpoint_set(root, sources, previous_commit)
    print(f"Checkpoint recovery clones: {backup}")
    git(root, "merge", "--ff-only", remote_commit)
    installed_commit = git(root, "rev-parse", "HEAD")
    if installed_commit != remote_commit:
        raise DeploymentBlocked("Git fast-forward did not reach expected commit; no restart sent.")

    # Send one signal to the supervisor, never PID-kill the worker or run
    # a second bot. wakeup.py forwards SIGHUP and saves through its owner.
    try:
        os.kill(parent, signal.SIGHUP)
    except OSError as exc:
        raise DeploymentBlocked(f"Could not request checkpoint-saving restart: {exc}") from exc
    print(f"Graceful SIGHUP sent to supervisor {parent}.")
    new_worker = wait_for_new_worker(root, parent, old_worker)
    print(f"DEPLOY_VERIFIED commit={installed_commit} new_worker={new_worker}")
    print(f"Recovery clones retained at {backup}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Read-only local deployment preflight")
    group.add_argument("--apply", action="store_true", help="Clone checkpoint, fast-forward, and request one restart")
    parser.add_argument(
        "--repo", type=Path,
        default=Path(__file__).resolve().parents[1],
        help="BabyLLM checkout root (needed if script is downloaded to /tmp)",
    )
    args = parser.parse_args()
    try:
        deploy(args.repo, apply=args.apply)
    except (DeploymentBlocked, OSError, subprocess.SubprocessError) as exc:
        print(f"DEPLOY_BLOCKED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
