"""Non-invasive tests for BabyLLM's guarded local Mac deployment helper.

No test invokes the real macOS process list, sends a signal, runs git fetch,
copies the user's actual checkpoints or connects to Baby's public services.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from scripts import deploy_unified_local as deployment


def test_parse_processes_matches_only_actual_unified_wakeup_commands():
    sample = """
       800 1 /opt/homebrew/bin/python3 /Users/charis/babyLLM/wakeup.py unified
       801 800 /opt/homebrew/bin/python3 /Users/charis/babyLLM/wakeup.py unified --unified-worker
       804 1 /opt/homebrew/bin/python3 /Users/charis/babyLLM/wakeup.py train
       805 1 grep wakeup.py unified
       806 1 /usr/bin/python3 other_script.py unified
    """
    assert deployment.parse_processes(sample) == [
        (800, 1, ["/opt/homebrew/bin/python3", "/Users/charis/babyLLM/wakeup.py", "unified"]),
        (801, 800, ["/opt/homebrew/bin/python3", "/Users/charis/babyLLM/wakeup.py", "unified", "--unified-worker"]),
    ]


def test_find_supervisor_fails_closed_when_worker_or_parent_ambiguous(tmp_path, monkeypatch):
    (tmp_path / "wakeup.py").write_text("")
    def fake_command(argv, *, cwd, check=True):
        if argv[0] == "ps":
            return subprocess.CompletedProcess(argv, 0, stdout=(
                f"800 1 python3 {tmp_path}/wakeup.py unified\n"
                f"801 800 python3 {tmp_path}/wakeup.py unified --unified-worker\n"
            ), stderr="")
        raise AssertionError(argv)
    monkeypatch.setattr(deployment, "command", fake_command)
    assert deployment.find_supervisor(tmp_path) == (800, 801)

    def duplicate_workers(argv, *, cwd, check=True):
        result = fake_command(argv, cwd=cwd)
        result.stdout += f"802 800 python3 {tmp_path}/wakeup.py unified --unified-worker\n"
        return result
    monkeypatch.setattr(deployment, "command", duplicate_workers)
    with pytest.raises(deployment.DeploymentBlocked, match="exactly one unified worker"):
        deployment.find_supervisor(tmp_path)


def test_repository_verification_rejects_wrong_remote_or_dirty_main(tmp_path, monkeypatch):
    (tmp_path / "wakeup.py").write_text("")
    (tmp_path / "babyLLM.py").write_text("")
    def git_ok(root, *args, **kwargs):
        key = tuple(args)
        if key == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if key == ("remote", "get-url", "origin"):
            return "git@github.com:ChildOfAnAndroid/babyLLM.git"
        if key == ("branch", "--show-current"):
            return "main"
        if key == ("status", "--porcelain", "--untracked-files=no"):
            return ""
        if key == ("rev-parse", "HEAD"):
            return "oldhead"
        raise AssertionError(key)
    monkeypatch.setattr(deployment, "git", git_ok)
    assert deployment.verify_repo(tmp_path) == "oldhead"
    def wrong_remote(root, *args, **kwargs):
        if args == ("remote", "get-url", "origin"):
            return "git@github.com:attacker/repository.git"
        return git_ok(root, *args, **kwargs)
    monkeypatch.setattr(deployment, "git", wrong_remote)
    with pytest.raises(deployment.DeploymentBlocked, match="origin remote"):
        deployment.verify_repo(tmp_path)


def test_checkpoint_missing_or_incomplete_save_blocks_before_deployment(tmp_path):
    with pytest.raises(deployment.DeploymentBlocked, match="checkpoint"):
        deployment.snapshot_checkpoints(tmp_path)
    soul = tmp_path / "SHKAIRA" / "soul"
    soul.mkdir(parents=True)
    for name in deployment.REQUIRED_CHECKPOINTS:
        (soul / name).write_bytes(b"x" * 4096)
    assert len(deployment.snapshot_checkpoints(tmp_path)) == 3
    (soul / "babyllm_S_4200.pth.optim.tmp").write_text("in progress")
    with pytest.raises(deployment.DeploymentBlocked, match="unfinished"):
        deployment.snapshot_checkpoints(tmp_path)


def test_clone_predeploy_checkpoints_and_metadata_in_isolated_tmp_path(tmp_path, monkeypatch):
    soul = tmp_path / "SHKAIRA" / "soul"
    soul.mkdir(parents=True)
    files = []
    for name in deployment.REQUIRED_CHECKPOINTS:
        path = soul / name
        path.write_bytes(b"data" * 1200)
        files.append(path)
    cp_calls = []
    def fake_cp(argv, *, cwd, check=True):
        assert argv[:2] == ["/bin/cp", "-c"]
        cp_calls.append(argv)
        shutil.copy2(argv[2], argv[3])
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    monkeypatch.setattr(deployment, "command", fake_cp)
    folder = deployment.backup_checkpoint_set(tmp_path, files, "previous-commit")
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["source_commit"] == "previous-commit"
    assert len(manifest["files"]) == len(files)
    assert len(cp_calls) == len(files)
    for file in files:
        copied = folder / file.name
        assert copied.exists()
        assert copied.read_bytes() == file.read_bytes()
        assert copied.stat().st_ino != file.stat().st_ino


def test_readonly_check_does_not_fetch_clone_or_signal(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(deployment.sys, "platform", "darwin")
    monkeypatch.setattr(deployment, "verify_repo", lambda _: "previous-commit")
    monkeypatch.setattr(deployment, "snapshot_checkpoints", lambda _: [Path("mock.pth")])
    monkeypatch.setattr(deployment, "available_space", lambda _: deployment.MINIMUM_FREE_BYTES)
    monkeypatch.setattr(deployment, "find_supervisor", lambda _: (11, 12))
    monkeypatch.setattr(deployment, "health_check", lambda: True)
    monkeypatch.setattr(deployment, "git", lambda *a, **kw: pytest.fail("Read-only check mutated Git"))
    monkeypatch.setattr(deployment, "backup_checkpoint_set", lambda *a: pytest.fail("Read-only check attempted backup"))
    monkeypatch.setattr(deployment.os, "kill", lambda *a: pytest.fail("Read-only check signalled the live bot"))
    deployment.deploy(tmp_path, apply=False)
    assert "READ-ONLY CHECK PASSED" in capsys.readouterr().out


def test_apply_fast_forwards_and_signals_supervisor_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(deployment.sys, "platform", "darwin")
    old, new = "old", "new"
    monkeypatch.setattr(deployment, "verify_repo", lambda _: old)
    monkeypatch.setattr(deployment, "snapshot_checkpoints", lambda _: [Path("checkpoint")])
    monkeypatch.setattr(deployment, "available_space", lambda _: deployment.MINIMUM_FREE_BYTES)
    monkeypatch.setattr(deployment, "find_supervisor", lambda _: (99, 100))
    monkeypatch.setattr(deployment, "health_check", lambda: True)
    calls = []
    def fake_git(root, *args, check=True):
        calls.append(args)
        if args == ("rev-parse", "refs/remotes/origin/main"):
            return new
        if args == ("rev-parse", "HEAD"):
            return new
        return ""
    monkeypatch.setattr(deployment, "git", fake_git)
    monkeypatch.setattr(deployment, "command", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 0, stdout="", stderr=""))
    monkeypatch.setattr(deployment, "backup_checkpoint_set", lambda *args: tmp_path / "backup")
    signals = []
    monkeypatch.setattr(deployment.os, "kill", lambda pid, signum: signals.append((pid, signum)))
    monkeypatch.setattr(deployment, "wait_for_new_worker", lambda *args: 101)
    deployment.deploy(tmp_path, apply=True)
    assert ("merge", "--ff-only", new) in calls
    assert signals == [(99, deployment.signal.SIGHUP)]
    assert "DEPLOY_VERIFIED" in capsys.readouterr().out


def test_apply_does_not_signal_after_failed_fastforward(tmp_path, monkeypatch):
    monkeypatch.setattr(deployment.sys, "platform", "darwin")
    monkeypatch.setattr(deployment, "verify_repo", lambda _: "previous")
    monkeypatch.setattr(deployment, "snapshot_checkpoints", lambda _: [Path("checkpoint")])
    monkeypatch.setattr(deployment, "available_space", lambda _: deployment.MINIMUM_FREE_BYTES)
    monkeypatch.setattr(deployment, "find_supervisor", lambda _: (99, 100))
    monkeypatch.setattr(deployment, "health_check", lambda: True)
    monkeypatch.setattr(deployment, "git", lambda root, *args, **kwargs: "new")
    monkeypatch.setattr(deployment, "command", lambda argv, **kw:
                        subprocess.CompletedProcess(argv, 1, stdout="", stderr="not ancestor"))
    monkeypatch.setattr(deployment, "backup_checkpoint_set", lambda *args:
                        pytest.fail("Backup attempted despite divergent Git history"))
    monkeypatch.setattr(deployment.os, "kill", lambda *args:
                        pytest.fail("Signal attempted despite divergent Git history"))
    with pytest.raises(deployment.DeploymentBlocked, match="diverged"):
        deployment.deploy(tmp_path, apply=True)
