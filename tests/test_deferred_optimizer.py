from __future__ import annotations

from types import SimpleNamespace

import pytest

from babyLLM import BABYLLM


def test_deferred_optimizer_restores_only_at_training_barrier() -> None:
    loaded = []
    holder = SimpleNamespace(
        _deferred_optimizer_path="/tmp/baby.optim",
        _optimizer_load_error=None,
        _load_optimizer_state=lambda path: loaded.append(path),
    )

    assert BABYLLM.wait_for_optimizer_ready(holder) is True
    assert loaded == ["/tmp/baby.optim"]
    assert holder._deferred_optimizer_path is None


def test_deferred_optimizer_failure_blocks_training() -> None:
    def fail(_path):
        raise RuntimeError("corrupt optimizer")

    holder = SimpleNamespace(
        _deferred_optimizer_path="/tmp/baby.optim",
        _optimizer_load_error=None,
        _load_optimizer_state=fail,
    )

    with pytest.raises(RuntimeError, match="training is blocked"):
        BABYLLM.wait_for_optimizer_ready(holder)

    assert holder._deferred_optimizer_path is None
    assert isinstance(holder._optimizer_load_error, RuntimeError)


def test_optimizer_load_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        BABYLLM.loadModel(
            SimpleNamespace(),
            async_optimizer=True,
            defer_optimizer=True,
        )
