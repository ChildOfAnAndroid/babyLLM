from contextlib import nullcontext
import json
from types import SimpleNamespace
import sqlite3

import pytest
import torch

from babyLLM import BABYLLM
from brain.LAYERS import tangling
from brain.LAYERS.stream_feedback import (
    CHANNELS, LOSS_WEIGHT, SOURCES, StreamFeedback, StreamFeedbackReader, decode,
)


def test_feedback_pairs_are_ordered_held_out_and_masked(tmp_path):
    path = tmp_path / "trace.sqlite3"
    with sqlite3.connect(path) as database:
        database.execute("CREATE TABLE trace (at REAL, kind TEXT, data TEXT)")
        database.executemany("INSERT INTO trace VALUES (?, ?, ?)", (
            (100, "broadcast", json.dumps({"rms_dbfs": -70})),
            (120, "broadcast", json.dumps({"rms_dbfs": -10})),
            (110, "world-learning", json.dumps({"livingFraction": .15})),
            (118, "broadcast-reply", json.dumps({"alignment": "unverified", "margin_db": 25})),
            (125, "world-learning", json.dumps({"livingFraction": .25})),
        ))
    reader = StreamFeedbackReader(path)
    sample = reader.read([0] * len(SOURCES), now=130)
    assert sample.target.at_ms == 120_000
    assert sample.target.mask == (1, 0, 0, 0)
    assert sample.target.values[0] == 1
    assert sample.context[:8] == (-1., 0., 0., 0., 1, 0, 0, 0)
    assert sample.context[8:16] == (0.,) * 8
    assert sample.context[16] == 0   # Previous World density .15, not the later .25 reading.
    sample = reader.read([120_000, 0, 0], now=130)
    assert sample.target.source == SOURCES.index("world-learning")
    assert reader.read([100_000, 0, 0], now=130).target.source == SOURCES.index("world-learning")
    assert reader.read([120_000, 0, 125_000], now=130).target is None
    assert reader.read([0, 0, 0], now=400).context is None
    missing = tmp_path / "missing.sqlite3"
    assert StreamFeedbackReader(missing).read([0, 0, 0], now=130).context is None
    assert not missing.exists()


@pytest.mark.parametrize("kind,measurement", [
    ("broadcast-reply", {"alignment": "verified", "margin_db": 21,
                         "brightness": .1, "release_windows": 3, "release_transients_detected": 2}),
    ("world-learning", {"livingFraction": .25, "nutrientPerTile": .08, "pigmentPerTile": 8}),
])
def test_measurement_loss_updates_real_tangle_and_sensor_weights(monkeypatch, kind, measurement):
    torch.manual_seed(9)
    monkeypatch.setattr(tangling, "embedDimension", 24)
    monkeypatch.setattr(tangling, "numNeurons", 32)
    counsellor = SimpleNamespace(infodump=lambda _: nullcontext())
    brain = tangling.MINI_INN_TANGLING(counsellor, 8, _device=torch.device("cpu"))
    branch = StreamFeedback(24, "cpu")
    optimizer = torch.optim.SGD([*brain.parameters(), *branch.parameters()], lr=.2)
    words = torch.randn(8, 24)
    context = torch.full((CHANNELS * 2,), .2)
    target = decode(kind, 100, measurement)
    blended = branch.condition(words, context)
    assert (blended - words).abs().max() <= .020001
    loss = branch.loss(blended + brain.refine(blended, stage_name="embed"), target)
    assert 0 < loss < LOSS_WEIGHT
    loss.backward()
    watched = [branch.encoder.weight, branch.predictor.weight, brain.embed_proj_down.weight]
    before = [param.detach().clone() for param in watched]
    for param in watched:
        assert torch.isfinite(param.grad).all()
        assert param.grad.abs().max() > 0
    optimizer.step()
    for original, trained in zip(before, watched):
        assert not torch.equal(original, trained)


def test_legacy_optimizer_keeps_its_moments_when_the_sensor_group_is_added():
    anchor = torch.nn.Parameter(torch.ones(2))
    old = torch.optim.AdamW([anchor])
    anchor.sum().backward()
    old.step()
    saved = old.state_dict()
    shell = BABYLLM.__new__(BABYLLM)
    torch.nn.Module.__init__(shell)
    shell.stream_feedback = StreamFeedback(4, "cpu")
    shell.optimizer = torch.optim.AdamW([anchor])
    shell.optimizer.add_param_group({"params": shell.stream_feedback.parameters()})
    shell._append_stream_optimizer_group(saved["param_groups"], shell.optimizer.state_dict()["param_groups"])
    shell.optimizer.load_state_dict(saved)
    torch.testing.assert_close(shell.optimizer.state[anchor]["exp_avg"], old.state[anchor]["exp_avg"])
    assert all(param not in shell.optimizer.state for param in shell.stream_feedback.parameters())
