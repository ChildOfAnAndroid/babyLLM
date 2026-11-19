"""Observation transport and trainable sensory branch for the broadcast."""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sqlite3
import time

import torch
from torch import nn

DEFAULT_JOURNAL = Path.home() / "Library/Application Support/BabyLLM/stream-trace.sqlite3"
# Fixed channel order is part of model and optimizer checkpoint compatibility.
FIELDS = {
    "broadcast": (("rms_dbfs", -70, -10), ("peak", 0, 1), ("brightness", 0, 1), ("quiet_fraction", 0, 1)),
    "broadcast-reply": (("margin_db", -12, 32), ("brightness", 0, 1), ("quiet_fraction", 0, 1), ("release_ratio", 0, 1)),
    "world-learning": (("livingFraction", 0, .30), ("organicMatterPerTile", 0, .05),
                       ("nutrientPerTile", 0, .15), ("pigmentPerTile", 0, 32)),
}
SOURCES = tuple(FIELDS)
CHANNELS = sum(map(len, FIELDS.values()))
MAX_AGE = 120
LOSS_WEIGHT = .005


@dataclass(frozen=True)
class Observation:
    source: int
    at_ms: int
    values: tuple
    mask: tuple


@dataclass(frozen=True)
class Feedback:
    context: tuple | None = None
    target: Observation | None = None


def decode(kind, at, data):
    if type(at) not in (int, float) or not math.isfinite(at) or not isinstance(data, dict):
        return None
    if kind == "broadcast-reply":
        if data.get("alignment") != "verified":
            return None
        windows = data.get("release_windows")
        hits = data.get("release_transients_detected")
        if type(windows) is int and windows > 0 and type(hits) is int and 0 <= hits <= windows:
            data = {**data, "release_ratio": hits / windows}
    values = []
    mask = []
    for key, low, high in FIELDS[kind]:
        value = data.get(key)
        present = type(value) in (int, float) and math.isfinite(value)
        mask.append(int(present))
        values.append(max(0., min(1., (value - low) / (high - low))) if present else .5)
    if any(mask):
        return Observation(SOURCES.index(kind), round(at * 1000), tuple(values), tuple(mask))
    return None


class StreamFeedbackReader:
    """Read the journal without starting a writer, sensor, or network client."""

    def __init__(self, path=None):
        self.path = Path(path or os.environ.get("BBY_STREAM_TRACE_DB", DEFAULT_JOURNAL)).expanduser()

    def read(self, watermarks, now=None):
        now = time.time() if now is None else now
        recent = [[] for _ in SOURCES]
        try:
            with sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=.05) as db:
                for index, kind in enumerate(SOURCES):
                    rows = db.execute(
                        "SELECT at, data FROM trace WHERE at BETWEEN ? AND ? AND kind = ? ORDER BY at DESC LIMIT 32",
                        (now - MAX_AGE, now, kind),
                    )
                    for at, payload in rows:
                        try:
                            observation = decode(kind, at, json.loads(payload))
                        except (ValueError, TypeError):
                            continue
                        if observation is not None:
                            recent[index].append(observation)
        except sqlite3.Error:
            return Feedback()
        finally:
            if "db" in locals():
                db.close()

        # A target is held out from its context. Require an older valid sample
        # of the same source so an isolated reading cannot supervise a step.
        candidates = [items[0] for index, items in enumerate(recent)
                      if len(items) >= 2 and items[0].at_ms > watermarks[index]]
        # The HLS clock usually trails the World clock. Prefer the source
        # trained least recently so that latency cannot starve the World.
        target = min(candidates, key=lambda item: (watermarks[item.source], item.at_ms), default=None)
        context = []
        for kind, items in zip(SOURCES, recent):
            previous = next((item for item in items if target is None or item.at_ms < target.at_ms), None)
            width = len(FIELDS[kind])
            if previous:
                context.extend((2 * value - 1) * present for value, present in zip(previous.values, previous.mask))
                context.extend(previous.mask)
            else:
                context.extend([0.] * (width * 2))
        return Feedback(tuple(context) if any(items for items in recent) else None, target)


class StreamFeedback(nn.Module):
    """A bounded input into the tangle and a readout of its resulting state."""

    def __init__(self, embed_dim, device):
        super().__init__()
        self.encoder = nn.Linear(CHANNELS * 2, embed_dim, bias=False, device=device)
        self.predictor = nn.Linear(embed_dim, CHANNELS, bias=False, device=device)
        nn.init.normal_(self.encoder.weight, std=.04)
        nn.init.normal_(self.predictor.weight, std=.01)
        self.register_buffer("watermarks", torch.zeros(len(SOURCES), dtype=torch.long, device=device))

    def condition(self, embedding, context):
        return embedding + .02 * torch.tanh(self.encoder(context.to(embedding.dtype)))

    def loss(self, tangled_embedding, target):
        embedding = tangled_embedding.reshape(-1, tangled_embedding.shape[-1])[-1]
        start = sum(len(FIELDS[kind]) for kind in SOURCES[:target.source])
        stop = start + len(target.values)
        prediction = self.predictor(embedding).float()[start:stop].sigmoid()
        wanted = prediction.new_tensor(target.values)
        mask = prediction.new_tensor(target.mask)
        return ((prediction - wanted).square() * mask).sum() / mask.sum() * LOSS_WEIGHT
