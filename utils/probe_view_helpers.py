"""Shared cache and label helpers for the distinct latent-probe views."""
import json
from pathlib import Path
import numpy as np


def available_cache_files(directory, *, include_sequences=False):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    prefixes = ("probe_w", "probe_seq_") if include_sequences else ("probe_w",)
    return sorted(str(p) for p in directory.iterdir()
                  if p.name.startswith(prefixes) and p.name.endswith(".npz"))


def load_probe_cache(path, *, include_extras=False):
    with np.load(path, allow_pickle=True) as data:
        entries = json.loads(str(data["entries_json"][0]))
        neutral = {k[8:]: data[k] for k in data.files if k.startswith("neutral_")}
        tiled = {k[6:]: data[k] for k in data.files if k.startswith("tiled_")}
        if not include_extras:
            return entries, neutral, tiled
        extras = {}
        if "logit_lens_json" in data.files:
            extras["logit_lens"] = json.loads(str(data["logit_lens_json"][0]))
        for key in ("blend_weights", "cereb_long_weights", "cereb_short_weights", "cereb_short_gate"):
            if key in data.files:
                extras[key] = data[key]
        return entries, neutral, tiled, extras


def token_label(entry):
    return entry["label"] if entry.get("labeled") else entry["tok"]


def safe_category(entry):
    return str(entry.get("cat") or "unknown")


def load_three_vendor_sources(directory):
    sources = []
    for name in ("three.module.min.js", "OrbitControls.js"):
        path = Path(directory) / name
        if not path.exists():
            raise FileNotFoundError(f"Missing vendored Three.js file: {path}")
        sources.append(path.read_text(encoding="utf-8"))
    return tuple(sources)
