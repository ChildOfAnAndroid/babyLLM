#!/usr/bin/env python3
# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm latent probe · interactive UI -*- ⊂ʕʘ‿ʘ૮ʔ ---
# utils/latent_probe_ui.py
#
# Run from project root:
#   python3 -m streamlit run utils/latent_probe_ui.py
#
# Interactive latent-space probe: animated t-SNE scatter across transformer layers.
# Analysis panels deferred behind "Show analysis panels" button.

import sys
import os
import json
import hashlib
import random
from functools import reduce
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import torch
from scipy.spatial.distance import cdist

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.latent_probe import (
    build_model,
    sample_tokens,
    probe_tokens,
    LABEL_WORDS,
    NEUTRAL_FILL,
    RANDOM_SEED,
    N_VOCAB_SAMPLE,
    _ALL_LAYER_DEFS,
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="babyLLM · latent probe",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
<style>
body, .stApp { background-color: #0d0d0d; color: #dddddd; }
section[data-testid="stSidebar"] { background-color: #161616; }
h1, h2, h3, label, .stCheckbox span { color: #eeeeee !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── constants ─────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(PROJECT_ROOT, ".probe_cache")
ALL_LAYER_KEYS = [ld[1] for ld in _ALL_LAYER_DEFS]

_VOWELS = set("aeiou")
_SENT_PUNCT = set(".!?,;:")
_MATH_PUNCT = set("+-*/=<>%^~|\\&@#")
_BRACKET_PUNCT = set("()[]{}")
_QUOTE_PUNCT = set("'\"`")


def _blake_array(arr: np.ndarray, person: str = "latent-probe") -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.blake2b(digest_size=20, person=person.encode("utf-8")[:16])
    h.update(str(a.shape).encode("utf-8"))
    h.update(str(a.dtype).encode("utf-8"))
    h.update(a.view(np.uint8))
    return h.hexdigest()


def _fine_cat(e: Dict[str, Any]) -> str:
    if e["labeled"]:
        return "labeled"

    c = e["cat"]
    tok = e["tok"].lstrip("\u0120").strip()

    if c == "letter":
        if len(tok) == 1 and tok.lower() in _VOWELS:
            return "letter_vowel"
        return "letter_cons"

    if c == "digit":
        return "digit_lo" if tok in "01234" else "digit_hi"

    if c == "punct":
        if tok in _SENT_PUNCT:
            return "punct_sent"
        if tok in _MATH_PUNCT:
            return "punct_math"
        if tok in _BRACKET_PUNCT:
            return "punct_bracket"
        if tok in _QUOTE_PUNCT:
            return "punct_quote"
        return "punct_other"

    if c == "bpe_word":
        return "bpe_word_short" if len(tok) <= 4 else "bpe_word_long"

    if c == "bpe_sub":
        return "bpe_sub_short" if len(tok) <= 3 else "bpe_sub_long"

    return c


CAT_COLOR = {
    "labeled": "#ffeb3b",
    "letter_vowel": "#4fc3f7",
    "letter_cons": "#0277bd",
    "digit_lo": "#a5d6a7",
    "digit_hi": "#2e7d32",
    "punct_sent": "#ff8a65",
    "punct_math": "#ef5350",
    "punct_bracket": "#ff6f00",
    "punct_quote": "#ffd54f",
    "punct_other": "#795548",
    "bpe_word_short": "#ce93d8",
    "bpe_word_long": "#7b1fa2",
    "bpe_sub_short": "#f48fb1",
    "bpe_sub_long": "#c2185b",
    "special": "#607d8b",
}

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("ʕっʘ‿ʘʔっ babyLLM probe")

st.sidebar.header("Window size")
window_max = st.sidebar.radio("tokens", [269, 64], horizontal=True)

st.sidebar.header("Traces")
show_neutral = st.sidebar.checkbox("Neutral prefix  ●", value=True)
show_tiled = st.sidebar.checkbox("Tiled / looped  ◆", value=False)
show_diff = st.sidebar.checkbox("Strategy diff   —", value=False)
show_trail = st.sidebar.checkbox("Layer trail     ···", value=True)

st.sidebar.header("Scatter budget")
scatter_max = st.sidebar.slider(
    "Analysis scatter tokens",
    min_value=100,
    max_value=4200,
    value=500,
    step=50,
    help=(
        "Always includes all base chars + labeled words. "
        "Remaining budget filled with a random BPE sample."
    ),
    key="scatter_max",
)

st.sidebar.header("Embedding")
stabilize_view = st.sidebar.checkbox(
    "Stabilize animated view",
    value=True,
    help="Recenters and rescales each layer so the whole cloud does not drift wildly.",
    key="stabilize_view",
)

st.sidebar.header("Layers")
layer_toggles = {}
for key in ALL_LAYER_KEYS:
    defn = next(d for d in _ALL_LAYER_DEFS if d[1] == key)
    label = f"**{key}** — {defn[2].replace(chr(10), ' ')}"
    layer_toggles[key] = st.sidebar.checkbox(label, value=True, key=f"layer_{key}")

active_layers = [k for k, v in layer_toggles.items() if v]

st.sidebar.header("Track tokens")
track_input = st.sidebar.text_input(
    "Words / tokens to highlight in purple",
    placeholder="hello boof cat fire",
    help="Space-separated. Matches vocab tokens (or Ġword variants).",
)

st.sidebar.header("Activation overlay")
overlay_mode = st.sidebar.selectbox(
    "Brightness overlay",
    [
        "off",
        "single neuron",
        "L2 norm  (all neurons)",
        "neuron range  (mean |act|)",
        "layer PCA component",
    ],
    index=2,
    key="overlay_mode",
    help="Brightness is computed from the actual displayed layer and strategy.",
)

spotlight_neuron_idx: Optional[int] = None
spotlight_range: Optional[Tuple[int, int]] = None
spotlight_pca_k: Optional[int] = None

if overlay_mode == "single neuron":
    spotlight_neuron_idx = int(
        st.sidebar.number_input(
            "Dimension / neuron index",
            min_value=0,
            max_value=9999,
            value=0,
            step=1,
            key="spot_idx",
        )
    )
elif overlay_mode == "neuron range  (mean |act|)":
    _sc1, _sc2 = st.sidebar.columns(2)
    _n_from = int(_sc1.number_input("from", 0, 9998, 0, 1, key="range_from"))
    _n_to = int(_sc2.number_input("to", 1, 9999, 499, 1, key="range_to"))
    spotlight_range = (_n_from, min(_n_to, 9999))
elif overlay_mode == "layer PCA component":
    spotlight_pca_k = int(
        st.sidebar.slider(
            "PCA component  (1 = most variance)",
            min_value=1,
            max_value=8,
            value=1,
            step=1,
            key="pca_k",
        )
    )

st.sidebar.header("Cache")
force_rerun = st.sidebar.button("Re-run probe (flush cache)")

st.sidebar.markdown("---")
st.sidebar.caption(
    "● circle  = neutral\n"
    "◆ diamond = tiled\n"
    "★ star    = labeled word\n"
    "◆ purple  = tracked tokens"
)

# ── cache helpers ─────────────────────────────────────────────────────────────
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(w: int) -> str:
    return os.path.join(CACHE_DIR, f"probe_w{w}.npz")


def _cache_exists(w: int) -> bool:
    return os.path.exists(_cache_path(w))


def _json_safe(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _save_cache(
    w: int,
    neutral_data: Dict[str, np.ndarray],
    tiled_data: Dict[str, np.ndarray],
    entries: list,
    probe_meta: Dict[str, Any],
) -> None:
    np.savez_compressed(
        _cache_path(w),
        entries_json=np.array([_json_safe(entries)], dtype=object),
        probe_meta_json=np.array([_json_safe(probe_meta)], dtype=object),
        **{f"neutral_{k}": v for k, v in neutral_data.items()},
        **{f"tiled_{k}": v for k, v in tiled_data.items()},
    )


def _load_cache(w: int):
    data = np.load(_cache_path(w), allow_pickle=True)
    entries = json.loads(str(data["entries_json"][0]))
    probe_meta = (
        json.loads(str(data["probe_meta_json"][0]))
        if "probe_meta_json" in data.files
        else {}
    )
    n_data, t_data = {}, {}
    for key in data.files:
        if key.startswith("neutral_"):
            n_data[key[8:]] = data[key]
        elif key.startswith("tiled_"):
            t_data[key[6:]] = data[key]
    return n_data, t_data, entries, probe_meta


# ── model + probe ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_lib(w: int):
    return build_model(w)


def read_window_state(model) -> Dict[str, Any]:
    inn = model.interneuronNetwork
    state: Dict[str, Any] = {}
    try:
        with torch.no_grad():
            if hasattr(inn, "floatWindowSizes_used"):
                state["long_sizes"] = inn.floatWindowSizes_used.detach().cpu().float().tolist()
            if hasattr(inn, "cerebellumSoft"):
                state["long_weights"] = inn.cerebellumSoft.detach().cpu().float().tolist()
            if hasattr(inn, "floatWindowSizes_short_used"):
                state["short_sizes"] = inn.floatWindowSizes_short_used.detach().cpu().float().tolist()
            if hasattr(inn, "cerebellumSoft_short"):
                state["short_weights"] = inn.cerebellumSoft_short.detach().cpu().float().tolist()
            if hasattr(inn, "short_gate_used"):
                state["short_gate"] = float(inn.short_gate_used)
            elif hasattr(inn, "short_window_gate"):
                state["short_gate"] = float(torch.sigmoid(inn.short_window_gate).item())
            if hasattr(inn, "window_softmax_temperature"):
                state["long_temp"] = float(torch.exp(inn.window_softmax_temperature).clamp(min=0.01).item())
            if hasattr(inn, "window_softmax_temperature_short"):
                state["short_temp"] = float(torch.exp(inn.window_softmax_temperature_short).clamp(min=0.01).item())
    except Exception as e:
        print(f"[read_window_state] {e}")
    return state


def read_gate_values(model) -> Dict[str, float]:
    gate_map = {
        "attn1": "attention.logit_gate",
        "attn2": "attention2.logit_gate",
        "scratch": "scratchpad.output_gate",
        "mem1": "memory.output_gate",
        "mem2": "memory2.output_gate",
        "tangle_e": "tangling.embed_gate",
        "tangle_n": "tangling.neuron_gate",
        "tangle_m": "tangling.memory_gate",
    }
    vals: Dict[str, float] = {}
    for layer_key, path in gate_map.items():
        try:
            param = reduce(getattr, path.split("."), model)
            vals[layer_key] = float(torch.sigmoid(param).item())
        except AttributeError:
            pass
    return vals


def run_probe(w: int, force: bool = False):
    if not force and _cache_exists(w):
        n_data, t_data, entries, probe_meta = _load_cache(w)
        if len(entries) >= int(N_VOCAB_SAMPLE * 0.9):
            st.sidebar.success(f"Loaded from cache  (w={w}, {len(entries)} tokens)")
            return n_data, t_data, entries, probe_meta

    model, librarian = load_model_and_lib(w)
    neutral_id = librarian.tokenToIndex.get(NEUTRAL_FILL, 0)

    with st.spinner("Sampling vocab tokens …"):
        entries = sample_tokens(
            librarian,
            N_VOCAB_SAMPLE,
            LABEL_WORDS,
            NEUTRAL_FILL,
            RANDOM_SEED,
        )

    layer_defs = _ALL_LAYER_DEFS

    with st.spinner(f"Probing {len(entries)} tokens × neutral prefix …"):
        n_data = probe_tokens(model, entries, neutral_id, w, layer_defs, strategy="neutral")

    with st.spinner(f"Probing {len(entries)} tokens × tiled …"):
        t_data = probe_tokens(model, entries, neutral_id, w, layer_defs, strategy="tiled")

    probe_meta = {
        "window_state": read_window_state(model),
        "gate_vals": read_gate_values(model),
    }

    _save_cache(w, n_data, t_data, entries, probe_meta)
    st.sidebar.success(f"Probe complete — cached  (w={w})")
    return n_data, t_data, entries, probe_meta


# ── embedding ─────────────────────────────────────────────────────────────────
def _joint_embed_key(
    n_data: Dict[str, np.ndarray],
    t_data: Dict[str, np.ndarray],
    layer_keys: list,
) -> str:
    h = hashlib.blake2b(digest_size=20, person=b"joint-embed")
    for k in layer_keys:
        h.update(k.encode("utf-8"))
        h.update(_blake_array(n_data[k], person="joint-n").encode("utf-8"))
        h.update(_blake_array(t_data[k], person="joint-t").encode("utf-8"))
    return h.hexdigest()


def _split_joint_2d(coords: np.ndarray, meta: list, n: int):
    n_xy, t_xy = {}, {}
    for i, (strat, k) in enumerate(meta):
        (n_xy if strat == "neutral" else t_xy)[k] = coords[i * n:(i + 1) * n]
    return n_xy, t_xy


def _prepare_joint_embed_inputs(
    n_data: Dict[str, np.ndarray],
    t_data: Dict[str, np.ndarray],
    layer_keys: list,
    show_progress: bool = False,
):
    from sklearn.decomposition import PCA

    key = _joint_embed_key(n_data, t_data, layer_keys)
    cache = st.session_state.setdefault("embed_input_cache", {})
    if key in cache:
        return key, cache[key]

    progress = None
    total_blocks = max(1, 2 * len(layer_keys))
    done = 0
    if show_progress:
        progress = st.progress(0.0, text=f"Embedding prep · reducing {total_blocks} layer blocks …")

    n = len(next(iter(n_data.values())))
    blocks, meta = [], []

    for strat, data in [("neutral", n_data), ("tiled", t_data)]:
        for k in layer_keys:
            arr = data[k].astype(np.float32)
            d = min(50, arr.shape[1], n - 1)
            if d < 2:
                block = np.zeros((n, 50), dtype=np.float32)
            else:
                block = PCA(n_components=d, random_state=42).fit_transform(arr)
                if block.shape[1] < 50:
                    block = np.hstack([block, np.zeros((n, 50 - block.shape[1]), dtype=np.float32)])
                block = block.astype(np.float32)
            blocks.append(block)
            meta.append((strat, k))
            done += 1
            if progress is not None:
                frac = 0.58 * (done / total_blocks)
                progress.progress(frac, text=f"Embedding prep · PCA {done}/{total_blocks}: {strat} · {k}")

    stacked = np.vstack(blocks)
    cache[key] = (stacked, meta, n)

    if progress is not None:
        progress.progress(0.62, text=f"Embedding prep complete · {len(stacked):,} points ready")
        progress.empty()

    return key, cache[key]


def compute_joint_2d(
    n_data: Dict[str, np.ndarray],
    t_data: Dict[str, np.ndarray],
    layer_keys: list,
    method: str = "pca",
):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    method = method.lower()
    if method not in ("pca", "tsne"):
        raise ValueError(f"Unsupported embedding method: {method}")

    h, prepared = _prepare_joint_embed_inputs(n_data, t_data, layer_keys, show_progress=(method == "tsne"))
    stacked, meta, n = prepared
    cache = st.session_state.setdefault("embed_cache", {})
    cache_key = f"{h}:{method}"

    if cache_key not in cache:
        if method == "pca":
            with st.spinner(f"PCA preview · {len(stacked):,} points …"):
                if stacked.shape[0] < 2:
                    coords = np.zeros((stacked.shape[0], 2), dtype=np.float32)
                else:
                    coords = PCA(
                        n_components=min(2, stacked.shape[1], stacked.shape[0] - 1),
                        random_state=42,
                    ).fit_transform(stacked)
                    if coords.shape[1] < 2:
                        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
                cache[cache_key] = coords.astype(np.float32)
        else:
            progress = st.progress(0.68, text="t-SNE launch · sklearn has no in-call progress callback …")
            perp = min(max(2, n // 5), 40)
            cache[cache_key] = TSNE(
                n_components=2,
                perplexity=perp,
                random_state=42,
                max_iter=1000,
            ).fit_transform(stacked).astype(np.float32)
            progress.progress(1.0, text=f"t-SNE complete · {len(stacked):,} points cached")
            progress.empty()

    return _split_joint_2d(cache[cache_key], meta, n)


def stabilize_layer_view(n_xy: Dict[str, np.ndarray], t_xy: Dict[str, np.ndarray], layer_keys: list):
    centers = []
    radii = []
    for lk in layer_keys:
        combined = np.vstack([n_xy[lk], t_xy[lk]]).astype(np.float32)
        center = combined.mean(axis=0)
        radius = float(np.sqrt(np.mean(np.sum((combined - center) ** 2, axis=1))))
        centers.append(center)
        radii.append(max(radius, 1e-6))

    target_center = np.mean(np.stack(centers, axis=0), axis=0)
    target_radius = float(np.median(np.asarray(radii, dtype=np.float32)))

    stab_n, stab_t = {}, {}
    for lk, center, radius in zip(layer_keys, centers, radii):
        scale = target_radius / max(radius, 1e-6)
        stab_n[lk] = ((n_xy[lk] - center) * scale + target_center).astype(np.float32)
        stab_t[lk] = ((t_xy[lk] - center) * scale + target_center).astype(np.float32)
    return stab_n, stab_t


def _brightness_colors(base_hex_list, activations):
    import colorsys
    a = np.asarray(activations, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return list(base_hex_list)

    result = []
    for hex_c, val in zip(base_hex_list, a):
        t = (float(val) - lo) / (hi - lo)
        h_str = hex_c.lstrip("#")
        r, g, b = (int(h_str[i:i + 2], 16) / 255 for i in (0, 2, 4))
        hue, _, sat = colorsys.rgb_to_hls(r, g, b)
        new_l = 0.07 + t * 0.83
        nr, ng, nb = colorsys.hls_to_rgb(hue, new_l, max(sat, 0.55))
        result.append(f"#{int(nr*255):02x}{int(ng*255):02x}{int(nb*255):02x}")
    return result


# ── stats ─────────────────────────────────────────────────────────────────────
def compute_layer_stats(n_data: Dict[str, np.ndarray], layer_keys: list):
    from sklearn.decomposition import PCA

    cache = st.session_state.setdefault("lstats_cache", {})
    h = hashlib.blake2b(digest_size=20, person=b"layer-stats")
    for k in layer_keys:
        h.update(k.encode("utf-8"))
        h.update(_blake_array(n_data[k], person="lstats").encode("utf-8"))
    key = h.hexdigest()

    if key in cache:
        return cache[key]

    stats = {}
    for k in layer_keys:
        arr = n_data[k].astype(np.float32)
        norms = np.linalg.norm(arr, axis=1)
        n, d = arr.shape
        nc = min(50, d, n - 1)
        if nc >= 2:
            ev = PCA(n_components=nc, random_state=42).fit(arr).explained_variance_ratio_
            pr = float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-12))
            top1 = float(ev[0])
        else:
            pr, top1 = 1.0, 1.0

        top_neurons = top_vals = None
        if d >= 500:
            K = 5
            abs_arr = np.abs(arr)
            topk_idx = np.argpartition(abs_arr, -K, axis=1)[:, -K:]
            topk_vals = arr[np.arange(n)[:, None], topk_idx]
            order = np.argsort(np.abs(topk_vals), axis=1)[:, ::-1]
            top_neurons = topk_idx[np.arange(n)[:, None], order]
            top_vals = topk_vals[np.arange(n)[:, None], order]

        stats[k] = {
            "norms": norms,
            "participation_ratio": pr,
            "top1_explained": top1,
            "top_neurons": top_neurons,
            "top_vals": top_vals,
        }

    cache[key] = stats
    return stats


def compute_layer_pca_loadings(arr: np.ndarray, n_components: int = 8):
    from sklearn.decomposition import PCA

    cache = st.session_state.setdefault("layer_pca_cache", {})
    key = f"{_blake_array(arr, person='layer-pca')}:{n_components}"
    if key in cache:
        return cache[key]

    a = arr.astype(np.float32)
    max_nc = min(n_components, a.shape[0] - 1, a.shape[1] - 1)
    if max_nc < 1:
        loadings = np.zeros((a.shape[0], 1), dtype=np.float32)
        explained = np.array([1.0], dtype=np.float32)
    else:
        pca = PCA(n_components=max_nc, random_state=42)
        loadings = pca.fit_transform(a).astype(np.float32)
        explained = pca.explained_variance_ratio_.astype(np.float32)

    cache[key] = (loadings, explained)
    return loadings, explained


def _resolve_trail_strategy(show_neutral: bool, show_tiled: bool, trail_source: str) -> str:
    if trail_source == "neutral":
        return "neutral"
    if trail_source == "tiled":
        return "tiled"
    return "neutral" if (show_neutral or not show_tiled) else "tiled"


def build_overlay_values(
    active_layers: list,
    data_n: Dict[str, np.ndarray],
    data_t: Dict[str, np.ndarray],
    overlay_mode: str,
    spotlight_neuron_idx: Optional[int],
    spotlight_range: Optional[Tuple[int, int]],
    spotlight_pca_k: Optional[int],
):
    if overlay_mode == "off":
        return None

    spotlight_vals = {}
    for lk in active_layers:
        spotlight_vals[lk] = {"neutral": None, "tiled": None}
        arr_n = data_n[lk].astype(np.float32)
        arr_t = data_t[lk].astype(np.float32)

        if overlay_mode == "single neuron" and spotlight_neuron_idx is not None:
            if arr_n.ndim == 2 and arr_n.shape[1] > spotlight_neuron_idx:
                spotlight_vals[lk]["neutral"] = arr_n[:, spotlight_neuron_idx]
            if arr_t.ndim == 2 and arr_t.shape[1] > spotlight_neuron_idx:
                spotlight_vals[lk]["tiled"] = arr_t[:, spotlight_neuron_idx]

        elif overlay_mode == "L2 norm  (all neurons)":
            spotlight_vals[lk]["neutral"] = np.linalg.norm(arr_n, axis=1)
            spotlight_vals[lk]["tiled"] = np.linalg.norm(arr_t, axis=1)

        elif overlay_mode == "neuron range  (mean |act|)" and spotlight_range is not None:
            n_from, n_to = spotlight_range
            if arr_n.ndim == 2 and arr_n.shape[1] > n_to:
                spotlight_vals[lk]["neutral"] = np.abs(arr_n[:, n_from:n_to + 1]).mean(axis=1)
            if arr_t.ndim == 2 and arr_t.shape[1] > n_to:
                spotlight_vals[lk]["tiled"] = np.abs(arr_t[:, n_from:n_to + 1]).mean(axis=1)

        elif overlay_mode == "layer PCA component" and spotlight_pca_k is not None:
            load_n, _ = compute_layer_pca_loadings(arr_n, n_components=max(spotlight_pca_k, 8))
            load_t, _ = compute_layer_pca_loadings(arr_t, n_components=max(spotlight_pca_k, 8))
            if load_n.shape[1] >= spotlight_pca_k:
                spotlight_vals[lk]["neutral"] = load_n[:, spotlight_pca_k - 1].astype(np.float32)
            if load_t.shape[1] >= spotlight_pca_k:
                spotlight_vals[lk]["tiled"] = load_t[:, spotlight_pca_k - 1].astype(np.float32)

    return spotlight_vals



# ── analysis figure ───────────────────────────────────────────────────────────
def build_analysis_figure(
    n_xy,
    t_xy,
    layer_keys,
    entries,
    show_neutral,
    show_tiled,
    show_diff,
    show_trail,
    trail_source,
    custom_ids=None,
    layer_stats=None,
    spotlight_vals=None,
):
    """
    Leaner Analysis mode: fewer traces than the old version, still scientific.
    Traces:
      0 combined ghost history
      1 combined trail line
      2 diff lines
      3 neutral markers
      4 tiled markers
      5 tracked markers
    """
    if custom_ids is None:
        custom_ids = set()

    n = len(entries)
    trail_strat = _resolve_trail_strategy(show_neutral, show_tiled, trail_source)
    trail_xy = n_xy if trail_strat == "neutral" else t_xy

    colors = []
    syms_n = []
    syms_t = []
    base_hover = []

    for e in entries:
        fc = _fine_cat(e)
        colors.append(CAT_COLOR.get(fc, "#aaa"))
        syms_n.append("star" if e["labeled"] else "circle")
        syms_t.append("star-diamond" if e["labeled"] else "diamond")
        lbl = e["label"] if e["labeled"] else e["tok"]
        base_hover.append(f"<b>{lbl}</b><br>id: {e['id']}<br>cat: {e['cat']}")

    def _sizes_for_layer(lk, base_n=9.0, base_t=6.0):
        if layer_stats and lk in layer_stats:
            raw = layer_stats[lk]["norms"]
            lo, hi = raw.min(), raw.max()
            if hi - lo > 1e-6:
                scale = (raw - lo) / (hi - lo)
                sn = (base_n * 0.90 + scale * base_n * 0.35).tolist()
                st_ = (base_t * 0.90 + scale * base_t * 0.30).tolist()
            else:
                sn = [base_n] * n
                st_ = [base_t] * n
        else:
            sn = [base_n] * n
            st_ = [base_t] * n
        for i, e in enumerate(entries):
            if e["labeled"]:
                sn[i] = max(sn[i], 13.0)
                st_[i] = max(st_[i], 9.2)
        return sn, st_

    def _hover_for_layer(lk):
        if not (layer_stats and lk in layer_stats):
            return base_hover
        ls = layer_stats[lk]
        norms = ls["norms"]
        top_neurons = ls.get("top_neurons")
        top_vals = ls.get("top_vals")
        result = []
        for i in range(n):
            h = f"{base_hover[i]}<br>‖v‖ = {norms[i]:.1f}"
            if top_neurons is not None:
                parts = "  ".join(
                    f"<b>#{top_neurons[i,j]}</b>:{top_vals[i,j]:+.2f}"
                    for j in range(top_neurons.shape[1])
                )
                h += f"<br><span style='font-size:10px;color:#aaa'>top: {parts}</span>"
            result.append(h)
        return result

    def _colors_for_layer(lk, strat):
        arr = None
        if spotlight_vals and lk in spotlight_vals:
            arr = spotlight_vals[lk].get(strat)
        return list(colors) if arr is None else _brightness_colors(colors, arr)

    def _ghost_history(fi: int):
        if not show_trail or fi <= 0:
            return go.Scattergl(
                x=[], y=[], mode="markers",
                hoverinfo="skip",
                name="layer trail ···",
                legendgroup="trail",
                showlegend=(fi == 0),
            )
        recent = list(range(max(0, fi - 6), fi))
        total = max(1, len(recent))
        xs, ys, cols = [], [], []
        rgb = np.asarray([[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for c in colors], dtype=np.int16)
        for j, idx in enumerate(recent):
            alpha = 0.02 + 0.10 * ((j + 1) / total)
            pos = trail_xy[layer_keys[idx]]
            xs.extend(pos[:, 0].tolist())
            ys.extend(pos[:, 1].tolist())
            cols.extend([f"rgba({r},{g},{b},{alpha:.4f})" for r, g, b in rgb])
        return go.Scattergl(
            x=xs, y=ys, mode="markers",
            marker=dict(size=4.2, color=cols),
            hoverinfo="skip",
            name="layer trail ···",
            legendgroup="trail",
            showlegend=(fi == 0),
        )

    def _trail_path(fi: int):
        if not show_trail or fi <= 0:
            return go.Scatter(
                x=[], y=[], mode="lines",
                hoverinfo="skip",
                name="_trailpath",
                legendgroup="trail",
                showlegend=False,
            )
        src = trail_xy[layer_keys[max(0, fi - 1)]]
        dst = trail_xy[layer_keys[fi]]
        xs, ys = [], []
        for i in range(n):
            xs += [float(src[i, 0]), float(dst[i, 0]), None]
            ys += [float(src[i, 1]), float(dst[i, 1]), None]
        return go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="rgba(180,220,255,0.025)", width=0.45),
            hoverinfo="skip",
            name="_trailpath",
            legendgroup="trail",
            showlegend=False,
        )

    def frame_traces(fi: int, lk: str):
        nxy = n_xy[lk]
        txy = t_xy[lk]
        sizes_n, sizes_t = _sizes_for_layer(lk)
        hover_lk = _hover_for_layer(lk)

        diff_trace = go.Scatter(
            x=[], y=[], mode="lines",
            line=dict(color="rgba(255,210,0,0.05)", width=0.35),
            hoverinfo="skip",
            name="strat diff —",
            legendgroup="diff",
            showlegend=(fi == 0),
        )
        if show_diff:
            xs, ys = [], []
            for i in range(n):
                xs += [float(nxy[i, 0]), float(txy[i, 0]), None]
                ys += [float(nxy[i, 1]), float(txy[i, 1]), None]
            diff_trace = go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color="rgba(255,210,0,0.05)", width=0.35),
                hoverinfo="skip",
                name="strat diff —",
                legendgroup="diff",
                showlegend=(fi == 0),
            )

        neutral_trace = go.Scattergl(
            x=nxy[:, 0].tolist() if show_neutral else [],
            y=nxy[:, 1].tolist() if show_neutral else [],
            mode="markers",
            marker=dict(
                color=_colors_for_layer(lk, "neutral"),
                size=sizes_n,
                symbol=syms_n,
                line=dict(width=0.25, color="#333"),
                opacity=0.94,
            ),
            hovertext=hover_lk if show_neutral else [],
            hoverinfo="text",
            name="neutral ●",
            legendgroup="neutral",
            showlegend=(fi == 0),
        )

        tiled_trace = go.Scattergl(
            x=txy[:, 0].tolist() if show_tiled else [],
            y=txy[:, 1].tolist() if show_tiled else [],
            mode="markers",
            marker=dict(
                color=_colors_for_layer(lk, "tiled"),
                size=sizes_t,
                symbol=syms_t,
                line=dict(width=0.22, color="#555"),
                opacity=0.68,
            ),
            hovertext=hover_lk if show_tiled else [],
            hoverinfo="text",
            name="tiled ◆",
            legendgroup="tiled",
            showlegend=(fi == 0),
        )

        ci = [i for i, e in enumerate(entries) if e["id"] in custom_ids]
        if ci:
            track_xy = nxy if (show_neutral or not show_tiled) else txy
            cx = [float(track_xy[i, 0]) for i in ci]
            cy = [float(track_xy[i, 1]) for i in ci]
            clbl = [entries[i]["label"] if entries[i]["labeled"] else entries[i]["tok"] for i in ci]
            chov = [f"<b>★ {clbl[j]}</b><br>id: {entries[ci[j]]['id']}" for j in range(len(ci))]
            tracked_trace = go.Scattergl(
                x=cx, y=cy, mode="markers+text",
                marker=dict(color="#cc66ff", size=16, symbol="diamond", line=dict(width=1.2, color="white")),
                text=clbl, textposition="top center",
                textfont=dict(size=10, color="#cc66ff"),
                hovertext=chov, hoverinfo="text",
                name="tracked ◆", legendgroup="custom",
                showlegend=(fi == 0),
            )
        else:
            tracked_trace = go.Scattergl(
                x=[], y=[], mode="markers",
                hoverinfo="skip", name="tracked ◆", legendgroup="custom",
                showlegend=(fi == 0),
            )

        return [
            _ghost_history(fi),
            _trail_path(fi),
            diff_trace,
            neutral_trace,
            tiled_trace,
            tracked_trace,
        ]

    frames = [go.Frame(data=frame_traces(fi, lk), name=lk) for fi, lk in enumerate(layer_keys)]

    all_x, all_y = [], []
    include_tiled_bounds = show_tiled or show_diff
    for k in layer_keys:
        all_x.extend(n_xy[k][:, 0].tolist())
        all_y.extend(n_xy[k][:, 1].tolist())
        if include_tiled_bounds:
            all_x.extend(t_xy[k][:, 0].tolist())
            all_y.extend(t_xy[k][:, 1].tolist())
    xpad = (max(all_x) - min(all_x)) * 0.07 if all_x else 1.0
    ypad = (max(all_y) - min(all_y)) * 0.07 if all_y else 1.0
    x_range = [min(all_x) - xpad, max(all_x) + xpad]
    y_range = [min(all_y) - ypad, max(all_y) + ypad]

    steps = [
        dict(
            args=[[f.name], dict(frame=dict(duration=260, redraw=False), mode="immediate", transition=dict(duration=200, easing="linear"))],
            label=f.name,
            method="animate",
        )
        for f in frames
    ]

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        paper_bgcolor="#0e0e0e",
        plot_bgcolor="#0e0e0e",
        font=dict(color="#aaaaaa"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=x_range, autorange=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=y_range, autorange=False, scaleanchor="x", scaleratio=1),
        legend=dict(
            bgcolor="#111",
            bordercolor="#333",
            font=dict(size=11, color="#ccc"),
            x=1.01, xanchor="left", y=1,
            itemclick="toggleothers",
            itemdoubleclick="toggle",
        ),
        updatemenus=[dict(
            type="buttons", showactive=False, direction="left",
            x=0.0, xanchor="left", y=-0.07, yanchor="top",
            buttons=[
                dict(label="▶ play", method="animate",
                     args=[None, dict(frame=dict(duration=320, redraw=False), fromcurrent=True, mode="immediate", transition=dict(duration=220, easing="linear"))]),
                dict(label="⏸ pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
            ],
            bgcolor="#1a1a1a", bordercolor="#444",
            font=dict(color="#ddd", size=12),
        )],
        sliders=[dict(
            active=0, steps=steps,
            x=0.08, xanchor="left", y=-0.03, yanchor="top", len=0.90,
            currentvalue=dict(prefix="layer: ", font=dict(color="#cccccc", size=14), xanchor="center"),
            font=dict(color="#777", size=10),
            bgcolor="#1a1a1a", bordercolor="#444", tickcolor="#555",
            transition=dict(duration=200, easing="linear"),
        )],
        margin=dict(l=10, r=170, t=50, b=130),
        height=740,
    )
    return fig


# ── charts for analysis panels ────────────────────────────────────────────────
def window_attention_fig(ws: Dict[str, Any], window_max: int):
    if not ws:
        return go.Figure()

    fig = go.Figure()
    long_sizes = np.asarray(ws.get("long_sizes", []), dtype=np.float32)
    long_weights = np.asarray(ws.get("long_weights", []), dtype=np.float32)
    short_sizes = np.asarray(ws.get("short_sizes", []), dtype=np.float32)
    short_weights = np.asarray(ws.get("short_weights", []), dtype=np.float32)
    gate = float(ws.get("short_gate", 0.0))
    l_tmp = float(ws.get("long_temp", 1.0))
    s_tmp = float(ws.get("short_temp", 1.0))

    if len(long_sizes) and len(long_weights):
        order = np.argsort(long_sizes)
        fig.add_trace(go.Bar(
            x=long_weights[order].tolist(),
            y=[f"{long_sizes[i]:.1f} tok" for i in order],
            orientation="h",
            name="long windows",
            marker=dict(color="rgba(79,195,247,0.80)", line=dict(color="rgba(79,195,247,1)", width=0.5)),
            hovertemplate="window: <b>%{y}</b><br>attention: %{x:.4f}<extra>long</extra>",
        ))

    if len(short_sizes) and len(short_weights):
        order = np.argsort(short_sizes)
        scaled = short_weights[order] * gate
        fig.add_trace(go.Bar(
            x=scaled.tolist(),
            y=[f"{short_sizes[i]:.1f} tok" for i in order],
            orientation="h",
            name=f"short windows  ×gate={gate:.4f}",
            marker=dict(color="rgba(255,138,101,0.80)", line=dict(color="rgba(255,138,101,1)", width=0.5)),
            hovertemplate="window: <b>%{y}</b><br>attn×gate: %{x:.4f}<extra>short</extra>",
        ))

    all_x = np.concatenate([
        long_weights if len(long_weights) else np.array([0]),
        (short_weights * gate if len(short_weights) else np.array([0])),
    ])
    x_max = float(all_x.max()) * 1.18 + 1e-6

    fig.update_layout(
        paper_bgcolor="#0e0e0e", plot_bgcolor="#0e0e0e",
        font=dict(color="#aaa", size=9),
        height=420,
        margin=dict(l=85, r=20, t=50, b=40),
        barmode="overlay",
        title=dict(
            text=(f"cerebellum · temporal attention  "
                  f"(long temp={l_tmp:.2f} · short temp={s_tmp:.2f} · "
                  f"short gate={gate:.4f} · context={window_max} tok)"),
            font=dict(color="#ccc", size=11),
        ),
        xaxis=dict(title="softmax attention weight", showgrid=True, gridcolor="#1a1a1a", range=[0, x_max], tickfont=dict(size=8)),
        yaxis=dict(showgrid=False, tickfont=dict(size=8)),
        legend=dict(bgcolor="#111", bordercolor="#333", font=dict(size=9), x=1.01, xanchor="left"),
    )
    return fig


def intrinsic_dim_chart(stats, layer_keys, gate_vals=None):
    prs = [stats[k]["participation_ratio"] for k in layer_keys]
    top1s = [stats[k]["top1_explained"] for k in layer_keys]
    max_pr = max(prs) if prs else 1.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=layer_keys, x=[p / max_pr for p in prs], orientation="h",
        name="intrinsic dim  (relative ↔)",
        marker_color="#4fc3f7", opacity=0.85,
        hovertemplate="<b>%{y}</b><br>PR = %{customdata:.1f}  (%{x:.0%} of max)<extra></extra>",
        customdata=prs,
    ))
    fig.add_trace(go.Bar(
        y=layer_keys, x=top1s, orientation="h",
        name="PC-1 explained var  (high = collapsed)",
        marker_color="#ff7043", opacity=0.60,
        hovertemplate="<b>%{y}</b><br>PC-1 = %{x:.1%}<extra></extra>",
    ))

    if gate_vals:
        gk = [k for k in layer_keys if k in gate_vals]
        gv = [gate_vals[k] for k in gk]
        fig.add_trace(go.Scatter(
            y=gk, x=gv, mode="markers",
            name="gate value",
            marker=dict(color="#cc66ff", size=11, symbol="diamond", line=dict(width=1, color="white")),
            hovertemplate="<b>%{y}</b><br>gate = %{x:.4f}<extra></extra>",
        ))

    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="#0e0e0e", plot_bgcolor="#0e0e0e",
        font=dict(color="#aaa", size=10),
        height=max(280, len(layer_keys) * 26 + 80),
        margin=dict(l=90, r=20, t=50, b=40),
        title=dict(text="intrinsic dimensionality · PC-1 collapse · gate values", font=dict(color="#ccc", size=12)),
        xaxis=dict(showgrid=True, gridcolor="#1e1e1e", range=[0, 1.05]),
        yaxis=dict(showgrid=False, autorange="reversed"),
        legend=dict(bgcolor="#111", bordercolor="#333", font=dict(size=9), x=1.01, xanchor="left"),
    )
    return fig


def compute_neuron_atlas(neurons_arr: np.ndarray):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    cache = st.session_state.setdefault("natlas_cache", {})
    key = _blake_array(neurons_arr, person="neuron-atlas")
    if key in cache:
        return cache[key]

    arr = neurons_arr.astype(np.float32)
    profiles = arr.T
    d = min(50, profiles.shape[1] - 1, profiles.shape[0] - 1)
    reduced = PCA(n_components=d, random_state=42).fit_transform(profiles)
    perp = min(50, max(5, reduced.shape[0] // 200))
    xy = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=500).fit_transform(reduced)
    cache[key] = xy
    return xy


def neuron_atlas_fig(atlas_xy, neurons_arr, entries, spotlight_idx=None):
    arr = neurons_arr.astype(np.float32)
    activity = np.abs(arr).mean(axis=0)
    std_act = arr.std(axis=0)

    top3_idx = np.argsort(arr, axis=0)[-3:][::-1]
    tok_labels = np.array([e["tok"] for e in entries])

    hover = [
        (f"neuron {j}<br>"
         f"mean|act|={activity[j]:.4f}  std={std_act[j]:.4f}<br>"
         f"top tokens: {tok_labels[top3_idx[0, j]]}, "
         f"{tok_labels[top3_idx[1, j]]}, {tok_labels[top3_idx[2, j]]}")
        for j in range(arr.shape[1])
    ]

    v_lo, v_hi = float(std_act.min()), float(std_act.max())
    if v_hi - v_lo > 1e-9:
        sizes = (2 + 6 * (std_act - v_lo) / (v_hi - v_lo)).tolist()
    else:
        sizes = [4.0] * len(std_act)

    colors = activity.tolist()
    marker = dict(
        size=sizes, color=colors, colorscale="Viridis", opacity=0.75, showscale=True,
        colorbar=dict(len=0.5, thickness=10, title=dict(text="mean |act|", font=dict(size=9, color="#aaa")), tickfont=dict(size=8, color="#aaa"))
    )

    fig = go.Figure(go.Scattergl(
        x=atlas_xy[:, 0].tolist(), y=atlas_xy[:, 1].tolist(),
        mode="markers", marker=marker,
        hovertext=hover, hoverinfo="text",
        name="neurons",
    ))

    if spotlight_idx is not None and 0 <= spotlight_idx < arr.shape[1]:
        fig.add_trace(go.Scatter(
            x=[float(atlas_xy[spotlight_idx, 0])],
            y=[float(atlas_xy[spotlight_idx, 1])],
            mode="markers",
            marker=dict(size=14, color="#ff5555", symbol="circle-open", line=dict(width=2, color="#ff5555")),
            hovertext=[f"spotlight: neuron {spotlight_idx}"],
            hoverinfo="text", name=f"spotlight n{spotlight_idx}",
        ))

    fig.update_layout(
        paper_bgcolor="#0e0e0e", plot_bgcolor="#0e0e0e",
        font=dict(color="#aaa"),
        height=520,
        margin=dict(l=20, r=20, t=45, b=20),
        title=dict(text="neuron atlas · 10 000 INN neurons", font=dict(color="#ccc", size=11)),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        legend=dict(bgcolor="#111", bordercolor="#333", font=dict(size=9)),
    )
    return fig


def neuron_heatmap_fig(neurons_arr, entries, n_top=64, max_tokens=500):
    import random as _rng_mod
    arr = neurons_arr.astype(np.float32)

    base_idx = [i for i, e in enumerate(entries) if e["labeled"] or e["cat"] in ("letter", "digit", "punct")]
    bpe_idx = [i for i, e in enumerate(entries) if not e["labeled"] and e["cat"] in ("bpe_word", "bpe_sub")]
    special_idx = [i for i, e in enumerate(entries) if not e["labeled"] and e["cat"] not in ("letter", "digit", "punct", "bpe_word", "bpe_sub")]

    n_bpe = max(0, max_tokens - len(base_idx) - len(special_idx))
    bpe_word = [i for i in bpe_idx if entries[i]["cat"] == "bpe_word"]
    bpe_sub = [i for i in bpe_idx if entries[i]["cat"] == "bpe_sub"]
    _r = _rng_mod.Random(42)
    _r.shuffle(bpe_word)
    _r.shuffle(bpe_sub)
    n_w = min(len(bpe_word), n_bpe // 2 + n_bpe % 2)
    n_s = min(len(bpe_sub), n_bpe - n_w)
    chosen_idx = sorted(set(base_idx + special_idx + bpe_word[:n_w] + bpe_sub[:n_s]))

    cat_order = {
        "labeled": 0, "letter_vowel": 1, "letter_cons": 2, "digit_lo": 3, "digit_hi": 4,
        "punct_sent": 5, "punct_math": 6, "punct_bracket": 7, "punct_quote": 8, "punct_other": 9,
        "bpe_word_short": 10, "bpe_word_long": 11, "bpe_sub_short": 12, "bpe_sub_long": 13, "special": 14,
    }

    def _key(i):
        e = entries[i]
        return (cat_order.get(_fine_cat(e), 99), e["tok"])

    chosen_idx = sorted(chosen_idx, key=_key)

    sub_arr = arr[chosen_idx]
    sub_entries = [entries[i] for i in chosen_idx]

    var = sub_arr.var(axis=0)
    top_idx = np.argsort(var)[::-1][:n_top]
    z = sub_arr[:, top_idx].T

    tok_labels = [e["label"] if e["labeled"] else e["tok"] for e in sub_entries]
    show_label = [lbl if (e["labeled"] or e["cat"] in ("letter", "digit")) else "" for e, lbl in zip(sub_entries, tok_labels)]
    x_pos = list(range(len(tok_labels)))

    hover_text = []
    for row_i in range(n_top):
        row = []
        for col_i, lbl in enumerate(tok_labels):
            row.append(f"token: <b>{lbl}</b><br>neuron: n{top_idx[row_i]}<br>activation: {z[row_i, col_i]:.3f}")
        hover_text.append(row)

    fig = go.Figure(go.Heatmap(
        z=z, x=x_pos, y=[f"n{top_idx[i]}" for i in range(n_top)],
        colorscale="RdBu_r", zmid=0,
        text=hover_text, hoverinfo="text",
        colorbar=dict(tickfont=dict(color="#aaa"), len=0.5, title=dict(text="activation", font=dict(color="#aaa", size=10))),
    ))
    fig.update_layout(
        paper_bgcolor="#0e0e0e", plot_bgcolor="#0e0e0e",
        font=dict(color="#aaa", size=9),
        height=560,
        margin=dict(l=60, r=20, t=50, b=100),
        title=dict(text=(f"INN neurons · top-{n_top} most variable · "
                         f"{len(chosen_idx)} individual tokens"), font=dict(color="#ccc", size=11)),
        xaxis=dict(tickvals=x_pos, ticktext=show_label, tickangle=-70, tickfont=dict(size=7), color="#666"),
        yaxis=dict(tickfont=dict(size=8), color="#777", autorange="reversed"),
    )
    return fig


def neuron_health_fig(neurons_arr, spotlight_idx=None, spotlight_range=None):
    arr = neurons_arr.astype(np.float32)
    activity = np.abs(arr).mean(axis=0)
    sorted_i = np.argsort(activity)
    y = activity[sorted_i]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=list(range(len(y))), y=y.tolist(), mode="markers",
        marker=dict(
            size=1.8, color=y.tolist(), colorscale="Viridis", opacity=0.85, showscale=True,
            colorbar=dict(len=0.6, thickness=8, title=dict(text="mean |act|", font=dict(size=9, color="#aaa")), tickfont=dict(size=8, color="#aaa")),
        ),
        hovertemplate="rank: %{x}<br>neuron id: %{customdata}<br>mean |act|: %{y:.5f}<extra></extra>",
        customdata=sorted_i.tolist(), name="neuron activity",
    ))

    if spotlight_idx is not None:
        hits = np.where(sorted_i == spotlight_idx)[0]
        if len(hits):
            r = int(hits[0])
            fig.add_vline(x=r, line_color="#ff5555", line_width=1.5,
                          annotation_text=f" n{spotlight_idx}",
                          annotation_font=dict(color="#ff5555", size=9),
                          annotation_position="top right")

    if spotlight_range is not None:
        n_from, n_to = spotlight_range
        in_range = np.where((sorted_i >= n_from) & (sorted_i <= n_to))[0]
        if len(in_range):
            fig.add_vrect(
                x0=int(in_range.min()), x1=int(in_range.max()),
                fillcolor="rgba(255,200,50,0.10)",
                line_color="rgba(255,200,50,0.35)", line_width=0.8,
                annotation_text=f" n{n_from}–{n_to}",
                annotation_font=dict(color="#ffc832", size=9),
            )

    threshold = float(y.max()) * 0.01 if len(y) else 0.0
    dead_count = int(np.searchsorted(y, threshold))
    if dead_count > 0:
        fig.add_vline(x=dead_count, line_color="#444", line_dash="dot", line_width=1,
                      annotation_text=f" >{dead_count} near-zero",
                      annotation_font=dict(color="#666", size=8),
                      annotation_position="top right")

    fig.update_layout(
        paper_bgcolor="#0e0e0e", plot_bgcolor="#0e0e0e",
        font=dict(color="#aaa"),
        height=210,
        margin=dict(l=55, r=20, t=38, b=35),
        title=dict(text="neuron health · mean |activation| across all tokens · sorted dead → alive", font=dict(color="#ccc", size=11)),
        xaxis=dict(title="neuron rank", showgrid=False, color="#555", tickfont=dict(size=8)),
        yaxis=dict(title="mean |act|", showgrid=True, gridcolor="#1a1a1a", tickfont=dict(size=8)),
    )
    return fig


def cosine_heatmap_fig(arr, labels, title):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    normed = arr / (norms + 1e-9)
    cos = 1.0 - cdist(normed, normed, metric="cosine")

    annotations = [
        dict(x=j, y=i, text=f"{cos[i, j]:.2f}", showarrow=False,
             font=dict(size=8, color="white" if abs(cos[i, j]) < 0.65 else "#111"))
        for i in range(len(labels)) for j in range(len(labels))
    ]
    fig = go.Figure(go.Heatmap(
        z=cos, x=labels, y=labels,
        colorscale="RdBu", zmin=-1, zmax=1,
        colorbar=dict(tickfont=dict(color="#aaa"), len=0.6),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#ccc", size=11)),
        paper_bgcolor="#0e0e0e", plot_bgcolor="#0e0e0e",
        font=dict(color="#aaa"),
        legend=dict(bgcolor="#111", bordercolor="#333"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=420,
        annotations=annotations,
    )
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9), color="#aaa", showgrid=False, zeroline=False, showticklabels=True)
    fig.update_yaxes(tickfont=dict(size=9), color="#aaa", showgrid=False, zeroline=False, showticklabels=True)
    return fig


# ── main ──────────────────────────────────────────────────────────────────────
st.title("ʕっʘ‿ʘʔっ babyLLM · latent space probe")
st.caption(
    f"window = **{window_max}** tokens  ·  "
    f"active layers: **{', '.join(active_layers) if active_layers else 'none'}**"
)

if not active_layers:
    st.warning("No layers selected — tick at least one in the sidebar.")
    st.stop()

if not (show_neutral or show_tiled or show_diff):
    st.warning("No traces selected — tick at least one in the Traces section.")
    st.stop()

if not _cache_exists(window_max):
    neutral_data, tiled_data, entries, probe_meta = run_probe(window_max, force=False)
else:
    neutral_data, tiled_data, entries, probe_meta = run_probe(window_max, force=force_rerun)

if force_rerun:
    for key in [
        "embed_cache",
        "embed_input_cache",
        "lstats_cache",
        "layer_pca_cache",
        "natlas_cache",
        "natlas_requested",
    ]:
        st.session_state.pop(key, None)
    st.rerun()

# ── tracked tokens ────────────────────────────────────────────────────────────
custom_ids = set()
tracked_idx = set()
missing_toks = []

if track_input.strip():
    tok_lookup = {e["tok"]: (i, e["id"]) for i, e in enumerate(entries)}
    for word in track_input.strip().lower().split():
        candidates = [word, "Ġ" + word, " " + word]
        found = False
        for c in candidates:
            if c in tok_lookup:
                idx, tok_id = tok_lookup[c]
                custom_ids.add(tok_id)
                tracked_idx.add(idx)
                found = True
                break
        if not found:
            missing_toks.append(word)

    if custom_ids:
        found_toks = [entries[i]["tok"] for i in sorted(tracked_idx)]
        st.info(f"Tracking: **{', '.join(found_toks)}**")
    if missing_toks:
        st.warning(f"Not in probed vocab sample: **{', '.join(missing_toks)}**")

# ── subsets ───────────────────────────────────────────────────────────────────
# Analysis subset
_base_idx = [i for i, e in enumerate(entries) if e["cat"] in ("letter", "digit", "punct")]
_labeled_idx = [i for i, e in enumerate(entries) if e["labeled"]]
_bpe_idx = [i for i, e in enumerate(entries) if e["cat"] in ("bpe_word", "bpe_sub") and not e["labeled"]]
_must_have = sorted(set(_base_idx) | set(_labeled_idx) | tracked_idx)
_bpe_budget = max(0, scatter_max - len(_must_have))
_rng_sc = random.Random(42)
_bpe_sample = _rng_sc.sample(_bpe_idx, min(_bpe_budget, len(_bpe_idx)))
_analysis_idx = sorted(set(_must_have) | set(_bpe_sample))

analysis_entries = [entries[i] for i in _analysis_idx]
analysis_n_data = {k: neutral_data[k][_analysis_idx] for k in active_layers}
analysis_t_data = {k: tiled_data[k][_analysis_idx] for k in active_layers}

# ── embedding ─────────────────────────────────────────────────────────────────
vis_entries = analysis_entries
vis_n_data = analysis_n_data
vis_t_data = analysis_t_data

neutral_xy, tiled_xy = compute_joint_2d(vis_n_data, vis_t_data, active_layers, method="tsne")
if stabilize_view:
    neutral_xy, tiled_xy = stabilize_layer_view(neutral_xy, tiled_xy, active_layers)

with st.spinner("Computing layer statistics …"):
    vis_layer_stats = compute_layer_stats(vis_n_data, active_layers)

gate_vals = probe_meta.get("gate_vals", {})
window_state = probe_meta.get("window_state", {})

spotlight_vals = build_overlay_values(
    active_layers=active_layers,
    data_n=vis_n_data,
    data_t=vis_t_data,
    overlay_mode=overlay_mode,
    spotlight_neuron_idx=spotlight_neuron_idx,
    spotlight_range=spotlight_range,
    spotlight_pca_k=spotlight_pca_k,
)

# ── main animated figure ──────────────────────────────────────────────────────
fig = build_analysis_figure(
    neutral_xy,
    tiled_xy,
    active_layers,
    vis_entries,
    show_neutral,
    show_tiled,
    show_diff,
    show_trail,
    "auto",
    custom_ids=custom_ids,
    layer_stats=vis_layer_stats,
    spotlight_vals=spotlight_vals,
)
st.caption(
    f"showing **{len(vis_entries)}** tokens · "
    f"embedding: **t-SNE**. "
    f"Trail: **{_resolve_trail_strategy(show_neutral, show_tiled, 'auto')}**."
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config=dict(
        displayModeBar=True,
        modeBarButtonsToRemove=["lasso2d", "select2d"],
        toImageButtonOptions=dict(format="png", filename="latent_probe", scale=2),
    ),
)

# ── analysis panels ───────────────────────────────────────────────────────────
if st.button("Show analysis panels", key="show_analysis"):
    st.session_state["analysis_open"] = True

if not st.session_state.get("analysis_open", False):
    st.stop()

with st.spinner("Computing full layer statistics …"):
    layer_stats = compute_layer_stats({k: neutral_data[k] for k in active_layers}, active_layers)

if any(lk in active_layers for lk in ("cereb", "neurons", "refine2")):
    with st.expander("🧠 cerebellum · temporal attention  (long + short windows)", expanded=True):
        st.caption(
            "**Blue bars** = long-range windows.  "
            "**Orange bars** = short-range windows scaled by the learned short_window_gate."
        )
        st.plotly_chart(
            window_attention_fig(window_state, window_max),
            use_container_width=True,
            config=dict(displayModeBar=False),
        )

with st.expander("📐 intrinsic dimensionality · collapse indicator · gate values", expanded=True):
    st.caption(
        "**Blue bar** = effective dimensions (participation ratio, relative).  \n"
        "**Orange bar** = fraction of variance explained by a single axis.  \n"
        "**Purple ◆** = learned gate value."
    )
    st.plotly_chart(
        intrinsic_dim_chart(layer_stats, active_layers, gate_vals),
        use_container_width=True,
        config=dict(displayModeBar=False),
    )

if "neurons" in active_layers:
    with st.expander("🔥 neuron activation heatmap  (INN · top-N most-variable neurons)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            n_top = st.slider("neurons shown (rows)", 16, 256, 64, 16, key="neuron_heatmap_n_top")
        with c2:
            max_tok = st.slider("tokens shown (columns)", 100, 4200, 500, 100, key="neuron_heatmap_max_tok")

        st.plotly_chart(
            neuron_health_fig(
                neutral_data["neurons"],
                spotlight_idx=(spotlight_neuron_idx if overlay_mode == "single neuron" else None),
                spotlight_range=(spotlight_range if overlay_mode == "neuron range  (mean |act|)" else None),
            ),
            use_container_width=True,
            config=dict(displayModeBar=False),
        )

        st.plotly_chart(
            neuron_heatmap_fig(neutral_data["neurons"], entries, n_top=n_top, max_tokens=max_tok),
            use_container_width=True,
            config=dict(displayModeBar=True, modeBarButtonsToRemove=["lasso2d", "select2d"]),
        )

        st.markdown("---")
        st.markdown("**Neuron atlas**")
        if st.button("Compute neuron atlas", key="compute_atlas"):
            st.session_state.pop("natlas_cache", None)
            st.session_state["natlas_requested"] = True

        if st.session_state.get("natlas_requested"):
            atlas_xy = compute_neuron_atlas(neutral_data["neurons"])
            spot_idx = spotlight_neuron_idx if overlay_mode == "single neuron" else None
            st.plotly_chart(
                neuron_atlas_fig(atlas_xy, neutral_data["neurons"], entries, spotlight_idx=spot_idx),
                use_container_width=True,
                config=dict(displayModeBar=True, modeBarButtonsToRemove=["lasso2d", "select2d"]),
            )
        else:
            st.info("Press the button above to compute the neuron atlas. Result is cached for the session.")

labeled_indices = [i for i, e in enumerate(entries) if e["labeled"]]
labeled_labels = [e["label"] for e in entries if e["labeled"]]

if labeled_indices and "refine2" in active_layers:
    st.subheader("cosine similarity · refine2 layer · labeled words")
    c1, c2 = st.columns(2)
    for col, (data, label) in zip(
        [c1, c2],
        [(neutral_data, "neutral prefix"), (tiled_data, "tiled / looped")],
    ):
        with col:
            vecs = data["refine2"][labeled_indices]
            st.plotly_chart(
                cosine_heatmap_fig(vecs, labeled_labels, label),
                use_container_width=True,
                config=dict(displayModeBar=False),
            )
