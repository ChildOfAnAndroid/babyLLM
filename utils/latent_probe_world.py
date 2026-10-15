#!/usr/bin/env python3
# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm brain world explorer -*- ⊂ʕʘʔ⊂ʔ ---
# utils/latent_probe_world.py
#
# Run from project root:
#   python3 -m streamlit run utils/latent_probe_world.py
#
# ANIMATED TRAVERSAL — watch ALL tokens roll through every layer of baby's brain.
#
# Z axis    = pipeline depth  (embed at bottom → logits at top)
# X, Y      = shared 2D latent projection (comparable across layers)
# Animation = tokens physically move between layers via interpolated subframes
#
# babyLLM forward pass (canonical order from _ALL_LAYER_DEFS):
#   EMBED:      embed → pos → char → blend
#   ATTENTION:  attn1 → tangle_e
#   INN:        neurons → cereb → refine2
#   POST-INN:   inn_raw → attn2 → tangle_n → pre_scratch → scratch
#   MEMORY:     mem1 → tangle_m → mem2
#   LOGITS:     logits

import os
import sys
import json
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, ".probe_cache")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.probe_view_helpers import (available_cache_files, load_probe_cache, token_label, safe_category, load_three_vendor_sources)
from utils.latent_probe import _ALL_LAYER_DEFS

# ── layer metadata ────────────────────────────────────────────────────────────
_LAYER_ORDER = [ld[1] for ld in _ALL_LAYER_DEFS]
_LAYER_DESC = {ld[1]: ld[2].replace("\n", " ").strip() for ld in _ALL_LAYER_DEFS}

_STAGE_OF = {
    "embed": 0, "pos": 0, "char": 0, "blend": 0,
    "attn1": 1, "tangle_e": 1,
    "neurons": 2, "cereb": 2, "refine2": 2,
    "inn_raw": 3, "attn2": 3, "tangle_n": 3, "pre_scratch": 3, "scratch": 3,
    "mem1": 4, "tangle_m": 4, "mem2": 4,
    "logits": 5,
}
_STAGE_LABEL = {
    0: "EMBED", 1: "ATTENTION", 2: "INN",
    3: "POST-INN", 4: "MEMORY", 5: "LOGITS",
}

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="babyLLM brain world", layout="wide")
st.markdown(
    """<style>
body, .stApp { background-color: #090909; color: #dddddd; }
section[data-testid="stSidebar"] { background-color: #131313; }
h1, h2, h3, label, .stCheckbox span, .stRadio label { color: #eeeeee !important; }
</style>""",
    unsafe_allow_html=True,
)

st.title("ʕっʘ‿ʘʔっ babyLLM · brain world")
st.caption(
    "Press **Play** to watch every token traverse the full forward pass.  "
    "Z = pipeline depth · X,Y = shared latent projection."
)


# ── helpers ───────────────────────────────────────────────────────────────────
def _load_npz(cache_path: str):
    return load_probe_cache(cache_path, include_extras=False)


def _available_cache_files() -> List[str]:
    return available_cache_files(CACHE_DIR, include_sequences=False)


def _token_label(e: Dict[str, Any]) -> str:
    return token_label(e)


def _safe_cat(e: Dict[str, Any]) -> str:
    return safe_category(e)


def _cat_color(cat: str, labeled: bool = False) -> str:
    if labeled:
        return "#ffeb3b"
    return {
        "letter": "#4fc3f7", "digit": "#81c784", "punct": "#ff8a65",
        "bpe_word": "#ce93d8", "bpe_sub": "#f48fb1", "special": "#90a4ae",
    }.get(cat, "#aaaaaa")


def _ease_in_out(t: float) -> float:
    # smoothstep
    return t * t * (3.0 - 2.0 * t)


def _lerp(a, b, t: float):
    return (1.0 - t) * a + t * b


def _padded_range(values, pad_frac: float = 0.08, min_pad: float = 0.25) -> List[float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    span = hi - lo
    pad = max(min_pad, span * pad_frac)
    return [lo - pad, hi + pad]


def _render_plotly_world(fig: go.Figure, height: int = 920) -> None:
    post_script = """
    (function() {
        const gd = document.getElementById('{plot_id}');
        if (!gd || typeof Plotly === 'undefined') return;

        const storageKey = 'latent-probe-world-camera';
        let applyingCamera = false;

        function readCamera() {
            try {
                const raw = window.localStorage.getItem(storageKey);
                return raw ? JSON.parse(raw) : null;
            } catch (err) {
                return null;
            }
        }

        function writeCamera(camera) {
            if (!camera) return;
            try {
                window.localStorage.setItem(storageKey, JSON.stringify(camera));
            } catch (err) {
                // Ignore storage failures; the plot still works without persistence.
            }
        }

        function getCurrentCamera() {
            return gd.layout && gd.layout.scene ? gd.layout.scene.camera : null;
        }

        function restoreCamera() {
            if (applyingCamera) return;
            const camera = readCamera();
            if (!camera) return;
            applyingCamera = true;
            Plotly.relayout(gd, {'scene.camera': camera}).finally(function() {
                applyingCamera = false;
            });
        }

        gd.on('plotly_relayout', function(evt) {
            if (applyingCamera) return;
            const camera = (evt && evt['scene.camera']) || getCurrentCamera();
            if (camera) writeCamera(camera);
        });

        gd.on('plotly_afterplot', restoreCamera);
        gd.on('plotly_animated', restoreCamera);
        gd.on('plotly_animationinterrupted', restoreCamera);

        if (readCamera()) {
            restoreCamera();
        }
    })();
    """

    html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True},
        div_id="latent-probe-world-plot",
        post_script=post_script,
    )
    components.html(html, height=height, scrolling=False)


# ── layer depth mapping ──────────────────────────────────────────────────────
def _layer_z_depths(layer_keys: Tuple[str, ...]) -> Dict[str, float]:
    """Z positions with gaps between pipeline stages."""
    z: Dict[str, float] = {}
    pos = 0.0
    prev_stage = -1
    for i, lk in enumerate(layer_keys):
        stage = _STAGE_OF.get(lk, -1)
        if i > 0:
            pos += 2.0 if stage != prev_stage else 1.0
        z[lk] = pos
        prev_stage = stage
    if len(z) > 1:
        lo, hi = min(z.values()), max(z.values())
        rng = hi - lo
        if rng > 1e-6:
            z = {k: (v - lo) / rng * 10.0 for k, v in z.items()}
    return z


# ── shared 2D projection ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing traversal positions …")
def compute_traversal_xy(
    data: Dict[str, np.ndarray], layer_keys: Tuple[str, ...]
) -> Dict[str, np.ndarray]:
    """
    Shared 2D (X, Y) for every token at every layer.
    Each layer compressed to 24 dims, normalised, stacked, final 2D PCA.
    """
    blocks: List[np.ndarray] = []
    n_tokens: Optional[int] = None

    for lk in layer_keys:
        arr = np.asarray(data[lk], dtype=np.float32)
        if n_tokens is None:
            n_tokens = arr.shape[0]
        d = min(24, arr.shape[0] - 1, arr.shape[1])
        if d < 2:
            block = np.zeros((arr.shape[0], 24), dtype=np.float32)
        else:
            block = PCA(n_components=d, random_state=42).fit_transform(arr).astype(np.float32)
            if block.shape[1] < 24:
                block = np.pad(block, ((0, 0), (0, 24 - block.shape[1])))
        # normalise per-layer so high-dim layers don't dominate the final PCA
        bscale = float(np.max(np.linalg.norm(block, axis=1)))
        if bscale > 1e-8:
            block /= bscale
        blocks.append(block)

    stacked = np.vstack(blocks)
    xy = PCA(n_components=2, random_state=42).fit_transform(stacked).astype(np.float32)
    xy -= xy.mean(axis=0, keepdims=True)
    scale = float(np.max(np.abs(xy)))
    if scale > 1e-8:
        xy /= scale
    xy *= 5.0  # ±5 range, proportional with Z=[0,10]

    assert n_tokens is not None
    out: Dict[str, np.ndarray] = {}
    for i, lk in enumerate(layer_keys):
        out[lk] = xy[i * n_tokens : (i + 1) * n_tokens]
    return out


# ── sidebar ───────────────────────────────────────────────────────────────────
cache_files = _available_cache_files()
if not cache_files:
    st.error("No probe cache in .probe_cache/. Run latent_probe_ui.py first.")
    st.stop()

selected_cache = st.sidebar.selectbox(
    "Probe cache",
    options=cache_files,
    format_func=lambda p: os.path.basename(p),
)

entries, neutral_data, tiled_data = _load_npz(selected_cache)

_known = set(neutral_data.keys())
layer_keys = tuple(
    [k for k in _LAYER_ORDER if k in _known]
    + sorted(_known - set(_LAYER_ORDER))
)

if len(layer_keys) < 2:
    st.error(f"Only {len(layer_keys)} layer(s) in cache — need at least 2.")
    st.stop()

n_tokens = len(entries)
n_layers = len(layer_keys)

st.sidebar.markdown("---")
st.sidebar.header("Focus token")
_labels = [_token_label(e) for e in entries]
selected_label = st.sidebar.selectbox("Token", options=_labels, index=0)
selected_idx = _labels.index(selected_label)

st.sidebar.header("Display")
step_stride = st.sidebar.slider(
    "Step-line thinning", 1, 30, 4, 1,
    help="How many tokens show blue step-lines between layers.",
)
show_bg_paths = st.sidebar.checkbox("Show full paths (static faint lines)", value=False)
if show_bg_paths:
    path_stride = st.sidebar.slider("Path thinning", 1, 50, 6, 1)

st.sidebar.header("Animation")
frame_ms = st.sidebar.slider("Speed (ms per layer)", 400, 3000, 1200, 100)
subframes = st.sidebar.slider(
    "Interpolated subframes",
    1, 12, 5, 1,
    help="Extra in-between motion frames between real layers.",
)
black_bg = st.sidebar.checkbox("Black void", value=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "▶ Play to watch all tokens traverse the forward pass.\n"
    "Scrub the layer slider to jump.\n\n"
    "● colored dots = all tokens at current layer\n"
    "◆ white diamond = selected (focus) token\n"
    "purple line = selected token's growing trail\n"
    "blue lines = step direction (prev → current layer)"
)


# ── compute ───────────────────────────────────────────────────────────────────
z_depths = _layer_z_depths(layer_keys)
traversal_xy = compute_traversal_xy(neutral_data, layer_keys)
all_xy = np.vstack([traversal_xy[lk] for lk in layer_keys]).astype(np.float32)
scene_x_range = _padded_range(
    np.concatenate([all_xy[:, 0], np.asarray([6.5], dtype=np.float32)]),
    pad_frac=0.06,
    min_pad=0.4,
)
scene_y_range = _padded_range(all_xy[:, 1], pad_frac=0.06, min_pad=0.4)
scene_z_range = _padded_range(list(z_depths.values()), pad_frac=0.05, min_pad=0.5)

# precompute per-token colors & hover
_tok_colors = [_cat_color(_safe_cat(e), bool(e.get("labeled"))) for e in entries]
_tok_hover_base = [
    f"<b>{_token_label(e)}</b><br>id: {e['id']} · {_safe_cat(e)}"
    for e in entries
]

# selected token data across all layers
sel_all_x = [float(traversal_xy[lk][selected_idx, 0]) for lk in layer_keys]
sel_all_y = [float(traversal_xy[lk][selected_idx, 1]) for lk in layer_keys]
sel_all_z = [z_depths[lk] for lk in layer_keys]

# step-line token indices
step_idx = list(range(0, n_tokens, step_stride))

# per-layer L2 norms for selected token
sel_norms: Dict[str, float] = {}
for lk in layer_keys:
    sel_norms[lk] = float(
        np.linalg.norm(np.asarray(neutral_data[lk][selected_idx], dtype=np.float32))
    )

# ── build figure ──────────────────────────────────────────────────────────────
fig = go.Figure()
bg = "#000000" if black_bg else "#0d0d0d"

# ══════════════════════════════════════════════════════════════════════════════
# STATIC TRACES  (don't change during animation)
# ══════════════════════════════════════════════════════════════════════════════

# T0: selected token full route (faint dotted — shows where it WILL go)
fig.add_trace(go.Scatter3d(
    x=sel_all_x, y=sel_all_y, z=sel_all_z,
    mode="lines",
    line=dict(color="rgba(180,100,255,0.22)", width=2, dash="dot"),
    hoverinfo="skip",
    name=f"full route · {selected_label}",
))

# T1: layer depth labels
label_texts = []
prev_sid = -1
for lk in layer_keys:
    sid = _STAGE_OF.get(lk, -1)
    prefix = f"{_STAGE_LABEL.get(sid, '?')}: " if sid != prev_sid else "  "
    prev_sid = sid
    dim = neutral_data[lk].shape[1]
    label_texts.append(f"{prefix}{lk}  ({dim}d)")

fig.add_trace(go.Scatter3d(
    x=[6.5] * n_layers,
    y=[0.0] * n_layers,
    z=[z_depths[lk] for lk in layer_keys],
    mode="text",
    text=label_texts,
    textposition="middle right",
    textfont=dict(size=9, color="#555555"),
    hoverinfo="skip",
    showlegend=False,
))

# T2: background full paths (optional static faint lines)
if show_bg_paths:
    px: List[Optional[float]] = []
    py: List[Optional[float]] = []
    pz: List[Optional[float]] = []
    for ti in range(0, n_tokens, path_stride):
        for lk in layer_keys:
            xy = traversal_xy[lk][ti]
            px.append(float(xy[0]))
            py.append(float(xy[1]))
            pz.append(z_depths[lk])
        px.append(None)
        py.append(None)
        pz.append(None)
    fig.add_trace(go.Scatter3d(
        x=px, y=py, z=pz,
        mode="lines",
        line=dict(color="rgba(100,160,255,0.04)", width=1),
        hoverinfo="skip",
        name="all paths",
    ))
else:
    # placeholder — keeps trace indices stable
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="lines", line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip", showlegend=False,
    ))

# ══════════════════════════════════════════════════════════════════════════════
# ANIMATED TRACES  (updated per frame)
# trace indices 3, 4, 5, 6, 7
# ══════════════════════════════════════════════════════════════════════════════

lk0 = layer_keys[0]
z0 = z_depths[lk0]
xy0 = traversal_xy[lk0]

# T3: step lines (prev→current, one segment per strided token)
#     at frame 0: zero-length (same start and end)
step_x0: List[Optional[float]] = []
step_y0: List[Optional[float]] = []
step_z0: List[Optional[float]] = []
for ti in step_idx:
    xv = float(xy0[ti, 0])
    yv = float(xy0[ti, 1])
    step_x0 += [xv, xv, None]
    step_y0 += [yv, yv, None]
    step_z0 += [z0, z0, None]

fig.add_trace(go.Scatter3d(
    x=step_x0, y=step_y0, z=step_z0,
    mode="lines",
    line=dict(color="rgba(100,180,255,0.22)", width=1.8),
    hoverinfo="skip",
    name="step lines",
))

# T4: ghost cloud (previous layer, faded)
fig.add_trace(go.Scatter3d(
    x=xy0[:, 0].tolist(),
    y=xy0[:, 1].tolist(),
    z=[z0] * n_tokens,
    mode="markers",
    marker=dict(size=2.5, color=_tok_colors, opacity=0.18),
    hoverinfo="skip",
    name="prev layer",
    showlegend=False,
))

# T5: current token cloud (bright, this is the main visual)
fig.add_trace(go.Scatter3d(
    x=xy0[:, 0].tolist(),
    y=xy0[:, 1].tolist(),
    z=[z0] * n_tokens,
    mode="markers",
    marker=dict(size=3.5, color=_tok_colors, opacity=0.85),
    hovertext=[f"{h}<br>{lk0}" for h in _tok_hover_base],
    hoverinfo="text",
    name="tokens",
))

# T6: selected token growing trail (stacked at start → extends each frame)
fig.add_trace(go.Scatter3d(
    x=[sel_all_x[0]] * (n_layers + 1),
    y=[sel_all_y[0]] * (n_layers + 1),
    z=[sel_all_z[0]] * (n_layers + 1),
    mode="lines+markers",
    line=dict(color="#cc66ff", width=5),
    marker=dict(size=3, color="#cc66ff"),
    hoverinfo="skip",
    name=f"trail · {selected_label}",
))

# T7: selected token spotlight (white diamond)
fig.add_trace(go.Scatter3d(
    x=[float(xy0[selected_idx, 0])],
    y=[float(xy0[selected_idx, 1])],
    z=[z0],
    mode="markers+text",
    marker=dict(size=10, color="#ffffff", symbol="diamond"),
    text=[selected_label],
    textposition="top center",
    textfont=dict(color="#ffffff", size=11),
    hoverinfo="skip",
    name="selected",
    showlegend=False,
))

# ══════════════════════════════════════════════════════════════════════════════
# INTERPOLATED MOTION STATES
# ══════════════════════════════════════════════════════════════════════════════
motion_states = []

for li in range(n_layers - 1):
    src_lk = layer_keys[li]
    dst_lk = layer_keys[li + 1]

    src_xy = traversal_xy[src_lk]
    dst_xy = traversal_xy[dst_lk]
    src_z = z_depths[src_lk]
    dst_z = z_depths[dst_lk]

    # real frame at source layer (only once, for first layer)
    if li == 0:
        motion_states.append({
            "name": src_lk,
            "label": src_lk,
            "display_layer": src_lk,
            "xy": src_xy,
            "z": np.full((n_tokens,), src_z, dtype=np.float32),
            "ghost_xy": src_xy,
            "ghost_z": np.full((n_tokens,), src_z, dtype=np.float32),
            "step_src_xy": src_xy[step_idx],
            "step_src_z": np.full((len(step_idx),), src_z, dtype=np.float32),
            "step_dst_xy": src_xy[step_idx],
            "step_dst_z": np.full((len(step_idx),), src_z, dtype=np.float32),
            "trail_real_upto": 0,
            "trail_interp_xy": src_xy[selected_idx],
            "trail_interp_z": src_z,
        })

    # subframes between src → dst, including exact destination frame
    for s in range(1, subframes + 2):
        t_raw = s / (subframes + 1)
        t = _ease_in_out(t_raw)

        interp_xy = _lerp(src_xy, dst_xy, t).astype(np.float32)
        interp_z = _lerp(
            np.full((n_tokens,), src_z, dtype=np.float32),
            np.full((n_tokens,), dst_z, dtype=np.float32),
            t,
        ).astype(np.float32)

        is_real_dest = (s == subframes + 1)

        motion_states.append({
            "name": f"{src_lk}→{dst_lk}#{s}",
            "label": dst_lk if is_real_dest else "",
            "display_layer": dst_lk if is_real_dest else f"{src_lk}→{dst_lk}",
            "xy": interp_xy,
            "z": interp_z,
            "ghost_xy": src_xy,
            "ghost_z": np.full((n_tokens,), src_z, dtype=np.float32),
            "step_src_xy": src_xy[step_idx],
            "step_src_z": np.full((len(step_idx),), src_z, dtype=np.float32),
            "step_dst_xy": interp_xy[step_idx],
            "step_dst_z": interp_z[step_idx],
            "trail_real_upto": li,
            "trail_interp_xy": interp_xy[selected_idx],
            "trail_interp_z": float(interp_z[selected_idx]),
        })

# ══════════════════════════════════════════════════════════════════════════════
# ANIMATION FRAMES
# ══════════════════════════════════════════════════════════════════════════════
frames = []
for state in motion_states:
    # T3: step lines
    sx: List[Optional[float]] = []
    sy: List[Optional[float]] = []
    sz: List[Optional[float]] = []
    for i in range(len(step_idx)):
        sx += [float(state["step_src_xy"][i, 0]), float(state["step_dst_xy"][i, 0]), None]
        sy += [float(state["step_src_xy"][i, 1]), float(state["step_dst_xy"][i, 1]), None]
        sz += [float(state["step_src_z"][i]), float(state["step_dst_z"][i]), None]

    # T4: ghost cloud
    gx = state["ghost_xy"][:, 0].tolist()
    gy = state["ghost_xy"][:, 1].tolist()
    gz = state["ghost_z"].tolist()

    # T5: current cloud
    cx = state["xy"][:, 0].tolist()
    cy = state["xy"][:, 1].tolist()
    cz = state["z"].tolist()

    # T6: selected trail
    tx = []
    ty = []
    tz = []

    # fully completed real layers so far
    for j in range(state["trail_real_upto"] + 1):
        lk = layer_keys[j]
        tx.append(float(traversal_xy[lk][selected_idx, 0]))
        ty.append(float(traversal_xy[lk][selected_idx, 1]))
        tz.append(float(z_depths[lk]))

    # current interpolated point
    tx.append(float(state["trail_interp_xy"][0]))
    ty.append(float(state["trail_interp_xy"][1]))
    tz.append(float(state["trail_interp_z"]))

    # keep point count stable
    while len(tx) < n_layers + 1:
        tx.append(float(state["trail_interp_xy"][0]))
        ty.append(float(state["trail_interp_xy"][1]))
        tz.append(float(state["trail_interp_z"]))

    # T7: spotlight
    spx = [float(state["xy"][selected_idx, 0])]
    spy = [float(state["xy"][selected_idx, 1])]
    spz = [float(state["z"][selected_idx])]

    frames.append(go.Frame(
        data=[
            go.Scatter3d(x=sx, y=sy, z=sz),  # T3
            go.Scatter3d(x=gx, y=gy, z=gz),  # T4
            go.Scatter3d(
                x=cx, y=cy, z=cz,
                hovertext=[f"{h}<br>{state['display_layer']}" for h in _tok_hover_base],
            ),                               # T5
            go.Scatter3d(x=tx, y=ty, z=tz),  # T6
            go.Scatter3d(x=spx, y=spy, z=spz),  # T7
        ],
        traces=[3, 4, 5, 6, 7],
        name=state["name"],
    ))

fig.frames = frames

# ── animation controls ────────────────────────────────────────────────────────
slider_steps = [
    dict(
        args=[[state["name"]], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
        label=state["label"],
        method="animate",
    )
    for state in motion_states
    if state["label"]
]

interp_frame_ms = max(35, frame_ms // (subframes + 1))
interp_transition_ms = max(20, int(interp_frame_ms * 0.75))
camera_uirevision = "latent-probe-world-camera"

fig.update_layout(
    uirevision=camera_uirevision,
    paper_bgcolor=bg,
    plot_bgcolor=bg,
    font=dict(color="#dddddd"),
    margin=dict(l=0, r=0, t=10, b=120),
    height=880,
    legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(size=10)),
    scene=dict(
        # Keep the user-moved 3D camera across Streamlit reruns.
        uirevision=camera_uirevision,
        bgcolor=bg,
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1.3),
        camera=dict(
            eye=dict(x=1.6, y=0.8, z=0.5),
            up=dict(x=0, y=0, z=1),
        ),
        xaxis=dict(
            range=scene_x_range,
            showbackground=False, showgrid=False, zeroline=False,
            showticklabels=False, title="",
        ),
        yaxis=dict(
            range=scene_y_range,
            showbackground=False, showgrid=False, zeroline=False,
            showticklabels=False, title="",
        ),
        zaxis=dict(
            range=scene_z_range,
            showbackground=False, showgrid=True, zeroline=False,
            showticklabels=False, title="pipeline depth",
            gridcolor="rgba(255,255,255,0.04)",
        ),
    ),
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        x=0.05,
        y=-0.02,
        xanchor="right",
        yanchor="top",
        buttons=[
            dict(
                label="▶ Play",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=interp_frame_ms, redraw=True),
                    transition=dict(duration=interp_transition_ms, easing="linear"),
                    fromcurrent=True,
                    mode="immediate",
                )],
            ),
            dict(
                label="⏸ Pause",
                method="animate",
                args=[[None], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
            ),
        ],
        bgcolor="#1a1a1a",
        bordercolor="#444",
        font=dict(color="#ddd", size=12),
    )],
    sliders=[dict(
        active=0,
        steps=slider_steps,
        x=0.08,
        xanchor="left",
        y=-0.03,
        yanchor="top",
        len=0.88,
        currentvalue=dict(
            prefix="layer: ",
            font=dict(color="#cccccc", size=13),
            xanchor="center",
        ),
        font=dict(color="#777", size=8),
        bgcolor="#1a1a1a",
        bordercolor="#444",
        tickcolor="#555",
    )],
)

_render_plotly_world(fig)

# ── readout ───────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.markdown("### world")
    st.write(f"cache: `{os.path.basename(selected_cache)}`")
    st.write(f"tokens: **{n_tokens}** · layers: **{len(layer_keys)}**")
    stages_present = sorted({_STAGE_OF.get(lk, -1) for lk in layer_keys})
    st.write(f"pipeline: {' → '.join(_STAGE_LABEL.get(s, '?') for s in stages_present)}")

with c2:
    st.markdown(f"### {selected_label}")
    e = entries[selected_idx]
    st.write(f"id: **{e['id']}** · cat: **{_safe_cat(e)}**")
    if e.get("labeled"):
        st.write("★ labeled word")

# ── traversal table ──────────────────────────────────────────────────────────
with st.expander(f"**{selected_label}** · full traversal ({len(layer_keys)} layers)"):
    rows = []
    for lk in layer_keys:
        xy = traversal_xy[lk][selected_idx]
        sid = _STAGE_OF.get(lk, -1)
        rows.append({
            "stage": _STAGE_LABEL.get(sid, "?"),
            "layer": lk,
            "description": _LAYER_DESC.get(lk, ""),
            "dim": int(neutral_data[lk].shape[1]),
            "L2 norm": round(sel_norms[lk], 4),
            "x": round(float(xy[0]), 3),
            "y": round(float(xy[1]), 3),
            "z": round(z_depths[lk], 3),
        })
    st.dataframe(rows, use_container_width=True)

# ── neuron detail ────────────────────────────────────────────────────────────
if "neurons" in neutral_data:
    with st.expander(f"**{selected_label}** · top neurons (INN layer)"):
        vec = np.asarray(neutral_data["neurons"][selected_idx], dtype=np.float32)
        top_k = 32
        idx = np.argsort(np.abs(vec))[::-1][:top_k]
        rows = [
            {
                "neuron": int(i),
                "activation": round(float(vec[i]), 5),
                "|act|": round(float(np.abs(vec[i])), 5),
            }
            for i in idx
        ]
        st.dataframe(rows, use_container_width=True)

# ── about ─────────────────────────────────────────────────────────────────────
with st.expander("what this view shows"):
    st.markdown(
        """
**babyLLM forward pass traversal** — animated 3D view of all layers.

| Stage | Layers | Dim |
|-------|--------|-----|
| EMBED | embed, pos, char, blend | 1024 |
| ATTENTION | attn1, tangle_e | 1024 |
| INN | neurons, cereb, refine2 | 10,000 |
| POST-INN | inn_raw, attn2, tangle_n, pre_scratch, scratch | 10,000 |
| MEMORY | mem1, tangle_m, mem2 | 10,000 |
| LOGITS | logits | vocab |

**How it works:**
- **Z axis** = pipeline depth (bottom = embed, top = logits)
- **X, Y** = shared 2D projection so positions are comparable across layers
- **Colored dots** = ALL tokens at the current animated layer (moving together)
- **Faded dots** = ghost of where tokens were at the previous real layer
- **Blue lines** = step direction showing how each token moved
- **Purple trail** = selected token's path built up layer by layer
- **White diamond** = selected token at current point in the traversal
- **Dotted line** = selected token's full route (preview of the entire path)

Press **Play** to watch the forward pass unfold. The entire token cloud
morphs at each layer — and now drifts through real interpolated subframes
between layers instead of snapping from one layer to the next.
"""
    )
