#!/usr/bin/env python3
# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm theatre (three, box architecture) -*- ⊂ʕʘʔ⊂ʔ ---
# utils/latent_probe_theatre.py
#
# Run from project root:
#   python3 -m streamlit run utils/latent_probe_theatre.py
#
# Forward-pass theatre:
# - continuous token river instead of chunked frame jumps
# - fixed camera unless you move it
# - real 3D depth, occlusion, orbit controls
# - layers as BOX / CHAMBER geometry sized from the plotted token cloud
# - tokens remain round / orb-like
# - selected token gets a stronger ribbon and spotlight
# - INN / POST-INN chambers include internal scaffold structure
#
# Install:
#   pip install streamlit numpy scipy scikit-learn
#
# Notes:
# - Browser-side Three.js is vendored locally and injected into the iframe at runtime.
# - No npm build step is required for this prototype.
#
# Vendored files expected:
#   utils/_vendor/three/three.module.min.js
#   utils/_vendor/three/OrbitControls.js

import math
import os
import sys
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, ".probe_cache")
THREE_VENDOR_DIR = os.path.join(PROJECT_ROOT, "utils", "_vendor", "three")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.probe_view_helpers import (available_cache_files, load_probe_cache, token_label, safe_category, load_three_vendor_sources)
from utils.latent_probe import _ALL_LAYER_DEFS

# ── metadata ──────────────────────────────────────────────────────────────────
_LAYER_ORDER = [ld[1] for ld in _ALL_LAYER_DEFS]
_LAYER_DESC = {ld[1]: ld[2].replace("", " ").strip() for ld in _ALL_LAYER_DEFS}

_STAGE_OF = {
    "embed": 0, "pos": 0, "char": 0, "blend": 0,
    "attn1": 1, "tangle_e_down": 1, "tangle_e_up": 1, "tangle_e": 1,
    "neurons_norm": 2, "neurons": 2, "cereb": 2,
    "refine2_down": 2, "refine2_up": 2, "refine2": 2,
    "inn_raw": 3, "attn2": 3,
    "tangle_n_down": 3, "tangle_n_up": 3, "tangle_n": 3,
    "pre_scratch": 3, "scratch_down": 3, "scratch_up": 3, "scratch": 3,
    "mem1_reduce": 4, "mem1_inf_down": 4, "mem1_inf_up": 4, "mem1": 4,
    "tangle_m_down": 4, "tangle_m_up": 4, "tangle_m": 4,
    "mem2_reduce": 4, "mem2_inf_down": 4, "mem2_inf_up": 4, "mem2": 4,
    "logits": 5,
}
_STAGE_LABEL = {
    0: "EMBED",
    1: "ATTENTION",
    2: "INN",
    3: "POST-INN",
    4: "MEMORY",
    5: "LOGITS",
}
_STAGE_COLOR = {
    0: "#4fc3f7",
    1: "#81c784",
    2: "#ff8a65",
    3: "#ce93d8",
    4: "#f48fb1",
    5: "#ffd54f",
}
TOK_CAT_COLOR = {
    "letter": "#4fc3f7",
    "digit": "#81c784",
    "punct": "#ff8a65",
    "bpe_word": "#ce93d8",
    "bpe_sub": "#f48fb1",
    "special": "#90a4ae",
    "unknown": "#aaaaaa",
}

_WORLD_RADIUS = 8.5  # kept for _normalize_xy fallback only

# ── architecture-driven box dimensions ────────────────────────────────────────
# W = f(input_dim), H = f(output_dim) — so the box face shows the transformation.
# Dimensions come from the actual model config (vocabSize=4200, embedDim=1024,
# numNeurons=10000, scratchpad slots=256, bottleneck=512, positions=2048).
_LAYER_ARCH_DIMS: Dict[str, Dict[str, int]] = {
    # ── Embedding stage ───────────────────────────────────────────────────────
    "embed":         {"in": 4200,  "out": 1024},   # vocab lookup → embed
    "pos":           {"in": 2048,  "out": 1024},   # positional lookup → embed
    "char":          {"in": 256,   "out": 1024},   # 256 byte values → char embed
    "blend":         {"in": 1024,  "out": 1024},   # 4-source softmax blend
    # ── Attention₁ + tangling (embed stage) ──────────────────────────────────
    "attn1":         {"in": 1024,  "out": 1024},   # 16-head causal attention
    "tangle_e_down": {"in": 1024,  "out": 256},    # embed context projection ↓
    "tangle_e_up":   {"in": 256,   "out": 1024},   # embed context projection ↑
    "tangle_e":      {"in": 1024,  "out": 1024},   # embed-stage context mixer
    # ── INN stage ─────────────────────────────────────────────────────────────
    "neurons_norm":  {"in": 1024,  "out": 1024},   # inputNorm before expansion
    "neurons":       {"in": 1024,  "out": 10000},  # THE EXPANSION: 1024 → 10 000
    "cereb":         {"in": 10000, "out": 10000},  # 18-window cerebellum aggregation
    "refine2_down":  {"in": 10000, "out": 512},    # bottleneck compress 10k → 512
    "refine2_up":    {"in": 512,   "out": 10000},  # bottleneck expand   512 → 10k
    "refine2":       {"in": 512,   "out": 10000},  # post-bottleneck output (10k)
    # ── Post-INN + tangling (neuron stage) ────────────────────────────────────
    "inn_raw":       {"in": 10000, "out": 10000},  # post-INN before attn₂
    "attn2":         {"in": 10000, "out": 10000},  # 16-head causal attention (neuron space)
    "tangle_n_down": {"in": 10000, "out": 128},    # neuron context projection ↓
    "tangle_n_up":   {"in": 128,   "out": 10000},  # neuron context projection ↑
    "tangle_n":      {"in": 10000, "out": 10000},  # neuron-stage context mixer
    "pre_scratch":   {"in": 10000, "out": 10000},  # signal entering scratchpad
    "scratch_down":  {"in": 10000, "out": 512},    # scratchpad project_down 10k→512
    "scratch_up":    {"in": 512,   "out": 10000},  # scratchpad project_up   512→10k
    "scratch":       {"in": 256,   "out": 10000},  # 256 content-addressable slots → 10k
    # ── Memory stage + tangling (memory stage) ────────────────────────────────
    "mem1_reduce":   {"in": 10000, "out": 1024},   # memory₁ inputReducer 10k→1024
    "mem1_inf_down": {"in": 1024,  "out": 512},    # memory₁ influence compress 1024→512
    "mem1_inf_up":   {"in": 512,   "out": 10000},  # memory₁ influence expand   512→10k
    "mem1":          {"in": 10000, "out": 10000},  # memory₁ STM+LTM gated blend
    "tangle_m_down": {"in": 10000, "out": 128},    # memory context projection ↓
    "tangle_m_up":   {"in": 128,   "out": 10000},  # memory context projection ↑
    "tangle_m":      {"in": 10000, "out": 10000},  # memory-stage context mixer
    "mem2_reduce":   {"in": 10000, "out": 1024},   # memory₂ inputReducer 10k→1024
    "mem2_inf_down": {"in": 1024,  "out": 512},    # memory₂ influence compress 1024→512
    "mem2_inf_up":   {"in": 512,   "out": 10000},  # memory₂ influence expand   512→10k
    "mem2":          {"in": 10000, "out": 10000},  # memory₂ second STM+LTM instance
    # ── Logits ────────────────────────────────────────────────────────────────
    "logits":        {"in": 10000, "out": 4200},   # 10 000 → vocab logits
}
_ARCH_DIM_LO, _ARCH_DIM_HI = 256, 10000
_ARCH_WORLD_LO, _ARCH_WORLD_HI = 3.5, 18.0


def _arch_to_world(dim: int) -> float:
    """Map a model dimension to a world-unit size on a log scale."""
    t = (math.log(max(dim, _ARCH_DIM_LO)) - math.log(_ARCH_DIM_LO)) / \
        (math.log(_ARCH_DIM_HI) - math.log(_ARCH_DIM_LO))
    return _ARCH_WORLD_LO + (_ARCH_WORLD_HI - _ARCH_WORLD_LO) * max(0.0, min(1.0, t))


def _arch_box_metrics(layer_key: str, depth: float) -> Dict[str, float]:
    """Entry face = square sized by input_dim; exit face = square sized by output_dim."""
    arch = _LAYER_ARCH_DIMS.get(layer_key, {"in": 1024, "out": 1024})
    s_in  = _arch_to_world(arch["in"])
    s_out = _arch_to_world(arch["out"])
    return {"w_in": s_in, "h_in": s_in, "w_out": s_out, "h_out": s_out, "d": float(depth)}
_BALANCED_PCA_DIMS = 8
_MAX_VISIBLE_TOKENS = 4200
_MAX_AMBIENT_RIBBONS = 192

# ── page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="babyLLM theatre (three)", layout="wide")
st.markdown(
    """
<style>
body, .stApp { background-color: #090909; color: #dddddd; }
section[data-testid="stSidebar"] { background-color: #131313; }
h1, h2, h3, label, .stCheckbox span, .stRadio label { color: #eeeeee !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("ʕっʘ‿ʘʔっ babyLLM · theatre")
st.caption("Continuous token flow through a box-shaped machine. Tokens are round; architecture is geometric.")

# ── helpers ───────────────────────────────────────────────────────────────────
def _available_cache_files() -> List[str]:
    return available_cache_files(CACHE_DIR, include_sequences=True)


def _load_npz(cache_path: str):
    return load_probe_cache(cache_path, include_extras=True)


def _token_label(e: Dict[str, Any]) -> str:
    return token_label(e)


def _safe_cat(e: Dict[str, Any]) -> str:
    return safe_category(e)


def _tok_color(e: Dict[str, Any]) -> str:
    if e.get("generated"):
        return "#ff9800"   # orange — model-generated tokens
    if e.get("labeled"):
        return "#ffeb3b"   # yellow — labeled/focus tokens
    return TOK_CAT_COLOR.get(_safe_cat(e), "#aaaaaa")


def _layer_z_depths(layer_keys: Tuple[str, ...]) -> Dict[str, float]:
    z: Dict[str, float] = {}
    pos = 0.0
    prev_stage = -1
    for i, lk in enumerate(layer_keys):
        stage = _STAGE_OF.get(lk, -1)
        if i > 0:
            pos += 2.0 if stage != prev_stage else 1.0
        z[lk] = pos
        prev_stage = stage

    lo, hi = min(z.values()), max(z.values())
    rng = hi - lo
    if rng > 1e-8:
        z = {k: (v - lo) / rng * 52.0 for k, v in z.items()}
    return z


def _normalize_xy(xy: np.ndarray, radius: float = _WORLD_RADIUS) -> np.ndarray:
    out = np.asarray(xy, dtype=np.float32).copy()
    out -= out.mean(axis=0, keepdims=True)
    scale = float(np.max(np.linalg.norm(out, axis=1)))
    if scale > 1e-8:
        out /= scale
    out *= radius
    return out


def _center_xy(xy: np.ndarray) -> np.ndarray:
    out = np.asarray(xy, dtype=np.float32).copy()
    out -= out.mean(axis=0, keepdims=True)
    return out


def _projection_mix_matrix(max_dims: int = _BALANCED_PCA_DIMS) -> np.ndarray:
    rng = np.random.default_rng(42)
    base = rng.standard_normal((max_dims, max_dims)).astype(np.float32)
    q, _ = np.linalg.qr(base)
    return q[:, :2].astype(np.float32)


_PROJECTION_MIX = _projection_mix_matrix()


def _layer_balanced_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.shape[0] < 2 or a.shape[1] < 2:
        return np.zeros((a.shape[0], 2), dtype=np.float32)

    a = a - a.mean(axis=0, keepdims=True)
    total_var = float(np.var(a, axis=0).sum())
    if total_var < 1e-12:
        return np.zeros((a.shape[0], 2), dtype=np.float32)

    nc = min(_BALANCED_PCA_DIMS, a.shape[0] - 1, a.shape[1])
    if nc < 1:
        return np.zeros((a.shape[0], 2), dtype=np.float32)

    z = PCA(n_components=nc, whiten=False, random_state=42).fit_transform(a).astype(np.float32)
    z = np.nan_to_num(z, copy=False)
    xy = z @ _PROJECTION_MIX[:nc]
    return _center_xy(xy)  # centre only — no radius normalisation


def _align_to_prev(prev_xy: np.ndarray, curr_xy: np.ndarray) -> np.ndarray:
    a0 = _center_xy(prev_xy)
    b0 = _center_xy(curr_xy)
    try:
        r, _ = orthogonal_procrustes(b0, a0)
        aligned = b0 @ r
    except Exception:
        aligned = b0
    return _center_xy(aligned)  # preserve natural scale, just centre


def _xy_flatness_ratio(xy: np.ndarray) -> float:
    a = np.asarray(xy, dtype=np.float32)
    if a.shape[0] < 3:
        return 1.0
    cov = np.cov(a.T)
    eig = np.linalg.eigvalsh(cov)
    major = float(max(eig[-1], 1e-8))
    minor = float(max(eig[0], 0.0))
    return minor / major


@st.cache_data(show_spinner="Computing token world …")
def compute_layer_xy_world(neutral_data: Dict[str, np.ndarray], layer_keys: Tuple[str, ...]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    prev_xy = None
    for lk in layer_keys:
        # Each layer's cloud is normalised to fill its own frustum face —
        # architectural box size sets the scale, raw PCA variance sets the shape.
        arch = _LAYER_ARCH_DIMS.get(lk, {"in": 1024, "out": 1024})
        radius = min(_arch_to_world(arch["in"]), _arch_to_world(arch["out"])) * 0.44

        xy = _layer_balanced_2d(neutral_data[lk])   # centred, unscaled
        if prev_xy is not None:
            if float(np.std(xy)) < 1e-6:
                xy = prev_xy * 0.96
            else:
                xy = _align_to_prev(prev_xy, xy)
                flatness = _xy_flatness_ratio(xy)
                if flatness < 0.08:
                    carry = 0.42 if flatness < 0.02 else 0.24
                    xy = _center_xy(((1.0 - carry) * xy) + (carry * prev_xy))
        xy = _normalize_xy(xy, radius=radius)        # scale to this layer's box
        out[lk] = xy.astype(np.float32)
        prev_xy = xy
    return out


def _layer_slab_depths(layer_keys: Tuple[str, ...], z_depths: Dict[str, float]) -> Dict[str, float]:
    if not layer_keys:
        return {}
    if len(layer_keys) == 1:
        return {layer_keys[0]: 0.5}

    zvals = [float(z_depths[lk]) for lk in layer_keys]
    out: Dict[str, float] = {}
    for i, lk in enumerate(layer_keys):
        prev_gap = zvals[i] - zvals[i - 1] if i > 0 else None
        next_gap = zvals[i + 1] - zvals[i] if i + 1 < len(zvals) else None
        local_gap = next_gap if prev_gap is None else prev_gap if next_gap is None else min(prev_gap, next_gap)
        out[lk] = float(np.clip((local_gap or 3.2) * 0.62, 1.4, 3.2))
    return out



@st.cache_resource
def _load_three_vendor_sources() -> Tuple[str, str]:
    return load_three_vendor_sources(THREE_VENDOR_DIR)


def _render_world(world_payload: Dict[str, Any], height: int = 920) -> None:
    payload_json = json.dumps(world_payload).replace("</", "<\\/")
    three_source, orbit_source = _load_three_vendor_sources()
    three_source_json = json.dumps(three_source).replace("</", "<\\/")
    orbit_source_json = json.dumps(orbit_source).replace("</", "<\\/")

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
html, body {{ margin:0; padding:0; background:#000; overflow:hidden; }}
#wrap {{ width:100%; height:{height}px; position:relative; background:#000; }}
#overlay {{
  position:absolute; left:12px; top:10px; color:#ddd; z-index:20;
  font:12px ui-monospace, SFMono-Regular, Menlo, monospace;
  line-height:1.35; pointer-events:none; white-space:pre-line;
  text-shadow:0 0 8px rgba(0,0,0,0.85);
}}
#hint {{ position:absolute; right:12px; top:10px; color:#888; z-index:20; font:12px monospace; }}
#fallback {{ position:absolute; left:12px; bottom:10px; color:#ff8a8a; z-index:20; font:12px monospace; display:none; }}
</style>
</head>
<body>
<div id="wrap">
  <div id="overlay"></div>
  <div id="hint">drag = orbit · wheel = zoom · camera persists</div>
  <div id="fallback"></div>
</div>
<script type="module">
(async function() {{
  const payload = {payload_json};
  const THREE_MODULE_SOURCE = {three_source_json};
  const ORBIT_CONTROLS_SOURCE = {orbit_source_json};
  const wrap = document.getElementById('wrap');
  const overlay = document.getElementById('overlay');
  const fallback = document.getElementById('fallback');
  const CAM_KEY = 'babyllm-theatre-camera-v4-corridor';

  function fail(msg) {{
    fallback.style.display = 'block';
    fallback.textContent = msg;
    console.error(msg);
  }}

  let THREE;
  let OrbitControls;
  let threeUrl = null;
  let orbitUrl = null;
  try {{
    threeUrl = URL.createObjectURL(new Blob([THREE_MODULE_SOURCE], {{ type: 'text/javascript' }}));
    const orbitPatched = ORBIT_CONTROLS_SOURCE.replace(
      /from\\s+['\"]three['\"]\\s*;/,
      `from '${{threeUrl}}';`
    );
    orbitUrl = URL.createObjectURL(new Blob([orbitPatched], {{ type: 'text/javascript' }}));
    THREE = await import(threeUrl);
    ({{ OrbitControls }} = await import(orbitUrl));
  }} catch (err) {{
    fail('Three.js failed to load from vendored bundle.');
    console.error(err);
    return;
  }}

  if (!THREE || !OrbitControls) {{
    fail('Three.js renderer failed to initialize.');
    return;
  }}

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);

  const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  wrap.appendChild(renderer.domElement);

  const camera = new THREE.PerspectiveCamera(50, wrap.clientWidth / wrap.clientHeight, 0.1, 1000);
  camera.position.set(28, 16, 64);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.autoRotate = false;
  controls.target.set(0, 0, 24);

  try {{
    const raw = localStorage.getItem(CAM_KEY);
    if (raw) {{
      const cam = JSON.parse(raw);
      if (cam && cam.pos && cam.target) {{
        camera.position.set(cam.pos.x, cam.pos.y, cam.pos.z);
        controls.target.set(cam.target.x, cam.target.y, cam.target.z);
      }}
    }}
  }} catch (err) {{}}
  controls.update();

  controls.addEventListener('change', () => {{
    try {{
      localStorage.setItem(CAM_KEY, JSON.stringify({{
        pos: {{ x: camera.position.x, y: camera.position.y, z: camera.position.z }},
        target: {{ x: controls.target.x, y: controls.target.y, z: controls.target.z }}
      }}));
    }} catch (err) {{}}
  }});

  window.addEventListener('resize', () => {{
    camera.aspect = wrap.clientWidth / wrap.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  }});

  const ambient = new THREE.AmbientLight(0xffffff, 0.82);
  scene.add(ambient);

  const key = new THREE.DirectionalLight(0xffffff, 0.50);
  key.position.set(8, 14, 20);
  scene.add(key);

  const layers = payload.layers;
  const tokenPaths = payload.token_paths;
  const labels = payload.token_labels;
  const cats = payload.token_cats;
  const colors = payload.token_colors;
  const tokenGenerated = payload.token_generated || [];
  const selectedToken = payload.selected_token_index;
  const activeCount = payload.active_count;
  const ambientRibbonCount = payload.ambient_ribbon_count;
  const spawnSpacing = payload.spawn_spacing;
  const globalSpeed = payload.global_speed;
  const seed = payload.seed || 42;
  const blendWeights = payload.blend_weights;    // 4-element array or null
  const cerebWeights = payload.cereb_weights;    // long/short/gate object or null
  const logitLens    = payload.logit_lens;       // per-step per-layer top-K or null

  function hexColor(hex) {{ return new THREE.Color(hex); }}

  // ── architecture-specific scaffold builders ──────────────────────────────

  // Generic INN processing chamber (used for inn_raw, tangle_n, pre_scratch)
  function buildInnScaffold(L) {{
    const g = new THREE.Group();
    const lineMat = new THREE.LineBasicMaterial({{
      color: hexColor(L.color), transparent: true, opacity: 0.11, depthTest: false
    }});
    const cols = 5, rows = 5;
    const x0 = -L.w * 0.34, x1 = L.w * 0.34;
    const y0 = -L.h * 0.34, y1 = L.h * 0.34;
    const z0 = L.z - L.d * 0.34, z1 = L.z + L.d * 0.34;
    const pts = [];
    for (let ix = 0; ix < cols; ix++) {{
      const tx = cols === 1 ? 0 : ix / (cols - 1);
      const x = x0 + (x1 - x0) * tx;
      pts.push(new THREE.Vector3(x, y0, z0), new THREE.Vector3(x, y1, z0));
      pts.push(new THREE.Vector3(x, y0, z1), new THREE.Vector3(x, y1, z1));
      pts.push(new THREE.Vector3(x, y0, z0), new THREE.Vector3(x, y0, z1));
      pts.push(new THREE.Vector3(x, y1, z0), new THREE.Vector3(x, y1, z1));
    }}
    for (let iy = 0; iy < rows; iy++) {{
      const ty = rows === 1 ? 0 : iy / (rows - 1);
      const y = y0 + (y1 - y0) * ty;
      pts.push(new THREE.Vector3(x0, y, z0), new THREE.Vector3(x1, y, z0));
      pts.push(new THREE.Vector3(x0, y, z1), new THREE.Vector3(x1, y, z1));
      pts.push(new THREE.Vector3(x0, y, z0), new THREE.Vector3(x0, y, z1));
      pts.push(new THREE.Vector3(x1, y, z0), new THREE.Vector3(x1, y, z1));
    }}
    pts.push(new THREE.Vector3(x0, 0, L.z), new THREE.Vector3(x1, 0, L.z));
    pts.push(new THREE.Vector3(0, y0, L.z), new THREE.Vector3(0, y1, L.z));
    pts.push(new THREE.Vector3(0, 0, z0), new THREE.Vector3(0, 0, z1));
    const lines = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), lineMat);
    lines.renderOrder = 3;
    g.add(lines);
    const nodeGeom = new THREE.SphereGeometry(0.045, 6, 6);
    const nodeMat = new THREE.MeshBasicMaterial({{
      color: hexColor(L.color), transparent: true, opacity: 0.16, depthTest: false, depthWrite: false
    }});
    const nodes = new THREE.InstancedMesh(nodeGeom, nodeMat, cols * rows);
    nodes.renderOrder = 4;
    const dummy2 = new THREE.Object3D();
    let idx2 = 0;
    for (let ix = 0; ix < cols; ix++) {{
      const tx = cols === 1 ? 0 : ix / (cols - 1);
      const x = x0 + (x1 - x0) * tx;
      for (let iy = 0; iy < rows; iy++) {{
        const ty = rows === 1 ? 0 : iy / (rows - 1);
        const y = y0 + (y1 - y0) * ty;
        dummy2.position.set(x, y, L.z); dummy2.scale.setScalar(1); dummy2.updateMatrix();
        nodes.setMatrixAt(idx2++, dummy2.matrix);
      }}
    }}
    nodes.instanceMatrix.needsUpdate = true;
    g.add(nodes);
    return {{ group: g, lines, nodes }};
  }}

  // NEURON (embed→10k expansion): fan of lines showing 1024→10000 projection
  function buildNeuronExpansionScaffold(L) {{
    const g = new THREE.Group();
    const lineMat = new THREE.LineBasicMaterial({{
      color: hexColor(L.color), transparent: true, opacity: 0.12, depthTest: false
    }});
    const pts = [];
    const nFan = 28;
    const z0 = L.z - L.d * 0.38, z1 = L.z + L.d * 0.38;
    for (let i = 0; i < nFan; i++) {{
      const t = i / (nFan - 1);
      const yIn  = (t - 0.5) * L.h * 0.28;   // narrow at input (1024-d)
      const yOut = (t - 0.5) * L.h * 0.90;   // wide at output (10k-d)
      const xIn  = (t - 0.5) * L.w * 0.22;
      const xOut = (t - 0.5) * L.w * 0.82;
      pts.push(new THREE.Vector3(xIn, yIn, z0), new THREE.Vector3(xOut, yOut, z1));
    }}
    // Input rail (narrow)
    pts.push(new THREE.Vector3(-L.w * 0.11, -L.h * 0.14, z0), new THREE.Vector3(-L.w * 0.11, L.h * 0.14, z0));
    pts.push(new THREE.Vector3( L.w * 0.11, -L.h * 0.14, z0), new THREE.Vector3( L.w * 0.11, L.h * 0.14, z0));
    pts.push(new THREE.Vector3(-L.w * 0.11, -L.h * 0.14, z0), new THREE.Vector3( L.w * 0.11, -L.h * 0.14, z0));
    pts.push(new THREE.Vector3(-L.w * 0.11,  L.h * 0.14, z0), new THREE.Vector3( L.w * 0.11,  L.h * 0.14, z0));
    // Output rail (wide)
    pts.push(new THREE.Vector3(-L.w * 0.41, -L.h * 0.45, z1), new THREE.Vector3(-L.w * 0.41, L.h * 0.45, z1));
    pts.push(new THREE.Vector3( L.w * 0.41, -L.h * 0.45, z1), new THREE.Vector3( L.w * 0.41, L.h * 0.45, z1));
    const lines = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), lineMat);
    lines.renderOrder = 3;
    g.add(lines);
    return {{ group: g, lines, nodes: null }};
  }}

  // CEREBELLUM (9 long + 9 short temporal windows): horizontal slot grid with central divider
  function buildWindowScaffold(L) {{
    const g = new THREE.Group();
    const lineMat = new THREE.LineBasicMaterial({{
      color: hexColor(L.color), transparent: true, opacity: 0.12, depthTest: false
    }});
    const pts = [];
    const nLong = 9, nShort = 9, total = nLong + nShort;
    const x0 = -L.w * 0.40, x1 = L.w * 0.40, xMid = 0;
    const ySpan = L.h * 0.78;
    // Row lines for each window
    for (let i = 0; i <= total; i++) {{
      const y = (i / total - 0.5) * ySpan;
      pts.push(new THREE.Vector3(x0, y, L.z), new THREE.Vector3(x1, y, L.z));
    }}
    // Vertical borders and central divider
    pts.push(new THREE.Vector3(x0, -ySpan * 0.5, L.z), new THREE.Vector3(x0, ySpan * 0.5, L.z));
    pts.push(new THREE.Vector3(x1, -ySpan * 0.5, L.z), new THREE.Vector3(x1, ySpan * 0.5, L.z));
    pts.push(new THREE.Vector3(xMid, -ySpan * 0.5, L.z), new THREE.Vector3(xMid, ySpan * 0.5, L.z));
    // Small tick marks on divider to show window sizes growing left→right
    for (let i = 0; i < nLong; i++) {{
      const y = ((i + 0.5) / total - 0.5) * ySpan;
      const tick = 0.06 + i * 0.018;  // longer ticks for wider windows
      pts.push(new THREE.Vector3(x0 + tick, y, L.z), new THREE.Vector3(x0 + tick * 0.3, y, L.z));
    }}
    const lines = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), lineMat);
    lines.renderOrder = 3;
    g.add(lines);
    // Node at each window slot centre
    const nodeGeom = new THREE.SphereGeometry(0.032, 5, 5);
    const nodeMat = new THREE.MeshBasicMaterial({{
      color: hexColor(L.color), transparent: true, opacity: 0.16, depthTest: false, depthWrite: false
    }});
    const nodes = new THREE.InstancedMesh(nodeGeom, nodeMat, total);
    nodes.renderOrder = 4;
    const dummyW = new THREE.Object3D();
    for (let i = 0; i < total; i++) {{
      const y = ((i + 0.5) / total - 0.5) * ySpan;
      const xNode = i < nLong ? x0 * 0.5 : x1 * 0.5;
      dummyW.position.set(xNode, y, L.z); dummyW.scale.setScalar(1); dummyW.updateMatrix();
      nodes.setMatrixAt(i, dummyW.matrix);
    }}
    nodes.instanceMatrix.needsUpdate = true;
    g.add(nodes);
    return {{ group: g, lines, nodes }};
  }}

  // REFINE2 (10k→512→10k bottleneck MLP): hourglass shape
  function buildBottleneckScaffold(L) {{
    const g = new THREE.Group();
    const lineMat = new THREE.LineBasicMaterial({{
      color: hexColor(L.color), transparent: true, opacity: 0.13, depthTest: false
    }});
    const pts = [];
    const nLines = 18;
    const z0 = L.z - L.d * 0.38, zMid = L.z, z1 = L.z + L.d * 0.38;
    const wFull = L.w * 0.40, wNeck = L.w * 0.07;  // 512/10000 ≈ 0.05
    for (let i = 0; i < nLines; i++) {{
      const t = i / (nLines - 1);
      const y = (t - 0.5) * L.h * 0.80;
      const yNeck = y * 0.14;  // compress heavily at neck
      const xFull = (t - 0.5) * wFull * 2.0;
      const xNeck = (t - 0.5) * wNeck * 2.0;
      pts.push(new THREE.Vector3(xFull, y, z0), new THREE.Vector3(xNeck, yNeck, zMid));
      pts.push(new THREE.Vector3(xNeck, yNeck, zMid), new THREE.Vector3(xFull, y, z1));
    }}
    // Neck rectangle
    pts.push(new THREE.Vector3(-wNeck, -L.h * 0.07, zMid), new THREE.Vector3( wNeck, -L.h * 0.07, zMid));
    pts.push(new THREE.Vector3( wNeck, -L.h * 0.07, zMid), new THREE.Vector3( wNeck,  L.h * 0.07, zMid));
    pts.push(new THREE.Vector3( wNeck,  L.h * 0.07, zMid), new THREE.Vector3(-wNeck,  L.h * 0.07, zMid));
    pts.push(new THREE.Vector3(-wNeck,  L.h * 0.07, zMid), new THREE.Vector3(-wNeck, -L.h * 0.07, zMid));
    const lines = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), lineMat);
    lines.renderOrder = 3;
    g.add(lines);
    return {{ group: g, lines, nodes: null }};
  }}

  // SCRATCHPAD (256 slots × 512-d content-addressable buffer): slot grid
  function buildSlotScaffold(L) {{
    const g = new THREE.Group();
    const lineMat = new THREE.LineBasicMaterial({{
      color: hexColor(L.color), transparent: true, opacity: 0.10, depthTest: false
    }});
    const pts = [];
    const nCols = 16, nRows = 16;  // 16×16 = 256 slots
    const x0 = -L.w * 0.37, x1 = L.w * 0.37;
    const y0 = -L.h * 0.37, y1 = L.h * 0.37;
    const cW = (x1 - x0) / nCols, cH = (y1 - y0) / nRows;
    for (let c = 0; c <= nCols; c++) {{
      const x = x0 + c * cW;
      pts.push(new THREE.Vector3(x, y0, L.z), new THREE.Vector3(x, y1, L.z));
    }}
    for (let r = 0; r <= nRows; r++) {{
      const y = y0 + r * cH;
      pts.push(new THREE.Vector3(x0, y, L.z), new THREE.Vector3(x1, y, L.z));
    }}
    const lines = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), lineMat);
    lines.renderOrder = 3;
    g.add(lines);
    return {{ group: g, lines, nodes: null }};
  }}

  // MEMORY (STM | LTM dual stores): left chamber = short-term decay, right = long-term persistence
  function buildMemoryScaffold(L) {{
    const g = new THREE.Group();
    const lineMat = new THREE.LineBasicMaterial({{
      color: hexColor(L.color), transparent: true, opacity: 0.13, depthTest: false
    }});
    const pts = [];
    const x0 = -L.w * 0.39, x1 = L.w * 0.39, xMid = 0;
    const y0 = -L.h * 0.39, y1 = L.h * 0.39;
    const z = L.z;
    // Outer frame
    pts.push(new THREE.Vector3(x0, y0, z), new THREE.Vector3(x1, y0, z));
    pts.push(new THREE.Vector3(x1, y0, z), new THREE.Vector3(x1, y1, z));
    pts.push(new THREE.Vector3(x1, y1, z), new THREE.Vector3(x0, y1, z));
    pts.push(new THREE.Vector3(x0, y1, z), new THREE.Vector3(x0, y0, z));
    // Central STM/LTM divider
    pts.push(new THREE.Vector3(xMid, y0, z), new THREE.Vector3(xMid, y1, z));
    // STM (left): tapered horizontal stripes — decay towards top
    const nStripes = 6;
    for (let i = 1; i < nStripes; i++) {{
      const ty = i / nStripes;
      const y = y0 + (y1 - y0) * ty;
      const decay = 1.0 - ty * 0.55;  // stripes get shorter as they rise
      pts.push(new THREE.Vector3(x0 * decay, y, z), new THREE.Vector3(-0.06, y, z));
    }}
    // LTM (right): full-width stripes — memory persists
    for (let i = 1; i < nStripes; i++) {{
      const ty = i / nStripes;
      const y = y0 + (y1 - y0) * ty;
      pts.push(new THREE.Vector3(0.06, y, z), new THREE.Vector3(x1 * 0.88, y, z));
    }}
    const lines = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), lineMat);
    lines.renderOrder = 3;
    g.add(lines);
    // One node per memory store
    const nodeGeom = new THREE.SphereGeometry(0.055, 6, 6);
    const nodeMat = new THREE.MeshBasicMaterial({{
      color: hexColor(L.color), transparent: true, opacity: 0.20, depthTest: false, depthWrite: false
    }});
    const nodes = new THREE.InstancedMesh(nodeGeom, nodeMat, 2);
    nodes.renderOrder = 4;
    const dummyM = new THREE.Object3D();
    dummyM.position.set(-L.w * 0.19, 0, z); dummyM.updateMatrix(); nodes.setMatrixAt(0, dummyM.matrix);
    dummyM.position.set( L.w * 0.19, 0, z); dummyM.updateMatrix(); nodes.setMatrixAt(1, dummyM.matrix);
    nodes.instanceMatrix.needsUpdate = true;
    g.add(nodes);
    return {{ group: g, lines, nodes }};
  }}

  // ATTENTION (causal): lower-triangular dot grid showing the causal mask
  function buildAttentionScaffold(L) {{
    const g = new THREE.Group();
    const lineMat = new THREE.LineBasicMaterial({{
      color: hexColor(L.color), transparent: true, opacity: 0.11, depthTest: false
    }});
    const pts = [];
    const n = 9;
    const x0 = -L.w * 0.35, x1 = L.w * 0.35;
    const y0 = -L.h * 0.35, y1 = L.h * 0.35;
    const cW = (x1 - x0) / n, cH = (y1 - y0) / n;
    // Lower-triangle crosses (causal: token i attends to j ≤ i)
    for (let r = 0; r < n; r++) {{
      for (let c = 0; c <= r; c++) {{
        const x = x0 + c * cW + cW * 0.5;
        const y = y0 + r * cH + cH * 0.5;
        pts.push(new THREE.Vector3(x - cW * 0.28, y, L.z), new THREE.Vector3(x + cW * 0.28, y, L.z));
        pts.push(new THREE.Vector3(x, y - cH * 0.28, L.z), new THREE.Vector3(x, y + cH * 0.28, L.z));
      }}
    }}
    // Diagonal boundary line
    pts.push(new THREE.Vector3(x0, y0, L.z), new THREE.Vector3(x1, y1, L.z));
    const lines = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), lineMat);
    lines.renderOrder = 3;
    g.add(lines);
    return {{ group: g, lines, nodes: null }};
  }}

  // EMBED BLEND (token + pos + char + pixel → weighted softmax blend): 4 converging sources
  function buildEmbedBlendScaffold(L) {{
    const g = new THREE.Group();
    const lineMat = new THREE.LineBasicMaterial({{
      color: hexColor(L.color), transparent: true, opacity: 0.13, depthTest: false
    }});
    const pts = [];
    const z0 = L.z - L.d * 0.38, z1 = L.z + L.d * 0.05;
    const nSrc = 4;
    const srcNames = ['tok', 'pos', 'chr', 'pix'];
    for (let i = 0; i < nSrc; i++) {{
      const t = (i + 0.5) / nSrc;
      const xOff = (i - 1.5) * L.w * 0.22;
      const yOff = (i % 2 === 0 ? -1 : 1) * L.h * 0.08;
      const sH = L.h * 0.18, sW = L.w * 0.09;
      // Small source rectangle
      pts.push(new THREE.Vector3(xOff - sW, yOff - sH, z0), new THREE.Vector3(xOff + sW, yOff - sH, z0));
      pts.push(new THREE.Vector3(xOff + sW, yOff - sH, z0), new THREE.Vector3(xOff + sW, yOff + sH, z0));
      pts.push(new THREE.Vector3(xOff + sW, yOff + sH, z0), new THREE.Vector3(xOff - sW, yOff + sH, z0));
      pts.push(new THREE.Vector3(xOff - sW, yOff + sH, z0), new THREE.Vector3(xOff - sW, yOff - sH, z0));
      // Converging line to blend point
      pts.push(new THREE.Vector3(xOff, yOff, z0), new THREE.Vector3(0, 0, z1));
    }}
    // Blend point star
    const bR = L.w * 0.04;
    pts.push(new THREE.Vector3(-bR, 0, z1), new THREE.Vector3(bR, 0, z1));
    pts.push(new THREE.Vector3(0, -bR, z1), new THREE.Vector3(0, bR, z1));
    const lines = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), lineMat);
    lines.renderOrder = 3;
    g.add(lines);
    return {{ group: g, lines, nodes: null }};
  }}

  // ── blend weight bars (inside the 'blend' frustum) ───────────────────────
  function buildBlendBars(L, weights) {{
    if (!weights || weights.length < 4) return null;
    const g = new THREE.Group();
    const names  = ['tok', 'pos', 'rgb', 'chr'];
    const bColors = ['#4fc3f7', '#81c784', '#f48fb1', '#ff8a65'];
    const nBars = 4;
    const barW = L.w_in * 0.10;
    const maxH = L.h_in * 0.70;
    const spacing = L.w_in * 0.30;
    for (let i = 0; i < nBars; i++) {{
      const w = weights[i];
      const barH = Math.max(w * maxH, 0.02);
      const x = (i - 1.5) * spacing;
      const geo = new THREE.BoxGeometry(barW, barH, 0.04);
      const mat = new THREE.MeshBasicMaterial({{
        color: hexColor(bColors[i]), transparent: true, opacity: 0.55, depthTest: false
      }});
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(x, (barH / 2) - maxH * 0.5, L.z - L.d * 0.42);
      mesh.renderOrder = 5;
      g.add(mesh);
    }}
    scene.add(g);
    return g;
  }}

  // ── cerebellum window bars (inside the 'cereb' frustum) ──────────────────
  function buildCerebBars(L, cw) {{
    if (!cw) return null;
    const g = new THREE.Group();
    const longW  = cw.long  || [];
    const shortW = cw.short || [];
    const gate   = cw.gate  != null ? cw.gate : 0.5;
    const allW = longW.concat(shortW);
    const n = allW.length;
    if (n === 0) return null;
    const maxH  = L.h_in * 0.78;
    const barW  = (L.w_in * 0.76) / n;
    for (let i = 0; i < n; i++) {{
      const isShort = i >= longW.length;
      const w = allW[i];
      const barH = Math.max(w * maxH, 0.015);
      const x = (i / (n - 1) - 0.5) * L.w_in * 0.76;
      const col = isShort ? '#ce93d8' : '#4fc3f7';
      const op  = isShort ? (0.25 + gate * 0.55) : (0.25 + (1 - gate) * 0.55);
      const geo = new THREE.BoxGeometry(barW * 0.72, barH, 0.04);
      const mat = new THREE.MeshBasicMaterial({{
        color: hexColor(col), transparent: true, opacity: op, depthTest: false
      }});
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(x, (barH / 2) - maxH * 0.5, L.z);
      mesh.renderOrder = 5;
      g.add(mesh);
    }}
    scene.add(g);
    return g;
  }}

  // ── build all static bars ─────────────────────────────────────────────────
  const blendLayer  = layers.find(l => l.key === 'blend');
  const cerebLayer  = layers.find(l => l.key === 'cereb');
  const blendBars   = blendLayer  ? buildBlendBars(blendLayer,  blendWeights) : null;
  const cerebBars   = cerebLayer  ? buildCerebBars(cerebLayer,  cerebWeights) : null;

  // ── logit-lens overlay helpers ────────────────────────────────────────────
  const LENS_KEYS = ['inn_raw', 'pre_scratch', 'mem1', 'mem2', 'logits'];
  function getLensStep(stepIndex) {{
    if (!logitLens || stepIndex < 0 || stepIndex >= logitLens.length) return null;
    return logitLens[stepIndex];
  }}
  function formatLensBlock(stepData, layerKey) {{
    if (!stepData) return '';
    const ld = stepData[layerKey];
    if (!ld || !ld.tokens || ld.tokens.length === 0) return '';
    const lines = ld.tokens.slice(0, 5).map((t, i) => {{
      const p = (ld.probs[i] * 100).toFixed(1);
      return `  ${{(i+1)}}. "${{t}}"  ${{p}}%`;
    }});
    return `[${{layerKey}}]\\n${{lines.join('\\n')}}`;
  }}

  // ── architecture / box chambers ─────────────────────────────────────────
  const layerMeshes = [];

  const spineMaterial = new THREE.LineBasicMaterial({{
    color: 0x2a2a2a,
    transparent: true,
    opacity: 0.22,
    depthTest: false
  }});
  const spinePts = layers.map(l => new THREE.Vector3(0, 0, l.z));
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(spinePts), spineMaterial));

  for (const L of layers) {{
    const group = new THREE.Group();

    // Entry face (z - d/2): square sized by input_dim
    // Exit  face (z + d/2): square sized by output_dim
    const xi = L.w_in  / 2, yi = L.h_in  / 2;  // entry half-extents
    const xo = L.w_out / 2, yo = L.h_out / 2;  // exit  half-extents
    const zF = L.z - L.d / 2;                   // front (entry) z
    const zB = L.z + L.d / 2;                   // back  (exit)  z

    // ── fill mesh (6-face frustum) ────────────────────────────────────────
    const fVerts = new Float32Array([
      -xi, -yi, zF,   xi, -yi, zF,   xi,  yi, zF,  -xi,  yi, zF,   // 0-3 front
      -xo, -yo, zB,   xo, -yo, zB,   xo,  yo, zB,  -xo,  yo, zB,   // 4-7 back
    ]);
    const fIdx = [
      0,1,2, 0,2,3,   // front face
      4,6,5, 4,7,6,   // back face
      0,4,5, 0,5,1,   // bottom
      3,2,6, 3,6,7,   // top
      0,3,7, 0,7,4,   // left
      1,5,6, 1,6,2,   // right
    ];
    const fillGeo = new THREE.BufferGeometry();
    fillGeo.setAttribute('position', new THREE.BufferAttribute(fVerts, 3));
    fillGeo.setIndex(fIdx);
    const fillMat = new THREE.MeshBasicMaterial({{
      color: hexColor(L.color), transparent: true, opacity: 0.008,
      depthWrite: false, depthTest: false, side: THREE.DoubleSide
    }});
    const fill = new THREE.Mesh(fillGeo, fillMat);
    fill.renderOrder = 1;
    group.add(fill);

    // ── wireframe edges (12 edges: 4 front + 4 back + 4 connecting) ──────
    const ePts = [
      // front face square
      new THREE.Vector3(-xi, -yi, zF), new THREE.Vector3( xi, -yi, zF),
      new THREE.Vector3( xi, -yi, zF), new THREE.Vector3( xi,  yi, zF),
      new THREE.Vector3( xi,  yi, zF), new THREE.Vector3(-xi,  yi, zF),
      new THREE.Vector3(-xi,  yi, zF), new THREE.Vector3(-xi, -yi, zF),
      // back face square
      new THREE.Vector3(-xo, -yo, zB), new THREE.Vector3( xo, -yo, zB),
      new THREE.Vector3( xo, -yo, zB), new THREE.Vector3( xo,  yo, zB),
      new THREE.Vector3( xo,  yo, zB), new THREE.Vector3(-xo,  yo, zB),
      new THREE.Vector3(-xo,  yo, zB), new THREE.Vector3(-xo, -yo, zB),
      // four connecting edges
      new THREE.Vector3(-xi, -yi, zF), new THREE.Vector3(-xo, -yo, zB),
      new THREE.Vector3( xi, -yi, zF), new THREE.Vector3( xo, -yo, zB),
      new THREE.Vector3( xi,  yi, zF), new THREE.Vector3( xo,  yo, zB),
      new THREE.Vector3(-xi,  yi, zF), new THREE.Vector3(-xo,  yo, zB),
    ];
    const edges = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints(ePts),
      new THREE.LineBasicMaterial({{
        color: hexColor(L.color), transparent: true, opacity: 0.18, depthTest: false
      }})
    );
    edges.renderOrder = 2;
    group.add(edges);

    const innScaffold = null;

    scene.add(group);
    layerMeshes.push({{ group, fill, edges, innScaffold, meta: L }});
  }}

  // ── particles / ribbons ──────────────────────────────────────────────────
  const particles = [];
  const ribbonGroup = new THREE.Group();
  scene.add(ribbonGroup);

  const tokenRadius = activeCount >= 3072 ? 0.05 : (activeCount >= 1536 ? 0.065 : 0.085);
  const tokenSegments = activeCount >= 1536 ? 7 : 10;
  const tokenGeom = new THREE.SphereGeometry(tokenRadius, tokenSegments, tokenSegments);
  const tokenMat = new THREE.MeshBasicMaterial({{
    color: 0xffffff,
    transparent: true,
    opacity: 0.88,
    depthWrite: false
  }});
  const tokenMesh = new THREE.InstancedMesh(tokenGeom, tokenMat, activeCount);
  tokenMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  tokenMesh.frustumCulled = false;
  tokenMesh.renderOrder = 10;
  scene.add(tokenMesh);
  const tokenDummy = new THREE.Object3D();

  const selectedGeom = new THREE.OctahedronGeometry(0.25, 0);

  function mulberry32(a) {{
    return function() {{
      let t = a += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }}
  }}

  const rng = mulberry32(seed);
  const shuffled = [...Array(tokenPaths.length).keys()];
  for (let i = shuffled.length - 1; i > 0; i--) {{
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }}

  for (let i = 0; i < activeCount; i++) {{
    let line = null;
    if (i < ambientRibbonCount) {{
      const lineMat = new THREE.LineBasicMaterial({{
        color: 0xffffff,
        transparent: true,
        opacity: 0.07,
        depthTest: false
      }});
      line = new THREE.Line(new THREE.BufferGeometry(), lineMat);
      line.renderOrder = 6;
      ribbonGroup.add(line);
    }}

    const tokenIndex = shuffled[i % shuffled.length];
    tokenMesh.setColorAt(i, hexColor(colors[tokenIndex]));

    particles.push({{
      line,
      tokenIndex,
      offset: i * spawnSpacing,
      speed: 1.0 + (rng() * 0.08 - 0.04)
    }});
  }}
  if (tokenMesh.instanceColor) {{
    tokenMesh.instanceColor.needsUpdate = true;
  }}

  const selectedMesh = new THREE.Mesh(
    selectedGeom,
    new THREE.MeshBasicMaterial({{ color: 0xffffff, depthTest: false }})
  );
  selectedMesh.renderOrder = 20;
  scene.add(selectedMesh);

  const selectedLine = new THREE.Line(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({{
      color: 0xcc66ff,
      transparent: true,
      opacity: 0.94,
      depthTest: false
    }})
  );
  selectedLine.renderOrder = 19;
  scene.add(selectedLine);

  function smoothstep(t) {{ return t * t * (3.0 - 2.0 * t); }}
  function lerp(a, b, t) {{ return a + (b - a) * t; }}

  function sampleTokenPath(tokenIndex, globalProgress) {{
    const path = tokenPaths[tokenIndex];
    const nSeg = path.length - 1;
    const wrapped = ((globalProgress % nSeg) + nSeg) % nSeg;
    const seg = Math.floor(wrapped);
    const local = wrapped - seg;
    const e = smoothstep(local);
    const a = path[seg];
    const b = path[seg + 1];
    return {{
      x: lerp(a[0], b[0], e),
      y: lerp(a[1], b[1], e),
      z: lerp(a[2], b[2], e),
      seg,
      local: e
    }};
  }}

  function buildTrail(tokenIndex, globalProgress, samples, length) {{
    const pts = [];
    for (let i = 0; i < samples; i++) {{
      const backT = globalProgress - (length * (1 - i / (samples - 1)));
      const s = sampleTokenPath(tokenIndex, backT);
      pts.push(new THREE.Vector3(s.x, s.y, s.z));
    }}
    return pts;
  }}

  function setLinePoints(line, pts) {{
    line.geometry.dispose();
    line.geometry = new THREE.BufferGeometry().setFromPoints(pts);
  }}

  let t0 = performance.now() * 0.001;

  function animate() {{
    requestAnimationFrame(animate);
    const now = performance.now() * 0.001;
    const elapsed = now - t0;

    const focusProg = elapsed * globalSpeed;
    const focus = sampleTokenPath(selectedToken, focusProg);
    const layerA = layers[Math.max(0, Math.min(layers.length - 1, focus.seg))];
    const layerB = layers[Math.max(0, Math.min(layers.length - 1, focus.seg + 1))];

    for (const item of layerMeshes) {{
      const L = item.meta;
      const isFocusWindow = (L.key === layerA.key || L.key === layerB.key);
      const pulse = isFocusWindow ? (1.0 + 0.08 * Math.sin(elapsed * 4.2 + L.z * 0.07)) : 1.0;

      item.group.scale.set(1, 1, 1);
      item.fill.material.opacity  = isFocusWindow ? 0.016 : 0.008;
      item.edges.material.opacity = isFocusWindow ? 0.30  : 0.18;

      if (item.innScaffold) {{
        const hot = isFocusWindow;
        item.innScaffold.lines.material.opacity = hot ? 0.22 : 0.11;
        if (item.innScaffold.nodes) {{
          item.innScaffold.nodes.material.opacity = hot ? 0.34 : 0.16;
        }}
        const innerPulse = hot ? (1.0 + 0.05 * Math.sin(elapsed * 7.5 + L.z * 0.11)) : 1.0;
        item.innScaffold.group.scale.set(innerPulse, innerPulse, 1.0);
      }}
    }}

    for (let i = 0; i < particles.length; i++) {{
      const p = particles[i];
      const prog = elapsed * globalSpeed * p.speed - p.offset;
      const s = sampleTokenPath(p.tokenIndex, prog);
      const pulse = 0.76 + 0.28 * (1.0 - Math.min(1.0, s.local * 0.75));
      tokenDummy.position.set(s.x, s.y, s.z);
      tokenDummy.scale.setScalar(pulse);
      tokenDummy.updateMatrix();
      tokenMesh.setMatrixAt(i, tokenDummy.matrix);

      if (p.line) {{
        const trailPts = buildTrail(p.tokenIndex, prog, 18, 2.25);
        setLinePoints(p.line, trailPts);
        p.line.material.color.set(colors[p.tokenIndex]);
        p.line.material.opacity = 0.07;
      }}
    }}
    tokenMesh.instanceMatrix.needsUpdate = true;

    selectedMesh.position.set(focus.x, focus.y, focus.z);
    const focusTrail = buildTrail(selectedToken, focusProg, 44, 4.4);
    setLinePoints(selectedLine, focusTrail);

    // ── logit lens: show predictions for current focus token at nearby layers ──
    const stepData = getLensStep(selectedToken);
    const nearKey  = layerA.key;
    let lensText = '';
    if (stepData) {{
      for (const lk of LENS_KEYS) {{
        const block = formatLensBlock(stepData, lk);
        if (block) lensText += '\\n' + block;
      }}
    }}

    // ── colour generated tokens orange vs normal ──────────────────────────────
    const isGen = tokenGenerated[selectedToken];

    overlay.textContent =
      `focus: ${{labels[selectedToken]}}${{isGen ? '  [generated]' : ''}}
` +
      `cat: ${{cats[selectedToken]}}   flow: ${{layerA.key}} → ${{layerB.key}}
` +
      `stage: ${{layerA.stage}}${{lensText}}`;

    controls.update();
    renderer.render(scene, camera);
  }}

  animate();

  window.addEventListener('beforeunload', () => {{
    if (orbitUrl) URL.revokeObjectURL(orbitUrl);
    if (threeUrl) URL.revokeObjectURL(threeUrl);
  }});
}})();
</script>
</body>
</html>
"""
    components.html(html, height=height, scrolling=False)


# ── UI / data ────────────────────────────────────────────────────────────────
cache_files = _available_cache_files()
if not cache_files:
    st.error("No probe cache in .probe_cache/. Run latent_probe_ui.py first.")
    st.stop()

selected_cache = st.sidebar.selectbox(
    "Probe cache",
    options=cache_files,
    format_func=lambda p: os.path.basename(p),
)
entries, neutral_data, tiled_data, seq_extras = _load_npz(selected_cache)

_known = set(neutral_data.keys())
layer_keys = tuple([k for k in _LAYER_ORDER if k in _known] + sorted(_known - set(_LAYER_ORDER)))
if len(layer_keys) < 2:
    st.error(f"Only {len(layer_keys)} layer(s) in cache — need at least 2.")
    st.stop()

n_tokens = len(entries)
max_visible_tokens = min(n_tokens, _MAX_VISIBLE_TOKENS)
token_slider_min = 16 if max_visible_tokens >= 16 else 1
token_slider_step = 16 if max_visible_tokens <= 512 else (32 if max_visible_tokens <= 2048 else 64)
token_slider_default = min(256, max_visible_tokens)

st.sidebar.markdown("---")
selected_idx = st.sidebar.selectbox(
    "Focus token",
    options=list(range(n_tokens)),
    format_func=lambda i: f"{_token_label(entries[i])}  [{i}]",
    index=0,
)
active_count = st.sidebar.slider(
    "Visible flowing tokens",
    token_slider_min,
    max_visible_tokens,
    token_slider_default,
    token_slider_step,
)
spawn_spacing = st.sidebar.slider("Spawn spacing", 0.02, 0.45, 0.09, 0.01)
global_speed = st.sidebar.slider("Flow speed", 0.06, 1.50, 0.32, 0.01)
seed = st.sidebar.number_input("Shuffle seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Install on Python side:\n"
    "pip install streamlit numpy scipy scikit-learn\n\n"
    "Browser side Three.js is vendored locally."
)

z_depths = _layer_z_depths(layer_keys)
layer_xy = compute_layer_xy_world(neutral_data, layer_keys)
layer_depths = _layer_slab_depths(layer_keys, z_depths)

layer_meta = []
for lk in layer_keys:
    sid = _STAGE_OF.get(lk, -1)
    dim = int(neutral_data[lk].shape[1])
    box = _arch_box_metrics(lk, layer_depths[lk])
    layer_meta.append({
        "key": lk,
        "stage": _STAGE_LABEL.get(sid, "?"),
        "z": float(z_depths[lk]),
        "dim": dim,
        "w_in":  box["w_in"],
        "h_in":  box["h_in"],
        "w_out": box["w_out"],
        "h_out": box["h_out"],
        "d":     box["d"],
        "color": _STAGE_COLOR.get(sid, "#aaaaaa"),
    })

ambient_ribbon_count = min(active_count if active_count <= 384 else _MAX_AMBIENT_RIBBONS, active_count)

all_paths = []
for ti in range(n_tokens):
    pts = []
    for lk in layer_keys:
        xy = layer_xy[lk][ti]
        pts.append([float(xy[0]), float(xy[1]), float(z_depths[lk])])
    all_paths.append(pts)

# ── sequence-probe extras for JS ─────────────────────────────────────────────
_token_generated = [bool(e.get("generated", False)) for e in entries]

# blend_weights: use step-0 values (learned params, same every step in eval)
_blend_weights = None
if "blend_weights" in seq_extras:
    bw = seq_extras["blend_weights"]
    _blend_weights = [float(x) for x in bw[0]]  # [tok, pos, rgb, char]

# cereb_weights: same (learned params)
_cereb_weights = None
if "cereb_long_weights" in seq_extras:
    cl = seq_extras["cereb_long_weights"][0].tolist()
    cs = seq_extras["cereb_short_weights"][0].tolist()
    cg = float(seq_extras["cereb_short_gate"][0])
    _cereb_weights = {"long": cl, "short": cs, "gate": cg}

# logit_lens: per-step, keyed by layer_key
_logit_lens = seq_extras.get("logit_lens", None)  # list[dict] or None

payload = {
    "layers": layer_meta,
    "token_paths": all_paths,
    "token_labels": [_token_label(e) for e in entries],
    "token_cats": [_safe_cat(e) for e in entries],
    "token_colors": [_tok_color(e) for e in entries],
    "token_generated": _token_generated,
    "selected_token_index": int(selected_idx),
    "active_count": int(min(active_count, n_tokens)),
    "ambient_ribbon_count": int(ambient_ribbon_count),
    "spawn_spacing": float(spawn_spacing),
    "global_speed": float(global_speed),
    "seed": int(seed),
    "blend_weights": _blend_weights,
    "cereb_weights": _cereb_weights,
    "logit_lens": _logit_lens,
}

_render_world(payload, height=920)

# ── readouts ──────────────────────────────────────────────────────────────────
a, b, c = st.columns(3)
with a:
    st.markdown("### world")
    st.write(f"cache: `{os.path.basename(selected_cache)}`")
    st.write(f"tokens in cache: **{n_tokens}**")
    st.write(f"visible looping tokens: **{min(active_count, n_tokens)}**")
    st.write(f"ambient ribbons: **{ambient_ribbon_count}**")

with b:
    st.markdown("### focus token")
    e = entries[selected_idx]
    st.write(f"label: **{_token_label(e)}**")
    st.write(f"id: **{e['id']}**")
    st.write(f"cat: **{_safe_cat(e)}**")

with c:
    st.markdown("### architecture")
    st.write(f"layers: **{len(layer_keys)}**")
    st.write("pipeline: **EMBED → ATTENTION → INN → POST-INN → MEMORY → LOGITS**")

with st.expander("selected token stations through the network"):
    rows = []
    for lk in layer_keys:
        xy = layer_xy[lk][selected_idx]
        rows.append({
            "stage": _STAGE_LABEL.get(_STAGE_OF.get(lk, -1), "?"),
            "layer": lk,
            "dim": int(neutral_data[lk].shape[1]),
            "x": round(float(xy[0]), 3),
            "y": round(float(xy[1]), 3),
            "z": round(float(z_depths[lk]), 3),
            "desc": _LAYER_DESC.get(lk, ""),
        })
    st.dataframe(rows, use_container_width=True)

with st.expander("what to install / what this is"):
    st.markdown(
        """
**Install:**
```bash
pip install streamlit numpy scipy scikit-learn
```

This renderer uses a vendored local Three.js bundle injected into the browser iframe, so there is no npm step for this prototype.

Each chamber's internal scaffold reflects the actual layer architecture:

| layer | scaffold |
|---|---|
| `neurons` | expansion fan — 1024-d input rail fans out to 10k-d output rail |
| `cereb` | 18-slot horizontal grid — 9 long-range + 9 short-range temporal windows |
| `refine2` | hourglass — 10k → 512-d bottleneck neck → 10k residual |
| `scratch` | 16×16 cell grid — 256 content-addressable scratchpad slots (512-d each) |
| `mem1` / `mem2` | split chamber — left = short-term (tapered, decays); right = long-term (full, persists) |
| `attn1` / `attn2` | lower-triangle cross grid — causal mask (token i sees only j ≤ i) |
| `blend` | 4 converging sources — token + positional + char + pixel embedding blend |
| `inn_raw`, `tangle_n`, `pre_scratch` | generic 5×5 node grid — general INN processing |

This is still a theatre / conceptual renderer, not yet a literal neuron-gate simulator.
"""
    )
