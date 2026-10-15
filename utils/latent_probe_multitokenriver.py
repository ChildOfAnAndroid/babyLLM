#!/usr/bin/env python3
# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm theatre (three, box architecture) -*- ⊂ʕʘʔ⊂ʔ ---
# utils/latent_probe_multitokenriver.py
#
# Run from project root:
#   python3 -m streamlit run utils/latent_probe_multitokenriver.py
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
    "attn1": 1, "tangle_e": 1,
    "neurons": 2, "cereb": 2, "refine2": 2,
    "inn_raw": 3, "attn2": 3, "tangle_n": 3, "pre_scratch": 3, "scratch": 3,
    "mem1": 4, "tangle_m": 4, "mem2": 4,
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

_WORLD_RADIUS = 8.5
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
    return available_cache_files(CACHE_DIR, include_sequences=False)


def _load_npz(cache_path: str):
    return load_probe_cache(cache_path, include_extras=False)


def _token_label(e: Dict[str, Any]) -> str:
    return token_label(e)


def _safe_cat(e: Dict[str, Any]) -> str:
    return safe_category(e)


def _tok_color(e: Dict[str, Any]) -> str:
    if e.get("labeled"):
        return "#ffeb3b"
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
        z = {k: (v - lo) / rng * 26.0 for k, v in z.items()}
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

    z = PCA(n_components=nc, whiten=True, random_state=42).fit_transform(a).astype(np.float32)
    z = np.nan_to_num(z, copy=False)
    xy = z @ _PROJECTION_MIX[:nc]
    return _normalize_xy(xy)


def _align_to_prev(prev_xy: np.ndarray, curr_xy: np.ndarray) -> np.ndarray:
    a0 = _center_xy(prev_xy)
    b0 = _center_xy(curr_xy)
    try:
        r, _ = orthogonal_procrustes(b0, a0)
        aligned = b0 @ r
    except Exception:
        aligned = b0
    return _normalize_xy(aligned)


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
        xy = _layer_balanced_2d(neutral_data[lk])
        if prev_xy is not None:
            if float(np.std(xy)) < 1e-6:
                xy = prev_xy * 0.96
            else:
                xy = _align_to_prev(prev_xy, xy)
                flatness = _xy_flatness_ratio(xy)
                if flatness < 0.08:
                    carry = 0.42 if flatness < 0.02 else 0.24
                    xy = _normalize_xy(((1.0 - carry) * xy) + (carry * prev_xy))
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
        out[lk] = float(np.clip(local_gap * 0.32, 0.34, 0.85))
    return out


def _layer_box_metrics(xy: np.ndarray, depth: float) -> Dict[str, float]:
    a = _center_xy(xy)

    x_lo, x_hi = float(np.min(a[:, 0])), float(np.max(a[:, 0]))
    y_lo, y_hi = float(np.min(a[:, 1])), float(np.max(a[:, 1]))
    span_x = max(2.0 * max(abs(x_lo), abs(x_hi)), 0.6)
    span_y = max(2.0 * max(abs(y_lo), abs(y_hi)), 0.6)

    pad_x = max(0.18, span_x * 0.06)
    pad_y = max(0.18, span_y * 0.06)
    w = span_x + 2.0 * pad_x
    h = span_y + 2.0 * pad_y

    qx_lo, qx_hi = np.quantile(a[:, 0], [0.18, 0.82])
    qy_lo, qy_hi = np.quantile(a[:, 1], [0.18, 0.82])
    inner_w = max(0.7, 2.0 * max(abs(float(qx_lo)), abs(float(qx_hi))) + 2.0 * max(0.08, pad_x * 0.35))
    inner_h = max(0.7, 2.0 * max(abs(float(qy_lo)), abs(float(qy_hi))) + 2.0 * max(0.08, pad_y * 0.35))
    inner_w = min(inner_w, w * 0.94)
    inner_h = min(inner_h, h * 0.94)

    return {
        "w": float(w),
        "h": float(h),
        "d": float(depth),
        "inner_w": float(inner_w),
        "inner_h": float(inner_h),
    }


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
  const CAM_KEY = 'babyllm-theatre-camera-v3-inn';

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
  camera.position.set(20, 12, 34);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.autoRotate = false;
  controls.target.set(0, 0, 12);

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
  const selectedToken = payload.selected_token_index;
  const activeCount = payload.active_count;
  const ambientRibbonCount = payload.ambient_ribbon_count;
  const spawnSpacing = payload.spawn_spacing;
  const globalSpeed = payload.global_speed;
  const seed = payload.seed || 42;

  function hexColor(hex) {{ return new THREE.Color(hex); }}

  function buildInnScaffold(L) {{
    const g = new THREE.Group();
    const lineMat = new THREE.LineBasicMaterial({{
      color: hexColor(L.color),
      transparent: true,
      opacity: 0.11,
      depthTest: false
    }});

    const cols = 5;
    const rows = 5;
    const x0 = -L.w * 0.34;
    const x1 =  L.w * 0.34;
    const y0 = -L.h * 0.34;
    const y1 =  L.h * 0.34;
    const z0 =  L.z - L.d * 0.34;
    const z1 =  L.z + L.d * 0.34;

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

    const lines = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints(pts),
      lineMat
    );
    lines.renderOrder = 3;
    g.add(lines);

    const nodeGeom = new THREE.SphereGeometry(0.045, 6, 6);
    const nodeMat = new THREE.MeshBasicMaterial({{
      color: hexColor(L.color),
      transparent: true,
      opacity: 0.16,
      depthTest: false,
      depthWrite: false
    }});
    const nodes = new THREE.InstancedMesh(nodeGeom, nodeMat, cols * rows);
    nodes.renderOrder = 4;
    const dummy = new THREE.Object3D();
    let idx = 0;
    for (let ix = 0; ix < cols; ix++) {{
      const tx = cols === 1 ? 0 : ix / (cols - 1);
      const x = x0 + (x1 - x0) * tx;
      for (let iy = 0; iy < rows; iy++) {{
        const ty = rows === 1 ? 0 : iy / (rows - 1);
        const y = y0 + (y1 - y0) * ty;
        dummy.position.set(x, y, L.z);
        dummy.scale.setScalar(1);
        dummy.updateMatrix();
        nodes.setMatrixAt(idx++, dummy.matrix);
      }}
    }}
    nodes.instanceMatrix.needsUpdate = true;
    g.add(nodes);

    return {{ group: g, lines, nodes }};
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

    const boxGeo = new THREE.BoxGeometry(L.w, L.h, L.d);
    const fillMat = new THREE.MeshBasicMaterial({{
      color: hexColor(L.color),
      transparent: true,
      opacity: 0.008,
      depthWrite: false,
      depthTest: false
    }});
    const fill = new THREE.Mesh(boxGeo, fillMat);
    fill.position.z = L.z;
    fill.renderOrder = 1;
    group.add(fill);

    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(boxGeo),
      new THREE.LineBasicMaterial({{
        color: hexColor(L.color),
        transparent: true,
        opacity: 0.18,
        depthTest: false
      }})
    );
    edges.position.z = L.z;
    edges.renderOrder = 2;
    group.add(edges);

    const innerGeo = new THREE.PlaneGeometry(L.inner_w, L.inner_h);
    const innerMat = new THREE.MeshBasicMaterial({{
      color: new THREE.Color(0x111111),
      transparent: true,
      opacity: 0.05,
      side: THREE.DoubleSide,
      depthWrite: false,
      depthTest: false
    }});
    const inner = new THREE.Mesh(innerGeo, innerMat);
    inner.position.z = L.z + (L.d * 0.5) + 0.005;
    inner.renderOrder = 1;
    group.add(inner);

    const innerEdge = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.PlaneGeometry(L.inner_w, L.inner_h)),
      new THREE.LineBasicMaterial({{
        color: hexColor(L.color),
        transparent: true,
        opacity: 0.14,
        depthTest: false
      }})
    );
    innerEdge.position.z = L.z + (L.d * 0.5) + 0.006;
    innerEdge.renderOrder = 2;
    group.add(innerEdge);

    const capGeo = new THREE.PlaneGeometry(L.w * 0.82, L.h * 0.82);
    const capMat = new THREE.MeshBasicMaterial({{
      color: new THREE.Color(0x080808),
      transparent: true,
      opacity: 0.035,
      side: THREE.DoubleSide,
      depthWrite: false,
      depthTest: false
    }});
    const cap = new THREE.Mesh(capGeo, capMat);
    cap.position.z = L.z - (L.d * 0.5) - 0.004;
    cap.renderOrder = 1;
    group.add(cap);

    let innScaffold = null;
    if (L.stage === 'INN' || L.stage === 'POST-INN') {{
      innScaffold = buildInnScaffold(L);
      group.add(innScaffold.group);
    }}

    scene.add(group);
    layerMeshes.push({{ group, fill, edges, inner, innerEdge, cap, innScaffold, meta: L }});
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
      item.fill.material.opacity = isFocusWindow ? 0.016 : 0.008;
      item.edges.material.opacity = isFocusWindow ? 0.28 : 0.18;
      item.inner.material.opacity = isFocusWindow ? 0.09 : 0.05;
      item.innerEdge.material.opacity = isFocusWindow ? 0.24 : 0.14;
      item.inner.scale.set(pulse, pulse, 1);
      item.innerEdge.scale.set(pulse, pulse, 1);

      if (item.innScaffold) {{
        const hot = isFocusWindow;
        item.innScaffold.lines.material.opacity = hot ? 0.22 : 0.11;
        item.innScaffold.nodes.material.opacity = hot ? 0.34 : 0.16;
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

    overlay.textContent =
      `focus token: ${{labels[selectedToken]}}
` +
      `cat: ${{cats[selectedToken]}}
` +
      `flow: ${{layerA.key}} → ${{layerB.key}}
` +
      `stage: ${{layerA.stage}}`;

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
entries, neutral_data, tiled_data = _load_npz(selected_cache)

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
    box = _layer_box_metrics(layer_xy[lk], layer_depths[lk])
    layer_meta.append({
        "key": lk,
        "stage": _STAGE_LABEL.get(sid, "?"),
        "z": float(z_depths[lk]),
        "dim": dim,
        "w": box["w"],
        "h": box["h"],
        "d": box["d"],
        "inner_w": box["inner_w"],
        "inner_h": box["inner_h"],
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

payload = {
    "layers": layer_meta,
    "token_paths": all_paths,
    "token_labels": [_token_label(e) for e in entries],
    "token_cats": [_safe_cat(e) for e in entries],
    "token_colors": [_tok_color(e) for e in entries],
    "selected_token_index": int(selected_idx),
    "active_count": int(min(active_count, n_tokens)),
    "ambient_ribbon_count": int(ambient_ribbon_count),
    "spawn_spacing": float(spawn_spacing),
    "global_speed": float(global_speed),
    "seed": int(seed),
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

What changed in this version:
- tokens stay round / orb-like
- layers are rectangular chambers / slabs / boxes
- the machine reads as a shaped architecture, not a stack of portals
- chamber footprints are sized from the plotted token spread at each layer
- aperture rectangles hint at pass-through gates inside each chamber
- large token counts automatically keep only a sampled set of background ribbons
- INN / POST-INN chambers now contain an internal scaffold so the middle of the machine feels denser

This is still a theatre / conceptual renderer, not yet a literal neuron-gate simulator.
"""
    )
