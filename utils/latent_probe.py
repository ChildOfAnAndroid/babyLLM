#!/usr/bin/env python3
# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm latent space probe -*- ⊂ʕʘ‿ʘ૮ʔ ---
# utils/latent_probe.py
#
# STRATEGIES — run both per window size, then show the diff:
#
#   "neutral"  : [NEUTRAL_FILL × (window_max-1)] + [probe_token]
#                Only the final position differs between tokens.
#                Cleanest per-token isolation.
#
#   "tiled"    : [probe_token × window_max]
#                Entire context is the same repeated token.
#                Activates all cerebellum window sizes equally.
#                May collapse tokens that share characters.
#
# DIFF PANEL — both strategies projected together into the SAME 2D space
#   (neutral + tiled stacked → single PCA/t-SNE run → split back).
#   Arrows connect each token's neutral position → tiled position.
#   Arrow colour = displacement magnitude (hot colourmap).
#   Long arrows = strategy-sensitive token.  Short = strategy-stable.
#
# WINDOW SIZES — each entry in WINDOW_SIZES produces its own figure.
#   Use a shorter window (e.g. 64) to see how context length changes topology.
#
# Run from project root:  python utils/latent_probe.py

import sys, os, random
from functools import reduce

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from babyLLM import BABYLLM
from school.staffroom.counsellor import COUNSELLOR
from school.staffroom.calligraphist import S_OUTPUT
from school.staffroom.librarian import LIBRARIAN
from school.staffroom.HE_IS_SCRIBE import SCRIBE
from config import modelDevice, learningRateGOAL

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
# Words to highlight and label on the plots.  bby is trained on lowercase only
# — uppercase tokens map to <UNK>.  All entries here must be lowercase.
LABEL_WORDS = [
    "boof", "butt", "sad", "geepy", "help", "computer",
    "cat", "dog", "happy", "angry", "water", "fire",
    "love", "hate", "hello", "goodbye", "yes", "no",
    "charis", "elodie", "ace"
]

N_VOCAB_SAMPLE  = 4200    # total probe tokens (incl. all base chars)
NEUTRAL_FILL    = "."    # token used as neutral context filler
RANDOM_SEED     = 4200
MAX_COLS        = 5      # max columns per row in the figure

# Run the probe at each of these context lengths.
# Each produces a separate output figure.
WINDOW_SIZES = [269]

# Output filename template — {w} is replaced with the window size.
OUTPUT_PNG_TEMPLATE = "latent_probe_w{w}.png"

# ── Layer toggles ─────────────────────────────────────────────────────────────
# Set a layer to False to skip it entirely (not probed, not plotted).
# Order here matches the pipeline order and the column order in the figure.
#
# NOTE: babyLLM.forward() calls interneuronNetwork / memory / logits via
# .forward() directly, bypassing nn.Module.__call__ and all hooks on those
# modules.  We therefore hook the innermost __call__-dispatched sublayer
# inside each block instead (see _ALL_LAYER_DEFS below).
LAYERS = {
    # ── Embedding stage ────────────────────────────────────────────────────────
    "embed":         True,   # raw token embedding lookup                  (1024-d)
    "pos":           True,   # positional embedding                        (1024-d)
    "char":          True,   # character-level embedding (char_projector)  (1024-d)
    "blend":         True,   # blended token+pos+char, pre-attention₁      (1024-d)
    # ── Attention + tangling (embed stage) ────────────────────────────────────
    "attn1":         True,   # attention₁ output                           (1024-d)
    "tangle_e_down": True,   # tangling embed projection down              (1024→256)
    "tangle_e_up":   True,   # tangling embed projection up                (256→1024)
    "tangle_e":      True,   # tangling embed delta (Δ)                    (1024-d)
    # ── INN stage ─────────────────────────────────────────────────────────────
    "neurons_norm":  True,   # INN inputNorm before expansion              (1024-d)
    "neurons":       True,   # INN neuron firings, last-token row          (10k-d)
    "cereb":         True,   # cerebellum output BEFORE refinement2        (10k-d)
    "refine2_down":  True,   # refinement2 bottleneck compress             (10k→512)
    "refine2_up":    True,   # refinement2 bottleneck expand               (512→10k)
    "refine2":       True,   # INN refinement2 bottleneck output           (10k-d)
    # ── Post-INN + tangling (neuron stage) ────────────────────────────────────
    "inn_raw":       True,   # raw INN FINALout, before attention₂         (10k-d)
    "attn2":         True,   # attention₂ delta (Δ)                        (10k-d)
    "tangle_n_down": True,   # tangling neuron projection down             (10k→128)
    "tangle_n_up":   True,   # tangling neuron projection up               (128→10k)
    "tangle_n":      True,   # tangling neuron delta (Δ)                   (10k-d)
    "pre_scratch":   True,   # INN+attn2+tangle combined, before scratchpad(10k-d)
    "scratch_down":  True,   # scratchpad project_down                     (10k→512)
    "scratch_up":    True,   # scratchpad project_up                       (512→10k)
    "scratch":       True,   # scratchpad delta (Δ)                        (10k-d)
    # ── Memory stage + tangling (memory stage) ────────────────────────────────
    "mem1_reduce":   True,   # memory₁ inputReducer                       (10k→1024)
    "mem1_inf_down": True,   # memory₁ memoryInfluence2 compress           (1024→512)
    "mem1_inf_up":   True,   # memory₁ memoryInfluence2 expand             (512→10k)
    "mem1":          True,   # memory₁ output_norm proxy                  (10k-d)
    "tangle_m_down": True,   # tangling memory projection down             (10k→128)
    "tangle_m_up":   True,   # tangling memory projection up               (128→10k)
    "tangle_m":      True,   # tangling memory delta (Δ)                   (10k-d)
    "mem2_reduce":   True,   # memory₂ inputReducer                       (10k→1024)
    "mem2_inf_down": True,   # memory₂ memoryInfluence2 compress           (1024→512)
    "mem2_inf_up":   True,   # memory₂ memoryInfluence2 expand             (512→10k)
    "mem2":          True,   # memory₂ output_norm proxy                  (10k-d)
    # ── Logits ────────────────────────────────────────────────────────────────
    "logits":        True,   # final logitNorm output                      (vocab-d)
}
# ─────────────────────────────────────────────────────────────────────────────

# Master layer registry — filtered by LAYERS at runtime.
# Tuple: (attr_path, short_label, long_title, extractor_key, hook_src)
#   extractor_key: "last"    → take last row of sequence dimension
#                  "squeeze" → squeeze leading dims, take last row if still 2-d
#   hook_src:      "output"  → hook fires on module return value
#                  "input"   → hook fires on inp[0] (used for cereb to get the
#                              tensor going INTO refinement2, i.e. pre-bottleneck)
_ALL_LAYER_DEFS = [
    # ── Embedding stage ────────────────────────────────────────────────────────
    ("embed",                                "embed",         "token embed\n(lookup, 1024-d)",                "last",    "output"),
    ("embed.posEmbedding",                   "pos",           "pos embed\n(positional, 1024-d)",              "last",    "output"),
    ("char_projector",                       "char",          "char embed\n(char-level, 1024-d)",             "last",    "output"),
    ("attention",                            "blend",         "blend\n(tok+pos+char pre-attn₁, 1024-d)",     "last",    "input"),
    # ── Attention₁ + tangling (embed stage) ───────────────────────────────────
    ("attention",                            "attn1",         "attention₁\n(post-attn, 1024-d)",             "last",    "output"),
    ("tangling.embed_proj_down",             "tangle_e_down", "tangle·embed ↓\n(1024→256)",                  "last",    "output"),
    ("tangling.embed_proj_up",               "tangle_e_up",   "tangle·embed ↑\n(256→1024)",                  "last",    "output"),
    ("tangling.embed_out_norm",              "tangle_e",      "tangle embed\n(Δ delta, 1024-d)",             "last",    "output"),
    # ── INN stage ─────────────────────────────────────────────────────────────
    ("interneuronNetwork.neurons.inputNorm", "neurons_norm",  "INN pre-norm\n(inputNorm, 1024-d)",           "last",    "output"),
    ("interneuronNetwork.neurons",           "neurons",       "INN neurons\n(last tok, 10k-d)",               "last",    "output"),
    ("interneuronNetwork.refinement2",       "cereb",         "cereb\n(long+short windows, 10k-d)",           "squeeze", "input"),
    ("interneuronNetwork.refinement2.0",     "refine2_down",  "refine2 ↓\n(10k→512 compress)",               "squeeze", "output"),
    ("interneuronNetwork.refinement2.3",     "refine2_up",    "refine2 ↑\n(512→10k expand)",                 "squeeze", "output"),
    ("interneuronNetwork.refinement2",       "refine2",       "refine2\n(post-bottleneck, 10k-d)",            "squeeze", "output"),
    # ── Post-INN + tangling (neuron stage) ────────────────────────────────────
    ("attention2",                           "inn_raw",       "INN raw\n(FINALout pre-attn₂, 10k-d)",        "squeeze", "input"),
    ("attention2",                           "attn2",         "attention₂\n(Δ delta, 10k-d)",                "squeeze", "output"),
    ("tangling.neuron_proj_down",            "tangle_n_down", "tangle·neuron ↓\n(10k→128)",                  "squeeze", "output"),
    ("tangling.neuron_proj_up",              "tangle_n_up",   "tangle·neuron ↑\n(128→10k)",                  "squeeze", "output"),
    ("tangling.neuron_out_norm",             "tangle_n",      "tangle neuron\n(Δ delta, 10k-d)",             "squeeze", "output"),
    ("scratchpad",                           "pre_scratch",   "pre-scratch\n(INN+attn₂+tangle, 10k-d)",      "squeeze", "input"),
    ("scratchpad.project_down",              "scratch_down",  "scratch ↓\n(10k→512)",                        "squeeze", "output"),
    ("scratchpad.project_up",                "scratch_up",    "scratch ↑\n(512→10k)",                        "squeeze", "output"),
    ("scratchpad",                           "scratch",       "scratchpad\n(Δ delta, 10k-d)",                "squeeze", "output"),
    # ── Memory stage + tangling (memory stage) ────────────────────────────────
    ("memory.inputReducer",                  "mem1_reduce",   "mem₁ reduce\n(10k→1024)",                     "squeeze", "output"),
    ("memory.memoryInfluence2.0",            "mem1_inf_down", "mem₁ inf ↓\n(1024→512)",                      "squeeze", "output"),
    ("memory.memoryInfluence2.3",            "mem1_inf_up",   "mem₁ inf ↑\n(512→10k)",                       "squeeze", "output"),
    ("memory.output_norm",                   "mem1",          "memory₁\n(output_norm proxy, 10k-d)",         "squeeze", "output"),
    ("tangling.memory_proj_down",            "tangle_m_down", "tangle·memory ↓\n(10k→128)",                  "squeeze", "output"),
    ("tangling.memory_proj_up",              "tangle_m_up",   "tangle·memory ↑\n(128→10k)",                  "squeeze", "output"),
    ("tangling.memory_out_norm",             "tangle_m",      "tangle memory\n(Δ delta, 10k-d)",             "squeeze", "output"),
    ("memory2.inputReducer",                 "mem2_reduce",   "mem₂ reduce\n(10k→1024)",                     "squeeze", "output"),
    ("memory2.memoryInfluence2.0",           "mem2_inf_down", "mem₂ inf ↓\n(1024→512)",                      "squeeze", "output"),
    ("memory2.memoryInfluence2.3",           "mem2_inf_up",   "mem₂ inf ↑\n(512→10k)",                       "squeeze", "output"),
    ("memory2.output_norm",                  "mem2",          "memory₂\n(output_norm proxy, 10k-d)",         "squeeze", "output"),
    # ── Logits ────────────────────────────────────────────────────────────────
    ("logits.logitNorm",                     "logits",        "logits\n(logitNorm, vocab-d)",                 "squeeze", "output"),
]

# Active layer defs — respects LAYERS toggles
LAYER_DEFS = [ld for ld in _ALL_LAYER_DEFS if LAYERS.get(ld[1], False)]

# ─────────────────────────────────────────────────────────────────────────────


from utils.diagnostic_model import build_model


def get_submodule(model, attr_path):
    return reduce(getattr, attr_path.split("."), model)


def categorize_token(tok):
    """Assign a display category to a vocab token string."""
    if tok.startswith("<") or tok in ("<UNK>",):
        return "special"
    if tok.startswith("Ġ") or tok.startswith(" "):
        inner = tok.lstrip("Ġ").lstrip()
        if len(inner) == 0:
            return "punct"
        if inner.isalpha():
            return "bpe_word"
        if inner.isdigit():
            return "bpe_word"
        return "bpe_word"
    if len(tok) == 1:
        if tok.isalpha():
            return "letter"
        if tok.isdigit():
            return "digit"
        return "punct"
    if tok.isalpha():
        return "bpe_sub"
    return "punct"


CATEGORY_COLORS = {
    "letter":   "#4fc3f7",   # light blue
    "digit":    "#81c784",   # light green
    "punct":    "#ff8a65",   # orange
    "bpe_word": "#ce93d8",   # purple
    "bpe_sub":  "#f48fb1",   # pink
    "special":  "#78909c",   # grey
    "labeled":  "#ffeb3b",   # yellow — overrides category colour for labeled tokens
}


def sample_tokens(librarian, n_total, label_words, neutral_tok, seed=42):
    """
    Build the probe token list:
    - Always include ALL single-char tokens (letters, digits, common punct).
    - Look up LABEL_WORDS: find the best single-token match in the vocab.
    - Fill remainder up to n_total with stratified random BPE samples.
    Returns list of (token_str, token_id, category, is_labeled, display_label).
    """
    rng = random.Random(seed)
    unk_id = librarian.tokenToIndex.get("<UNK>", 0)

    # Collect every vocab token with its id and category
    all_tokens = []  # (tok_str, tok_id, category)
    for tok_str, tok_id in librarian.tokenToIndex.items():
        if tok_str == "<UNK>":
            continue
        if tok_str == neutral_tok:
            continue  # exclude the filler itself
        cat = categorize_token(tok_str)
        all_tokens.append((tok_str, tok_id, cat))

    # Always include all base single-char tokens
    base_chars = [(ts, ti, c) for ts, ti, c in all_tokens if c in ("letter", "digit", "punct")]

    # Find labeled words — try exact, then space-prefixed, then first subtoken
    labeled_set = set()
    label_entries = []  # (tok_str, tok_id, category, display_label)
    for word in label_words:
        word_lc = word.lower()
        candidates = [word_lc, "Ġ" + word_lc, " " + word_lc]
        found = None
        for cand in candidates:
            if cand in librarian.tokenToIndex:
                found = (cand, librarian.tokenToIndex[cand], categorize_token(cand), word_lc)
                break
        if found is None:
            subtoks = librarian.tokenizeText(word_lc)
            if subtoks:
                st  = subtoks[0]
                tid = librarian.tokenToIndex.get(st, unk_id)
                if tid != unk_id:
                    found = (st, tid, categorize_token(st), f"{word_lc}({st})")
        if found:
            if found[1] not in labeled_set:
                labeled_set.add(found[1])
                label_entries.append(found)

    # BPE tokens not already included
    bpe_pool = [(ts, ti, c) for ts, ti, c in all_tokens
                if c in ("bpe_word", "bpe_sub") and ti not in labeled_set
                and ti not in {t[1] for t in base_chars}]

    # Stratified sample from BPE pool
    n_base    = len(base_chars)
    n_labeled = len(label_entries)
    n_bpe     = max(0, n_total - n_base - n_labeled)

    bpe_word_pool = [x for x in bpe_pool if x[2] == "bpe_word"]
    bpe_sub_pool  = [x for x in bpe_pool if x[2] == "bpe_sub"]
    n_bpe_word = min(len(bpe_word_pool), int(n_bpe * 0.7))
    n_bpe_sub  = min(len(bpe_sub_pool),  n_bpe - n_bpe_word)

    sampled_bpe = rng.sample(bpe_word_pool, n_bpe_word) + rng.sample(bpe_sub_pool, n_bpe_sub)

    # Assemble final list
    entries   = []
    seen_ids  = set()
    for ts, ti, c in base_chars + sampled_bpe:
        if ti not in seen_ids:
            seen_ids.add(ti)
            entries.append({"tok": ts, "id": ti, "cat": c, "labeled": False, "label": None})
    for ts, ti, c, lbl in label_entries:
        if ti not in seen_ids:
            seen_ids.add(ti)
        for e in entries:
            if e["id"] == ti:
                e["labeled"] = True
                e["label"]   = lbl
                break
        else:
            entries.append({"tok": ts, "id": ti, "cat": c, "labeled": True, "label": lbl})

    print(f"  probe set: {len(entries)} tokens "
          f"({n_base} base-chars, {len(label_entries)} labeled words, "
          f"{len(sampled_bpe)} BPE samples)")
    return entries


# ── input builders ────────────────────────────────────────────────────────────

def build_neutral_input(probe_id, neutral_id, window_max, device):
    """[neutral × (window_max-1)] + [probe_token].  Only last position varies."""
    ids = [neutral_id] * (window_max - 1) + [probe_id]
    return torch.tensor(ids, dtype=torch.long, device=device)


def build_tiled_input(probe_id, window_max, device):
    """[probe_token × window_max].  Entire context = same repeated token."""
    ids = [probe_id] * window_max
    return torch.tensor(ids, dtype=torch.long, device=device)


# ── vector extraction ─────────────────────────────────────────────────────────

def extract_vec(raw_output, extractor_key):
    if isinstance(raw_output, (tuple, list)):
        raw_output = raw_output[0]
    t = raw_output.detach().cpu().float()
    if t.dim() == 0:
        return t.unsqueeze(0).numpy()
    if extractor_key == "last":
        if t.dim() == 1:
            return t.numpy()
        while t.dim() > 2:
            t = t.squeeze(0)
        return t[-1].numpy()
    while t.dim() > 1 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.dim() > 1:
        t = t[-1]
    return t.numpy()


def snapshot_memory(model):
    """
    Save a copy of bby's current memory buffers.
    We restore from this before each probe token so probes don't bleed into
    each other — but we NEVER zero the buffers, that would erase learned state.
    """
    snap = {}
    for attr in ("memory", "memory2"):
        m = getattr(model, attr, None)
        if m is None:
            continue
        snap[attr] = {}
        for buf in ("shortTermMemory", "longTermMemory"):
            if hasattr(m, buf):
                snap[attr][buf] = getattr(m, buf).clone()
    return snap


def restore_memory(model, snap):
    """Restore memory buffers to the snapshotted state (not zero)."""
    for attr, bufs in snap.items():
        m = getattr(model, attr, None)
        if m is None:
            continue
        for buf, val in bufs.items():
            if hasattr(m, buf):
                getattr(m, buf).copy_(val)


# ── forward-pass probe ────────────────────────────────────────────────────────

@torch.no_grad()
def probe_tokens(model, token_entries, neutral_id, window_max, layer_defs, strategy="neutral"):
    """
    Run one forward pass per token, collect activations for every hooked layer.

    strategy: "neutral"  — [neutral × (window_max-1)] + [probe_token]
              "tiled"    — [probe_token × window_max]
    """
    captured = {}
    hooks    = []
    device   = next(model.parameters()).device

    for attr_path, short_label, _, extractor_key, hook_src in layer_defs:
        try:
            module = get_submodule(model, attr_path)
        except AttributeError:
            print(f"  [SKIP] {attr_path} not found")
            continue

        def make_hook(label, ekey, src):
            def _hook(mod, inp, out):
                raw = inp[0] if src == "input" else out
                captured[label] = extract_vec(raw, ekey)
            return _hook

        h = module.register_forward_hook(make_hook(short_label, extractor_key, hook_src))
        hooks.append(h)

    layer_shorts = [sl for _, sl, _, _, _ in layer_defs]
    results = {sl: [] for sl in layer_shorts}

    # Snapshot memory ONCE before the loop.  Restored before every token so
    # probes don't bleed into each other, but his learned state is preserved.
    mem_snap = snapshot_memory(model)

    n = len(token_entries)
    for i, entry in enumerate(token_entries):
        captured.clear()
        restore_memory(model, mem_snap)

        if strategy == "tiled":
            tok_ids = build_tiled_input(entry["id"], window_max, device)
        else:
            tok_ids = build_neutral_input(entry["id"], neutral_id, window_max, device)

        try:
            model.forward(_inputSeq=tok_ids, _pixel=None, _use_lock=False)
        except Exception as e:
            print(f"  [WARN] forward failed for '{entry['tok']}': {e}")

        for sl in layer_shorts:
            if sl in captured:
                results[sl].append(captured[sl].copy())
            else:
                ref = next((v for v in results[sl] if len(v) > 0), None)
                results[sl].append(np.zeros(len(ref) if ref else 1))

        if (i + 1) % 50 == 0 or i == n - 1:
            norms = {sl: f"{np.linalg.norm(results[sl][-1]):.1f}" for sl in layer_shorts}
            print(f"  [{i+1}/{n}] '{entry['tok']}'  " +
                  "  ".join(f"{k}={v}" for k, v in norms.items()))

    for h in hooks:
        h.remove()

    return {k: np.stack(v, axis=0) for k, v in results.items()}


# ── dimensionality reduction ──────────────────────────────────────────────────

def reduce_to_2d(embeddings, label):
    from sklearn.decomposition import PCA

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 1)

    n, d = embeddings.shape
    if n < 2:
        return np.zeros((n, 2))
    if d < 2:
        print(f"  [{label}] d={d} degenerate, padding")
        return np.pad(embeddings, ((0, 0), (0, 2 - d)))

    n_pca = max(2, min(50, n - 1, d))
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_pca, random_state=42)
        red = pca.fit_transform(embeddings)
        var = pca.explained_variance_ratio_.sum()
        print(f"  [{label}] PCA({d}→{n_pca}): {var*100:.1f}% var  ", end="")
    except Exception as e:
        print(f"  [{label}] PCA failed: {e}")
        return np.zeros((n, 2))

    if n >= 4 and n_pca >= 2:
        try:
            from sklearn.manifold import TSNE
            perp   = min(max(2, n // 5), 40)
            result = TSNE(n_components=2, perplexity=perp, n_iter=2000,
                          random_state=42, init="pca",
                          learning_rate="auto").fit_transform(red)
            print(f"t-SNE(perp={perp})")
            return result
        except Exception as e:
            print(f"t-SNE failed ({e}), PCA(2)")

    n_fin = min(2, n - 1, d)
    if n_fin < 2:
        raw = PCA(n_components=n_fin, random_state=42).fit_transform(embeddings)
        return np.pad(raw, ((0, 0), (0, 2 - n_fin)))
    print("PCA(2)")
    return PCA(n_components=2, random_state=42).fit_transform(embeddings)


def reduce_comparison_to_2d(neutral_arr, tiled_arr, label):
    """
    Project neutral + tiled together into the SAME 2D space so we can draw
    arrows from each token's neutral position to its tiled position.
    Returns (neutral_coords_2d, tiled_coords_2d) both shape (n, 2).
    """
    n = len(neutral_arr)
    combined = np.vstack([neutral_arr, tiled_arr])
    combined_2d = reduce_to_2d(combined, f"{label}_combined")
    return combined_2d[:n], combined_2d[n:]


# ── analysis helpers ──────────────────────────────────────────────────────────

def cosine_matrix_print(embeddings, labels, layer_name):
    from scipy.spatial.distance import cdist
    norms  = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-9)
    cos    = 1.0 - cdist(normed, normed, metric="cosine")

    print(f"\n── Cosine Similarity · {layer_name} (labeled words only) ──")
    cw = max(len(l) for l in labels) + 2
    print(f"{'':>{cw}}" + "".join(f"{l[:9]:>10}" for l in labels))
    for i, li in enumerate(labels):
        print(f"{li[:cw]:>{cw}}" + "".join(f"{cos[i,j]:>10.3f}" for j in range(len(labels))))
    off = cos[np.triu_indices(len(labels), k=1)]
    print(f"\n  off-diag:  mean={off.mean():.4f}  std={off.std():.4f}  "
          f"min={off.min():.4f}  max={off.max():.4f}")
    if off.std() < 0.005:
        print("  ⚠️  COLLAPSE — all identical (std < 0.005)")
    elif off.mean() > 0.97:
        print("  ⚠️  HIGH MEAN — very little spread")
    else:
        print("  ✓  Healthy spread")
    return cos


def layer_correlation_report(layer_data, layer_defs):
    """
    For each pair of layers, compute the Pearson correlation of their
    pairwise cosine distance matrices.  Near 1.0 = 'mirrors'.
    """
    from scipy.spatial.distance import pdist
    from scipy.stats import pearsonr

    shorts = [sl for _, sl, _, _, _ in layer_defs if sl in layer_data]
    n = next((layer_data[sl].shape[0] for sl in shorts if sl in layer_data), 0)
    n_pairs = n * (n - 1) // 2
    print(f"\n── Layer-pair distance correlation (1.0 = mirror) ──")
    print(f"  {len(shorts)} layers × {n} tokens → {n_pairs:,} pairs each  (may be slow)")

    dist_vecs = {}
    for i, sl in enumerate(shorts):
        arr = layer_data[sl]
        if arr.std() < 1e-9:
            dist_vecs[sl] = None
            print(f"  [{i+1}/{len(shorts)}] {sl:>14}  COLLAPSED")
        else:
            norms  = np.linalg.norm(arr, axis=1, keepdims=True)
            normed = arr / (norms + 1e-9)
            dist_vecs[sl] = pdist(normed, metric="cosine")
            print(f"  [{i+1}/{len(shorts)}] {sl:>14}  pdist done")

    cw = max(len(s) for s in shorts) + 1
    print(f"\n  computing {len(shorts)**2} Pearson correlations …")
    print(f"{'':>{cw}}" + "".join(f"{s:>10}" for s in shorts))
    for i, sa in enumerate(shorts):
        row = f"{sa:>{cw}}"
        for sb in shorts:
            if dist_vecs[sa] is None or dist_vecs[sb] is None:
                row += f"{'--':>10}"
            else:
                r, _ = pearsonr(dist_vecs[sa], dist_vecs[sb])
                row += f"{r:>10.3f}"
        print(row)
        if (i + 1) % 5 == 0:
            print(f"  … {i+1}/{len(shorts)} rows done")


# ── scatter panels ────────────────────────────────────────────────────────────

def scatter_panel(ax, coords, entries, title, show_legend=False):
    """
    Draw one t-SNE panel.  Labeled tokens get yellow outlines + text.
    Base-char tokens always get their character as a tiny label.
    """
    ax.set_facecolor("#0e0e0e")
    x, y = coords[:, 0], coords[:, 1]

    for labeled_pass in (False, True):
        for i, e in enumerate(entries):
            if e["labeled"] != labeled_pass:
                continue
            col = CATEGORY_COLORS["labeled"] if e["labeled"] else CATEGORY_COLORS[e["cat"]]
            ec  = "white" if e["labeled"] else "#444444"
            lw  = 0.8     if e["labeled"] else 0.2
            sz  = 90      if e["labeled"] else 30
            ax.scatter(x[i], y[i], color=col, s=sz, zorder=3 + int(e["labeled"]),
                       edgecolors=ec, linewidths=lw, alpha=0.9)

            lbl = e["label"] if e["labeled"] else None
            if lbl is None and e["cat"] in ("letter", "digit"):
                lbl = e["tok"]

            if lbl:
                ax.annotate(lbl, xy=(x[i], y[i]), xytext=(4, 2),
                            textcoords="offset points",
                            fontsize=7 if e["labeled"] else 5.5,
                            color="#ffffcc" if e["labeled"] else "#aaaaaa",
                            fontfamily="monospace",
                            fontweight="bold" if e["labeled"] else "normal",
                            zorder=6)

    ax.set_title(title, color="white", fontsize=9, pad=6)
    ax.tick_params(colors="#444444", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    if show_legend:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=col, markersize=6, label=cat)
                   for cat, col in CATEGORY_COLORS.items() if cat != "labeled"]
        handles.append(Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=CATEGORY_COLORS["labeled"],
                               markersize=8, label="labeled word"))
        ax.legend(handles=handles, fontsize=6, framealpha=0.3,
                  loc="lower right", facecolor="#111111", labelcolor="white")


def scatter_diff_panel(ax, neutral_coords, tiled_coords, entries, title):
    """
    Joint-projection diff panel.
    Circles = neutral position.  Diamonds = tiled position.
    Arrows connect them; colour = displacement magnitude (L2 in 2D space).
    """
    ax.set_facecolor("#0e0e0e")

    # displacement magnitude (in the 2D projected space)
    diff_vec  = tiled_coords - neutral_coords          # (n, 2)
    disp_norm = np.linalg.norm(diff_vec, axis=1)       # (n,)
    max_disp  = disp_norm.max() + 1e-9
    cmap      = plt.get_cmap("hot")

    for i, e in enumerate(entries):
        nx, ny = neutral_coords[i]
        tx, ty = tiled_coords[i]
        mag    = disp_norm[i] / max_disp
        color  = cmap(0.15 + 0.85 * mag)              # 0.15 floor so arrows aren't black

        # arrow neutral → tiled
        ax.annotate("", xy=(tx, ty), xytext=(nx, ny),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=0.6 + mag * 1.2,
                                   mutation_scale=6),
                    zorder=3)

        # neutral dot
        ax.scatter(nx, ny, color=color, s=18, zorder=4,
                   edgecolors="#666666", linewidths=0.15, alpha=0.85)

        # label if labeled or base-char
        lbl = e["label"] if e["labeled"] else (e["tok"] if e["cat"] in ("letter", "digit") else None)
        if lbl:
            ax.annotate(lbl, xy=(nx, ny), xytext=(4, 2),
                        textcoords="offset points",
                        fontsize=7 if e["labeled"] else 5,
                        color="#ffffcc" if e["labeled"] else "#888888",
                        fontfamily="monospace",
                        fontweight="bold" if e["labeled"] else "normal",
                        zorder=7)

    # colourbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=max_disp))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("displacement", color="white", fontsize=6)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=5)

    ax.set_title(title, color="white", fontsize=9, pad=6)
    ax.tick_params(colors="#444444", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


def cosine_heatmap_panel(ax, cos_matrix, labels, title=""):
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    im   = ax.imshow(cos_matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7, color="white")
    ax.set_yticklabels(labels, fontsize=7, color="white")
    ax.set_title(title, color="white", fontsize=8, pad=6)
    ax.set_facecolor("#0e0e0e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cos_matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=5, color="white" if abs(cos_matrix[i,j]) < 0.65 else "#111111")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03).ax.yaxis.set_tick_params(
        color="white", labelcolor="white", labelsize=6)


# ── figure builder ────────────────────────────────────────────────────────────

def build_strategy_figure(neutral_data, tiled_data, layer_defs, entries,
                          labeled_indices, labeled_labels, window_max):
    """
    Layout (all in one figure):

      ╔═ section header ══════════════════════════════╗
      ║  neutral prefix  │ layer0 │ layer1 │ ...       ║  ← wraps if > MAX_COLS
      ╠═ section header ══════════════════════════════╣
      ║  tiled / looped  │ layer0 │ layer1 │ ...       ║
      ╠═ section header ══════════════════════════════╣
      ║  diff (arrows)   │ layer0 │ layer1 │ ...       ║
      ╠═══════════════════════════════════════════════╣
      ║  cosine heatmap  │ neutral │ tiled  (refine2)  ║
      ╚═══════════════════════════════════════════════╝

    When n_layers > MAX_COLS each section wraps into multiple sub-rows.
    Toggle layers via the LAYERS dict at the top of the file.
    """
    import math
    from scipy.spatial.distance import cdist

    long_labels  = {sl: ll for _, sl, ll, _, _ in layer_defs}
    layer_shorts = [sl for _, sl, _, _, _ in layer_defs
                    if sl in neutral_data and sl in tiled_data]

    n_layers  = len(layer_shorts)
    n_cols    = min(n_layers, MAX_COLS)
    n_subcols = max(1, n_cols)                           # grid columns
    n_subrows = math.ceil(n_layers / n_subcols)          # sub-rows per section

    # ── cosine matrices (terminal) ─────────────────────────────────────────
    for strat_label, data in [("neutral", neutral_data), ("tiled", tiled_data)]:
        if "refine2" in data and labeled_indices:
            labeled_vecs = data["refine2"][labeled_indices]
            cosine_matrix_print(labeled_vecs, labeled_labels,
                                f"refine2 [{strat_label}]")

    # ── compute 2d coords ─────────────────────────────────────────────────
    print(f"\n  dim-reduction — neutral ...")
    neutral_2d = {sl: reduce_to_2d(neutral_data[sl], f"{sl}_N") for sl in layer_shorts}
    print(f"\n  dim-reduction — tiled ...")
    tiled_2d   = {sl: reduce_to_2d(tiled_data[sl],   f"{sl}_T") for sl in layer_shorts}
    print(f"\n  dim-reduction — joint diff ...")
    diff_neutral_2d, diff_tiled_2d = {}, {}
    for sl in layer_shorts:
        n2d, t2d = reduce_comparison_to_2d(neutral_data[sl], tiled_data[sl], sl)
        diff_neutral_2d[sl] = n2d
        diff_tiled_2d[sl]   = t2d

    # ── heatmap data (refine2 cosine, labeled words) ───────────────────────
    cos_neutral = cos_tiled = None
    if "refine2" in neutral_data and labeled_indices:
        def _cos(arr):
            norms  = np.linalg.norm(arr, axis=1, keepdims=True)
            normed = arr / (norms + 1e-9)
            return 1.0 - cdist(normed, normed, metric="cosine")
        cos_neutral = _cos(neutral_data["refine2"][labeled_indices])
        cos_tiled   = _cos(tiled_data["refine2"][labeled_indices])

    has_heatmap = cos_neutral is not None and len(labeled_labels) > 1

    # ── grid geometry ─────────────────────────────────────────────────────
    # 3 sections (neutral / tiled / diff), each n_subrows tall.
    # Optional heatmap section at the bottom (1 sub-row, split into 2 halves).
    SECTION_ROWS = n_subrows                   # rows per section
    N_SECTIONS   = 3                           # neutral, tiled, diff
    n_scatter_rows = N_SECTIONS * SECTION_ROWS
    n_heat_rows    = 1 if has_heatmap else 0
    n_total_rows   = n_scatter_rows + n_heat_rows

    panel_w = 6.2
    panel_h = 5.5
    fig = plt.figure(
        figsize=(n_subcols * panel_w, n_total_rows * panel_h),
        facecolor="#0d0d0d",
    )
    outer = gridspec.GridSpec(n_total_rows, n_subcols, figure=fig,
                              hspace=0.50, wspace=0.10)

    # ── draw section headers as text in the figure margin ─────────────────
    section_names = ["neutral prefix", "tiled / looped", "diff  (○ neutral → tiled)"]
    section_colors = ["#4fc3f7", "#ce93d8", "#ffb74d"]

    def _section_row_offset(sec_idx):
        return sec_idx * SECTION_ROWS

    for sec_idx, (sec_name, sec_col) in enumerate(zip(section_names, section_colors)):
        for sub_row in range(SECTION_ROWS):
            global_row = _section_row_offset(sec_idx) + sub_row

            # which layers appear in this sub-row?
            start_l = sub_row * n_subcols
            end_l   = min(start_l + n_subcols, n_layers)

            for col_idx, layer_idx in enumerate(range(start_l, end_l)):
                sl  = layer_shorts[layer_idx]
                ax  = fig.add_subplot(outer[global_row, col_idx])

                # prepend section label only on first column of first sub-row
                prefix = f"[{sec_name}]\n" if (sub_row == 0 and col_idx == 0) else ""
                title  = f"{prefix}{long_labels.get(sl, sl)}"

                if sec_idx == 0:
                    scatter_panel(ax, neutral_2d[sl], entries, title,
                                  show_legend=(layer_idx == 0))
                elif sec_idx == 1:
                    scatter_panel(ax, tiled_2d[sl], entries, title)
                else:
                    scatter_diff_panel(ax, diff_neutral_2d[sl], diff_tiled_2d[sl],
                                       entries, title)

                # colour the title by section
                ax.title.set_color(sec_col)

            # blank out unused columns in this sub-row
            for empty_col in range(end_l - start_l, n_subcols):
                ax_empty = fig.add_subplot(outer[global_row, empty_col])
                ax_empty.set_visible(False)

    # ── heatmap row ────────────────────────────────────────────────────────
    if has_heatmap:
        mid = max(1, n_subcols // 2)
        ax_n = fig.add_subplot(outer[n_scatter_rows, :mid])
        ax_t = fig.add_subplot(outer[n_scatter_rows, mid:])
        cosine_heatmap_panel(ax_n, cos_neutral, labeled_labels,
                             title="cosine sim · refine2 · neutral prefix")
        cosine_heatmap_panel(ax_t, cos_tiled, labeled_labels,
                             title="cosine sim · refine2 · tiled")

    active = [k for k, v in LAYERS.items() if v]
    fig.suptitle(
        f"babyLLM · latent space probe  |  window = {window_max}  |  "
        f"n = {len(entries)} tokens  |  layers: {', '.join(active)}",
        color="white", fontsize=9, y=1.002,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    active_layers = [k for k, v in LAYERS.items() if v]
    print("=" * 72)
    print("  babyLLM · latent space probe  (multi-strategy + multi-window)")
    print("  strategies : neutral-prefix  +  tiled/looped")
    print(f"  windows    : {WINDOW_SIZES}")
    print(f"  layers     : {active_layers}")
    print(f"  n tokens   : {N_VOCAB_SAMPLE}")
    print("=" * 72)

    # Use the largest window to load the model once; reload for smaller windows.
    for w_idx, window_max in enumerate(WINDOW_SIZES):
        print(f"\n{'━'*72}")
        print(f"  WINDOW SIZE = {window_max} tokens  ({w_idx+1}/{len(WINDOW_SIZES)})")
        print(f"{'━'*72}")

        print("\n[1] loading model ...")
        model, librarian = build_model(window_max)

        neutral_id = librarian.tokenToIndex.get(NEUTRAL_FILL, 0)
        print(f"  neutral fill token: '{NEUTRAL_FILL}' → id {neutral_id}")

        print(f"\n[2] sampling {N_VOCAB_SAMPLE} probe tokens ...")
        entries = sample_tokens(librarian, N_VOCAB_SAMPLE, LABEL_WORDS, NEUTRAL_FILL, RANDOM_SEED)
        labeled_indices = [i for i, e in enumerate(entries) if e["labeled"]]
        labeled_labels  = [e["label"] for e in entries if e["labeled"]]

        print(f"\n[3] probing — neutral-prefix strategy ...")
        neutral_data = probe_tokens(
            model, entries, neutral_id, window_max, LAYER_DEFS, strategy="neutral"
        )
        print(f"  layers: {list(neutral_data.keys())}")

        print(f"\n[4] probing — tiled/looped strategy ...")
        tiled_data = probe_tokens(
            model, entries, neutral_id, window_max, LAYER_DEFS, strategy="tiled"
        )
        print(f"  layers: {list(tiled_data.keys())}")

        print(f"\n[5] layer correlation report — neutral ...")
        layer_correlation_report(neutral_data, LAYER_DEFS)

        print(f"\n[5b] layer correlation report — tiled ...")
        layer_correlation_report(tiled_data, LAYER_DEFS)

        print(f"\n[6] building figure ...")
        fig = build_strategy_figure(
            neutral_data, tiled_data, LAYER_DEFS, entries,
            labeled_indices, labeled_labels, window_max,
        )

        out_path = OUTPUT_PNG_TEMPLATE.format(w=window_max)
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  saved → {os.path.abspath(out_path)}")

        # free model memory before next iteration
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n" + "=" * 72)
    print("  done.  output files:")
    for window_max in WINDOW_SIZES:
        p = os.path.abspath(OUTPUT_PNG_TEMPLATE.format(w=window_max))
        print(f"    {p}")

    print("\nReading the plots:")
    print("  Row A (neutral) — each token probed in isolation via a neutral context.")
    print("  Row B (tiled)   — entire window filled with the same token repeated.")
    print("  Row C (diff)    — joint projection.  arrows: neutral→tiled per token.")
    print("                    hot arrow = strategy-sensitive token.")
    print("                    cold/short arrow = strategy-stable token.")
    print("  Bottom heatmaps — cosine similarity of labeled words at refine2.")


if __name__ == "__main__":
    main()
