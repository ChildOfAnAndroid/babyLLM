#!/usr/bin/env python3
# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm sequence probe -*- ⊂ʕʘ‿ʘ૮ʔ ---
# utils/latent_probe_sequence.py
#
# Traces a real text prompt through bbyLLM token-by-token and captures
# activations at every probe point.
#
# Unlike latent_probe.py (vocab atlas — each token probed in isolation),
# here memory is LIVE and accumulates naturally between steps, so the INN
# window activations, scratchpad, and STM/LTM all reflect real sequential
# processing.  Each "token" in the output = one step in the forward pass.
#
# Optionally continues generating past the prompt so you can watch bby
# decide what to say next.
#
# Output: .probe_cache/probe_seq_{name}.npz
#         Same format as the vocab probe → load directly into the theatre.
#
# Usage (from project root):
#   python utils/latent_probe_sequence.py "hello charis how are you" --name hello
#   python utils/latent_probe_sequence.py "i like cats" --name cats --gen 30

import argparse
import json
import os
import sys
from functools import reduce

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CACHE_DIR = os.path.join(PROJECT_ROOT, ".probe_cache")

from babyLLM import BABYLLM
from config import modelDevice, learningRateGOAL
from school.staffroom.counsellor import COUNSELLOR
from school.staffroom.calligraphist import S_OUTPUT
from school.staffroom.librarian import LIBRARIAN
from school.staffroom.HE_IS_SCRIBE import SCRIBE
from utils.latent_probe import (
    _ALL_LAYER_DEFS,
    LAYERS,
    build_model,
    categorize_token,
    extract_vec,
    get_submodule,
)

LAYER_DEFS = [ld for ld in _ALL_LAYER_DEFS if LAYERS.get(ld[1], False)]

# Layers to apply logit-lens projection through the output weight matrix.
# Must be 10k-d (numNeurons-d) so the projection is meaningful.
LOGIT_LENS_LAYERS = ("inn_raw", "pre_scratch", "mem1", "mem2")
LOGIT_LENS_TOP_K  = 10


# ── helpers ───────────────────────────────────────────────────────────────────

def encode_prompt(librarian, text: str) -> list[int]:
    """Return a list of integer token IDs for the given text."""
    return list(librarian.tokenizer.encode(text))


def sample_next(logit_vec: np.ndarray, temperature: float = 0.9, top_k: int = 50) -> int:
    """Sample the next token id from a raw logit vector."""
    logits = torch.from_numpy(logit_vec).float()
    if top_k > 0:
        top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        threshold = top_vals[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
    return int(torch.multinomial(probs, 1).item())


def clear_scratchpad(model) -> None:
    sp = getattr(model, "scratchpad", None)
    if sp is not None and hasattr(sp, "clear_buffer"):
        sp.clear_buffer()


def zero_memory(model) -> None:
    """Set STM/LTM to zero — clean slate for a new conversation."""
    for attr in ("memory", "memory2"):
        m = getattr(model, attr, None)
        if m is None:
            continue
        for buf in ("shortTermMemory", "longTermMemory"):
            if hasattr(m, buf):
                getattr(m, buf).zero_()


def capture_blend_weights(model) -> np.ndarray:
    """Return the 4-element embed-blend softmax: [tok, pos, rgb, char]."""
    try:
        raw = torch.cat([model.inputBlend.detach(), model.charBlendWeight.detach()], dim=0)
        return F.softmax(raw, dim=0).cpu().numpy().astype(np.float32)
    except Exception:
        return np.zeros(4, dtype=np.float32)


def capture_cereb_weights(model):
    """Return (long_weights [n_long], short_weights [n_short], short_gate scalar)."""
    inn = getattr(model, "interneuronNetwork", None)
    try:
        long_w  = inn.cerebellumSoft.detach().cpu().numpy().astype(np.float32)
    except Exception:
        long_w  = np.zeros(9, dtype=np.float32)
    try:
        short_w = inn.cerebellumSoft_short.detach().cpu().numpy().astype(np.float32)
    except Exception:
        short_w = np.zeros(9, dtype=np.float32)
    try:
        gate    = float(inn.short_gate_used.item())
    except Exception:
        gate    = 0.5
    return long_w, short_w, gate


@torch.no_grad()
def compute_logit_lens(captured: dict, model, device, librarian,
                       top_k: int = LOGIT_LENS_TOP_K) -> dict:
    """
    For each layer in LOGIT_LENS_LAYERS (10k-d), project the captured activation
    through the output weight matrix to get proto-logit predictions.
    Also runs on 'logits' directly (vocab-d, just softmax).
    Returns {layer_key: {"ids": [int, ...], "probs": [float, ...], "tokens": [str, ...]}}
    """
    result = {}
    l_weights = model.logits.l_weights.detach()   # [10000, 4200]
    l_bias    = model.logits.l_bias.detach()       # [4200]
    act_norm  = model.logits.activationNorm        # LayerNorm(10000)

    for lk in LOGIT_LENS_LAYERS:
        if lk not in captured:
            continue
        try:
            act = torch.from_numpy(captured[lk]).float().to(device)
            scaled = act + act_norm(act)           # mirrors logits.forward
            proto  = scaled @ l_weights + l_bias   # [4200]
            probs  = F.softmax(proto, dim=-1)
            top_p, top_i = torch.topk(probs, min(top_k, probs.size(-1)))
            ids    = top_i.cpu().tolist()
            result[lk] = {
                "ids":    ids,
                "probs":  top_p.cpu().tolist(),
                "tokens": [librarian.indexToToken.get(i, "<UNK>") for i in ids],
            }
        except Exception:
            pass

    if "logits" in captured:
        try:
            lv    = torch.from_numpy(captured["logits"]).float()
            probs = F.softmax(lv, dim=-1)
            top_p, top_i = torch.topk(probs, min(top_k, probs.size(-1)))
            ids   = top_i.cpu().tolist()
            result["logits"] = {
                "ids":    ids,
                "probs":  top_p.cpu().tolist(),
                "tokens": [librarian.indexToToken.get(i, "<UNK>") for i in ids],
            }
        except Exception:
            pass

    return result


# ── main ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_sequence_probe(
    prompt: str,
    name: str,
    window_max: int = 269,
    n_gen: int = 0,
    temperature: float = 0.9,
    top_k: int = 50,
) -> str:
    """
    Run the sequence probe and save to .probe_cache/probe_seq_{name}.npz.
    Returns the path to the saved file.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"\n[seq-probe] loading model …")
    model, librarian = build_model(window_max)
    device = next(model.parameters()).device

    # ── encode prompt ────────────────────────────────────────────────────────
    prompt_ids = encode_prompt(librarian, prompt)
    if not prompt_ids:
        raise ValueError(f"Prompt encoded to zero tokens: {prompt!r}")
    print(f"[seq-probe] prompt: {prompt!r}")
    print(f"[seq-probe] {len(prompt_ids)} tokens: "
          f"{[librarian.indexToToken.get(i,'<UNK>') for i in prompt_ids]}")

    # ── register hooks ───────────────────────────────────────────────────────
    captured: dict = {}
    hooks = []
    for attr_path, short_label, _, extractor_key, hook_src in LAYER_DEFS:
        try:
            module = get_submodule(model, attr_path)
        except AttributeError:
            print(f"  [SKIP] {attr_path}")
            continue

        def make_hook(label, ekey, src):
            def _hook(mod, inp, out):
                raw = inp[0] if src == "input" else out
                captured[label] = extract_vec(raw, ekey)
            return _hook

        hooks.append(module.register_forward_hook(
            make_hook(short_label, extractor_key, hook_src)
        ))

    layer_shorts = [ld[1] for ld in LAYER_DEFS]
    results: dict[str, list] = {sl: [] for sl in layer_shorts}

    # ── clean slate ──────────────────────────────────────────────────────────
    # Memory and scratchpad start fresh so this sequence builds its own state.
    zero_memory(model)
    clear_scratchpad(model)

    # ── step through prompt + optional generation ─────────────────────────────
    all_ids: list[int] = list(prompt_ids)
    entries: list[dict] = []
    n_prompt = len(prompt_ids)

    # Extra capture accumulators
    blend_track:      list[np.ndarray] = []   # [step] → [4]
    cereb_long_track: list[np.ndarray] = []   # [step] → [n_long]
    cereb_short_track:list[np.ndarray] = []   # [step] → [n_short]
    cereb_gate_track: list[float]      = []   # [step] → scalar
    logit_lens_track: list[dict]       = []   # [step] → {layer_key: {ids, probs, tokens}}

    step = 0
    while step < len(all_ids):
        tok_id = all_ids[step]
        tok_str = librarian.indexToToken.get(tok_id, "<UNK>")
        is_generated = step >= n_prompt

        # Context window: up to window_max tokens ending at current step
        ctx_start = max(0, step + 1 - window_max)
        ctx_ids = all_ids[ctx_start : step + 1]
        input_tensor = torch.tensor(ctx_ids, dtype=torch.long, device=device)

        captured.clear()
        try:
            model.forward(_inputSeq=input_tensor, _pixel=None, _use_lock=False)
        except Exception as e:
            print(f"  [WARN] step {step} '{tok_str}': {e}")

        # ── collect layer activations ────────────────────────────────────────
        for sl in layer_shorts:
            if sl in captured:
                results[sl].append(captured[sl].copy())
            else:
                ref = next((v for v in results[sl] if len(v)), None)
                results[sl].append(np.zeros(len(ref) if ref is not None else 1))

        # ── capture blend weights ────────────────────────────────────────────
        blend_track.append(capture_blend_weights(model))

        # ── capture cerebellum window weights ────────────────────────────────
        cl, cs, cg = capture_cereb_weights(model)
        cereb_long_track.append(cl)
        cereb_short_track.append(cs)
        cereb_gate_track.append(cg)

        # ── logit lens projections ────────────────────────────────────────────
        logit_lens_track.append(
            compute_logit_lens(captured, model, device, librarian)
        )

        entries.append({
            "tok":     tok_str,
            "id":      tok_id,
            "cat":     categorize_token(tok_str),
            "step":    step,
            "labeled": True,
            "label":   f"[{step}] {tok_str}",
            "generated": is_generated,
        })

        # Optionally generate next token from this step's logits
        if step == len(all_ids) - 1 and len(all_ids) < n_prompt + n_gen:
            logit_vec = captured.get("logits")
            if logit_vec is not None:
                next_id = sample_next(logit_vec, temperature=temperature, top_k=top_k)
                all_ids.append(next_id)
                next_str = librarian.indexToToken.get(next_id, "<UNK>")
                print(f"  [gen] step {step+1}: {next_str!r}")

        if (step + 1) % 10 == 0 or step == len(all_ids) - 1:
            norms = {sl: f"{np.linalg.norm(results[sl][-1]):.1f}"
                     for sl in layer_shorts if results[sl]}
            ll_top = logit_lens_track[-1].get("logits", {}).get("tokens", [])
            print(f"  step {step+1}/{len(all_ids)}  '{tok_str}'"
                  + ("  [gen]" if is_generated else "")
                  + (f"  → top: {ll_top[:3]}" if ll_top else ""))
            print("    " + "  ".join(f"{k}={v}" for k, v in list(norms.items())[:6]))

        step += 1

    for h in hooks:
        h.remove()

    # ── save ─────────────────────────────────────────────────────────────────
    neutral_data = {k: np.stack(v, axis=0) for k, v in results.items() if v}
    probe_meta = {
        "kind":       "sequence",
        "prompt":     prompt,
        "name":       name,
        "window_max": window_max,
        "n_prompt":   n_prompt,
        "n_gen":      n_gen,
        "n_steps":    len(entries),
    }

    # Blend + cerebellum arrays
    blend_arr       = np.stack(blend_track,       axis=0)  # [n_steps, 4]
    cereb_long_arr  = np.stack(cereb_long_track,  axis=0)  # [n_steps, n_long]
    cereb_short_arr = np.stack(cereb_short_track, axis=0)  # [n_steps, n_short]
    cereb_gate_arr  = np.array(cereb_gate_track,  dtype=np.float32)  # [n_steps]

    out_path = os.path.join(CACHE_DIR, f"probe_seq_{name}.npz")
    np.savez_compressed(
        out_path,
        entries_json=np.array([json.dumps(entries, separators=(",", ":"),
                                ensure_ascii=False)], dtype=object),
        probe_meta_json=np.array([json.dumps(probe_meta, separators=(",", ":"),
                                  ensure_ascii=False)], dtype=object),
        logit_lens_json=np.array([json.dumps(logit_lens_track, separators=(",", ":"),
                                  ensure_ascii=False)], dtype=object),
        blend_weights=blend_arr,
        cereb_long_weights=cereb_long_arr,
        cereb_short_weights=cereb_short_arr,
        cereb_short_gate=cereb_gate_arr,
        **{f"neutral_{k}": v for k, v in neutral_data.items()},
    )

    print(f"\n[seq-probe] saved → {out_path}")
    print(f"            {len(entries)} steps × {len(neutral_data)} layers")
    print(f"            blend: {blend_arr.shape}  "
          f"cereb: {cereb_long_arr.shape}+{cereb_short_arr.shape}  "
          f"logit_lens: {len(LOGIT_LENS_LAYERS)+1} layers × {len(entries)} steps")

    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trace a text prompt through bbyLLM and save activations.")
    parser.add_argument("prompt",             help="Text to trace (quoted string)")
    parser.add_argument("--name",  default="seq",
                        help="Short name used in the output filename (default: seq)")
    parser.add_argument("--gen",   type=int, default=0,
                        help="Number of tokens to generate past the prompt (default: 0)")
    parser.add_argument("--temp",  type=float, default=0.9,
                        help="Sampling temperature for generation (default: 0.9)")
    parser.add_argument("--topk",  type=int, default=50,
                        help="Top-k for generation sampling (default: 50)")
    parser.add_argument("--window", type=int, default=269,
                        help="Context window size (default: 269)")
    args = parser.parse_args()

    run_sequence_probe(
        prompt=args.prompt,
        name=args.name,
        window_max=args.window,
        n_gen=args.gen,
        temperature=args.temp,
        top_k=args.topk,
    )
