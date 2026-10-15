#!/usr/bin/env python3
"""Inspect the current BabyLLM checkpoint for numerical instability.

This uses the canonical ``wakeup`` construction and checkpoint-loading path,
but does not train, save, or connect any chat service. Use
``--checkpoint-only`` for a cheap CPU-only attention-weight inspection.
"""

import argparse
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import (
    learningRateGOAL,
    modelFilePath,
    modelDevice,
    numTokensPerStepSTART,
    perfectionistPassRateSTART,
    trainingDataPairNumber,
    trainingDataStride,
    trainingLogFreq_A,
)


def diagnose_checkpoint_attention() -> bool:
    """Inspect attention tensors directly from the checkpoint on CPU."""
    checkpoint = torch.load(modelFilePath, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    attention = {
        name: tensor
        for name, tensor in state_dict.items()
        if isinstance(tensor, torch.Tensor)
        and ("attention." in name or "attention2." in name)
        and tensor.dtype != torch.bool
    }

    print(f"Loaded checkpoint: {modelFilePath}")
    print(f"Attention tensors: {len(attention)}")
    unstable = False
    for name, tensor in attention.items():
        has_nan = bool(torch.isnan(tensor).any().item())
        has_inf = bool(torch.isinf(tensor).any().item())
        norm = tensor.norm().item()
        maximum = tensor.abs().max().item() if tensor.numel() else 0.0
        rms = norm / math.sqrt(tensor.numel()) if tensor.numel() else 0.0
        unstable = unstable or has_nan or has_inf or norm > 1e6
        print(
            f"{name}: shape={tuple(tensor.shape)} norm={norm:.3e} "
            f"rms={rms:.3e} max={maximum:.3e} nan={has_nan} inf={has_inf}"
        )

    if not attention:
        print("No attention tensors found in the checkpoint.")
        return True
    print("EXPLOSION SIGNAL DETECTED" if unstable else "Attention weights look finite.")
    return unstable


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--checkpoint-only",
    action="store_true",
    help="Inspect attention weights on CPU without constructing BabyLLM.",
)
args = parser.parse_args()

if args.checkpoint_only:
    raise SystemExit(1 if diagnose_checkpoint_attention() else 0)

from wakeup import wakeup

print("=" * 80)
print("CURRENT CHECKPOINT STABILITY DIAGNOSTIC")
print("=" * 80)

print("\nLoading the current checkpoint through wakeup...")
model, tutor, librarian, scribe, calligraphist = wakeup(
    windowMAX=numTokensPerStepSTART,
    dataStride=trainingDataStride,
    passRateSTART=perfectionistPassRateSTART,
    lrGoal=learningRateGOAL,
    trainingDataPairNum=trainingDataPairNumber,
    log_A=trainingLogFreq_A,
    mode="mps_stress",
)

print("\n" + "=" * 80)
print("RUNNING DIAGNOSTICS")
print("=" * 80)

# ============================================================================
# DIAGNOSTIC 1: Check weight norms
# ============================================================================
print("\n1. Checking ATTENTION2 weight norms...")
explosion_detected = False
for name, param in model.attention2.attn.named_parameters():
    norm = param.norm().item()
    mean = param.mean().item()
    has_nan = torch.isnan(param).any().item()
    has_inf = torch.isinf(param).any().item()

    status = ""
    if has_nan:
        status += " ⚠️ NaN!"
        explosion_detected = True
    if has_inf:
        status += " ⚠️ Inf!"
        explosion_detected = True
    if norm > 1e6:
        status += " 🔥 EXPLODED!"
        explosion_detected = True

    print(f"   {name}: norm={norm:.2e}, mean={mean:.2e}{status}")

# ============================================================================
# DIAGNOSTIC 2: Check expansion weights
# ============================================================================
print("\n2. Checking TANGLING and SCRATCHPAD weights...")
print(f"   Tangling.embed_proj_up norm: {model.tangling.embed_proj_up.weight.norm().item():.2e}")
print(f"   Tangling.embed_proj_down norm: {model.tangling.embed_proj_down.weight.norm().item():.2e}")
print(f"   Tangling.embed_gate: {torch.sigmoid(model.tangling.embed_gate).item():.6f}")
print(f"   Tangling.memory_gate: {torch.sigmoid(model.tangling.memory_gate).item():.6f}")
print(
    f"   Scratchpad.write_strength: {torch.sigmoid(model.scratchpad.write_strength).item():.6f}"
)
print(
    f"   Scratchpad.erase_strength: {torch.sigmoid(model.scratchpad.erase_strength).item():.6f}"
)

# ============================================================================
# DIAGNOSTIC 3: Test attention2 on clean input
# ============================================================================
print("\n3. Testing ATTENTION2 on clean random input...")
test_10k = torch.randn(64, 10000, device=model.device) * 0.1  # Small random values
print(f"   Test input norm: {test_10k.norm().item():.2e}")

with torch.no_grad():
    try:
        out = model.attention2(test_10k)
        out_norm = out.norm().item()
        print(f"   Attention2 output norm: {out_norm:.2e}")

        if out_norm > 1e9:
            print("   🔥 EXPLOSION CONFIRMED!")
            print(f"      Contains NaN: {torch.isnan(out).any().item()}")
            print(f"      Contains Inf: {torch.isinf(out).any().item()}")
            print(f"      Max value: {out.max().item():.2e}")
            print(f"      Min value: {out.min().item():.2e}")
            explosion_detected = True
        else:
            print("   ✓ Output is reasonable")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        explosion_detected = True

# ============================================================================
# DIAGNOSTIC 4: Test with expansion disabled
# ============================================================================
print("\n4. Testing with EXPANSION DISABLED...")
with torch.no_grad():
    # Save original values
    embed_gate = model.tangling.embed_gate
    memory_gate = model.tangling.memory_gate
    orig_embed = embed_gate.clone()
    orig_memory = memory_gate.clone()
    orig_write = model.scratchpad.write_strength.clone()
    orig_erase = model.scratchpad.erase_strength.clone()

    # Disable expansion
    embed_gate.fill_(-100)  # sigmoid(-100) ≈ 0
    memory_gate.fill_(-100)
    model.scratchpad.write_strength.fill_(-100)
    model.scratchpad.erase_strength.fill_(-100)

    # Test forward pass
    test_input = torch.randint(0, len(librarian.vocabList), (64,), device=model.device)
    try:
        output = model(test_input)
        stats = model.getForwardStats()
        att2_norm = stats.get("4A_1_0_attnOut_norm", "N/A")
        mem_norm = stats.get("5M_memory_4M_x_FINAL_norm", "N/A")

        print(f"   Attention2 attnOut_norm: {att2_norm}")
        print(f"   Memory FINAL_norm: {mem_norm}")

        if isinstance(att2_norm, (int, float)) and att2_norm > 1e9:
            print("   🔥 ATTENTION2 EXPLODES EVEN WITHOUT EXPANSION!")
            print("   → This is a PRE-EXISTING issue, not caused by new modules")
            explosion_detected = True
        else:
            print("   ✓ Model is stable when expansion disabled")
            print("   → Explosion is CAUSED BY NEW MODULES")
    except Exception as e:
        print(f"   ❌ ERROR during forward pass: {e}")
        explosion_detected = True

    # Restore values
    embed_gate.copy_(orig_embed)
    memory_gate.copy_(orig_memory)
    model.scratchpad.write_strength.copy_(orig_write)
    model.scratchpad.erase_strength.copy_(orig_erase)

# ============================================================================
# DIAGNOSTIC 5: Test with only tangling enabled
# ============================================================================
print("\n5. Testing with ONLY TANGLING ENABLED...")
with torch.no_grad():
    # Disable scratchpad only
    model.scratchpad.write_strength.fill_(-100)
    model.scratchpad.erase_strength.fill_(-100)

    test_input = torch.randint(0, len(librarian.vocabList), (64,), device=model.device)
    try:
        output = model(test_input)
        stats = model.getForwardStats()
        att2_norm = stats.get("4A_1_0_attnOut_norm", "N/A")

        print(f"   Attention2 attnOut_norm: {att2_norm}")

        if isinstance(att2_norm, (int, float)) and att2_norm > 1e9:
            print("   🔥 TANGLING CAUSES EXPLOSION!")
        else:
            print("   ✓ Tangling alone is stable")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")

    # Restore
    model.scratchpad.write_strength.copy_(orig_write)
    model.scratchpad.erase_strength.copy_(orig_erase)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

if explosion_detected:
    print("⚠️  EXPLOSION DETECTED")
    print("\nRecommended actions:")
    print("1. Check if attention2 weights were already exploded before expansion")
    print("2. Review the diagnostic output above to identify the source")
    print("3. Consider reverting to a checkpoint before the explosion")
    print("4. Add gradient clipping if not already present")
else:
    print("✓ No explosion detected in diagnostics")
    print("\nIf you're seeing explosions in training:")
    print("1. The issue may be intermittent or input-dependent")
    print("2. Try running diagnostics during an actual training step")
    print("3. Check for NaN/Inf in gradients during training")

print("\n" + "=" * 80)
