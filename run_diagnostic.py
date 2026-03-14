#!/usr/bin/env python3
# Standalone diagnostic that loads the model like wakeup.py does

import torch
import sys
from babyLLM import BABYLLM
from school.staffroom.counsellor import COUNSELLOR
from school.staffroom.calligraphist import S_OUTPUT
from school.staffroom.librarian import LIBRARIAN
from school.staffroom.HE_IS_SCRIBE import SCRIBE
from config import *

print("=" * 80)
print("ATTENTION2 EXPLOSION DIAGNOSTIC")
print("=" * 80)

# Initialize components like wakeup.py does
print("\n[1/5] Initializing counsellor...")
counsellor = COUNSELLOR("diagnostic", _debug=False, _durations=False)

print("[2/5] Initializing librarian...")
librarian = LIBRARIAN(_counsellor=counsellor, _baseTokenizerPath=None, _forceRetrain=False)

print("[3/5] Initializing calligraphist...")
calligraphist = S_OUTPUT(_counsellor=counsellor)

print("[4/5] Initializing scribe...")
scribe = SCRIBE(_counsellor=counsellor, _calligraphist=calligraphist,
                _librarian=librarian, _numTokensPerStep=264)

print("[5/5] Loading babyLLM...")
model = BABYLLM(_counsellor=counsellor, _calligraphist=calligraphist,
                _scribe=scribe, _librarian=librarian,
                _device=modelDevice, _numTokensPerStep=264,
                _first=True, _learningRateGOAL=learningRateGOAL)

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
print("\n2. Checking EXPANSION module weights...")
print(f"   Tangling.project_up norm: {model.tangling.project_up.weight.norm().item():.2e}")
print(f"   Tangling.project_down norm: {model.tangling.project_down.weight.norm().item():.2e}")
print(f"   Tangling.embed_gate: {torch.sigmoid(model.tangling.embed_tangle_gate).item():.6f}")
print(f"   Tangling.memory_gate: {torch.sigmoid(model.tangling.memory_tangle_gate).item():.6f}")
print(f"   Scratchpad.write_strength: {torch.sigmoid(model.scratchpad.write_strength).item():.6f}")
print(f"   Scratchpad.erase_strength: {torch.sigmoid(model.scratchpad.erase_strength).item():.6f}")

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
            print(f"   🔥 EXPLOSION CONFIRMED!")
            print(f"      Contains NaN: {torch.isnan(out).any().item()}")
            print(f"      Contains Inf: {torch.isinf(out).any().item()}")
            print(f"      Max value: {out.max().item():.2e}")
            print(f"      Min value: {out.min().item():.2e}")
            explosion_detected = True
        else:
            print(f"   ✓ Output is reasonable")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        explosion_detected = True

# ============================================================================
# DIAGNOSTIC 4: Test with expansion disabled
# ============================================================================
print("\n4. Testing with EXPANSION DISABLED...")
with torch.no_grad():
    # Save original values
    orig_embed = model.tangling.embed_tangle_gate.clone()
    orig_memory = model.tangling.memory_tangle_gate.clone()
    orig_write = model.scratchpad.write_strength.clone()
    orig_erase = model.scratchpad.erase_strength.clone()

    # Disable expansion
    model.tangling.embed_tangle_gate.fill_(-100)  # sigmoid(-100) ≈ 0
    model.tangling.memory_tangle_gate.fill_(-100)
    model.scratchpad.write_strength.fill_(-100)
    model.scratchpad.erase_strength.fill_(-100)

    # Test forward pass
    test_input = torch.randint(0, len(librarian.vocabList), (64,), device=model.device)
    try:
        output = model(test_input)
        stats = model.getForwardStats()
        att2_norm = stats.get('4A_1_0_attnOut_norm', 'N/A')
        mem_norm = stats.get('5M_memory_4M_x_FINAL_norm', 'N/A')

        print(f"   Attention2 attnOut_norm: {att2_norm}")
        print(f"   Memory FINAL_norm: {mem_norm}")

        if isinstance(att2_norm, (int, float)) and att2_norm > 1e9:
            print(f"   🔥 ATTENTION2 EXPLODES EVEN WITHOUT EXPANSION!")
            print(f"   → This is a PRE-EXISTING issue, not caused by new modules")
            explosion_detected = True
        else:
            print(f"   ✓ Model is stable when expansion disabled")
            print(f"   → Explosion is CAUSED BY NEW MODULES")
    except Exception as e:
        print(f"   ❌ ERROR during forward pass: {e}")
        explosion_detected = True

    # Restore values
    model.tangling.embed_tangle_gate.copy_(orig_embed)
    model.tangling.memory_tangle_gate.copy_(orig_memory)
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
        att2_norm = stats.get('4A_1_0_attnOut_norm', 'N/A')

        print(f"   Attention2 attnOut_norm: {att2_norm}")

        if isinstance(att2_norm, (int, float)) and att2_norm > 1e9:
            print(f"   🔥 TANGLING CAUSES EXPLOSION!")
        else:
            print(f"   ✓ Tangling alone is stable")
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
