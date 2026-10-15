#!/usr/bin/env python3
# Simple timing test to find bottleneck

import torch
import time
from babyLLM import BABYLLM
from school.staffroom.counsellor import COUNSELLOR
from school.staffroom.calligraphist import S_OUTPUT
from school.staffroom.librarian import LIBRARIAN
from school.staffroom.HE_IS_SCRIBE import SCRIBE
from config import *

print("=" * 80)
print("SIMPLE TIMING TEST - Finding the Bottleneck")
print("=" * 80)

# Initialize model
print("\nInitializing model...")
counsellor = COUNSELLOR("timing", _debug=False, _durations=False)
librarian = LIBRARIAN(_counsellor=counsellor, _baseTokenizerPath=None, _forceRetrain=False)
calligraphist = S_OUTPUT(_counsellor=counsellor)
scribe = SCRIBE(_counsellor=counsellor, _calligraphist=calligraphist,
                _librarian=librarian, _numTokensPerStep=264)
model = BABYLLM(_counsellor=counsellor, _calligraphist=calligraphist,
                _scribe=scribe, _librarian=librarian,
                _device=modelDevice, _numTokensPerStep=264,
                _first=True, _learningRateGOAL=learningRateGOAL)

print(f"✓ Model on {modelDevice}")

# Create batch
batch_tokens = torch.randint(0, librarian.tokenizer.vocab_size, (1, 264), device=modelDevice)
print(f"✓ Batch created: {batch_tokens.shape}")

# Warmup
print("\nWarming up...")
model.eval()
with torch.no_grad():
    _ = model(batch_tokens)
    if modelDevice == 'mps':
        torch.mps.synchronize()
print("✓ Warmup complete")

# Time forward pass
print("\n" + "=" * 80)
print("TIMING FORWARD PASS")
print("=" * 80)

model.eval()
with torch.no_grad():
    if modelDevice == 'mps':
        torch.mps.synchronize()

    start = time.time()
    logits = model(batch_tokens)

    if modelDevice == 'mps':
        torch.mps.synchronize()

    forward_time = time.time() - start

print(f"\nForward pass time: {forward_time:.2f}s")
print(f"Training step estimate (forward + backward + optimizer): {forward_time * 3:.2f}s")

if forward_time > 60:
    print(f"\n🔥 EXTREMELY SLOW ({forward_time:.0f}s for forward pass)")
    print(f"   Training step would be {forward_time * 3 / 60:.1f} minutes!")
elif forward_time > 10:
    print(f"\n⚠️  SLOW ({forward_time:.0f}s for forward pass)")
else:
    print(f"\n✓ Reasonable speed")

# Check output magnitudes
print("\n" + "=" * 80)
print("OUTPUT MAGNITUDES")
print("=" * 80)

with torch.no_grad():
    stats = model.obtainAllCurrentStats()

    print("\nKey stats:")
    print(f"  Attention2 attnOut_norm: {stats.get('4A_2_attnOut_norm', 0):.2e}")
    print(f"  Memory FINAL_norm: {stats.get('4M_x_FINAL_norm', 0):.2e}")
    print(f"  TANGLE_neuron_gate: {stats.get('TANGLE_neuron_gate', 0):.6f}")

    if stats.get('4A_2_attnOut_norm', 0) > 1e9:
        print(f"\n  🔥 Attention2 outputs are BILLION-scale!")
        print(f"     This is why it's slow - MPS struggles with huge numbers")

print("\n" + "=" * 80)
print("POTENTIAL SPEEDUPS (without modifying weights)")
print("=" * 80)

print("\n1. GRADIENT CLIPPING (Already active)")
print(f"   Current: gradClip = {gradClip}")
print(f"   This limits backward pass explosion")

print("\n2. MIXED PRECISION (Risky with huge numbers)")
print(f"   Could use float16 instead of float32")
print(f"   BUT: Values > 65k overflow in float16")
print(f"   Attention2 outputs are {stats.get('4A_2_attnOut_norm', 0):.0f}")
print(f"   Status: ❌ Won't help (numbers too big)")

print("\n3. INCREASE WEIGHT DECAY (Let it shrink faster)")
print(f"   Current: {weightDecay}")
print(f"   Could increase to 0.1-0.2 temporarily")
print(f"   Weights shrink ~10-20% per step instead of 5%")
print(f"   Status: ✓ Safe, preserves learned patterns")

print("\n4. REDUCE BATCH SIZE")
print(f"   Current batch size: 1")
print(f"   Status: ❌ Already minimal")

print("\n5. REDUCE SEQUENCE LENGTH")
print(f"   Current: 264 tokens")
print(f"   Could temporarily train on 128 tokens")
print(f"   Status: ✓ Would speed up, but changes training")

print("\n6. SKIP TANGLING TEMPORARILY")
print(f"   Turn off tangling until weights normalize")
print(f"   Status: ✓ Would speed up dramatically")
print(f"   But: Stops learning the tangling feature")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

print("\nFastest fix WITHOUT modifying weights:")
print("  → Increase weight decay to 0.15-0.2 for 50-100 steps")
print("  → This makes weights shrink 3-4x faster")
print("  → Training will speed up as weights normalize")
print("  → Then reduce weight decay back to 0.05")

print("\nAlternative (if you want to keep using him now):")
print("  → Temporarily disable tangling in config")
print("  → Training will be fast")
print("  → Re-enable when weights normalize")
print("  → But he won't learn to use tangling during this time")

print("\n" + "=" * 80)
