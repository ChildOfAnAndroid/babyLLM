#!/usr/bin/env python3
# Compare training speed WITH and WITHOUT tangling/scratchpad

import torch
import time
from babyLLM import BABYLLM
from school.staffroom.counsellor import COUNSELLOR
from school.staffroom.calligraphist import S_OUTPUT
from school.staffroom.librarian import LIBRARIAN
from school.staffroom.HE_IS_SCRIBE import SCRIBE
from config import *

print("=" * 80)
print("MODULE SPEED TEST: Tangling & Scratchpad Impact")
print("=" * 80)

# Initialize
print("\n[1/2] Initializing model...")
counsellor = COUNSELLOR("speed_test", _debug=False, _durations=False)
librarian = LIBRARIAN(_counsellor=counsellor, _baseTokenizerPath=None, _forceRetrain=False)
calligraphist = S_OUTPUT(_counsellor=counsellor)
scribe = SCRIBE(_counsellor=counsellor, _calligraphist=calligraphist,
                _librarian=librarian, _numTokensPerStep=264)
model = BABYLLM(_counsellor=counsellor, _calligraphist=calligraphist,
                _scribe=scribe, _librarian=librarian,
                _device=modelDevice, _numTokensPerStep=264,
                _first=True, _learningRateGOAL=learningRateGOAL)

print(f"✓ Model loaded on {modelDevice}")

# Create batch
print("\n[2/2] Creating test batch...")
batch_tokens = torch.randint(0, librarian.tokenizer.vocab_size, (1, 264), device=modelDevice)
batch_targets = torch.randint(0, librarian.tokenizer.vocab_size, (1, 264), device=modelDevice)

print("\n" + "=" * 80)
print("SPEED TEST")
print("=" * 80)

def time_forward_backward(model, batch_tokens, batch_targets, name, num_runs=3):
    """Time forward + backward pass"""
    times = []

    for i in range(num_runs):
        if modelDevice == 'mps':
            torch.mps.synchronize()
        elif modelDevice == 'cuda':
            torch.cuda.synchronize()

        start = time.time()

        # Forward
        model.train()
        logits = model(batch_tokens)

        # Loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch_targets.view(-1)
        )

        # Backward
        loss.backward()

        if modelDevice == 'mps':
            torch.mps.synchronize()
        elif modelDevice == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start
        times.append(elapsed)

        # Clear gradients
        model.zero_grad()

    avg_time = sum(times) / len(times)
    print(f"\n{name}:")
    print(f"  Runs: {times}")
    print(f"  Average: {avg_time:.2f}s")
    return avg_time

# Test 1: WITH tangling and scratchpad (current state)
print("\n--- Test 1: WITH Tangling & Scratchpad (Current) ---")
time_with = time_forward_backward(model, batch_tokens, batch_targets, "With expansion modules", num_runs=2)

# Test 2: WITHOUT tangling and scratchpad
print("\n--- Test 2: WITHOUT Tangling & Scratchpad ---")
print("Temporarily disabling modules...")

# Save original methods
original_tangling_refine = model.tangling.refine if hasattr(model, 'tangling') else None
original_scratchpad_forward = model.scratchpad.forward if hasattr(model, 'scratchpad') else None

# Replace with no-op
if hasattr(model, 'tangling'):
    model.tangling.refine = lambda x, stage_name="unknown": torch.zeros_like(x)
if hasattr(model, 'scratchpad'):
    model.scratchpad.forward = lambda x: torch.zeros_like(x)

time_without = time_forward_backward(model, batch_tokens, batch_targets, "Without expansion modules", num_runs=2)

# Restore
if original_tangling_refine:
    model.tangling.refine = original_tangling_refine
if original_scratchpad_forward:
    model.scratchpad.forward = original_scratchpad_forward

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\nWith tangling & scratchpad:    {time_with:.2f}s")
print(f"Without tangling & scratchpad: {time_without:.2f}s")
print(f"Overhead from new modules:     {time_with - time_without:.2f}s ({(time_with/time_without - 1)*100:.1f}% slower)")

print(f"\nTraining step estimate (3x forward+backward for optimizer):")
print(f"  With modules:    {time_with * 3:.2f}s = {time_with * 3 / 60:.1f} mins")
print(f"  Without modules: {time_without * 3:.2f}s = {time_without * 3 / 60:.1f} mins")

if time_with > time_without * 2:
    print(f"\n⚠️  MODULES ADD SIGNIFICANT OVERHEAD ({(time_with/time_without - 1)*100:.0f}%)")
    print(f"   This explains the slowdown when you added them!")
    print(f"\n   Potential causes:")
    print(f"   - Tangling calls attention2 THREE times (embed, neuron, memory)")
    print(f"   - Scratchpad content-addressable memory operations")
    print(f"   - Projection layers (1k→10k→1k) for tangling")
    print(f"   - Additional matrix operations")
else:
    print(f"\n✓ Module overhead is reasonable")

# Check if fix is active
print(f"\n" + "=" * 80)
print("CHECKING FIX STATUS")
print("=" * 80)

model.eval()
with torch.no_grad():
    _ = model(batch_tokens)
    stats = model.obtainAllCurrentStats()

    neuron_gate = stats.get('TANGLE_neuron_gate', None)
    if neuron_gate:
        print(f"\n✓ Feedback loop fix is ACTIVE")
        print(f"  TANGLE_neuron_gate: {neuron_gate:.6f}")
        if 0.005 < neuron_gate < 0.02:
            print(f"  ✓ Gate value is correct (~0.01)")
        else:
            print(f"  ⚠️  Gate value unexpected (should be ~0.01)")
    else:
        print(f"\n⚠️  TANGLE_neuron_gate not found in stats!")

print("\n" + "=" * 80)
