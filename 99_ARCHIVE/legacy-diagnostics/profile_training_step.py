#!/usr/bin/env python3
# Profile a single training step to see where time is being spent

import torch
import time
from babyLLM import BABYLLM
from school.staffroom.counsellor import COUNSELLOR
from school.staffroom.calligraphist import S_OUTPUT
from school.staffroom.librarian import LIBRARIAN
from school.staffroom.HE_IS_SCRIBE import SCRIBE
from config import *

print("=" * 80)
print("TRAINING STEP PROFILER")
print("=" * 80)

# Initialize model with checkpoint
print("\n[1/3] Initializing model components...")
counsellor = COUNSELLOR("profiler", _debug=False, _durations=False)
librarian = LIBRARIAN(_counsellor=counsellor, _baseTokenizerPath=None, _forceRetrain=False)
calligraphist = S_OUTPUT(_counsellor=counsellor)
scribe = SCRIBE(_counsellor=counsellor, _calligraphist=calligraphist,
                _librarian=librarian, _numTokensPerStep=264)

print("\n[2/3] Loading model with checkpoint...")
model = BABYLLM(_counsellor=counsellor, _calligraphist=calligraphist,
                _scribe=scribe, _librarian=librarian,
                _device=modelDevice, _numTokensPerStep=264,
                _first=True, _learningRateGOAL=learningRateGOAL)

print(f"✓ Model loaded on {modelDevice}")

# Get a sample batch
print("\n[3/3] Preparing sample batch...")
try:
    # Get a sample from the dataset
    import os
    dataset_path = "/Users/charis/Dropbox/00_Icharis/01_DATASETS/FINEWEB/fineweb-edu-score-2/data"
    sample_file = None
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if f.endswith('.parquet'):
                sample_file = os.path.join(root, f)
                break
        if sample_file:
            break

    if sample_file:
        import pandas as pd
        df = pd.read_parquet(sample_file)
        sample_text = df['text'].iloc[0][:1000]  # First 1000 chars

        # Tokenize
        tokens = librarian.tokenizer.encode(sample_text)
        if len(tokens) < 264:
            tokens = tokens * ((264 // len(tokens)) + 1)
        tokens = tokens[:264]

        # Create batch
        batch_tokens = torch.tensor([tokens], device=modelDevice)
        batch_targets = torch.tensor([tokens[1:] + [tokens[0]]], device=modelDevice)

        print(f"✓ Sample batch ready: {batch_tokens.shape}")
    else:
        print("⚠️  Could not find dataset, using dummy batch")
        batch_tokens = torch.randint(0, librarian.tokenizer.vocab_size, (1, 264), device=modelDevice)
        batch_targets = torch.randint(0, librarian.tokenizer.vocab_size, (1, 264), device=modelDevice)
except Exception as e:
    print(f"⚠️  Error loading data ({e}), using dummy batch")
    batch_tokens = torch.randint(0, 50000, (1, 264), device=modelDevice)
    batch_targets = torch.randint(0, 50000, (1, 264), device=modelDevice)

print("\n" + "=" * 80)
print("PROFILING FORWARD PASS")
print("=" * 80)

# Profile forward pass
model.eval()
times = {}

with torch.no_grad():
    # Total forward pass
    start = time.time()

    try:
        # Just do forward pass (no backward)
        logits = model(batch_tokens)

        torch.mps.synchronize() if modelDevice == 'mps' else torch.cuda.synchronize() if modelDevice == 'cuda' else None

        forward_time = time.time() - start
        times['forward_pass'] = forward_time

        print(f"\n✓ Forward pass completed in {forward_time:.2f}s")
        print(f"  Output shape: {logits.shape}")

    except Exception as e:
        print(f"\n❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

# Check current stats
print("\n" + "=" * 80)
print("CURRENT MODEL STATS")
print("=" * 80)

stats = model.obtainAllCurrentStats()

print("\n=== GATE VALUES ===")
gate_stats = {k: v for k, v in stats.items() if 'gate' in k.lower() and 'TANGLE' in k}
if gate_stats:
    for key, value in sorted(gate_stats.items()):
        expected = " ✓" if 0.005 < value < 0.02 else " ⚠️"
        print(f"  {key:30s}: {value:.6f}{expected}")
else:
    print("  No gate stats found!")

print("\n=== WEIGHT NORMS ===")
norm_stats = {k: v for k, v in stats.items() if 'norm' in k.lower()}
for key, value in sorted(norm_stats.items())[:10]:  # First 10
    status = ""
    if value > 1e9:
        status = " 🔥🔥🔥 EXTREME"
    elif value > 1e6:
        status = " 🔥🔥 VERY HIGH"
    elif value > 10000:
        status = " 🔥 HIGH"
    elif value > 1000:
        status = " ⚠️ ELEVATED"
    print(f"  {key:30s}: {value:.2e}{status}")

print("\n" + "=" * 80)
print("WEIGHT INSPECTION")
print("=" * 80)

# Check actual parameter norms
print("\nDirect parameter inspection:")
param_norms = {}
for name, param in model.named_parameters():
    if param.dim() > 0:  # Skip scalars
        norm = param.norm().item()
        param_norms[name] = norm

# Sort by norm
sorted_params = sorted(param_norms.items(), key=lambda x: x[1], reverse=True)

print("\nTop 15 largest parameter norms:")
for name, norm in sorted_params[:15]:
    status = ""
    if norm > 1e6:
        status = " 🔥🔥🔥"
    elif norm > 10000:
        status = " 🔥🔥"
    elif norm > 1000:
        status = " 🔥"
    elif norm > 100:
        status = " ⚠️"
    print(f"  {name:50s}: {norm:12.2e}{status}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

# Check if fix is active
neuron_gate = stats.get('TANGLE_neuron_gate', None)
if neuron_gate is not None:
    if 0.005 < neuron_gate < 0.02:
        print("\n✓ FEEDBACK LOOP FIX IS ACTIVE")
        print(f"  Neuron gate = {neuron_gate:.6f} (expected ~0.01)")
    else:
        print(f"\n⚠️  NEURON GATE UNEXPECTED: {neuron_gate:.6f}")
        print("   Expected ~0.01 after fix")
else:
    print("\n❌ NEURON GATE NOT FOUND IN STATS")
    print("   Fix may not be active!")

# Check if weights are still exploded
attention2_norms = [norm for name, norm in param_norms.items() if 'attention2' in name]
if attention2_norms:
    max_attn2 = max(attention2_norms)
    print(f"\n{'✓' if max_attn2 < 200 else '⚠️'} Attention2 max norm: {max_attn2:.2e}")
    if max_attn2 > 1000:
        print("   Weights still exploded - will take time to stabilize")

# Performance assessment
print(f"\n⏱️  Forward pass time: {forward_time:.2f}s")
if forward_time > 30:
    print("   🔥 EXTREMELY SLOW - likely still processing exploded weights")
elif forward_time > 10:
    print("   ⚠️ SLOW - weights may still be large")
elif forward_time > 2:
    print("   Slower than optimal but acceptable for MPS")
else:
    print("   ✓ Normal performance")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

if neuron_gate and 0.005 < neuron_gate < 0.02:
    print("\n1. ✓ Fix is active - neuron gate properly initialized")
else:
    print("\n1. ⚠️ Fix may not be active - check if code was reloaded")

if max(param_norms.values()) > 10000:
    print("\n2. ⚠️ Weights still very large (checkpoint contains exploded weights)")
    print("   Options:")
    print("   a) Continue training - weights should decay over time")
    print("   b) Start fresh - delete checkpoint and reinitialize")
    print("   c) Use a pre-explosion checkpoint if available")

if forward_time > 10:
    print(f"\n3. ⚠️ Forward pass is slow ({forward_time:.1f}s)")
    print("   With exploded weights, this is expected")
    print("   Training step = 2x forward pass + backward pass")
    print(f"   Estimated training time: ~{forward_time * 3:.1f}s = {(forward_time * 3)/60:.1f} mins")

print("\n" + "=" * 80)
