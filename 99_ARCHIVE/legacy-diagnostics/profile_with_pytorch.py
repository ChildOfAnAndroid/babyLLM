#!/usr/bin/env python3
# Use PyTorch profiler to find bottlenecks

import torch
import time
from torch.profiler import profile, ProfilerActivity

print("=" * 80)
print("PYTORCH PROFILER - FINDING BOTTLENECKS")
print("=" * 80)

# Load model
print("\n[1/3] Loading model...")
from babyLLM import BABYLLM
from school.staffroom.counsellor import COUNSELLOR
from school.staffroom.calligraphist import S_OUTPUT
from school.staffroom.librarian import LIBRARIAN
from school.staffroom.HE_IS_SCRIBE import SCRIBE
from config import modelDevice, learningRateGOAL

counsellor = COUNSELLOR("profiler", _debug=False, _durations=False)
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
print("\n[2/3] Creating test batch...")
batch_tokens = torch.randint(0, 4200, (1, 264), device=modelDevice)

print("\n[3/3] Warming up...")
model.eval()
with torch.no_grad():
    _ = model(batch_tokens)  # Warmup

print("\n" + "=" * 80)
print("PROFILING FORWARD PASS")
print("=" * 80)

# Simple timing first
print("\n[1/2] Simple timing...")
model.eval()
with torch.no_grad():
    if modelDevice == 'mps':
        torch.mps.synchronize()

    start = time.time()
    logits = model(batch_tokens)

    if modelDevice == 'mps':
        torch.mps.synchronize()

    total_time = time.time() - start

print(f"✓ Forward pass: {total_time:.2f}s")

# Detailed profiling
print("\n[2/2] Detailed profiling...")
model.eval()

activities = [ProfilerActivity.CPU]
if modelDevice == 'mps':
    # MPS profiling might not be fully supported, but try CPU profiling
    pass

with profile(activities=activities, record_shapes=True) as prof:
    with torch.no_grad():
        logits = model(batch_tokens)

print("\n" + "=" * 80)
print("TOP 20 SLOWEST OPERATIONS")
print("=" * 80)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

print("\n" + "=" * 80)
print("GROUPED BY OPERATION TYPE")
print("=" * 80)

# Group by operation name
ops_by_name = {}
for event in prof.key_averages():
    name = event.key
    cpu_time = event.cpu_time_total / 1000000  # Convert to ms
    if cpu_time > 10:  # Only show ops taking > 10ms
        if name in ops_by_name:
            ops_by_name[name] += cpu_time
        else:
            ops_by_name[name] = cpu_time

print("\nOperations taking > 10ms:")
for name, time_ms in sorted(ops_by_name.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {time_ms:8.0f}ms  {name}")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

print(f"\nTotal forward pass: {total_time:.2f}s")
print(f"Training step estimate: {total_time * 2.5:.2f}s = {total_time * 2.5 / 60:.2f} mins")

if total_time > 30:
    print(f"\n🔥 EXTREMELY SLOW!")
    print(f"   This is way too slow for this model size.")
    print(f"\n   Possible causes:")
    print(f"   1. MPS falling back to CPU for some operations")
    print(f"   2. Memory swapping (model too large for GPU memory)")
    print(f"   3. Inefficient operation implementations")
    print(f"   4. Background processes using GPU")
elif total_time > 10:
    print(f"\n⚠️  SLOW but possibly expected")
    print(f"   For 640M params with tangling on MPS")
else:
    print(f"\n✓ Speed is reasonable")

# Check for CPU fallback signs
print(f"\n" + "=" * 80)
print("CHECKING FOR MPS→CPU FALLBACK")
print("=" * 80)

# Look for operations that might be falling back
cpu_heavy_ops = []
for event in prof.key_averages():
    cpu_time = event.cpu_time_total / 1000000
    # If CPU time is very high for a single operation, might be fallback
    if cpu_time > 1000:  # > 1 second on CPU
        cpu_heavy_ops.append((event.key, cpu_time))

if cpu_heavy_ops:
    print(f"\n⚠️  Found {len(cpu_heavy_ops)} operations with high CPU time:")
    for name, time_ms in sorted(cpu_heavy_ops, key=lambda x: x[1], reverse=True):
        print(f"  {time_ms:8.0f}ms  {name}")
    print(f"\n   These might be falling back to CPU!")
else:
    print(f"\n✓ No obvious CPU fallback detected")

print("\n" + "=" * 80)
