#!/usr/bin/env python3
# Check if buffers are actually on MPS or stuck on CPU

import torch
from babyLLM import BABYLLM
from school.staffroom.counsellor import COUNSELLOR
from school.staffroom.calligraphist import S_OUTPUT
from school.staffroom.librarian import LIBRARIAN
from school.staffroom.HE_IS_SCRIBE import SCRIBE
from config import modelDevice, learningRateGOAL

print("=" * 80)
print("BUFFER LOCATION DIAGNOSTIC")
print("=" * 80)

print("\n[1/2] Loading model...")
counsellor = COUNSELLOR("diagnostic", _debug=False, _durations=False)
librarian = LIBRARIAN(_counsellor=counsellor, _baseTokenizerPath=None, _forceRetrain=False)
calligraphist = S_OUTPUT(_counsellor=counsellor)
scribe = SCRIBE(_counsellor=counsellor, _calligraphist=calligraphist,
                _librarian=librarian, _numTokensPerStep=264)
model = BABYLLM(_counsellor=counsellor, _calligraphist=calligraphist,
                _scribe=scribe, _librarian=librarian,
                _device=modelDevice, _numTokensPerStep=264,
                _first=True, _learningRateGOAL=learningRateGOAL)

print(f"✓ Model should be on: {modelDevice}")

print("\n[2/2] Loading checkpoint to MPS...")
model.loadModel()
print(f"✓ Checkpoint loaded")

print("\n" + "=" * 80)
print("CHECKING BUFFER/PARAMETER LOCATIONS")
print("=" * 80)

wrong_device_count = 0
cpu_buffers = []
cpu_params = []

print("\nChecking all buffers and parameters...")
for name, tensor in model.named_buffers():
    if tensor.device.type != modelDevice:
        cpu_buffers.append((name, tensor.shape, tensor.device))
        wrong_device_count += 1

for name, param in model.named_parameters():
    if param.device.type != modelDevice:
        cpu_params.append((name, param.shape, param.device))
        wrong_device_count += 1

if wrong_device_count > 0:
    print(f"\n🔥 FOUND {wrong_device_count} TENSORS ON WRONG DEVICE!")

    if cpu_buffers:
        print(f"\n❌ BUFFERS ON CPU (should be on {modelDevice}):")
        for name, shape, device in cpu_buffers[:10]:  # Show first 10
            print(f"  {name:50s} {str(shape):30s} on {device}")
        if len(cpu_buffers) > 10:
            print(f"  ... and {len(cpu_buffers) - 10} more")

    if cpu_params:
        print(f"\n❌ PARAMETERS ON CPU (should be on {modelDevice}):")
        for name, shape, device in cpu_params[:10]:
            print(f"  {name:50s} {str(shape):30s} on {device}")
        if len(cpu_params) > 10:
            print(f"  ... and {len(cpu_params) - 10} more")

    print(f"\n" + "=" * 80)
    print("DIAGNOSIS: CPU↔GPU TRANSFER BOTTLENECK!")
    print("=" * 80)
    print(f"""
This is DEFINITELY your problem!

Every forward pass:
1. Input tokens (on MPS)
2. Hit CPU buffer → CPU↔MPS transfer
3. Process on MPS
4. Hit another CPU buffer → CPU↔MPS transfer
5. Repeat hundreds of times

CPU↔GPU transfers are EXTREMELY slow (10-100x slower than computation)!

This explains:
- Why 3 mins/step is so slow
- Why MPS profiling fails
- Why the model works but is sluggish

FIX: Need to explicitly move these buffers to MPS after loading checkpoint.
""")

else:
    print(f"\n✓ ALL TENSORS ON CORRECT DEVICE ({modelDevice})")
    print(f"\n   If training is still slow, the issue is elsewhere:")
    print(f"   - MPS might be genuinely slow for this model size")
    print(f"   - Need to profile actual MPS operations")
    print(f"   - Check for other bottlenecks (data loading, etc.)")

print("\n" + "=" * 80)
print("DETAILED BUFFER CHECK")
print("=" * 80)

# Specifically check the problematic buffers
critical_buffers = ['char_lookup_data', 'char_mask_data',
                    'memory.shortTermMemory', 'memory.longTermMemory',
                    'memory2.shortTermMemory', 'memory2.longTermMemory']

print("\nCritical buffers:")
for buf_name in critical_buffers:
    try:
        # Navigate nested attributes
        parts = buf_name.split('.')
        obj = model
        for part in parts:
            obj = getattr(obj, part)

        print(f"  {buf_name:30s}: {obj.device} {list(obj.shape)}")
    except AttributeError:
        print(f"  {buf_name:30s}: NOT FOUND")

print("\n" + "=" * 80)
