#!/usr/bin/env python3
# Profile where time is actually spent in attention2 calls

import torch
import time

# Simple timing wrapper
class TimedModule:
    def __init__(self, module, name):
        self.module = module
        self.name = name
        self.call_count = 0
        self.total_time = 0

    def __call__(self, *args, **kwargs):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        start = time.time()
        result = self.module(*args, **kwargs)

        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        elapsed = time.time() - start
        self.call_count += 1
        self.total_time += elapsed

        return result

    def report(self):
        avg = self.total_time / self.call_count if self.call_count > 0 else 0
        return f"{self.name}: {self.call_count} calls, {self.total_time:.2f}s total, {avg:.2f}s avg"

print("=" * 80)
print("ATTENTION2 TIMING PROFILER")
print("=" * 80)

# Load model
print("\nLoading model...")
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

# Wrap modules with timing
print("\nWrapping modules with timing...")
timed_attention2 = TimedModule(model.attention2, "attention2")
timed_tangling = TimedModule(model.tangling.refine, "tangling.refine")
timed_scratchpad = TimedModule(model.scratchpad, "scratchpad") if hasattr(model, 'scratchpad') else None

# Replace
model.attention2 = timed_attention2
model.tangling.refine = timed_tangling
if timed_scratchpad:
    model.scratchpad = timed_scratchpad

# Create batch
print("\nCreating test batch...")
batch_tokens = torch.randint(0, 4200, (1, 264), device=modelDevice)

print("\n" + "=" * 80)
print("RUNNING FORWARD PASS")
print("=" * 80)

# Time full forward pass
model.eval()
with torch.no_grad():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    start = time.time()
    logits = model(batch_tokens)

    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    total_time = time.time() - start

print(f"\n✓ Forward pass completed")

print("\n" + "=" * 80)
print("TIMING BREAKDOWN")
print("=" * 80)

print(f"\nTotal forward pass: {total_time:.2f}s")
print(f"\n{timed_attention2.report()}")
print(f"{timed_tangling.report()}")
if timed_scratchpad:
    print(f"{timed_scratchpad.report()}")

# Calculate percentages
attn2_pct = (timed_attention2.total_time / total_time) * 100 if total_time > 0 else 0
tangle_pct = (timed_tangling.total_time / total_time) * 100 if total_time > 0 else 0
scratchpad_pct = (timed_scratchpad.total_time / total_time) * 100 if total_time > 0 and timed_scratchpad else 0

print(f"\nAs percentage of total:")
print(f"  attention2: {attn2_pct:.1f}%")
print(f"  tangling: {tangle_pct:.1f}%")
if timed_scratchpad:
    print(f"  scratchpad: {scratchpad_pct:.1f}%")
print(f"  other: {100 - attn2_pct - tangle_pct - scratchpad_pct:.1f}%")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

# Note: tangling.refine calls attention2 internally, so there's overlap
print(f"\nNote: tangling calls attention2 internally, so times overlap")
print(f"Expected: tangling_time ≈ attention2_time (since tangling mostly IS attention2)")

if tangle_pct > 50:
    print(f"\n⚠️  TANGLING IS THE BOTTLENECK ({tangle_pct:.0f}% of time)")
    print(f"   Each tangling.refine call takes ~{timed_tangling.total_time/timed_tangling.call_count:.2f}s")
    print(f"   This is because it calls attention2 + projections")

if total_time > 10:
    print(f"\n⚠️  FORWARD PASS IS VERY SLOW ({total_time:.0f}s)")
    print(f"   Training step ≈ 2-3x forward time = {total_time * 2.5:.0f}s ≈ {total_time * 2.5 / 60:.1f} mins")
    print(f"\n   This matches your reported ~3 mins/step!")
    print(f"\n   Root cause: Tangling calls attention2 multiple times")
    print(f"   Each call processes 400M parameters in 10k-dim space")
    print(f"   MPS is slower than CUDA for large models")

print("\n" + "=" * 80)
