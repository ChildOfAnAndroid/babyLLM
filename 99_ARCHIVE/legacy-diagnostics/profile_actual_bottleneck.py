#!/usr/bin/env python3
# Profile to find ACTUAL bottleneck causing 3 min/step

import torch
import time
from babyLLM import BABYLLM
from school.staffroom.counsellor import COUNSELLOR
from school.staffroom.calligraphist import S_OUTPUT
from school.staffroom.librarian import LIBRARIAN
from school.staffroom.HE_IS_SCRIBE import SCRIBE
from config import *

print("=" * 80)
print("ACTUAL BOTTLENECK PROFILER")
print("=" * 80)

# Initialize model
print("\n[1/2] Initializing model...")
start_init = time.time()
counsellor = COUNSELLOR("profiler", _debug=False, _durations=False)
librarian = LIBRARIAN(_counsellor=counsellor, _baseTokenizerPath=None, _forceRetrain=False)
calligraphist = S_OUTPUT(_counsellor=counsellor)
scribe = SCRIBE(_counsellor=counsellor, _calligraphist=calligraphist,
                _librarian=librarian, _numTokensPerStep=264)
model = BABYLLM(_counsellor=counsellor, _calligraphist=calligraphist,
                _scribe=scribe, _librarian=librarian,
                _device='cpu',  # Start on CPU to avoid MPS issues
                _numTokensPerStep=264,
                _first=True, _learningRateGOAL=learningRateGOAL)
init_time = time.time() - start_init
print(f"✓ Model initialized in {init_time:.2f}s")

# Create dummy batch
print("\n[2/2] Creating dummy batch...")
batch_size = 1
seq_len = 264
batch_tokens = torch.randint(0, librarian.tokenizer.vocab_size, (batch_size, seq_len))
batch_targets = torch.randint(0, librarian.tokenizer.vocab_size, (batch_size, seq_len))
print(f"✓ Batch shape: {batch_tokens.shape}")

print("\n" + "=" * 80)
print("TIMING BREAKDOWN")
print("=" * 80)

# Test on CPU first (to establish baseline)
print("\n--- CPU TIMING ---")
model = model.to('cpu')
batch_tokens_cpu = batch_tokens.to('cpu')
batch_targets_cpu = batch_targets.to('cpu')

model.eval()
with torch.no_grad():
    # Warmup
    _ = model(batch_tokens_cpu)

    # Time forward pass
    start = time.time()
    logits = model(batch_tokens_cpu)
    forward_time_cpu = time.time() - start

    print(f"  Forward pass: {forward_time_cpu:.2f}s")

# Test on MPS if available
if torch.backends.mps.is_available():
    print("\n--- MPS TIMING ---")
    try:
        model = model.to('mps')
        batch_tokens_mps = batch_tokens.to('mps')
        batch_targets_mps = batch_targets.to('mps')

        model.eval()
        with torch.no_grad():
            # Warmup
            _ = model(batch_tokens_mps)
            torch.mps.synchronize()

            # Time forward pass
            start = time.time()
            logits = model(batch_tokens_mps)
            torch.mps.synchronize()
            forward_time_mps = time.time() - start

            print(f"  Forward pass: {forward_time_mps:.2f}s")
            print(f"  MPS vs CPU: {forward_time_mps/forward_time_cpu:.2f}x")

            if forward_time_mps > forward_time_cpu * 2:
                print(f"\n  ⚠️  MPS is SLOWER than CPU! This is unusual.")
                print(f"     Possible causes:")
                print(f"     - MPS is falling back to CPU for some operations")
                print(f"     - Operations not optimized for MPS")
                print(f"     - Memory bandwidth issues")
    except Exception as e:
        print(f"  ❌ MPS failed: {e}")

print("\n" + "=" * 80)
print("COMPONENT BREAKDOWN (CPU)")
print("=" * 80)

model = model.to('cpu')
batch_tokens = batch_tokens.to('cpu')

# Profile individual components
print("\nTiming major components:")

model.eval()
with torch.no_grad():
    # Embedding
    start = time.time()
    embeds = model.embed(batch_tokens)
    embed_time = time.time() - start
    print(f"  Embedding: {embed_time:.3f}s ({embed_time/forward_time_cpu*100:.1f}%)")

    # Sensory processing
    if hasattr(model, 'sensory'):
        start = time.time()
        sensed = model.sensory(embeds, embeds)
        sensory_time = time.time() - start
        print(f"  Sensory: {sensory_time:.3f}s ({sensory_time/forward_time_cpu*100:.1f}%)")
        embeds = embeds + sensed

    # Attention
    start = time.time()
    attn_out = model.attention(embeds)
    attention_time = time.time() - start
    print(f"  Attention: {attention_time:.3f}s ({attention_time/forward_time_cpu*100:.1f}%)")
    embeds = embeds + attn_out

    # Tangling embed
    if hasattr(model, 'tangling'):
        start = time.time()
        tangle_out = model.tangling.refine(embeds, "embed")
        tangling_embed_time = time.time() - start
        print(f"  Tangling (embed): {tangling_embed_time:.3f}s ({tangling_embed_time/forward_time_cpu*100:.1f}%)")
        embeds = embeds + tangle_out

    # INN
    start = time.time()
    interneuron_output = model.interneuronNetwork(embeds)
    inn_time = time.time() - start
    print(f"  INN: {inn_time:.3f}s ({inn_time/forward_time_cpu*100:.1f}%)")

    # Attention2
    start = time.time()
    attn2_out = model.attention2(interneuron_output)
    attention2_time = time.time() - start
    print(f"  Attention2: {attention2_time:.3f}s ({attention2_time/forward_time_cpu*100:.1f}%)")
    INNOutput = interneuron_output + attn2_out

    # Tangling neuron
    if hasattr(model, 'tangling'):
        start = time.time()
        tangle_neuron = model.tangling.refine(INNOutput, "neuron")
        tangling_neuron_time = time.time() - start
        print(f"  Tangling (neuron): {tangling_neuron_time:.3f}s ({tangling_neuron_time/forward_time_cpu*100:.1f}%)")
        INNOutput = INNOutput + tangle_neuron

    # Scratchpad
    if hasattr(model, 'scratchpad'):
        start = time.time()
        scratch_out = model.scratchpad(INNOutput)
        scratchpad_time = time.time() - start
        print(f"  Scratchpad: {scratchpad_time:.3f}s ({scratchpad_time/forward_time_cpu*100:.1f}%)")
        INNOutput = INNOutput + scratch_out

    # Memory
    start = time.time()
    mem_out = model.memory(INNOutput)
    memory_time = time.time() - start
    print(f"  Memory: {memory_time:.3f}s ({memory_time/forward_time_cpu*100:.1f}%)")

    # Memory2
    start = time.time()
    mem2_out = model.memory2(mem_out)
    memory2_time = time.time() - start
    print(f"  Memory2: {memory2_time:.3f}s ({memory2_time/forward_time_cpu*100:.1f}%)")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print(f"\nTotal forward pass time: {forward_time_cpu:.2f}s")
print(f"Training step ≈ 2-3x forward time (forward + backward + optimizer)")
print(f"Estimated training step: {forward_time_cpu * 2.5:.2f}s")

if forward_time_cpu > 30:
    print(f"\n⚠️  FORWARD PASS IS VERY SLOW (>{forward_time_cpu:.0f}s)")
    print(f"   This is unusual for a 640M parameter model")
    print(f"   Possible causes:")
    print(f"   - Model is running on CPU (very slow for this size)")
    print(f"   - MPS performance issues")
    print(f"   - Inefficient operations in the model")
    print(f"   - Debug prints or logging slowing things down")
elif forward_time_cpu > 10:
    print(f"\n⚠️  Forward pass is slow ({forward_time_cpu:.0f}s)")
    print(f"   Expected: 1-5s on CPU for this model size")
else:
    print(f"\n✓ Forward pass time is reasonable ({forward_time_cpu:.1f}s)")

# Check if actual training time matches our estimate
print(f"\nYou reported: ~180s (3 minutes) per training step")
print(f"Our estimate: {forward_time_cpu * 2.5:.0f}s per training step")
print(f"Ratio: {180 / (forward_time_cpu * 2.5):.2f}x")

if 180 / (forward_time_cpu * 2.5) > 3:
    print(f"\n⚠️  ACTUAL TRAINING IS MUCH SLOWER THAN EXPECTED")
    print(f"   The slowness is NOT from the forward pass!")
    print(f"   Likely causes:")
    print(f"   - Data loading bottleneck")
    print(f"   - Optimizer step is very slow")
    print(f"   - Gradient computation issues")
    print(f"   - Checkpointing/saving overhead")
    print(f"   - Logging/stats collection overhead")

print("\n" + "=" * 80)
