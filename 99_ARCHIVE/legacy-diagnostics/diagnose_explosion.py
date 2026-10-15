# DIAGNOSTIC SCRIPT: Find the cause of attention2 explosion
# Run this to understand WHY the massive norms are happening

import torch
import sys
from babyLLM import BABYLLM
from config import *

print("=" * 80)
print("ATTENTION2 EXPLOSION DIAGNOSTIC")
print("=" * 80)

# Load model
print("\n1. Loading model...")
model = BABYLLM()
model.eval()

# Create test input
test_input = torch.randint(0, vocabularySize, (1, 64), device=modelDevice)

print("\n2. Testing BASELINE (all expansion disabled)...")
# Temporarily disable expansion modules by setting gates to zero
with torch.no_grad():
    # Disable tangling
    model.tangling.embed_tangle_gate.fill_(-100)  # sigmoid(-100) ≈ 0
    model.tangling.memory_tangle_gate.fill_(-100)
    # Disable scratchpad
    model.scratchpad.write_strength.fill_(-100)
    model.scratchpad.erase_strength.fill_(-100)

try:
    with torch.no_grad():
        output = model(test_input)
        stats = model.getForwardStats()
        print(f"   ATTENTION2 attnOut_norm: {stats.get('4A_1_0_attnOut_norm', 'N/A'):.2e}")
        print(f"   MEMORY FINAL_norm: {stats.get('5M_memory_4M_x_FINAL_norm', 'N/A'):.2e}")
        print(f"   Loss: {stats.get('loss', 'N/A'):.4f}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n3. Testing TANGLING ONLY (scratchpad disabled)...")
with torch.no_grad():
    # Re-enable tangling
    model.tangling.embed_tangle_gate.fill_(math.log(0.01/0.99))
    model.tangling.memory_tangle_gate.fill_(math.log(0.01/0.99))
    # Keep scratchpad disabled
    model.scratchpad.write_strength.fill_(-100)
    model.scratchpad.erase_strength.fill_(-100)

try:
    with torch.no_grad():
        output = model(test_input)
        stats = model.getForwardStats()
        print(f"   ATTENTION2 attnOut_norm: {stats.get('4A_1_0_attnOut_norm', 'N/A'):.2e}")
        print(f"   TANGLE_embed_refinement_norm: {stats.get('TANGLE_embed_refinement_norm', 'N/A'):.2e}")
        print(f"   TANGLE_neuron_refinement_norm: {stats.get('TANGLE_neuron_refinement_norm', 'N/A'):.2e}")
        print(f"   Loss: {stats.get('loss', 'N/A'):.4f}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n4. Testing SCRATCHPAD ONLY (tangling disabled)...")
with torch.no_grad():
    # Disable tangling
    model.tangling.embed_tangle_gate.fill_(-100)
    model.tangling.memory_tangle_gate.fill_(-100)
    # Re-enable scratchpad
    model.scratchpad.write_strength.fill_(math.log(0.01/0.99))
    model.scratchpad.erase_strength.fill_(math.log(0.01/0.99))

try:
    with torch.no_grad():
        output = model(test_input)
        stats = model.getForwardStats()
        print(f"   ATTENTION2 attnOut_norm: {stats.get('4A_1_0_attnOut_norm', 'N/A'):.2e}")
        print(f"   SCRATCH_buffer_norm: {stats.get('SCRATCH_buffer_norm', 'N/A'):.2e}")
        print(f"   Loss: {stats.get('loss', 'N/A'):.4f}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n5. Checking ATTENTION2 for stateful components...")
print(f"   Checking attention2 buffers:")
for name, buffer in model.attention2.named_buffers():
    print(f"      {name}: shape={buffer.shape}, norm={buffer.norm().item():.2e}")

print(f"\n   Checking attention2 parameters:")
for name, param in model.attention2.named_parameters():
    print(f"      {name}: shape={param.shape}, norm={param.norm().item():.2e}")

print("\n6. Testing SINGLE vs MULTIPLE attention2 calls...")
print("   Calling attention2 once on test tensor:")
test_10k = torch.randn(1, 64, numNeurons, device=modelDevice)
with torch.no_grad():
    out1 = model.attention2(test_10k)
    print(f"      Output norm: {out1.norm().item():.2e}")

print("   Calling attention2 four times sequentially:")
with torch.no_grad():
    for i in range(4):
        out = model.attention2(test_10k)
        print(f"      Call {i+1} output norm: {out.norm().item():.2e}")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
print("\nNext steps based on results:")
print("- If BASELINE shows explosion → Pre-existing issue, not expansion")
print("- If TANGLING causes explosion → Issue with multiple attention2 calls")
print("- If SCRATCHPAD causes explosion → Issue with scratchpad integration")
print("- If attention2 buffers are large → Stateful accumulation problem")
print("- If sequential calls show growing norms → Attention2 has hidden state")
