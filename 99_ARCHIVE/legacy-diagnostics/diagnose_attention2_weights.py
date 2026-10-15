# DIAGNOSTIC: Check if attention2 weights have exploded
# This will help us understand WHY the norms are so massive

import torch
from babyLLM import BABYLLM
from config import *

print("=" * 80)
print("ATTENTION2 WEIGHT ANALYSIS")
print("=" * 80)

# Load model
print("\n1. Loading model...")
model = BABYLLM()

print("\n2. Checking ATTENTION2 weight norms...")
print("\n   MultiheadAttention internal weights:")
for name, param in model.attention2.attn.named_parameters():
    norm = param.norm().item()
    mean = param.mean().item()
    std = param.std().item()
    has_nan = torch.isnan(param).any().item()
    has_inf = torch.isinf(param).any().item()
    print(f"      {name}:")
    print(f"         shape={param.shape}, norm={norm:.2e}, mean={mean:.2e}, std={std:.2e}")
    if has_nan:
        print(f"         ⚠️  Contains NaN!")
    if has_inf:
        print(f"         ⚠️  Contains Inf!")
    if norm > 1e6:
        print(f"         🔥 EXPLODED WEIGHT! (norm > 1M)")

print("\n   Attention2 gate:")
gate_val = model.attention2.logit_gate.item()
gate_sigmoid = torch.sigmoid(model.attention2.logit_gate).item()
print(f"      logit_gate = {gate_val:.4f}")
print(f"      sigmoid(gate) = {gate_sigmoid:.4f}")

print("\n3. Checking TANGLING projection weight norms...")
print("\n   Tangling.project_up (1024 → 10k):")
norm = model.tangling.project_up.weight.norm().item()
mean = model.tangling.project_up.weight.mean().item()
std = model.tangling.project_up.weight.std().item()
print(f"      norm={norm:.2e}, mean={mean:.2e}, std={std:.2e}")
if norm > 0.01:
    print(f"      ⚠️  Weight norm is > 0.01 (initialized to 1e-5)")
    print(f"      This suggests optimizer has been updating these weights!")

print("\n   Tangling.project_down (10k → 1024):")
norm = model.tangling.project_down.weight.norm().item()
mean = model.tangling.project_down.weight.mean().item()
std = model.tangling.project_down.weight.std().item()
print(f"      norm={norm:.2e}, mean={mean:.2e}, std={std:.2e}")
if norm > 0.01:
    print(f"      ⚠️  Weight norm is > 0.01 (initialized to 1e-5)")

print("\n   Tangling gates:")
embed_gate_val = model.tangling.embed_tangle_gate.item()
embed_gate_sigmoid = torch.sigmoid(model.tangling.embed_tangle_gate).item()
memory_gate_val = model.tangling.memory_tangle_gate.item()
memory_gate_sigmoid = torch.sigmoid(model.tangling.memory_tangle_gate).item()
print(f"      embed_gate logit={embed_gate_val:.4f}, sigmoid={embed_gate_sigmoid:.6f}")
print(f"      memory_gate logit={memory_gate_val:.4f}, sigmoid={memory_gate_sigmoid:.6f}")

print("\n4. Simulating a forward pass to trace the explosion...")
test_input = torch.randint(0, vocabularySize, (1, 64), device=modelDevice)

print("\n   Testing normal INN → attention2 flow:")
with torch.no_grad():
    # Temporarily disable expansion to isolate attention2
    orig_embed_gate = model.tangling.embed_tangle_gate.clone()
    orig_memory_gate = model.tangling.memory_tangle_gate.clone()
    model.tangling.embed_tangle_gate.fill_(-100)
    model.tangling.memory_tangle_gate.fill_(-100)
    model.scratchpad.write_strength.fill_(-100)
    model.scratchpad.erase_strength.fill_(-100)

    # Run forward
    embeds = model.embed(test_input)
    print(f"      Embeddings norm: {embeds.norm().item():.2e}")

    att_out = model.attention(embeds)
    print(f"      After attention: {att_out.norm().item():.2e}")

    inn_out = model.interneuronNetwork(att_out)
    print(f"      INN output norm: {inn_out.norm().item():.2e}")

    # This is where explosion happens
    att2_out = model.attention2(inn_out)
    print(f"      Attention2 output norm: {att2_out.norm().item():.2e}")

    if att2_out.norm().item() > 1e9:
        print(f"      🔥 EXPLOSION CONFIRMED in attention2!")
        print(f"      Checking for NaN/Inf:")
        print(f"         Contains NaN: {torch.isnan(att2_out).any().item()}")
        print(f"         Contains Inf: {torch.isinf(att2_out).any().item()}")
        print(f"         Max value: {att2_out.max().item():.2e}")
        print(f"         Min value: {att2_out.min().item():.2e}")

    # Restore gates
    model.tangling.embed_tangle_gate.copy_(orig_embed_gate)
    model.tangling.memory_tangle_gate.copy_(orig_memory_gate)

print("\n5. Testing if explosion is pre-existing (before this session)...")
print("   If attention2 weights are already exploded, this is a pre-existing issue.")
print("   If attention2 weights are normal but output explodes, it's a new issue.")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nKey questions to answer:")
print("1. Are attention2's internal weights exploded? (norm > 1M)")
print("2. Are tangling projection weights still ~1e-5, or have they been updated?")
print("3. Does attention2 explode even with expansion disabled?")
print("4. Is the explosion happening in the attention computation itself?")
