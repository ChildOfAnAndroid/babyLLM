# SIMPLE DIAGNOSTIC: Run this with your existing model loaded
# Usage: In your training session, run: exec(open('diagnose_simple.py').read())

import torch

print("=" * 80)
print("ATTENTION2 EXPLOSION DIAGNOSTIC (Simple Version)")
print("=" * 80)

# Assuming 'model' variable exists in your namespace
if 'model' not in dir():
    print("ERROR: No 'model' variable found. Run this from your training session.")
    exit(1)

print("\n1. Checking ATTENTION2 weight norms...")
print("\n   MultiheadAttention internal weights:")
for name, param in model.attention2.attn.named_parameters():
    norm = param.norm().item()
    mean = param.mean().item()
    has_nan = torch.isnan(param).any().item()
    has_inf = torch.isinf(param).any().item()
    print(f"      {name}: norm={norm:.2e}, mean={mean:.2e}", end="")
    if has_nan:
        print(f" ⚠️ NaN!", end="")
    if has_inf:
        print(f" ⚠️ Inf!", end="")
    if norm > 1e6:
        print(f" 🔥 EXPLODED!", end="")
    print()

print("\n2. Checking TANGLING projection weights...")
print(f"   project_up norm: {model.tangling.project_up.weight.norm().item():.2e}")
print(f"   project_down norm: {model.tangling.project_down.weight.norm().item():.2e}")
print(f"   embed_gate: {torch.sigmoid(model.tangling.embed_tangle_gate).item():.6f}")
print(f"   memory_gate: {torch.sigmoid(model.tangling.memory_tangle_gate).item():.6f}")

print("\n3. Checking SCRATCHPAD weights...")
print(f"   write_strength: {torch.sigmoid(model.scratchpad.write_strength).item():.6f}")
print(f"   erase_strength: {torch.sigmoid(model.scratchpad.erase_strength).item():.6f}")

print("\n4. Testing ATTENTION2 on clean input...")
test_10k = torch.randn(64, 10000, device=model.device)
print(f"   Test input norm: {test_10k.norm().item():.2e}")

with torch.no_grad():
    out = model.attention2(test_10k)
    out_norm = out.norm().item()
    print(f"   Attention2 output norm: {out_norm:.2e}")
    if out_norm > 1e9:
        print(f"   🔥 EXPLOSION CONFIRMED!")
        print(f"      Contains NaN: {torch.isnan(out).any().item()}")
        print(f"      Contains Inf: {torch.isinf(out).any().item()}")

print("\n5. Testing with EXPANSION DISABLED...")
with torch.no_grad():
    # Save original values
    orig_embed = model.tangling.embed_tangle_gate.clone()
    orig_memory = model.tangling.memory_tangle_gate.clone()
    orig_write = model.scratchpad.write_strength.clone()
    orig_erase = model.scratchpad.erase_strength.clone()

    # Disable expansion
    model.tangling.embed_tangle_gate.fill_(-100)
    model.tangling.memory_tangle_gate.fill_(-100)
    model.scratchpad.write_strength.fill_(-100)
    model.scratchpad.erase_strength.fill_(-100)

    # Test forward pass
    test_input = torch.randint(0, len(model.librarian.vocabList), (64,), device=model.device)
    try:
        output = model(test_input)
        stats = model.getForwardStats()
        att2_norm = stats.get('4A_1_0_attnOut_norm', 'N/A')
        print(f"   With expansion disabled, attention2 norm: {att2_norm}")
        if isinstance(att2_norm, (int, float)) and att2_norm > 1e9:
            print(f"   🔥 ATTENTION2 EXPLODES EVEN WITHOUT EXPANSION!")
            print(f"   This is a PRE-EXISTING issue, not caused by new modules.")
        else:
            print(f"   ✓ Attention2 is stable when expansion disabled")
            print(f"   Explosion is caused by NEW MODULES")
    except Exception as e:
        print(f"   ERROR during forward pass: {e}")

    # Restore values
    model.tangling.embed_tangle_gate.copy_(orig_embed)
    model.tangling.memory_tangle_gate.copy_(orig_memory)
    model.scratchpad.write_strength.copy_(orig_write)
    model.scratchpad.erase_strength.copy_(orig_erase)

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
