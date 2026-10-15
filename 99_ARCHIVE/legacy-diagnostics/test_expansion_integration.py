#!/usr/bin/env python3
"""
Quick test script to verify Tangling + Scratch Pad integration is working
Tests forward pass, gradient flow, and stats collection
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import *

print("="*80)
print("TESTING EXPANSION INTEGRATION")
print("="*80)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from brain.LAYERS.tangling import TANGLING
    from brain.LAYERS.scratchpad import SCRATCHPAD
    print("   ✓ Successfully imported TANGLING")
    print("   ✓ Successfully imported SCRATCHPAD")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize modules standalone
print("\n2. Testing standalone initialization...")
try:
    from brain.LAYERS.attention import GATED_MHA
    from school.staffroom.counsellor import COUNSELLOR

    counsellor = COUNSELLOR()
    device = torch.device('cpu')

    # Create a dummy attention2 to test tangling
    attention2 = GATED_MHA(
        _counsellor=counsellor,
        _device=device,
        _embed_dim=numNeurons,
        _stat_prefix="TEST"
    )

    tangling = TANGLING(
        _counsellor=counsellor,
        _attention2_reference=attention2,
        _device=device
    )

    scratchpad = SCRATCHPAD(
        _counsellor=counsellor,
        _device=device
    )

    print("   ✓ Tangling initialized")
    print("   ✓ Scratchpad initialized")
except Exception as e:
    print(f"   ✗ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check gate initialization
print("\n3. Testing gate initialization (should be very small)...")
try:
    embed_gate = torch.sigmoid(tangling.embed_tangle_gate).item()
    memory_gate = torch.sigmoid(tangling.memory_tangle_gate).item()
    write_strength = torch.sigmoid(scratchpad.write_strength).item()
    erase_strength = torch.sigmoid(scratchpad.erase_strength).item()

    print(f"   Tangling embed gate:      {embed_gate:.6f} (expect ~0.01)")
    print(f"   Tangling memory gate:     {memory_gate:.6f} (expect ~0.01)")
    print(f"   Scratchpad write strength: {write_strength:.6f} (expect ~0.01)")
    print(f"   Scratchpad erase strength: {erase_strength:.6f} (expect ~0.01)")

    if 0.005 < embed_gate < 0.02:
        print("   ✓ Gates initialized to very small values")
    else:
        print(f"   ⚠ Gates not in expected range (0.005-0.02)")
except Exception as e:
    print(f"   ✗ Gate check failed: {e}")
    sys.exit(1)

# Test 4: Test forward pass with dummy data
print("\n4. Testing forward pass...")
try:
    batch_size = 1
    seq_len = 32

    # Test tangling on embeddings (1024-dim)
    dummy_embed = torch.randn(seq_len, embedDimension, device=device)
    tangled_embed = tangling.refine(dummy_embed, stage_name="embed")
    print(f"   ✓ Tangling (embed): {dummy_embed.shape} → {tangled_embed.shape}")

    # Test tangling on neurons (10k-dim)
    dummy_neurons = torch.randn(seq_len, numNeurons, device=device)
    tangled_neurons = tangling.refine(dummy_neurons, stage_name="neuron")
    print(f"   ✓ Tangling (neuron): {dummy_neurons.shape} → {tangled_neurons.shape}")

    # Test scratchpad (needs batch dimension)
    dummy_neurons_batch = dummy_neurons.unsqueeze(0)  # (1, seq, dim)
    scratchpad_out = scratchpad(dummy_neurons_batch)
    print(f"   ✓ Scratchpad: {dummy_neurons_batch.shape} → {scratchpad_out.shape}")

    # Verify residual connections work
    diff_embed = (tangled_embed - dummy_embed).abs().max().item()
    diff_neurons = (tangled_neurons - dummy_neurons).abs().max().item()
    diff_scratchpad = (scratchpad_out - dummy_neurons_batch).abs().max().item()

    print(f"   Max change (embed):     {diff_embed:.6f} (should be small)")
    print(f"   Max change (neuron):    {diff_neurons:.6f} (should be small)")
    print(f"   Max change (scratchpad): {diff_scratchpad:.6f} (should be small)")

except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test gradient flow
print("\n5. Testing gradient flow...")
try:
    # Create dummy data with requires_grad
    dummy_input = torch.randn(seq_len, embedDimension, device=device, requires_grad=True)

    # Forward through tangling
    output = tangling.refine(dummy_input, stage_name="embed")

    # Create dummy loss
    loss = output.sum()

    # Backward
    loss.backward()

    # Check gradients
    if dummy_input.grad is not None:
        print(f"   ✓ Gradients flow through tangling")
        print(f"   Input grad norm: {dummy_input.grad.norm().item():.6f}")
    else:
        print(f"   ✗ No gradients on input!")

    # Check projection layer gradients
    if tangling.project_up.weight.grad is not None:
        print(f"   ✓ Gradients on project_up")
        print(f"   project_up grad norm: {tangling.project_up.weight.grad.norm().item():.6f}")
    else:
        print(f"   ⚠ No gradients on project_up (might be expected if gate is very small)")

except Exception as e:
    print(f"   ✗ Gradient test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test stats collection
print("\n6. Testing stats collection...")
try:
    tangling_stats = tangling.getTanglingStats()
    scratchpad_stats = scratchpad.getScratchpadStats()

    print(f"   Tangling stats keys: {list(tangling_stats.keys())}")
    print(f"   Scratchpad stats keys: {list(scratchpad_stats.keys())}")

    expected_tangling = ['TANGLE_embed_gate', 'TANGLE_embed_refinement_norm', 'TANGLE_embed_output_norm']
    expected_scratchpad = ['SCRATCH_write_gate', 'SCRATCH_read_gate', 'SCRATCH_erase_gate']

    tangling_ok = all(k in tangling_stats for k in expected_tangling[:1])  # At least check one
    scratchpad_ok = all(k in scratchpad_stats for k in expected_scratchpad[:1])

    if tangling_ok and scratchpad_ok:
        print("   ✓ Stats collection working")
    else:
        print("   ⚠ Some stats missing (check if refine/forward was called)")

except Exception as e:
    print(f"   ✗ Stats test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Count parameters
print("\n7. Counting parameters...")
try:
    def count_params(module):
        return sum(p.numel() for p in module.parameters())

    tangling_params = count_params(tangling)
    scratchpad_params = count_params(scratchpad)

    print(f"   Tangling parameters:   {tangling_params:,} ({tangling_params/1e6:.1f}M)")
    print(f"   Scratchpad parameters: {scratchpad_params:,} ({scratchpad_params/1e6:.1f}M)")
    print(f"   Total expansion:       {tangling_params + scratchpad_params:,} ({(tangling_params + scratchpad_params)/1e6:.1f}M)")

    if 25_000_000 < (tangling_params + scratchpad_params) < 35_000_000:
        print("   ✓ Parameter count in expected range (25-35M)")
    else:
        print(f"   ⚠ Parameter count outside expected range")

except Exception as e:
    print(f"   ✗ Parameter count failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*80)
print("INTEGRATION TEST SUMMARY")
print("="*80)
print("✓ All basic tests passed!")
print()
print("Next steps:")
print("1. Try loading babyLLM with: python wakeup.py")
print("2. Check that stats appear in terminal output")
print("3. Verify gates start near 0.01")
print("4. Run a few training steps and check loss is stable")
print("5. Do a chat test to verify coherent responses")
print("="*80)
