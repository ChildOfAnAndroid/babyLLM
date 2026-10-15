#!/usr/bin/env python3
"""
Migration script to add Tangling + Scratch Pad expansion to existing babyLLM checkpoint
SAFELY adds 30M new parameters without breaking existing 610M trained weights
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import math
from collections import OrderedDict
from config import modelFilePath, embedDimension, numNeurons, modelDevice

def count_parameters(state_dict):
    """Count total parameters in state dict."""
    total = 0
    for key, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            if param.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
                total += param.numel()
    return total

def initialize_tangling_weights(device):
    """
    Initialize tangling module weights.

    Returns dict with:
    - tangling.project_up.weight: (10000, 1024)
    - tangling.project_up.bias: (10000,)
    - tangling.project_down.weight: (1024, 10000)
    - tangling.project_down.bias: (1024,)
    - tangling.embed_tangle_gate: scalar
    - tangling.memory_tangle_gate: scalar
    """
    weights = {}

    # Project up: 1024 → 10000
    # Initialize with Xavier/Glorot to maintain variance
    fan_in, fan_out = embedDimension, numNeurons
    std = math.sqrt(2.0 / (fan_in + fan_out))
    weights['tangling.project_up.weight'] = torch.randn(fan_out, fan_in, device=device) * std
    weights['tangling.project_up.bias'] = torch.zeros(fan_out, device=device)

    # Project down: 10000 → 1024
    fan_in, fan_out = numNeurons, embedDimension
    std = math.sqrt(2.0 / (fan_in + fan_out))
    weights['tangling.project_down.weight'] = torch.randn(fan_out, fan_in, device=device) * std
    weights['tangling.project_down.bias'] = torch.zeros(fan_out, device=device)

    # Gates: start at -2.0 (sigmoid ≈ 0.12, minimal initial impact)
    weights['tangling.embed_tangle_gate'] = torch.tensor(-2.0, device=device)
    weights['tangling.memory_tangle_gate'] = torch.tensor(-2.0, device=device)

    params = sum(w.numel() for w in weights.values())
    print(f"  Tangling: {params:,} parameters")

    return weights

def initialize_scratchpad_weights(device):
    """
    Initialize scratch pad module weights.

    Returns dict with all scratch pad Linear layers and parameters.
    """
    weights = {}

    dim = embedDimension  # 1024

    # Write controller (3 linear layers + 1 gate)
    for name in ['write_query', 'write_key', 'write_value']:
        std = math.sqrt(2.0 / (dim + dim))
        weights[f'scratchpad.{name}.weight'] = torch.randn(dim, dim, device=device) * std
        weights[f'scratchpad.{name}.bias'] = torch.zeros(dim, device=device)

    # Write gate: 1024 → 1
    std = math.sqrt(2.0 / (dim + 1))
    weights['scratchpad.write_gate.weight'] = torch.randn(1, dim, device=device) * std
    weights['scratchpad.write_gate.bias'] = torch.zeros(1, device=device)

    # Read controller
    for name in ['read_query', 'read_key']:
        std = math.sqrt(2.0 / (dim + dim))
        weights[f'scratchpad.{name}.weight'] = torch.randn(dim, dim, device=device) * std
        weights[f'scratchpad.{name}.bias'] = torch.zeros(dim, device=device)

    # Read gate: 1024 → 1
    weights['scratchpad.read_gate.weight'] = torch.randn(1, dim, device=device) * std
    weights['scratchpad.read_gate.bias'] = torch.zeros(1, device=device)

    # Erase controller
    std = math.sqrt(2.0 / (dim + dim))
    weights['scratchpad.erase_query.weight'] = torch.randn(dim, dim, device=device) * std
    weights['scratchpad.erase_query.bias'] = torch.zeros(dim, device=device)

    # Erase gate: 1024 → 1
    std = math.sqrt(2.0 / (dim + 1))
    weights['scratchpad.erase_gate.weight'] = torch.randn(1, dim, device=device) * std
    weights['scratchpad.erase_gate.bias'] = torch.zeros(1, device=device)

    # Integrate: 2048 → 1024
    fan_in, fan_out = dim * 2, dim
    std = math.sqrt(2.0 / (fan_in + fan_out))
    weights['scratchpad.integrate.weight'] = torch.randn(fan_out, fan_in, device=device) * std
    weights['scratchpad.integrate.bias'] = torch.zeros(fan_out, device=device)

    # Learnable parameters
    weights['scratchpad.write_strength'] = torch.tensor(0.5, device=device)
    weights['scratchpad.erase_strength'] = torch.tensor(0.3, device=device)
    weights['scratchpad.read_temperature'] = torch.tensor(1.0, device=device)

    # Buffers (non-trainable, will be registered separately)
    # buffer: (256, 1024)
    # slot_usage: (256,)
    num_slots = 256
    weights['scratchpad.buffer'] = torch.zeros(num_slots, dim, device=device)
    weights['scratchpad.slot_usage'] = torch.zeros(num_slots, device=device)

    params = sum(w.numel() for w in weights.values())
    print(f"  Scratch Pad: {params:,} parameters (includes {num_slots * dim:,} non-trainable buffer)")

    return weights

def migrate_checkpoint(input_path, output_path):
    """
    Load existing checkpoint, add expansion modules, save new checkpoint.
    """
    print("="*80)
    print("BABYLLM EXPANSION MIGRATION")
    print("Adding Tangling + Scratch Pad (30M params)")
    print("="*80)

    # Load existing checkpoint
    print(f"\n1. Loading existing checkpoint from: {input_path}")
    try:
        checkpoint = torch.load(input_path, map_location='cpu')
        print("   ✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load checkpoint: {e}")
        return False

    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("   ✓ Found model_state_dict in checkpoint")
    else:
        state_dict = checkpoint
        print("   ⚠ Using checkpoint directly as state_dict")

    # Count existing parameters
    old_param_count = count_parameters(state_dict)
    print(f"\n2. Existing model: {old_param_count:,} parameters")

    # Initialize new modules
    print("\n3. Initializing new modules:")
    device = torch.device('cpu')  # Initialize on CPU, will be moved to device on load

    tangling_weights = initialize_tangling_weights(device)
    scratchpad_weights = initialize_scratchpad_weights(device)

    # Merge new weights into state dict
    print("\n4. Merging new weights into state dict:")
    new_keys_added = []

    for key, weight in tangling_weights.items():
        if key in state_dict:
            print(f"   ⚠ WARNING: Key {key} already exists! Skipping.")
        else:
            state_dict[key] = weight
            new_keys_added.append(key)

    for key, weight in scratchpad_weights.items():
        if key in state_dict:
            print(f"   ⚠ WARNING: Key {key} already exists! Skipping.")
        else:
            state_dict[key] = weight
            new_keys_added.append(key)

    print(f"   ✓ Added {len(new_keys_added)} new parameter tensors")

    # Count new parameters
    new_param_count = count_parameters(state_dict)
    added_params = new_param_count - old_param_count
    print(f"\n5. New model: {new_param_count:,} parameters")
    print(f"   Added: {added_params:,} parameters ({added_params/1e6:.1f}M)")

    # Verify key modules still exist
    print("\n6. Verifying existing modules intact:")
    required_keys = [
        'embed.e_weights',
        'attention.attn.in_proj_weight',
        'attention2.attn.in_proj_weight',
        'interneuronNetwork.neurons.n_weights',
        'memory.memoryProjector.weight',
        'memory2.memoryProjector.weight',
        'logits.l_weights',
    ]

    all_present = True
    for key in required_keys:
        if key in state_dict:
            print(f"   ✓ {key}")
        else:
            print(f"   ✗ MISSING: {key}")
            all_present = False

    if not all_present:
        print("\n   ✗ ERROR: Some required keys are missing!")
        return False

    # Create new checkpoint with metadata
    print("\n7. Creating new checkpoint:")

    new_checkpoint = {
        'model_state_dict': state_dict,
        'migration_info': {
            'version': 'v1.13_expansion',
            'added_modules': ['tangling', 'scratchpad'],
            'old_param_count': old_param_count,
            'new_param_count': new_param_count,
            'added_param_count': added_params,
        }
    }

    # Preserve other checkpoint data if it exists
    preserve_keys = [
        'optimizer_state_dict',
        'scheduler_state_dict',
        'epoch',
        'step',
        'loss',
        'config',
    ]

    for key in preserve_keys:
        if key in checkpoint:
            new_checkpoint[key] = checkpoint[key]
            print(f"   ✓ Preserved {key}")

    # Save new checkpoint
    print(f"\n8. Saving expanded checkpoint to: {output_path}")
    try:
        torch.save(new_checkpoint, output_path)
        print("   ✓ Checkpoint saved successfully")
    except Exception as e:
        print(f"   ✗ Failed to save checkpoint: {e}")
        return False

    # Summary
    print("\n" + "="*80)
    print("MIGRATION COMPLETE")
    print("="*80)
    print(f"Old model: {old_param_count:,} params ({old_param_count/1e6:.1f}M)")
    print(f"New model: {new_param_count:,} params ({new_param_count/1e6:.1f}M)")
    print(f"Expansion: +{added_params:,} params (+{added_params/1e6:.1f}M)")
    print()
    print("Next steps:")
    print("1. Update babyLLM.py to include tangling and scratchpad modules")
    print("2. Update config.py to track new stats")
    print("3. Update tutor.py to collect new stats")
    print("4. Test forward pass with new checkpoint")
    print("5. Verify gradients flow through new modules")
    print("6. Resume training!")
    print("="*80)

    return True

if __name__ == "__main__":
    # Default paths
    input_checkpoint = modelFilePath
    output_checkpoint = modelFilePath.replace('.pth', '_expanded.pth')

    print(f"""
Migration Configuration:
  Input:  {input_checkpoint}
  Output: {output_checkpoint}

This will create a NEW checkpoint with expansion modules.
Your original checkpoint will NOT be modified.

Press Enter to continue, or Ctrl+C to cancel...
""")

    try:
        input()
    except KeyboardInterrupt:
        print("\n\nMigration cancelled.")
        sys.exit(0)

    success = migrate_checkpoint(input_checkpoint, output_checkpoint)

    if success:
        print("\n✓ Migration successful!")
        sys.exit(0)
    else:
        print("\n✗ Migration failed!")
        sys.exit(1)
