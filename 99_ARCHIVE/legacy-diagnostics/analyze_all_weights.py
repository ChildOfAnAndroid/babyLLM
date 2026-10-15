#!/usr/bin/env python3
# Compare ALL weights in checkpoint vs fresh initialization

import torch
from babyLLM import BABYLLM
from school.staffroom.counsellor import COUNSELLOR
from school.staffroom.calligraphist import S_OUTPUT
from school.staffroom.librarian import LIBRARIAN
from school.staffroom.HE_IS_SCRIBE import SCRIBE
from config import *

print("=" * 80)
print("COMPREHENSIVE WEIGHT ANALYSIS: CHECKPOINT vs FRESH INIT")
print("=" * 80)

# Load checkpoint
print("\n[1/2] Loading checkpoint...")
checkpoint_path = modelFilePath
state_dict = torch.load(checkpoint_path, map_location='cpu')
print(f"✓ Loaded checkpoint with {len(state_dict.keys())} parameters")

# Initialize fresh model
print("\n[2/2] Initializing fresh model...")
counsellor = COUNSELLOR("diagnostic", _debug=False, _durations=False)
librarian = LIBRARIAN(_counsellor=counsellor, _baseTokenizerPath=None, _forceRetrain=False)
calligraphist = S_OUTPUT(_counsellor=counsellor)
scribe = SCRIBE(_counsellor=counsellor, _calligraphist=calligraphist,
                _librarian=librarian, _numTokensPerStep=264)
fresh_model = BABYLLM(_counsellor=counsellor, _calligraphist=calligraphist,
                      _scribe=scribe, _librarian=librarian,
                      _device='cpu', _numTokensPerStep=264,
                      _first=True, _learningRateGOAL=learningRateGOAL)
fresh_dict = fresh_model.state_dict()
print(f"✓ Fresh model initialized with {len(fresh_dict.keys())} parameters")

print("\n" + "=" * 80)
print("WEIGHT GROWTH ANALYSIS")
print("=" * 80)

# Group weights by module
modules = {
    'embed': [],
    'attention': [],
    'attention2': [],
    'interneuronNetwork': [],
    'memory': [],
    'memory2': [],
    'logits': [],
    'tangling': [],
    'scratchpad': [],
    'sensory': [],
    'other': []
}

# Categorize all parameters
for key in state_dict.keys():
    param = state_dict[key]

    # Skip non-float tensors
    if param.dtype not in [torch.float32, torch.float16, torch.bfloat16, torch.float64]:
        continue

    # Skip scalars and tiny tensors
    if param.numel() < 10:
        continue

    # Categorize
    if 'tangling' in key and 'attention2' not in key:
        modules['tangling'].append(key)
    elif 'scratchpad' in key:
        modules['scratchpad'].append(key)
    elif 'attention2' in key and 'tangling' not in key:
        modules['attention2'].append(key)
    elif 'attention.' in key or key.startswith('attention.'):
        modules['attention'].append(key)
    elif 'interneuronNetwork' in key:
        modules['interneuronNetwork'].append(key)
    elif 'memory2' in key:
        modules['memory2'].append(key)
    elif 'memory' in key:
        modules['memory'].append(key)
    elif 'embed' in key:
        modules['embed'].append(key)
    elif 'logits' in key:
        modules['logits'].append(key)
    elif 'sensory' in key:
        modules['sensory'].append(key)
    else:
        modules['other'].append(key)

# Analyze each module
results = []

for module_name, keys in modules.items():
    if not keys:
        continue

    print(f"\n{module_name.upper()}:")
    print("-" * 60)

    growth_ratios = []

    for key in sorted(keys):
        ckpt_param = state_dict[key]

        # Check if exists in fresh model
        if key not in fresh_dict:
            print(f"  {key}: ONLY IN CHECKPOINT (not in fresh init)")
            continue

        fresh_param = fresh_dict[key]

        ckpt_norm = ckpt_param.norm().item()
        fresh_norm = fresh_param.norm().item()

        # Calculate growth ratio (avoid division by zero)
        if fresh_norm > 1e-10:
            ratio = ckpt_norm / fresh_norm
            growth_ratios.append(ratio)

            status = ""
            if ratio > 100:
                status = " 🔥🔥🔥"
            elif ratio > 10:
                status = " 🔥🔥"
            elif ratio > 3:
                status = " 🔥"
            elif ratio > 1.5:
                status = " ⚠️"

            print(f"  {key}")
            print(f"    Fresh: {fresh_norm:.2e}, Checkpoint: {ckpt_norm:.2e}, Ratio: {ratio:.1f}x{status}")
        else:
            print(f"  {key}: fresh_norm too small to compare")

    if growth_ratios:
        avg_ratio = sum(growth_ratios) / len(growth_ratios)
        max_ratio = max(growth_ratios)
        min_ratio = min(growth_ratios)

        results.append({
            'module': module_name,
            'avg': avg_ratio,
            'max': max_ratio,
            'min': min_ratio,
            'count': len(growth_ratios)
        })

        print(f"\n  SUMMARY: {len(growth_ratios)} params")
        print(f"    Avg growth: {avg_ratio:.1f}x")
        print(f"    Max growth: {max_ratio:.1f}x")
        print(f"    Min growth: {min_ratio:.1f}x")

# Final summary
print("\n" + "=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)

print("\nModule Growth Rankings (by average):")
results.sort(key=lambda x: x['avg'], reverse=True)

for r in results:
    status = ""
    if r['avg'] > 100:
        status = " 🔥🔥🔥 EXTREME"
    elif r['avg'] > 10:
        status = " 🔥🔥 HIGH"
    elif r['avg'] > 3:
        status = " 🔥 MODERATE"
    elif r['avg'] > 1.5:
        status = " ⚠️ SLIGHT"
    else:
        status = " ✓ NORMAL"

    print(f"  {r['module']:20s}: {r['avg']:6.1f}x avg, {r['max']:6.1f}x max ({r['count']:3d} params){status}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

# Check if expansion grew more than others
expansion_modules = [r for r in results if r['module'] in ['tangling', 'scratchpad']]
core_modules = [r for r in results if r['module'] not in ['tangling', 'scratchpad']]

if expansion_modules and core_modules:
    avg_expansion = sum(r['avg'] for r in expansion_modules) / len(expansion_modules)
    avg_core = sum(r['avg'] for r in core_modules) / len(core_modules)

    print(f"\nExpansion modules avg growth: {avg_expansion:.1f}x")
    print(f"Core modules avg growth: {avg_core:.1f}x")
    print(f"Expansion/Core ratio: {avg_expansion/avg_core:.1f}x")

    if avg_expansion > avg_core * 3:
        print("\n⚠️  EXPANSION MODULES GROWING MUCH FASTER THAN CORE!")
        print("   This suggests the expansion is destabilizing training.")
    elif avg_expansion > avg_core * 1.5:
        print("\n⚠️  Expansion modules growing faster than core (but not extreme)")
    else:
        print("\n✓ Expansion growth proportional to core modules")
        print("  All weights are growing together (normal with training)")

print("\n" + "=" * 80)
