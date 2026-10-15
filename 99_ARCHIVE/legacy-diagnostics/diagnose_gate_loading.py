#!/usr/bin/env python3
# Diagnose why neuron_tangle_gate loads as 0.677 instead of -4.5951

import torch
from babyLLM import BABYLLM
from school.staffroom.counsellor import COUNSELLOR
from school.staffroom.calligraphist import S_OUTPUT
from school.staffroom.librarian import LIBRARIAN
from school.staffroom.HE_IS_SCRIBE import SCRIBE
from config import *

print("=" * 80)
print("GATE LOADING DIAGNOSTIC")
print("=" * 80)

# Step 1: Check what's in the checkpoint file
print("\n[1/4] Checking checkpoint file...")
checkpoint_path = modelFilePath
state_dict = torch.load(checkpoint_path, map_location='cpu')

neuron_gate_key = 'tangling.neuron_tangle_gate'
if neuron_gate_key in state_dict:
    checkpoint_value = state_dict[neuron_gate_key].item()
    checkpoint_sigmoid = torch.sigmoid(torch.tensor(checkpoint_value)).item()
    print(f"✓ Checkpoint has {neuron_gate_key}")
    print(f"  Raw value: {checkpoint_value:.6f}")
    print(f"  Sigmoid: {checkpoint_sigmoid:.6f}")
else:
    print(f"❌ Checkpoint missing {neuron_gate_key}!")
    exit(1)

# Step 2: Initialize fresh model (before loading checkpoint)
print("\n[2/4] Initializing fresh model...")
counsellor = COUNSELLOR("diagnostic", _debug=False, _durations=False)
librarian = LIBRARIAN(_counsellor=counsellor, _baseTokenizerPath=None, _forceRetrain=False)
calligraphist = S_OUTPUT(_counsellor=counsellor)
scribe = SCRIBE(_counsellor=counsellor, _calligraphist=calligraphist,
                _librarian=librarian, _numTokensPerStep=264)
model = BABYLLM(_counsellor=counsellor, _calligraphist=calligraphist,
                _scribe=scribe, _librarian=librarian,
                _device='cpu',  # Use CPU for faster init
                _numTokensPerStep=264,
                _first=True, _learningRateGOAL=learningRateGOAL)

fresh_value = model.tangling.neuron_tangle_gate.item()
fresh_sigmoid = torch.sigmoid(model.tangling.neuron_tangle_gate).item()
print(f"✓ Fresh model initialized")
print(f"  neuron_tangle_gate raw: {fresh_value:.6f}")
print(f"  sigmoid: {fresh_sigmoid:.6f}")

# Step 3: Load checkpoint
print("\n[3/4] Loading checkpoint...")
model.loadModel(checkpoint_path)

loaded_value = model.tangling.neuron_tangle_gate.item()
loaded_sigmoid = torch.sigmoid(model.tangling.neuron_tangle_gate).item()
print(f"✓ Checkpoint loaded")
print(f"  neuron_tangle_gate raw: {loaded_value:.6f}")
print(f"  sigmoid: {loaded_sigmoid:.6f}")

# Step 4: Check if optimizer is overwriting it
print("\n[4/4] Checking optimizer state...")
if hasattr(model, 'optimizer'):
    # Check if optimizer has state for this parameter
    param_id = id(model.tangling.neuron_tangle_gate)
    optimizer_state = model.optimizer.state

    # Find this parameter in optimizer state
    for group in model.optimizer.param_groups:
        for param in group['params']:
            if id(param) == param_id:
                if param in optimizer_state:
                    print(f"✓ Optimizer has state for neuron_tangle_gate")
                    state = optimizer_state[param]
                    print(f"  Optimizer state keys: {list(state.keys())}")
                    if 'exp_avg' in state:
                        print(f"  exp_avg norm: {state['exp_avg'].norm().item():.6f}")
                    if 'exp_avg_sq' in state:
                        print(f"  exp_avg_sq norm: {state['exp_avg_sq'].norm().item():.6f}")
                else:
                    print(f"⚠️  Optimizer has NO state for neuron_tangle_gate (first step)")
                break
else:
    print(f"⚠️  Model has no optimizer")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if abs(checkpoint_value - loaded_value) < 0.001:
    print(f"\n✓ CHECKPOINT LOADED CORRECTLY")
    print(f"  Checkpoint value: {checkpoint_value:.6f}")
    print(f"  Loaded value: {loaded_value:.6f}")
    print(f"  Sigmoid: {loaded_sigmoid:.6f}")

    if abs(loaded_sigmoid - 0.660339) < 0.01:
        print(f"\n🔥 BUT THIS MATCHES THE WRONG VALUE FROM TRAINING!")
        print(f"  Expected sigmoid: ~0.01")
        print(f"  Actual sigmoid: {loaded_sigmoid:.6f}")
        print(f"\n  This means the checkpoint file itself has the WRONG value!")
        print(f"  The migration script may not have worked correctly.")
    else:
        print(f"\n✓ Value is correct (~0.01)")
else:
    print(f"\n❌ VALUE CHANGED AFTER LOADING!")
    print(f"  Checkpoint: {checkpoint_value:.6f} (sigmoid {checkpoint_sigmoid:.6f})")
    print(f"  After load: {loaded_value:.6f} (sigmoid {loaded_sigmoid:.6f})")
    print(f"\n  Possible causes:")
    print(f"  - loadModel() is overwriting the parameter")
    print(f"  - Optimizer is restoring old state")
    print(f"  - Some initialization code runs after loading")

print("\n" + "=" * 80)
