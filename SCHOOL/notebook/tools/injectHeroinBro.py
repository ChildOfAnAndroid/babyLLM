# convert_babyLLM_9to32.py

import torch
import torch.nn as nn
import numpy as np
import os
from config import *
from SCHOOL.staffroom.librarian import VOCAB
from BRAIN.LAYERS.interneuronNetwork import PARALLELNEURONLAYER
from BRAIN.LAYERS.embed import EMBEDLAYER
from BRAIN.LAYERS.logits import OUTPUTLAYER
from BRAIN.LAYERS.memory import MEMORYLAYER
from babyLLM import BABYLLM

def upgrade_parallel_neuron_layer(pnl, new_window_sizes):
    old_weights = pnl.windowWeighting.detach().cpu() if hasattr(pnl, 'windowWeighting') else pnl.cerebellum.detach().cpu()
    old_combos = pnl.windowCombos
    old_window_sizes = allWindowSizes

    print(f"🔧 Old window sizes: {old_window_sizes}")
    print(f"🔁 Interpolating cerebellum from {len(old_weights)} → {len(new_window_sizes)}")

    # Interpolate cerebellum
    old_indices = torch.linspace(0, 1, steps=len(old_weights))
    new_indices = torch.linspace(0, 1, steps=len(new_window_sizes))
    interpolated_weights = np.interp(new_indices, old_indices, old_weights.numpy())
    new_weights = torch.tensor(interpolated_weights, dtype=torch.float32, device=modelDevice)
    pnl.cerebellum = nn.Parameter(new_weights)

    # Replace combo layers by copying closest matches
    new_combos = nn.ModuleList()
    for w in new_window_sizes:
        closest_idx = min(range(len(old_window_sizes)), key=lambda j: abs(w - old_window_sizes[j]))
        old_combo = old_combos[closest_idx]
        combo = nn.Linear(old_combo.in_features, old_combo.out_features, device=modelDevice)
        combo.load_state_dict(old_combo.state_dict())
        new_combos.append(combo)

    pnl.windowCombos = new_combos
    pnl.allWindowSizes = new_window_sizes
    print(f"✅ Rebuilt combo layers with {len(new_window_sizes)} entries.")
    return pnl


if __name__ == "__main__":
    print("🚀 Starting model upgrade from 9→32 windows...")

    # Setup
    legacy_path = modelLegacyFilePath
    new_path = modelFilePath
    new_window_sizes = allWindowSizes_new

    # Build model skeleton
    vocab = VOCAB(vocabSize)
    model = BABYLLM(vocab=vocab, embedDimension=embedDimension, numNeurons=numNeurons, activationFunction=activationFunction)

    # Load legacy weights
    print(f"📦 Loading from {legacy_path}...")
    model.load_state_dict(torch.load(legacy_path, map_location=modelDevice), strict=False)

    # Upgrade
    model.parallelNeuronLayer = upgrade_parallel_neuron_layer(model.parallelNeuronLayer, new_window_sizes)

    # Save upgraded model
    torch.save(model.state_dict(), new_path)
    print(f"\n🎉 Upgrade complete! Saved updated model to {new_path}")
