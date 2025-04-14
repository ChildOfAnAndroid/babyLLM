import torch

# Load the original state_dict
state_dict = torch.load("babyLLM_legacy.pth")

for key in state_dict.keys():
    print(key)

allWindowSizes_new = [32, 2, 4, 8, 12, 16, 20, 24, 28]  

"""-*-*-*- COMMENT EVERYTHING UNDER HERE TO JUST CHECK WHATS IN DICT FIRST -*-*-*-"""

# OTHERWISE UR GONNA RENAME UR WHOLE FILE LOL

weights = []
biases = []
for i in range(len(allWindowSizes_new)):
    w_key = f"interneuronNetwork.windowCombos.{i}.weight"
    b_key = f"interneuronNetwork.windowCombos.{i}.bias"
    weights.append(state_dict[w_key])
    biases.append(state_dict[b_key])
    # Optionally clean up
    del state_dict[w_key]
    del state_dict[b_key]
state_dict[f"interneuronNetwork.windowComboWeights"] = torch.stack(weights)
state_dict[f"interneuronNetwork.windowComboBiases"] = torch.stack(biases)

#for key in list(state_dict.keys()):
#    if "#" in key:
#        print(f"🧹 Removing: {key}")
#        del state_dict[key]

# Mapping of old names to new names
"""rename_map = {
    # EMBEDDING
    "logits.weights": "logits.l_weights"}

    # INTERNEURON NETWORK
    "parallelNeuronLayer.windowWeighting": "interneuronNetwork.cerebellum",
    "parallelNeuronLayer.attentionProjection.0.weight": "interneuronNetwork.queryProj.weight",
    "parallelNeuronLayer.attentionProjection.0.bias": "interneuronNetwork.queryProj.bias",
    "parallelNeuronLayer.attentionProjection.1.weight": "interneuronNetwork.keyProj.weight",
    "parallelNeuronLayer.attentionProjection.1.bias": "interneuronNetwork.keyProj.bias",
    "parallelNeuronLayer.combinationLayer.weight": "interneuronNetwork.combinationLayer.weight",
    "parallelNeuronLayer.combinationLayer.bias": "interneuronNetwork.combinationLayer.bias",

    # WindowCombos
    "parallelNeuronLayer.windowCombos.0.weight": "interneuronNetwork.windowCombos.0.weight",
    "parallelNeuronLayer.windowCombos.0.bias": "interneuronNetwork.windowCombos.0.bias",
    "parallelNeuronLayer.windowCombos.1.weight": "interneuronNetwork.windowCombos.1.weight",
    "parallelNeuronLayer.windowCombos.1.bias": "interneuronNetwork.windowCombos.1.bias",
    "parallelNeuronLayer.windowCombos.2.weight": "interneuronNetwork.windowCombos.2.weight",
    "parallelNeuronLayer.windowCombos.2.bias": "interneuronNetwork.windowCombos.2.bias",
    "parallelNeuronLayer.windowCombos.3.weight": "interneuronNetwork.windowCombos.3.weight",
    "parallelNeuronLayer.windowCombos.3.bias": "interneuronNetwork.windowCombos.3.bias",
    "parallelNeuronLayer.windowCombos.4.weight": "interneuronNetwork.windowCombos.4.weight",
    "parallelNeuronLayer.windowCombos.4.bias": "interneuronNetwork.windowCombos.4.bias",
    "parallelNeuronLayer.windowCombos.5.weight": "interneuronNetwork.windowCombos.5.weight",
    "parallelNeuronLayer.windowCombos.5.bias": "interneuronNetwork.windowCombos.5.bias",
    "parallelNeuronLayer.windowCombos.6.weight": "interneuronNetwork.windowCombos.6.weight",
    "parallelNeuronLayer.windowCombos.6.bias": "interneuronNetwork.windowCombos.6.bias",
    "parallelNeuronLayer.windowCombos.7.weight": "interneuronNetwork.windowCombos.7.weight",
    "parallelNeuronLayer.windowCombos.7.bias": "interneuronNetwork.windowCombos.7.bias",
    "parallelNeuronLayer.windowCombos.8.weight": "interneuronNetwork.windowCombos.8.weight",
    "parallelNeuronLayer.windowCombos.8.bias": "interneuronNetwork.windowCombos.8.bias",

    # Neuron weights and biases
    "interneuronNetwork.neuron.n_weights": "interneuronNetwork.neurons.n_weights",
    "interneuronNetwork.neuron.n_biases": "interneuronNetwork.neurons.n_biases",

    # OUTPUT
    "logits.weights": "logits.l_weights",
    "logits.bias": "logits.l_bias",

    # MEMORY
    "memoryLayer.shortTermDecay": "memory.shortTermDecay",
    "memoryLayer.longTermDecay": "memory.longTermDecay",
    "memoryLayer.shortGate": "memory.shortGate",
    "memoryLayer.longGate": "memory.longGate",
    "memoryLayer.currentGate": "memory.currentGate"""

# Create a new state_dict with renamed keys
"""new_state_dict = {}
for key, value in state_dict.items():
    new_key = key
    for old_name, new_name in rename_map.items():
        if old_name in new_key:
            print(f"Renaming {new_key} → {new_key.replace(old_name, new_name)}")
            new_key = new_key.replace(old_name, new_name)
    new_state_dict[new_key] = value"""

# Save the fixed version
torch.save(state_dict, "babyLLM_legacy_t.pth")
print("Saved updated model to babyLLM_legacy_t.pth")

#unchanged = [k for k in state_dict if k in new_state_dict]
#print(f"Keys unchanged: {len(unchanged)} / {len(state_dict)}")