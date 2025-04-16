import torch
from VER1_config import *
from VER1_babyLLM import BABYLLM

old_path = "babyLLM_legacy.pth"
new_path = "babyLLM_batched.pth"
num_windows = len(allWindowSizes)

# === LOAD OLD STATE ===
print("📥 Loading old model...")
old_state = torch.load(old_path, map_location="mps", weights_only=False)

# === Extract the shared neuron weights ===
shared_weights = old_state["parallelNeuronLayer.neurons.weights"]  # shape: [numNeurons, embedDim]
shared_biases = old_state["parallelNeuronLayer.neurons.biases"]    # shape: [numNeurons]

# === Strategy: make the per-window weights slightly offset to encourage divergence ===
print("🔧 Generating diversified per-window weights...")
per_window_weights = []
per_window_biases = []

for i in range(num_windows):
    weight = shared_weights.clone()
    bias = shared_biases.clone()

    # Slight tweak per window to avoid identical behavior
    weight += (torch.randn_like(weight) * 0.005 * (i + 1))
    bias += (torch.randn_like(bias) * 0.002 * (i + 1))

    per_window_weights.append(weight)
    per_window_biases.append(bias)

# === Load fresh model and insert the new neurons ===
print("🌱 Initializing new BabyLLM...")
vocab = None  # not needed just for state transfer
model = BABYLLM(vocab, embedDimension, numNeurons, activationFunction, startIndex=0)

# === Replace weights directly ===
new_state = model.state_dict()
for i in range(num_windows):
    new_state[f"parallelNeuronLayer.windowed_neurons.{i}.weights"] = per_window_weights[i]
    new_state[f"parallelNeuronLayer.windowed_neurons.{i}.biases"] = per_window_biases[i]

# === Copy all other keys ===
for key in new_state:
    if "windowed_neurons" in key:
        continue  # already handled
    if key in old_state:
        new_state[key] = old_state[key]

# === Save ===
torch.save(new_state, new_path)
print(f"✅ Saved diversified per-window model to: {new_path}")
