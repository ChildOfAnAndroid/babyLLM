# convert_old_to_batched.py

import torch
from babyLLM import BABYLLM
from parallelNeuronLayer import BATCHEDNEURONLAYER
from config import *

# ==== LOAD THE OLD MODEL (with ModuleList neurons) ====
class LegacyNEURON(torch.nn.Module):
    def __init__(self, embedDimension, activationFunction):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(embedDimension) * 0.01)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.activation = activationFunction

    def forward(self, x):
        return x  # unused here

class LegacyPARALLELNEURONLAYER(torch.nn.Module):
    def __init__(self, numNeurons, embedDimension, activationFunction):
        super().__init__()
        self.neurons = torch.nn.ModuleList([
            LegacyNEURON(embedDimension, activationFunction)
            for _ in range(numNeurons)
        ])
        self.numNeurons = numNeurons
        self.embedDimension = embedDimension
        self.activationFunction = activationFunction

class LegacyBABYLLM(torch.nn.Module):
    def __init__(self, vocab, embedDimension, numNeurons, activationFunction):
        super().__init__()
        self.vocab = vocab
        self.embedDimension = embedDimension
        self.numNeurons = numNeurons
        self.activationFunction = activationFunction
        self.parallelNeuronLayer = LegacyPARALLELNEURONLAYER(numNeurons, embedDimension, activationFunction)
        self.loadFullModel = True  # you might not need this

# ==== BEGIN MIGRATION ====

old_weights_path = "babyllm_legacy.pth"
new_weights_path = "babyllm_batched.pth"

print("🧠 Loading legacy model...")
legacy_model = LegacyBABYLLM(vocab=None, embedDimension=embedDimension, numNeurons=numNeurons, activationFunction=activationFunction)
legacy_model.load_state_dict(torch.load(old_weights_path, map_location="cpu"), strict=False)

print("🌱 Creating fresh batched model...")
new_model = BABYLLM(vocab=None, embedDimension=embedDimension, numNeurons=numNeurons, activationFunction=activationFunction)

# ==== COPY NEURON WEIGHTS ====
print("🔁 Migrating neurons...")
old_neurons = legacy_model.parallelNeuronLayer.neurons
batched_layer = new_model.parallelNeuronLayer.neurons

batched_layer.weights.data = torch.stack([n.weight.data for n in old_neurons])
batched_layer.biases.data = torch.cat([n.bias.data for n in old_neurons])

# ==== COPY REMAINING PARAMETERS ====
print("📦 Copying rest of weights...")
old_state = torch.load(old_weights_path, map_location="cpu")
new_state = new_model.state_dict()

for key in new_state:
    if "neurons" in key:
        continue  # already handled
    if key in old_state:
        new_state[key] = old_state[key]
    else:
        print(f"⚠️ Missing in old state: {key}")

new_model.load_state_dict(new_state)

# ==== SAVE NEW MODEL ====
torch.save(new_model.state_dict(), new_weights_path)
print(f"✅ Done! Saved new model to: {new_weights_path}")
