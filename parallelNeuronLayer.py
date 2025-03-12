# CHARIS CAT 2025

import torch
import torch.nn as nn
from neuron import NEURON
from config import *

"""layer that applies the same set of neurons to each token embedding independently."""
"""no sequence awareness!"""
class PARALLELNEURONLAYER(nn.Module):
    def __init__(self, numNeurons, embedDimension, activationFunction):
        super().__init__()
        self.numNeurons = numNeurons
        self.embedDimension = embedDimension
        self.activationFunction = activationFunction
        """puts each neuron into an array/nn list"""
        self.neurons = nn.ModuleList(
            [NEURON(embedDimension=embedDimension, activationFunction=activationFunction) 
            for _ in range(numNeurons)])

    def forward(self, inputEmbedsList):
        """iterates through the list of input embeddings, applies all neurons in parallel for each embedding, produces a vector of neuron outputs"""
        layerActivations = []
        for embedVector in inputEmbedsList:
            """Stacks outputs from every neuron (for the current 'embedVector') into a tensor"""
            neuronOutputs = torch.stack([neuron(embedVector) for neuron in self.neurons])
            #print(f"Debug TRANSFORMERLAYER: Shape of neuronOutputTensor: {neuronOutputTensor.shape}")
            layerActivations.append(neuronOutputs)
            """Stacks the list of activation vectors into a 2D tensor"""
            layerActivationsTensor = torch.stack(layerActivations, dim=0)  # Now shape: [seq_len, numNeurons]
            #layerActivationsTensor = layerActivationsTensor.mean(dim=0, keepdim=True)
        return layerActivationsTensor
    
if __name__ == "__main__":
    parallelNeuronLayer = PARALLELNEURONLAYER(numNeurons = numNeurons, embedDimension = embedDimension, activationFunction = activationFunction)

    TESTinputEmbeds = torch.randn(embedDimension)
    #TESTinputEmbedsList = [TESTinputEmbeds] # for future when more input tokens, currently 1 can ignore

    layerOutputActivations = parallelNeuronLayer.forward(TESTinputEmbeds)

    print("--- PARALLEL NEURON LAYER TESTING STARTS ---")
    print(f"Parallel Neuron Layer created with {parallelNeuronLayer.numNeurons} neurons.")
    print(f"Inputs per Neuron (embed dimension): {parallelNeuronLayer.embedDimension}")
    print(f"Output Activations (first 10):")
    print(layerOutputActivations[:10])
    print(f"Output Activations Shape: {layerOutputActivations.shape}")
    print("\n--- PARALLEL NEURON LAYER TESTING COMPLETE ---")