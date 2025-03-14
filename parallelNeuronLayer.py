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
        # Check what type and shape perTokenActivationsTensor is
        layerActivations = []
        for embedVector in inputEmbedsList:
            """Stacks outputs from every neuron (for the current 'embedVector') into a tensor"""
            neuronOutputs = torch.stack([neuron(embedVector) for neuron in self.neurons])
            #print(f"Debug TRANSFORMERLAYER: Shape of neuronOutputTensor: {neuronOutputTensor.shape}")
            layerActivations.append(neuronOutputs)

        """Stacks the list of activation vectors into a 2D tensor"""
        """output without mean = Output Activations Shape: torch.Size([seqLen, numNeurons, 1])"""
        perTokenActivationsTensor = torch.stack(layerActivations, dim=0)
        """output with mean = Output Activations Shape: torch.Size([1, numNeurons, 1])"""
        meanActivationsTensor = perTokenActivationsTensor.mean(dim=0, keepdim=True)
        
        """DEBUG PRINTS"""
        print(f"Type of perTokenActivationsTensor: {type(perTokenActivationsTensor)}")
        if isinstance(perTokenActivationsTensor, torch.Tensor):
            print(f"Shape of perTokenActivationsTensor: {perTokenActivationsTensor.shape}")
        elif isinstance(perTokenActivationsTensor, list):
            print(f"Length of perTokenActivationsTensor list: {len(perTokenActivationsTensor)}")
            print(f"Shape of first element in list: {perTokenActivationsTensor[0].shape}")
        else:
            print("Unknown format for perTokenActivationsTensor!")
        return meanActivationsTensor#, perTokenActivationsTensor
    
if __name__ == "__main__":
    parallelNeuronLayer = PARALLELNEURONLAYER(numNeurons = numNeurons, embedDimension = embedDimension, activationFunction = activationFunction)

    TESTinputEmbeds = torch.randn(trainingWindow)
    TESTinputEmbedsList = [TESTinputEmbeds] # for future when more input tokens, currently 1 can ignore

    meanActivationsTensor = parallelNeuronLayer.forward(TESTinputEmbeds)

    print("--- PARALLEL NEURON LAYER TESTING START ---")
    print(f"Parallel Neuron Layer created with {parallelNeuronLayer.numNeurons} neurons.")
    print(f"Inputs per Neuron (embed dimension): {parallelNeuronLayer.embedDimension}")
    print(f"Output Activations (first 10):")
    print(meanActivationsTensor[:10])
    print(f"Output Activations Shape: {meanActivationsTensor.shape}")
    print("\n--- PARALLEL NEURON LAYER TESTING COMPLETED ---")