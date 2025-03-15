# CHARIS CAT 2025

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuron import NEURON
from tinyAttentionLayer import TINYATTENTIONLAYER
from config import *

"""layer that applies the same set of neurons to each token embedding independently."""
"""no sequence awareness!"""
class PARALLELNEURONLAYER(nn.Module):
    def __init__(self, numNeurons, embedDimension, activationFunction):
        super().__init__()
        self.numNeurons = numNeurons
        self.embedDimension = embedDimension
        self.activationFunction = activationFunction
        self.allWindowSizes = allWindowSizes
        self.windowAttn = TINYATTENTIONLAYER(embedDimension)
        self.attentionProjection = nn.Sequential(nn.LayerNorm(embedDimension), nn.Linear(embedDimension, numNeurons))
        self.windowWeighting = nn.Parameter(torch.ones(len(allWindowSizes)))
        self.combinationLayer = nn.Linear(self.numNeurons * len(self.allWindowSizes), self.numNeurons)
        """puts each neuron into an array/nn list"""
        self.neurons = nn.ModuleList(
            [NEURON(embedDimension=embedDimension, activationFunction=activationFunction) 
            for _ in range(numNeurons)])

    def forward(self, inputEmbeds):
        """iterates through the list of input embeddings, applies all neurons in parallel for each embedding, produces a vector of neuron outputs"""
        inputEmbedsTensor = torch.stack(inputEmbeds, dim=0)
        attendedEmbeds = self.windowAttn(inputEmbedsTensor)

        layerActivations = []
        for embedVector in inputEmbeds:
            """Stacks outputs from every neuron (for the current 'embedVector') into a tensor"""
            neuronOutputs = torch.stack([neuron(embedVector) for neuron in self.neurons])
            neuronOutputs = torch.clamp(neuronOutputs, min=-5, max=5)  # Prevent neurons from exploding
            #print(f"Debug TRANSFORMERLAYER: Shape of neuronOutputTensor: {neuronOutputTensor.shape}")
            layerActivations.append(neuronOutputs)

        """Stacks the list of activation vectors into a 2D tensor"""
        perTokenActivationsTensor = torch.stack(layerActivations, dim=0)

        windowMeanActivations = []
        for windowSize in allWindowSizes:
            if perTokenActivationsTensor.shape[0] < windowSize:
                print(f"Not enough tokens for a window! Need at least {windowSize}, got {perTokenActivationsTensor.shape[0]}.")
                emptyWindow = torch.zeros_like(perTokenActivationsTensor[0]).unsqueeze(0)
                windowMeanActivations.append(emptyWindow)
                continue
            if windowSize == attentionWindow:
                inputEmbedsTensor = torch.stack(inputEmbeds, dim=0)  # Convert list to tensor
                attentionApplied = self.windowAttn(inputEmbedsTensor[-windowSize:].unsqueeze(0))  # Apply attention
                attentionMean = attentionApplied.mean(dim=1).squeeze(1)  # Shape: [1, embedDimension]

                # Convert attention output to numNeurons for stacking
                projectedAttention = self.attentionProjection(attentionMean) # Convert [1, embedDim] → [1, numNeurons]
                windowMeanActivations.append(projectedAttention)
            else:
                windowMean = torch.mean(perTokenActivationsTensor[-windowSize:], dim=0, keepdim=True)  # [1, numNeurons]
                windowMeanActivations.append(windowMean)

        if not windowMeanActivations:
            print("No valid window sizes")
            return torch.zeros_like(perTokenActivationsTensor[0])

        """combine activations into their own learnable layer"""
        windowMeanStack = torch.stack(windowMeanActivations, dim=0)  # Shape: [num_windows, 1, numNeurons]
        normalizedWeights = (self.windowWeighting + 0.1) / (self.windowWeighting.sum() + 0.1)
        weightedWindowMeanList = []
        for i in range(windowMeanStack.shape[0]): # Iterate through windows
            weightedWindowMean = windowMeanStack[i] * normalizedWeights[i] # Apply weight to each window mean
            weightedWindowMeanList.append(weightedWindowMean)

        weightedWindowMean = torch.cat(weightedWindowMeanList, dim=1) # Concatenate along feature dimension (dim=1)

        combinedActivationsTensor = self.combinationLayer(weightedWindowMean)

        """DEBUG PRINTS"""
        #print(f"Type of perTokenActivationsTensor: {type(perTokenActivationsTensor)}")
        #if isinstance(perTokenActivationsTensor, torch.Tensor):
        #    print(f"Shape of perTokenActivationsTensor: {perTokenActivationsTensor.shape}")
        #elif isinstance(perTokenActivationsTensor, list):
        #    print(f"Length of perTokenActivationsTensor list: {len(perTokenActivationsTensor)}")
        #    print(f"Shape of first element in list: {perTokenActivationsTensor[0].shape}")
        #else:
        #    print("Unknown format for perTokenActivationsTensor!")

        return combinedActivationsTensor#, perTokenActivationsTensor

    #def smallContextWindow(self, perTokenActivationsTensor):
    #    if perTokenActivationsTensor.shape[0] < windowMIN:
    #        print(f"Not enough tokens for a window! Need at least 2, got {perTokenActivationsTensor.shape[0]}.")
    #        return None
    #    """find the first two token activations"""
    #    smallContextActivations = perTokenActivationsTensor[:windowMIN]  # [2, numNeurons]
    #    #window 1 act = cut last two from tensor thing
    #    #mean 
    #    #cat window 1 act tensor
    #    """take the mean"""
    #    smallContextActivationsTensor = torch.mean(windowCatSlices, dim=0, keepdim=True)  # [1, numNeurons]
    #    #return smallContextActivationsTensor
            
    
if __name__ == "__main__":
    parallelNeuronLayer = PARALLELNEURONLAYER(numNeurons = numNeurons, embedDimension = embedDimension, activationFunction = activationFunction)

    TESTinputSeq = torch.randn(window1, embedDimension)
    TESTinputEmbeds = [TESTinputSeq]

    meanActivationsTensor = parallelNeuronLayer.forward(TESTinputEmbeds)

    print("--- PARALLEL NEURON LAYER TESTING START ---")
    print(f"Parallel Neuron Layer created with {parallelNeuronLayer.numNeurons} neurons.")
    print(f"Inputs per Neuron (embed dimension): {parallelNeuronLayer.embedDimension}")
    print(f"Output Activations (first 10):")
    print(meanActivationsTensor[:10])
    print(f"Output Activations Shape: {meanActivationsTensor.shape}")
    print("\n--- PARALLEL NEURON LAYER TESTING COMPLETED ---")