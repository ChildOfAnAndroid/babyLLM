# CHARIS CAT 2025
# BABYLLM - parallelNeuronLayer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
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
        self.attentionProjection = nn.Sequential(nn.LayerNorm(embedDimension, device = modelDevice), nn.Linear(embedDimension, numNeurons, device = modelDevice))
        self.windowWeighting = nn.Parameter(torch.ones(len(allWindowSizes), device = modelDevice))
        self.combinationLayer = nn.Linear(self.numNeurons * len(self.allWindowSizes), self.numNeurons)
        self.windowCombos = nn.ModuleList([nn.Linear(self.numNeurons, self.numNeurons, device = modelDevice) for _ in range(len(self.allWindowSizes))])
        """puts each neuron into an array/nn list"""
        #self.neurons = nn.ModuleList(
        #    [NEURON(embedDimension=embedDimension, activationFunction=activationFunction) 
        #    for _ in range(numNeurons)]) # makes it 32 (embed) x 10000 (numNeurons)
        self.neurons = BATCHEDNEURONLAYER(numNeurons=numNeurons, embedDimension=embedDimension, activationFunction=activationFunction)

    """iterates through the list of input embeddings, applies all neurons in parallel for each embedding, produces a vector of neuron outputs"""
    def forward(self, inputEmbeds, trainingStepCounter = None):
        timings = {}
        #inputEmbedsTensor = torch.stack(inputEmbeds, dim=0)
        tinyWindowCount = 0

        #layerActivations = []
        #for embedVector in inputEmbeds:
        #    """Stacks outputs from every neuron (for the current 'embedVector') into a tensor"""
        #    #neuronOutputs = torch.stack([neuron(embedVector) for neuron in self.neurons])
        #    neuronOutputs = self.neurons(embedVector.unsqueeze(0)).squeeze(0) 
        #    neuronOutputs = torch.clamp(neuronOutputs, min=-5, max=5)  # Prevent neurons from exploding
        #    #print(f"Debug TRANSFORMERLAYER: Shape of neuronOutputTensor: {neuronOutputTensor.shape}")
        #    layerActivations.append(neuronOutputs)
        #"""Stacks the list of activation vectors into a 2D tensor"""
        #perTokenActivationsTensor = torch.stack(layerActivations, dim=0)

        #neuronTimestamp = time.time()
        perTokenActivationsTensor = self.neurons(inputEmbeds)
        #timings["NeuronLayer"] = (time.time() - neuronTimestamp) * 1000

        """combine activations into their own learnable layer"""
        #windowTimestamp = time.time()
        windowMeanActivations = []
        for windowSize in allWindowSizes:
            if perTokenActivationsTensor.shape[0] < windowSize:
                tinyWindowCount += 1
                emptyWindow = torch.zeros_like(perTokenActivationsTensor[0]).unsqueeze(0)
                windowMeanActivations.append(emptyWindow)
            else:
                windowMean = torch.mean(perTokenActivationsTensor[-windowSize:], dim=0, keepdim=True)
                windowMeanActivations.append(windowMean)

        if not windowMeanActivations:
            print("no valid window sizes")
            return torch.zeros_like(perTokenActivationsTensor[0])
        #timings["WindowMeans"] = (time.time() - windowTimestamp) * 1000

        #windowMeanTimestamp = time.time()
        windowMeanStack = torch.stack(windowMeanActivations, dim=0).squeeze(1)
        normalizedWeights = F.softmax(self.windowWeighting, dim=0)
        weightedWindowStack = windowMeanStack * normalizedWeights.view(-1, 1)
        #weightedWindowMean = torch.sum(weightedWindowStack, dim=0, keepdim=True)
        #timings["WindowWeighting"] = (time.time() - windowMeanTimestamp) * 1000

        combinedActivationsTensor = weightedWindowStack.sum(dim=0, keepdim=True)

        if tinyWindowCount > 0:
            print(f"saw {perTokenActivationsTensor.shape[0]} tokens; created {tinyWindowCount} empty windows.")

        return combinedActivationsTensor

        #combineTimestamp = time.time()
        combinedActivationsTensor = self.combinationLayer(weightedWindowMean)
        #combinedActivationsTensor = sum(transformedWindows).unsqueeze(0)
        #timings["CombineLayer"] = (time.time() - combineTimestamp) * 1000

        if tinyWindowCount > 0:
            print(f"saw {perTokenActivationsTensor.shape[0]} tokens; created {tinyWindowCount} empty windows.")
            
        tinyWindowCount = 0

        """DEBUG PRINTS"""
        #print(f"Type of perTokenActivationsTensor: {type(perTokenActivationsTensor)}")
        #if isinstance(perTokenActivationsTensor, torch.Tensor):
        #    print(f"Shape of perTokenActivationsTensor: {perTokenActivationsTensor.shape}")
        #elif isinstance(perTokenActivationsTensor, list):
        #    print(f"Length of perTokenActivationsTensor list: {len(perTokenActivationsTensor)}")
        #    print(f"Shape of first element in list: {perTokenActivationsTensor[0].shape}")
        #else:
        #    print("Unknown format for perTokenActivationsTensor!")

        #print(f"Type of combinedActivationsTensor: {type(combinedActivationsTensor)}")
        #if isinstance(combinedActivationsTensor, torch.Tensor):
        #    print(f"Shape of combinedActivationsTensor: {combinedActivationsTensor.shape}")
        #elif isinstance(combinedActivationsTensor, list):
        #    print(f"Length of combinedActivationsTensor list: {len(combinedActivationsTensor)}")
        #    print(f"Shape of first element in list: {combinedActivationsTensor[0].shape}")
        #else:
        #    print("Unknown format for combinedActivationsTensor!")

        #if trainingStepCounter is not None and trainingStepCounter % trainingLogFreq_100 == 0:
        #    durationLogNeurons_1 = (f"Step {trainingStepCounter} | PNL Durations: NeuronLayer: {timings['NeuronLayer']:.2f}ms | "f"WindowMeans: {timings['WindowMeans']:.2f}ms | "f"WindowWeighting: {timings['WindowWeighting']:.2f}ms | "f"CombineLayer: {timings['CombineLayer']:.2f}ms")
        #    with open(durationLogNeuronsPath_1, "a") as logFile:
        #        logFile.write(durationLogNeurons_1 + "\n")
        #        logFile.flush()

        return combinedActivationsTensor#, perTokenActivationsTensor
    
class BATCHEDNEURONLAYER(nn.Module):
    def __init__(self, numNeurons=numNeurons, embedDimension=embedDimension, activationFunction=activationFunction):
        super().__init__()
        self.numNeurons = numNeurons
        self.embedDimension = embedDimension
        self.weights = nn.Parameter(torch.randn(numNeurons, embedDimension, device = modelDevice) * 0.01)
        self.biases = nn.Parameter(torch.zeros(numNeurons, device = modelDevice))
        self.activation_name = activationFunction

    def forward(self, embed):  # embed: (batch_size, embed_size)
        # Compute batched dot product + bias: (batch_size, num_neurons)
        output = torch.matmul(embed, self.weights.T) + self.biases

        if self.activation_name == 'leaky_relu':
            output = F.leaky_relu(output, 0.01)
        elif self.activation_name == 'relu':
            output = F.relu(output)
        elif self.activation_name == 'sigmoid':
            output = torch.sigmoid(output)
        elif self.activation_name == 'tanh':
            output = torch.tanh(output)

        return torch.clamp(output, -5, 5)
            
    
if __name__ == "__main__":
    parallelNeuronLayer = PARALLELNEURONLAYER(numNeurons = numNeurons, embedDimension = embedDimension, activationFunction = activationFunction)

    TESTinputSeq = torch.randn(window1, embedDimension)
    TESTinputEmbeds = TESTinputSeq

    meanActivationsTensor = parallelNeuronLayer.forward(TESTinputEmbeds)

    print("--- PARALLEL NEURON LAYER TESTING START ---")
    print(f"parallel neuron layer created with {parallelNeuronLayer.numNeurons} neurons.")
    print(f"inputs per neuron (embed dimension): {parallelNeuronLayer.embedDimension}")
    print(f"output activations (first 10):")
    print(meanActivationsTensor[:10])
    print(f"output activations shape: {meanActivationsTensor.shape}")
    print("\n--- PARALLEL NEURON LAYER TESTING COMPLETED ---")