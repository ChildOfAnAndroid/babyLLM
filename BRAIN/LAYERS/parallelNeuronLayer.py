# CHARIS CAT 2025
# BABYLLM - parallelNeuronLayer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from config import *
from BRAIN.LAYERS import S_output

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
        self.neurons = BATCHEDNEURONLAYER(numNeurons=numNeurons, embedDimension=embedDimension, activationFunction=activationFunction)
        self.stats = {}

    """iterates through the list of input embeddings, applies all neurons in parallel for each embedding, produces a vector of neuron outputs"""
    def forward(self, inputEmbeds):
        #inputEmbedsTensor = torch.stack(inputEmbeds, dim=0)
        tinyWindowCount = 0

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
        #timings["WindowWeighting"] = (time.time() - windowMeanTimestamp) * 1000


        combinedActivationsTensor = weightedWindowStack.sum(dim=0, keepdim=True)

        if tinyWindowCount > 0:
            print(f"saw {perTokenActivationsTensor.shape[0]} tokens; created {tinyWindowCount} empty windows.")

        tinyWindowCount = 0

        #combineTimestamp = time.time()
        #timings["CombineLayer"] = (time.time() - combineTimestamp) * 1000
        return combinedActivationsTensor#, perTokenActivationsTensor
    
    def getNeuronStats(self):
        with torch.no_grad():
            #weight_norms = torch.norm(self.neurons.weights, dim=1)
            #weight_sparsity = (self.neurons.weights.abs() < 1e-6).float().mean().item()
            #mean_weight_norm = weight_norms.mean().item()
            max_weight_norm = weight_norms.max().item()
            min_weight_norm = weight_norms.min().item()

            #biases_mean = self.neurons.biases.mean().item()
            #biases_std = self.neurons.biases.std().item()
            biases_min = self.neurons.biases.min().item()
            biases_max = self.neurons.biases.max().item()

            #weights_mean = self.neurons.weights.mean().item()
            #weights_std = self.neurons.weights.std().item()
            weights_min = self.neurons.weights.min().item()
            weights_max = self.neurons.weights.max().item()

            weight_stats = {
                #"weightSparsity": weight_sparsity,
                #"meanWeightNorm": mean_weight_norm,
                "maxWeightNorm": max_weight_norm,
                "minWeightNorm": min_weight_norm,
                #"biasMean": biases_mean,
                #"biasStd": biases_std,
                "biasMin": biases_min,
                "biasMax": biases_max,
                #"weightMean": weights_mean,
                #"weightStd": weights_std,
                "weightMin": weights_min,
                "weightMax": weights_max,
            }

            window_stats = {
                #"windowWeights": self.latestWindowWeights.cpu().tolist() if hasattr(self, 'latestWindowWeights') else []
            }

            return {**weight_stats, **window_stats}
    
class BATCHEDNEURONLAYER(nn.Module):
    def __init__(self, numNeurons=numNeurons, embedDimension=embedDimension, activationFunction=activationFunction):
        super().__init__()
        self.numNeurons = numNeurons
        self.embedDimension = embedDimension
        self.weights = nn.Parameter(torch.randn(numNeurons, embedDimension, device = modelDevice) * 0.01)
        self.biases = nn.Parameter(torch.zeros(numNeurons, device = modelDevice))
        self.activationFunction = activationFunction

    def forward(self, embed):  # embed: (batch_size, embed_size)
        # Compute batched dot product + bias: (batch_size, num_neurons)
        output = torch.matmul(embed, self.weights.T) + self.biases
    
        """magic activation function applied to this weighted sum, which outputs a single number from the neuron"""
        output = self.activationFunction(output)

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