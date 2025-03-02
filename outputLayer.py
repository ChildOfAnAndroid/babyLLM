# CHARIS CAT 2025

import torch
import torch.nn as nn
from config import *

class OUTPUTLAYER(nn.Module):
    def __init__(self, numNeurons, vocabSize):
        super().__init__()
        self.numNeurons = numNeurons
        self.vocabSize = vocabSize

        self.weights = nn.Parameter(torch.randn(numNeurons, vocabSize))
        self.bias = nn.Parameter(torch.randn(vocabSize))

    def forward(self, layerActivations):
        print(f"Debug: layerActivations shape: {layerActivations.shape}")
        print(f"Debug: layerActivations (first 10): {layerActivations[:10]}")

        #activationsTensor = torch.tensor(layerActivations).clone().detach()
        self.activationsTensor = layerActivations

        print(f"Debugging: activationsTensor shape: {self.activationsTensor.shape}")

        linearOutput = torch.matmul(self.activationsTensor, self.weights) + self.bias

        print(f"Debug: linearOutput shape: {linearOutput.shape}")

        probabilityDist = torch.softmax(linearOutput, dim=0)

        return probabilityDist
    
if __name__ == "__main__":
    TESTlayerActivations = torch.randn(numNeurons)

    outputLayer = OUTPUTLAYER(numNeurons, vocabSize = vocabSize)
    probabilityDist = outputLayer.forward(TESTlayerActivations)

    print("--- Output Layer Testing ---")
    print(f"Output Layer created with {outputLayer.vocabSize} vocabulary tokens.")
    print(f"Weight matrix shape: {outputLayer.weights.shape}")
    print(f"Bias vector shape: {outputLayer.bias.shape}")
    print(f"Probability Distribution (first 100):")
    print(probabilityDist[:10])
    print(f"Probability Distribution Shape: {probabilityDist.shape}")
    print(f"Sum of Probabilities: {torch.sum(probabilityDist).item():.4f}")
    print("\n--- Output Layer Testing Completed ---")