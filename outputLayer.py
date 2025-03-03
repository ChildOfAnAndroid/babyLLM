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
        self.weights.data *= 0.01
        self.bias = nn.Parameter(torch.zeros(vocabSize))

    def forward(self, layerActivations):

        if isinstance(layerActivations, list):
            self.activationsTensor = torch.stack(layerActivations, dim=0) 
        else:
            self.activationsTensor = layerActivations

        if self.activationsTensor.dim() == 1:
            self.activationsTensor = self.activationsTensor.unsqueeze(0) 

        self.activationsTensor = self.activationsTensor.view(1, -1) 

        logits = self.activationsTensor @ self.weights + self.bias

        # Ensure we return logits (not softmax) for better gradient computation in cross-entropy loss
        return logits

    
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