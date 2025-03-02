# CHARIS CAT 2025

import torch
from config import *

class OUTPUTLAYER:
    def __init__(self, numNeurons, vocabSize):
        self.numNeurons = numNeurons
        self.vocabSize = vocabSize

        self.weights = torch.randn(numNeurons, vocabSize)
        self.bias = torch.randn(vocabSize)

    def forward(self, layerActivations):
        print(f"Debug: layerActivations shape: {len(layerActivations)}")
        print(f"Debug: layerActivations (first 10): {layerActivations[:10]}")
        activationsTensor = torch.tensor(layerActivations).clone().detach()
        print(f"Debugging: activationsTensor shape: {activationsTensor.shape}")
        linearOutput = torch.matmul(activationsTensor, self.weights) + self.bias
        print(f"Debug: linearOutput shape: {linearOutput.shape}")
        probabilityDist = torch.softmax(linearOutput, dim=1)

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