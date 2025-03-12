# CHARIS CAT 2025

import torch
import torch.nn as nn
from config import *

"""final layer, maps neuron activations to logits for each token in the vocab"""
class OUTPUTLAYER(nn.Module):
    def __init__(self, numNeurons, vocabSize):
        super().__init__()
        self.numNeurons = numNeurons
        self.vocabSize = vocabSize
        self.weights = nn.Parameter(torch.randn(numNeurons, vocabSize))
        self.weights.data *= 0.01
        self.bias = nn.Parameter(torch.zeros(vocabSize))

    def forward(self, layerActivations):
        """imports the activations from parallelNeuronLayer, assuming that is is a tensor"""
        self.activationsTensor = layerActivations
        """if the activation tensor is 1d, we make it 2d, but this shouldnt happen!"""
        if self.activationsTensor.dim() == 1:
            self.activationsTensor = self.activationsTensor.unsqueeze(0) 
        """return logits (not softmax) for better gradient computation in cross-entropy loss"""
        logits = self.activationsTensor @ self.weights + self.bias
        return logits

    
if __name__ == "__main__":
    TESTlayerActivations = torch.randn(numNeurons)

    outputLayer = OUTPUTLAYER(numNeurons, vocabSize = vocabSize)
    logits = outputLayer.forward(TESTlayerActivations)

    print("--- OUTPUT LAYER TESTING START ---")
    print(f"Output Layer created with {outputLayer.vocabSize} vocabulary tokens.")
    print(f"Weight matrix shape: {outputLayer.weights.shape}")
    print(f"Bias vector shape: {outputLayer.bias.shape}")
    print(f"Logits (first 100):")
    print(logits[:10])
    print(f"Logits Shape: {logits.shape}")
    print("--- OUTPUT LAYER TESTING COMPLETE ---")