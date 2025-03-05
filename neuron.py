# CHARIS CAT 2025

import torch
import torch.nn as nn
from config import *

class NEURON(nn.Module):
    def __init__(self, embedDimension, activationFunction):
        super().__init__()
        self.embedDimension = embedDimension
        self.activationFunction = activationFunction

        # initialises itself with 32-dimension weights list
        self.weights = nn.Parameter(torch.randn(embedDimension))
        #self.weights.data *= 0.01
        # small number that is added after activation function
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, embedVector):
        #embedVector = torch.tensor(embedVector)
        # takes embed vector (iteratively) * its own weights and adds bias
        weightedSum = torch.sum(embedVector * self.weights) + self.bias
        # magic reLU outputs a single number from the neuron
        output = self.activationFunction(weightedSum)

        return output
    
if __name__ == "__main__":
    TESTembedDimension = 5
    TESTembedVector = [0.5, 0.1, -0.2, 0.8, -0.9]
    
    neuron = NEURON(embedDimension = TESTembedDimension, activationFunction = activationFunction)
    # output activation
    outputActivation = neuron.forward(TESTembedVector)

    print("--- Neuron Testing ---")
    print(f"Neuron created with {neuron.embedDimension} inputs and ReLU activation.")
    print(f"Weights shape: {neuron.weights.shape}") # Check weights shape
    print(f"Bias value: {neuron.bias.item():.4f}") # Check bias value (formatted to 4 decimal places)
    print(f"Example inputs: {TESTembedVector}")
    print(f"Output Activation Value: {outputActivation.item():.4f}") # Check output activation (formatted to 4 decimal places)