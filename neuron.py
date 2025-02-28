# CHARIS CAT 2025

import torch
from config import *

class NEURON:
    def __init__(self, numInputs, activationFunction):
        self.numInputs = numInputs
        self.activationFunction = activationFunction

        self.weights = torch.randn(numInputs)
        self.bias = torch.randn(1)

    def forward(self, inputs):
        inputsTensor = torch.tensor(inputs)
        weightedSum = torch.sum(inputsTensor * self.weights) + self.bias
        output = self.activationFunction(weightedSum)

        return output
    
if __name__ == "__main__":
    TESTnumInputs = 5
    neuron = NEURON(numInputs = TESTnumInputs, activationFunction = activationFunction)

    TESTinputs = [0.5, 0.1, -0.2, 0.8, -0.9]
    outputActivation = neuron.forward(TESTinputs)

    print("--- Neuron Testing ---")
    print(f"Neuron created with {neuron.numInputs} inputs and ReLU activation.")
    print(f"Weights shape: {neuron.weights.shape}") # Check weights shape
    print(f"Bias value: {neuron.bias.item():.4f}") # Check bias value (formatted to 4 decimal places)
    print(f"Example inputs: {TESTinputs}")
    print(f"Output Activation Value: {outputActivation.item():.4f}") # Check output activation (formatted to 4 decimal places)