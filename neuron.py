# CHARIS CAT 2025

import torch
import torch.nn as nn
from config import *

"""defines a single neuron that processes an embedding vector (in parallelNeuronLayer)"""
class NEURON(nn.Module):
    def __init__(self, embedDimension, activationFunction):
        super().__init__()
        self.embedDimension = embedDimension
        self.activationFunction = activationFunction
        """initialises the weights and bias for a single neuron"""
        self.weights = nn.Parameter(torch.randn(embedDimension))
        #self.weights.data *= 0.01 # start weights at smaller random values for stable training
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, embedVector):
        """neurons forward pass, weighted sum + activation"""
        weightedSum = torch.sum(embedVector * self.weights) + self.bias
        """magic activation function applied to this weighted sum, which outputs a single number from the neuron"""
        output = self.activationFunction(weightedSum)

        return output.squeeze()
    
if __name__ == "__main__":
    TESTembedDimension = 5
    TESTembedVector = [0.5, 0.1, -0.2, 0.8, -0.9]
    
    neuron = NEURON(embedDimension = TESTembedDimension, activationFunction = activationFunction)
    outputActivation = neuron.forward(TESTembedVector)

    print("--- NEURON TESTING START ---")
    print(f"Neuron created with {neuron.embedDimension} inputs and {activationFunction.__name__} activation.")
    print(f"Weights shape: {neuron.weights.shape}") # Check weights shape
    print(f"Bias value: {neuron.bias.item():.4f}") # Check bias value
    print(f"Example inputs: {TESTembedVector}")
    print(f"Output Activation Value: {outputActivation.item():.4f}") # Check output activation
    print("--- NEURON TESTING COMPLETE ---")
