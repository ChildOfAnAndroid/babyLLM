# CHARIS CAT 2025

import torch
from neuron import NEURON
from config import *

class TRANSFORMERLAYER:
    def __init__(self, numNeurons, numInputsPerNeuron, activationFunction):
        self.numNeurons = numNeurons
        self.numInputsPerNeuron = numInputsPerNeuron
        self.activationFunction = activationFunction

        self.neurons = []
        for _ in range(numNeurons):
            neuron = NEURON(numInputs = numInputsPerNeuron, activationFunction = activationFunction)
            self.neurons.append(neuron)

    def forward(self, inputEmbedsList):
        layerActivations = []
        for neuron in self.neurons:
            neuronOutput = neuron.forward(inputEmbedsList)
            layerActivations.append(neuronOutput)
        return layerActivations
    
if __name__ == "__main__":
    transformerLayer = TRANSFORMERLAYER(numNeurons = numNeurons_L1, numInputsPerNeuron = embedDimension, activationFunction = activationFunction)

    TESTinputEmbeds = torch.randn(embedDimension)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!

    layerOutputActivations = transformerLayer.forward(TESTinputEmbeds)

    print("--- Transformer Layer Testing ---")
    print(f"Transformer Layer created with {transformerLayer.numNeurons} neurons.")
    print(f"Each neuron has {transformerLayer.numInputsPerNeuron} inputs.")
    print(f"Output Activations (first 10 neurons - simplified for testing):") # Just print first 10 for brevity
    print(layerOutputActivations[:10]) # Print first 10 neuron activations (or just the first neuron's activation if using Option 2)
    print(f"Output Activations Shape (if returning all activations - Option 1): {len(layerOutputActivations)}") # Check length of activations list (should be num_neurons)


    print("\n--- Transformer Layer Testing Completed ---")