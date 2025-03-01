# CHARIS CAT 2025

import torch
from neuron import NEURON
from config import *

class TRANSFORMERLAYER:
    def __init__(self, numNeurons, embedDimension, activationFunction):
        self.numNeurons = numNeurons
        self.embedDimension = embedDimension
        self.activationFunction = activationFunction

        # iterates through initialisation for each neuron, into an array
        self.neurons = []
        for _ in range(numNeurons):
            neuron = NEURON(embedDimension = embedDimension, activationFunction = activationFunction)
            self.neurons.append(neuron)

    def forward(self, inputEmbedsList):
        # iterates through every neuron on list, does functions, outputs 1 number per neuron
        layerActivations = []
        for neuron in self.neurons:
            neuronOutput = neuron.forward(inputEmbedsList)
            layerActivations.append(neuronOutput)
        return layerActivations
    
if __name__ == "__main__":
    transformerLayer = TRANSFORMERLAYER(numNeurons = numNeurons, embedDimension = embedDimension, activationFunction = activationFunction)

    TESTinputEmbeds = torch.randn(embedDimension)
    #TESTinputEmbedsList = [TESTinputEmbeds] # for future when more input tokens, currently 1 can ignore

    layerOutputActivations = transformerLayer.forward(TESTinputEmbeds)

    print("--- Transformer Layer Testing ---")
    print(f"Transformer Layer created with {transformerLayer.numNeurons} neurons.")
    print(f"Inputs per Neuron (embed dimension): {transformerLayer.embedDimension}")
    print(f"Output Activations (first 10):")
    print(layerOutputActivations[:10])
    print(f"Output Activations Shape: {len(layerOutputActivations)}")
    print("\n--- Transformer Layer Testing Completed ---")