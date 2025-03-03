# CHARIS CAT 2025

import torch
import torch.nn as nn
from neuron import NEURON
from config import *

class TRANSFORMERLAYER(nn.Module):
    def __init__(self, numNeurons, embedDimension, activationFunction):
        super().__init__()
        self.numNeurons = numNeurons
        self.embedDimension = embedDimension
        self.activationFunction = activationFunction

        # iterates through initialisation for each neuron, into an array
        #self.neurons = []
        #for _ in range(numNeurons):
        #    self.neuron = NEURON(embedDimension = embedDimension, activationFunction = activationFunction)
        #    self.neurons.append(self.neuron)
        self.neurons = nn.ModuleList(
            [NEURON(embedDimension=embedDimension, activationFunction=activationFunction) 
            for _ in range(numNeurons)])

    def forward(self, inputEmbedsList):
        # iterates through every neuron on list, does functions, outputs 1 number per neuron
        layerActivations = []
        for embedVector in inputEmbedsList:
            #neuronOutput = self.neuron.forward(embedVector)
            #neuronOutputs = [neuron(embedVector) for neuron in self.neurons]
            neuronOutputs = torch.stack([neuron(embedVector) for neuron in self.neurons])
            #neuronOutputTensor = torch.stack(neuronOutputs)
            #neuronOutputTensor = torch.stack(neuronOutputs) # list to tensor
            #print(f"Debug TRANSFORMERLAYER: Shape of neuronOutputTensor: {neuronOutputTensor.shape}")
            layerActivations.append(neuronOutputs)
            layerActivationsTensor = torch.stack(layerActivations, dim=0)  # Now shape: [seq_len, numNeurons]
            layerActivationsTensor = layerActivationsTensor.mean(dim=0, keepdim=True)  # Reduce to [1, numNeurons]
        return layerActivationsTensor
    
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