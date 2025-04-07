#CHARIS CAT 2025
# BABYLLM - memoryLayer.py

import torch
import torch.nn as nn
from config import *

"""this makes a rolling buffer of past activations"""
class MEMORYLAYER(nn.Module):
    def __init__(self, numNeurons = numNeurons):
        super().__init__()
        self.numNeurons = numNeurons
        # Learnable decay rates
        self.shortTermDecay = nn.Parameter(torch.tensor(0.7, device = modelDevice))
        self.longTermDecay = nn.Parameter(torch.tensor(0.95, device = modelDevice))
        # lists for the memory
        #self.shortTermMemory = torch.zeros(numNeurons)
        #self.longTermMemory = torch.zeros(numNeurons)
        self.shortTermMemory = torch.zeros(self.numNeurons, device = modelDevice)
        self.longTermMemory = torch.zeros(self.numNeurons, device = modelDevice)
        # gates for it to learn when to use the memory or not, learnable average
        self.shortGate = nn.Parameter(torch.tensor(0.25, device = modelDevice))  
        self.longGate = nn.Parameter(torch.tensor(0.25, device = modelDevice))
        self.currentGate = nn.Parameter(torch.tensor(0.5, device = modelDevice))

    """learns when to forget more or less"""
    def forward(self, combinedActivationsTensor):
        device = self.shortTermMemory.device
        combinedActivationsTensor = combinedActivationsTensor.to(device)
        self.shortTermMemory = self.shortTermMemory.to(device)
        self.longTermMemory = self.longTermMemory.to(device)
        # make sure decay values stay within [0, 1] range
        shortDecay = torch.sigmoid(self.shortTermDecay)  # Force between 0-1
        longDecay = torch.sigmoid(self.longTermDecay)
        # update memories with learned decay rates
        newShortTermMemory = (shortDecay * self.shortTermMemory) + ((1 - shortDecay) * combinedActivationsTensor)
        newLongTermMemory = (longDecay * self.longTermMemory) + ((1 - longDecay) * combinedActivationsTensor)
        self.shortTermMemory = newShortTermMemory.detach()
        self.longTermMemory = newLongTermMemory.detach()
        # log the memory gate sizes
        gateSum = self.shortGate + self.longGate + self.currentGate + 1e-9
        self.latestMemoryGates = torch.stack([
        self.shortGate / gateSum,
        self.longGate / gateSum,
        self.currentGate / gateSum])
        # blend memories using weighted sum of the memories, using gates as weights
        blendedActivations = (
            self.shortGate * self.shortTermMemory) + (
            self.longGate * self.longTermMemory) + (
            self.currentGate * combinedActivationsTensor)
        
        #print(f"Type of blendedActivations: {type(blendedActivations)}")
        #if isinstance(blendedActivations, torch.Tensor):
        #    print(f"Shape of blendedActivations: {blendedActivations.shape}")
        #elif isinstance(blendedActivations, list):
        #    print(f"Length of blendedActivations list: {len(blendedActivations)}")
        #    print(f"Shape of first element in list: {blendedActivations[0].shape}")
        #else:
        #    print("Unknown format for blendedActivations!")

        return blendedActivations
    
    def resetMemory(self):
        device = self.shortTermMemory.device  # grab current device
        self.shortTermMemory = torch.zeros(self.numNeurons, device = modelDevice)
        self.longTermMemory = torch.zeros(self.numNeurons, device = modelDevice)


if __name__ == "__main__":
    memoryLayer = MEMORYLAYER(numNeurons = numNeurons)

    TESTinputSeq = torch.randn(window1, embedDimension)
    TESTinputEmbeds = [TESTinputSeq]

    meanActivationsTensor = parallelNeuronLayer.forward(TESTinputEmbeds)

    print("--- PARALLEL NEURON LAYER TESTING START ---")
    print(f"parallel neuron layer created with {parallelNeuronLayer.numNeurons} neurons.")
    print(f"inputs per neuron (embed dimension): {parallelNeuronLayer.embedDimension}")
    print(f"output activations (first 10):")
    print(meanActivationsTensor[:10])
    print(f"output activations shape: {meanActivationsTensor.shape}")
    print("\n--- PARALLEL NEURON LAYER TESTING COMPLETED ---")