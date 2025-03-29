#CHARIS CAT 2025
import torch
import torch.nn as nn
from config import *

"""this makes a rolling buffer of past activations"""
class MEMORYLAYER(nn.Module):
    def __init__(self, numNeurons = numNeurons):
        super().__init__()
        self.numNeurons = numNeurons
        # Learnable decay rates
        self.shortTermDecay = nn.Parameter(torch.tensor(0.7))
        self.longTermDecay = nn.Parameter(torch.tensor(0.95))
        # lists for the memory
        self.shortTermMemory = torch.zeros(numNeurons)
        self.longTermMemory = torch.zeros(numNeurons)
        # gates for it to learn when to use the memory or not, learnable average
        self.shortGate = nn.Parameter(torch.tensor(0.25))  
        self.longGate = nn.Parameter(torch.tensor(0.25))
        self.currentGate = nn.Parameter(torch.tensor(0.5))

    def forward(self, combinedActivationsTensor):
        """learns when to forget more or less."""
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

if __name__ == "__main__":
    memoryLayer = MEMORYLAYER(numNeurons = numNeurons)

    TESTinputSeq = torch.randn(window1, embedDimension)
    TESTinputEmbeds = [TESTinputSeq]

    meanActivationsTensor = parallelNeuronLayer.forward(TESTinputEmbeds)

    print("--- PARALLEL NEURON LAYER TESTING START ---")
    print(f"Parallel Neuron Layer created with {parallelNeuronLayer.numNeurons} neurons.")
    print(f"Inputs per Neuron (embed dimension): {parallelNeuronLayer.embedDimension}")
    print(f"Output Activations (first 10):")
    print(meanActivationsTensor[:10])
    print(f"Output Activations Shape: {meanActivationsTensor.shape}")
    print("\n--- PARALLEL NEURON LAYER TESTING COMPLETED ---")