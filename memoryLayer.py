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

    def forward(self, currentActivations):
        """learns when to forget more or less."""
        # make sure decay values stay within [0, 1] range
        shortDecay = torch.sigmoid(self.shortTermDecay)  # Force between 0-1
        longDecay = torch.sigmoid(self.longTermDecay)
        # update memories with learned decay rates
        self.shortTermMemory = (shortDecay * self.shortTermMemory) + ((1 - shortDecay) * currentActivations)
        self.longTermMemory = (longDecay * self.longTermMemory) + ((1 - longDecay) * currentActivations)
        # blend memories using weighted sum of the memories, using gates as weights
        blendedActivations = (
            self.shortGate * self.shortTermMemory) + (
            self.longGate * self.longTermMemory) + (
            self.currentGate * currentActivations)

        return blendedActivations