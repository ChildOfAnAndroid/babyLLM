# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# MEMORY LAYER // BRAIN/LAYERS/memory.py

import torch
import torch.nn as nn
from config import *

"""this makes a rolling buffer of past activations"""
class MEMORY(nn.Module):
    def __init__(self, _counsellor, _device):
        super().__init__()
        self.device = _device
        self.counsellor = _counsellor

        # learnable decay rates and gates
        self.shortTermDecay = nn.Parameter(torch.tensor(0.7, device = self.device))
        self.longTermDecay = nn.Parameter(torch.tensor(0.95, device = self.device))
        self.shortGate = nn.Parameter(torch.tensor(0.25, device = self.device))
        self.longGate = nn.Parameter(torch.tensor(0.25, device = self.device))
        self.currentGate = nn.Parameter(torch.tensor(0.5, device = self.device))

        # buffers to store memory (outside gradient)
        self.register_buffer("shortTermMemory", torch.zeros(1, numNeurons))
        self.register_buffer("longTermMemory", torch.zeros(1, numNeurons))

        # stats
        self.shortGateWeightHistory = []
        self.longGateWeightHistory = []
        self.currentGateWeightHistory = []

    @whocalled
    def forward(self, _activationsTensor):
        with self.counsellor.infodump("forward"):
            shortDecay = torch.sigmoid(self.shortTermDecay)
            longDecay = torch.sigmoid(self.longTermDecay)

            newShort = (shortDecay * self.shortTermMemory) + ((1 - shortDecay) * _activationsTensor)
            newLong  = (longDecay * self.longTermMemory) + ((1 - longDecay) * _activationsTensor)

            clampedShort = torch.clamp(self.shortGate, min=1e-3)
            clampedLong = torch.clamp(self.longGate, min=1e-3)
            clampedCurrent = torch.clamp(self.currentGate, min=1e-3)

            gateSum = clampedShort + clampedLong + clampedCurrent + 1e-9
            shortGateWeight = clampedShort / gateSum
            longGateWeight = clampedLong / gateSum
            currentGateWeight = clampedCurrent / gateSum

            blendedAct = (
                shortGateWeight * newShort +
                longGateWeight * newLong +
                currentGateWeight * _activationsTensor
            )

            self.latestShortGateWeight = shortGateWeight
            self.latestLongGateWeight = longGateWeight
            self.latestCurrentGateWeight = currentGateWeight
            self.latestMemoryGates = torch.stack([shortGateWeight, longGateWeight, currentGateWeight])

            self.shortGateWeightHistory.append(shortGateWeight.item())
            self.longGateWeightHistory.append(longGateWeight.item())
            self.currentGateWeightHistory.append(currentGateWeight.item())

            # Trim if too long
            if len(self.shortGateWeightHistory) >= windowMAX:
                self.stats = {
                    "4M_shortGateWeight": sum(self.shortGateWeightHistory) / len(self.shortGateWeightHistory),
                    "4M_longGateWeight": sum(self.longGateWeightHistory) / len(self.longGateWeightHistory),
                    "4M_currentGateWeight": sum(self.currentGateWeightHistory) / len(self.currentGateWeightHistory),
                    "4M_shortDecay": torch.sigmoid(self.shortTermDecay),
                    "4M_longDecay": torch.sigmoid(self.longTermDecay),
                }

                self.shortGateWeightHistory = []
                self.longGateWeightHistory = []
                self.currentGateWeightHistory = []

            # store computed memories for after backward
            self.newShort = newShort
            self.newLong = newLong

            return blendedAct

    def updateMemoryBuffers(self):
        with self.counsellor.infodump("updateMemoryBuffers"):
            with torch.no_grad():
                self.shortTermMemory.copy_(self.newShort.detach())
                self.longTermMemory.copy_(self.newLong.detach())

    def resetMemory(self):
        with self.counsellor.infodump("resetMemory"):
            with torch.no_grad():
                self.shortTermMemory.zero_()
                self.longTermMemory.zero_()

    def getMemoryStats(self):
        with self.counsellor.infodump("getMemoryStats") as ʕっʘ‿ʘʔっ:
            with torch.no_grad():
                stats = {}
                ʕっʘ‿ʘʔっ("decayStats")

                return stats

if __name__ == "__main__":
    memory = MEMORY(numNeurons = numNeurons)
    print("--- MEMORY TESTING STARTED ---")
    print("\n--- MEMORY TESTING COMPLETE ---")