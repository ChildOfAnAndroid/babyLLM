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

        # Learnable decay rates and gates
        self.shortTermDecay = nn.Parameter(torch.tensor(0.7, device = self.device))
        self.longTermDecay = nn.Parameter(torch.tensor(0.95, device = self.device))
        self.shortGate = nn.Parameter(torch.tensor(0.25, device = self.device))
        self.longGate = nn.Parameter(torch.tensor(0.25, device = self.device))
        self.currentGate = nn.Parameter(torch.tensor(0.5, device = self.device))

        # Buffers to store memory (outside gradient)
        self.register_buffer("shortTermMemory", torch.zeros(1, numNeurons))
        self.register_buffer("longTermMemory", torch.zeros(1, numNeurons))

    def forward(self, _activationsTensor):
        with self.counsellor.infodump("forward"):
            shortDecay = torch.sigmoid(self.shortTermDecay)
            longDecay = torch.sigmoid(self.longTermDecay)

            newShort = (shortDecay * self.shortTermMemory) + ((1 - shortDecay) * _activationsTensor)
            newLong  = (longDecay * self.longTermMemory) + ((1 - longDecay) * _activationsTensor)

            gateSum = self.shortGate + self.longGate + self.currentGate + 1e-9
            shortGateNorm = self.shortGate / gateSum
            longGateNorm = self.longGate / gateSum
            currentGateNorm = self.currentGate / gateSum

            blendedAct = (
                shortGateNorm * newShort +
                longGateNorm * newLong +
                currentGateNorm * _activationsTensor
            )

            self.latestMemoryGates = torch.stack([shortGateNorm, longGateNorm, currentGateNorm])

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
                stats["shortDecay"] = torch.sigmoid(self.shortTermDecay)
                stats["longDecay"] = torch.sigmoid(self.longTermDecay)
                stats["latestMemoryGates"] = self.latestMemoryGates

                return stats

if __name__ == "__main__":
    memory = MEMORY(numNeurons = numNeurons)
    print("--- MEMORY TESTING STARTED ---")
    print("\n--- MEMORY TESTING COMPLETE ---")