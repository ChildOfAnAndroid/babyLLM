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
        self.shortGateScaleHistory = []
        self.longGateScaleHistory = []
        self.activationsGateScaleHistory = []
        self.rawActivationsHistory = []
        self.shortTermMemoryHistory = []
        self.longTermMemoryHistory = []
        self.FINALmemoryHistory = []

    @whocalled
    def forward(self, _activationsTensor):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            self.activationsTensor = _activationsTensor

            ʕっʘ‿ʘʔっ("shortTermDecay")
            shortDecay = torch.sigmoid(self.shortTermDecay)
            ʕっʘ‿ʘʔっ("longTermDecay")
            longDecay = torch.sigmoid(self.longTermDecay)

            ʕっʘ‿ʘʔっ("newShortTermMemory")
            newShort = (shortDecay * self.shortTermMemory) + ((1 - shortDecay) * self.activationsTensor)
            ʕっʘ‿ʘʔっ("newLongTermMemory")
            newLong  = (longDecay * self.longTermMemory) + ((1 - longDecay) * self.activationsTensor)

            ʕっʘ‿ʘʔっ("clamp memory gates")
            clampedShort = torch.clamp(self.shortGate, min=1e-3)
            clampedLong = torch.clamp(self.longGate, min=1e-3)
            clampedactivations = torch.clamp(self.currentGate, min=1e-3)

            ʕっʘ‿ʘʔっ("get gateSum")
            gateSum = clampedShort + clampedLong + clampedactivations + 1e-9
            shortGateScale = clampedShort / gateSum
            longGateScale = clampedLong / gateSum
            activationsGateScale = clampedactivations / gateSum
            self.latestMemoryGates = torch.stack([shortGateScale, longGateScale, activationsGateScale]) # needed to be used in babyLLM for processing

            self.FINALmemory = ((shortGateScale * newShort) + (longGateScale * newLong) +(activationsGateScale * self.activationsTensor))

            self.shortGateScaleHistory.append(shortGateScale.item())
            self.longGateScaleHistory.append(longGateScale.item())
            self.activationsGateScaleHistory.append(activationsGateScale.item())

            self.rawActivationsHistory.append(self.activationsTensor.norm().item())
            self.shortTermMemoryHistory.append(self.shortTermMemory.norm().item())
            self.longTermMemoryHistory.append(self.longTermMemory.norm().item())
            self.FINALmemoryHistory.append(self.FINALmemory.norm().item())

            if len(self.shortGateScaleHistory) >= windowMAX:
                self.stats = {
                    "4M_0_rawActivations_norm": sum(self.rawActivationsHistory) / len(self.rawActivationsHistory),

                    "4M_1_shortTermMemory_norm": sum(self.shortTermMemoryHistory) / len(self.shortTermMemoryHistory),
                    "4M_1_longTermMemory_norm": sum(self.longTermMemoryHistory) / len(self.longTermMemoryHistory),

                    "_4M_shortGateScale": sum(self.shortGateScaleHistory) / len(self.shortGateScaleHistory),
                    "_4M_longGateScale": sum(self.longGateScaleHistory) / len(self.longGateScaleHistory),
                    "_4M_activationsGateScale": sum(self.activationsGateScaleHistory) / len(self.activationsGateScaleHistory),
                    
                    "4M_x_FINALmemory_norm": sum(self.FINALmemoryHistory) / len(self.FINALmemoryHistory),

                    "4M_shortDecay": torch.sigmoid(self.shortTermDecay),
                    "4M_longDecay": torch.sigmoid(self.longTermDecay),
                }

                self.shortGateScaleHistory = []
                self.longGateScaleHistory = []
                self.activationsGateScaleHistory = []
                self.rawActivationsHistory = []
                self.shortTermMemoryHistory = []
                self.longTermMemoryHistory = []
                self.FINALmemoryHistory = []

            # store computed memories for after backward
            self.newShort = newShort
            self.newLong = newLong

            return self.FINALmemory

    def updateMemoryBuffers(self):
        with self.counsellor.infodump("updateMemoryBuffers") as ʕっʘ‿ʘʔっ:
            with torch.no_grad():
                self.shortTermMemory.copy_(self.newShort.detach())
                self.longTermMemory.copy_(self.newLong.detach())

    def resetMemory(self):
        with self.counsellor.infodump("resetMemory") as ʕっʘ‿ʘʔっ:
            with torch.no_grad():
                self.shortTermMemory.zero_()
                #self.longTermMemory.zero_() #retaining long term cause, yk, long term! i felt mean!

    def getMemoryStats(self): return self.stats

if __name__ == "__main__":
    memory = MEMORY(numNeurons = numNeurons)
    print("--- MEMORY TESTING STARTED ---")
    print("\n--- MEMORY TESTING COMPLETE ---")