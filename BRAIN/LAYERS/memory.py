# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# MEMORY LAYER // BRAIN/LAYERS/memory.py

import torch
import torch.nn as nn
from config import *

"""this makes a rolling buffer of past activations"""
class MEMORY(nn.Module):
    def __init__(self, _counsellor, _numTokensPerStep, _device = modelDevice):
        super().__init__()
        self.device = _device
        self.counsellor = _counsellor
        self.numTokensPerStep = _numTokensPerStep

        # learnable decay rates and gates
        self.shortTermDecay = nn.Parameter(torch.tensor(0.7, device = self.device))
        self.longTermDecay = nn.Parameter(torch.tensor(0.95, device = self.device))
        self.shortGate = nn.Parameter(torch.tensor(0.25, device = self.device))
        self.longGate = nn.Parameter(torch.tensor(0.25, device = self.device))
        self.currentGate = nn.Parameter(torch.tensor(0.5, device = self.device))

        self.inputReducer = nn.Linear(numNeurons, embedDimension, device = self.device)       # 10k → 1k
        self.gateLayer     = nn.Linear(embedDimension, 3 * numNeurons, device = self.device)  # 1k → 30k

        # buffers to store memory (outside gradient)
        self.register_buffer("shortTermMemory", torch.zeros(1, numNeurons))
        self.register_buffer("longTermMemory", torch.zeros(1, numNeurons))

        # stats
        self.shortGateScaleHistory = []
        self.longGateScaleHistory = []
        self.activationsGateScaleHistory = []
        self.gateLayerHistory = []

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
            with torch.no_grad(): self.shortTermDecay.clamp_(-5, 5) # keeps sigmoid ~[0.0067, 0.9933], so memory doesnt vanish or freeze forever
            ʕっʘ‿ʘʔっ("longTermDecay")
            longDecay = torch.sigmoid(self.longTermDecay)
            with torch.no_grad(): self.longTermDecay.clamp_(-5, 5)

            ʕっʘ‿ʘʔっ("newShortTermMemory")
            newShort = (shortDecay * self.shortTermMemory) + ((1 - shortDecay) * self.activationsTensor)
            ʕっʘ‿ʘʔっ("newLongTermMemory")
            newLong  = (longDecay * self.longTermMemory) + ((1 - longDecay) * self.activationsTensor)

            reducedInput    = self.inputReducer(self.activationsTensor) # shape: (1, embedDimension)
            gateLogits      = self.gateLayer(reducedInput)              # shape: (1, 3 * numNeurons)
            gateLogits      = gateLogits.view(3, numNeurons)            # shape: (3, numNeurons)
            gateLogits      = gateLogits.clamp(-30, 30)                 # clamp
            gateWeights     = torch.softmax(gateLogits, dim=0)          # across gate types

            shortGateScale, longGateScale, actGateScale = gateWeights

            self.FINALmemory = ((shortGateScale * newShort) + (longGateScale * newLong) + (actGateScale * self.activationsTensor))

            self.shortGateScaleHistory.append(shortGateScale.mean().item())
            self.longGateScaleHistory.append(longGateScale.mean().item())
            self.activationsGateScaleHistory.append(actGateScale.mean().item())
            self.gateLayerHistory.append(self.gateLayer.weight.norm().item())

            self.rawActivationsHistory.append(self.activationsTensor.norm().item())
            self.shortTermMemoryHistory.append(self.shortTermMemory.norm().item())
            self.longTermMemoryHistory.append(self.longTermMemory.norm().item())
            self.FINALmemoryHistory.append(self.FINALmemory.norm().item())

            if len(self.shortGateScaleHistory) >= self.numTokensPerStep:
                self.stats = {
                    "4M_0_rawActivations_norm": sum(self.rawActivationsHistory) / len(self.rawActivationsHistory),

                    "4M_1_shortTermMemory_norm": sum(self.shortTermMemoryHistory) / len(self.shortTermMemoryHistory),
                    "4M_1_longTermMemory_norm": sum(self.longTermMemoryHistory) / len(self.longTermMemoryHistory),

                    "4M_gateLayer": sum(self.gateLayerHistory) / len(self.gateLayerHistory),
                    "4M_shortGateScale": sum(self.shortGateScaleHistory) / len(self.shortGateScaleHistory),
                    "4M_longGateScale": sum(self.longGateScaleHistory) / len(self.longGateScaleHistory),
                    "4M_activationsGateScale": sum(self.activationsGateScaleHistory) / len(self.activationsGateScaleHistory),
                    
                    "4M_x_FINALmemory_norm": sum(self.FINALmemoryHistory) / len(self.FINALmemoryHistory),

                    "4M_shortDecay": torch.sigmoid(self.shortTermDecay).item(),
                    "4M_longDecay": torch.sigmoid(self.longTermDecay).item(),
                }

                self.shortGateScaleHistory = []
                self.longGateScaleHistory = []
                self.activationsGateScaleHistory = []
                self.gateLayerHistory = []

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
            #with torch.no_grad():
                #self.longTermDecay += 0.1
                #self.shortTermDecay += 0.001
                #self.shortTermMemory = self.shortTermMemory * 0.1
                #self.longTermMemory.zero_() #retaining long term cause, yk, long term! i felt mean!
            pass

    def getMemoryStats(self): return self.stats

if __name__ == "__main__":
    memory = MEMORY(numNeurons = numNeurons)
    print("--- MEMORY TESTING STARTED ---")
    print("\n--- MEMORY TESTING COMPLETE ---")