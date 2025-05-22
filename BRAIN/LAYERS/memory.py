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

        self.inputReducer = nn.Linear(numNeurons, embedDimension, device = self.device)       # 10k → 1k
        self.gateLayer2     = nn.Linear(embedDimension, 4 * numNeurons, device = self.device)  # 1k → 30k

        self.memoryProjector = nn.Linear(numNeurons, embedDimension, device = self.device)
        self.memoryInfluence2 = torch.nn.Sequential(
                                nn.Linear(embedDimension, 512, device = self.device),   # bottleneck layer
                                nn.GELU(),                                              # smoother activation
                                nn.LayerNorm(512, device = self.device),                # mid normalization
                                nn.Linear(512, numNeurons, device = self.device),         # expand back
                                nn.LayerNorm(numNeurons, device = self.device)            # final safety net
                                )

        # buffers to store memory (outside gradient)
        self.register_buffer("shortTermMemory", torch.zeros(1, numNeurons))
        self.register_buffer("longTermMemory", torch.zeros(1, numNeurons))

        # stats
        self.shortGateScaleHistory = []
        self.longGateScaleHistory = []
        self.activationsGateScaleHistory = []
        self.gateLayer2History = []
        self.reducedInputHistory = []

        self.rawActivationsHistory = []
        self.shortTermMemoryHistory = []
        self.longTermMemoryHistory = []
        self.FINALmemoryHistory = []

        self.memGateScaleHistory = []
        self.projectedMemoryHistory = []
        self.memoryGateHistory = []
        self.mixedEmbedHistory = []

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

            reducedInput = self.inputReducer(self.activationsTensor)  # [1, embedDim]

            # unified gate logits -> shape: [1, 4 * numNeurons]
            gateLogits = self.gateLayer2(reducedInput).view(4, numNeurons)
            gateLogits = gateLogits.clamp(-30, 30)

            # softmax across sources (dim=0), sum to 1 per neuron
            gateWeights = torch.softmax(gateLogits, dim=0)
            shortGateScale, longGateScale, actGateScale, memGateScale = gateWeights

            firstGatedMemory = (
                (shortGateScale * newShort) +
                (longGateScale * newLong) +
                (actGateScale * self.activationsTensor)
            )

            projectedMemory = self.memoryProjector(firstGatedMemory)
            mixedEmbed = reducedInput + projectedMemory
            memoryGate = self.memoryInfluence2(mixedEmbed)

            self.gatedMemory = (
                (shortGateScale * newShort) +
                (longGateScale * newLong) +
                (actGateScale * self.activationsTensor) +
                (memGateScale * memoryGate)
            )
            self.FINALmemory = self.gatedMemory

            self.shortGateScaleHistory.append(shortGateScale.mean().item()) # 1
            self.longGateScaleHistory.append(longGateScale.mean().item()) # 2
            self.activationsGateScaleHistory.append(actGateScale.mean().item()) # 0
            self.memGateScaleHistory.append(memGateScale.mean().item()) # 7

            self.rawActivationsHistory.append(self.activationsTensor.norm().item()) # 0
            self.shortTermMemoryHistory.append(self.shortTermMemory.norm().item()) # 1

            self.longTermMemoryHistory.append(self.longTermMemory.norm().item()) # 2

            self.reducedInputHistory.append(reducedInput.norm().item()) # 3
            self.gateLayer2History.append(self.gateLayer2.weight.norm().item()) # 4

            self.projectedMemoryHistory.append(projectedMemory.norm().item()) # 5
            self.mixedEmbedHistory.append(mixedEmbed.norm().item()) # 6
            self.memoryGateHistory.append(memoryGate.norm().item()) # 7

            self.FINALmemoryHistory.append(self.FINALmemory.norm().item())

            if len(self.shortGateScaleHistory) >= self.numTokensPerStep:
                self.stats = {
                    "4M_0_rawActivations_norm": sum(self.rawActivationsHistory) / len(self.rawActivationsHistory),

                    "4M_1_shortTermMemory_norm": sum(self.shortTermMemoryHistory) / len(self.shortTermMemoryHistory),
                    "4M_2_longTermMemory_norm": sum(self.longTermMemoryHistory) / len(self.longTermMemoryHistory),

                    "4M_4_gateLayer": sum(self.gateLayer2History) / len(self.gateLayer2History),
                    "4M_1_shortGateScale": sum(self.shortGateScaleHistory) / len(self.shortGateScaleHistory),
                    "4M_2_longGateScale": sum(self.longGateScaleHistory) / len(self.longGateScaleHistory),
                    "4M_0_activationsGateScale": sum(self.activationsGateScaleHistory) / len(self.activationsGateScaleHistory),
                    "4M_7_memoryGateScale": sum(self.memGateScaleHistory) / len(self.memGateScaleHistory),
                    "4M_5_projectedMemory_norm": sum(self.projectedMemoryHistory) / len(self.projectedMemoryHistory),
                    "4M_7_memoryGate_norm": sum(self.memoryGateHistory) / len(self.memoryGateHistory),
                    "4M_6_mixedEmbed_norm": sum(self.mixedEmbedHistory) / len(self.mixedEmbedHistory),
                    
                    "4M_x_FINALmemory_norm": sum(self.FINALmemoryHistory) / len(self.FINALmemoryHistory),

                    "4M_1_shortDecay": torch.sigmoid(self.shortTermDecay).item(),
                    "4M_1_longDecay": torch.sigmoid(self.longTermDecay).item(),
                }

                self.shortGateScaleHistory = []
                self.longGateScaleHistory = []
                self.activationsGateScaleHistory = []
                self.gateLayer2History = []
                self.reducedInputHistory = []

                self.rawActivationsHistory = []
                self.shortTermMemoryHistory = []
                self.longTermMemoryHistory = []
                self.FINALmemoryHistory = []

                self.memGateScaleHistory = []
                self.projectedMemoryHistory = []
                self.memoryGateHistory = []
                self.mixedEmbedHistory = []

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