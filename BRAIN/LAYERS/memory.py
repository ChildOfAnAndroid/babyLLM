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
        self.stats = {}
        self.shortGateScaleHistory = []
        self.longGateScaleHistory = []
        self.activationsGateScaleHistory = []
        self.gateLayer2History = []
        self.gateLayer2MaxHistory = []
        self.gateLayer2MinHistory = []
        self.reducedInputHistory = []
        self.reducedInputMaxHistory = []
        self.reducedInputMinHistory = []

        self.rawActivationsHistory = []
        self.rawActivationsMaxHistory = []
        self.rawActivationsMinHistory = []
        self.shortTermMemoryHistory = []
        self.shortTermMemoryMaxHistory = []
        self.shortTermMemoryMinHistory = []
        self.longTermMemoryHistory = []
        self.longTermMemoryMaxHistory = []
        self.longTermMemoryMinHistory = []
        self.FINALmemoryHistory = []
        self.FINALmemoryMaxHistory = []
        self.FINALmemoryMinHistory = []

        self.memGateScaleHistory = []
        self.projectedMemoryHistory = []
        self.projectedMemoryMaxHistory = []
        self.projectedMemoryMinHistory = []
        self.memoryGateHistory = []
        self.memoryGateMaxHistory = []
        self.memoryGateMinHistory = []
        self.mixedEmbedHistory = []
        self.mixedEmbedMaxHistory = []
        self.mixedEmbedMinHistory = []
        self.gateLayer2NormHistory = []
        self.reducedInputNormHistory = []

        self.rawActivationsNormHistory = []
        self.shortTermMemoryNormHistory = []
        self.longTermMemoryNormHistory = []
        self.FINALmemoryNormHistory = []

        self.projectedMemoryNormHistory = []
        self.memoryGateNormHistory = []
        self.mixedEmbedNormHistory = []

    @whocalled
    def forward(self, _activationsTensor):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            self.activationsTensor = _activationsTensor

            if debugPrints: ʕっʘ‿ʘʔっ("shortTermDecay")
            shortDecay = torch.sigmoid(self.shortTermDecay)
            with torch.no_grad(): self.shortTermDecay.clamp_(-5, 5) # keeps sigmoid ~[0.0067, 0.9933], so memory doesnt vanish or freeze forever
            if debugPrints: ʕっʘ‿ʘʔっ("longTermDecay")
            longDecay = torch.sigmoid(self.longTermDecay)
            with torch.no_grad(): self.longTermDecay.clamp_(-5, 5)

            if debugPrints: ʕっʘ‿ʘʔっ("newShortTermMemory")
            newShort = (shortDecay * self.shortTermMemory) + ((1 - shortDecay) * self.activationsTensor)
            if debugPrints: ʕっʘ‿ʘʔっ("newLongTermMemory")
            newLong  = (longDecay * self.longTermMemory) + ((1 - longDecay) * self.activationsTensor)

            if debugPrints: ʕっʘ‿ʘʔっ("self.inputReducer")
            reducedInput = self.inputReducer(self.activationsTensor)  # [1, embedDim]

            # unified gate logits -> shape: [1, 4 * numNeurons]
            if debugPrints: ʕっʘ‿ʘʔっ("self.gateLater2")
            gateLogits = self.gateLayer2(reducedInput).view(4, numNeurons)
            if debugPrints: ʕっʘ‿ʘʔっ("clamp gatelayer2 -> gateLogits")
            gateLogits = gateLogits.clamp(-30, 30)

            # softmax across sources (dim=0), sum to 1 per neuron
            if debugPrints: ʕっʘ‿ʘʔっ("softmax gateLogits")
            gateWeights = torch.softmax(gateLogits, dim=0)
            shortGateScale, longGateScale, actGateScale, memGateScale = gateWeights

            if debugPrints: ʕっʘ‿ʘʔっ("firstGatedMemory")
            firstGatedMemory = (
                (shortGateScale * newShort) +
                (longGateScale * newLong) +
                (actGateScale * self.activationsTensor)
            )

            if debugPrints: ʕっʘ‿ʘʔっ("self.memoryProjector")
            projectedMemory = self.memoryProjector(firstGatedMemory)
            if debugPrints: ʕっʘ‿ʘʔっ("mix embeds")
            mixedEmbed = reducedInput + projectedMemory
            if debugPrints: ʕっʘ‿ʘʔっ("self.memoryInfluence2")
            memoryGate = self.memoryInfluence2(mixedEmbed)

            if debugPrints: ʕっʘ‿ʘʔっ("self.gatedMemory")
            self.gatedMemory = (
                (shortGateScale * newShort) +
                (longGateScale * newLong) +
                (actGateScale * self.activationsTensor) +
                (memGateScale * memoryGate)
            )
            self.FINALmemory = self.gatedMemory

            if debugPrints: ʕっʘ‿ʘʔっ("shortGateScale stats")
            self.shortGateScaleHistory.append(shortGateScale.mean().item()) # 1
            self.longGateScaleHistory.append(longGateScale.mean().item()) # 2
            self.activationsGateScaleHistory.append(actGateScale.mean().item()) # 0
            self.memGateScaleHistory.append(memGateScale.mean().item()) # 7

            if debugPrints: ʕっʘ‿ʘʔっ("raw activation stats")
            self.rawActivationsNormHistory.append(self.activationsTensor.norm().item()) # 0
            self.rawActivationsHistory.append(self.activationsTensor.mean().item()) # 0
            self.rawActivationsMaxHistory.append(self.activationsTensor.max().item()) # 0
            self.rawActivationsMinHistory.append(self.activationsTensor.min().item()) # 0

            """if debugPrints: ʕっʘ‿ʘʔっ("STM stats")
            self.shortTermMemoryNormHistory.append(self.shortTermMemory.norm().item()) # 1
            self.shortTermMemoryHistory.append(self.shortTermMemory.mean().item()) # 1
            self.shortTermMemoryMaxHistory.append(self.shortTermMemory.max().item()) # 1
            self.shortTermMemoryMinHistory.append(self.shortTermMemory.min().item()) # 1

            if debugPrints: ʕっʘ‿ʘʔっ("LTM stats")
            self.longTermMemoryNormHistory.append(self.longTermMemory.norm().item()) # 2
            self.longTermMemoryHistory.append(self.longTermMemory.mean().item()) # 2
            self.longTermMemoryMaxHistory.append(self.longTermMemory.max().item()) # 2
            self.longTermMemoryMinHistory.append(self.longTermMemory.min().item()) # 2

            if debugPrints: ʕっʘ‿ʘʔっ("reduced Input stats")
            self.reducedInputNormHistory.append(reducedInput.norm().item()) # 3
            self.reducedInputHistory.append(reducedInput.mean().item()) # 3
            self.reducedInputMaxHistory.append(reducedInput.max().item()) # 3
            self.reducedInputMinHistory.append(reducedInput.min().item()) # 3

            if debugPrints: ʕっʘ‿ʘʔっ("gate layer 2 stats")
            self.gateLayer2NormHistory.append(self.gateLayer2.weight.norm().item()) # 4
            self.gateLayer2History.append(self.gateLayer2.weight.mean().item()) # 4
            self.gateLayer2MaxHistory.append(self.gateLayer2.weight.max().item()) # 4
            self.gateLayer2MinHistory.append(self.gateLayer2.weight.min().item()) # 4

            if debugPrints: ʕっʘ‿ʘʔっ("projected memory stats")
            self.projectedMemoryNormHistory.append(projectedMemory.norm().item()) # 5
            self.projectedMemoryHistory.append(projectedMemory.mean().item()) # 5
            self.projectedMemoryMaxHistory.append(projectedMemory.max().item()) # 5
            self.projectedMemoryMinHistory.append(projectedMemory.min().item()) # 5

            if debugPrints: ʕっʘ‿ʘʔっ("mixed embed stats")
            self.mixedEmbedNormHistory.append(mixedEmbed.norm().item()) # 6
            self.mixedEmbedHistory.append(mixedEmbed.mean().item()) # 6
            self.mixedEmbedMaxHistory.append(mixedEmbed.max().item()) # 6
            self.mixedEmbedMinHistory.append(mixedEmbed.min().item()) # 6

            if debugPrints: ʕっʘ‿ʘʔっ("memory gate stats")
            self.memoryGateNormHistory.append(memoryGate.norm().item()) # 7
            self.memoryGateHistory.append(memoryGate.mean().item()) # 7
            self.memoryGateMaxHistory.append(memoryGate.max().item()) # 7
            self.memoryGateMinHistory.append(memoryGate.min().item()) # 7"""

            if debugPrints: ʕっʘ‿ʘʔっ("final memory stats")
            self.FINALmemoryNormHistory.append(self.FINALmemory.norm().item())
            self.FINALmemoryHistory.append(self.FINALmemory.mean().item())
            self.FINALmemoryMaxHistory.append(self.FINALmemory.max().item())
            self.FINALmemoryMinHistory.append(self.FINALmemory.min().item())

            if len(self.shortGateScaleHistory) >= self.numTokensPerStep:
                if debugPrints: ʕっʘ‿ʘʔっ("updateStats if short gate scale history >= self.numTokensPerStep")
                statShortDecay = torch.sigmoid(self.shortTermDecay).item()
                statLongDecay = torch.sigmoid(self.longTermDecay).item()
                self.stats = {
                    "4M_0_rawActs_norm": sum(self.rawActivationsNormHistory) / len(self.rawActivationsNormHistory),
                    "4M_0_rawActs_mean": sum(self.rawActivationsHistory) / len(self.rawActivationsHistory),
                    "4M_0_rawActs_max": sum(self.rawActivationsMaxHistory) / len(self.rawActivationsMaxHistory),
                    "4M_0_rawActs_min": sum(self.rawActivationsMinHistory) / len(self.rawActivationsMinHistory),

                    #"4M_1_STM_norm": sum(self.shortTermMemoryNormHistory) / len(self.shortTermMemoryNormHistory),
                    #"4M_1_STM_mean": sum(self.shortTermMemoryHistory) / len(self.shortTermMemoryHistory),
                    #"4M_1_STM_max": sum(self.shortTermMemoryMaxHistory) / len(self.shortTermMemoryMaxHistory),
                    #"4M_1_STM_min": sum(self.shortTermMemoryMinHistory) / len(self.shortTermMemoryMinHistory),

                    #"4M_2_LTM_norm": sum(self.longTermMemoryNormHistory) / len(self.longTermMemoryNormHistory),
                    #"4M_2_LTM_mean": sum(self.longTermMemoryHistory) / len(self.longTermMemoryHistory),
                    #"4M_2_LTM_max": sum(self.longTermMemoryMaxHistory) / len(self.longTermMemoryMaxHistory),
                    #"4M_2_LTM_min": sum(self.longTermMemoryMinHistory) / len(self.longTermMemoryMinHistory),

                    #"4M_3_reducedInput_norm": sum(self.reducedInputNormHistory) / len(self.reducedInputNormHistory),
                    #"4M_3_reducedInput_mean": sum(self.reducedInputHistory) / len(self.reducedInputHistory),
                    #"4M_3_reducedInput_max": sum(self.reducedInputMaxHistory) / len(self.reducedInputMaxHistory),
                    #"4M_3_reducedInput_min": sum(self.reducedInputMinHistory) / len(self.reducedInputMinHistory),

                    #"4M_4_gateLayer_norm": sum(self.gateLayer2NormHistory) / len(self.gateLayer2NormHistory),
                    #"4M_4_gateLayer_mean": sum(self.gateLayer2History) / len(self.gateLayer2History),
                    #"4M_4_gateLayer_max": sum(self.gateLayer2MaxHistory) / len(self.gateLayer2MaxHistory),
                    #"4M_4_gateLayer_min": sum(self.gateLayer2MinHistory) / len(self.gateLayer2MinHistory),

                    #"4M_5_projected_norm": sum(self.projectedMemoryNormHistory) / len(self.projectedMemoryNormHistory),
                    #"4M_5_projected_mean": sum(self.projectedMemoryHistory) / len(self.projectedMemoryHistory),
                    #"4M_5_projected_max": sum(self.projectedMemoryMaxHistory) / len(self.projectedMemoryMaxHistory),
                    #"4M_5_projected_min": sum(self.projectedMemoryMinHistory) / len(self.projectedMemoryMinHistory),

                    #"4M_6_mixedEmbed_norm": sum(self.mixedEmbedNormHistory) / len(self.mixedEmbedNormHistory),
                    #"4M_6_mixedEmbed_mean": sum(self.mixedEmbedHistory) / len(self.mixedEmbedHistory),
                    #"4M_6_mixedEmbed_max": sum(self.mixedEmbedMaxHistory) / len(self.mixedEmbedMaxHistory),
                    #"4M_6_mixedEmbed_min": sum(self.mixedEmbedMinHistory) / len(self.mixedEmbedMinHistory),

                    #"4M_7_memoryGate_norm": sum(self.memoryGateNormHistory) / len(self.memoryGateNormHistory),
                    #"4M_7_memoryGate_mean": sum(self.memoryGateHistory) / len(self.memoryGateHistory),
                    #"4M_7_memoryGate_max": sum(self.memoryGateMaxHistory) / len(self.memoryGateMaxHistory),
                    #"4M_7_memoryGate_min": sum(self.memoryGateMinHistory) / len(self.memoryGateMinHistory),

                    "4M_x_FINAL_norm": sum(self.FINALmemoryNormHistory) / len(self.FINALmemoryNormHistory),
                    "4M_x_FINAL_mean": sum(self.FINALmemoryHistory) / len(self.FINALmemoryHistory),
                    "4M_x_FINAL_max": sum(self.FINALmemoryMaxHistory) / len(self.FINALmemoryMaxHistory),
                    "4M_x_FINAL_min": sum(self.FINALmemoryMinHistory) / len(self.FINALmemoryMinHistory),

                    "4M_1_shortGateScale": sum(self.shortGateScaleHistory) / len(self.shortGateScaleHistory),
                    "4M_2_longGateScale": sum(self.longGateScaleHistory) / len(self.longGateScaleHistory),
                    "4M_0_actGateScale": sum(self.activationsGateScaleHistory) / len(self.activationsGateScaleHistory),
                    "4M_7_memoryGateScale": sum(self.memGateScaleHistory) / len(self.memGateScaleHistory),

                    "4M_1_shortDecay": statShortDecay,
                    "4M_1_longDecay": statLongDecay,
                }

                """if debugPrints: ʕっʘ‿ʘʔっ("clear stats")
                self.shortGateScaleHistory = []
                self.longGateScaleHistory = []
                self.activationsGateScaleHistory = []
                self.gateLayer2History = []
                self.gateLayer2MaxHistory = []
                self.gateLayer2MinHistory = []
                self.reducedInputHistory = []
                self.reducedInputMaxHistory = []
                self.reducedInputMinHistory = []

                self.rawActivationsHistory = []
                self.rawActivationsMaxHistory = []
                self.rawActivationsMinHistory = []
                self.shortTermMemoryHistory = []
                self.shortTermMemoryMaxHistory = []
                self.shortTermMemoryMinHistory = []
                self.longTermMemoryHistory = []
                self.longTermMemoryMaxHistory = []
                self.longTermMemoryMinHistory = []
                self.FINALmemoryHistory = []
                self.FINALmemoryMaxHistory = []
                self.FINALmemoryMinHistory = []

                self.memGateScaleHistory = []
                self.projectedMemoryHistory = []
                self.projectedMemoryMaxHistory = []
                self.projectedMemoryMinHistory = []
                self.memoryGateHistory = []
                self.memoryGateMaxHistory = []
                self.memoryGateMinHistory = []
                self.mixedEmbedHistory = []
                self.mixedEmbedMaxHistory = []
                self.mixedEmbedMinHistory = []
                self.gateLayer2NormHistory = []
                self.reducedInputNormHistory = []

                self.rawActivationsNormHistory = []
                self.shortTermMemoryNormHistory = []
                self.longTermMemoryNormHistory = []
                self.FINALmemoryNormHistory = []

                self.projectedMemoryNormHistory = []
                self.memoryGateNormHistory = []
                self.mixedEmbedNormHistory = []"""

            if debugPrints: ʕっʘ‿ʘʔっ("store computed memories for after backward")
            self.newShort = newShort
            self.newLong = newLong

            if debugPrints: ʕっʘ‿ʘʔっ("return finalmemory")
            return self.FINALmemory

    @whocalled
    def updateMemoryBuffers(self):
        with self.counsellor.infodump("updateMemoryBuffers") as ʕっʘ‿ʘʔっ:
            with torch.no_grad():
                if debugPrints: ʕっʘ‿ʘʔっ("self.shortTermMemory.copy_")
                self.shortTermMemory.copy_(self.newShort.detach())
                if debugPrints: ʕっʘ‿ʘʔっ("self.longTermMemory.copy_")
                self.longTermMemory.copy_(self.newLong.detach())

    @whocalled
    def resetMemory(self, _memoryLength):
        with self.counsellor.infodump("resetMemory") as ʕっʘ‿ʘʔっ:
            with torch.no_grad(): # Ensure all ops here are outside graph
                # Detach _resetStrength if it's a tensor from a graph
                current_reset_strength_tensor = _memoryLength.detach() if isinstance(_memoryLength, torch.Tensor) else torch.tensor(float(_memoryLength), device=self.device)
                reset_strength_scalar = current_reset_strength_tensor.item() if current_reset_strength_tensor.numel() == 1 else float(current_reset_strength_tensor) # Fallback if not scalar
                reset_strength_scalar = max(0.0, min(1.0, reset_strength_scalar))
                keep_factor = 1.0 - reset_strength_scalar
                print(f"resetting memory from {self.shortTermMemory.mean().item():.10f}", end = "")
                self.shortTermMemory.mul_(keep_factor) # In-place multiplication with a float
                # self.longTermMemory.mul_(keep_factor)
                print(f" to {self.shortTermMemory.mean().item():.10f}!")
                #self.longTermDecay += 0.1
                #self.shortTermDecay += 0.001
                #self.longTermMemory.zero_() #retaining long term cause, yk, long term! i felt mean!
            #pass

    @whocalled
    def getMemoryStats(self): return self.stats

if __name__ == "__main__":
    memory = MEMORY(numNeurons = numNeurons)
    print("--- MEMORY TESTING STARTED ---")
    print("\n--- MEMORY TESTING COMPLETE ---")