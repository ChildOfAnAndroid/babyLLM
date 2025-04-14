#CHARIS CAT 2025
# BABYLLM - memoryLayer.py

import torch
import torch.nn as nn
from config import *
from SCHOOL.staffroom.counsellor import *

"""this makes a rolling buffer of past activations"""
class MEMORY(nn.Module):
    def __init__(self, numNeurons=numNeurons):
        super().__init__()
        self.counsellor = COUNSELLOR("MEMORY", debug=debugPrints, durations=durationLogging)
        # Learnable decay rates
        self.shortTermDecay = nn.Parameter(torch.tensor(0.7, device = modelDevice))
        self.longTermDecay = nn.Parameter(torch.tensor(0.95, device = modelDevice))
        # gates for it to learn when to use the memory or not, learnable average
        self.shortGate = nn.Parameter(torch.tensor(0.25, device = modelDevice))
        self.longGate = nn.Parameter(torch.tensor(0.25, device = modelDevice))
        self.currentGate = nn.Parameter(torch.tensor(0.5, device = modelDevice))

        # Buffers to hold state outside graph
        self.register_buffer("shortTermMemory", torch.zeros(1, numNeurons))
        self.register_buffer("longTermMemory", torch.zeros(1, numNeurons))

    def forward(self, activationsTensor): # learns when to forget more or less
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            ʕっʘ‿ʘʔっ("tensorsToDevice")
            device = modelDevice
            activationsTensor = activationsTensor.to(device)

            ʕっʘ‿ʘʔっ("detach memories") # Get detached historical memory (not part of current graph)
            oldShort = self.shortTermMemory.detach()
            oldLong = self.longTermMemory.detach()
            actClone = activationsTensor.detach()

            ʕっʘ‿ʘʔっ("sigmoid gate decays") # make sure decay values stay within [0, 1] range
            shortDecay = torch.sigmoid(self.shortTermDecay)
            longDecay = torch.sigmoid(self.longTermDecay)

            ʕっʘ‿ʘʔっ("updateMemories (compute)") # Compute new memory (attached to current graph)

            ʕっʘ‿ʘʔっ("update no grad memory") # Update the state (detached, won’t break graph)
            with torch.no_grad():
                newShort = (shortDecay * oldShort) + ((1 - shortDecay) * actClone)
                newLong = (longDecay * oldLong) + ((1 - longDecay) * actClone)
                self.shortTermMemory.copy_(newShort.clone())
                self.longTermMemory.copy_(newLong.clone())
            #print("memory output requires_grad?", self.longTermMemory.requires_grad)

            ʕっʘ‿ʘʔっ("logGateSizes") # log the memory gate sizes
            gateSum = self.shortGate + self.longGate + self.currentGate + 1e-9
            self.latestMemoryGates = torch.stack([
                (self.shortGate / gateSum),
                (self.longGate / gateSum),
                (self.currentGate / gateSum)
            ])

            ʕっʘ‿ʘʔっ("blendMemories") # Blend memories, weighted sum, hopefully gradient safe lol
            blendedAct = (
                ((shortDecay * newShort) +
                (longDecay * newLong)) +
                activationsTensor  # keep original graph for this one
            )

            return blendedAct

    def resetMemory(self):
        with self.counsellor.infodump("resetMemory") as ʕっʘ‿ʘʔっ:
            with torch.no_grad():
                ʕっʘ‿ʘʔっ("shortTerm")
                self.shortTermMemory.zero_()
                ʕっʘ‿ʘʔっ("longTerm")
                self.longTermMemory.zero_()

if __name__ == "__main__":
    memory = MEMORY(numNeurons = numNeurons)
    print("--- MEMORY TESTING STARTED ---")
    print("\n--- MEMORY TESTING COMPLETE ---")