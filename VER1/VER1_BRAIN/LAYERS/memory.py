#CHARIS CAT 2025
# BABYLLM - memoryLayer.py

import torch
import torch.nn as nn
from VER1_config import *
from VER1_SCHOOL.staffroom.counsellor import COUNSELLOR

"""this makes a rolling buffer of past activations"""
class MEMORY(nn.Module):
    def __init__(self):
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
        self.lastInput = None  # for after backward
        
    def forward(self, activationsTensor):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            ʕっʘ‿ʘʔっ("tensorsToDevice")
            device = modelDevice

            ʕっʘ‿ʘʔっ("detach memories")
            oldShort = self.shortTermMemory.detach()
            oldLong = self.longTermMemory.detach()

            ʕっʘ‿ʘʔっ("sigmoid gate decays")
            shortDecay = torch.sigmoid(self.shortTermDecay)
            longDecay = torch.sigmoid(self.longTermDecay)

            ʕっʘ‿ʘʔっ("logGateSizes")
            gateSum = self.shortGate + self.longGate + self.currentGate + 1e-9
            shortGateNorm = self.shortGate / gateSum
            longGateNorm = self.longGate / gateSum
            currentGateNorm = self.currentGate / gateSum

            newShort = (shortDecay * oldShort) + ((1 - shortDecay) * activationsTensor.detach())
            newLong = (longDecay * oldLong) + ((1 - longDecay) * activationsTensor.detach())

            blendedAct = (
                (shortGateNorm * newShort) +
                (longGateNorm * newLong) +
                (currentGateNorm * activationsTensor)
            )

            self.latestMemoryGates = torch.stack([
                shortGateNorm,
                longGateNorm,
                currentGateNorm
            ])

            self.lastInput = activationsTensor.detach()

            return blendedAct
        
    def updateMemoryBuffers(self):
        with self.counsellor.infodump("updateMemoryBuffers") as ʕっʘ‿ʘʔっ:
            if self.lastInput is None:
                return  # Nothing to update from

            shortDecay = torch.sigmoid(self.shortTermDecay)
            longDecay = torch.sigmoid(self.longTermDecay)

            with torch.no_grad():
                newShort = (shortDecay * self.shortTermMemory) + ((1 - shortDecay) * self.lastInput)
                newLong  = (longDecay * self.longTermMemory) + ((1 - longDecay) * self.lastInput)

                self.shortTermMemory.copy_(newShort)
                self.longTermMemory.copy_(newLong)


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