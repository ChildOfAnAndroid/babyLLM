# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# OUTPUT LAYER FOR LOGIT PREDICTION
# BRAIN/LAYERS/logits.py

import torch
import torch.nn as nn
from config import *

"""final layer, maps neuron activations to logits for each token in the vocab"""
class LOGITS(nn.Module):
    def __init__(self, _counsellor, _device):
        super().__init__()
        self.device = _device
        self.counsellor = _counsellor
        self.lastSavedWeights = 0 # for stats

        self.l_weights = nn.Parameter(torch.randn(numNeurons, vocabSize, device = self.device)) # this is set to move the NEURON ACTIVATIONS (10000) onto VOCAB SIZE (2000)
        self.l_bias = nn.Parameter(torch.zeros(vocabSize, device = self.device))

    def forward(self, _meanActivationsTensor):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            """imports the activations from interneuronNetwork, assuming that is is a tensor"""
            activationsTensor = _meanActivationsTensor
            #activationsTensor = activationsTensor.to(self.device)
            if debugPrints: print(f"Debug logits: activationsTensor shape before @ weights: {activationsTensor.shape}")
            if debugPrints: print(f"Debug logits: weights shape: {self.l_weights.shape}")
            """return logits (not softmax) for better gradient computation in cross-entropy loss"""
            logitOutputNormalized = (activationsTensor @ self.l_weights) / (numNeurons ** 0.5) + self.l_bias
            logitOutputOriginal = activationsTensor @ self.l_weights + self.l_bias
            logitOutput = (logitOutputOriginal + logitOutputNormalized)/2
            if debugPrints: print(f"Debug logits: logitOutput shape AFTER @ weights: {logitOutput.shape}")
            return logitOutput
    
    def getLogitStats(self):
        with self.counsellor.infodump("getLogitStats") as ʕっʘ‿ʘʔっ:
            with torch.no_grad():
                stats = {}
                ʕっʘ‿ʘʔっ("weightNormStats")
                weightNorms = torch.norm(self.l_weights, dim = 0)
                stats["logitWeightNormMean"] = weightNorms.mean()
                stats["logitWeightNormStd"] = weightNorms.std()
                stats["logitWeightNormMax"] = weightNorms.max()

                ʕっʘ‿ʘʔっ("sparsityStat")
                sparsity = (self.l_weights.abs() < 1e-5).float().mean()
                stats["logitWeightSparsity"] = sparsity

                ʕっʘ‿ʘʔっ("weightDriftStat")
                drift = torch.norm(self.l_weights - self.lastSavedWeights)
                stats["logitWeightDrift"] = drift
                self.lastSavedWeights = self.l_weights.clone().detach()

                ʕっʘ‿ʘʔっ("biasStats")
                stats["logitBiasMean"] = self.l_bias.mean()
                stats["logitBiasStd"] = self.l_bias.std()
                stats["logitBiasMax"] = self.l_bias.max()

                if hasattr(self, 'latestActivations'):
                    ʕっʘ‿ʘʔっ("activationStats")
                    act = self.latestActivations
                    stats["activationStd"] = act.std()
                    stats["activationMean"] = act.mean()
                    stats["activationMax"] = act.max()
                    stats["activationMin"] = act.min()
                    stats["activationSparsity"] = (act.abs() < 1e-6).float().mean()

        return stats

if __name__ == "__main__":
    TESTlayerActivations = torch.randn(numNeurons)

    logits = LOGITS(numNeurons = numNeurons, vocabSize = vocabSize)
    logitOutput = logits.forward(TESTlayerActivations)

    print("--- LOGITS TESTING START ---")
    print(f"Output Layer created with {logits.vocabSize} vocabulary tokens.")
    print(f"Weight matrix shape: {logits.weights.shape}")
    print(f"Bias vector shape: {logits.bias.shape}")
    print(f"Logits (first 100):")
    print(logitOutput[:10])
    print(f"Logits Shape: {logitOutput.shape}")
    print("--- LOGITS TESTING COMPLETE ---")