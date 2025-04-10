# CHARIS CAT 2025
# BABYLLM - outputLayer.py

import torch
import torch.nn as nn
from config import *

"""final layer, maps neuron activations to logits for each token in the vocab"""
class OUTPUTLAYER(nn.Module):
    def __init__(self, numNeurons, vocabSize):
        super().__init__()
        self.numNeurons = numNeurons
        self.vocabSize = vocabSize
        self.weights = nn.Parameter(torch.randn(numNeurons, vocabSize, device = modelDevice)) # this is set to move the NEURON ACTIVATIONS (10000) onto VOCAB SIZE (2000)
        self.weights.data *= 0.01
        self.bias = nn.Parameter(torch.zeros(vocabSize, device = modelDevice))

    def forward(self, meanActivationsTensor):
        """imports the activations from parallelNeuronLayer, assuming that is is a tensor"""
        activationsTensor = meanActivationsTensor
        activationsTensor = activationsTensor.to(self.weights.device)
        """DEBUG PRINTS"""
        #print(f"Debug outputLayer: activationsTensor shape before @ weights: {self.activationsTensor.shape}")
        #print(f"Debug outputLayer: weights shape: {self.weights.shape}")
        """return logits (not softmax) for better gradient computation in cross-entropy loss"""
        logits = activationsTensor @ self.weights + self.bias
        return logits
    
    def getOutputStats(self):
        with torch.no_grad():
            stats = {}
            # weight stats
            #weightNorms = torch.norm(self.weights, dim=0)
            #stats["logitWeightNormMean"] = weightNorms.mean().item()
            #stats["logitWeightNormStd"] = weightNorms.std().item()
            #stats["logitWeightNormMax"] = weightNorms.max().item()

            # sparsity
            #sparsity = (self.weights.abs() < 1e-5).float().mean().item()
            #stats["logitWeightSparsity"] = sparsity

            #drift = torch.norm(self.weights - self.lastSavedWeights).item()
            #stats["logitWeightDrift"] = drift
            #self.lastSavedWeights = self.weights.clone().detach()

            # bias stats
            #stats["logitBiasMean"] = self.bias.mean().item()
            #stats["logitBiasStd"] = self.bias.std().item()
            #stats["logitBiasMax"] = self.bias.max().item()

            # Activation stats
            if hasattr(self, 'latestActivations'):
                act = self.latestActivations
                #stats["activationStd"] = act.std().item()
                #stats["activationMean"] = act.mean().item()
                stats["activationMax"] = act.max().item()
                stats["activationMin"] = act.min().item()
                #stats["activationSparsity"] = (act.abs() < 1e-6).float().mean().item()


            return stats

    
if __name__ == "__main__":
    TESTlayerActivations = torch.randn(numNeurons)

    outputLayer = OUTPUTLAYER(numNeurons, vocabSize = vocabSize)
    logits = outputLayer.forward(TESTlayerActivations)

    print("--- OUTPUT LAYER TESTING START ---")
    print(f"Output Layer created with {outputLayer.vocabSize} vocabulary tokens.")
    print(f"Weight matrix shape: {outputLayer.weights.shape}")
    print(f"Bias vector shape: {outputLayer.bias.shape}")
    print(f"Logits (first 100):")
    print(logits[:10])
    print(f"Logits Shape: {logits.shape}")
    print("--- OUTPUT LAYER TESTING COMPLETE ---")