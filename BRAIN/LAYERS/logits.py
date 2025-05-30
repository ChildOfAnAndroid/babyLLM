# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# OUTPUT LAYER FOR LOGIT PREDICTION
# BRAIN/LAYERS/logits.py

import torch
import torch.nn as nn
from config import *

"""final layer, maps neuron activations to logits for each token in the vocab"""
class LOGITS(nn.Module):
    def __init__(self, _counsellor, _device, _numTokensPerStep):
        super().__init__()
        self.device = _device
        self.counsellor = _counsellor
        self.numTokensPerStep = _numTokensPerStep
        self.lastSavedWeights = 0 # for stats

        self.l_weights = nn.Parameter(torch.randn(numNeurons, vocabSize, device = self.device)) # this is set to move the NEURON ACTIVATIONS (10000) onto VOCAB SIZE (2000)
        self.l_bias = nn.Parameter(torch.zeros(vocabSize, device = self.device))
        self.activationNorm = nn.LayerNorm(numNeurons, device = self.device)
        self.rawActivationsScale = nn.Parameter(torch.tensor(0.5)) 
        self.normedActivationsScale = nn.Parameter(torch.tensor(0.5)) 

        self.logitNorm = nn.LayerNorm(vocabSize, device = self.device)
        self.outputScale = nn.Parameter(torch.tensor(0.5)) 
        self.normOutputScale = nn.Parameter(torch.tensor(0.5)) 

        self.stats = {}
        self.tensorHist = []
        self.normedHist = []
        self.activHist = []
        self.logitHist = []
        self.logitNormHist = []
        self.finalLogitHist = []

    @whocalled
    def forward(self, _meanActivationsTensor):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            # <- = from
            # INN? -> L1 -> L2 -> L3 -> L4 -> L5 -> L6 -> *
            """imports the activations from interneuronNetwork, assuming that is is a tensor"""
            ʕっʘ‿ʘʔっ("L1: activationsTensor") # <- INN? no? seems to come from babyLLM? maybe through babyLLM?
            self.activationsTensor = _meanActivationsTensor # _1
            rawActScale = torch.sigmoid(self.rawActivationsScale)
            normActScale = torch.sigmoid(self.normedActivationsScale)
            ʕっʘ‿ʘʔっ("L2: normedActivationsTensor") # <- L1
            self.normedActivationsTensor = self.activationNorm(self.activationsTensor) # _2
            self.scaledActivations = (self.activationsTensor * rawActScale + self.normedActivationsTensor * normActScale)
            self.scaledActivations = self.scaledActivations.clamp(-10, 10)

            if debugPrints: print(f"Debug logits: activations shape before @ weights: {self.scaledActivations.shape}")
            if debugPrints: print(f"Debug logits: weights shape: {self.l_weights.shape}")
                        
            ʕっʘ‿ʘʔっ("L3: scaledActivations") # <- L1 + L2
            logitOutput = (self.scaledActivations @ self.l_weights / (numNeurons ** 0.5)) + self.l_bias
            logitOutput = logitOutput.clamp(-60, 60)
            #logitNormed = (self.scaledActivations @ self.l_weights) / (numNeurons ** 0.5) + self.l_bias
            logitNormed = self.logitNorm(logitOutput)  # softly smooth

            ʕっʘ‿ʘʔっ("L4: logitOutput") # <- L3 (with weights and bias)
            outScale = torch.sigmoid(self.outputScale)
            normOutScale = torch.sigmoid(self.normOutputScale)
            self.finalLogit = (logitOutput * outScale) + (logitNormed * normOutScale)
            if debugPrints: print(f"Debug logits: logitOutput shape AFTER @ weights: {logitOutput.shape}")

            ʕっʘ‿ʘʔっ("clamp scalar parameters")
            with torch.no_grad():
                self.rawActivationsScale.data.clamp_(-10, 10)
                self.normedActivationsScale.data.clamp_(-10, 10)
                self.outputScale.data.clamp_(-10, 10)
                self.normOutputScale.data.clamp_(-10, 10)

            ʕっʘ‿ʘʔっ("append rolling self.stats")
            self.tensorHist.append(self.activationsTensor.norm().item())
            self.normedHist.append(self.normedActivationsTensor.norm().item())
            self.activHist.append(self.scaledActivations.norm().item())
            self.logitHist.append(logitOutput.norm().item())
            self.logitNormHist.append(logitNormed.norm().item())
            self.finalLogitHist.append(self.finalLogit.norm().item())

            if len(self.tensorHist) >= self.numTokensPerStep:
                ʕっʘ‿ʘʔっ("clear rolling self.stats at end of window")
                self.stats = {
                    "6L_0_activationsTensor_norm": sum(self.tensorHist) / len(self.tensorHist),
                    "6L_1_normedActivationsTensor_norm": sum(self.normedHist) / len(self.normedHist),
                    "6L_2_scaledActivations_norm": sum(self.activHist) / len(self.activHist),
                    "6L_3_out_norm": sum(self.logitHist) / len(self.logitHist),
                    "6L_4_outNorm_norm": sum(self.logitNormHist) / len(self.logitNormHist),
                    "6L_x_finalLogit_norm": sum(self.finalLogitHist) / len(self.finalLogitHist),
                }
                self.tensorHist = []
                self.normedHist = []
                self.activHist = []
                self.logitHist = []
                self.logitNormHist = []
                self.finalLogitHist = []

            with torch.no_grad():
                #topValues, topIndices = torch.topk(self.finalLogit, 5)
                #self.stats["6L_topLogits"] = topValues.tolist()
                #self.stats["6L_topIndices"] = topIndices.tolist()
                #self.stats["6L_logitMax"] = self.finalLogit.max().item()
                #self.stats["6L_logitMin"] = self.finalLogit.min().item()
                #self.stats["6L_logitMean"] = self.finalLogit.mean().item()
                #self.stats["6L_logitStd"] = self.finalLogit.std().item()
                #self.stats["6L_0_activationsTensor_scale"] = rawScale.item()
                #self.stats["6L_1_normedActivationsTensor_scale"] = normedScale.item()
                self.stats["6L_3_outSigmoid_scale"] = outScale.detach().item()
                self.stats["6L_4_outNormSigmoid_scale"] = normOutScale.detach().item()
            if debugPrints: print("activation norm:", self.scaledActivations.norm().item())
            if debugPrints: print("weight norm mean:", self.l_weights.norm(dim=0).mean().item())
            if debugPrints: print("weight norm max:", self.l_weights.norm(dim=0).max().item())

            # return logits (not softmax) for better gradient computation in cross-entropy loss
            return self.finalLogit # L6 ->
    
    def getLogitStats(self):
        with self.counsellor.infodump("getLogitStats") as ʕっʘ‿ʘʔっ:
            with torch.no_grad():
                ʕっʘ‿ʘʔっ("weightNormStats")
                weightNorms = torch.norm(self.l_weights, dim = 0)
                self.stats["logitWeightNormMean"] = weightNorms.mean()
                self.stats["logitWeightNormStd"] = weightNorms.std()
                self.stats["logitWeightNormMax"] = weightNorms.max()

                # scales (dont need on per token history as only updated in backward)
                self.stats["6L_0_activationsTensor_scale"] = self.rawActivationsScale.norm().item()
                self.stats["6L_1_normedActivationsTensor_scale"] = self.normedActivationsScale.norm().item()
                self.stats["6L_3_out_scale"] = self.outputScale.detach().norm().item()
                self.stats["6L_4_normOut_scale"] = self.normOutputScale.detach().norm().item()

                ʕっʘ‿ʘʔっ("sparsityStat")
                sparsity = (self.l_weights.abs() < 1e-5).float().mean()
                self.stats["logitWeightSparsity"] = sparsity

                ʕっʘ‿ʘʔっ("weightDriftStat")
                drift = torch.norm(self.l_weights - self.lastSavedWeights)
                self.stats["logitWeightDrift"] = drift
                self.lastSavedWeights = self.l_weights.clone().detach()

                ʕっʘ‿ʘʔっ("biasStats")
                self.stats["logitBiasMean"] = self.l_bias.mean()
                self.stats["logitBiasStd"] = self.l_bias.std()
                self.stats["logitBiasMax"] = self.l_bias.max()

                if hasattr(self, 'latestActivations'):
                    ʕっʘ‿ʘʔっ("activationStats")
                    act = self.latestActivations
                    self.stats["activationStd"] = act.std()
                    self.stats["activationMean"] = act.mean()
                    self.stats["activationMax"] = act.max()
                    self.stats["activationMin"] = act.min()
                    self.stats["activationSparsity"] = (act.abs() < 1e-6).float().mean()

        return self.stats

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