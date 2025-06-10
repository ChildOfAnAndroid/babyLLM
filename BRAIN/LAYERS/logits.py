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

        self.tensorNormHist = []
        self.normedNormHist = []
        self.activNormHist = []
        self.logitNormHist = []
        self.normLayerNormHist = []
        self.finalNormHist = []

        self.tensorHist = []
        self.normedHist = []
        self.activHist = []
        self.logitHist = []
        self.normLayerHist = []
        self.finalHist = []

        self.tensorMinHist = []
        self.normedMinHist = []
        self.activMinHist = []
        self.logitMinHist = []
        self.normLayerMinHist = []
        self.finalMinHist = []

        self.tensorMaxHist = []
        self.normedMaxHist = []
        self.activMaxHist = []
        self.logitMaxHist = []
        self.normLayerMaxHist = []
        self.finalMaxHist = []

    @whocalled
    def forward(self, _meanActivationsTensor):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            # <- = from
            # INN? -> L1 -> L2 -> L3 -> L4 -> L5 -> L6 -> *
            """imports the activations from interneuronNetwork, assuming that is is a tensor"""
            if debugPrints: ʕっʘ‿ʘʔっ("L1: activationsTensor") # <- INN? no? seems to come from babyLLM? maybe through babyLLM?
            actsTensor = _meanActivationsTensor # _1
            rawActScale = torch.sigmoid(self.rawActivationsScale)
            normActScale = torch.sigmoid(self.normedActivationsScale)
            if debugPrints: ʕっʘ‿ʘʔっ("L2: normedActivationsTensor") # <- L1
            normedActsTensor = self.activationNorm(actsTensor) # _2
            scaledActs = (actsTensor * rawActScale + normedActsTensor * normActScale)
            scaledActs = scaledActs.clamp(-10, 10)

            if debugPrints: print(f"Debug logits: activations shape before @ weights: {scaledActs.shape}")
            if debugPrints: print(f"Debug logits: weights shape: {self.l_weights.shape}")
                        
            if debugPrints: ʕっʘ‿ʘʔっ("L3: scaledActivations") # <- L1 + L2
            logitOutput = (scaledActs @ self.l_weights / (numNeurons ** 0.5)) + self.l_bias
            logitOutput = logitOutput.clamp(-60, 60)
            #logitNormed = (scaledActs @ self.l_weights) / (numNeurons ** 0.5) + self.l_bias
            logitNormed = self.logitNorm(logitOutput)  # softly smooth

            if debugPrints: ʕっʘ‿ʘʔっ("L4: logitOutput") # <- L3 (with weights and bias)
            outScale = torch.sigmoid(self.outputScale)
            normOutScale = torch.sigmoid(self.normOutputScale)
            finalLogit = (logitOutput * outScale) + (logitNormed * normOutScale)
            if debugPrints: print(f"Debug logits: logitOutput shape AFTER @ weights: {logitOutput.shape}")

            if debugPrints: ʕっʘ‿ʘʔっ("clamp scalar parameters")
            with torch.no_grad():
                self.rawActivationsScale.data.clamp_(-10, 10)
                self.normedActivationsScale.data.clamp_(-10, 10)
                self.outputScale.data.clamp_(-10, 10)
                self.normOutputScale.data.clamp_(-10, 10)

            if debugPrints: ʕっʘ‿ʘʔっ("append rolling self.stats")
            self.tensorNormHist.append(actsTensor.norm().item())
            #self.normedNormHist.append(normedActsTensor.norm().item())
            #self.activNormHist.append(scaledActs.norm().item())
            #self.logitNormHist.append(logitOutput.norm().item())
            #self.normLayerNormHist.append(logitNormed.norm().item())
            self.finalNormHist.append(finalLogit.norm().item())

            self.tensorHist.append(actsTensor.mean().item())
            #self.normedHist.append(normedActsTensor.mean().item())
            #self.activHist.append(scaledActs.mean().item())
            #self.logitHist.append(logitOutput.mean().item())
            #self.normLayerHist.append(logitNormed.mean().item())
            self.finalHist.append(finalLogit.mean().item())

            self.tensorMinHist.append(actsTensor.min().item())
            #self.normedMinHist.append(normedActsTensor.min().item())
            #self.activMinHist.append(scaledActs.min().item())
            #self.logitMinHist.append(logitOutput.min().item())
            #self.normLayerMinHist.append(logitNormed.min().item())
            self.finalMinHist.append(finalLogit.min().item())

            self.tensorMaxHist.append(actsTensor.max().item())
            #self.normedMaxHist.append(normedActsTensor.max().item())
            #self.activMaxHist.append(scaledActs.max().item())
            #self.logitMaxHist.append(logitOutput.max().item())
            #self.normLayerMaxHist.append(logitNormed.max().item())
            self.finalMaxHist.append(finalLogit.max().item())

            if len(self.tensorHist) >= self.numTokensPerStep:
                if debugPrints: ʕっʘ‿ʘʔっ("clear rolling self.stats at end of window")
                self.stats = {
                    "6L_0_actsTensor_norm": sum(self.tensorNormHist) / len(self.tensorNormHist),
                    #"6L_1_normActsTensor_norm": sum(self.normedNormHist) / len(self.normedNormHist),
                    #"6L_2_scaledActsTensor_norm": sum(self.activNormHist) / len(self.activNormHist),
                    #"6L_3_out_norm": sum(self.logitNormHist) / len(self.logitNormHist),
                    #"6L_4_outNorm_norm": sum(self.normLayerNormHist) / len(self.normLayerNormHist),
                    "6L_x_final_norm": sum(self.finalNormHist) / len(self.finalNormHist),

                    "6L_0_actsTensor_mean": sum(self.tensorHist) / len(self.tensorHist),
                    #"6L_1_normActsTensor_mean": sum(self.normedHist) / len(self.normedHist),
                    #"6L_2_scaledActsTensor_mean": sum(self.activHist) / len(self.activHist),
                    #"6L_3_out_mean": sum(self.logitHist) / len(self.logitHist),
                    #"6L_4_outNorm_mean": sum(self.normLayerHist) / len(self.normLayerHist),
                    "6L_x_final_mean": sum(self.finalHist) / len(self.finalHist),

                    "6L_0_actsTensor_min": sum(self.tensorMinHist) / len(self.tensorMinHist),
                    #"6L_1_normActsTensor_min": sum(self.normedMinHist) / len(self.normedMinHist),
                    #"6L_2_scaledActsTensor_min": sum(self.activMinHist) / len(self.activMinHist),
                    #"6L_3_out_min": sum(self.logitMinHist) / len(self.logitMinHist),
                    #"6L_4_outNorm_min": sum(self.normLayerMinHist) / len(self.normLayerMinHist),
                    "6L_x_final_min": sum(self.finalMinHist) / len(self.finalMinHist),

                    "6L_0_actsTensor_max": sum(self.tensorMaxHist) / len(self.tensorMaxHist),
                    #"6L_1_normActsTensor_max": sum(self.normedMaxHist) / len(self.normedMaxHist),
                    #"6L_2_scaledActsTensor_max": sum(self.activMaxHist) / len(self.activMaxHist),
                    #"6L_3_out_max": sum(self.logitMaxHist) / len(self.logitMaxHist),
                    #"6L_4_outNorm_max": sum(self.normLayerMaxHist) / len(self.normLayerMaxHist),
                    "6L_x_final_max": sum(self.finalMaxHist) / len(self.finalMaxHist),

                }

                self.tensorNormHist = []
                self.normedNormHist = []
                self.activNormHist = []
                self.logitNormHist = []
                self.normLayerNormHist = []
                self.finalNormHist = []

                self.tensorHist = []
                self.normedHist = []
                self.activHist = []
                self.logitHist = []
                self.normLayerHist = []
                self.finalHist = []

                self.tensorMinHist = []
                self.normedMinHist = []
                self.activMinHist = []
                self.logitMinHist = []
                self.normLayerMinHist = []
                self.finalMinHist = []

                self.tensorMaxHist = []
                self.normedMaxHist = []
                self.activMaxHist = []
                self.logitMaxHist = []
                self.normLayerMaxHist = []
                self.finalMaxHist = []

            #with torch.no_grad():
                #topValues, topIndices = torch.topk(finalLogit, 5)
                #self.stats["6L_topLogits"] = topValues.tolist()
                #self.stats["6L_topIndices"] = topIndices.tolist()
                #self.stats["6L_logitMax"] = finalLogit.max().item()
                #self.stats["6L_logitMin"] = finalLogit.min().item()
                #self.stats["6L_logitMean"] = finalLogit.mean().item()
                #self.stats["6L_logitStd"] = finalLogit.std().item()
                #self.stats["6L_0_actsTensor_scale"] = rawScale.item()
                #self.stats["6L_1_normActsTensor_scale"] = normedScale.item()
                #self.stats["6L_3_outSigmoid_scale"] = outScale.detach().item()
                #self.stats["6L_4_outNormSigmoid_scale"] = normOutScale.detach().item()
            if debugPrints: print("activation norm:", scaledActs.norm().item())
            if debugPrints: print("weight norm mean:", self.l_weights.norm(dim = 0).mean().item())
            if debugPrints: print("weight norm max:", self.l_weights.norm(dim = 0).max().item())

            # return logits (not softmax) for better gradient computation in cross-entropy loss
            return finalLogit # L6 ->
    
    @whocalled
    def getLogitStats(self):
        with self.counsellor.infodump("getLogitStats") as ʕっʘ‿ʘʔっ:
            """with torch.no_grad():
                if debugPrints: ʕっʘ‿ʘʔっ("weightNormStats")
                weightNorms = torch.norm(self.l_weights.detach(), dim = 0)
                self.stats["logitWeightNormMean"] = weightNorms.mean().item()
                self.stats["logitWeightNormStd"] = weightNorms.std().item()
                self.stats["logitWeightNormMax"] = weightNorms.max().item()

                # scales (dont need on per token history as only updated in backward)
                self.stats["6L_0_actsTensor_scale"] = self.rawActivationsScale.norm().item()
                self.stats["6L_1_normActsTensor_scale"] = self.normedActivationsScale.norm().item()
                self.stats["6L_3_out_scale"] = self.outputScale.detach().norm().item()
                self.stats["6L_4_normOut_scale"] = self.normOutputScale.detach().norm().item()

                if debugPrints: ʕっʘ‿ʘʔっ("sparsityStat")
                sparsity = (self.l_weights.detach().abs() < 1e-5).float().mean().item()
                self.stats["logitWeightSparsity"] = sparsity

                if debugPrints: ʕっʘ‿ʘʔっ("weightDriftStat")
                drift = torch.norm(self.l_weights.detach() - self.lastSavedWeights)
                self.stats["logitWeightDrift"] = drift
                self.lastSavedWeights = self.l_weights.clone().detach()

                if debugPrints: ʕっʘ‿ʘʔっ("biasStats")
                self.stats["logitBiasMean"] = self.l_bias.mean().item()
                self.stats["logitBiasStd"] = self.l_bias.std().item()
                self.stats["logitBiasMax"] = self.l_bias.max().item()

                if hasattr(self, 'latestActivations'):
                    if debugPrints: ʕっʘ‿ʘʔっ("activationStats")
                    act = self.latestActivations
                    self.stats["activationStd"] = act.std().item()
                    self.stats["activationMean"] = act.mean().item()
                    self.stats["activationMax"] = act.max().item()
                    self.stats["activationMin"] = act.min().item()
                    self.stats["activationSparsity"] = (act.abs() < 1e-6).float().mean().item()"""

        return self.stats

    @whocalled
    def clearStats(self):
        for attr in list(vars(self)):
            if attr.endswith("History") or attr.endswith("Hist"):
                setattr(self, attr, [])

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