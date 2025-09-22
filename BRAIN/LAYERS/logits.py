# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# OUTPUT LAYER FOR LOGIT PREDICTION
# brain/LAYERS/logits.py
# v4.2

import torch
import torch.nn as nn
from collections import deque
from config import *
import torch.nn.functional as F
from helpers import clamp_param

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

        self.stats = {}
        self._history_attrs = [
            "tensorNormHist",
            "normedNormHist",
            "activNormHist",
            "logitNormHist",
            "normLayerNormHist",
            "finalNormHist",
            "tensorHist",
            "normedHist",
            "activHist",
            "logitHist",
            "normLayerHist",
            "finalHist",
            "tensorMinHist",
            "normedMinHist",
            "activMinHist",
            "logitMinHist",
            "normLayerMinHist",
            "finalMinHist",
            "tensorMaxHist",
            "normedMaxHist",
            "activMaxHist",
            "logitMaxHist",
            "normLayerMaxHist",
            "finalMaxHist",
        ]
        self._init_history_buffers()

    def _init_history_buffers(self):
        maxlen = max(1, self.numTokensPerStep)
        for attr in self._history_attrs:
            setattr(self, attr, deque(maxlen=maxlen))

    @whocalled
    def forward(self, _meanActivationsTensor):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            # <- = from
            # INN? -> L1 -> L2 -> L3 -> L4 -> L5 -> L6 -> *
            """imports the activations from interneuronNetwork, assuming that is is a tensor"""
            if debugPrints: ʕっʘ‿ʘʔっ("L1: activationsTensor") # <- INN? no? seems to come from babyLLM? maybe through babyLLM?
            actsTensor = _meanActivationsTensor # _1
            #rawActScale = torch.sigmoid(self.rawActivationsScale)
            #normActScale = torch.sigmoid(self.normedActivationsScale)
            if debugPrints: ʕっʘ‿ʘʔっ("L2: normedActivationsTensor") # <- L1
            normedActsTensor = self.activationNorm(actsTensor) # _2
            #scaledActs = (actsTensor * rawActScale + normedActsTensor * normActScale)
            scaledActs = actsTensor + normedActsTensor # direct pass through skipping scaling
            #scaledActs = scaledActs.clamp(-10, 10)

            if debugPrints: print(f"Debug logits: activations shape before @ weights: {scaledActs.shape}")
            if debugPrints: print(f"Debug logits: weights shape: {self.l_weights.shape}")
                        
            if debugPrints: ʕっʘ‿ʘʔっ("L3: scaledActivations") # <- L1 + L2
            #logitOutput = (scaledActs @ self.l_weights / (numNeurons ** 0.5)) + self.l_bias
            logitOutput = (scaledActs @ self.l_weights) + self.l_bias
            #logitOutput = logitOutput.clamp(-60, 60) # DO NOT CLAMP, softens too much!
            logitNormed = self.logitNorm(logitOutput) #+ logitOutput  # softly smooth
            finalLogit  = logitNormed

            if debugPrints: print(f"Debug logits: logitOutput shape AFTER @ weights: {logitOutput.shape}")

            if debugPrints: ʕっʘ‿ʘʔっ("clamp scalar parameters")
            clamp_param(self.rawActivationsScale, 0, 0.75)
            clamp_param(self.normedActivationsScale, 0, 0.75)

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
                    "7L_0_actsTensor_norm": sum(self.tensorNormHist) / len(self.tensorNormHist),
                    #"7L_1_normActsTensor_norm": sum(self.normedNormHist) / len(self.normedNormHist),
                    #"7L_2_scaledActsTensor_norm": sum(self.activNormHist) / len(self.activNormHist),
                    #"7L_3_out_norm": sum(self.logitNormHist) / len(self.logitNormHist),
                    #"7L_4_outNorm_norm": sum(self.normLayerNormHist) / len(self.normLayerNormHist),
                    "7L_x_final_norm": sum(self.finalNormHist) / len(self.finalNormHist),

                    "7L_0_actsTensor_mean": sum(self.tensorHist) / len(self.tensorHist),
                    #"7L_1_normActsTensor_mean": sum(self.normedHist) / len(self.normedHist),
                    #"7L_2_scaledActsTensor_mean": sum(self.activHist) / len(self.activHist),
                    #"7L_3_out_mean": sum(self.logitHist) / len(self.logitHist),
                    #"7L_4_outNorm_mean": sum(self.normLayerHist) / len(self.normLayerHist),
                    "7L_x_final_mean": sum(self.finalHist) / len(self.finalHist),

                    "7L_0_actsTensor_min": sum(self.tensorMinHist) / len(self.tensorMinHist),
                    #"7L_1_normActsTensor_min": sum(self.normedMinHist) / len(self.normedMinHist),
                    #"7L_2_scaledActsTensor_min": sum(self.activMinHist) / len(self.activMinHist),
                    #"7L_3_out_min": sum(self.logitMinHist) / len(self.logitMinHist),
                    #"7L_4_outNorm_min": sum(self.normLayerMinHist) / len(self.normLayerMinHist),
                    "7L_x_final_min": sum(self.finalMinHist) / len(self.finalMinHist),

                    "7L_0_actsTensor_max": sum(self.tensorMaxHist) / len(self.tensorMaxHist),
                    #"7L_1_normActsTensor_max": sum(self.normedMaxHist) / len(self.normedMaxHist),
                    #"7L_2_scaledActsTensor_max": sum(self.activMaxHist) / len(self.activMaxHist),
                    #"7L_3_out_max": sum(self.logitMaxHist) / len(self.logitMaxHist),
                    #"7L_4_outNorm_max": sum(self.normLayerMaxHist) / len(self.normLayerMaxHist),
                    "7L_x_final_max": sum(self.finalMaxHist) / len(self.finalMaxHist),

                }

                for attr in self._history_attrs:
                    getattr(self, attr).clear()

            #with torch.no_grad():
                #topValues, topIndices = torch.topk(finalLogit, 5)
                #self.stats["7L_topLogits"] = topValues.tolist()
                #self.stats["7L_topIndices"] = topIndices.tolist()
                #self.stats["7L_logitMax"] = finalLogit.max().item()
                #self.stats["7L_logitMin"] = finalLogit.min().item()
                #self.stats["7L_logitMean"] = finalLogit.mean().item()
                #self.stats["7L_logitStd"] = finalLogit.std().item()
                #self.stats["7L_0_actsTensor_scale"] = rawScale.item()
                #self.stats["7L_1_normActsTensor_scale"] = normedScale.item()
                #self.stats["7L_3_outSigmoid_scale"] = outScale.detach().item()
                #self.stats["7L_4_outNormSigmoid_scale"] = normOutScale.detach().item()
            #if debugPrints: print("activation norm:", scaledActs.norm().item())
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
                self.stats["7L_0_actsTensor_scale"] = self.rawActivationsScale.norm().item()
                self.stats["7L_1_normActsTensor_scale"] = self.normedActivationsScale.norm().item()

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
        self._init_history_buffers()

# __main__ test harness removed (vanity)
