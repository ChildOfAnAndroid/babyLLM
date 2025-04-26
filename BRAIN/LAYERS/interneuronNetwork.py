# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# INTERNEURON NETWORK & NEURONS
# BRAIN/LAYERS/interneuronNetwork.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from config import *

class NEURON(nn.Module):
    def __init__(self, _counsellor, _device = modelDevice):
        super().__init__()
        self.device = _device
        self.n_counsellor = _counsellor
        # SELF ALLOWED - nn.parameter!
        self.n_weights = nn.Parameter(torch.randn(numNeurons, embedDimension, device = self.device) * 0.01)
        self.n_biases = nn.Parameter(torch.zeros(numNeurons, device = self.device))
        #self.n_counsellor = COUNSELLOR("NEURON", debug = debugPrints, durations = durationLogging)

        # MUST NOT BE ON SELF - global parameters that may be used by backward pass
        #numNeuron, embedDimension, activationFunction, etc

    def forward(self, _inputEmbeds):  # embed: (batch_size, embed_size)
        with self.n_counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            if skipNeuron: 
                ʕっʘ‿ʘʔっ("skipping forward")
                activations = _inputEmbeds.mean(dim=-1) # Mean across embed_dim, shape (sequence_length,)
                return activations[-1].unsqueeze(0) # Take LAST activation, unsqueeze to (1, ) for batch dim
            
            ʕっʘ‿ʘʔっ("computeBatchedDotProduct+bias") # Compute batched dot product + bias: (batch_size, num_neurons)
            output = torch.matmul(_inputEmbeds, self.n_weights.T) + self.n_biases 

            ʕっʘ‿ʘʔっ("activationFunction") # magic activation function applied to this weighted sum, which outputs a single number from the neuron
            output = activationFunction(output)
            if debugPrints: print("Device check:")
            if debugPrints: print("inputEmbeds:", _inputEmbeds.device)
            if debugPrints: print("output tensor device:", output.device)
            #output = torch.clamp(output, -5, 5) # ENSURE OUT-OF-PLACE
            output.clamp_(-5, 5) # IN PLACE VER

            return output

"""layer that applies the same set of neurons to each token embedding independently. - no sequence awareness!"""
class INTERNEURON_NETWORK(nn.Module):
    def __init__(self, _model, _counsellor, _calligraphist, _device = modelDevice):
        super().__init__()
        #self.inn_counsellor = COUNSELLOR("INN", debug = debugPrints, durations = durationLogging)
        self.model = _model
        self.inn_counsellor = _counsellor
        self.device = _device
        self.calligraphist = _calligraphist

        # SELF ALLOWED - nn.parameter!
        self.neurons = NEURON(_counsellor = self.inn_counsellor)

        self.cerebellum = nn.Parameter(torch.ones(len(allWindowSizes_new), device = self.device)) # THIS WAS THE WINDOW WEIGHTING LAYER
        self.windowCombos = nn.ModuleList([nn.Linear(numNeurons, numNeurons, device = self.device) for _ in range(len(allWindowSizes_new))])

        self.queryProj = nn.Linear(numNeurons, embedDimension, bias = True, device = self.device)
        self.keyProj = nn.Linear(numNeurons, embedDimension, bias = True, device = self.device)

        self.judgeBias = nn.Parameter(torch.zeros(len(allWindowSizes_new), device = self.device))
        self.credibilityBias = nn.Parameter(torch.zeros(len(allWindowSizes_new), device = self.device))
        self.parliamentBlend = nn.Parameter(torch.tensor(0.5, device = self.device))

        # MUST NOT BE ON SELF - global parameters that may be used by backward pass
        #numNeurons, embedDimension, activationFunction, allWindowSizes_new, etc


    def forward(self, _inputEmbeds):  
        with self.inn_counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            # --- iterates through input embeddings, applies all neurons in parallel for each, produces a vector of neuron outputs
            ʕっʘ‿ʘʔっ("localParamInit") # AVOIDING SELF - parameters only used in this function and never passed
            tinyWindowCount = 0
            if skipINN: 
                ʕっʘ‿ʘʔっ("skipping INNforward")
                perTokenActivationsTensor = self.neurons(_inputEmbeds)

                windowMeanActivations = []
                for windowSize in allWindowSizes_new:
                    if perTokenActivationsTensor.shape[0] < windowSize:
                        tinyWindowCount += 1
                        emptyWindow = torch.zeros_like(perTokenActivationsTensor[0]).unsqueeze(0)
                        windowMeanActivations.append(emptyWindow)
                    else:
                        windowMean = torch.mean(perTokenActivationsTensor[-windowSize:], dim = 0, keepdim = True)
                        windowMeanActivations.append(windowMean)

                if not windowMeanActivations:
                    print("no valid window sizes")
                    return torch.zeros_like(perTokenActivationsTensor[0])
                
                windowMeanStack = torch.stack(windowMeanActivations, dim = 0).squeeze(1)
                clampedCerebellum = self.cerebellum.clamp(min=-1.0, max = 1.0)
                #self.parliamentBlendClamped = torch.sigmoid(self.parliamentBlend)

                self.cerebellumSoft = F.softmax(clampedCerebellum + (clampedCerebellum.abs() / 2), dim = 0)
                mix = (0.5 * self.cerebellumSoft) + (0.5 * torch.tanh(clampedCerebellum))
                weightedWindowStack = windowMeanStack * mix.reshape(-1, 1)
                #weightedWindowStack = windowMeanStack * self.cerebellumSoft.reshape(-1, 1)
                
                combinedActivationsTensor = weightedWindowStack.sum(dim = 0, keepdim = True)

                if tinyWindowCount > 0:
                    print(f"saw {perTokenActivationsTensor.shape[0]} tokens; created {tinyWindowCount} empty windows.")

                ʕっʘ‿ʘʔっ("entropyReward?")
                self.windowEntropy = -torch.sum(self.cerebellumSoft * torch.log(self.cerebellumSoft + 1e-12))
                self.entropyBonus = self.windowEntropy

                #target = 0.0
                #rate = 0.001
                #with torch.no_grad():
                    #delta = (target - self.cerebellum) * rate
                    #self.cerebellum.add_(delta)

                return combinedActivationsTensor
            
            # --- DO NOT TAKE ANYTHING TO SELF PAST HERE, IT SHOULD ALL PASS THROUGH BACKWARD WITHOUT SAVING! --- #
            ʕっʘ‿ʘʔっ("CALL NEURON FORWARD")
            #if debugPrints: print(f"Device check - inputEmbeds: {_inputEmbeds.device}, neuron weights: {self.neurons.n_weights.device}")
            neuronActivations = self.neurons(_inputEmbeds)
            ʕっʘ‿ʘʔっ("windowOutputs") # --- combine activations into their own learnable layer
            windowOutputsTensor = self.stackedWindowMeans(neuronActivations, allWindowSizes_new)
            comboViews = torch.stack([
                self.windowCombos[i](windowOutputsTensor[i]) for i in range(len(allWindowSizes_new))
            ], dim = 0)

            # Project summaries to queries and keys for attention scoring
            ʕっʘ‿ʘʔっ("cerebellumSoft")
            self.cerebellumSoft = F.softmax(self.cerebellum, dim = 0) # THIS WAS THE WINDOW WEIGHTING LAYER
            ʕっʘ‿ʘʔっ("query")
            query = self.queryProj(comboViews) + self.judgeBias.unsqueeze(1) * self.cerebellum.unsqueeze(1)  # shape: (32, numNeurons)
            ʕっʘ‿ʘʔっ("key")
            key = self.keyProj(windowOutputsTensor) + self.credibilityBias.unsqueeze(1) * self.cerebellum.unsqueeze(1)   # shape: (32, numNeurons)
            # Compute attention scores between every pair of windows (32x32 matrix)

            ʕっʘ‿ʘʔっ("scores")
            self.scores = torch.matmul(query, key.T) / self.model.temperature

            ʕっʘ‿ʘʔっ("selfScores & peerScores") # separate self scores (diagonal) and peer scores (off-diagonals)
            self.selfScores = torch.diag(self.scores) # self score for window i: scores[i, i] (shape: (32,))
            self.peerScores = self.scores.sum(dim = 0) - self.selfScores # peer scores for window j: sum of scores[i, j] for all i != j (shape: (32,))
            ʕっʘ‿ʘʔっ("combinedScores")
            self.combinedScores = self.selfScores + self.peerScores # shape: (32,)
            softCombinedScores = F.softmax(self.combinedScores, dim = 0)
            self.attentionWindowWeights = softCombinedScores  # shape: (32,), sum = 1

            ʕっʘ‿ʘʔっ("parliamentBlendMix") # learned mix instinct vs votes
            parliamentBlendClamped = torch.sigmoid(self.parliamentBlend)  # stays in [0, 1]
            #combinedWindowWeights = ((1.0 - self.parliamentBlend) * self.cerebellumSoft + self.parliamentBlend * softCombinedScores)
            if skipINNparliament:
                combinedWindowWeights = self.cerebellumSoft
            else:
                blended = ((1.0 - parliamentBlendClamped) * self.cerebellum + parliamentBlendClamped * self.combinedScores) * self.cerebellum
                combinedWindowWeights = F.softmax(blended, dim = 0)
                
            ʕっʘ‿ʘʔっ("formatWindowVoteString")
            #parliamentBlend_str = f"{self.parliamentBlend.item():.3f}"
            topWeights = sorted(zip(allWindowSizes_new, combinedWindowWeights.tolist()), key = lambda x: x[1], reverse = True)
            topWindows_str = ",".join(f"W{w}:{wgt:.2f}" for w, wgt in topWeights[:4])
            self.windowVotes_str = f"top windows: {topWindows_str}"

            ʕっʘ‿ʘʔっ("weightedWindows")
            windowContextVector = torch.sum(comboViews + combinedWindowWeights.unsqueeze(1), dim = 0, keepdim = True)

            ʕっʘ‿ʘʔっ("finalActions")
            if tinyWindowCount > 0: print(f"saw {neuronActivations.shape[0]} tokens; created {tinyWindowCount} empty windows.")

            raw = comboViews.mean(dim = 0, keepdim = True) # keeps gradients open
            windowContextVector = windowContextVector + 0.2 * raw

            ʕっʘ‿ʘʔっ("entropyReward?")
            self.windowEntropy = -torch.sum(self.attentionWindowWeights * torch.log(self.attentionWindowWeights + 1e-12))
            self.entropyBonus = self.windowEntropy.item()

            return windowContextVector
        
    def stackedWindowMeans(self, activations: torch.Tensor, windowSizes: list[int]) -> torch.Tensor:
        """
        Fully vectorized, loop-free window mean calculation.
        Returns: (len(windowSizes), embedDim)
        """
        seqLen, embedDim = activations.shape

        # Right-aligned padding: (maxW, embedDim)
        padded = torch.zeros((windowMAX, embedDim), device = self.device)
        padded[-min(seqLen, windowMAX):] = activations[-min(seqLen, windowMAX):]

        # Stack: (numWindows, maxW, embedDim)
        stacked = padded.unsqueeze(0).repeat(len(windowSizes), 1, 1)

        # Build mask: (numWindows, maxW, 1)
        range_mask = torch.arange(windowMAX, device = self.device).unsqueeze(0)
        window_tensor = torch.tensor(windowSizes, device = self.device).unsqueeze(1)
        mask = (range_mask < window_tensor).float().unsqueeze(2)

        # Apply mask, compute sums and divide by window sizes
        masked = stacked * mask
        sums = masked.sum(dim = 1)
        means = sums / window_tensor

        return means  # shape: (numWindows, embedDim)
    
    def INN_getStats(self):
        with self.inn_counsellor.infodump("INN_getStats") as ʕっʘ‿ʘʔっ:
            INN_cerebellum_str = ""
            INN_credibilityBias_str = ""
            INN_judgeBias_str = ""
            self.windowVotes_str = ""
            stats = {}
            if collectStats and n_collectStats:
                ʕっʘ‿ʘʔっ("torch.no_grad♥")
                with torch.no_grad():
                    if n_weightStats:
                        ʕっʘ‿ʘʔっ("♥n_weightStats")
                        stats["n_weightMean"] = self.neurons.n_weights.mean()
                        stats["n_weightStd"] = self.neurons.n_weights.std()
                        stats["n_weightMin"] = self.neurons.n_weights.min()
                        stats["n_weightMax"] = self.neurons.n_weights.max()
                        if debugPrints: print(f"neuron weight mean: {stats["n_weightMean"]} std: {stats["n_weightStd"]} min: {stats["n_weightMin"]} max: {stats["n_weightMax"]}")
                    
                    if n_weightNormStats:
                        ʕっʘ‿ʘʔっ("♥n_weightNormStats")
                        stats["n_weightNorm"] = torch.norm(self.neurons.n_weights, dim = 1)
                        stats["n_weightNormMean"] = stats["n_weightNorm"].mean()
                        stats["n_weightNormMin"] = stats["n_weightNorm"].min()
                        stats["n_weightNormMax"] = stats["n_weightNorm"].max()
                        if debugPrints: print(f"neuron weightNorm: {stats["n_weightNorm"]} mean: {stats["n_weightNormMean"]} min: {stats["n_weightNormMax"]} max: {stats["n_weightNormMin"]}")

                    if n_biasesStats:
                        ʕっʘ‿ʘʔっ("♥n_biasesStats")                    
                        stats["n_biasesMean"] = self.neurons.n_biases.mean()
                        stats["n_biasesStd"] = self.neurons.n_biases.std()
                        stats["n_biasesMin"] = self.neurons.n_biases.min()
                        stats["n_biasesMax"] = self.neurons.n_biases.max()
                        if debugPrints: print(f"neuron biases mean: {stats["n_biasesMean"]} std: {stats["n_biasesStd"]} min: {stats["n_biasesMin"]} max: {stats["n_biasesMax"]}")

                    if n_sparsityStat:
                        ʕっʘ‿ʘʔっ("♥getSparsityStat")
                        stats["n_sparsity"] = (self.neurons.n_weights.abs() < 1e-5).float().mean()
                        if debugPrints: print(f"neuron sparsity: {stats["n_sparsity"]}")

            if collectStats and INN_collectStats:
                ʕっʘ‿ʘʔっ("torch.no_grad♥")
                with torch.no_grad():
                    if INN_cerebellumStats:
                        ʕっʘ‿ʘʔっ("♥getCerebellumStats") #THIS WAS WINDOWWEIGHTING
                        stats["INN_cerebellum"] = self.cerebellum
                        stats["INN_cerebellumSoft"] = self.cerebellumSoft
                        stats["INN_cerebellumMean"] = self.cerebellum.mean()
                        stats["INN_cerebellumStd"] = self.cerebellum.std()
                        #stats["INN_parliament"] = self.parliamentBlend
                        #stats["INN_parliamentSoft"] = self.parliamentBlendClamped
                        if debugPrints: print(f"cerebellum: {stats["INN_cerebellum"]}, soft: {stats["INN_cerebellumSoft"]} mean: {stats['INN_cerebellumMean']} std: {stats['INN_cerebellumStd']}")
                        ʕっʘ‿ʘʔっ("♥cerebellumString")
                        #INN_hybridCerebellum = sorted(zip(allWindowSizes_new, stats["INN_cerebellum"], stats["INN_cerebellumSoft"]), key = lambda x: x[1], reverse = True)
                        #INN_cerebellum_str = ",".join(f"W{w}:{RAW:.5f} ({SOFTMAX:.2f})" for w, RAW, SOFTMAX in INN_hybridCerebellum)
                        #windowVotes_str = self.calligraphist.S_formatWindowBiasTriplets(label="INN_parliament", rawTensor = stats["INN_parliament"], softTensor = stats["INN_parliamentSoft"], windowSizes = allWindowSizes_new)
                        INN_cerebellum_str = self.calligraphist.S_formatWindowBiasTriplets(label="INN_cerebellum", rawTensor = stats["INN_cerebellum"], softTensor = stats["INN_cerebellumSoft"], windowSizes = allWindowSizes_new)
                        if debugPrints: print(f"{INN_cerebellum_str}")


                    if INN_credibilityBiasStats:
                        ʕっʘ‿ʘʔっ("♥getCredibilityBiasStats")
                        stats["INN_credibilityBias"] = self.credibilityBias
                        stats["INN_credibilityBiasSoft"] = F.softmax(self.credibilityBias, dim = 0)
                        stats["INN_credibilityBiasMean"] = self.credibilityBias.mean()
                        stats["INN_credibilityBiasStd"] = self.credibilityBias.std()
                        if debugPrints: print(f"credibilityBias: {stats["INN_credibilityBias"]} soft: {stats["INN_credibilityBiasSoft"]} mean: {stats["INN_credibilityBiasMean"]} std: {stats["INN_credibilityBiasStd"]}")
                        ʕっʘ‿ʘʔっ("♥credibilityBiasString")
                        #INN_hybridCredibilityBias = sorted(zip(allWindowSizes_new, stats["INN_credibilityBias"], stats["INN_credibilityBiasSoft"]), key = lambda x: x[1], reverse = True)
                        #INN_credibilityBias_str = ",".join(f"W{w}:{RAW:.5f} ({SOFTMAX:.2f})" for w, RAW, SOFTMAX in INN_hybridCredibilityBias)
                        INN_credibilityBias_str = self.calligraphist.S_formatWindowBiasTriplets(label="INN_credibilityBias", rawTensor = stats["INN_credibilityBias"], softTensor = stats["INN_credibilityBiasSoft"], windowSizes = allWindowSizes_new)
                        if debugPrints: print(f"{INN_credibilityBias_str}") 

                    if INN_judgeBiasStats:
                        ʕっʘ‿ʘʔっ("♥getJudgeBiasStats")
                        stats["INN_judgeBias"] = self.judgeBias
                        stats["INN_judgeBiasSoft"] = F.softmax(self.judgeBias, dim = 0)
                        stats["INN_judgeBiasMean"] = self.judgeBias.mean()
                        stats["INN_judgeBiasStd"] = self.judgeBias.std()
                        if debugPrints: print(f"judgeBias: {stats["INN_judgeBias"]} soft: {stats["INN_judgeBiasSoft"]} mean: {stats["INN_judgeBiasMean"]} std: {stats["INN_judgeBiasStd"]}")
                        ʕっʘ‿ʘʔっ("♥judgeBiasString")
                        #INN_hybridJudgeBias = sorted(zip(allWindowSizes_new, stats["INN_judgeBiasSoft"], stats["INN_judgeBias"]), key = lambda x: x[1], reverse = True)
                        #INN_judgeBias_str = ",".join(f"W{w}:{RAW:.5f} ({SOFT:.2f})" for w, SOFT, RAW in INN_hybridJudgeBias)
                        INN_judgeBias_str = self.calligraphist.S_formatWindowBiasTriplets(label="INN_judgeBias", rawTensor = stats["INN_judgeBias"], softTensor = stats["INN_judgeBiasSoft"], windowSizes = allWindowSizes_new)
                        if debugPrints: print(f"{INN_judgeBias_str}")
                        
                    if INN_scoringStats:
                        ʕっʘ‿ʘʔっ("♥getScoringStats")
                        #stats["INN_scores"] = self.scores
                        #stats["INN_selfScores"] = self.selfScores
                        #stats["INN_peerScores"] = self.peerScores
                        #stats["INN_combinedScores"] = self.combinedScores
                        #if debugPrints: print(f"INN Parliament Scores: {INN_scores} self: {INN_selfScores} peer: {INN_peerScores} combined: {INN_combinedScores}")

                    """window stats"""
                    if INN_windowStats:
                        ʕっʘ‿ʘʔっ("♥windowStats")
                        windowVotes_str = self.windowVotes_str
                        #stats["INN_topWindowWeight"] = self.attentionWindowWeights.max()
                        #stats["INN_windowStd"] = self.attentionWindowWeights.std()
                        #stats["INN_windowEntropy"] = self.entropyBonus
                        #stats["INN_effectiveWindowCount"] = torch.exp(torch.tensor(self.windowEntropy))
                        #if debugPrints: print(f"window stats: top: {stats["INN_topWindowWeight"]} std: {stats["INN_windowStd"]} entropy: {stats["INN_windowEntropy"]} effective window count: {stats["INN_effectiveWindowCount"]}")

        return stats, INN_cerebellum_str, INN_judgeBias_str, INN_credibilityBias_str, windowVotes_str 
    
if __name__ == "__main__":
    interneuronNetwork = INTERNEURON_NETWORK()

    TESTinputSeq = torch.randn(window1, embedDimension)
    TESTinputEmbeds = TESTinputSeq

    meanActivationsTensor = interneuronNetwork.forward(TESTinputEmbeds)

    print("--- INTERNEURON NETWORK TESTING START ---")
    print(f"parallel neuron layer created with {interneuronNetwork.numNeurons} neurons.")
    print(f"inputs per neuron (embed dimension): {interneuronNetwork.embedDimension}")
    print(f"output activations (first 10):")
    print(meanActivationsTensor[:10])
    print(f"output activations shape: {meanActivationsTensor.shape}")
    print("\n--- INTERNEURON NETWORK TESTING COMPLETED ---")