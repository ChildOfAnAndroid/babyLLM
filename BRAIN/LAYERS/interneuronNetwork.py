# CHARIS CAT 2025
# BABYLLM - interneuronNetwork.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from BRAIN.LAYERS.S_output import *
import math

class NEURON(nn.Module):
    def __init__(self):
        super().__init__()
        # SELF ALLOWED - nn.parameter!
        self.n_weights = nn.Parameter(torch.randn(numNeurons, embedDimension, device = modelDevice) * 0.01)
        self.n_biases = nn.Parameter(torch.zeros(numNeurons, device = modelDevice))
        self.n_counsellor = COUNSELLOR("NEURON", debug=debugPrints, durations=durationLogging)

        # MUST NOT BE ON SELF - global parameters that may be used by backward pass
        #numNeuron, embedDimension, activationFunction, etc

    def forward(self, inputEmbeds):  # embed: (batch_size, embed_size)
        with self.n_counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            if skipNeuron: 
                ʕっʘ‿ʘʔっ("skipping forward")
                activations = inputEmbeds.mean(dim=-1) # Mean across embed_dim, shape (sequence_length,)
                return activations[-1].unsqueeze(0) # Take LAST activation, unsqueeze to (1, ) for batch dim
            ʕっʘ‿ʘʔっ("computeBatchedDotProduct+bias") # Compute batched dot product + bias: (batch_size, num_neurons)
            output = torch.matmul(inputEmbeds, self.n_weights.T) + self.n_biases   

            ʕっʘ‿ʘʔっ("activationFunction") # magic activation function applied to this weighted sum, which outputs a single number from the neuron
            output = activationFunction(output)
            output = torch.clamp(output, -5, 5) # ENSURE OUT-OF-PLACE

            return output

"""layer that applies the same set of neurons to each token embedding independently. - no sequence awareness!"""
class INTERNEURON_NETWORK(nn.Module):
    def __init__(self):
        super().__init__()

        # SELF ALLOWED - nn.parameter!
        self.cerebellum = nn.Parameter(torch.ones(len(allWindowSizes_new), device = modelDevice)) # THIS WAS THE WINDOW WEIGHTING LAYER
        self.windowCombos = nn.ModuleList([nn.Linear(numNeurons, numNeurons, device = modelDevice) for _ in range(len(allWindowSizes_new))])

        self.queryProj = nn.Linear(numNeurons, embedDimension, bias=True, device=modelDevice)
        self.keyProj = nn.Linear(numNeurons, embedDimension, bias=True, device=modelDevice)

        self.judgeBias = nn.Parameter(torch.zeros(len(allWindowSizes_new), device = modelDevice))
        self.credibilityBias = nn.Parameter(torch.zeros(len(allWindowSizes_new), device = modelDevice))
        self.parliamentBlend = nn.Parameter(torch.tensor(0.5, device=modelDevice))

        self.neurons = NEURON()

        # MUST NOT BE ON SELF - global parameters that may be used by backward pass
        #numNeurons, embedDimension, activationFunction, allWindowSizes_new, etc

        # NO IDEA IF THESE CAN BE ON SELF (other class calls)
        self.inn_counsellor = COUNSELLOR("INN", debug = debugPrints, durations = durationLogging)

    def forward(self, inputEmbeds):  
        with self.inn_counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
        # --- iterates through input embeddings, applies all neurons in parallel for each, produces a vector of neuron outputs
            ʕっʘ‿ʘʔっ("localParamInit") # AVOIDING SELF - parameters only used in this function and never passed
            tinyWindowCount = 0
            # --- DO NOT TAKE ANYTHING TO SELF PAST HERE, IT SHOULD ALL PASS THROUGH BACKWARD WITHOUT SAVING! --- #
            ʕっʘ‿ʘʔっ("CALL NEURON FORWARD")
            neuronActivations = self.neurons(inputEmbeds)
            if skipINN:
                ʕっʘ‿ʘʔっ("skipping INNforward")
                return neuronActivations
            ʕっʘ‿ʘʔっ("windowOutputs") # --- combine activations into their own learnable layer
            windowOutputs = []
            if debugPrints: 
                for name, param in self.named_parameters():
                    print(name, param.requires_grad, param.grad is not None)
            for windowSize in allWindowSizes_new:
                if inputEmbeds.shape[0] < windowSize: 
                    ʕっʘ‿ʘʔっ("not enough tokens for window (neurons.n_weights.mean)") # --- Not enough tokens for this window; use a zero vector
                    tinyWindowCount += 1
                    #summary = torch.zeros_like(numNeurons, device = modelDevice)
                    summary = neuronActivations.mean(dim=0) * 0  # KEEPS GRADIENTS FLOWING EVEN WHEN ZERO - shape: [numNeurons], safe to stack
                else: # Mean pooling over the last 'windowSize' token activations
                    ʕっʘ‿ʘʔっ("mean pooling all over all tokens (torch.mean)") # --- MEAN IS OVER WINDOW SIZE
                    summary = torch.mean(neuronActivations[-windowSize:], dim=0)
                ʕっʘ‿ʘʔっ("append window summaries")
                if debugPrints: 
                    for name, param in self.named_parameters():
                        print(name, param.requires_grad, param.grad is not None)
                windowOutputs.append(summary)

            ʕっʘ‿ʘʔっ("windowCombos")
            comboViews = []
            for i, summary in enumerate(windowOutputs):
                transformed = self.windowCombos[i](summary)
                comboViews.append(transformed)
            windowOutputsTensor = torch.stack(comboViews, dim=0)  # shape: (numWindows, numNeurons)

            # Project summaries to queries and keys for attention scoring
            ʕっʘ‿ʘʔっ("cerebellumSoft")
            self.cerebellumSoft = F.softmax(self.cerebellum, dim=0) # THIS WAS THE WINDOW WEIGHTING LAYER
            ʕっʘ‿ʘʔっ("query")
            query = self.queryProj(windowOutputsTensor) + self.judgeBias.unsqueeze(1) + self.cerebellum.unsqueeze(1)  # shape: (32, numNeurons)
            ʕっʘ‿ʘʔっ("key")
            key = self.keyProj(windowOutputsTensor) + self.credibilityBias.unsqueeze(1) + self.cerebellum.unsqueeze(1)   # shape: (32, numNeurons)
            # Compute attention scores between every pair of windows (32x32 matrix)

            ʕっʘ‿ʘʔっ("scores")
            scores = torch.matmul(query, key.T) / temperature

            ʕっʘ‿ʘʔっ("selfScores & peerScores") # separate self scores (diagonal) and peer scores (off-diagonals)
            selfScores = torch.diag(scores) # self score for window i: scores[i, i] (shape: (32,))
            peerScores = scores.sum(dim=0) - selfScores # peer scores for window j: sum of scores[i, j] for all i != j (shape: (32,))
            ʕっʘ‿ʘʔっ("combinedScores")
            combinedScores = selfScores + peerScores # shape: (32,)
            softCombinedScores = F.softmax(combinedScores, dim=0)
            attentionWindowWeights = softCombinedScores  # shape: (32,), sum = 1

            if statPrints or debugPrints:
                print(f"attentionWindowWeights: {attentionWindowWeights}")
                windowEntropy = -torch.sum(attentionWindowWeights * torch.log(attentionWindowWeights + 1e-12))
                print(f"windowEntropy: {windowEntropy}")

            ʕっʘ‿ʘʔっ("parliamentBlendMix") # learned mix instinct vs votes
            parliamentBlendClamped = torch.sigmoid(self.parliamentBlend)  # stays in [0, 1]
            combinedWindowWeights = (
                (1.0 - parliamentBlendClamped) * self.cerebellumSoft +
                parliamentBlendClamped * attentionWindowWeights)

            ʕっʘ‿ʘʔっ("weightedWindows")
            weightedViews = windowOutputsTensor * combinedWindowWeights.unsqueeze(1)
            windowContextVector = weightedViews.sum(dim=0, keepdim=True)  # (1, numNeurons)

            ʕっʘ‿ʘʔっ("finalActions")
            if tinyWindowCount > 0: print(f"saw {neuronActivations.shape[0]} tokens; created {tinyWindowCount} empty windows.")
            torch.mps.empty_cache()

            return windowContextVector
    
    def INN_getStats(self):
        with self.inn_counsellor.infodump("INN_getStats") as ʕっʘ‿ʘʔっ:
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
                        if statPrints or debugPrints: print(f"neuron weight mean: {stats["n_weightMean"]} std: {stats["n_weightStd"]} min: {stats["n_weightMin"]} max: {stats["n_weightMax"]}")
                    
                    if n_weightNormStats:
                        ʕっʘ‿ʘʔっ("♥n_weightNormStats")
                        stats["n_weightNorm"] = torch.norm(self.neurons.n_weights, dim=1)
                        stats["n_weightNormMean"] = stats["n_weightNorm"].mean()
                        stats["n_weightNormMax"] = stats["n_weightNorm"].min()
                        stats["n_weightNormMin"] = stats["n_weightNorm"].max()
                        if statPrints or debugPrints: print(f"neuron weightNorm: {stats["n_weightNorm"]} mean: {stats["n_weightNormMean"]} min: {stats["n_weightNormMax"]} max: {stats["n_weightNormMin"]}")

                    if n_biasesStats:
                        ʕっʘ‿ʘʔっ("♥n_biasesStats")                    
                        stats["n_biasesMean"] = self.neurons.n_biases.mean()
                        stats["n_biasesStd"] = self.neurons.n_biases.std()
                        stats["n_biasesMin"] = self.neurons.n_biases.min()
                        stats["n_biasesMax"] = self.neurons.n_biases.max()
                        if statPrints or debugPrints: print(f"neuron biases mean: {stats["n_biasesMean"]} std: {stats["n_biasesStd"]} min: {stats["n_biasesMin"]} max: {stats["n_biasesMax"]}")

                    if n_sparsityStat:
                        ʕっʘ‿ʘʔっ("♥getSparsityStat")
                        stats["n_sparsity"] = (self.neurons.n_weights.abs() < 1e-6).float().mean()
                        if statPrints or debugPrints: print(f"neuron sparsity: {stats["n_sparsity"]}")

            if collectStats and INN_collectStats:
                ʕっʘ‿ʘʔっ("torch.no_grad♥")
                with torch.no_grad():
                    if INN_cerebellumStats:
                        ʕっʘ‿ʘʔっ("♥getCerebellumStats") #THIS WAS WINDOWWEIGHTING
                        stats["INN_cerebellum"] = self.cerebellum
                        stats["INN_cerebellumSoft"] = self.cerebellumSoft
                        stats["INN_cerebellumMean"] = self.cerebellum.mean()
                        stats["INN_cerebellumStd"] = self.cerebellum.std()
                        if statPrints or debugPrints: print(f"cerebellum: {stats["INN_cerebellum"]}, soft: {stats["INN_cerebellumSoft"]} mean: {stats['INN_cerebellumMean']} std: {stats['INN_cerebellumStd']}")
                        ʕっʘ‿ʘʔっ("♥cerebellumString")
                        INN_hybridCerebellum = sorted(zip(allWindowSizes_new, stats["INN_cerebellumSoft"], stats["INN_cerebellum"]), key=lambda x: x[1], reverse=True)
                        INN_cerebellum_str = ",".join(f"W{w}:{RAW:.5f} ({SOFTMAX:.2f})" for w, SOFTMAX, RAW in INN_hybridCerebellum)
                        if statPrints or debugPrints: print(f"{INN_cerebellum_str}")

                    if INN_credibilityBiasStats:
                        ʕっʘ‿ʘʔっ("♥getCredibilityBiasStats")
                        stats["INN_credibilityBias"] = self.credibilityBias
                        stats["INN_credibilityBiasSoft"] = F.softmax(self.credibilityBias, dim=0)
                        stats["INN_credibilityBiasMean"] = self.credibilityBias.mean()
                        stats["INN_credibilityBiasStd"] = self.credibilityBias.std()
                        if statPrints or debugPrints: print(f"credibilityBias: {stats["INN_credibilityBias"]} soft: {stats["INN_credibilityBiasSoft"]} mean: {stats["INN_credibilityBiasMean"]} std: {stats["INN_credibilityBiasStd"]}")
                        ʕっʘ‿ʘʔっ("♥credibilityBiasString")
                        INN_hybridCredibilityBias = sorted(zip(allWindowSizes_new, stats["INN_credibilityBias"], stats["INN_credibilityBiasSoft"]), key=lambda x: x[1], reverse=True)
                        INN_credibilityBias_str = ",".join(f"W{w}:{RAW:.5f} ({SOFT:.2f})" for w, SOFT, RAW in INN_hybridCredibilityBias)
                        if statPrints or debugPrints: print(f"{INN_credibilityBias_str}")     

                    if INN_judgeBiasStats:
                        ʕっʘ‿ʘʔっ("♥getJudgeBiasStats")
                        stats["INN_judgeBias"] = self.judgeBias
                        stats["INN_judgeBiasSoft"] = F.softmax(self.judgeBias, dim=0)
                        stats["INN_judgeBiasMean"] = self.judgeBias.mean()
                        stats["INN_judgeBiasStd"] = self.judgeBias.std()
                        if debugPrints: print(f"judgeBias: {stats["INN_judgeBias"]} soft: {stats["INN_judgeBiasSoft"]} mean: {stats["INN_judgeBiasMean"]} std: {stats["INN_judgeBiasStd"]}")
                        ʕっʘ‿ʘʔっ("♥judgeBiasString")
                        INN_hybridJudgeBias = sorted(zip(allWindowSizes_new, stats["INN_judgeBiasSoft"], stats["INN_judgeBias"]), key=lambda x: x[1], reverse=True)
                        INN_judgeBias_str = ",".join(f"W{w}:{RAW:.5f} ({SOFT:.2f})" for w, SOFT, RAW in INN_hybridJudgeBias)
                        if debugPrints: print(f"{INN_judgeBias_str}")
                        
                    #if INN_scoringStats:
                        #ʕっʘ‿ʘʔっ("♥getScoringStats")
                        #stats["INN_scores"] = self.scores
                        #stats["INN_selfScores"] = self.selfScores
                        #stats["INN_peerScores"] = self. peerScores
                        #stats["INN_combinedScores"] = self.combinedScores
                        #if statPrints or debugPrints: print(f"INN Parliament Scores: {INN_scores} self: {INN_selfScores} peer: {INN_peerScores} combined: {INN_combinedScores}")

                    """window stats"""
                    if INN_windowStats:
                        ʕっʘ‿ʘʔっ("♥windowStats")
                        #stats["INN_topWindowWeight"] = attentionWindowWeights.max()
                        #stats["INN_windowStd"] = INN_attentionWindowWeights.std()
                        #stats["INN_windowEntropy"] = 
                        #stats["INN_effectiveWindowCount"] = torch.exp(torch.tensor(INN_windowEntropy))
                        #if debugPrints: print(f"window stats: top: {stats["INN_topWindowWeight"]} std: {stats["INN_windowStd"]} entropy: {stats["INN_windowEntropy"]} effective window count: {stats["INN_effectiveWindowCount"]}")

            return stats, INN_cerebellum_str, INN_judgeBias_str, INN_credibilityBias_str      
    
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