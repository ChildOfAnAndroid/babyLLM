# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# INTERNEURON NETWORK & NEURONS
# BRAIN/LAYERS/interneuronNetwork.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from config import *
from SCHOOL.staffroom.counsellor import COUNSELLOR

"""def tensorStats(tensor: torch.Tensor, prefix: str, statsDict: dict):
    statsDict[f"{prefix}_norm"] = tensor.norm().item()
    statsDict[f"{prefix}_norm_token"] = tensor.norm(dim=1).mean().item()
    statsDict[f"{prefix}_norm_neuron"] = tensor.norm(dim=0).mean().item()"""

class NEURON(nn.Module):
    def __init__(self, _counsellor, _numTokensPerStep, _device = modelDevice):
        super().__init__()
        self.device = _device
        self.n_counsellor = _counsellor
        self.numTokensPerStep = _numTokensPerStep
        
        # SELF ALLOWED - nn.parameter!
        self.inputNorm = nn.LayerNorm(embedDimension, elementwise_affine=True, device=self.device)
        self.n_weights = nn.Parameter(torch.randn(numNeurons, embedDimension, device = self.device) * 0.01)
        self.n_biases = nn.Parameter(torch.zeros(numNeurons, device = self.device))
        self.neuronNorm = nn.LayerNorm(numNeurons, elementwise_affine=True, device=self.device)
        self.n_counsellor = COUNSELLOR("NEURON", _debug = debugPrints, _durations = durationLogging)

        self.stats = {}

        self.rawInputHistory = []
        self.rawInputHistory_tokens = []
        self.rawInputHistory_neurons = []

        self.normedInputHistory = []
        self.normedInputHistory_tokens = []
        self.normedInputHistory_neurons = []

        self.rawOutputHistory = []
        self.rawOutputHistory_tokens = []
        self.rawOutputHistory_neurons = []

        self.activatedOutputHistory = []
        self.activatedOutputHistory_tokens = []
        self.activatedOutputHistory_neurons = []

        #self.normedOutputHistory = []
        #self.normedOutputHistory_tokens = []
        #self.normedOutputHistory_neurons = []

        # MUST NOT BE ON SELF - global parameters that may be used by backward pass
        #numNeuron, embedDimension, activationFunction, etc

    @whocalled
    def forward(self, _inputEmbeds):  # embed: (batch_size, embed_size)
        with self.n_counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            self.inputEmbeds = _inputEmbeds
            with torch.no_grad():
                weightNorm = self.n_weights.norm(dim=1, keepdim=True)
                clippedWeights = self.n_weights / weightNorm.clamp(min=1.0, max=100.0)
                self.n_weights.data = clippedWeights
                
            if skipNeuron: 
                ʕっʘ‿ʘʔっ("skipping forward")
                activations = _inputEmbeds.mean(dim=-1) # Mean across embed_dim, shape (sequence_length,)
                return activations[-1].unsqueeze(0) # Take LAST activation, unsqueeze to (1, ) for batch dim
            
            ʕっʘ‿ʘʔっ("inputNorm")
            normedInput = self.inputNorm(_inputEmbeds)

            ʕっʘ‿ʘʔっ("computeBatchedDotProduct+bias") # Compute batched dot product + bias: (batch_size, num_neurons)
            #rawOutput = torch.matmul(normedInput, self.n_weights.T) + self.n_biases  # shape: (seq_len, numNeurons)
            scale = math.sqrt(embedDimension)
            rawOutput = (torch.matmul(normedInput, self.n_weights.T) + self.n_biases) / scale # new fancy clamping attempt bdsm idfk nipple clamps its 12 noon help me

            ʕっʘ‿ʘʔっ("activationFunction") # magic activation function applied to this weighted sum, which outputs a single number from the neuron
            activated = activationFunction(rawOutput)

            #ʕっʘ‿ʘʔっ("layerNorm")
            #normed = self.neuronNorm(activated)  # keeps shape: (seq_len, numNeurons)

            if debugPrints: print("Device check:")
            if debugPrints: print("inputEmbeds:", _inputEmbeds.device)
            #if debugPrints: print("normed tensor device:", normed.device)
            #output = torch.clamp(output, -5, 5) # ENSURE OUT-OF-PLACE
            #output.clamp_(-5, 5) # IN PLACE VER

            if True:
                self.rawInputHistory.append(self.inputEmbeds.norm().item())
                self.rawInputHistory_tokens.append(self.inputEmbeds.norm(dim=1).mean().item())
                self.rawInputHistory_neurons.append(self.inputEmbeds.norm(dim=0).mean().item())

                self.normedInputHistory.append(normedInput.norm().item())
                self.normedInputHistory_tokens.append(normedInput.norm(dim=1).mean().item())
                self.normedInputHistory_neurons.append(normedInput.norm(dim=0).mean().item())

                self.rawOutputHistory.append(rawOutput.norm().item())
                self.rawOutputHistory_tokens.append(rawOutput.norm(dim=1).mean().item())
                self.rawOutputHistory_neurons.append(rawOutput.norm(dim=0).mean().item())

                self.activatedOutputHistory.append(activated.norm().item())
                self.activatedOutputHistory_tokens.append(activated.norm(dim=1).mean().item())
                self.activatedOutputHistory_neurons.append(activated.norm(dim=0).mean().item())

                # --- More diagnostic stats (do not break grid)
                self.activatedOutputHistory_std_token = getattr(self, 'activatedOutputHistory_std_token', [])
                self.activatedOutputHistory_std_neuron = getattr(self, 'activatedOutputHistory_std_neuron', [])
                self.activatedOutputHistory_saturation = getattr(self, 'activatedOutputHistory_saturation', [])
                self.activatedOutputHistory_min = getattr(self, 'activatedOutputHistory_min', [])
                self.activatedOutputHistory_max = getattr(self, 'activatedOutputHistory_max', [])

                self.activatedOutputHistory_std_token.append(activated.std(dim=1).mean().item())
                self.activatedOutputHistory_std_neuron.append(activated.std(dim=0).mean().item())
                self.activatedOutputHistory_saturation.append((activated.abs() < 1e-3).float().mean().item())
                self.activatedOutputHistory_min.append(activated.min().item())
                self.activatedOutputHistory_max.append(activated.max().item())

                #self.normedOutputHistory.append(normed.norm().item())
                #self.normedOutputHistory_tokens.append(normed.norm(dim=1).mean().item())
                #self.normedOutputHistory_neurons.append(normed.norm(dim=0).mean().item())

                if len(self.rawOutputHistory) >= self.numTokensPerStep:
                    self.stats = {
                        "2N_0_rawInput_norm": sum(self.rawInputHistory) / len(self.rawInputHistory),
                        "2N_0_rawInput_norm_token": sum(self.rawInputHistory_tokens) / len(self.rawInputHistory_tokens),
                        "2N_0_rawInput_norm_neuron": sum(self.rawInputHistory_neurons) / len(self.rawInputHistory_neurons),

                        "2N_1_normedInput_norm": sum(self.normedInputHistory) / len(self.normedInputHistory),
                        "2N_1_normedInput_norm_token": sum(self.normedInputHistory_tokens) / len(self.normedInputHistory_tokens),
                        "2N_1_normedInput_norm_neuron": sum(self.normedInputHistory_neurons) / len(self.normedInputHistory_neurons),

                        "2N_2_rawOutput_norm": sum(self.rawOutputHistory) / len(self.rawOutputHistory),
                        "2N_2_rawOutput_norm_token": sum(self.rawOutputHistory_tokens) / len(self.rawOutputHistory_tokens),
                        "2N_2_rawOutput_norm_neuron": sum(self.rawOutputHistory_neurons) / len(self.rawOutputHistory_neurons),

                        "2N_x_actOut_norm": sum(self.activatedOutputHistory) / len(self.activatedOutputHistory),
                        "2N_x_actOut_norm_token": sum(self.activatedOutputHistory_tokens) / len(self.activatedOutputHistory_tokens),
                        "2N_x_actOut_norm_neuron": sum(self.activatedOutputHistory_neurons) / len(self.activatedOutputHistory_neurons),

                        "2N_x_actOut_std_token": sum(self.activatedOutputHistory_std_token) / len(self.activatedOutputHistory_std_token),
                        "2N_x_actOut_std_neuron": sum(self.activatedOutputHistory_std_neuron) / len(self.activatedOutputHistory_std_neuron),
                        "2N_x_actOut_saturation": sum(self.activatedOutputHistory_saturation) / len(self.activatedOutputHistory_saturation),
                        "2N_x_actOut_min": sum(self.activatedOutputHistory_min) / len(self.activatedOutputHistory_min),
                        "2N_x_actOut_max": sum(self.activatedOutputHistory_max) / len(self.activatedOutputHistory_max),

                        #"2N_x_normedOutput_norm": sum(self.normedOutputHistory) / len(self.normedOutputHistory),
                        #"2N_x_normedOutput_norm_token": sum(self.normedOutputHistory_tokens) / len(self.normedOutputHistory_tokens),
                        #"2N_x_normedOutput_norm_neuron": sum(self.normedOutputHistory_neurons) / len(self.normedOutputHistory_neurons),
                        }

                    self.rawInputHistory = []
                    self.rawInputHistory_tokens = []
                    self.rawInputHistory_neurons = []

                    self.normedInputHistory = []
                    self.normedInputHistory_tokens = []
                    self.normedInputHistory_neurons = []

                    self.rawOutputHistory = []
                    self.rawOutputHistory_tokens = []
                    self.rawOutputHistory_neurons = []

                    self.activatedOutputHistory = []
                    self.activatedOutputHistory_tokens = []
                    self.activatedOutputHistory_neurons = []

                    self.activatedOutputHistory_std_token = []
                    self.activatedOutputHistory_std_neuron = []
                    self.activatedOutputHistory_saturation = []
                    self.activatedOutputHistory_min = []
                    self.activatedOutputHistory_max = []

                    #self.normedOutputHistory = []
                    #self.normedOutputHistory_tokens = []
                    #self.normedOutputHistory_neurons = []

        return activated

    def getStats(self): return self.stats

"""layer that applies the same set of neurons to each token embedding independently. - no sequence awareness!"""
class INTERNEURON_NETWORK(nn.Module):
    def __init__(self, _model, _counsellor, _calligraphist, _numTokensPerStep, _device = modelDevice):
        super().__init__()
        #self.inn_counsellor = COUNSELLOR("3INN", debug = debugPrints, durations = durationLogging)
        self.model = _model
        self.inn_counsellor = _counsellor
        self.device = _device
        self.calligraphist = _calligraphist
        self.numTokensPerStep = _numTokensPerStep
        self.entropyBonus = 0

        self.stats = {}

        self.activationsHistory = [] 
        self.activationsHistory_token = []
        self.activationsHistory_neuron = []

        self.normedMeanInputHistory = []
        self.normedMeanInputHistory_token = []
        self.normedMeanInputHistory_neuron = []

        self.combHistory = []
        self.combHistory_token = []
        self.combHistory_neuron = []

        self.refHistory = []
        self.refHistory_token = []
        self.refHistory_neuron = []

        self.scaledHistory = []
        self.scaledHistory_token = []
        self.scaledHistory_neuron = []

        self.combiOutHistory = [] 
        self.combiOutHistory_token = [] 
        self.combiOutHistory_neuron = [] 

        self.logitHistory = []
        self.combiScaleHistory = []

        # SELF ALLOWED - nn.parameter!
        self.neurons = NEURON(_counsellor = self.inn_counsellor, _numTokensPerStep = self.numTokensPerStep)

        self.cerebellum = nn.Parameter(torch.ones(len(allWindowSizes_new), device = self.device)) # THIS WAS THE WINDOW WEIGHTING LAYER

        self.windowFractionality = nn.Parameter(torch.full((len(allWindowSizes_new),), -6.0, device=self.device))

        self.logWindowSizes = nn.Parameter(torch.log(torch.tensor(allWindowSizes_new, dtype=torch.float32, device=self.device))) # one tensor per window size!
        self.refinement2 = torch.nn.Sequential(
                            nn.Linear(numNeurons, 512, device=self.device), # bottleneck layer
                            nn.GELU(),                                      # smoother activation
                            nn.LayerNorm(512, device=self.device),          # mid normalization
                            nn.Linear(512, numNeurons, device=self.device), # expand back
                            nn.LayerNorm(numNeurons, device=self.device)    # final safety net
                            )

        
        self.logitScale = nn.Parameter(torch.tensor(1.0, device=self.device))  
        with torch.no_grad(): self.logitScale.clamp_(0.001, 10)      
        self.combiScale = nn.Parameter(torch.tensor(1.0, device=self.device))    
        with torch.no_grad(): self.combiScale.clamp_(0.001, 10)
        self.windowMeanNorm = nn.LayerNorm(numNeurons, elementwise_affine=True, device=self.device)
        self.combiOutNorm = nn.LayerNorm(numNeurons, elementwise_affine=True, device=self.device)

        # MUST NOT BE ON SELF - global parameters that may be used by backward pass
        #numNeurons, embedDimension, activationFunction, allWindowSizes_new, etc

    @whocalled
    def forward(self, _inputEmbeds):  
        with self.inn_counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            ʕっʘ‿ʘʔっ("INN1: neuronActivationsPerToken")
            self.neuronActivationsPerToken = self.neurons(_inputEmbeds)

            ʕっʘ‿ʘʔっ("compute fresh floatWindowSizes + fractionality")
            # learnable fractionality, allows it to decide how descrite the windows should be
            fractionality = torch.sigmoid(self.windowFractionality)  # (numWindows,)
            with torch.no_grad(): self.windowFractionality.clamp_(-6.0, 6.0)
            
            minWindowSize = 0.1
            maxWindowSize = float(self.numTokensPerStep)

            #clampedLogWindowSizes = self.logWindowSizes + (self.logWindowSizes.clamp(-1.5, 1.5) - self.logWindowSizes).detach()
            clampedLogWindowSizes = torch.tanh(self.logWindowSizes / 2) * 1.5  # output ∈ [-1.5, 1.5]
            scaledTanh = (torch.tanh(clampedLogWindowSizes) + 1) / 2  # ∈ [0, 1]
            floatWindowSizes = minWindowSize + (maxWindowSize - minWindowSize) * scaledTanh
            intWindowSizes = torch.round(floatWindowSizes).clamp(min=1.0)

            ʕっʘ‿ʘʔっ("blend int+float using fractionality") # straight-through estimator: gradient flows only through floatWindow
            windowTensor = (1 - fractionality) * intWindowSizes + fractionality * floatWindowSizes
            windowTensor = windowTensor.unsqueeze(1)  # (numWindows, 1)

            # Store for stats
            self.windowTensor_used = windowTensor.squeeze(1).detach()
            self.floatWindowSizes_used = floatWindowSizes.detach()

            ʕっʘ‿ʘʔっ("INN2: stackedWindowMeans")
            windowMeanStack = self.stackedWindowMeans(self.neuronActivationsPerToken, windowTensor)

            ʕっʘ‿ʘʔっ("softmax weights from cerebellum")
            sigmoidWeights = torch.sigmoid(self.cerebellum) # squish raw values into [0, 1]
            with torch.no_grad(): self.cerebellum.clamp_(-5.0, 5.0)
            clamped = torch.clamp(sigmoidWeights, min=1e-4) # avoid 0s
            self.cerebellumSoft = clamped / clamped.sum()   # normalize across all windows

            weightedWindowStack = windowMeanStack * self.cerebellumSoft.unsqueeze(1)

            ʕっʘ‿ʘʔっ("entropy reward")
            self.windowEntropy = -torch.sum(self.cerebellumSoft * torch.log(self.cerebellumSoft + 1e-12))
            self.entropyBonus = self.windowEntropy

            ʕっʘ‿ʘʔっ("combine + refine")
            combinedActivationsTensor = weightedWindowStack.sum(dim=0, keepdim=True)
            refinedActivations = self.refinement2(combinedActivationsTensor)

            FINALout = refinedActivations

            # --- logging
            self.activationsHistory.append(self.neuronActivationsPerToken.norm().item())
            self.activationsHistory_token.append(self.neuronActivationsPerToken.norm(dim=1).mean().item())
            self.activationsHistory_neuron.append(self.neuronActivationsPerToken.norm(dim=0).mean().item())
            self.combHistory.append(combinedActivationsTensor.norm().item())
            self.combHistory_neuron.append(combinedActivationsTensor.norm(dim=0).mean().item())
            self.refHistory.append(refinedActivations.norm().item())
            self.refHistory_neuron.append(refinedActivations.norm(dim=0).mean().item())

            if len(self.combHistory) >= self.numTokensPerStep:
                self.stats = {
                    "3INN_0_rawActivations_norm": sum(self.activationsHistory) / len(self.activationsHistory),
                    "3INN_0_rawActivations_norm_token": sum(self.activationsHistory_token) / len(self.activationsHistory_token),
                    "3INN_0_rawActivations_norm_neuron": sum(self.activationsHistory_neuron) / len(self.activationsHistory_neuron),
                    "3INN_2_combinedActivations_norm": sum(self.combHistory) / len(self.combHistory),
                    "3INN_2_combinedActivations_norm_neuron": sum(self.combHistory_neuron) / len(self.combHistory_neuron),
                    "3INN_x_refinedActivations_norm": sum(self.refHistory) / len(self.refHistory),
                    "3INN_3_refinedActivations_norm_neuron": sum(self.refHistory_neuron) / len(self.refHistory_neuron),
                    "3INN_windowSizesMean": floatWindowSizes.mean().item()
                }

                self.activationsHistory = []
                self.activationsHistory_token = []
                self.activationsHistory_neuron = []
                self.combHistory = []
                self.combHistory_neuron = []
                self.refHistory = []
                self.refHistory_neuron = []

            return FINALout
    
    def stackedWindowMeans(self, activations: torch.Tensor, windowTensor: torch.Tensor) -> torch.Tensor:
        """ Fully vectorized, loop-free window mean calculation. Expects: windowTensor already built. """
        with self.inn_counsellor.infodump("stackedWindowMeans") as ʕっʘ‿ʘʔっ:
            #self.neuronActivationsPerToken = activations
            seqLen, embedDim = activations.shape

            # PADDING ENSURES UNDERLYING DATA HAS CORRECT TENSOR/VECTOR SHAPE FOR THE MASK
            padded = torch.zeros((self.numTokensPerStep, embedDim), device=self.device)
            padded[-min(seqLen, self.numTokensPerStep):] = activations[-min(seqLen, self.numTokensPerStep):]

            stackedWindows = padded.unsqueeze(0).repeat(windowTensor.shape[0], 1, 1)

            # THE RANGE MASK IS ONLY EVER AS LONG AS WINDOWMAX, SO THAT WINDOWS DONT EXCEED IT
            rangeMask = torch.arange(self.numTokensPerStep, device=self.device).unsqueeze(0)  # (1, maxW)

            #'mask' CHECKS TO SEE HOW LONG WINDOWS ARE, AND IF THEY ARE LONGER CROPS IT BEFORE MEANING
            mask = (rangeMask < windowTensor).float().unsqueeze(2)  # (numWindows, maxW, 1)

            maskedWindows = stackedWindows * mask
            sums = maskedWindows.sum(dim=1)  # (numWindows, embedDim)
            means = sums / windowTensor  # divide each by its window length

            return means
    
    def INN_getStats(self):
        with self.inn_counsellor.infodump("INN_getStats") as ʕっʘ‿ʘʔっ:
            INN_cerebellum_str = ""
            if collectStats and n_collectStats:
                ʕっʘ‿ʘʔっ("torch.no_grad♥")
                with torch.no_grad():
                    if n_weightStats:
                        ʕっʘ‿ʘʔっ("♥n_weightStats")
                        self.stats["2N_weightMean"] = self.neurons.n_weights.mean()
                        self.stats["2N_weightStd"] = self.neurons.n_weights.std()
                        self.stats["2N_weightMin"] = self.neurons.n_weights.min()
                        self.stats["2N_weightMax"] = self.neurons.n_weights.max()
                        if debugPrints: print(f"neuron weight mean: {self.stats["2N_weightMean"]} std: {self.stats["2N_weightStd"]} min: {self.stats["2N_weightMin"]} max: {self.stats["2N_weightMax"]}")
                    
                    if n_weightNormStats:
                        ʕっʘ‿ʘʔっ("♥n_weightNormStats")
                        self.n_weightNorm = torch.norm(self.neurons.n_weights, dim = 1)
                        self.stats["2N_weightNormMean"] = self.n_weightNorm.mean()
                        self.stats["2N_weightNormMin"] = self.n_weightNorm.min()
                        self.stats["2N_weightNormMax"] = self.n_weightNorm.max()
                        if debugPrints: print(f"neuron weightNorm: {self.n_weightNorm} mean: {self.stats["2N_weightNormMean"]} min: {self.stats["2N_weightNormMax"]} max: {self.stats["2N_weightNormMin"]}")

                    if n_biasesStats:
                        ʕっʘ‿ʘʔっ("♥n_biasesStats")                    
                        self.stats["2N_biasesMean"] = self.neurons.n_biases.mean()
                        self.stats["2N_biasesStd"] = self.neurons.n_biases.std()
                        self.stats["2N_biasesMin"] = self.neurons.n_biases.min()
                        self.stats["2N_biasesMax"] = self.neurons.n_biases.max()
                        if debugPrints: print(f"neuron biases mean: {self.stats["2N_biasesMean"]} std: {self.stats["2N_biasesStd"]} min: {self.stats["2N_biasesMin"]} max: {self.stats["2N_biasesMax"]}")

                    if n_sparsityStat:
                        ʕっʘ‿ʘʔっ("♥getSparsityStat")
                        self.stats["2N_sparsity"] = (self.neurons.n_weights.abs() < 1e-5).float().mean()
                        if debugPrints: print(f"neuron sparsity: {self.stats["2N_sparsity"]}")

            if collectStats and INN_collectStats:
                ʕっʘ‿ʘʔっ("torch.no_grad♥")
                with torch.no_grad():
                    if INN_cerebellumStats:
                        ʕっʘ‿ʘʔっ("♥getCerebellumStats") #THIS WAS WINDOWWEIGHTING
                        self.stats["3INN_windowFractionalityMean"] = torch.sigmoid(self.windowFractionality).mean().item()
                        self.stats["3INN_cerebellumMean"] = self.cerebellum.mean().item()
                        self.stats["3INN_cerebellumStd"] = self.cerebellum.std().item()
                        INN_cerebellumStats_fullValues = zip(self.floatWindowSizes_used, self.cerebellum, self.cerebellumSoft, self.windowTensor_used)
                        for w, raw, soft, tensor in INN_cerebellumStats_fullValues:
                            self.stats[f"INN_cerebellum_W{int(w)}_float"] = w.item()
                            self.stats[f"INN_cerebellum_W{int(w)}"] = raw.item()
                            self.stats[f"INN_cerebellum_W{int(w)}_softMax"] = soft.item()
                            self.stats[f"INN_cerebellum_W{int(w)}_tensor"] = tensor.item()
                        if debugPrints: print(f"cerebellum: {self.cerebellum}, soft: {self.cerebellumSoft} mean: {self.stats['3INN_cerebellumMean']} std: {self.stats['3INN_cerebellumStd']}")
                        ʕっʘ‿ʘʔっ("♥cerebellumString")
                        INN_cerebellum_str = self.calligraphist.S_formatWindowBiasTriplets(label="INN_cerebellum", rawTensor = self.cerebellum, softTensor = self.cerebellumSoft, windowSizes = self.floatWindowSizes_used, windowTensor = self.windowTensor_used, per_window_style = True)
                        if debugPrints: print(f"{INN_cerebellum_str}")

            self.stats.update({f"{k}": v for k, v in self.neurons.getStats().items()})
        return self.stats, INN_cerebellum_str
    
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