# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# BABYLLM // babyLLM.py

import random, os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim 
from adan_pytorch import Adan
import torch.optim.lr_scheduler
import math
from collections import Counter

from BRAIN.LAYERS.embed import EMBED
from BRAIN.LAYERS.interneuronNetwork import INTERNEURON_NETWORK
from BRAIN.LAYERS.logits import LOGITS
from BRAIN.LAYERS.memory import MEMORY
#from BRAIN.LAYERS.sensoryWobble import WOBBLE
from config import *

"""this class combines all the core components of the babyLLM:"""
"""EMBED: token embedding layer"""
"""INTERNEURON_NETWORK: layer of parallel neurons for feature extraction"""
"""LOGITS: output layer to generate logits"""
"""it also manages training, loss computation, backpropagation, and response generation."""
class BABYLLM(nn.Module):
    def __init__(self, _counsellor, _calligraphist, _scribe, _librarian, _numTokensPerStep, _learningRateGOAL = learningRateGOAL, _device = modelDevice, _first = True):
        super().__init__()
        self.device = _device
        self.counsellor = _counsellor
        self.calligraphist = _calligraphist
        self.scribe = _scribe
        self.librarian = _librarian
        self.numTokensPerStep = _numTokensPerStep
        #self.wobble = _wobble

        # MUST BE ON SELF - ONLY ACCESSED IN THIS CLASS AND NOT NN.PARAMS
        self.totalTokenEvaluations = 0
        self.learningRateGOAL = _learningRateGOAL
        self.latestLossDelta = 0
        self.totalTokenEvaluations_A = 0
        self.recentGeneratedTokens = []  # used for repetition penalty
        self.lastLossBaby = 0
        self.computeLossCount = 0
        self.repeatedPercent = 0
        self.normalisedActivations = 0
        self.rollingTokenTotals = Counter()
        self.gumBellend = 0
        self.pixelLoss_used = 0

        self.stats = {}
        self.normalisedHistory = []
        self.INNOutputHistory = []
        self.memoryOutputHistory = []
        self.penalisedOutputHistory = []
        self.inputEmbedsHistory = []
        self.FINALlogitsHistory = []
        self.predPixel = torch.tensor([0.0, 0.0, 0.0], device=self.device)


        """CEREBRAL LAYERS // BRAIN"""
        self.embed = EMBED(_counsellor = self.counsellor, _device = self.device)
        self.interneuronNetwork = INTERNEURON_NETWORK(_model = BABYLLM, _counsellor = self.counsellor, _calligraphist = self.calligraphist, _device = self.device, _numTokensPerStep = self.numTokensPerStep)
        self.logits = LOGITS(_counsellor = self.counsellor, _device = self.device, _numTokensPerStep=self.numTokensPerStep)
        self.memory = MEMORY(_counsellor = self.counsellor, _device = self.device, _numTokensPerStep = self.numTokensPerStep)
        self.finalNormLayer = nn.LayerNorm(numNeurons, device=self.device)
        self.pixelPupil = nn.Sequential(nn.Linear(embedDimension, embedDimension), nn.GELU(), nn.Linear(embedDimension, 3), nn.Sigmoid())

        """LEARNABLE LEARNING PARAMETERS"""
        self.repetitionPenalty = nn.Parameter(torch.tensor(1.0, device = self.device))
        self.logTemp = nn.Parameter(torch.tensor(math.log(0.8), device = self.device))
        self.logLR = nn.Parameter(torch.tensor(math.log(1e-4), device = self.device))
        self.logGradClip = nn.Parameter(torch.tensor(math.log(1.0), device = self.device))
        self.scheduledSamplingRate = nn.Parameter(torch.tensor(0.2, device = self.device))
        self.logMemoryLength = nn.Parameter(torch.tensor(math.log(memoryLengthGOAL), device = self.device))
        self.logRepetitionWindow = nn.Parameter(torch.tensor(math.log(repetitionWindowGOAL), device = self.device))

        self.blendPixel = nn.Parameter(torch.tensor(0.05, device=self.device))  # 0 = token only, 1 = pixel only
        #self.pixelPupil = nn.Sequential(nn.Linear(embedDimension, embedDimension), nn.GELU(), nn.Linear(embedDimension, 3))


        """stuff"""
        self.gradientClipMaxNorm = torch.exp(self.logGradClip)
        self.temperature = None

        """OPTIMIZER - this updates all of the layers learnable parameters"""
        if debugPrints: 
            print("registered parameters: ")
            for name, param in BABYLLM.named_parameters(self): print(name, param.shape)

        if optimizerName == "Adan":
            self.optimizer = Adan(self.parameters(), lr=learningRate, betas=(0.02, 0.08, 0.01), eps=1e-6, weight_decay=0.005)
        else:
            optimizerClass = getattr(optim, optimizerName)
            self.optimizer = optimizerClass(self.parameters(), lr=learningRate, weight_decay=0.005, fused=True)


        if debugPrints:
            for name, param in self.named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")

        #self.to(self.device)
        self.statsCategories = {"loss": 0, "gradNorm": 0, "logitMin": 0, "logitMax": 0, "scheduledSamplingRate": 0, "tokenCount": 0, "memoryGateShort": 0, "memoryGateLong": 0, "memoryGateCurrent": 0, "shortDecay": 0, "longDecay": 0,}

    @whocalled
    def forward(self, _inputSeq = None, _pixel = None):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ: # processes input sequence of tokens (str) to generate logits to predict the next token
            if debugPrints: print(f"Debug: Input to forward: {_inputSeq}")
            self.temperature = torch.exp(self.logTemp)
            self.pixel = _pixel

            ʕっʘ‿ʘʔっ("B0: inputEmbeds") # convert indices to embeddings
            #inputEmbeds = self.embed(_inputSeq) # DIRECTLY TAKING A TENSOR NOW
            tokenEmbed = self.embed(_tokenIndex = _inputSeq)
            if not skipPixels and (_pixel is not None):
                rgbEmbed = self.embed(_pixel = _pixel)
                blendPixelClamped = self.blendPixel.clamp(0.0, 1.0)
                inputEmbeds = ((1.0 - blendPixelClamped) * tokenEmbed) + (blendPixelClamped * rgbEmbed)
                #inputEmbeds = (tokenEmbed * 0.95) + (rgbEmbed * 0.05) # or a learnable fusion
                #inputEmbeds = tokenEmbed
            else:
                inputEmbeds = tokenEmbed
            self.latestTokenEmbed = inputEmbeds
            if debugPrints: print(f"Debug BABYLLM.forward: inputEmbeds requires_grad: {inputEmbeds.requires_grad} [EXPECTED: TRUE]")

            ʕっʘ‿ʘʔっ("B1: interneuronNetworkOutput") # PARALLEL NEURON LAYER input/processing (feature extraction)
            INNOutput = self.interneuronNetwork.forward(inputEmbeds) 
            if debugPrints: print(f"Debug BABYLLM.forward: interneuronNetworkOutput length: {len(INNOutput)}") 
            if debugPrints: print("combinedActivationsTensor.requires_grad:", INNOutput.requires_grad)
            if debugPrints: print("combinedActivationsTensor.grad_fn:", INNOutput.grad_fn)

            ʕっʘ‿ʘʔっ("B2: memoryOutput") # MEMORY LAYER PROCESSING - NOW PROCESS THE COMBINED ACTIVATIONS
            if skipMemory:
                if debugPrints: print("skipping memory layer...")
                #self.latestMemGates = torch.tensor([0.0, 0.0, 1.0], device = self.device)  # dummy gates
                memoryOutput = INNOutput.detach()  # no grad path, super light
            else:
                memoryOutput = self.memory.forward(INNOutput)
                #self.latestMemGates = self.memory.latestMemoryGates

            ʕっʘ‿ʘʔっ("B3: logits.forward BEFORE penalty")
            logitsBeforePenalty = self.logits.forward(memoryOutput)
            if debugPrints: print("combinedActivations.requires_grad:", memoryOutput.requires_grad)

            ʕっʘ‿ʘʔっ("B4: applyRepetitionPenalty to logits")
            if not torch.isfinite(self.logRepetitionWindow):
                print("logRepetitionWindow has gone non-finite. Resetting.")
                self.logRepetitionWindow.data = torch.tensor(math.log(repetitionWindowGOAL), device = self.device)
            penalisedLogits = self.applyRepetitionPenalty(logitsBeforePenalty)
            
            if debugPrints: print("before memory output requires_grad?", self.memory.longTermMemory.requires_grad)
            if debugPrints: print("before cerebellum requires_grad?", self.interneuronNetwork.cerebellum.requires_grad)
            if debugPrints: print("before logRepetitionWindow requires_grad?", self.logRepetitionWindow.requires_grad)
            if debugPrints: print("before logMemoryLength requires_grad?", self.logMemoryLength.requires_grad)
            if skipFINALlogitNorm:
                ʕっʘ‿ʘʔっ("Bx: logits.forward")
                FINALlogits = penalisedLogits
                #FINALlogits = self.logits.forward(memoryOutput)
            if False:
                ʕっʘ‿ʘʔっ("B4: finalNormLayer")
                self.normedOutput = self.finalNormLayer(penalisedOutput)
                FINALlogits = self.logits.forward(self.normedOutput) 
                self.normalisedHistory.append(self.normedOutput.norm().item())
            if debugPrints: print("AFTER logMemoryLength requires_grad?", self.logMemoryLength.requires_grad)
            if debugPrints: print("AFTER logRepetitionWindow requires_grad?", self.logRepetitionWindow.requires_grad)
            if debugPrints: print("AFTER cerebellum requires_grad?", self.interneuronNetwork.cerebellum.requires_grad)
            if debugPrints: print("AFTER memory output requires_grad?", self.memory.longTermMemory.requires_grad)

            if True:
                ʕっʘ‿ʘʔっ("stats collection!")
                self.inputEmbedsHistory.append(inputEmbeds.norm().item())
                self.INNOutputHistory.append(INNOutput.norm().item())
                self.memoryOutputHistory.append(memoryOutput.norm().item())
                self.penalisedOutputHistory.append(penalisedLogits.norm().item())
                self.FINALlogitsHistory.append(FINALlogits.norm().item())

                if len(self.inputEmbedsHistory) >= self.numTokensPerStep:
                    self.forwardStats = {
                        "2B_0_inputEmbeds_norm": sum(self.inputEmbedsHistory) / len(self.inputEmbedsHistory),
                        "3B_1_INNOutput_norm": sum(self.INNOutputHistory) / len(self.INNOutputHistory),
                        "5B_0_memoryOutput_norm": sum(self.memoryOutputHistory) / len(self.memoryOutputHistory),
                        "7B_1_penalisedOutput_norm": sum(self.penalisedOutputHistory) / len(self.penalisedOutputHistory),
                        #"5B_x_finalNormLayer_norm": sum(self.normalisedHistory) / len(self.normalisedHistory),
                        "7B_x_FINALlogits_norm": sum(self.FINALlogitsHistory) / len(self.FINALlogitsHistory),
                        "B_blendPixel": self.blendPixel.item(),
                    }
                    self.stats.update(self.forwardStats)
                    
                    self.inputEmbedsHistory = []
                    self.INNOutputHistory = []
                    self.memoryOutputHistory = []
                    self.penalisedOutputHistory = []
                    self.FINALlogitsHistory = []
                    self.normalisedHistory = []

            """returns a logits tensor of shape (1, vocabSize) showing predicted probabilities for the next token"""
            #tokenEmbed = self.embed(_tokenIndex = _inputSeq)
            #self.latestTokenEmbed = tokenEmbed
            return FINALlogits #, self.latestTokenEmbed

    """computes the cross-entropy loss between the models logits and the target token, essentially checking how good the models prediction was"""        
    def computeLoss(self, _logits, _targetTokenIndex, _totalAvgAbsDelta = 1, _learningRateGOAL = learningRateGOAL, _perfectTokens = 0, _training = False):
        with self.counsellor.infodump("computeLoss") as ʕっʘ‿ʘʔっ:
            self.perfectTokens = _perfectTokens
            self.totalAvgAbsDelta = _totalAvgAbsDelta
            self.learningRateGOAL = _learningRateGOAL
            if skipComputeLoss:
                ʕっʘ‿ʘʔっ("skipping loss!")
                return torch.tensor([0.1], requires_grad = True, device = self.device)  # Constant scalar tensor
            
            ʕっʘ‿ʘʔっ("targetTensor")          
            targetTensor = torch.tensor([_targetTokenIndex], dtype = torch.long, device = self.device)
            
            if debugPrints: print(f"logits shape: {_logits.shape} | target: {_targetTokenIndex}")
            if _logits.dim() == 1: 
                _logits = _logits.unsqueeze(0) # ensure logits are at least 2d
            
            ʕっʘ‿ʘʔっ("cross Entropy Loss")
            #LOSSlogits = torch.clamp(_logits, min=-50, max=50)
            loss = F.cross_entropy(_logits, targetTensor)

            if not torch.isfinite(loss):
                print("NaN/Inf loss detected — logits:", _logits)
                return torch.tensor(10.0, device = self.device, requires_grad = True)  # or skip/backoff

            if debugPrints: print(f"crossentropy raw loss: {F.cross_entropy(_logits, targetTensor)}")
            
            self.CELossDelta = loss - ((self.lastLossBaby) if self.lastLossBaby is not None else 0)
            #tempReg = (torch.clamp(self.logTemp, 0.7, 0.9) - 0.8).pow(2)

            if debugPrints: print(f"{self.lastLossBaby:0.1f}", end = ", ") # take delta

            #entropy = 0.001 * self.interneuronNetwork.entropyBonus

            lrSoftClamp = 0.001 * (self.logLR - math.log(learningRateGOAL)).pow(2)
            #lrSoftClamp = (self.totalAvgAbsDelta ** 1.5) * (self.logLR - math.log(self.learningRateGOAL)).pow(2)
            tempSoftClamp = 2 * (self.logTemp - math.log(temperatureGOAL)).pow(2)
            if self.repetitionPenalty >= 0: 
                repetitionPenaltySoftClamp = 0.000000000001 * (self.repetitionPenalty - repetitionPenaltyGOAL).pow(2)
            elif self.repetitionPenalty >= -1:
                repetitionPenaltySoftClamp = 0.0000001 * (self.repetitionPenalty - repetitionPenaltyGOAL).pow(2)
            elif self.repetitionPenalty < -1:
                repetitionPenaltySoftClamp = 0.0002 * (self.repetitionPenalty - repetitionPenaltyGOAL).pow(2)
            elif self.repetitionPenalty < 0:
                repetitionPenaltySoftClamp = 0.00002 * (self.repetitionPenalty - repetitionPenaltyGOAL).pow(2)

            loss += lrSoftClamp # use .detach() to avoid .backward()
            loss += tempSoftClamp
            loss += repetitionPenaltySoftClamp
            self.lastLossBaby = loss.item()
            FINALloss = loss
            #print(f"{FINALloss} + loss")

            if _training and self.lastSoftSample is not None and not skipAuxLoss:
                target = F.one_hot(targetTensor, num_classes = _logits.shape[1]).float()
                auxLoss = F.kl_div(self.lastSoftSample.log(), target, reduction = 'batchmean')
                AUXloss = auxLoss * torch.sigmoid(loss - auxLoss) # low weight for anti-dominatrix
                #print(f"{AUXloss} + aux")

            if not skipPixels and (self.nextPixelTarget is not None and hasattr(self, "pixelPupil")):
                ʕっʘ‿ʘʔっ("RGB regression loss")
                if debugPrints: print(f"latestTokenEmbed is {self.latestTokenEmbed} ({self.latestTokenEmbed.shape}), [-1] is {self.latestTokenEmbed[-1]} ({self.latestTokenEmbed[-1].shape})")
                predictedRGB = self.pixelPupil(self.latestTokenEmbed[-1])
                self.predPixel = predictedRGB
                rgbLoss = F.mse_loss(self.predPixel, self.nextPixelTarget)
                #self.PIXELloss = rgbLoss * torch.sigmoid(loss - rgbLoss)
                pixelWeight = rgbLoss / (rgbLoss + loss)
                self.PIXELloss = max(min((pixelWeight * 1), 1),-1)
                #self.print_rgb_block(self.pixel, "prompt")
                #self.print_rgb_block(predictedRGB, "guess")
                #self.print_rgb_block(self.nextPixelTarget, "truth")
                #print(f"{rgbLoss} + rgb")
                #print(f"{self.PIXELloss} + pixel")

            else:
                FINALloss = loss
                #print(f"{FINALloss} + final")

            #tempSoftClamp = 0.4 * (self.logTemp - math.log(0.5)).pow(2)

                # more tokens (better) > perfTokens > less tokens (worse)
                # HIGHER NUMBER > 2 > LOWER NUMBER
                # 0.3x > 2 > 1.3x

                # worse (explore) > latestlossdelta > better (stay still)
                # POSITIVE NUMBER > 0 > NEGATIVE NUMBER 
                # +4 Delta (worse) > 0 > -4 Delta (better)
                # [0-25]x0.1 > 0 > [0-1]
                # 0-2.5 > 0 > 0-1
            if not skipPixels and (self.nextPixelTarget is not None and hasattr(self, "pixelPupil")): 
                FINALloss += (self.PIXELloss * 50)
                self.pixelLoss_used = (self.PIXELloss * 50)
                #print(f"{FINALloss} pixel + final")
            if _training and self.lastSoftSample is not None and not skipAuxLoss: 
                FINALloss += AUXloss
                #print(f"{FINALloss} aux + final")
            #if debugPrints: print(f"[LOSS DEBUG] requires_grad: {loss.requires_grad} | value: {loss.detach().cpu().item():.4f}")
            return FINALloss
    
    """backpropagation and optimization, computes gradients of the loss and uses the optimizer to update the models weights"""
    def backward(self, _loss):
        with self.counsellor.infodump("backward") as ʕっʘ‿ʘʔっ:
            if debugPrints:
                for name, p in self.named_parameters():
                    if p.grad is None:
                        print(f"before = {self.calligraphist.S_apply("dim", f"no grad: {name}")}")
                    else:
                        grad = p.grad
                        shape = tuple(grad.shape)
                        norm = grad.norm().item()
                        nonzero = grad.count_nonzero().item()
                        total = grad.numel()
                        sparsity = 1 - (nonzero / total)
                        mean = grad.mean().item()
                        std = grad.std().item()
                        print(f"before = {self.calligraphist.S_apply("almostPerfect", f"yes grad: {name} | shape: {shape} | norm: {norm:.4f} | sparsity: {sparsity:.2%} | mean: {mean:.4f} | std: {std:.4f}")}")
            ʕっʘ‿ʘʔっ("loss.backward")
            if debugPrints: print("Loss:", _loss.item())
            _loss.backward()
            if debugPrints: print("Logit weights grad norm:", self.logits.l_weights.grad.norm())
            if debugPrints: print("LogWindowSizes grad norm:", self.interneuronNetwork.logWindowSizes.grad.norm())
            if debugPrints: print("Cerebellum grad norm:", self.interneuronNetwork.cerebellum.grad.norm())
            if debugPrints: print("Repetition penalty grad norm:", self.repetitionPenalty.grad.norm())
            #print(next(self.parameters()).grad)
            if debugPrints:
                for name, p in self.named_parameters():
                    if p.grad is None:
                        print(f"after = {self.calligraphist.S_apply("emergency", f"NO GRAD: {name}")}")
                    else: 
                        grad = p.grad
                        shape = tuple(grad.shape)
                        norm = grad.norm().item()
                        nonzero = grad.count_nonzero().item()
                        total = grad.numel()
                        sparsity = 1 - (nonzero / total)
                        mean = grad.mean().item()
                        std = grad.std().item()
                        print(f"after = {self.calligraphist.S_apply("almostPerfect", f"yes grad: {name} | shape: {shape} | norm: {norm:.4f} | sparsity: {sparsity:.2%} | mean: {mean:.4f} | std: {std:.4f}")}")

            with torch.no_grad(): # RESET LEARNABLE PARAMETERS
                #self.logLR.data.fill_(math.log(0.00035))  # Learning rate back to 1e-4
                self.scheduledSamplingRate.data.fill_(0.001)  # Scheduled sampling full (no scheduled sampling yet)
                #self.temperature.data.fill_(math.exp(self.logTemp))  # Temperature normal
                #self.repetitionPenalty.data.fill_(1.0)  # Repetition penalty normal
                self.logMemoryLength.data.fill_(math.log(2))  # Memory length default
                #self.logRepetitionWindow.data.fill_(math.log(16))  # Repetition window default
                #self.interneuronNetwork.logWindowSizes.data.copy_(
                #    torch.log(torch.tensor(allWindowSizes_new, dtype=torch.float32, device=self.device))
                #)
                #for module in self.interneuronNetwork.windowMeta:
                #    if isinstance(module, torch.nn.Linear):
               #        module.reset_parameters()

            if True:
                with torch.no_grad():
                    self.logLR.clamp_(math.log(0.0001), math.log(0.001))  # CLAMP IT! IN MEMORY OF THE AMAZING 1.00 SELF LEARNED LOSS RUN OF 27-APRIL-2025! - you certainly dropped the delta! you win!
                learnedLR = torch.exp(self.logLR).item()
                for g in self.optimizer.param_groups:
                    g['lr'] = learnedLR
                #self.gradientClipMaxNorm = torch.exp(self.logGradClip).item()
                #self.repetitionWindow = torch.exp(self.logRepetitionWindow).item()
                self.memoryLength = torch.exp(self.logMemoryLength).item()
                #self.logLR.data.fill_(self.logLR+0.000001) # increment LR manually (break grid)

            ʕっʘ‿ʘʔっ("clip_grad_norm")
            clipValue = torch.exp(self.logGradClip).item()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clipValue)
            ʕっʘ‿ʘʔっ("optimizer.step")
            self.optimizer.step()  # Update weights
            repWindow = torch.exp(self.logRepetitionWindow)
            self.repetitionWindow = repWindow / (1 + repWindow / self.numTokensPerStep)  # asymptotes near windowMAX

            self.backwardStats = {
                "B_floatMemoryLength": torch.exp(self.logMemoryLength).item(),
                "B_expWindow": repWindow.item(),
                "B_repetitionWindow": self.repetitionWindow.item(),
                "B_temperature": torch.exp(self.logTemp).item(),
                "B_PIXELloss": self.PIXELloss,
                "B_PIXELloss_scaled": self.pixelLoss_used
            }
            self.stats.update(self.backwardStats)
            self.pixelLoss_used = 0

            #with torch.no_grad(): # FORCE RESET THE MEMORY GATES IF OVER USING LONG
                #self.memory.currentGate.data = self.memory.currentGate.data.abs()
                #self.memory.shortGate.data = self.memory.shortGate.data.abs()
                    
    """this takes the output logits, does temperature scaling and softmax to create a probability distribution over the vocab, and then selects most likely response token"""
    """def getResponseFromLogits(self, _logits, _temperature = temperature):
        with self.counsellor.infodump("getResponseFromLogits") as ʕっʘ‿ʘʔっ:
            _logits /= _temperature
            if debugPrints: print(f"Debug BABYLLM.getResponseFromLogits: logits shape BEFORE softmax: {_logits.shape}")
            if _logits.dim() == 1: _logits = _logits.unsqueeze(0)
            probs = torch.softmax(_logits, dim = 1)
            responseFromLogits = torch.multinomial(probs, 1)
            return responseFromLogits"""

    @whocalled
    def getResponseFromLogits(self, _logits, _training = False):
        with self.counsellor.infodump("getResponseFromLogits") as ʕっʘ‿ʘʔっ:
            if not torch.isfinite(_logits).all():
                print("logits not finite before response gen:", _logits)
                _logits = torch.nan_to_num(_logits, nan=0.0, posinf=1e3, neginf=-1e3)
            ʕっʘ‿ʘʔっ("update logarithmic parameters")
            #self.repetitionWindow = torch.exp(self.logRepetitionWindow)#.clamp(min=1.0)
            self.temperature = torch.exp(self.logTemp)  # TORCH.exp keeps gradient path!
            _logits /= self.temperature
            ʕっʘ‿ʘʔっ("check for nan logits")
            if torch.isnan(_logits).any():
                ʕっʘ‿ʘʔっ("NaN yes = nan_to_num on _logits")
                print("NaN in logits after temperature scaling!")
                print("logTemp:", self.logTemp.item(), "Temp:", self.temperature.item())
                print("logits stats:", _logits.min().item(), _logits.max().item(), _logits.mean().item())
                _logits = torch.nan_to_num(_logits, nan=0.0, posinf=1e3, neginf=-1e3)

            ʕっʘ‿ʘʔっ("if logits dim(1), unsqueeze(0)")
            if _logits.dim() == 1: _logits = _logits.unsqueeze(0)  # ensure [1, vocabSize]

            if _training:
                ʕっʘ‿ʘʔっ("training, use gumbel")
                ʕっʘ‿ʘʔっ("cloning _logits to logitForSample")
                logitsForSample = _logits.clone()
                if not torch.isfinite(logitsForSample).all():
                    ʕっʘ‿ʘʔっ("non-finite logits detected BEFORE GUMBEL, nan_to_num logitsForSample")
                    print("non-finite logits detected BEFORE GUMBEL")
                    print("logits:", logitsForSample)
                    logitsForSample = torch.nan_to_num(logitsForSample, nan=0.0, posinf=1e3, neginf=-1e3)
                try:
                    ʕっʘ‿ʘʔっ("gumbel softmax")
                    gumbelProbs = F.gumbel_softmax(logitsForSample, tau=self.temperature, hard=False)
                    assert torch.isfinite(gumbelProbs).all(), "gumbelProbs has NaN or Inf!"
                except Exception as e:
                    self.gumBellend += 1
                    ʕっʘ‿ʘʔっ("gumbel softmax failed")
                    print("gumbel softmax failed:", e)
                    print(f"falling back to softmax sampling (total fallbacks: {self.gumBellend})...")
                    gumbelProbs = torch.softmax(logitsForSample, dim=1)

                self.lastSoftSample = gumbelProbs
                responseFromLogits = gumbelProbs.argmax(dim = 1, keepdim = True)
                self.lastSoftSample = gumbelProbs

                ʕっʘ‿ʘʔっ("topK sampling")
                topk = torch.topk(gumbelProbs, 10, dim=1)
                indices = topk.indices[0].tolist()
                values = topk.values[0].tolist()
                #self.lastTopGuesses = []
                ʕっʘ‿ʘʔっ("forloop get rolling token totals")
                for i, p in zip(indices, values):
                    token = self.librarian.indexToToken.get(i, "<UNK>")
                    try:
                        if isinstance(p, float) and math.isfinite(p):
                            #self.lastTopGuesses.append((token, round(p, 4)))
                            self.rollingTokenTotals[token] += round(p, 4)
                        else:
                            ʕっʘ‿ʘʔっ("skipping non-finite topk")
                            print(f"skipping non-finite top guess: {token} → {p}")
                    except Exception as e:
                        print(f"error processing top guess: {token} → {p} | {e}")

                #print("Top guesses + confidences:", [(self.librarian.indexToToken[i.item()], f"{p.item():.3f}") for i, p in zip(indices, values)])

            else:
                ʕっʘ‿ʘʔっ("not training, using softmax")
                probs = torch.softmax(_logits, dim=1)
                ʕっʘ‿ʘʔっ("multinomial")
                responseFromLogits = torch.multinomial(probs, 1)
                self.lastSoftSample = None  # or keep the probs if you want analysis

            #if debugPrints:
                #print(f"[REP PENALTY] {self.repeatedPercent:.2%} repeated | repetition slice: {self.repetitionSlice} | Penalised: {[self.librarian.indexToToken.get(t, '<UNK>') for t in uniqueTokens]}")

            ʕっʘ‿ʘʔっ("create windows using rolling buffer")
            repWindow = torch.exp(self.logRepetitionWindow)
            self.repetitionWindow = repWindow / (1 + repWindow / self.numTokensPerStep)  # asymptotes near windowMAX
            self.recentGeneratedTokens.append(responseFromLogits.item())
            if len(self.recentGeneratedTokens) > int(self.repetitionWindow):
                self.recentGeneratedTokens.pop(0)

            return responseFromLogits
        
    def applyRepetitionPenalty(self, _logits):
        with self.counsellor.infodump("applyRepetitionPenalty") as ʕっʘ‿ʘʔっ:
            if not self.recentGeneratedTokens:
                ʕっʘ‿ʘʔっ("no recent generated tokens, returning _logits")
                return _logits

            ʕっʘ‿ʘʔっ("repWindow = torch.exp(self.logRepetitionWindow)")
            repWindow = torch.exp(self.logRepetitionWindow)
            repWindow = repWindow / (1 + repWindow / self.numTokensPerStep)  # asymptotes near windowMAX
            ʕっʘ‿ʘʔっ("penalty = self.repetitionPenalty")
            penalty = self.repetitionPenalty

            ʕっʘ‿ʘʔっ("recentTokens to tensor")
            recentTokens = torch.tensor(self.recentGeneratedTokens, device = self.device)
            ʕっʘ‿ʘʔっ("vocabSize = _logits.shape[1]")
            vocabSize = _logits.shape[1]

            ʕっʘ‿ʘʔっ("positions = torch.arange(len(recentTokens)).float()")
            positions = torch.arange(len(recentTokens), device = self.device).float()
            ʕっʘ‿ʘʔっ("windowCenter")
            windowCenter = len(recentTokens) - 0.5  # so token 0 gets proper suppression
            ʕっʘ‿ʘʔっ("softMask = torch.sigmoid((positions - (windowCenter - repWindow)) * 0.5)")
            softMask = torch.sigmoid((positions - (windowCenter - repWindow)) * 0.5)

            ʕっʘ‿ʘʔっ("oneHots")
            oneHots = F.one_hot(recentTokens, num_classes=vocabSize).float()
            ʕっʘ‿ʘʔっ("weightedFreqs = (oneHots.T @ softMask).view(1, -1)")
            weightedFreqs = (oneHots.T @ softMask).view(1, -1)

            # Scale penalty based on confidence (entropy)
            entropy = -(_logits.softmax(dim=-1) * _logits.log_softmax(dim=-1)).sum(dim=-1, keepdim=True)
            dynamicPenalty = 0.01 * penalty / (1 + entropy)  # soft & bounded

        return _logits - (weightedFreqs * dynamicPenalty)


    def getNextToken(self, _inputSeq):  
        with self.counsellor.infodump("getNextToken(FORWARD)") as ʕっʘ‿ʘʔっ:
            logits, *_ = self.forward(_inputSeq) # unpacks the first value of the tuple and ignores the rest
            nextToken = self.getResponseFromLogits(logits, _training = True)
            return nextToken
        
    def saveModel(self, _trainingStepCounter, _totalAvgLoss, _first, filePath = modelFilePath, _newStartIndex = trainingStartIndex):
        with self.counsellor.infodump("saveModel") as ʕっʘ‿ʘʔっ:
            tmpPath = filePath + ".tmp"
            torch.save(self.state_dict(), tmpPath)
            print(f"model temp file created at {tmpPath}...")
            # save optimizer to a separate file (if present)
            if hasattr(self, "optimizer") and self.optimizer is not None:
                optimPath = filePath + ".optim"
                tmpOptimPath = optimPath + ".tmp"
                torch.save(self.optimizer.state_dict(), tmpOptimPath)
                print(f"optimizer saved to {optimPath}")
                os.replace(tmpOptimPath, optimPath)
            os.replace(tmpPath, filePath)
            print(f"model successfully saved to {filePath}!")
            with open(stepCheckpointFilePath, "w") as f:
                if debugPrints or True: print(f"HELLO I AM SAVEMODEL STEPCOUNTER IS {_trainingStepCounter} AND START INDEX IS {_newStartIndex} I SHOULD WRITE {str(_trainingStepCounter+_newStartIndex)} to {stepCheckpointFilePath}")
                f.write(str(_trainingStepCounter+_newStartIndex)) # THIS ISNT REAL, FIX LATER, MAYBE MOVE SAVE AND LOAD TO WAKEUP?
            with open(lossCheckpointFilePath, "w") as f:
                if debugPrints or True: print(f"HELLO I AM SAVEMODEL AVGLOSS IS {_totalAvgLoss} I SHOULD WRITE {str(_totalAvgLoss)} to {lossCheckpointFilePath}")
                f.write(str(_totalAvgLoss))


    """loads the model from a file"""
    def loadModel(self, filePath = modelFilePath):
        with self.counsellor.infodump("loadModel") as ʕっʘ‿ʘʔっ:
            try:
                ʕっʘ‿ʘʔっ("update logarithmic parameters")
                repWindow = torch.exp(self.logRepetitionWindow)
                self.repetitionWindow = repWindow / (1 + repWindow / self.numTokensPerStep)  # asymptotes near windowMAX
                self.temperature = torch.exp(self.logTemp)  # TORCH.exp keeps gradient path!
                print(f"loading model from path: {filePath}") 
                self.load_state_dict(torch.load(filePath), strict = saveStrict)
                # try loading optimizer separately
                if hasattr(self, "optimizer"):
                    optimPath = filePath + ".optim"
                    if os.path.exists(optimPath):
                        try:
                            self.optimizer.load_state_dict(torch.load(optimPath))
                            for state in self.optimizer.state.values():
                                for k, v in state.items():
                                    if isinstance(v, torch.Tensor):
                                        state[k] = v.to(self.device)
                            print(f"optimizer restored from {optimPath}")
                        except Exception as e:
                            print(f"failed to load optimizer: {e}")
                print(f"model loaded from {filePath}!")
                self.to(self.device)
                print(f"device set to {self.device}!")
                self.resetMemory(context="inference")
                
            except FileNotFoundError: print("no saved model found")

    def babyllm_diary_entry(self, interneuronNetwork, step):
        with self.counsellor.infodump("babyllm_diary_entry") as ʕっʘ‿ʘʔっ:
            # Grab current window weightings
            weights = interneuronNetwork.cerebellum
            windows = interneuronNetwork.allWindowSizes

            # Find the current favourite and least favourite
            fav_idx = weights.argmax()
            worst_idx = weights.argmin()
            fav_window = windows[fav_idx]
            worst_window = windows[worst_idx]

            moods = ["chaotic", "curious", "crunchy", "a bit overwhelmed", "spicy", "thoughtful", "itchy", "playful"]
            actions = [
                f"I still trust window {fav_window} the most",
                f"Window {fav_window} makes me feel safe",
                f"Window {worst_window} keeps confusing me!", 
                f"I'll start listening to window {fav_window} more!",
                f"Window {worst_window} tastes like static",
                f"I'm starting to wonder about window {fav_window}... is it my destiny?",
                f"Window {worst_window} is just noise, I swear!",
                f"Today I felt {random.choice(moods)}.",
                f"Window {fav_window} whispered secrets to me."
            ]

            diaryLine = f"Step {step+1}: BabyLLM diary update: '{random.choice(actions)}'"
            print(diaryLine)

    def resetMemory(self, context="inference"):
        with self.counsellor.infodump("resetMemory") as ʕっʘ‿ʘʔっ:
            """Reset memory depending on the context: inference always resets, training resets every n turns"""
            if context == "inference": 
                ʕっʘ‿ʘʔっ("context = inference")
                self.memory.resetMemory()
                print(f"resetting memory for new conversation...")
            elif context == "training":
                ʕっʘ‿ʘʔっ("context = training")
                if hasattr(self, "stepsSinceMemoryReset"): 
                    self.stepsSinceMemoryReset += 1
                else: 
                    self.stepsSinceMemoryReset = 1
                if self.stepsSinceMemoryReset >= int(torch.exp(self.logMemoryLength).item()): 
                    self.memory.resetMemory()
                    if debugPrints: print(f"resetting memory after {self.stepsSinceMemoryReset} steps... (learned mem length: {self.logMemoryLength})")
                    self.stepsSinceMemoryReset = 0 

    def setLearningRate(self, _newLearningRate):
        self.learningRate = max(1e-6, min(_newLearningRate, 0.01))  # clamp it a bit
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learningRate

    def print_rgb_block(self, rgb_tensor, label="RGB"):
        #rgb_tensor = rgb_tensor.detach().cpu().clamp(0, 1).numpy()
        rgb_tensor = rgb_tensor.detach().cpu().numpy()

        # If it's a 1D array, convert it to shape (1, 3)
        if rgb_tensor.ndim == 1:
            print("DIM1 reshape")
            rgb_tensor = rgb_tensor.reshape(1, 3)

        for i, rgb in enumerate(rgb_tensor):
            r, g, b = (rgb * 255).astype(int)
            print(f"{label}[{i}]: \x1b[48;2;{r};{g};{b}m     \x1b[0m  ({r}, {g}, {b})")


    def getBabyStats(self): return self.stats
    
if __name__ == "__main__":
    exit(0)