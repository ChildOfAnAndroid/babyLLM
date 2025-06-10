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
import torch_optimizer
import math
from collections import Counter

from BRAIN.LAYERS.embed import EMBED
from BRAIN.LAYERS.interneuronNetwork import INTERNEURON_NETWORK
from BRAIN.LAYERS.logits import LOGITS
from BRAIN.LAYERS.memory import MEMORY
#from BRAIN.LAYERS.sensoryWobble import WOBBLE
from config import *
from secret import *

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
        self.rollingTokenTotals_tensor = torch.zeros(len(self.librarian.vocabList), device=self.device)
        self.gumBellend = 0
        self.pixelLoss_used = 0

        self.stats = {}
        self.normalisedHistory = []
        self.INNOutputHistory = []
        self.memoryOutputHistory = []
        self.totalTurns = 1
        self.memory2OutputHistory = []
        self.penalisedOutputHistory = []
        self.inputEmbedsHistory = []
        self.FINALlogitsHistory = []
        self.predPixel = torch.tensor([0.0, 0.0, 0.0], device = self.device)

        """CEREBRAL LAYERS // BRAIN"""
        self.embed = EMBED(_counsellor = self.counsellor, _device = self.device)
        self.interneuronNetwork = INTERNEURON_NETWORK(_model = BABYLLM, _counsellor = self.counsellor, _calligraphist = self.calligraphist, _device = self.device, _numTokensPerStep = self.numTokensPerStep)
        self.logits = LOGITS(_counsellor = self.counsellor, _device = self.device, _numTokensPerStep = self.numTokensPerStep)
        self.memory = MEMORY(_counsellor = self.counsellor, _device = self.device, _numTokensPerStep = self.numTokensPerStep)
        self.memory2 = MEMORY(_counsellor = self.counsellor, _device = self.device, _numTokensPerStep = self.numTokensPerStep)
        self.pixelPupil = nn.Sequential(nn.Linear(embedDimension, embedDimension), nn.GELU(), nn.Linear(embedDimension, 3), nn.Sigmoid())

        """LEARNABLE LEARNING PARAMETERS"""
        self.repetitionPenalty = nn.Parameter(torch.tensor(1.0, device = self.device))
        self.logTemp = nn.Parameter(torch.tensor(math.log(0.8), device = self.device))
        self.logLR = nn.Parameter(torch.tensor(math.log(1e-4), device = self.device))
        self.logGradClip = nn.Parameter(torch.tensor(math.log(1.0), device = self.device))
        self.scheduledSamplingRate = nn.Parameter(torch.tensor(0.2, device = self.device))
        self.logMemoryLength = nn.Parameter(torch.tensor(math.log(memoryLengthGOAL), device = self.device))
        self.logMemory2Length = nn.Parameter(torch.tensor(math.log(memoryLengthGOAL), device = self.device))
        self.logRepetitionWindow = nn.Parameter(torch.tensor(math.log(repetitionWindowGOAL), device = self.device))
        self.inputBlend = nn.Parameter(torch.ones(3, device = self.device))
        self.memoryLength = torch.sigmoid((1 - torch.exp(self.logMemoryLength)) * 0.1)
        self.memory2Length = torch.sigmoid((1 - torch.exp(self.logMemory2Length)) * 0.1)

        """stuff"""
        self.gradientClipMaxNorm = torch.exp(self.logGradClip)
        self.temperature = None

        """OPTIMIZER - this updates all of the layers learnable parameters"""
        if debugPrints: 
            print("registered parameters: ")
            for name, param in BABYLLM.named_parameters(self): print(name, param.shape)

        #baseOptim = optim.RAdam(self.parameters(), lr = learningRate)
        #baseOptim = torch_optimizer.Lion(self.parameters(), lr = 1e-4)
        #self.optimizer = optim.Lookahead(baseOptim)
        #self.optimizer = baseOptim

        if optimizerName == "Adan":
            self.optimizer = Adan(self.parameters(), lr = learningRate, betas=(0.98, 0.92, 0.99), eps = 1e-6, weight_decay = 0.005)
        else:
            optimizerClass = getattr(optim, optimizerName)
            self.optimizer = optimizerClass(self.parameters(), lr = learningRate, weight_decay = 0.005, fused = True)


        if debugPrints:
            for name, param in self.named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")

        #self.to(self.device)
        self.statsCategories = {"loss": 0, "gradNorm": 0, "logitMin": 0, "logitMax": 0, "scheduledSamplingRate": 0, "tokenCount": 0, "memoryGateShort": 0, "memoryGateLong": 0, "memoryGateCurrent": 0, "shortDecay": 0, "longDecay": 0,}

    @whocalled
    def forward(self, _inputSeq = None, _pixel = None):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ: # processes input sequence of tokens (str) to generate logits to predict the next token
            if debugPrints:
                tensor_snitch(self, "babyllm forward start")
                tensor_snitch(self.memory, "babyllm forward start")
                tensor_snitch(self.memory2, "babyllm forward start")
                tensor_snitch(self.embed, "babyllm forward start")
                tensor_snitch(self.interneuronNetwork, "babyllm forward start")
                tensor_snitch(self.logits, "babyllm forward start")
            if debugPrints: print(f"Debug: Input to forward: {_inputSeq}")
            self.temperature = torch.exp(self.logTemp)
            self.pixel = _pixel

            if debugPrints: ʕっʘ‿ʘʔっ("B0: inputEmbeds") # convert indices to embeddings
            #inputEmbeds = self.embed(_inputSeq) # DIRECTLY TAKING A TENSOR NOW
            tokenEmbed = self.embed(_tokenIndex = _inputSeq)
            seq_len = tokenEmbed.shape[0]
            pos_indices = torch.arange(seq_len, device = tokenEmbed.device)
            posEmbed = self.embed.posEmbedding(pos_indices)  # [seq_len, embed_dim]
            if not skipPixels and (_pixel is not None):
                rgbEmbed = self.embed(_pixel = _pixel)
                if debugPrints:
                    print("tokenEmbed:", tokenEmbed.shape)
                    print("posEmbed:", posEmbed.shape)
                    print("rgbEmbed:", rgbEmbed.shape)
                #blendPixelClamped = self.blendPixel.clamp(0.0, 1.0)
                #inputEmbeds = ((1.0 - blendPixelClamped) * tokenEmbed) + (blendPixelClamped * rgbEmbed)
                blend = F.softmax(self.inputBlend, dim = 0)
                inputEmbeds = blend[0] * tokenEmbed + blend[1] * posEmbed + blend[2] * rgbEmbed
            else:
                inputEmbeds = tokenEmbed
            self.latestTokenEmbed = inputEmbeds
            if debugPrints: print(f"Debug BABYLLM.forward: inputEmbeds requires_grad: {inputEmbeds.requires_grad} [EXPECTED: TRUE]")

            if debugPrints: ʕっʘ‿ʘʔっ("B1: interneuronNetworkOutput") # PARALLEL NEURON LAYER input/processing (feature extraction)
            INNOutput = self.interneuronNetwork.forward(inputEmbeds) 
            if debugPrints: print(f"Debug BABYLLM.forward: interneuronNetworkOutput length: {len(INNOutput)}") 
            if debugPrints: print("combinedActivationsTensor.requires_grad:", INNOutput.requires_grad)
            if debugPrints: print("combinedActivationsTensor.grad_fn:", INNOutput.grad_fn)

            if debugPrints: ʕっʘ‿ʘʔっ("B2: memoryOutput") # MEMORY LAYER PROCESSING - NOW PROCESS THE COMBINED ACTIVATIONS
            if skipMemory:
                if debugPrints: print("skipping memory layer...")
                memoryOutput = INNOutput.detach()  # no grad path, super light
            else:
                memoryOutput = self.memory.forward(INNOutput)
                memory2Input = (INNOutput * 0.5) + (memoryOutput * 0.5)
                memory2Output = self.memory2.forward(memory2Input)
                #self.latestMemGates = self.memory.latestMemoryGates

            if debugPrints: ʕっʘ‿ʘʔっ("B3: logits.forward BEFORE penalty")
            logitsBeforePenalty = self.logits.forward(memory2Output)
            if debugPrints: print("combinedActivations.requires_grad:", memoryOutput.requires_grad)

            if debugPrints: ʕっʘ‿ʘʔっ("B4: applyRepetitionPenalty to logits")
            if not torch.isfinite(self.logRepetitionWindow):
                print("logRepetitionWindow has gone non-finite. Resetting.")
                self.logRepetitionWindow.data = torch.tensor(math.log(repetitionWindowGOAL), device = self.device)
            penalisedLogits = self.applyRepetitionPenalty(logitsBeforePenalty)
            
            if debugPrints: print("before memory output requires_grad?", self.memory.longTermMemory.requires_grad)
            if debugPrints: print("before cerebellum requires_grad?", self.interneuronNetwork.cerebellum.requires_grad)
            if debugPrints: print("before logRepetitionWindow requires_grad?", self.logRepetitionWindow.requires_grad)
            if debugPrints: print("before logMemoryLength requires_grad?", self.logMemoryLength.requires_grad)
            if skipFINALlogitNorm:
                if debugPrints: ʕっʘ‿ʘʔっ("Bx: logits.forward")
                FINALlogits = penalisedLogits
                #FINALlogits = self.logits.forward(memoryOutput)
            if debugPrints: print("AFTER logMemoryLength requires_grad?", self.logMemoryLength.requires_grad)
            if debugPrints: print("AFTER logRepetitionWindow requires_grad?", self.logRepetitionWindow.requires_grad)
            if debugPrints: print("AFTER cerebellum requires_grad?", self.interneuronNetwork.cerebellum.requires_grad)
            if debugPrints: print("AFTER memory output requires_grad?", self.memory.longTermMemory.requires_grad)

            if True:
                if debugPrints: ʕっʘ‿ʘʔっ("stats collection!")
                if _pixel is not None:
                    blend_vals = blend.detach().cpu().tolist()
                #self.inputEmbedsHistory.append(inputEmbeds.norm().item())
                #self.INNOutputHistory.append(INNOutput.norm().item())
                #self.memoryOutputHistory.append(memoryOutput.norm().item())
                #self.memory2OutputHistory.append(memory2Output.norm().item())
                #self.penalisedOutputHistory.append(penalisedLogits.norm().item())
                self.FINALlogitsHistory.append(FINALlogits.norm().item())

                if len(self.inputEmbedsHistory) >= self.numTokensPerStep:
                    self.forwardStats = {
                        #"2B_0_inputEmbeds_norm": sum(self.inputEmbedsHistory) / len(self.inputEmbedsHistory),
                        #"3B_1_INNOutput_norm": sum(self.INNOutputHistory) / len(self.INNOutputHistory),
                        #"5B_0_memoryOutput_norm": sum(self.memoryOutputHistory) / len(self.memoryOutputHistory),
                        #"5B_0b_memory2Output_norm": sum(self.memory2OutputHistory) / len(self.memory2OutputHistory),
                        #"7B_1_penalisedOutput_norm": sum(self.penalisedOutputHistory) / len(self.penalisedOutputHistory),
                        "7B_x_FINALlogits_norm": sum(self.FINALlogitsHistory) / len(self.FINALlogitsHistory),
                        #"B_blendPixel": self.blendPixel.item(),
                    }
                    self.forwardStats["B_blendToken"] = blend_vals[0]
                    self.forwardStats["B_blendPos"] = blend_vals[1]
                    self.forwardStats["B_blendPixel"] = blend_vals[2]
                    self.stats.update(self.forwardStats)
                    
                    self.inputEmbedsHistory = []
                    self.INNOutputHistory = []
                    self.memoryOutputHistory = []
                    self.memory2OutputHistory = []
                    self.penalisedOutputHistory = []
                    self.FINALlogitsHistory = []
                    self.normalisedHistory = []

            """returns a logits tensor of shape (1, vocabSize) showing predicted probabilities for the next token"""
            #tokenEmbed = self.embed(_tokenIndex = _inputSeq)
            #self.latestTokenEmbed = tokenEmbed
            #self.log_all_learnable_params(prefix="FORWARD_")
            if debugPrints:
                tensor_snitch(self, "babyllm forward end")
                tensor_snitch(self.memory, "babyllm forward end")
                tensor_snitch(self.memory2, "babyllm forward end")
                tensor_snitch(self.embed, "babyllm forward end")
                tensor_snitch(self.interneuronNetwork, "babyllm forward end")
                tensor_snitch(self.logits, "babyllm forward end")
            return FINALlogits #, self.latestTokenEmbed

    """computes the cross-entropy loss between the models logits and the target token, essentially checking how good the models prediction was"""        
    @whocalled
    def computeLoss(self, _logits, _targetTokenIndex, _totalAvgAbsDelta = 1, _learningRateGOAL = learningRateGOAL, _perfectTokens = 0, _training = False):
        with self.counsellor.infodump("computeLoss") as ʕっʘ‿ʘʔっ:
            self.perfectTokens = _perfectTokens
            self.totalAvgAbsDelta = _totalAvgAbsDelta
            self.learningRateGOAL = _learningRateGOAL
            if skipComputeLoss:
                if debugPrints: ʕっʘ‿ʘʔっ("skipping loss!")
                return torch.tensor([0.1], requires_grad = True, device = self.device)  # Constant scalar tensor
            
            if debugPrints: ʕっʘ‿ʘʔっ("targetTensor")          
            targetTensor = torch.tensor([_targetTokenIndex], dtype = torch.long, device = self.device)
            
            if debugPrints: print(f"logits shape: {_logits.shape} | target: {_targetTokenIndex}")
            if _logits.dim() == 1: 
                _logits = _logits.unsqueeze(0) # ensure logits are at least 2d
            
            if debugPrints: ʕっʘ‿ʘʔっ("cross Entropy Loss")
            loss = F.cross_entropy(_logits, targetTensor)
            self.CEloss_used = loss

            if not torch.isfinite(loss):
                print("NaN/Inf loss detected — logits:", _logits)
                return torch.tensor(10.0, device = self.device, requires_grad = True)

            if debugPrints: print(f"crossentropy raw loss: {F.cross_entropy(_logits, targetTensor)}")
            
            self.CELossDelta = loss - ((self.lastLossBaby) if self.lastLossBaby is not None else 0)

            if debugPrints: print(f"{self.lastLossBaby:0.1f}", end = ", ") # take delta

            # regulate the learned LR, temperature, repetition penalty (etc) towards target values
            lrSoftClamp = 0.001 * (self.logLR - math.log(learningRateGOAL)).pow(2)
            #lrSoftClamp = (self.totalAvgAbsDelta ** 1.5) * (self.logLR - math.log(self.learningRateGOAL)).pow(2)
            tempSoftClamp = (self.CEloss_used * 4) * (self.logTemp - math.log(temperatureGOAL)).pow(2)
            if self.repetitionPenalty >= 0: 
                repetitionPenaltySoftClamp = 0.000000000001 * (self.repetitionPenalty - repetitionPenaltyGOAL).pow(2)
            elif self.repetitionPenalty >= -1:
                repetitionPenaltySoftClamp = 0.0000001 * (self.repetitionPenalty - repetitionPenaltyGOAL).pow(2)
            elif self.repetitionPenalty < -1:
                repetitionPenaltySoftClamp = 0.0002 * (self.repetitionPenalty - repetitionPenaltyGOAL).pow(2)
            elif self.repetitionPenalty < 0:
                repetitionPenaltySoftClamp = 0.00002 * (self.repetitionPenalty - repetitionPenaltyGOAL).pow(2)

            loss += lrSoftClamp # use .detach() to avoid .backward()
            self.lrSoftClamp_used = lrSoftClamp
            loss += tempSoftClamp
            self.tempSoftClamp_used = tempSoftClamp
            loss += repetitionPenaltySoftClamp
            self.repPenSoftClamp_used = repetitionPenaltySoftClamp
            self.lastLossBaby = loss.item()
            FINALloss = loss
            if debugPrints: print(f"{FINALloss} + loss")

            if self.lastSoftSample is not None and not skipAuxLoss:
                target = F.one_hot(targetTensor, num_classes = _logits.shape[1]).float()
                kl_loss = F.kl_div(self.lastSoftSample.log(), target, reduction = 'batchmean')
                AUXloss_kl = kl_loss * 0.1
                self.AUXlossKL_used = AUXloss_kl
                #AUXloss = auxLoss * torch.sigmoid(loss - auxLoss) # low weight for anti-dominatrix
                cosSim = F.cosine_similarity(self.lastSoftSample, target)
                AUXloss_cos = (1.0 - cosSim.mean())
                self.AUXlossCos_used = AUXloss_cos
                AUXloss = AUXloss_cos + AUXloss_kl
                if debugPrints: print(f"{AUXloss} + aux")
            else:
                AUXloss = 0

            if not skipPixels and (self.nextPixelTarget is not None and hasattr(self, "pixelPupil")):
                if debugPrints: ʕっʘ‿ʘʔっ("RGB regression loss")
                if debugPrints: print(f"latestTokenEmbed is {self.latestTokenEmbed} ({self.latestTokenEmbed.shape}), [-1] is {self.latestTokenEmbed[-1]} ({self.latestTokenEmbed[-1].shape})")
                predictedRGB = self.pixelPupil(self.latestTokenEmbed[-1])
                self.predPixel = predictedRGB
                rgbLoss = F.mse_loss(self.predPixel, self.nextPixelTarget)
                #self.PIXELloss = rgbLoss * torch.sigmoid(loss - rgbLoss)
                pixelWeight = rgbLoss / (rgbLoss + loss)
                self.PIXELloss = max(min((pixelWeight * 1), 1),-1)
                if debugPrints: self.print_rgb_block(self.pixel, "prompt")
                if debugPrints: self.print_rgb_block(predictedRGB, "guess")
                if debugPrints: self.print_rgb_block(self.nextPixelTarget, "truth")
                if debugPrints: print(f"{rgbLoss} + rgb")
                if debugPrints: print(f"{self.PIXELloss} + pixel")
                # Detach the token embedding once it's no longer needed for gradient computation
                if self.latestTokenEmbed is not None:
                    self.latestTokenEmbed = self.latestTokenEmbed.detach()

            else:
                FINALloss = loss
                if debugPrints: print(f"{FINALloss} + final")

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
                FINALloss += (self.PIXELloss * 1.5)
                self.pixelLoss_used = (self.PIXELloss * 1.5)
                if debugPrints: print(f"{FINALloss} pixel + final")
            if self.lastSoftSample is not None and not skipAuxLoss: 
                if torch.isnan(AUXloss) or not torch.isfinite(AUXloss):
                    print(f"AUXloss contains NaN!")
                    AUXloss = torch.tensor(0.0, device = self.device)
                FINALloss += AUXloss
                if debugPrints: print(f"{FINALloss} aux ({AUXloss}) + final")
            if debugPrints: print(f"[LOSS DEBUG] requires_grad: {loss.requires_grad} | value: {loss.detach().cpu().item():.4f}")
            token_freqs = self.lastSoftSample.mean(dim = 0)
            repLoss = (token_freqs**2).mean() * self.repetitionPenalty
            self.repLoss_used = repLoss
            FINALloss += repLoss
            return FINALloss
    
    """backpropagation and optimization, computes gradients of the loss and uses the optimizer to update the models weights"""
    @whocalled
    def backward(self, _loss):
        with self.counsellor.infodump("backward") as ʕっʘ‿ʘʔっ:
            if debugPrints:
                tensor_snitch(self, "babyllm backward start")
                tensor_snitch(self.memory, "babyllm backward start")
                tensor_snitch(self.memory2, "babyllm backward start")
                tensor_snitch(self.embed, "babyllm backward start")
                tensor_snitch(self.interneuronNetwork, "babyllm backward start")
                tensor_snitch(self.logits, "babyllm backward start")
                if debugPrints: ʕっʘ‿ʘʔっ("print named parameters")
                printTensorAttrs(self, name='babyllm')
                printTensorAttrs(self.memory, name='memory')
                printTensorAttrs(self.memory2, name='memory2')
                printTensorAttrs(self.embed, name='embed')
                printTensorAttrs(self.interneuronNetwork, name='interneuronNetwork')
                printTensorAttrs(self.logits, name='logits')
                for name, p in self.named_parameters():
                    if p.grad is None:
                        if debugPrints: ʕっʘ‿ʘʔっ("print no grads")
                        print(f"before = {self.calligraphist.S_apply('dim', f'no grad: {name}')}")
                    else:
                        if debugPrints: ʕっʘ‿ʘʔっ("set yes grads")
                        grad = p.grad
                        if debugPrints: ʕっʘ‿ʘʔっ("set yes grad shape")
                        shape = tuple(grad.shape)
                        if debugPrints: ʕっʘ‿ʘʔっ("set yes grad norm")
                        norm = grad.norm().item()
                        if debugPrints: ʕっʘ‿ʘʔっ("set yes grad nonzero")
                        nonzero = grad.count_nonzero().item()
                        if debugPrints: ʕっʘ‿ʘʔっ("set yes grad numEl \ numan")
                        total = grad.numel()
                        if debugPrints: ʕっʘ‿ʘʔっ("set yes grad sparsity")
                        sparsity = 1 - (nonzero / total)
                        if debugPrints: ʕっʘ‿ʘʔっ("set yes grad mean")
                        mean = grad.mean().item()
                        if debugPrints: ʕっʘ‿ʘʔっ("set yes grad std")
                        std = grad.std().item()
                        if debugPrints: ʕっʘ‿ʘʔっ("print yes grads")
                        print(f"before = {self.calligraphist.S_apply('almostPerfect', f'yes grad: {name} | shape: {shape} | norm: {norm:.4f} | sparsity: {sparsity:.2%} | mean: {mean:.4f} | std: {std:.4f}')}")
                        if debugPrints: print("Loss:", _loss.item())
            if debugPrints: ʕっʘ‿ʘʔっ("loss.backward")
            if debugPrints: print(f"windowMAX: {self.numTokensPerStep}")
            _loss.backward()
            if debugPrints: print("Logit weights grad norm:", self.logits.l_weights.grad.norm())
            if debugPrints: print("LogWindowSizes grad norm:", self.interneuronNetwork.logWindowSizes.grad.norm())
            if debugPrints: print("Cerebellum grad norm:", self.interneuronNetwork.cerebellum.grad.norm())
            if debugPrints: print("Repetition penalty grad norm:", self.repetitionPenalty.grad.norm())
            #print(next(self.parameters()).grad)
            if debugPrints:
                if debugPrints: ʕっʘ‿ʘʔっ("print named parameters")
                printTensorAttrs(self, name='babyllm')
                printTensorAttrs(self.memory, name='memory')
                printTensorAttrs(self.memory2, name='memory2')
                printTensorAttrs(self.embed, name='embed')
                printTensorAttrs(self.interneuronNetwork, name='interneuronNetwork')
                printTensorAttrs(self.logits, name='logits')
                for name, p in self.named_parameters():
                    if p.grad is None:
                        if debugPrints: ʕっʘ‿ʘʔっ("print no grads")
                        print(f"after = {self.calligraphist.S_apply('emergency', f'NO GRAD: {name}')}")
                    else: 
                        if debugPrints: ʕっʘ‿ʘʔっ("set yes grads")
                        grad = p.grad
                        if debugPrints: ʕっʘ‿ʘʔっ("set yes grad shape")
                        shape = tuple(grad.shape)
                        if debugPrints: ʕっʘ‿ʘʔっ("set yes grad norm")
                        norm = grad.norm().item()
                        if debugPrints: ʕっʘ‿ʘʔっ("set yes grad nonzero")
                        nonzero = grad.count_nonzero().item()
                        if debugPrints: ʕっʘ‿ʘʔっ("set yes grad numEl \ numan")
                        total = grad.numel()
                        if debugPrints: ʕっʘ‿ʘʔっ("set yes grad sparsity")
                        sparsity = 1 - (nonzero / total)
                        if debugPrints: ʕっʘ‿ʘʔっ("set yes grad mean")
                        mean = grad.mean().item()
                        if debugPrints: ʕっʘ‿ʘʔっ("set yes grad std")
                        std = grad.std().item()
                        if debugPrints: ʕっʘ‿ʘʔっ("print yes grads")
                        print(f"after = {self.calligraphist.S_apply('almostPerfect', f'yes grad: {name} | shape: {shape} | norm: {norm:.4f} | sparsity: {sparsity:.2%} | mean: {mean:.4f} | std: {std:.4f}')}")

            if debugPrints: ʕっʘ‿ʘʔっ("torch.no_grad")
            with torch.no_grad(): # RESET LEARNABLE PARAMETERS
                #self.logLR.data.fill_(math.log(0.00035))  # Learning rate back to 1e-4
                if debugPrints: ʕっʘ‿ʘʔっ("fill scheduledSamplingRate")
                self.scheduledSamplingRate.data.fill_(0.001)  # Scheduled sampling full (no scheduled sampling yet)
                #self.temperature.data.fill_(math.exp(self.logTemp))  # Temperature normal
                #self.repetitionPenalty.data.fill_(1.0)  # Repetition penalty normal
                #self.logMemoryLength.data.fill_(math.log(5))  # Memory length default
                #self.logRepetitionWindow.data.fill_(math.log(16))  # Repetition window default
                #self.interneuronNetwork.logWindowSizes.data.copy_(
                #    torch.log(torch.tensor(allWindowSizes_new, dtype = torch.float32, device = self.device))
                #)
                #for module in self.interneuronNetwork.windowMeta:
                #    if isinstance(module, torch.nn.Linear):
               #        module.reset_parameters()

            if True:
                if debugPrints: ʕっʘ‿ʘʔっ("torch.no_grad")
                with torch.no_grad():
                    if debugPrints: ʕっʘ‿ʘʔっ("clamp logLR")
                    self.logLR.clamp_(math.log(0.0001), math.log(0.001))  # CLAMP IT! IN MEMORY OF THE AMAZING 1.00 SELF LEARNED LOSS RUN OF 27-APRIL-2025! - you certainly dropped the delta! you win!
                if debugPrints: ʕっʘ‿ʘʔっ("set self.memoryLength")
                self.memoryLength = torch.sigmoid((self.totalTurns - torch.exp(self.logMemoryLength)) * 0.5)
                if debugPrints: ʕっʘ‿ʘʔっ("set self.memoryLength2")
                self.memory2Length = torch.sigmoid((self.totalTurns - torch.exp(self.logMemory2Length)) * 0.5)
                if debugPrints: ʕっʘ‿ʘʔっ("set learnedLR")
                learnedLR = torch.exp(self.logLR).item()
                for g in self.optimizer.param_groups:
                    if debugPrints: ʕっʘ‿ʘʔっ("update self.optimizer.param_groups")
                    g['lr'] = learnedLR # send the learned LR to the optimizer
                #self.gradientClipMaxNorm = torch.exp(self.logGradClip).item()
                #self.repetitionWindow = torch.exp(self.logRepetitionWindow).item()
                #self.logLR.data.fill_(self.logLR+0.000001) # increment LR manually (break grid)

            if debugPrints: ʕっʘ‿ʘʔっ("clip_grad_norm")
            clipValue = torch.exp(self.logGradClip).item()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = clipValue)
            if debugPrints: ʕっʘ‿ʘʔっ("optimizer.step")
            self.optimizer.step()  # Update weights
            if debugPrints: ʕっʘ‿ʘʔっ("torch.exp(self.logRepetionWindow)")
            repWindow = torch.exp(self.logRepetitionWindow)
            if debugPrints: ʕっʘ‿ʘʔっ("set self.repetitionWindow")
            self.repetitionWindow = repWindow / (1 + repWindow / self.numTokensPerStep)  # asymptotes near windowMAX

            if debugPrints: ʕっʘ‿ʘʔっ("set backwardStats")
            if True:
                self.backwardStats = {
                    "B_floatMemoryLength": torch.exp(self.logMemoryLength).item(),
                    "B_floatMemory2Length": torch.exp(self.logMemory2Length).item(),
                    #"B_expWindow": repWindow.item(),
                    "B_repetitionWindow": self.repetitionWindow.item(),
                    "B_temperature": torch.exp(self.logTemp).item(),
                    "L_CEloss": self.CEloss_used,
                    #"L_PIXELloss": self.PIXELloss,
                    "L_PIXELloss_scaled": self.pixelLoss_used,
                    "L_AUXlossCos": self.AUXlossCos_used,
                    "L_AUXlossKL": self.AUXlossKL_used,
                    #"L_LRclamp": self.lrSoftClamp_used,
                    #"L_tempClamp": self.tempSoftClamp_used,
                    #"L_repPenClamp": self.repPenSoftClamp_used,
                    "L_repLoss": self.repLoss_used,
                }
                if debugPrints: ʕっʘ‿ʘʔっ("update self.stats with self.backwardStats")
                self.stats.update(self.backwardStats)
            #self.log_all_learnable_params(prefix="BACKWARD_")
            self.pixelLoss_used = 0

            #with torch.no_grad(): # FORCE RESET THE MEMORY GATES IF OVER USING LONG
                #self.memory.currentGate.data = self.memory.currentGate.data.abs()
                #self.memory.shortGate.data = self.memory.shortGate.data.abs()

            if debugPrints:
                tensor_snitch(self, "babyllm backward end")
                tensor_snitch(self.memory, "babyllm backward end")
                tensor_snitch(self.memory2, "babyllm backward end")
                tensor_snitch(self.embed, "babyllm backward end")
                tensor_snitch(self.interneuronNetwork, "babyllm backward end")
                tensor_snitch(self.logits, "babyllm backward end")

    @whocalled
    def getResponseFromLogits(self, _logits, _training = False):
        with self.counsellor.infodump("getResponseFromLogits") as ʕっʘ‿ʘʔっ:
            if not torch.isfinite(_logits).all():
                print("logits not finite before response gen:", _logits)
                _logits = torch.nan_to_num(_logits, nan = 0.0, posinf = 1e3, neginf=-1e3)
            if debugPrints: ʕっʘ‿ʘʔっ("update logarithmic parameters")
            #self.repetitionWindow = torch.exp(self.logRepetitionWindow)#.clamp(min = 1.0)
            if debugPrints: ʕっʘ‿ʘʔっ("torch.exp(self.logTemp)")
            self.temperature = torch.exp(self.logTemp)  # TORCH.exp keeps gradient path!
            if debugPrints: ʕっʘ‿ʘʔっ("_logits /= self.temperature")
            _logits /= self.temperature
            if debugPrints: ʕっʘ‿ʘʔっ("check for NaN logits")
            if torch.isnan(_logits).any():
                if debugPrints: ʕっʘ‿ʘʔっ("NaN yes = nan_to_num on _logits")
                print("NaN in logits after temperature scaling!")
                print("logTemp:", self.logTemp.item(), "Temp:", self.temperature.item())
                print("logits stats:", _logits.min().item(), _logits.max().item(), _logits.mean().item())
                _logits = torch.nan_to_num(_logits, nan = 0.0, posinf = 1e3, neginf=-1e3)

            if debugPrints: ʕっʘ‿ʘʔっ("if logits dim(1), unsqueeze(0)")
            if _logits.dim() == 1: _logits = _logits.unsqueeze(0)  # ensure [1, vocabSize]

            if _training:
                if debugPrints: ʕっʘ‿ʘʔっ("training, use gumbel")
                if debugPrints: ʕっʘ‿ʘʔっ("cloning _logits to logitForSample")
                logitsForSample = _logits.clone()
                if not torch.isfinite(logitsForSample).all():
                    if debugPrints: ʕっʘ‿ʘʔっ("non-finite logits detected BEFORE GUMBEL, nan_to_num logitsForSample")
                    print("non-finite logits detected BEFORE GUMBEL")
                    print("logits:", logitsForSample)
                    logitsForSample = torch.nan_to_num(logitsForSample, nan = 0.0, posinf = 1e3, neginf=-1e3)
                try:
                    if debugPrints: ʕっʘ‿ʘʔっ("gumbel softmax")
                    gumbelProbs = F.gumbel_softmax(logitsForSample, tau = self.temperature, hard = False)
                    assert torch.isfinite(gumbelProbs).all(), "gumbelProbs has NaN or Inf!"
                except Exception as e:
                    self.gumBellend += 1
                    if debugPrints: ʕっʘ‿ʘʔっ("gumbel softmax failed")
                    if debugPrints: print("gumbel softmax failed:", e)
                    if debugPrints: print(f"falling back to softmax sampling (total fallbacks: {self.gumBellend})...")
                    if debugPrints: ʕっʘ‿ʘʔっ("torch.softmax")
                    gumbelProbs = torch.softmax(logitsForSample, dim = 1)

                self.lastSoftSample = gumbelProbs
                if debugPrints: ʕっʘ‿ʘʔっ("gumbelProbs.argmax")
                responseFromLogits = gumbelProbs.argmax(dim = 1, keepdim = True)
                self.lastSoftSample = gumbelProbs

                if debugPrints: ʕっʘ‿ʘʔっ("topK sampling")
                topk = torch.topk(gumbelProbs, 10, dim = 1)
                finite = torch.isfinite(topk.values[0])
                self.rollingTokenTotals_tensor.index_add_(0, topk.indices[0][finite], topk.values[0][finite])

                #print("Top guesses + confidences:", [(self.librarian.indexToToken[i.item()], f"{p.item():.3f}") for i, p in zip(indices, values)])

            else:
                if debugPrints: ʕっʘ‿ʘʔっ("not training, using softmax")
                probs = torch.softmax(_logits, dim = 1)
                if debugPrints: ʕっʘ‿ʘʔっ("multinomial")
                responseFromLogits = torch.multinomial(probs, 1)
                self.lastSoftSample = None  # or keep the probs if you want analysis

            #if debugPrints:
                #print(f"[REP PENALTY] {self.repeatedPercent:.2%} repeated | repetition slice: {self.repetitionSlice} | Penalised: {[self.librarian.indexToToken.get(t, '<UNK>') for t in uniqueTokens]}")

            if debugPrints: ʕっʘ‿ʘʔっ("create windows using rolling buffer")
            repWindow = torch.exp(self.logRepetitionWindow)
            self.repetitionWindow = repWindow / (1 + repWindow / self.numTokensPerStep)  # asymptotes near windowMAX
            self.recentGeneratedTokens.append(responseFromLogits.item())
            if len(self.recentGeneratedTokens) > int(self.repetitionWindow):
                self.recentGeneratedTokens.pop(0)

            return responseFromLogits
        
    @whocalled    
    def applyRepetitionPenalty(self, _logits):
        with self.counsellor.infodump("applyRepetitionPenalty") as ʕっʘ‿ʘʔっ:
            if not self.recentGeneratedTokens:
                if debugPrints: ʕっʘ‿ʘʔっ("no recent generated tokens, returning _logits")
                return _logits

            if debugPrints: ʕっʘ‿ʘʔっ("repWindow = torch.exp(self.logRepetitionWindow)")
            repWindow = torch.exp(self.logRepetitionWindow)
            repWindow = repWindow / (1 + repWindow / self.numTokensPerStep)
            if debugPrints: ʕっʘ‿ʘʔっ("penalty = self.repetitionPenalty")
            penalty = self.repetitionPenalty

            if debugPrints: ʕっʘ‿ʘʔっ("recentTokens to tensor")
            recentTokens = torch.tensor(self.recentGeneratedTokens, device = self.device)
            if debugPrints: ʕっʘ‿ʘʔっ("vocabSize = _logits.shape[1]")
            vocabSize = _logits.shape[1]

            if debugPrints: ʕっʘ‿ʘʔっ("positions = torch.arange(len(recentTokens)).float()")
            positions = torch.arange(len(recentTokens), device = self.device).float()
            if debugPrints: ʕっʘ‿ʘʔっ("windowCenter")
            windowCenter = len(recentTokens) - 0.5  # so token 0 gets proper suppression
            if debugPrints: ʕっʘ‿ʘʔっ("softMask = torch.sigmoid((positions - (windowCenter - repWindow)) * 0.5)")
            softMask = torch.sigmoid((positions - (windowCenter - repWindow)) * 0.5)

            if debugPrints: ʕっʘ‿ʘʔっ("oneHots")
            oneHots = F.one_hot(recentTokens, num_classes = vocabSize).float()
            if debugPrints: ʕっʘ‿ʘʔっ("weightedFreqs = (oneHots.T @ softMask).view(1, -1)")
            weightedFreqs = (oneHots.T @ softMask).view(1, -1)

            if debugPrints: ʕっʘ‿ʘʔっ("setting penalty to 0 for target token!")
            if self.targetTokenFromTutor is not None:
                weightedFreqs[0, self.targetTokenFromTutor] = 0.0

        return _logits - (weightedFreqs * penalty)

    @whocalled
    def getNextToken(self, _inputSeq):  
        with self.counsellor.infodump("getNextToken(FORWARD)") as ʕっʘ‿ʘʔっ:
            ʕっʘ‿ʘʔっ(f"unpack logits from self.forward{_inputSeq}")
            logits, *_ = self.forward(_inputSeq) # unpacks the first value of the tuple and ignores the rest
            if debugPrints: ʕっʘ‿ʘʔっ("get next token")
            nextToken = self.getResponseFromLogits(logits, _training = True)
            if debugPrints: 
                print("nextToken: ")
            print(f"{nextToken}")
            return nextToken

    @whocalled    
    def saveModel(self, _trainingStepCounter, _totalAvgLoss, _first, filePath = modelFilePath, _newStartIndex = trainingStartIndex):
        with self.counsellor.infodump("saveModel") as ʕっʘ‿ʘʔっ:
            with open(stepCheckpointFilePath, "w") as f:
                if debugPrints or True: print(f"HELLO I AM SAVEMODEL STEPCOUNTER IS {_trainingStepCounter} AND START INDEX IS {_newStartIndex} I SHOULD WRITE {str(_trainingStepCounter+_newStartIndex)} to {stepCheckpointFilePath}")
                f.write(str(_trainingStepCounter+_newStartIndex)) # THIS ISNT REAL, FIX LATER, MAYBE MOVE SAVE AND LOAD TO WAKEUP?
            with open(lossCheckpointFilePath, "w") as f:
                if debugPrints or True: print(f"HELLO I AM SAVEMODEL AVGLOSS IS {_totalAvgLoss} I SHOULD WRITE {str(_totalAvgLoss)} to {lossCheckpointFilePath}")
                f.write(str(_totalAvgLoss))
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
            # (existing model and optimizer saving)
            memory_buffers_state = {
                'memory1_short': self.memory.shortTermMemory.detach().cpu(),
                'memory1_long': self.memory.longTermMemory.detach().cpu(),
                'memory2_short': self.memory2.shortTermMemory.detach().cpu(),
                'memory2_long': self.memory2.longTermMemory.detach().cpu(),
            }
            buffers_path = filePath + ".membuff"
            tmp_buffers_path = buffers_path + ".tmp"
            torch.save(memory_buffers_state, tmp_buffers_path)
            print(f"Memory buffers temp file created at {tmp_buffers_path}...")
            os.replace(tmp_buffers_path, buffers_path)
            print(f"Memory buffers successfully saved to {buffers_path}!")

    @whocalled
    def loadModel(self, filePath = modelFilePath):
        with self.counsellor.infodump("loadModel") as ʕっʘ‿ʘʔっ:
            try:
                if debugPrints: ʕっʘ‿ʘʔっ("update logarithmic parameters")
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
                #self.resetMemory(context="inference", _memoryLength = self.memoryLength)
                # (existing model and optimizer loading)
                buffers_path = filePath + ".membuff"
                if os.path.exists(buffers_path):
                    try:
                        memory_buffers_state = torch.load(buffers_path, map_location = self.device) # Load to current device
                        self.memory.shortTermMemory.data.copy_(memory_buffers_state['memory1_short'])
                        self.memory.longTermMemory.data.copy_(memory_buffers_state['memory1_long'])
                        self.memory2.shortTermMemory.data.copy_(memory_buffers_state['memory2_short'])
                        self.memory2.longTermMemory.data.copy_(memory_buffers_state['memory2_long'])
                        print(f"Memory buffers restored from {buffers_path}")
                    except Exception as e:
                        print(f"Failed to load memory buffers: {e}. Initializing to zeros.")
                        # Ensure they are zeroed if loading fails
                        self.memory.shortTermMemory.zero_()
                        self.memory.longTermMemory.zero_()
                        self.memory2.shortTermMemory.zero_()
                        self.memory2.longTermMemory.zero_()
                else:
                    print(f"No memory buffer file found at {buffers_path}. Initializing to zeros.")
                    # Ensure they are zeroed if file not found
                    self.memory.shortTermMemory.zero_()
                    self.memory.longTermMemory.zero_()
                    self.memory2.shortTermMemory.zero_()
                    self.memory2.longTermMemory.zero_()
                self.memory.to(self.device)
                self.memory2.to(self.device)
                print(f"memory device set to {self.device}!")
                
            except FileNotFoundError: print("no saved model found")

    @whocalled
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

    @whocalled
    def resetMemory(self, context="inference"):
        with self.counsellor.infodump("resetMemory") as ʕっʘ‿ʘʔっ:
            """Reset memory depending on the context: inference always resets, training resets every n turns"""
            self.memoryLength = torch.sigmoid((self.totalTurns - torch.exp(self.logMemoryLength)) * 0.1)
            self.memory2Length = torch.sigmoid((self.totalTurns - torch.exp(self.logMemory2Length)) * 0.1)
            #print(f"resetting memory... (learned mem length: {self.memoryLength})")
            #self.memory.resetMemory(_memoryLength = self.memoryLength)
            #self.memory2.resetMemory(_memoryLength = self.memoryLength)
            if context == "inference":
                if debugPrints: ʕっʘ‿ʘʔっ("context = inference")
                self.memory.resetMemory(self.memoryLength)
                self.memory2.resetMemory(self.memory2Length)
                print(f"resetting memory for new conversation...")
            elif context == "training":
                if debugPrints: ʕっʘ‿ʘʔっ("context = training")
                if hasattr(self, "stepsSinceMemoryReset"): 
                    self.stepsSinceMemoryReset += 1
                else: 
                    self.stepsSinceMemoryReset = 1
                if hasattr(self, "stepsSinceMemory2Reset"): 
                    self.stepsSinceMemory2Reset += 1
                else: 
                    self.stepsSinceMemory2Reset = 1
                if self.stepsSinceMemoryReset > 3: 
                    if debugPrints: print(f"resetting memory1 after {self.stepsSinceMemoryReset} steps... (learned mem length: {torch.exp(self.logMemoryLength)} ({self.memoryLength}))")
                    self.memory.resetMemory(_memoryLength = self.memoryLength)
                    self.stepsSinceMemoryReset = 0
                if self.stepsSinceMemory2Reset > 3:
                    if debugPrints: print(f"resetting memory2 after {self.stepsSinceMemory2Reset} steps... (learned mem length: {torch.exp(self.logMemory2Length)} ({self.memory2Length}))")
                    self.memory2.resetMemory(_memoryLength = self.memory2Length)
                    self.stepsSinceMemory2Reset = 0 

    @whocalled
    def setLearningRate(self, _newLearningRate):
        self.learningRate = max(1e-6, min(_newLearningRate, 0.01))  # clamp it a bit
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learningRate

    @whocalled
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

    def getRollingTokenTotalsDict(self):
        counts = self.rollingTokenTotals_tensor.detach().cpu()
        non_zero = torch.nonzero(counts).squeeze()
        if non_zero.numel() == 0:
            return {}
        if non_zero.dim() == 0:
            non_zero = non_zero.unsqueeze(0)
        return {self.librarian.indexToToken[int(i)]: float(counts[int(i)]) for i in non_zero}


    """def log_all_learnable_params(self, prefix="PARAM_"):
        Logs all learnable scalar parameters and basic stats for tensors in self.stats dict.
        Also ensures mostImportantStats includes new param keys matching include_patterns.
        for name, param in self.named_parameters():
            if param.requires_grad:
                if param.numel() == 1:
                    self.stats[f"{prefix}{name}"] = param.item()
                else:
                    self.stats[f"{prefix}{name}_mean"] = param.data.mean().item()
                    self.stats[f"{prefix}{name}_norm"] = param.data.norm().item()

        new_keys = [k for k in self.stats if k.startswith(prefix)]
        for key in new_keys:
            if re.search(pat, key, re.IGNORECASE):
                if key not in mostImportantStats:
                    mostImportantStats.append(key)"""

    @whocalled
    def getBabyStats(self): return self.stats
    
if __name__ == "__main__":
    exit(0)