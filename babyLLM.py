# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# BABYLLM // babyLLM.py
# v4.1

import random, os, threading
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim 
from adan_pytorch import Adan
import math
from sophia.sophia import SophiaG

from brain.LAYERS.embed import EMBED
from brain.LAYERS.interneuronNetwork import INTERNEURON_NETWORK
from brain.LAYERS.logits import LOGITS
from brain.LAYERS.memory import MEMORY
from brain.LAYERS.attention import GATED_MHA
# Creative modules removed
#from brain.LAYERS.sensoryWobble import WOBBLE
from config import *
from secret import *
from helpers import clamp_param, get_grad_stats, debug_print

GRAD_SNAPSHOT_LIMIT = 8

def log_param_to_length(log_param): return torch.sigmoid((1 - torch.exp(log_param)) * 0.1)

"""this class combines all the core components of the babyLLM:"""
"""EMBED: token embedding layer"""
"""INTERNEURON_NETWORK: layer of parallel neurons for feature extraction"""
"""LOGITS: output layer to generate logits"""
"""it also manages training, loss computation, backpropagation, and response generation."""
class BABYLLM(nn.Module):
    def __init__(self, _counsellor, _calligraphist, _scribe, _librarian, _numTokensPerStep, _learningRateGOAL = learningRateGOAL, _device = modelDevice, _first = True, _model_thread_lock = None):
        super().__init__()
        self.device = _device
        self.counsellor = _counsellor
        self.calligraphist = _calligraphist
        self.scribe = _scribe
        self.librarian = _librarian
        self.model_thread_lock = _model_thread_lock or threading.Lock()
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
        self.PIXELloss = 0
        self.CEloss_used = 0.0
        self.lrSoftClamp_used = 0.0
        self.tempSoftClamp_used = 0.0
        self.repPenSoftClamp_used = 0.0
        self.repLoss_used = 0.0
        self.targetTokenFromTutor = None

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

        self.cerebralLoad = 0.0
        self.dreamIntensity = 0.0
        self.memoryFlux = 0.0
        self.learningStability = 0.0
        self.AUXlossCos_used = 0.0
        self.AUXlossKL_used = 0.0
        self.lastSoftSample = None
        self._lastSoftSample_for_loss = None

        """CEREBRAL LAYERS // brain"""
        self.embed = EMBED(_counsellor = self.counsellor, _device = self.device)
        self.attention = GATED_MHA(_counsellor = self.counsellor, _device = self.device)
        self.interneuronNetwork = INTERNEURON_NETWORK(_model = BABYLLM, _counsellor = self.counsellor, _calligraphist = self.calligraphist, _device = self.device, _numTokensPerStep = self.numTokensPerStep)
        self.logits = LOGITS(_counsellor = self.counsellor, _device = self.device, _numTokensPerStep = self.numTokensPerStep)
        self.memory = MEMORY(_counsellor = self.counsellor, _device = self.device, _numTokensPerStep = self.numTokensPerStep)
        self.memory2 = MEMORY(_counsellor = self.counsellor, _device = self.device, _numTokensPerStep = self.numTokensPerStep)
        #self.pixelPupil = nn.Sequential(nn.Linear(embedDimension, embedDimension), nn.GELU(), nn.Linear(embedDimension, 3), nn.Sigmoid())
        self.pixelPupil = PIXEL(embedDimension, embedDimension, 3, _device=self.device)
        
        # Creative modules removed

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
        self.memoryLength = log_param_to_length(self.logMemoryLength)
        self.memory2Length = log_param_to_length(self.logMemory2Length)

        """self.transformer_block = nn.TransformerEncoderLayer(
            d_model=embedDimension, 
            nhead=8,  # A reasonable number of attention heads
            dim_feedforward=embedDimension * 4, # Standard practice
            dropout=0.1,
            activation='gelu',
            batch_first=True # IMPORTANT
        ).to(self.device)"""

        """stuff"""
        #self.gradientClipMaxNorm = torch.exp(self.logGradClip)
        self.temperature = None

        """OPTIMIZER - this updates all of the layers learnable parameters"""
        debug_print("registered parameters: ")
        for name, param in BABYLLM.named_parameters(self):
            debug_print(name, param.shape)

        #baseOptim = optim.RAdam(self.parameters(), lr = learningRate)
        #baseOptim = torch_optimizer.Lion(self.parameters(), lr = 1e-4)
        #self.optimizer = optim.Lookahead(baseOptim)
        #self.optimizer = baseOptim

        if optimizerName == "Adan":
            self.optimizer = Adan(
                self.parameters(),
                lr=learningRate,
                betas=(0.98, 0.92, 0.99),
                eps=1e-6,
                weight_decay=0.05,
            )
        elif optimizerName == "Sophia":
            self.optimizer = SophiaG(self.parameters(), lr=learningRate,  # start slightly lower LR than AdamW, Sophia can be aggressive
            betas=(0.965, 0.99), rho=0.04, weight_decay=0.05)
        else:
            optimizerClass = getattr(optim, optimizerName)
            self.optimizer = optimizerClass(self.parameters(), lr = learningRate, weight_decay = 0.05, fused = True)
        #print("!!! RUNNING WITH SGD OPTIMIZER !!!")
        #self.optimizer = optim.SGD(self.parameters(), lr=learningRate, momentum=0.9)


        for name, param in self.named_parameters():
            debug_print(f"{name}: requires_grad={param.requires_grad}")

        #self.to(self.device)
        self.statsCategories = {
            "loss": 0,
            "gradNorm": 0,
            "logitMin": 0,
            "logitMax": 0,
            "scheduledSamplingRate": 0,
            "tokenCount": 0,
            "memoryGateShort": 0,
            "memoryGateLong": 0,
            "memoryGateCurrent": 0,
            "shortDecay": 0,
            "longDecay": 0,
        }

    def _snapshot_gradients(self, limit: int = GRAD_SNAPSHOT_LIMIT):
        snapshot = []
        for name, param in self.named_parameters():
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            try:
                stats = get_grad_stats(grad)
            except Exception:
                continue
            snapshot.append((name, stats))
            if len(snapshot) >= limit:
                break
        return snapshot

    @whocalled
    def forward(self, _inputSeq = None, _pixel = None, _use_lock: bool = True):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ: # processes input sequence of tokens (str) to generate logits to predict the next token
            lock_ctx = self.model_thread_lock if _use_lock else nullcontext()
            with lock_ctx:
                if debugPrints:
                    tensor_snitch(self, "babyllm forward start")
                    tensor_snitch(self.memory, "babyllm forward start")
                    tensor_snitch(self.memory2, "babyllm forward start")
                    tensor_snitch(self.embed, "babyllm forward start")
                    tensor_snitch(self.interneuronNetwork, "babyllm forward start")
                    tensor_snitch(self.logits, "babyllm forward start")
                self.temperature = torch.exp(self.logTemp)
                self.interneuronNetwork.temperature = self.temperature
                self.pixel = _pixel

                if debugPrints: ʕっʘ‿ʘʔっ("B0: inputEmbeds") # convert indices to embeddings
                tokenEmbed = self.embed(_tokenIndex = _inputSeq)
                seq_len = tokenEmbed.shape[0]
                pos_indices = torch.arange(seq_len, device = tokenEmbed.device)
                posEmbed = self.embed.posEmbedding(pos_indices)
                posEmbed = self.embed.posDropout(posEmbed * self.embed.scale)  # [seq_len, embed_dim]
                if not skipPixels and (_pixel is not None):
                    rgbEmbed = self.embed(_pixel = _pixel)
                    debug_print("tokenEmbed:", tokenEmbed.shape)
                    debug_print("posEmbed:", posEmbed.shape)
                    debug_print("rgbEmbed:", rgbEmbed.shape)
                    #blendPixelClamped = self.blendPixel.clamp(0.0, 1.0)
                    #inputEmbeds = ((1.0 - blendPixelClamped) * tokenEmbed) + (blendPixelClamped * rgbEmbed)
                    blend = F.softmax(self.inputBlend, dim = 0)
                    inputEmbeds = blend[0] * tokenEmbed + blend[1] * posEmbed + blend[2] * rgbEmbed
                else: inputEmbeds = tokenEmbed
                token_embed_for_pixel = inputEmbeds
                # Store a detached copy so we don't hold on to the autograd graph between
                # forward passes (this was causing a memory leak when pixels were skipped).
                self.latestTokenEmbed = token_embed_for_pixel.detach()
                # Ensure latestTokenEmbed has proper dimensions for pixel regression
                if hasattr(self, "pixelPupil") and len(self.latestTokenEmbed.shape) == 1:
                    # If 1D, ensure it matches expected embedding dimension
                    debug_print(f"[DEBUG] latestTokenEmbed is 1D with shape {self.latestTokenEmbed.shape}")
                inputEmbeds = self.attention(inputEmbeds)
                debug_print(f"Debug BABYLLM.forward: inputEmbeds requires_grad: {inputEmbeds.requires_grad} [EXPECTED: TRUE]")

                if debugPrints: ʕっʘ‿ʘʔっ("B1: interneuronNetworkOutput") # PARALLEL NEURON LAYER input/processing (feature extraction)

                if True:
                    INNOutput = self.interneuronNetwork.forward(inputEmbeds)
                    debug_print(f"Debug BABYLLM.forward: interneuronNetworkOutput length: {len(INNOutput)}") 
                    debug_print("combinedActivationsTensor.requires_grad:", INNOutput.requires_grad)
                    debug_print("combinedActivationsTensor.grad_fn:", INNOutput.grad_fn)

                    if debugPrints: ʕっʘ‿ʘʔっ("B2: memoryOutput") # MEMORY LAYER PROCESSING - NOW PROCESS THE COMBINED ACTIVATIONS
                    if skipMemory:
                        debug_print("skipping memory layer...")
                        memoryOutput = INNOutput
                    else:
                        # --- RESIDUAL A: pass the raw thought past the first memory layer ---
                        memoryOutput = self.memory.forward(INNOutput) + INNOutput

                        memory2Input = (INNOutput * 0.5) + (memoryOutput * 0.5)

                        # --- RESIDUAL B: bypass the second Memory Layer ---
                        memory2Output = self.memory2.forward(memory2Input) + memory2Input
                        #self.latestMemGates = self.memory.latestMemoryGates
                    
                    if debugPrints: ʕっʘ‿ʘʔっ("B3: logits.forward BEFORE penalty")
                    logitsBeforePenalty = self.logits.forward(memory2Output)
                    debug_print("combinedActivations.requires_grad:", memoryOutput.requires_grad)

                if debugPrints: ʕっʘ‿ʘʔっ("B4: applyRepetitionPenalty to logits")
                if not torch.isfinite(self.logRepetitionWindow):
                    print("logRepetitionWindow has gone non-finite. Resetting.")
                    self.logRepetitionWindow.data = torch.tensor(math.log(repetitionWindowGOAL), device = self.device)
                if self.logRepetitionWindow > math.log(windowMAXSTART):
                    print("logRepetitionWindow is higher than windowMAXSTART. Resetting.")
                    self.logRepetitionWindow.data = torch.tensor(math.log(repetitionWindowGOAL), device = self.device)
                penalisedLogits = self.applyRepetitionPenalty(logitsBeforePenalty, _inputSeq)
                
                debug_print("before memory output requires_grad?", self.memory.longTermMemory.requires_grad)
                debug_print("before cerebellum requires_grad?", self.interneuronNetwork.cerebellum.requires_grad)
                debug_print("before logRepetitionWindow requires_grad?", self.logRepetitionWindow.requires_grad)
                debug_print("before logMemoryLength requires_grad?", self.logMemoryLength.requires_grad)
                if skipFINALlogitNorm:
                    if debugPrints: ʕっʘ‿ʘʔっ("Bx: logits.forward")
                    FINALlogits = penalisedLogits
                else:
                    FINALlogits = penalisedLogits 

                debug_print("AFTER logMemoryLength requires_grad?", self.logMemoryLength.requires_grad)
                debug_print("AFTER logRepetitionWindow requires_grad?", self.logRepetitionWindow.requires_grad)
                debug_print("AFTER cerebellum requires_grad?", self.interneuronNetwork.cerebellum.requires_grad)
                debug_print("AFTER memory output requires_grad?", self.memory.longTermMemory.requires_grad)

                if True:
                    if debugPrints: ʕっʘ‿ʘʔっ("stats collection!")
                    blend_vals = None
                    if (not skipPixels) and (_pixel is not None):
                        blend_vals = blend.detach().cpu().tolist()
                    #self.inputEmbedsHistory.append(inputEmbeds.norm().item())
                    #self.INNOutputHistory.append(INNOutput.norm().item())
                    #self.memoryOutputHistory.append(memoryOutput.norm().item())
                    #self.memory2OutputHistory.append(memory2Output.norm().item())
                    #self.penalisedOutputHistory.append(penalisedLogits.norm().item())
                    self.FINALlogitsHistory.append(FINALlogits.norm().item())
                    if len(self.FINALlogitsHistory) >= self.numTokensPerStep:
                        self.forwardStats = {
                            #"2B_0_inputEmbeds_norm": sum(self.inputEmbedsHistory) / len(self.inputEmbedsHistory),
                            #"3B_1_INNOutput_norm": sum(self.INNOutputHistory) / len(self.INNOutputHistory),
                            #"5B_0_memoryOutput_norm": sum(self.memoryOutputHistory) / len(self.memoryOutputHistory),
                            #"5B_0b_memory2Output_norm": sum(self.memory2OutputHistory) / len(self.memory2OutputHistory),
                            #"7B_1_penalisedOutput_norm": sum(self.penalisedOutputHistory) / len(self.penalisedOutputHistory),
                            "7B_x_FINALlogits_norm": sum(self.FINALlogitsHistory) / len(self.FINALlogitsHistory),
                            #"B_blendPixel": self.blendPixel.item(),
                        }
                        if blend_vals is not None:
                            self.forwardStats["B_blendToken"] = blend_vals[0]
                            self.forwardStats["B_blendPos"] = blend_vals[1]
                            self.forwardStats["B_blendPixel"] = blend_vals[2]
                            debug_print(f"token {blend_vals[0]}, pos {blend_vals[1]}, pixel {blend_vals[2]}")
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
            
            debug_print(f"logits shape: {_logits.shape} | target: {_targetTokenIndex}")
            if _logits.dim() == 1: 
                _logits = _logits.unsqueeze(0) # ensure logits are at least 2d
            
            if debugPrints: ʕっʘ‿ʘʔっ("cross Entropy Loss")
            loss = F.cross_entropy(_logits, targetTensor)
            loss_value = loss.detach().item()
            self.CEloss_used = loss_value

            if not torch.isfinite(loss):
                print("NaN/Inf loss detected — logits:", _logits)
                return torch.tensor(10.0, device = self.device, requires_grad = True)

            debug_print(f"crossentropy raw loss: {F.cross_entropy(_logits, targetTensor)}")
            
            self.CELossDelta = loss - ((self.lastLossBaby) if self.lastLossBaby is not None else 0)

            debug_print(f"{self.lastLossBaby:0.1f}", end = ", ") # take delta

            # regulate the learned LR, temperature, repetition penalty (etc) towards target values
            lrSoftClamp = 0.0015 * (self.logLR - math.log(learningRateGOAL)).pow(2)
            #lrSoftClamp = (self.totalAvgAbsDelta ** 1.5) * (self.logLR - math.log(self.learningRateGOAL)).pow(2)
            tempSoftClamp = (loss_value * 0.4) * (self.logTemp - math.log(temperatureGOAL)).pow(2)
            repetitionPenaltySoftClamp = 0.04 * (self.repetitionPenalty - repetitionPenaltyGOAL).pow(2)

            # Creative gate regulation removed

            loss += lrSoftClamp # use .detach() to avoid .backward()
            self.lrSoftClamp_used = lrSoftClamp.detach().item()
            loss += tempSoftClamp
            self.tempSoftClamp_used = tempSoftClamp.detach().item()
            loss += repetitionPenaltySoftClamp
            self.repPenSoftClamp_used = repetitionPenaltySoftClamp.detach().item()

            
            self.lastLossBaby = loss.item()
            FINALloss = loss
            debug_print(f"{FINALloss} + loss")

            soft_sample = getattr(self, "_lastSoftSample_for_loss", None)
            if soft_sample is None:
                soft_sample = self.lastSoftSample

            if soft_sample is not None and not skipAuxLoss:
                target = F.one_hot(targetTensor, num_classes = _logits.shape[1]).float()
                # Clamp to avoid log(0) producing -inf and destabilising KL
                eps = 1e-8
                safe_probs = soft_sample.clamp(min=eps)
                kl_loss = F.kl_div(safe_probs.log(), target, reduction = 'batchmean')
                AUXloss_kl = kl_loss * 0.01
                self.AUXlossKL_used = AUXloss_kl.detach().item()
                #AUXloss = auxLoss * torch.sigmoid(loss - auxLoss) # low weight for anti-dominatrix
                # Ensure cosine similarity is well-defined (avoid zero-norm vectors)
                safe_probs_norm = safe_probs / safe_probs.norm(dim=-1, keepdim=True).clamp_min(eps)
                target_norm = target / target.norm(dim=-1, keepdim=True).clamp_min(eps)
                cosSim = (safe_probs_norm * target_norm).sum(dim=-1)
                AUXloss_cos = (1.0 - cosSim.mean())
                self.AUXlossCos_used = AUXloss_cos.detach().item()
                AUXloss = AUXloss_cos + AUXloss_kl
                debug_print(f"{AUXloss} + aux")
            else:
                self.AUXlossKL_used = 0.0
                self.AUXlossCos_used = 0.0
                AUXloss = 0

            if soft_sample is not None:
                token_freqs = soft_sample.mean(dim = 0)
                repLoss_raw = (token_freqs**2).mean()
                repLoss = repLoss_raw * 100.0
                self.repLoss_used = repLoss.detach().item()
                FINALloss += repLoss
                debug_print(f"{FINALloss} repLoss ({repLoss}) + final")
            else:
                self.repLoss_used = 0.0

            if not skipPixels and (self.nextPixelTarget is not None and hasattr(self, "pixelPupil")):
                if debugPrints: ʕっʘ‿ʘʔっ("RGB regression loss with creative synesthetic enhancement")

                # Handle different tensor shapes properly
                token_embed_for_pixel = getattr(self, "latestTokenEmbed", None)
                if token_embed_for_pixel is None:
                    debug_print("latestTokenEmbed is None; using zero embedding for pixel loss")
                    token_embed_for_pixel = torch.zeros(self.pixelPupil.linear1.in_features, device=self.device)
                else:
                    debug_print(f"latestTokenEmbed is {token_embed_for_pixel} ({token_embed_for_pixel.shape})")

                if len(token_embed_for_pixel.shape) == 1:
                    # Already 1D embedding - use directly
                    embedding = token_embed_for_pixel
                elif len(token_embed_for_pixel.shape) == 2:
                    # 2D tensor [seq_len, embed_dim] - take the last token
                    embedding = token_embed_for_pixel[-1]
                else:
                    # Unexpected shape - flatten and take appropriate slice
                    embedding = token_embed_for_pixel.flatten()
                    if embedding.size(0) > self.pixelPupil.linear1.in_features:
                        embedding = embedding[:self.pixelPupil.linear1.in_features]
                
                # Debug tensor shapes before pixelPupil
                debug_print(f"[DEBUG] About to pass embedding to pixelPupil: shape={embedding.shape}, dtype={embedding.dtype}")
                debug_print(f"[DEBUG] pixelPupil.linear1 expects input size: {self.pixelPupil.linear1.in_features}")
                
                # Ensure embedding is the right shape - should be [embedDimension] for single token
                # Fix potential batch dimension issues
                if len(embedding.shape) == 0:
                    raise RuntimeError(f"Embedding is a scalar! Original latestTokenEmbed shape: {self.latestTokenEmbed.shape}")
                elif len(embedding.shape) == 2:
                    debug_print(f"[DEBUG] Embedding has 2D shape {embedding.shape}, taking mean across sequence dimension")
                    embedding = embedding.mean(dim=0)  # Average across sequence if needed
                elif len(embedding.shape) > 2:
                    debug_print(f"[DEBUG] Embedding has unexpected shape {embedding.shape}, flattening to expected size...")
                    embedding = embedding.flatten()
                    if embedding.size(0) != self.pixelPupil.linear1.in_features:
                        debug_print(f"[DEBUG] Flattened size {embedding.size(0)} doesn't match expected {self.pixelPupil.linear1.in_features}, truncating/padding")
                        if embedding.size(0) > self.pixelPupil.linear1.in_features:
                            embedding = embedding[:self.pixelPupil.linear1.in_features]
                        else:
                            padding_size = self.pixelPupil.linear1.in_features - embedding.size(0)
                            embedding = torch.cat([embedding, torch.zeros(padding_size, device=embedding.device)])
                
                # Ensure we have the right final shape
                if embedding.size(0) != self.pixelPupil.linear1.in_features:
                    raise RuntimeError(f"Embedding final size {embedding.size(0)} doesn't match pixelPupil input size {self.pixelPupil.linear1.in_features}")
                
                # Base pixel prediction
                base_predicted_rgb = self.pixelPupil(embedding)
                
                # Synesthetic enhancement removed; use base prediction
                predictedRGB = base_predicted_rgb
                if debugPrints: debug_print("Using base pixel prediction (no synesthesia)")
                
                self.predPixel = predictedRGB
                rgbLoss = F.mse_loss(self.predPixel, self.nextPixelTarget)
                # Weight pixel loss safely; avoid 0/0 when both rgbLoss and loss are ~0
                eps = 1e-8
                pixelWeight = rgbLoss / (rgbLoss + loss + eps)
                if not torch.isfinite(pixelWeight):
                    pixelWeight = torch.tensor(0.0, device=self.device)
                self.PIXELloss = max(min((pixelWeight * 1), 1), -1)
                if debugPrints: self.print_rgb_block(self.pixel, "prompt")
                if debugPrints: self.print_rgb_block(predictedRGB, "guess")
                if debugPrints: self.print_rgb_block(self.nextPixelTarget, "truth")
                debug_print(f"{rgbLoss} + rgb")
                debug_print(f"{self.PIXELloss} + pixel")
                # Detach the token embedding once it's no longer needed for gradient computation
                if self.latestTokenEmbed is not None:
                    self.latestTokenEmbed = self.latestTokenEmbed.detach()

            else:
                FINALloss = loss
                debug_print(f"{FINALloss} + final")

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
                FINALloss += (self.PIXELloss * 0.5)
                self.pixelLoss_used = (self.PIXELloss * 0.5)
                debug_print(f"{FINALloss} pixel + final")

            if soft_sample is not None and not skipAuxLoss:
                if torch.isnan(AUXloss) or not torch.isfinite(AUXloss):
                    print(f"AUXloss contains NaN!")
                    AUXloss = torch.tensor(0.0, device = self.device)
                FINALloss += AUXloss
                debug_print(f"{FINALloss} aux ({AUXloss}) + final")
            debug_print(f"[LOSS DEBUG] requires_grad: {loss.requires_grad} | value: {loss.detach().cpu().item():.4f}")

            # Drop references to the computation graph so successive calls do not accumulate memory.
            self._lastSoftSample_for_loss = None
            if soft_sample is not None:
                self.lastSoftSample = soft_sample.detach()

            if not torch.isfinite(FINALloss):
                print("computeLoss produced non-finite FINALloss; resetting to fallback.")
                FINALloss = torch.tensor(10.0, device=self.device, requires_grad=True)

            return FINALloss
    
    """backpropagation and optimization, computes gradients of the loss and uses the optimizer to update the models weights"""
    @whocalled
    def backward(self, _loss, _lossDelta):
        with self.counsellor.infodump("backward") as ʕっʘ‿ʘʔっ:
            collect_grad_stats = (self.totalTurns % 100 == 0)
            grad_snapshot = None

            if debugPrints:
                tensor_snitch(self, "babyllm backward start")
                tensor_snitch(self.memory, "babyllm backward start")
                tensor_snitch(self.memory2, "babyllm backward start")
                tensor_snitch(self.embed, "babyllm backward start")
                tensor_snitch(self.interneuronNetwork, "babyllm backward start")
                tensor_snitch(self.logits, "babyllm backward start")
                ʕっʘ‿ʘʔっ("print named parameters")
                printTensorAttrs(self, name='babyllm')
                printTensorAttrs(self.memory, name='memory')
                printTensorAttrs(self.memory2, name='memory2')
                printTensorAttrs(self.embed, name='embed')
                printTensorAttrs(self.interneuronNetwork, name='interneuronNetwork')
                printTensorAttrs(self.logits, name='logits')
                for name, p in self.named_parameters():
                    if p.grad is None:
                        ʕっʘ‿ʘʔっ("print no grads")
                        print(f"before = {self.calligraphist.S_apply('dim', f'no grad: {name}')}")
                    else:
                        if debugPrints: ʕっʘ‿ʘʔっ("set yes grads")
                        stats = get_grad_stats(p.grad)
                        shape = stats["shape"]
                        norm = stats["norm"]
                        sparsity = stats["sparsity"]
                        mean = stats["mean"]
                        std = stats["std"]
                        if debugPrints: ʕっʘ‿ʘʔっ("print yes grads")
                        print(f"before = {self.calligraphist.S_apply('almostPerfect', f'yes grad: {name} | shape: {shape} | norm: {norm:.4f} | sparsity: {sparsity:.2%} | mean: {mean:.4f} | std: {std:.4f}')}")
                        debug_print("Loss:", _loss.item())

            if debugPrints: ʕっʘ‿ʘʔっ("loss.backward")
            debug_print(f"windowMAX: {self.numTokensPerStep}")
            _loss.backward()
            debug_print("Logit weights grad norm:", self.logits.l_weights.grad.norm())
            debug_print("LogWindowSizes grad norm:", self.interneuronNetwork.logWindowSizes.grad.norm())
            debug_print("Cerebellum grad norm:", self.interneuronNetwork.cerebellum.grad.norm())
            debug_print("Repetition penalty grad norm:", self.repetitionPenalty.grad.norm())
            #print(next(self.parameters()).grad)

            # --- MOVE GRAD SNAPSHOT/REPORTING HERE (after backward, before zero_grad) ---
            if collect_grad_stats:
                grad_snapshot = self._snapshot_gradients()
                grad_total_norm = None
                if grad_snapshot:
                    grad_log_output = ["\n--- Gradient Snapshot (pre-zero_grad) ---"]
                    for name, stats in grad_snapshot:
                        norm_val = stats["norm"]
                        sparsity_val = stats["sparsity"]
                        mean_val = stats["mean"]
                        std_val = stats["std"]
                        norm_style = self.calligraphist.S_getStat(f"{name}_norm", norm_val)
                        sparsity_style = self.calligraphist.S_getStat(f"{name}_sparsity", sparsity_val)
                        mean_style = self.calligraphist.S_getStat(f"{name}_mean", mean_val)
                        std_style = self.calligraphist.S_getStat(f"{name}_std", std_val)
                        grad_log_output.append(
                            f"{name:<50} | "
                            f"norm: {self.calligraphist.S_apply(norm_style, f'{norm_val:.6f}')} | "
                            f"sparsity: {self.calligraphist.S_apply(sparsity_style, f'{sparsity_val:.6%}')} | "
                            f"mean: {self.calligraphist.S_apply(mean_style, f'{mean_val:.6f}')} | "
                            f"std: {self.calligraphist.S_apply(std_style, f'{std_val:.6f}')}")
                    print("\n".join(grad_log_output))
                else:
                    print("\n--- Gradient Snapshot (pre-zero_grad) ---\n(no gradients recorded)")

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
                        stats = get_grad_stats(p.grad)
                        shape = stats["shape"]
                        norm = stats["norm"]
                        sparsity = stats["sparsity"]
                        mean = stats["mean"]
                        std = stats["std"]
                        if debugPrints: ʕっʘ‿ʘʔっ("print yes grads")
                        print(f"after = {self.calligraphist.S_apply('almostPerfect', f'yes grad: {name} | shape: {shape} | norm: {norm:.4f} | sparsity: {sparsity:.2%} | mean: {mean:.4f} | std: {std:.4f}')}")
            if debugPrints: ʕっʘ‿ʘʔっ("torch.no_grad")
            with torch.no_grad(): # RESET LEARNABLE PARAMETERS
                #self.logLR.data.fill_(math.log(0.00035))  # Learning rate back to 1e-4
                if debugPrints: ʕっʘ‿ʘʔっ("fill scheduledSamplingRate")
                self.scheduledSamplingRate.data.fill_(0.02)  # Scheduled sampling full (no scheduled sampling yet)
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
                if debugPrints: ʕっʘ‿ʘʔっ("clamp logLR")
                clamp_param(self.logLR, math.log(0.0001), math.log(0.001))  # CLAMP IT! IN MEMORY OF THE AMAZING 1.00 SELF LEARNED LOSS RUN OF 27-APRIL-2025! - you certainly dropped the delta! you win!
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
            with torch.no_grad():
                base_clip = 5.0
                sensitivity = 2.5 

                lossDelta_tensor = torch.tensor(_lossDelta, device=self.device)
                adjustment = (lossDelta_tensor * sensitivity)
                clipValue = (base_clip + adjustment).clamp(min=1.0, max=2.5)

            # Clip gradients BEFORE the lock to prevent NaNs
            total_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clipValue.item())
            self.gradientClipMaxNorm = clipValue.item()

            if debugPrints: ʕっʘ‿ʘʔっ("optimizer.step") # Acquire the lock only for the weight update step
            with self.model_thread_lock: self.optimizer.step()
            self.optimizer.zero_grad()

            if collect_grad_stats:
                grad_snapshot = self._snapshot_gradients()
                grad_total_norm = float(total_grad_norm)
            else:
                grad_total_norm = None
                
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
                    "L_PIXELloss": self.PIXELloss,
                    "L_PIXELloss_scaled": self.pixelLoss_used,
                    "L_AUXlossCos": self.AUXlossCos_used,
                    "L_AUXlossKL": self.AUXlossKL_used,
                    "L_LRclamp": self.lrSoftClamp_used,
                    "L_tempClamp": self.tempSoftClamp_used,
                    "L_repPenClamp": self.repPenSoftClamp_used,
                    "L_repLoss": self.repLoss_used,
                    "B_gradClip": self.gradientClipMaxNorm,
                }
                if debugPrints: ʕっʘ‿ʘʔっ("update self.stats with self.backwardStats")
                self.stats.update(self.backwardStats)

            if collect_grad_stats:
                if grad_snapshot:
                    grad_log_output = ["\n--- Gradient Snapshot ---"]
                    for name, stats in grad_snapshot:
                        norm_val = stats["norm"]
                        sparsity_val = stats["sparsity"]
                        mean_val = stats["mean"]
                        std_val = stats["std"]
                        norm_style = self.calligraphist.S_getStat(f"{name}_norm", norm_val)
                        sparsity_style = self.calligraphist.S_getStat(f"{name}_sparsity", sparsity_val)
                        mean_style = self.calligraphist.S_getStat(f"{name}_mean", mean_val)
                        std_style = self.calligraphist.S_getStat(f"{name}_std", std_val)
                        grad_log_output.append(
                            f"{name:<50} | "
                            f"norm: {self.calligraphist.S_apply(norm_style, f'{norm_val:.6f}')} | "
                            f"sparsity: {self.calligraphist.S_apply(sparsity_style, f'{sparsity_val:.6%}')} | "
                            f"mean: {self.calligraphist.S_apply(mean_style, f'{mean_val:.6f}')} | "
                            f"std: {self.calligraphist.S_apply(std_style, f'{std_val:.6f}')}"
                        )
                    if grad_total_norm is not None:
                        grad_log_output.append(f"total grad norm: {grad_total_norm:.6f}")
                    print("\n".join(grad_log_output))
                else:
                    print("\n--- Gradient Snapshot ---\n(no gradients recorded)")
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
    def getResponseFromLogits(self, _logits, _training=False, _totAvgAbsDelta = 0.0, _use_lock: bool = True):
        with self.counsellor.infodump("getResponseFromLogits") as ʕっʘ‿ʘʔっ:
            lock_ctx = self.model_thread_lock if _use_lock else nullcontext()
            with lock_ctx:
                # Ensure incoming logits are finite
                if not torch.isfinite(_logits).all():
                    _logits = torch.nan_to_num(_logits, nan=0.0, posinf=1e3, neginf=-1e3)

                # Clamp temperature to a safe, non-zero range
                raw_temp = torch.exp(self.logTemp)
                safe_temp = raw_temp.clamp(min=0.1, max=5.0)
                # Keep attrs up-to-date for any downstream consumers
                self.temperature = safe_temp
                self.interneuronNetwork.temperature = safe_temp

                # Scale logits and sanitize again to avoid inf/NaN after division
                logits_scaled = _logits / safe_temp
                logits_scaled = torch.nan_to_num(logits_scaled, nan=0.0, posinf=1e3, neginf=-1e3)
                # Optional safety clamp to keep within softmax-stable range
                logits_scaled = logits_scaled.clamp(min=-80.0, max=80.0)

                if logits_scaled.dim() == 1:
                    logits_scaled = logits_scaled.unsqueeze(0)

                # Gumbel-Softmax (robust to tiny tau)
                try:
                    tau = float(safe_temp.detach().cpu().item())
                    tau = max(tau, 1e-2)
                    base_probs = F.gumbel_softmax(logits_scaled, tau=tau, hard=False)
                    assert torch.isfinite(base_probs).all(), "gumbelProbs has NaN or Inf!"
                except Exception as e:
                    self.gumBellend += 1
                    debug_print(f"Gumbel softmax failed: {e}. Falling back to softmax.")
                    base_probs = F.softmax(logits_scaled, dim=-1)
                # Clamp and renormalize to avoid zeros that cause log(0) downstream

            eps = 1e-8
            base_probs = torch.nan_to_num(base_probs, nan=0.0)
            base_probs = base_probs.clamp(min=eps)
            base_probs = base_probs / base_probs.sum(dim=-1, keepdim=True)
            
            if _training:
                self._lastSoftSample_for_loss = base_probs
                self.lastSoftSample = base_probs.detach()
                with torch.no_grad():
                    # Existing creativity metrics
                    eps = 1e-8
                    a = self.memory.FINALmemory
                    b = self.memory2.FINALmemory
                    if a.dim() == 2 and a.size(0) > 1:
                        a = a.mean(dim=0, keepdim=True)
                    if b.dim() == 2 and b.size(0) > 1:
                        b = b.mean(dim=0, keepdim=True)
                    denom = (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(eps)
                    cos_val = (a * b).sum(dim=-1) / denom
                    cos_val = torch.nan_to_num(cos_val, nan=0.0)
                    self.memoryFlux = (1 - cos_val).item()
                    self.cerebralLoad = self.interneuronNetwork.cerebellum.std().item()
                    self.learningStability = _totAvgAbsDelta
                    self.dreamIntensity = (self.memoryFlux * 2.0) + (self.cerebralLoad * 5.0) + (self.learningStability * 1.0)

                    # Simplified sampling without creative modules
                    augmented_probs = base_probs
                    base_p = 0.92
                    top_p = base_p
                    sorted_probs, sorted_indices = torch.sort(augmented_probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    augmented_probs[indices_to_remove] = 0

                    if _training:
                        # take the argmax from the *augmented* distribution - can still influence the training choice
                        responseFromLogits = augmented_probs.argmax(dim=1, keepdim=True)
                    else:
                        if torch.sum(augmented_probs) > 0:
                            responseFromLogits = torch.multinomial(augmented_probs, num_samples=1)
                        else:
                            responseFromLogits = torch.topk(base_probs, 1).indices

                repWindow = torch.exp(self.logRepetitionWindow).item()
                effective_repWindow = repWindow / (1 + repWindow / self.numTokensPerStep)
                self.recentGeneratedTokens.append(responseFromLogits.item())
                if len(self.recentGeneratedTokens) > int(effective_repWindow):
                    self.recentGeneratedTokens.pop(0)

                return responseFromLogits

    def forward_and_sample(self, _inputSeq, _pixel=None, _training=False, _totAvgAbsDelta=0.0):
        """Run ``forward`` and ``getResponseFromLogits`` while holding the model lock once."""

        with self.model_thread_lock:
            logits = self.forward(_inputSeq, _pixel=_pixel, _use_lock=False)
            response = self.getResponseFromLogits(
                logits,
                _training=_training,
                _totAvgAbsDelta=_totAvgAbsDelta,
                _use_lock=False,
            )
        return logits, response
            
    @whocalled    
    def applyRepetitionPenalty(self, _logits, _contextTokens = None):
        with self.counsellor.infodump("applyRepetitionPenalty") as ʕっʘ‿ʘʔっ:
            if not self.recentGeneratedTokens:
                if _contextTokens is None:
                    if debugPrints: ʕっʘ‿ʘʔっ("no recent tokens or context, returning _logits")
                    return _logits
                if debugPrints: ʕっʘ‿ʘʔっ("using context tokens for repetition penalty")
                recentTokens = _contextTokens[-int(self.numTokensPerStep):].detach()
                recentTokens = recentTokens.to(self.device)
                if recentTokens.dtype != torch.long:
                    recentTokens = recentTokens.long()
                recentTokens = recentTokens.reshape(-1)
            else:
                recentTokens = torch.tensor(self.recentGeneratedTokens, device=self.device, dtype=torch.long)
                recentTokens = recentTokens.reshape(-1)

            if debugPrints: ʕっʘ‿ʘʔっ("repWindow = torch.exp(self.logRepetitionWindow)")
            repWindow = torch.exp(self.logRepetitionWindow)
            repWindow = repWindow / (1 + repWindow / self.numTokensPerStep)
            if debugPrints: ʕっʘ‿ʘʔっ("penalty = self.repetitionPenalty")
            penalty = self.repetitionPenalty
            if penalty < 0.0:
                new_value = repetitionPenaltyGOAL
                self.repetitionPenalty.data.copy_(new_value)
                penalty = self.repetitionPenalty

            if isinstance(recentTokens, list):
                if debugPrints: ʕっʘ‿ʘʔっ("recentTokens list -> tensor")
                recentTokens = torch.tensor(recentTokens, device=self.device, dtype=torch.long)
                recentTokens = recentTokens.reshape(-1)
            if debugPrints: ʕっʘ‿ʘʔっ("vocabSize = _logits.shape[1]")
            vocabSize = _logits.shape[1]

            if debugPrints: ʕっʘ‿ʘʔっ("positions = torch.arange(len(recentTokens)).float()")
            positions = torch.arange(len(recentTokens), device = self.device).float()
            if debugPrints: ʕっʘ‿ʘʔっ("windowCenter")
            windowCenter = len(recentTokens) - 0.5  # so token 0 gets proper suppression
            if debugPrints: ʕっʘ‿ʘʔっ("softMask = torch.sigmoid((positions - (windowCenter - repWindow)) * 0.5)")
            #softMask = torch.sigmoid((positions - (windowCenter - repWindow)) * 0.5)
            distance_from_window_start = positions - (len(recentTokens) - repWindow)
            relative_position_in_window = distance_from_window_start / repWindow
            softMask = torch.clamp(relative_position_in_window, 0.0, 1.0)

            if debugPrints: ʕっʘ‿ʘʔっ("computing weighted frequencies")
            softMask = softMask.to(dtype=_logits.dtype)
            weightedFreqs = torch.zeros(vocabSize, device=_logits.device, dtype=_logits.dtype)
            if recentTokens.numel() > 0: weightedFreqs.index_add_(0, recentTokens, softMask)
            weightedFreqs = weightedFreqs.view(1, -1)

            if debugPrints: ʕっʘ‿ʘʔっ("setting penalty to 0 for target token!")
            if self.targetTokenFromTutor is not None:
                weightedFreqs[0, self.targetTokenFromTutor] = 0.0

        return _logits - (weightedFreqs * penalty)

    @whocalled
    def getNextToken(self, _inputSeq):
        with self.counsellor.infodump("getNextToken(FORWARD)") as ʕっʘ‿ʘʔっ:
            ʕっʘ‿ʘʔっ("unpack logits from self.forward")
            if debugPrints:
                try: seq_len = len(_inputSeq)
                except TypeError: seq_len = "unknown"
                ʕっʘ‿ʘʔっ("♥input_seq_len", seq_len)
            with torch.no_grad():
                logits, nextToken = self.forward_and_sample(_inputSeq, _training=True)
            if debugPrints: print("nextToken: ")
            print(f"{nextToken}")
            return nextToken

    @whocalled    
    def saveModel(self, _trainingStepCounter, _totalAvgLoss, _first, filePath = modelFilePath, _newStartIndex = trainingStartIndex):
        with self.counsellor.infodump("saveModel") as ʕっʘ‿ʘʔっ:
            with open(stepCheckpointFilePath, "w") as f:
                if debugPrints or True: print(f"HELLO I AM SAVEMODEL STEPCOUNTER IS {_trainingStepCounter} AND START INDEX IS {_newStartIndex} I SHOULD WRITE {str(_trainingStepCounter+_newStartIndex)} to {stepCheckpointFilePath}")
                f.write(str(_trainingStepCounter+_newStartIndex)) # THIS ISNT REAL, FIX LATER, MAYBE MOVE SAVE AND LOAD TO WAKEUP?
            with open(lossCheckpointAppendFilePath, "a") as f:
                if debugPrints or True: print(f"hi :) i am saveModel... avgLoss is: {_totalAvgLoss}, so... i'm writing {str(_totalAvgLoss)} to {lossCheckpointAppendFilePath}!")
                f.write(str(_totalAvgLoss))
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
                self.interneuronNetwork.temperature = self.temperature
                print(f"loading model from path: {filePath}") 
                self.load_state_dict(torch.load(filePath, map_location=self.device), strict = saveStrict)
                # try loading optimizer separately
                if hasattr(self, "optimizer"):
                    optimPath = filePath + ".optim"
                    if os.path.exists(optimPath):
                        try:
                            self.optimizer.load_state_dict(torch.load(optimPath, map_location=self.device))
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

    def generate(self, _prompt, _numTokens, _temperature):
        """
        Generates a sequence of tokens autoregressively based on a prompt.
        This is the main function called by the bots for inference.
        """
        self.eval() # Set the model to evaluation mode (disables things like dropout)
        self.temperature = torch.exp(self.logTemp).item()
        prompt_tokens = self.librarian.tokenizeText(_prompt)
        gen_token_ids = [self.librarian.tokenToIndex.get(t, 0) for t in prompt_tokens]
        
        response_ids = []

        with torch.no_grad(): # We don't need to calculate gradients during generation
            for _ in range(_numTokens):
                # The context window is the last `numTokensPerStep` tokens
                input_ids = gen_token_ids[-self.numTokensPerStep:]
                input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)
                logits, next_token_tensor = self.forward_and_sample(input_tensor, _training=False)
                next_token_id = next_token_tensor.item()
                gen_token_ids.append(next_token_id)
                response_ids.append(next_token_id)
                # if next_token_id == self.librarian.tokenToIndex.get("<EOS>", -1):
                #     break
        
        # Decode the generated IDs back into a string
        return self.librarian.decodeIDs(response_ids)

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
                    debug_print(f"resetting memory1 after {self.stepsSinceMemoryReset} steps... (learned mem length: {torch.exp(self.logMemoryLength)} ({self.memoryLength}))")
                    self.memory.resetMemory(_memoryLength = self.memoryLength)
                    self.stepsSinceMemoryReset = 0
                if self.stepsSinceMemory2Reset > 3:
                    debug_print(f"resetting memory2 after {self.stepsSinceMemory2Reset} steps... (learned mem length: {torch.exp(self.logMemory2Length)} ({self.memory2Length}))")
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
    def getBabyStats(self): 
        # Creative modules removed; return core stats only
        return dict(self.stats)
    
class PIXEL(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int = 3, *, output_mode: str = "sigmoid", use_layernorm: bool = True, res_scale_init: float = 0.5, _device=modelDevice,):
        super().__init__()
        self.device = _device
        self.linear1 = nn.Linear(in_features, hidden_features, device=self.device)
        self.gelu    = nn.GELU()
        self.linear2 = nn.Linear(hidden_features, out_features, device=self.device)

        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln = nn.LayerNorm(hidden_features, device=self.device)

        self.alpha = nn.Parameter(torch.tensor(res_scale_init, device=self.device))
        self.beta  = nn.Parameter(torch.tensor(res_scale_init, device=self.device))

        self.register_buffer("inv_sqrt2", torch.tensor(1 / math.sqrt(2), device=self.device))

        assert output_mode in {"sigmoid", "clamp", "raw"}
        self.output_mode = output_mode

    def forward(self, x):
        x_res = x
        x = self.linear1(x)
        x = x + self.alpha * x_res

        x_res_gelu = x
        x = self.gelu(x)
        x = x + self.beta * x_res_gelu
        x = x * self.inv_sqrt2

        if self.use_layernorm:
            x = self.ln(x)

        logits = self.linear2(x)

        if self.output_mode == "sigmoid":
            return torch.sigmoid(logits)
        elif self.output_mode == "clamp":
            return torch.clamp(logits, 0.0, 1.0)
        else:
            return logits
    
if __name__ == "__main__":
    exit(0)
