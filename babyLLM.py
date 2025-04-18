# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# BABYLLM // babyLLM.py

import random, os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim 

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
    def __init__(self, _counsellor, _s_output, _scribe, _librarian, _wobble, _device = modelDevice):
        super().__init__()
        self.device = _device
        self.counsellor = _counsellor
        self.s_output = _s_output
        self.scribe = _scribe
        self.librarian = _librarian
        #self.wobble = _wobble

        # MUST BE ON SELF - ONLY ACCESSED IN THIS CLASS AND NOT NN.PARAMS
        self.stats = {}
        self.scheduledSampling = 0
        self.totalTokenEvaluations = 0
        self.totalTokenEvaluations_100 = 0
        self.recentGeneratedTokens = []  # used for repetition penalty
        self.learningRate = learningRate

        """CEREBRAL LAYERS // BRAIN"""
        self.embed = EMBED(_counsellor = self.counsellor, _device = self.device)
        self.interneuronNetwork = INTERNEURON_NETWORK(_counsellor = self.counsellor, _device = self.device)
        self.logits = LOGITS(_counsellor = self.counsellor, _device = self.device)
        self.memory = MEMORY(_counsellor = self.counsellor, _device = self.device)

        """OPTIMIZER - this updates all of the layers learnable parameters"""
        if debugPrints: print("registered paarameters: ")
        if debugPrints: 
            for name, param in BABYLLM.named_parameters(self): print(name, param.shape)

        optimizerClass = getattr(optim, optimizerName)
        self.optimizer = optimizerClass(self.parameters(), lr=learningRate, weight_decay=0.001)

        if debugPrints:
            for name, param in self.named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")

        """self.optimizer = optimizerClass(
            list(self.embed.parameters()) +
            list(self.interneuronNetwork.parameters()) + 
            list(self.logits.parameters()) +
            list(self.memory.parameters()),
            lr=learningRate, weight_decay=0.001
        )"""

        #self.to(self.device)
        self.statsCategories = {"loss": 0, "gradNorm": 0, "logitMin": 0, "logitMax": 0, "scheduledSampling": 0, "tokenCount": 0, "memoryGateShort": 0, "memoryGateLong": 0, "memoryGateCurrent": 0, "shortDecay": 0, "longDecay": 0,}

    def forward(self, _inputSeq):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ: # processes input sequence of tokens (str) to generate logits to predict the next token
            if debugPrints: print(f"Debug: Input to forward: {_inputSeq}")

            ʕっʘ‿ʘʔっ("inputEmbeds") # convert indices to embeddings
            inputEmbeds = []
            inputEmbeds = self.embed(_inputSeq) # DIRECTLY TAKING A TENSOR NOW
            if debugPrints: print(f"Debug BABYLLM.forward: inputEmbeds requires_grad: {inputEmbeds.requires_grad} [EXPECTED: TRUE]")

            ʕっʘ‿ʘʔっ("interneuronNetworkOutput") # PARALLEL NEURON LAYER input/processing (feature extraction)
            interneuronNetworkOutput = self.interneuronNetwork.forward(inputEmbeds) 
            if debugPrints: print(f"Debug BABYLLM.forward: interneuronNetworkOutput length: {len(interneuronNetworkOutput)}") 

            ʕっʘ‿ʘʔっ("combinedActivationsTensor") # RESIZE NEURON LAYER TO STANDARD SIZE FOR COMBINED FORWARD PROCESSING
            combinedActivationsTensor = interneuronNetworkOutput
            if debugPrints: print("combinedActivationsTensor.requires_grad:", combinedActivationsTensor.requires_grad)
            if debugPrints: print("combinedActivationsTensor.grad_fn:", combinedActivationsTensor.grad_fn)

            ʕっʘ‿ʘʔっ("memoryLayer") # MEMORY LAYER PROCESSING - NOW PROCESS THE COMBINED ACTIVATIONS
            if skipMemory:
                if debugPrints: print("skipping memory layer...")
                latestMemGates = torch.tensor([0.0, 0.0, 1.0], device=self.device)  # dummy gates
                combinedActivations = combinedActivationsTensor.detach()  # no grad path, super light
            else:
                memoryOutput = self.memory.forward(combinedActivationsTensor)
                latestMemGates = self.memory.latestMemoryGates
                combinedActivations = memoryOutput
            if debugPrints: print("combinedActivations.requires_grad:", combinedActivations.requires_grad)

            ʕっʘ‿ʘʔっ("logits.forward")
            if debugPrints: print("memory output requires_grad?", self.memory.longTermMemory.requires_grad)
            logits = self.logits.forward(combinedActivations)  
            if debugPrints: print("memory output requires_grad?", self.memory.longTermMemory.requires_grad)

            """returns a logits tensor of shape (1, vocabSize) showing predicted probabilities for the next token"""
            return logits
    
    """computes the cross-entropy loss between the models logits and the target token, essentially checking how good the models prediction was"""        
    def computeLoss(self, _logits, _targetTokenIndex):
        with self.counsellor.infodump("computeLoss") as ʕっʘ‿ʘʔっ:
            if skipComputeLoss:
                ʕっʘ‿ʘʔっ("skipping loss!")
                return torch.tensor([0.1], requires_grad=True, device=self.device)  # Constant scalar tensor
            else:     
                ʕっʘ‿ʘʔっ("targetTensor")          
                targetTensor = torch.tensor([_targetTokenIndex], dtype=torch.long, device = self.device)
                #if debugPrints: print(f"[LOSS DEBUG] logits shape: {logits.shape} | target: {targetTokenIndex}")
                if _logits.dim() == 1: _logits = _logits.unsqueeze(0) # ensure logits are at least 2d
                ʕっʘ‿ʘʔっ("cross Entropy Loss")
                loss = F.cross_entropy(_logits, targetTensor)
            #if debugPrints: print(f"[LOSS DEBUG] requires_grad: {loss.requires_grad} | value: {loss.detach().cpu().item():.4f}")
            return loss
    
    """backpropagation and optimization, computes gradients of the loss and uses the optimizer to update the models weights"""
    def backward(self, _loss):
        with self.counsellor.infodump("backward") as ʕっʘ‿ʘʔっ:
            #if not torch.isfinite(_loss): 
                #ʕっʘ‿ʘʔっ("if not torch.isfinite(loss)")
                #print("babyLLM.backward.loss.backward !!! Loss is NaN or Inf:", _loss)
                #return
            for name, p in self.named_parameters():
                if p.grad is None:
                    if debugPrints: print(f"NO GRAD before backward: No grad for {name}")
                else:
                    if debugPrints: print(f"grad before backward for {name} - requires_grad: {p.requires_grad}")
            with torch.autograd.set_detect_anomaly(anomalyDetect):
                ʕっʘ‿ʘʔっ("loss.backward")
                _loss.backward()
                #print(next(self.parameters()).grad)
            for name, p in self.named_parameters():
                if p.grad is None:
                    if debugPrints: print(f"NO GRAD after backward: No grad for {name}")
                else: 
                    if debugPrints: 
                        print(f"grad after backward for {name} - requires_grad: {p.requires_grad}")
            #for name, p in self.named_parameters():
                #if p.grad is not None and not torch.isfinite(p.grad).all():
                    #print(f"babyLLM.backward - non-finite grad in: {name}") 
                    #return
            ʕっʘ‿ʘʔっ("optimizer.step")
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = gradientClipMaxNorm)
            self.optimizer.step()  # Update weights
                
    """this takes the output logits, does temperature scaling and softmax to create a probability distribution over the vocab, and then selects most likely response token"""
    """def getResponseFromLogits(self, _logits, _temperature = temperature):
        with self.counsellor.infodump("getResponseFromLogits") as ʕっʘ‿ʘʔっ:
            _logits /= _temperature
            if debugPrints: print(f"Debug BABYLLM.getResponseFromLogits: logits shape BEFORE softmax: {_logits.shape}")
            if _logits.dim() == 1: _logits = _logits.unsqueeze(0)
            probs = torch.softmax(_logits, dim=1)
            responseFromLogits = torch.multinomial(probs, 1)
            return responseFromLogits"""
        

    def getResponseFromLogits(self, _logits, _temperature = temperature, _repetitionPenalty = repetitionPenalty):
        with self.counsellor.infodump("getResponseFromLogits") as ʕっʘ‿ʘʔっ:
            self.repetitionPenalty = _repetitionPenalty
            _logits / _temperature

            if _logits.dim() == 1: _logits = _logits.unsqueeze(0)  # ensure [1, vocabSize]

            # repetition penalty
            if self.recentGeneratedTokens:
                last_tokens = self.recentGeneratedTokens[-penaltyWindow:]
                unique_tokens = set(last_tokens)
                repeated_tokens = [t for t in last_tokens if last_tokens.count(t) > 1]
                repeatedPercent = len(repeated_tokens) / len(last_tokens)

                self.stats.update({"repetitionRatio": repeatedPercent})

                for token in unique_tokens:
                    _logits[0, token] /= self.repetitionPenalty  # reduce score of repeated token

                if debugPrints:
                    print(f"[REP PENALTY] {repeatedPercent:.2%} repeated | Penalised: {[self.librarian.indexToToken.get(t, '<UNK>') for t in unique_tokens]}")

            # Sample from softmaxed logits
            probs = torch.softmax(_logits, dim=1)
            responseFromLogits = torch.multinomial(probs, 1)

            # history buffer
            self.recentGeneratedTokens.append(responseFromLogits.item())
            if len(self.recentGeneratedTokens) > penaltyWindow * 2:
                self.recentGeneratedTokens.pop(0)

            return responseFromLogits

    def getNextToken(self, _inputSeq, _temperature = temperature):  
        with self.counsellor.infodump("getNextToken(FORWARD)") as ʕっʘ‿ʘʔっ:
            logits, *_ = self.forward(_inputSeq) # unpacks the first value of the tuple and ignores the rest
            nextToken = self.getResponseFromLogits(logits, _temperature)
            return nextToken
        
    def saveModel(self, filePath = modelFilePath, _newStartIndex = trainingStartIndex, _trainingStepCounter = 0):
        with self.counsellor.infodump("saveModel") as ʕっʘ‿ʘʔっ:
            tmpPath = filePath + ".tmp"
            torch.save(self.state_dict(), tmpPath)
            print(f"model temp file created at {tmpPath}...")
            os.replace(tmpPath, filePath)
            print(f"model successfully saved to {filePath}!")
            with open(stepCheckpointFilePath, "w") as f:
                f.write(str(_trainingStepCounter+_newStartIndex)) # THIS ISNT REAL, FIX LATER, MAYBE MOVE SAVE AND LOAD TO WAKEUP?

    """loads the model from a file"""
    def loadModel(self, filePath = modelFilePath):
        with self.counsellor.infodump("loadModel") as ʕっʘ‿ʘʔっ:
            try:
                print(f"loading model from path: {filePath}") 
                self.load_state_dict(torch.load(filePath), strict=saveStrict)
                print(f"model loaded from {filePath}!")
                self.to(self.device)
                print(f"device set to {self.device}!")
                self.resetMemory(context="inference")
                
            except FileNotFoundError: print("No saved model found")

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
                ʕっʘ‿ʘʔっ("context=inference")
                self.memory.resetMemory()
                print(f"resetting memory for new conversation...")
            elif context == "training":
                ʕっʘ‿ʘʔっ("context=training")
                if hasattr(self, "stepsSinceMemoryReset"): 
                    self.stepsSinceMemoryReset += 1
                else: 
                    self.stepsSinceMemoryReset = 1
                if self.stepsSinceMemoryReset >= memoryLength: 
                    self.memory.resetMemory()
                    self.stepsSinceMemoryReset = 0 
                    print(f"resetting memory after {memoryLength} steps...")

    def setLearningRate(self, _newLearningRate):
        self.learningRate = max(1e-6, min(_newLearningRate, 0.01))  # clamp it a bit
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learningRate
    
if __name__ == "__main__":
    exit(0)