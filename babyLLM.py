# CHARIS CAT 2025
# BABYLLM - babyLLM.py

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim 
from BRAIN.LAYERS.vocab import VOCAB
from BRAIN.LAYERS.embed import EMBED
from BRAIN.LAYERS.interneuronNetwork import INTERNEURON_NETWORK
from BRAIN.LAYERS.logits import LOGITS
from BRAIN.LAYERS.memory import MEMORY
from BRAIN.LAYERS.S_output import *
from SCHOOL.staffroom import languageAndLiterature
from SCHOOL.staffroom.tutor import TUTOR
from SCHOOL.staffroom.counsellor import *
from config import *
from datetime import datetime
import random, os, sys, shutil, time
from collections import Counter

"""this class combines all the core components of the babyLLM:"""
"""EMBED: token embedding layer"""
"""INTERNEURON_NETWORK: layer of parallel neurons for feature extraction"""
"""LOGITS: output layer to generate logits"""
"""it also manages training, loss computation, backpropagation, and response generation."""
class BABYLLM(nn.Module):
    def __init__(self, vocab, embedDimension, numNeurons, activationFunction):
        super().__init__()
        self.counsellor = COUNSELLOR("BabyLLM", debug=debugPrints, durations=durationLogging)

        # MUST BE ON SELF - ONLY ACCESSED IN THIS CLASS AND NOT NN.PARAMS
        self.stats = {}
        self.scheduledSamplingProb = 0
        self.trainingStepCounter = 1
        self.totalTokenEvaluations = 0
        self.totalTokenEvaluations_100 = 0
       
        """CONFIG"""
        optimizerClass = getattr(optim, optimizerName)

        """LAYERS"""
        self.embed = EMBED(vocabSize, embedDimension)
        self.interneuronNetwork = INTERNEURON_NETWORK()
        self.logits = LOGITS(numNeurons = numNeurons, vocabSize = vocabSize)
        self.memory = MEMORY(numNeurons = numNeurons)

        """OPTIMIZER - this updates all of the layers learnable parameters"""
        if debugPrints: print("registered paarameters: ")
        if debugPrints: 
            for name, param in BABYLLM.named_parameters(self): print(name, param.shape)

        self.optimizer = optimizerClass(
        self.parameters(),  # <- collects EVERYTHING
        lr=learningRate, weight_decay=0.001)

        if debugPrints or lossPrints:
            for name, param in self.named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")

        """self.optimizer = optimizerClass(
            list(self.embed.parameters()) +
            list(self.interneuronNetwork.parameters()) + 
            list(self.logits.parameters()) +
            list(self.memory.parameters()),
            lr=learningRate, weight_decay=0.001
        )"""

        # Hold durations counters in a counter object, this dict ensures values are always defined before print (the value needs to be 0 to ensure a noop)
        #self.durationCategories = {"Step": 0, "Save": 0, "Load": 0, "Print": 0, "Logits": 0, "Combine": 0, "Token": 0}
        #self.duration = Counter(self.durationCategories)
        #self.duration_100 = Counter(self.durationCategories)
        self.statsCategories = {"loss": 0, "gradNorm": 0, "logitMin": 0, "logitMax": 0, "scheduledSampling": 0, "tokenCount": 0, "memoryGateShort": 0, "memoryGateLong": 0, "memoryGateCurrent": 0, "shortDecay": 0, "longDecay": 0,}

    def forward(self, inputSeq):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ: # processes input sequence of tokens (str) to generate logits to predict the next token
            if debugPrints: print(f"Debug: Input to forward: {inputSeq}")

            ʕっʘ‿ʘʔっ("inputIndices") # convert inputted tokens to indices (batch processing instead of looping)
            inputIndices = [vocab.tokenToIndex.get(tokenString, vocab.tokenToIndex["<UNK>"]) if not isinstance(tokenString, int) else tokenString for tokenString in inputSeq]

            ʕっʘ‿ʘʔっ("inputEmbeds") # convert indices to embeddings
            inputEmbeds = []
            inputIndicesTensor = torch.tensor(inputIndices, device = modelDevice)
            if lossPrints: print(f"Debug BABYLLM.forward: inputIndicesTensor requires_grad: {inputIndicesTensor.requires_grad} [EXPECTED: FALSE]")
            inputEmbeds = self.embed(inputIndicesTensor)
            if lossPrints: print(f"Debug BABYLLM.forward: inputEmbeds requires_grad: {inputEmbeds.requires_grad} [EXPECTED: TRUE]")

            ʕっʘ‿ʘʔっ("interneuronNetworkOutput") # PARALLEL NEURON LAYER input/processing (feature extraction)
            interneuronNetworkOutput = self.interneuronNetwork.forward(inputEmbeds) 
            if debugPrints: print(f"Debug BABYLLM.forward: interneuronNetworkOutput length: {len(interneuronNetworkOutput)}") 

            ʕっʘ‿ʘʔっ("combinedActivationsTensor") # RESIZE NEURON LAYER TO STANDARD SIZE FOR COMBINED FORWARD PROCESSING
            #combinedActivationsTensor = torch.mean(interneuronNetworkOutput, dim=0, keepdim=True)
            combinedActivationsTensor = interneuronNetworkOutput
            if debugPrints: print("combinedActivationsTensor.requires_grad:", combinedActivationsTensor.requires_grad)
            if debugPrints: print("combinedActivationsTensor.grad_fn:", combinedActivationsTensor.grad_fn)

            ʕっʘ‿ʘʔっ("memoryLayer") # MEMORY LAYER PROCESSING - NOW PROCESS THE COMBINED ACTIVATIONS
            if skipMemory:
                if debugPrints: print("skipping memory layer...")
                latestMemGates = torch.tensor([0.0, 0.0, 1.0], device=modelDevice)  # dummy gates
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
            return logits, interneuronNetworkOutput, inputEmbeds

    
    """computes the cross-entropy loss between the models logits and the target token, essentially checking how good the models prediction was"""        
    def computeLoss(self, logits, targetTokenIndex):
        with self.counsellor.infodump("computeLoss") as ʕっʘ‿ʘʔっ:
            targetTensor = torch.tensor([targetTokenIndex], dtype=torch.long, device = modelDevice)
            if statPrints or debugPrints: print(f"Debug BABYLLM.computeLoss: predictions shape: {logits.shape}\nDebug BABYLLM.computeLoss: predictions (first 10): {logits[0][:10]}\nDebug BABYLLM.computeLoss: targetTokenIndex: {targetTokenIndex}")
            if logits.dim() == 1: logits = logits.unsqueeze(0) 
            ʕっʘ‿ʘʔっ("cross Entropy Loss")
            loss = F.cross_entropy(logits, targetTensor)
            if debugPrints or lossPrints: print(f"Debug BABYLLM.computeLoss: Loss requires_grad: {loss.requires_grad} [EXPECTED: TRUE]")
            if statPrints or debugPrints: print(f"Debug BABYLLM.computeLoss: Loss value: {loss:.4f}")
            return loss
    
    """backpropagation and optimization, computes gradients of the loss and uses the optimizer to update the models weights"""
    def backward(self, loss, durationLogging = durationLogging):
        with self.counsellor.infodump("backward") as ʕっʘ‿ʘʔっ:
            if debugPrints: 
                if not torch.isfinite(loss): 
                    ʕっʘ‿ʘʔっ("if not torch.isfinite(loss)")
                    print("babyLLM.backward.loss.backward !!! Loss is NaN or Inf:", loss)
                    return
                else: print("babyLLM.backward.loss.backward - loss is not NaN or Inf:", loss)
            ʕっʘ‿ʘʔっ("optimizer.zero_grad")
            self.optimizer.zero_grad()
            try:
                for name, p in self.named_parameters():
                    if p.grad is None:
                        if debugPrints or lossPrints: print(f"BEFORE backward: No grad for {name}")
                    else:
                        if debugPrints or lossPrints: print(f"Grad BEFORE backward for {name} - requires_grad: {p.requires_grad}")
                ʕっʘ‿ʘʔっ("loss.backward")
                with torch.autograd.set_detect_anomaly(True):
                    loss.backward()
                for name, p in self.named_parameters():
                    if p.grad is None:
                        if debugPrints or lossPrints: print(f"AFTER backward: No grad for {name}")
                    else:
                        if debugPrints or lossPrints: print(f"Grad AFTER backward for {name} - requires_grad: {p.requires_grad}")
            except RuntimeError as e:
                print("babyLLM.backward.loss.backward failed!", e)
                ʕっʘ‿ʘʔっ("emptyCache")
                
                return
            
            if debugPrints: 
                for name, p in self.named_parameters(): 
                    if p.grad is not None and not torch.isfinite(p.grad).all(): 
                        print(f"babyLLM.backward - non-finite grad in: {name}") 
                        return
                    else: 
                        print("babyLLM.backward - all gradients are finite")
            try:
                ʕっʘ‿ʘʔっ("clip_grad_norm")
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=gradientClipMaxNorm)
                ʕっʘ‿ʘʔっ("optimizer.step")
                self.optimizer.step()
                ʕっʘ‿ʘʔっ("emptyCache")
                #del loss  # as soon as possible after optimizer.step()
                #
            except RuntimeError as e: 
                print("crash during optimizer.step:", e) 
                return

            #if modelDevice.type == 'mps': 

    """this takes the output logits, does temperature scaling and softmax to create a probability distribution over the vocab, and then selects most likely response token"""
    def getResponseFromLogits(self, logits, temperature = temperature, durationLogging = durationLogging):
        with self.counsellor.infodump("getResponseFromLogits") as ʕっʘ‿ʘʔっ:
            logits /= temperature
            if debugPrints or logitPrints: print(f"Debug BABYLLM.getResponseFromLogits: logits shape BEFORE softmax: {logits.shape}")
            probs = torch.softmax(logits, dim=1)
            responseFromLogits = torch.multinomial(probs, 1)
            topVals, topIdxs = torch.topk(probs, 5)
            if statPrints or debugPrints or logitPrints: print("Top token probs:", [(self.getTokenIndexAsString(i), float(v)) for i, v in zip(topIdxs[0], topVals[0])])
            return responseFromLogits

    def getTokenIndexAsString(self, tokenIndex): 
        with self.counsellor.infodump("getTokenIndexAsString") as ʕっʘ‿ʘʔっ:
            tokenIndexAsString = vocab.indexToToken.get(int(tokenIndex), "<UNK>") # tmp fix for token 1999
            return tokenIndexAsString
    
    def getNextToken(self, inputSeq, temperature = temperature):  
        with self.counsellor.infodump("getNextToken(FORWARD)") as ʕっʘ‿ʘʔっ:
            logits, *_ = self.forward(inputSeq) # unpacks the first value of the tuple and ignores the rest
            nextToken = self.getResponseFromLogits(logits, temperature)
            return nextToken

    def getBasicStats(self, logitSeq):
        with self.counsellor.infodump("getBasicStats") as ʕっʘ‿ʘʔっ:
            #gradNorm = (sum((p.grad.norm(2)**2 for p in self.parameters() if p.grad is not None)))**0.5
            stats = {}

            if logitSeq:
                stats["logitMin"] = logitSeq[-1].min(dim=-1).values.mean()
                stats["logitMax"] = logitSeq[-1].max(dim=-1).values.mean()

            stats["scheduledSampling"] = self.scheduledSamplingProb
            return stats
    
    def getComplexStats(self, embeds):
        with self.counsellor.infodump("getComplexStats") as ʕっʘ‿ʘʔっ:
            stats = self.interneuronNetwork.INN_getStats()
            
            stats["embedMean"] = embeds.mean()
            stats["embedStd"] = embeds.std()
            #stats["meanActivation"] = activations.mean()
            #stats["activationSparsity"] = (activations.abs() < 1e-6).float().mean()
            
            stats["shortDecay"] = torch.sigmoid(self.memory.shortTermDecay)
            stats["longDecay"] = torch.sigmoid(self.memory.longTermDecay)
            
            return stats
    
    """calculates and returns display stats, non numbers, as a string"""
    def getStringStats(self, guessedTokenSeq, tokenCounts, tokenCounts_100, logFreq_100=False):
        with self.counsellor.infodump("getStringStats") as ʕっʘ‿ʘʔっ:
            if guessedTokenSeq: 
                ʕっʘ‿ʘʔっ("guessedTokenSeq")
                tokenCounts.update(guessedTokenSeq)
                tokenCounts_100.update(guessedTokenSeq)
            topTokens = tokenCounts.most_common(10)
            topTokens_100 = tokenCounts_100.most_common(10)

            if self.totalTokenEvaluations > 0:
                ʕっʘ‿ʘʔっ("tokenPerfectRate")
                tokenPerfectRate = (self.perfectTokenCount / self.totalTokenEvaluations) * 100
                tokenPerfect_str = f"{S_OUTPUT.S_apply('perfect', f'tokenPerfect: {self.perfectTokenCount} / {self.totalTokenEvaluations}')} → {tokenPerfectRate:.2f}%"
            else: tokenPerfect_str = ""

            if self.totalTokenEvaluations_100 > 0:
                ʕっʘ‿ʘʔっ("tokenPerfectRate_100")
                tokenPerfectRate_100 = (self.perfectTokenCount_100 / self.totalTokenEvaluations_100) * 100
                tokenPerfect_100_str = f"{S_OUTPUT.S_apply('perfect', f'tokenPerfect: {self.perfectTokenCount_100} / {self.totalTokenEvaluations_100}')} → {tokenPerfectRate_100:.2f}%"
            else: tokenPerfect_100_str = ""

            stringStats = {"tokenPerfect": str(tokenPerfect_100_str if logFreq_100 else tokenPerfect_str), "topTokens": str(topTokens_100 if logFreq_100 else topTokens)}

            return stringStats
        
    """saves the model to a file"""    
    def saveModel(self, filePath = modelFilePath):
        with self.counsellor.infodump("saveModel") as ʕっʘ‿ʘʔっ:
            tmpPath = filePath + ".tmp"
            torch.save(self.state_dict(), tmpPath)
            print(f"model temp file created at {tmpPath}...")
            os.replace(tmpPath, filePath)
            print(f"model successfully saved to {filePath}!")
            with open(stepCheckpointFilePath, "w") as f:
                f.write(str({self.trainingStepCounter+startIndex}))

    """loads the model from a file"""
    def loadModel(self, filePath = modelFilePath):
        with self.counsellor.infodump("loadModel") as ʕっʘ‿ʘʔっ:
            try:
                print(f"loading model from path: {filePath}") 
                self.load_state_dict(torch.load(filePath), strict=saveStrict)
                print(f"model loaded from {filePath}!")
                self.to(modelDevice)
                print(f"device set to {modelDevice}!")
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

    
if __name__ == "__main__":
    print(">> trainingFilePath =", trainingFilePath)
    print(">> type =", type(trainingFilePath))
    m_counsellor = COUNSELLOR("__main__", debug=debugPrints, durations=durationLogging)
    with m_counsellor.infodump("babyLLM") as ʕっʘ‿ʘʔっ:
        startIndex = trainingStartIndex # default
        vocab = VOCAB(vocabSize = vocabSize, vocabPath = vocabLoad)
        babyLLM = BABYLLM(vocab = vocab, embedDimension = embedDimension, numNeurons = numNeurons, activationFunction = activationFunction)
        tutor = TUTOR(model = babyLLM, vocab = vocab)

        ʕっʘ‿ʘʔっ("stepCheckpointFilePathCheck")
        if os.path.exists(stepCheckpointFilePath):
            with open(stepCheckpointFilePath, "r") as f:
                try: savedStep = int(f.read().strip())
                except ValueError:
                    babyNote_loadCheckpoint = f"{babyName} 'ah, i couldn't load step checkpoint file from {stepCheckpointFilePath}, resetting to 0...' "
                    print(babyNote_loadCheckpoint)
                    savedStep = 0
        else:
            babyNote_loadCheckpoint = f"{babyName} 'ah, the step checkpoint file {stepCheckpointFilePath} doesn't exist, resetting to 0...' "
            print(babyNote_loadCheckpoint)
            savedStep = 0

        babyNote_loadCheckpointCheck = f"{babyName} 'right, last time i got to step {savedStep}... want to restart from there?' "
        ʕっʘ‿ʘʔっ("choice = input♥")
        choice = input(babyNote_loadCheckpointCheck + f"\n{userName}: ").lower()

        if choice == "" or choice.startswith("y"):
            ʕっʘ‿ʘʔっ("♥choice = y")
            startIndex = savedStep
            babyNote_loadCheckpoint = f"{babyName} 'ok! let's go to step {savedStep}! "
            print(babyNote_loadCheckpoint, end="")

        elif choice.startswith("r") or choice in ["random", "i dont care", "i don't care", "idc"]:
            ʕっʘ‿ʘʔっ("♥choice = r")
            startIndex = random.randint(0, len(vocab.tokens) - windowMAX - 1)
            babyNote_loadCheckpoint = f"{babyName} 'oh, cool! i'll pick a random spot to start from... umm... let's go to step {startIndex}! "
            print(babyNote_loadCheckpoint, end="")

        elif choice.startswith("n") or choice in ["start again", "restart"]:
            ʕっʘ‿ʘʔっ("♥choice = n")
            startIndex = trainingStartIndex
            babyNote_loadCheckpoint = f"{babyName} 'alright, step {startIndex}, let's go back to the beginning :) "
            print(babyNote_loadCheckpoint, end="")
            
        elif choice.isdigit():
            ʕっʘ‿ʘʔっ("♥choice = digit")
            startIndex = int(choice)
            babyNote_loadCheckpoint = f"{babyName} 'damn that's specific! heading to step {startIndex}... "
            print(babyNote_loadCheckpoint, end="")

        else:
            ʕっʘ‿ʘʔっ("♥choice = None")
            startIndex = trainingStartIndex
            babyNote_loadCheckpoint = f"{babyName} 'umm... i don't think i heard you properly, i'll just start from step {startIndex} :) but, "
            print(babyNote_loadCheckpoint, end="")

        ʕっʘ‿ʘʔっ("runStart")
        babyNote_runStart = f"what am i learning today?'" # no tag of 'babyllm:' because it merges with the end of above message in logs
        userNote_runStart = f"{userName}: '" + input(babyNote_runStart + f"\n{userName}: ").strip().lower() + "'"
        userNote_loadCheckpoint = f"{userName}: '{choice}'"

        #TESTinputSeq = ["what","will","you","do","out","there","now","?"]
        TESTinputSeq = ["i","love","you","this","is","good","music","is","life","hey","how","are","you","?"]
        #TESTinputSeq = ["what"] 

        #ʕっʘ‿ʘʔっ("babyHello")
        #babyLLM.to(modelDevice)
        #guessedToken = babyLLM.getNextToken(TESTinputSeq[-windowMAX:])
        #babyHello = print(f"{babyName}: '{vocab.indexToToken.get(guessedToken, '<UNK>')}'")

        ʕっʘ‿ʘʔっ("loadModel")
        babyLLM.loadModel()
        #if retokenizeOnLoad: 
            #ʕっʘ‿ʘʔっ("genTrainingData")
            #trainingDataPairs = vocab.genTrainingData(windowMAX, startIndex=startIndex)
        #if saveTokenizedData: 
            #ʕっʘ‿ʘʔっ("getOrCreateTrainingData")
            #trainingDataPairs = vocab.getOrCreateTrainingData(windowMAX, )
        ʕっʘ‿ʘʔっ("genTrainingDataFromTokens")
        babyLLM.babyNote_loadCheckpointCheck = babyNote_loadCheckpointCheck
        babyLLM.userNote_loadCheckpoint = userNote_loadCheckpoint
        babyLLM.babyNote_loadCheckpoint = babyNote_loadCheckpoint
        babyLLM.babyNote_runStart = babyNote_runStart
        babyLLM.userNote_runStart = userNote_runStart
        trainingDataPairs = vocab.genTrainingData(windowMAX, startIndex = startIndex)
        print(f"Total trainingDataPairs: {len(trainingDataPairs)}")
        babyLLM.to(modelDevice)
        tutor.trainModel(trainingDataPairs, epochs = epochs, startIndex = startIndex, model=babyLLM)
