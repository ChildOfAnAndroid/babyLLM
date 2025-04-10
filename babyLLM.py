# CHARIS CAT 2025
# BABYLLM - babyLLM.py

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim 
from BRAIN.LAYERS.vocab import VOCAB
from BRAIN.LAYERS.embedLayer import EMBEDLAYER
from BRAIN.LAYERS.parallelNeuronLayer import PARALLELNEURONLAYER
from BRAIN.LAYERS.outputLayer import OUTPUTLAYER
from BRAIN.LAYERS.memoryLayer import MEMORYLAYER
import BRAIN.LAYERS.S_output as S_output
from config import *
from datetime import datetime
import random, os, sys, shutil, time
from collections import Counter

"""this class combines all the core components of the babyLLM:"""
"""EMBEDLAYER: token embedding layer"""
"""PARALLELNEURONLAYER: layer of parallel neurons for feature extraction"""
"""OUTPUTLAYER: output layer to generate logits"""
"""MULTIWINDOWLAYER: (New) layer to incorporate multi-window context"""
"""it also manages training, loss computation, backpropagation, and response generation."""
class BABYLLM(nn.Module):
    def __init__(self, vocab, embedDimension, numNeurons, activationFunction):
        super().__init__()
        """CONFIG"""
        self.vocabSize = vocabSize
        self.vocab = vocab
        self.embedDimension = embedDimension
        self.numNeurons = numNeurons
        self.learningRate = learningRate
        self.temperature = temperature
        self.activationFunction = activationFunction
        self.recentPrintLosses = []
        optimizerClass = getattr(optim, optimizerName)

        """LAYERS"""
        self.embedLayer = EMBEDLAYER(vocabSize, self.embedDimension)
        self.parallelNeuronLayer = PARALLELNEURONLAYER(numNeurons = self.numNeurons, embedDimension = self.embedDimension, activationFunction = self.activationFunction)
        self.outputLayer = OUTPUTLAYER(numNeurons = self.numNeurons, vocabSize = self.vocabSize)
        self.memoryLayer = MEMORYLAYER(numNeurons = self.numNeurons)

        """OPTIMIZER - this updates all of the layers learnable parameters"""
        #print("Registered Parameters:")
        #for name, param in BABYLLM.named_parameters(self):
        #    print(name, param.shape)
        self.optimizer = optimizerClass(
            list(self.embedLayer.parameters()) +
            list(self.parallelNeuronLayer.parameters()) + 
            list(self.outputLayer.parameters()) +
            list(self.memoryLayer.parameters()),
            lr=learningRate, weight_decay=0.001
        )

        self.scheduledSamplingProb = 0.0
        self.perfectTokenCount = 0
        self.perfectTokenCount_100 = 0
        self.totalTokenEvaluations = 0
        self.totalTokenEvaluations_100 = 0

        # Hold durations counters in a counter object, this dict ensures values are always defined before print (the value needs to be 0 to ensure a noop)
        self.durationCategories = {"Step": 0, "Save": 0, "Load": 0, "Print": 0, "Logits": 0, "Combine": 0, "Token": 0}

        self.duration = Counter(self.durationCategories)
        self.duration_100 = Counter(self.durationCategories)

        self.statsCategories = {"loss": 0, "gradNorm": 0, "logitMin": 0, "logitMax": 0, "scheduledSampling": 0, "tokenCount": 0, "memoryGateShort": 0, "memoryGateLong": 0, "memoryGateCurrent": 0, "shortDecay": 0, "longDecay": 0,}

    """processes input sequence of tokens (str) to generate logits to predict the next token"""
    def forward(self, inputSeq):
        #forwardStart = time.time()
        #print(f"Debug: Input to forward: {inputSeq}")

        """convert inputted tokens to indices (batch processing instead of looping)"""
        #print(f"Debug BABYLLM.forward: inputIndices: {inputIndices}")
        #inputTimestamp = time.time()
        inputIndices = [self.vocab.tokenToIndex.get(tokenString, self.vocab.tokenToIndex["<UNK>"]) if not isinstance(tokenString, int) else tokenString for tokenString in inputSeq]
        #idxTime = time.time() - inputTimestamp

        """convert indices to embeddings"""
        #embedTimestamp = time.time()
        inputEmbeds = []
        inputIndicesTensor = torch.tensor(inputIndices, device = modelDevice)
        inputEmbeds = self.embedLayer(inputIndicesTensor)
        #embedTime = time.time() - embedTimestamp

        #neuronTimestamp = time.time()
        """PARALLEL NEURON LAYER input/processing (feature extraction)"""
        parallelNeuronOutput = self.parallelNeuronLayer.forward(inputEmbeds) 
        #print(f"Debug BABYLLM.forward: parallelNeuronOutput length: {len(parallelNeuronOutput)}") 

        """RESIZE NEURON LAYER TO STANDARD SIZE FOR COMBINED FORWARD PROCESSING"""
        combinedActivationsTensor = torch.mean(parallelNeuronOutput, dim=0, keepdim=True)
        #neuronTime = time.time() - neuronTimestamp

        #memoryTimestamp = time.time()
        """MEMORY LAYER PROCESSING - NOW PROCESS THE COMBINED ACTIVATIONS"""
        memoryLayerOutput = self.memoryLayer.forward(combinedActivationsTensor)
        self.latestMemGates = self.memoryLayer.latestMemoryGates.detach() 
        combinedActivations = memoryLayerOutput
        #memoryTime = time.time() - memoryTimestamp

        #outputTimestamp = time.time()
        logits = self.outputLayer.forward(combinedActivations)  
        #outputTime = time.time() - outputTimestamp

        #forwardTotal = time.time() - forwardStart
        """returns a logits tensor of shape (1, vocabSize) showing predicted probabilities for the next token"""
        return logits, parallelNeuronOutput, inputEmbeds, F.softmax(self.parallelNeuronLayer.windowWeighting, dim=0), self.memoryLayer.longTermMemory, self.memoryLayer.shortTermMemory

    
    """computes the cross-entropy loss between the models logits and the target token, essentially checking how good the models prediction was"""
    def computeLoss(self, logits, targetTokenIndex):
        targetTensor = torch.tensor([targetTokenIndex], dtype=torch.long, device = modelDevice)
        #print(f"Debug BABYLLM.computeLoss: predictions shape: {logits.shape}\nDebug BABYLLM.computeLoss: predictions (first 10): {logits[:10]}\nDebug BABYLLM.computeLoss: targetTokenIndex: {targetTokenIndex}")
        if logits.dim() == 1: logits = logits.unsqueeze(0) 
        #print(f"Debug BABYLLM.computeLoss: Loss value: {self.loss.item():.4f}")
        """returns a scalar tensor representing the cross-entropy loss value"""
        return  F.cross_entropy(logits, targetTensor)
    
    """backpropagation and optimization, computes gradients of the loss and uses the optimizer to update the models weights"""
    def backward(self, loss, durationLogging = durationLogging):
        if durationLogging: babyLLM_backwardStart = time.time()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=gradientClipMaxNorm)
        self.optimizer.step()
        if modelDevice.type == 'mps':
            torch.mps.empty_cache()

        if durationLogging: babyLLM_backwardDuration = time.time() - babyLLM_backwardStart, self.duration.update({"babyLLM_backward": babyLLM_backwardDuration}), self.duration_100.update({"babyLLM_backward": babyLLM_backwardDuration})

    """this takes the output logits, does temperature scaling and softmax to create a probability distribution over the vocab, and then selects most likely response token"""
    def getResponseFromLogits(self, logits, temperature=None, durationLogging = durationLogging):
        if durationLogging: babyLLM_getResponseFromLogitsStart = time.time()

        if temperature is None: temperature = self.temperature
        logits /= temperature

        if durationLogging: babyLLM_getResponseFromLogitsDuration = time.time() - babyLLM_getResponseFromLogitsStart, self.duration.update({"BabyLLM_getResponseFromLogits": babyLLM_getResponseFromLogitsDuration}), self.duration_100.update({"babyLLM_getResponseFromLogits": babyLLM_getResponseFromLogitsDuration})
        return torch.multinomial(torch.softmax(logits, dim=1), 1).item()
    
    def getTokenIndexAsString(self, tokenIndex): return self.vocab.indexToToken.get(int(tokenIndex), "<UNK>") # tmp fix for token 1999
    
    def getNextToken(self, inputSeq, temperature=None, durationLogging = durationLogging):  
        if durationLogging: babyLLM_getNextTokenStart = time.time()

        if temperature is None: temperature = self.temperature
        logits, *_ = self.forward(inputSeq)  # <-- just grab logits
        
        if durationLogging: babyLLM_getNextTokenDuration = time.time() - babyLLM_getNextTokenStart, self.duration.update({"babyLLM_getNextToken": babyLLM_getNextTokenDuration}), self.duration_100.update({"babyLLM_getNextToken": babyLLM_getNextTokenDuration})
        return self.getResponseFromLogits(logits, temperature)
    
    def trainStep(self, inputTokenIndices, targetTokenIndexSeq):
        if durationLogging: babyLLM_trainStepStart = time.time()
        predictedTokenIndices = []
        inputSeqPredictions = list(inputTokenIndices) # Start with input context, create a COPY!
        losses = []
        logitSeq = []
        cumulativeLoss = 0.0 # Sum of losses for THIS sequence

        # Predict multiple tokens in a sequence
        for j in range(numTokensPerStep):
            if durationLogging: babyLLM_trainStep_forwardStart = time.time()
            logits, activations, embeds, windowWeights, longMem, shortMem = self.forward(inputSeqPredictions)
            if durationLogging: babyLLM_trainStep_forwardDuration = time.time() - babyLLM_trainStep_forwardStart, self.duration.update({"babyLLM_trainStep_forward": babyLLM_trainStep_forwardDuration}), self.duration_100.update({"babyLLM_trainStep_forward": babyLLM_trainStep_forwardDuration})
            
            #predictTimestamp = time.time()
            predictedTokenIndex = self.getResponseFromLogits(logits)
            logitSeq.append(logits)
            predictedTokenIndices.append(predictedTokenIndex)

            if scheduledSampling and random.random() < self.scheduledSamplingProb: nextTokenInput = predictedTokenIndex
            else: nextTokenInput = targetTokenIndexSeq[j] if j < len(targetTokenIndexSeq) else predictedTokenIndex
            inputSeqPredictions.append(nextTokenInput)
            #predictTime = time.time() - predictTimestamp

            #lossTimestamp = time.time()
            if j < len(targetTokenIndexSeq):
                self.totalTokenEvaluations += 1
                self.totalTokenEvaluations_100 += 1
                if predictedTokenIndex == targetTokenIndexSeq[j]:
                    self.perfectTokenCount += 1
                    self.perfectTokenCount_100 += 1
                stepLoss = self.computeLoss(logits, targetTokenIndexSeq[j])
                losses.append(stepLoss)
                cumulativeLoss += stepLoss
                self.recentPrintLosses.append(stepLoss.item())
                if len(self.recentPrintLosses) > printFreq: self.recentPrintLosses.pop(0)

        loss = cumulativeLoss / len(losses) if losses else torch.tensor(0.0, device=modelDevice)
        self.scheduledSamplingProb = min(self.scheduledSamplingProb + scheduledSamplingProbIncrement, 1.0)
        #lossTime = time.time() - lossTimestamp

        self.backward(loss)

        if durationLogging: babyLLM_trainStepDuration = time.time() - babyLLM_trainStepStart, self.duration.update({"babyLLM_trainStep": babyLLM_trainStepDuration}), self.duration_100.update({"babyLLM_trainStep": babyLLM_trainStepDuration})
        return loss, predictedTokenIndices, logitSeq, embeds, activations, windowWeights, longMem, shortMem, losses

    def getBasicStats(self, loss, logitSeq, latestMemGates, losses, durationLogging = durationLogging):
        if durationLogging: babyLLM_getBasicStatsStart = time.time()
        gradNorm = (sum((p.grad.data.norm(2)**2 for p in self.parameters() if p.grad is not None)))**0.5
        stats = {"loss": loss.item(), "gradNorm": gradNorm, "tokenCount": len(losses),}

        if logitSeq:
            stats["logitMin"] = logitSeq[-1].min(dim=-1).values.mean().item()
            stats["logitMax"] = logitSeq[-1].max(dim=-1).values.mean().item()

        stats["scheduledSampling"] = self.scheduledSamplingProb

        if durationLogging: babyLLM_getBasicStatsDuration = time.time() - babyLLM_getBasicStatsStart, self.duration.update({"babyLLM_getBasicStats": babyLLM_getBasicStatsDuration}), self.duration_100.update({"babyLLM_getBasicStats": babyLLM_getBasicStatsDuration})
        return stats
    
    def getComplexStats(self, embeds, activations, windowWeights, latestMemGates):
        if durationLogging: babyLLM_getComplexStatsStart = time.time()
        probs = torch.softmax(self.parallelNeuronLayer.windowWeighting, dim=0)
        stats = {}

        stats["embedMean"] = embeds.mean().item() ##-0
        stats["embedStd"] = embeds.std().item() ##??
        stats["meanActivation"] = activations.mean().item() ##0
        stats["activationSparsity"] = (activations.abs() < 1e-6).float().mean().item() ##0

        stats["windowStd"] = probs.std().item() ##??
        stats["windowEntropy"] = -(probs * torch.log(probs + 1e-8)).sum().item()
        stats["topWindowWeight"] = probs.max().item()
        stats["effectiveWindowCount"] = torch.exp(torch.tensor(stats["windowEntropy"])).item()

        stats["shortDecay"] = torch.sigmoid(self.memoryLayer.shortTermDecay).item() ##
        stats["longDecay"] = torch.sigmoid(self.memoryLayer.longTermDecay).item() ##

        if latestMemGates is not None: ## all seem v tiny numbers
            stats["memoryGateShort"] = latestMemGates[0].item()
            stats["memoryGateLong"] = latestMemGates[1].item()
            stats["memoryGateCurrent"] = latestMemGates[2].item()
            stats["memoryGateMean"] = latestMemGates.mean().item()
            stats["memoryGateStd"] = latestMemGates.std().item()

        if durationLogging: babyLLM_getComplexStatsDuration = time.time() - babyLLM_getComplexStatsStart, self.duration.update({"babyLLM_getComplexStats": babyLLM_getComplexStatsDuration}), self.duration_100.update({"babyLLM_getComplexStats": babyLLM_getComplexStatsDuration})
        return stats
    
    """calculates and returns display stats, non numbers, as a string"""
    def getStringStats(self, guessedTokenSeq, tokenCounts, tokenCounts_100, logFreq_100=False):
        if guessedTokenSeq: tokenCounts.update(guessedTokenSeq), tokenCounts_100.update(guessedTokenSeq)
        topTokens = tokenCounts.most_common(10)
        topTokens_100 = tokenCounts_100.most_common(10)

        if self.totalTokenEvaluations > 0:
            tokenPerfectRate = (self.perfectTokenCount / self.totalTokenEvaluations) * 100
            tokenPerfect_str = f"{S_output.S_apply('perfect', f'tokenPerfect: {self.perfectTokenCount} / {self.totalTokenEvaluations}')} → {tokenPerfectRate:.2f}%"
        else: tokenPerfect_str = ""

        if self.totalTokenEvaluations_100 > 0:
            tokenPerfectRate_100 = (self.perfectTokenCount_100 / self.totalTokenEvaluations_100) * 100
            tokenPerfect_100_str = f"{S_output.S_apply('perfect', f'tokenPerfect: {self.perfectTokenCount_100} / {self.totalTokenEvaluations_100}')} → {tokenPerfectRate_100:.2f}%"
        else: tokenPerfect_100_str = ""

        with torch.no_grad():
            raw = self.parallelNeuronLayer.windowWeighting.detach().cpu().numpy()
            softmaxed = torch.softmax(self.parallelNeuronLayer.windowWeighting, dim=0).detach().cpu().numpy()
            hybridWeights = sorted(zip(allWindowSizes, softmaxed, raw), key=lambda x: x[1], reverse=True)
            windowWeights_str = ",".join(
                f"W{weight}:{raw:.5f} ({softmaxed:.2f})" if logFreq_100 else f"W{weight}:{raw:.3f} ({softmaxed:.1f})"
                for weight, softmaxed, raw in hybridWeights)

        return {"tokenPerfect": str(tokenPerfect_100_str if logFreq_100 else tokenPerfect_str), "topTokens": str(topTokens_100 if logFreq_100 else topTokens), "windowWeights": windowWeights_str,}

    """this iterates through training data, performing forward passes, loss computation, backpropagation, and optimization for each step."""
    def trainModel(self, trainingDataPairs, epochs, startIndex = trainingStartIndex):
        #print(f"Debug tokenToIndex (First 20): {list(vocab.tokenToIndex.items())[:20]}")
        self.startIndex = startIndex
        self.trainingStepCounter = 1  

        numTokens = numTokensPerStep
        if isinstance(numTokens, torch.Tensor): numTokens = numTokens.item()
        numTokens = int(numTokens)
     
        tokenCounts = Counter()
        tokenCounts_100 = Counter()
        self.stats = Counter(self.statsCategories)
        self.stats_100 = Counter(self.statsCategories)
        
        print("babyLLM is heading back to school...")
        """EPOCH LOOP"""
        for epoch in range(epochs):
            print(f"--- Epoch {epoch+1}/{epochs} Started ---")
            
            """TRAINING DATA (batches)"""
            try:
                for i, (inputSeq, targetSeq) in enumerate(trainingDataPairs):
                    stepStartTime = time.time()
                    inputTokenIndices = [vocab.tokenToIndex.get(token, vocab.tokenToIndex["<UNK>"]) for token in inputSeq]
                    targetTokenIndexSeq = [vocab.tokenToIndex.get(target_token, vocab.tokenToIndex["<UNK>"]) for target_token in targetSeq]

                    self.resetMemory(context="training")

                    loss, predictedTokenIndices, logitSeq, embeds, activations, windowWeights, longMem, shortMem, losses = self.trainStep(inputTokenIndices = inputTokenIndices, targetTokenIndexSeq = targetTokenIndexSeq)

                    """CALCULATE BASIC STATS"""
                    basicStats = self.getBasicStats(loss, logitSeq, self.latestMemGates, losses)
                    self.stats.update(basicStats)
                    self.stats_100.update(basicStats)

                    stepDuration = {"Step": time.time() - stepStartTime}
                    self.duration.update(stepDuration)
                    self.duration_100.update(stepDuration)

                    """SAVE THE MODEL EVERY x STEPS"""
                    if self.trainingStepCounter % saveModelFreq == 0:
                        print(f"{S_output.S_apply('dim', 'autosaving...')}{S_output.S_apply('reset', '')}")
                        babyLLM.saveModel()
                        success = f"autosave successful! saving every {saveModelFreq} steps, the next autosave will be at step {self.trainingStepCounter+saveModelFreq}..."
                        print(f"{S_output.S_apply('dim', success)}{S_output.S_apply('reset', '')}")

                    """PRINTING LOSS TO LOGS AND TERMINAL"""
                    #terminalPrintStartTime = time.time()
                    if self.trainingStepCounter == 1:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        runStart = f"\n--- {timestamp} ---\n{babyNote_loadCheckpointCheck}\n{userNote_loadCheckpoint}\n{babyNote_loadCheckpoint}{babyNote_runStart}\n{userNote_runStart}\n"
                        print(runStart)
                        with open(chatLogPath_forHumans, "a") as logFile: logFile.write(runStart)

                        trainingChatLine = f"\n--- {timestamp} --- {babyNote_loadCheckpointCheck} - {userNote_loadCheckpoint} - {babyNote_loadCheckpoint}{babyNote_runStart} - {userNote_runStart}\n"
                        with open(trainingLogPath_100, "a") as logFile: logFile.write(trainingChatLine)
                        with open(trainingLogPath_1000, "a") as logFile: logFile.write(trainingChatLine)
                        with open(chatLogPath_trainingLog, "a") as logFile: logFile.write(trainingChatLine)

                    # Track loss every 1000 steps
                    if self.trainingStepCounter % trainingLogFreq_1000 == 0:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                        complexStats = self.getComplexStats(embeds, activations, windowWeights, self.latestMemGates)
                        self.stats.update(complexStats)

                        trainingDataRemaining = len(trainingDataPairs) - self.trainingStepCounter
                        trainingDataPercent = (trainingDataRemaining / len(trainingDataPairs)) * 100
                        print(f"step {self.trainingStepCounter} | tokens remaining: {len(trainingDataPairs) - self.trainingStepCounter} ({trainingDataPercent:.2f}%)")

                        # eventually want this at the end only
                        self.duration.update(self.durationCategories) # Force a noop update to ensure we have every category
                        durationLog_1000 = "Durations: " + ", ".join([
                            f"{name}: {(duration * 1000 / trainingLogFreq_1000 if trainingLogFreq_1000 > 0 else 0):.2f}ms"
                            for name, duration in self.duration.most_common() # most_common with no parameter returns everything, already sorted in reverse
                        ])
                        self.duration.clear()
                        with open(durationLogPath_1000, "a") as logFile: logFile.write(durationLog_1000 + "\n")

                        stringStats = self.getStringStats(predictedTokenIndices, tokenCounts, tokenCounts_100, logFreq_100=False)

                        S_output.S_logTraining(
                            trainingLogPath = trainingLogPath_1000,
                            trainingStepCounter = self.trainingStepCounter,
                            freq = trainingLogFreq_1000,
                            stats = self.stats,
                            windowWeights_str = stringStats["windowWeights"],
                            topTokens_str = stringStats["topTokens"],
                            otherInfo_str = f"{stringStats['tokenPerfect']} | babyLLM.py {trainingLogFreq_1000}",
                        )
                        self.stats.clear()
                        
                        self.perfectTokenCount = 0
                        self.perfectTokenCount_100 = 0
                        self.totalTokenEvaluations = 0
                        self.totalTokenEvaluations_100 = 0
                        
                    # Track loss every 100 steps
                    if self.trainingStepCounter % trainingLogFreq_100 == 0:
                        complexStats = self.getComplexStats(embeds, activations, windowWeights, self.latestMemGates)
                        self.stats_100.update(complexStats)
                        
                        trainingDataRemaining = len(trainingDataPairs) - self.trainingStepCounter
                        trainingDataPercent = (trainingDataRemaining / len(trainingDataPairs)) * 100
                        print(f"step {self.trainingStepCounter} | tokens remaining: {len(trainingDataPairs) - self.trainingStepCounter} ({trainingDataPercent:.2f}%)")

                        self.duration_100.update(self.durationCategories) # Force a noop update to ensure we have every category
                        #durationLogBabyLLM_inner_100 = (f"inner step {j+1}: forward: {forwardTime*1000:.2f}ms | predict: {predictTime*1000:.2f}ms | loss: {lossTime*1000:.2f}ms")
                        #durationLogBabyLLM_100 = (f"DEBUG: forward() timings: Index: {idxTime*1000:.2f}ms | Embed: {embedTime*1000:.2f}ms | Neuron: {neuronTime*1000:.2f}ms | Memory: {memoryTime*1000:.2f}ms | Output: {outputTime*1000:.2f}ms | Total: {forwardTotal*1000:.2f}ms")
                        durationLog_100 = "Durations: " + ", ".join([
                            f"{name}: {(duration * 1000 / trainingLogFreq_100 if trainingLogFreq_100 > 0 else 0):.2f}ms"
                            for name, duration in self.duration_100.most_common()
                        ])
                        #durationLogCombined_100 = f"\n--- {timestamp} --- \n{durationLog_100} \n{durationLogBabyLLM_100} \n{durationLogBabyLLM_inner_100}\n"
                        with open(durationLogPath_100, "a") as logFile: logFile.write(durationLog_100 + "\n")
                        self.duration_100.clear()

                        stringStats = self.getStringStats(predictedTokenIndices, tokenCounts, tokenCounts_100, logFreq_100 = True)

                        S_output.S_logTraining(
                            trainingLogPath = trainingLogPath_100,
                            trainingStepCounter = self.trainingStepCounter,
                            freq = trainingLogFreq_100,
                            stats = self.stats_100,
                            windowWeights_str = stringStats["windowWeights"],
                            topTokens_str = stringStats["topTokens"],
                            otherInfo_str = f"{stringStats['tokenPerfect']} | babyLLM.py {trainingLogFreq_100}",
                        )
                        self.stats_100.clear()

                    """PRINTING GUESSES TO THE TERMINAL"""
                    if self.trainingStepCounter % printFreq == 0:
                        guessedTokenSeq = [self.getTokenIndexAsString(idx) if idx != -1 else "<UNK>" for idx in predictedTokenIndices]
                        if guessedTokenSeq: tokenCounts_100.update(guessedTokenSeq), tokenCounts.update(guessedTokenSeq)

                        S_output.S_colourPrintTraining(
                            step = self.trainingStepCounter,
                            inputSeq = inputSeq,
                            guessedSeq_str = guessedTokenSeq[:windowMAX],
                            targetSeq_str = targetSeq[:windowMAX],
                            loss = loss.item(),
                            recentLoss = sum(self.recentPrintLosses) / len(self.recentPrintLosses) if self.recentPrintLosses else None,
                            totalLoss = loss.item(),
                            totalTokenCount = self.stats["tokenCount"]
                        )

                    #terminalPrintDuration = {"Print": time.time() - terminalPrintStartTime}
                    #self.duration.update(terminalPrintDuration)
                    #self.duration_100.update(terminalPrintDuration)
                    #self.totalTerminalPrintDuration += terminalPrintDuration
                    self.trainingStepCounter += 1
                    """END OF ONE TURN"""    

                self.saveModel()

            except KeyboardInterrupt:
                choice = input("save, cancel or interact?" + f"\n{userName}: ").lower()
                if choice == "save" or choice == (""): babyLLM.saveModel(), print("\nit's rude to interrupt people.. but, bye bye! :)")
                elif choice == "cancel": babyLLM.saveModel(), print("\nhey! i wanted to remember that! :(")
                elif choice == "interact":
                    babyLLM.saveModel()
                    import code
                    print("try:\nbabyLLM.stats\nbabyLLM.scheduledSamplingProb\nbabyLLM.memoryLayer.memory\nbabyLLM.parallelNeuronLayer.windowWeighting\nbabyLLM.outputLayer.forward(...)\nUse `exit()` to return to terminal.\n")
                    vars = globals().copy()
                    vars.update(locals())
                    code.interact(local=vars)
                else: babyLLM.saveModel(), print("\nwait, what did you say? i didn't quite hear you.. but, bye bye! :)")
                
                sys.exit(8)

        print("--- Training Completed! ---")
        
    """saves the model to a file"""    
    def saveModel(self, filePath = modelFilePath):
        saveStartTime = time.time()

        tmpPath = filePath + ".tmp"
        torch.save(self.state_dict(), tmpPath)
        print(f"model temp file created at {tmpPath}")
        os.replace(tmpPath, filePath)
        print(f"model successfully saved to {filePath}!")
        with open(stepCheckpointFilePath, "w") as f:
            f.write(str(self.trainingStepCounter+self.startIndex))

        saveDuration = {"Save": time.time() - saveStartTime}
        self.duration.update(saveDuration)
        self.duration_100.update(saveDuration)

    """loads the model from a file"""
    def loadModel(self, filePath = modelFilePath):
        if durationLogging: loadStartTime = time.time()
        try:
            print(f"loading model from path: {filePath}") 
            self.load_state_dict(torch.load(filePath), strict = saveLock)
            print(f"model loaded from {filePath}!")
            self.to(modelDevice)
            print(f"device set to {modelDevice}!")
            self.resetMemory(context="inference")
            
            if durationLogging: loadDuration = {"Load": time.time() - loadStartTime}, self.duration.update(loadDuration), self.duration_100.update(loadDuration)
        except FileNotFoundError: print("No saved model found.")

    def babyllm_diary_entry(parallelNeuronLayer, step):
        # Grab current window weightings
        weights = parallelNeuronLayer.windowWeighting.detach().cpu().numpy()
        windows = parallelNeuronLayer.allWindowSizes

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
        """Reset memory depending on the context: inference always resets, training resets every n turns"""
        if context == "inference": self.memoryLayer.resetMemory(), print(f"resetting memory for new conversation...")
        elif context == "training":
            if hasattr(self, "stepsSinceMemoryReset"): self.stepsSinceMemoryReset += 1
            else: 
                self.stepsSinceMemoryReset = 1
            if self.stepsSinceMemoryReset >= memoryLength: 
                self.memoryLayer.resetMemory()
                self.stepsSinceMemoryReset = 0 
                print(f"resetting memory after {memoryLength} steps...")

    
if __name__ == "__main__":

    embedDimension = embedDimension
    numNeurons = numNeurons
    activationFunction = activationFunction
    startIndex = trainingStartIndex  # default
    vocab = VOCAB(vocabSize = vocabSize)
    babyLLM = BABYLLM(vocab = vocab, embedDimension = embedDimension, numNeurons = numNeurons, activationFunction = activationFunction)
    babyLLM.loadModel()

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
    choice = input(babyNote_loadCheckpointCheck + f"\n{userName}: ").lower()

    if choice == "" or choice.startswith("y"):
        startIndex = savedStep
        babyNote_loadCheckpoint = f"{babyName} 'ok! let's go to step {savedStep}! "
        print(babyNote_loadCheckpoint, end="")

    elif choice.startswith("r") or choice in ["random", "i dont care", "i don't care", "idc"]:
        startIndex = random.randint(0, len(vocab.tokens) - windowMAX - 1)
        babyNote_loadCheckpoint = f"{babyName} 'oh, cool! i'll pick a random spot to start from... umm... let's go to step {startIndex}! "
        print(babyNote_loadCheckpoint, end="")

    elif choice.startswith("n") or choice in ["start again", "restart"]:
        startIndex = trainingStartIndex
        babyNote_loadCheckpoint = f"{babyName} 'alright, step {startIndex}, let's go back to the beginning :) "
        print(babyNote_loadCheckpoint, end="")
        
    elif choice.isdigit():
        startIndex = int(choice)
        babyNote_loadCheckpoint = f"{babyName} 'damn that's specific! heading to step {startIndex}... "
        print(babyNote_loadCheckpoint, end="")

    else:
        startIndex = trainingStartIndex
        babyNote_loadCheckpoint = f"{babyName} 'umm... i don't think i heard you properly, i'll just start from step {startIndex} :) but, "
        print(babyNote_loadCheckpoint, end="")

    babyNote_runStart = f"what am i learning today?'" # no tag of 'babyllm:' because it merges with the end of above message in logs
    userNote_runStart = f"{userName}: '" + input(babyNote_runStart + f"\n{userName}: ").strip().lower() + "'"
    userNote_loadCheckpoint = f"{userName}: '{choice}'"

    #TESTinputSeq = ["what","will","you","do","out","there","now","?"]
    TESTinputSeq = ["i","love","you","this","is","good","music","is","life",]
    #TESTinputSeq = ["what"] 

    trainingDataPairs = vocab.genTrainingData(windowMAX, startIndex = startIndex)
    print(f"Total trainingDataPairs: {len(trainingDataPairs)}")
    babyLLM.to(modelDevice)
    babyLLM.trainModel(trainingDataPairs, epochs = epochs, startIndex = startIndex)
