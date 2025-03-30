# CHARIS CAT 2025

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim 
from vocab import VOCAB
from embedLayer import EMBEDLAYER
from parallelNeuronLayer import PARALLELNEURONLAYER
from outputLayer import OUTPUTLAYER
from neuron import NEURON
from tinyAttentionLayer import TINYATTENTIONLAYER
from memoryLayer import MEMORYLAYER
from config import *
from datetime import datetime
import random
from torch.profiler import profile, record_function, ProfilerActivity
import os
import outputStyles
import time
from collections import Counter
from trainingHUD import *
import sys
import shutil  

"""this class combines all the core components of the babyLLM:"""
"""EMBEDLAYER: token embedding layer"""
"""PARALLELNEURONLAYER: layer of parallel neurons for feature extraction"""
"""OUTPUTLAYER: output layer to generate logits"""
"""MULTIWINDOWLAYER: (New) layer to incorporate multi-window context"""
"""it also manages training, loss computation, backpropagation, and response generation."""

class BABYLLM(nn.Module):
    def __init__(self, vocab, embedDimension, numNeurons, activationFunction):
        super().__init__()
        #os.system('clear')
        """CONFIG"""
        self.vocabSize = vocabSize
        self.vocab = vocab
        self.embedDimension = embedDimension
        self.numNeurons = numNeurons
        self.learningRate = learningRate
        self.temperature = temperature
        self.activationFunction = activationFunction
        optimizerClass = getattr(optim, optimizerName)
        self.guessHUD = rainbowHUD(maxArms=60)

        """LAYERS"""
        self.embedLayer = EMBEDLAYER(vocabSize, self.embedDimension)
        self.parallelNeuronLayer = PARALLELNEURONLAYER(numNeurons = self.numNeurons, embedDimension = self.embedDimension, activationFunction = self.activationFunction)
        self.outputLayer = OUTPUTLAYER(numNeurons = self.numNeurons, vocabSize = self.vocabSize)
        self.memoryLayer = MEMORYLAYER(numNeurons = self.numNeurons)
        #self.multiWindowLayer = MULTIWINDOWLAYER(embedDimension = self.embedDimension, windowSizes = [window1, window2, window3])

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

        self.totalLoss = 0
        self.totalLossDetail = 0
        self.totalTokenCount = 0 
        self.totalTokenCountDetail = 0 
        self.totalLogitMin = 0 
        self.totalLogitMax = 0
        self.totalLogitMinDetail = 0 
        self.totalLogitMaxDetail = 0 
        self.scheduledSamplingProb = 0.0
        self.perfectTokenCount = 0
        self.totalTokenEvaluations = 0

        self.totalStepDuration = 0
        self.totalStepDurationDetail = 0
        self.totalSaveDuration = 0
        self.totalSaveDurationDetail = 0
        self.totalLoadDuration = 0
        self.totalLoadDurationDetail = 0
        self.totalLogitsDuration = 0
        self.totalLogitsDurationDetail = 0 
        self.totalCombineDuration = 0
        self.totalCombineDurationDetail = 0 
        self.totalGetTokenDuration = 0
        self.totalGetTokenDurationDetail = 0 
        self.totalTerminalPrintDuration = 0
        self.totalTerminalPrintDurationDetail = 0

        self.hud_height = 5
        self.term_height = shutil.get_terminal_size().lines
        self.hud_start_line = self.term_height - self.hud_height + 1

        #def HUD_fixScroll(self):
        #    sys.stdout.write("\033[999B")  # Move cursor down a lot
        #    sys.stdout.flush()

        """def HUD_fixScroll(self):
            sys.stdout.write("\033[999B")  # Move cursor to bottom
            sys.stdout.flush()
            printHUD(
                windowWeights=(self.parallelNeuronLayer.windowWeighting + 0.1).detach().cpu().numpy(),
                guessHUD=self.guessHUD
            )
            sys.stdout.write("\033[6A")  # Move cursor 6 lines up to hover above HUD
            sys.stdout.flush()"""

    """def HUD_fixScroll(self):
        # Move cursor to top-left
        sys.stdout.write("\033[H")
        sys.stdout.flush()

        # Clear HUD area (overwrite with spaces)
        for _ in range(self.hud_height):
            sys.stdout.write("\033[K") # Clear line
            sys.stdout.write("\n")     # Move to next line

        # Move cursor back to top-left to redraw HUD
        sys.stdout.write("\033[H")
        sys.stdout.flush()

        printHUD(
            windowWeights=(self.parallelNeuronLayer.windowWeighting + 0.1).detach().cpu().numpy(),
            guessHUD=self.guessHUD
        )"""

    def forward(self, inputSeq):
        """processes input sequence of tokens (str) to generate logits to predict the next token"""
        #print(f"Debug: Input to forward: {inputSeq}")

        """convert inputted tokens to indices (batch processing instead of looping)"""
        inputIndices = [self.vocab.tokenToIndex.get(tokenString, self.vocab.tokenToIndex["<UNK>"]) if not isinstance(tokenString, int) else tokenString for tokenString in inputSeq]
        #print(f"Debug BABYLLM.forward: inputIndices: {inputIndices}")

        """convert indices to embeddings"""
        inputEmbeds = []
        inputIndicesTensor = torch.tensor(inputIndices)
        inputEmbedsBatch = self.embedLayer(inputIndicesTensor)
        inputEmbeds = [embed for embed in inputEmbedsBatch]     # list of (embed_dim,) tensors

        """DEBUG PRINTS"""
        if len(inputEmbeds) > 0: # Check if inputEmbeds is not empty
            #print(f"Debug BABYLLM.forward: Type of first element in inputEmbeds: {type(inputEmbeds[0])}")
            #print(f"Debug BABYLLM.forward: Shape of first element in inputEmbeds: {inputEmbeds[0].shape}")
            #print(f"Debug BABYLLM.forward: Shapes of first 5 elements in inputEmbeds: {[embed.shape for embed in inputEmbeds[:min(5, len(inputEmbeds))] ]}... (first 5)")
            pass
        else:
            #print(f"Debug BABYLLM.forward: inputEmbeds list is EMPTY!")
            pass

        """make sure inputEmbeds is a LIST of tensors"""
        if not isinstance(inputEmbeds, list):
            inputEmbeds = [inputEmbeds]

        """PARALLEL NEURON LAYER input/processing (feature extraction)"""
        parallelNeuronOutput = self.parallelNeuronLayer.forward(inputEmbeds) 
        #print(f"Debug BABYLLM.forward: parallelNeuronOutput length: {len(parallelNeuronOutput)}") 

        """RESIZE NEURON LAYER TO STANDARD SIZE FOR COMBINED FORWARD PROCESSING"""
        combinedActivationsTensor = torch.mean(parallelNeuronOutput, dim=0, keepdim=True)

        """MEMORY LAYER PROCESSING - NOW PROCESS THE COMBINED ACTIVATIONS"""
        memoryLayerOutput = self.memoryLayer.forward(combinedActivationsTensor)
        self.latestMemGates = self.memoryLayer.latestMemoryGates.detach() 
        combinedActivations = memoryLayerOutput
        #print(f"Debug BABYLLM: Shape of lastTokenActivations BEFORE outputLayer: {lastTokenActivations.shape}")
        logits = self.outputLayer.forward(combinedActivations)  
        #print(f"Debug BABYLLM.forward: probabilityDist shape: {probabilityDist.shape}")
        """returns a logits tensor of shape (1, vocabSize) showing predicted probabilities for the next token"""
        return logits
    
    """computes the cross-entropy loss between the models logits and the target token, essentially checking how good the models prediction was"""
    def computeLoss(self, logits, targetTokenIndex):
        targetTensor = torch.tensor([targetTokenIndex], dtype=torch.long)
        #print(f"Debug BABYLLM.computeLoss: predictions shape: {logits.shape}")
        #print(f"Debug BABYLLM.computeLoss: predictions (first 10): {logits[:10]}")
        #print(f"Debug BABYLLM.computeLoss: targetTokenIndex: {targetTokenIndex}")
        #print(f"Debug BABYLLM.computeLoss: self.target: {self.target}")
        """Handle cases where logits might be 1D (unsqueeze to make it 2D for cross_entropy)"""
        """SUS!!"""
        if logits.dim() == 1: 
            logits = logits.unsqueeze(0) 

        self.loss = F.cross_entropy(logits, targetTensor) 
        #print(f"Debug BABYLLM.computeLoss: Loss value: {self.loss.item():.4f}")
        """returns a scalar tensor representing the cross-entropy loss value"""
        return self.loss
    
    """backpropagation and optimization, computes gradients of the loss and uses the optimizer to update the models weights"""
    def backward(self, loss):
        self.optimizer.zero_grad()  # Reset gradients
        loss.backward()
        self.optimizer.step()  # Update weights

    """this iterates through training data, performing forward passes, loss computation, backpropagation, and optimization for each step."""
    def trainModel(self, trainingDataPairs, epochs):
        #print(f"Debug tokenToIndex (First 20): {list(vocab.tokenToIndex.items())[:20]}")
        numTokens = numTokensPerStep
        if isinstance(numTokens, torch.Tensor):
            numTokens = numTokens.item()
        numTokens = int(numTokens)

        totalMemGatesDetail = 0
        totalMemGates = 0
        avgLossDetail = 0
        avgLoss = 0
        avgGradNorm = 0
        avgGradNormDetail = 0
        
        #os.system('clear')
        print("babyLLM is heading back to school...")

        """EPOCH LOOP"""
        trainingStepCounter = 1
        for epoch in range(epochs):
            print(f"--- Epoch {epoch+1}/{epochs} Started ---")

            """TRAINING DATA (batches)"""
            try:
                for i, (inputSeq, targetSeq) in enumerate(trainingDataPairs):
                    logitRange_str = ""        
                    logitRangeDetail_str = ""
                    stepStartTime = time.time()
                    inputTokenIndices = [vocab.tokenToIndex.get(token, vocab.tokenToIndex["<UNK>"]) for token in inputSeq]
                    targetTokenIndexSeq = [vocab.tokenToIndex.get(target_token, vocab.tokenToIndex["<UNK>"]) for target_token in targetSeq]

                    if targetSeq: # Check if targetSeq is not empty
                        target = targetSeq[0] # Get the FIRST token of targetSeq
                        targetTokenIndex = vocab.tokenToIndex.get(target, vocab.tokenToIndex["<UNK>"])
                    else:
                        targetTokenIndex = vocab.tokenToIndex["<UNK>"] # Default to <UNK> if no target

                    trainingStepCounter += 1
                    guessedTokenSeq = []

                    """MULTI-TOKEN TRAINING STEP"""
                    self.optimizer.zero_grad() # reset gradients from previous step
                    predictedTokenIndices = []
                    guessedTokenSeq = []
                    losses = []
                    cumulativeLoss = 0.0 # Sum of losses for THIS sequence

                    inputSeqPredictions = list(inputTokenIndices) # Start with input context, create a COPY!
                    logitSeq = [] # Store logits for each prediction step

                    # Predict multiple tokens in a sequence
                    for j in range(numTokens):
                        logits = self.forward(inputSeqPredictions) # Feed current input sequence
                        logitSeq.append(logits) # Store logits
                        predictedTokenIndex = self.getResponseFromLogits(logits) # Get predicted token
                        predictedTokenIndices.append(predictedTokenIndex) # Store predicted token index

                        # Decide whether to use teacher forcing or model prediction for next input
                        if False: #scheduledSampling and random.random() < scheduledSamplingProb:
                            # Use model's prediction (scheduled sampling)
                            nextTokenInput = predictedTokenIndex
                        else:
                            # Use teacher forcing (ground truth)
                            if j < len(targetTokenIndexSeq): # Check if target exists for this step
                                nextTokenInput = targetTokenIndexSeq[j] # Use ground truth target token
                            else:
                                nextTokenInput = predictedTokenIndex # if no more targets, use prediction

                        inputSeqPredictions.append(nextTokenInput) # Append token index (predicted or ground truth) as next input
                        if j < len(targetTokenIndexSeq): # compute loss only if target exists
                            self.totalTokenEvaluations += 1
                            if predictedTokenIndex == targetTokenIndexSeq[j]:
                                self.perfectTokenCount += 1
                            stepLoss = self.computeLoss(logits, targetTokenIndexSeq[j]) # calculate loss for this step against target
                            losses.append(stepLoss) # append loss
                            cumulativeLoss += stepLoss # Cumulative loss (sum)

                    # Average loss over the sequence of predicted tokens
                    if losses:
                        loss = cumulativeLoss / len(losses) # Average loss for THIS SEQUENCE (per token)
                    else:
                        loss = torch.tensor(0.0)

                    loss.backward() # Backpropagate the averaged loss
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = gradientClipMaxNorm)
                    self.optimizer.step()

                    self.totalLoss += cumulativeLoss.item() # Accumulate SUM of token losses
                    self.totalLossDetail += cumulativeLoss.item() # Accumulate SUM of token losses
                    self.totalTokenCount += len(losses) # Count tokens processed
                    self.totalTokenCountDetail += len(losses) # Count tokens processed

                    with torch.no_grad():
                        logitMin = logits.min(dim=-1).values.mean().item()
                        logitMax = logits.max(dim=-1).values.mean().item()

                        self.totalLogitMin += logitMin
                        self.totalLogitMax += logitMax
                        self.totalLogitMinDetail += logitMin
                        self.totalLogitMaxDetail += logitMax
                        memGatesTensor = self.latestMemGates
                        if memGatesTensor is not None:
                            memoryGates_str = f"Short:{memGatesTensor[0]:.3f}, Long:{memGatesTensor[1]:.3f}, Current:{memGatesTensor[2]:.3f}"
                            totalMemGatesDetail += memGatesTensor.mean().item()
                            totalMemGates += memGatesTensor.mean().item()
                        else:
                            memoryGates_str = "N/A"

                    firstPredictedTokenIndex = predictedTokenIndices[0] if predictedTokenIndices else -1 # Get the first predicted token index for display
                    guessedTokenIndex = firstPredictedTokenIndex # for single token display
                    stepEndTime = time.time()
                    stepDuration = stepEndTime - stepStartTime
                    self.totalStepDuration += stepDuration
                    self.totalStepDurationDetail += stepDuration
                    guessedTokenSeq = []

                    """SAVE THE MODEL EVERY x STEPS"""
                    if trainingStepCounter % saveModelFreq == 0:
                        print(f"{outputStyles.S_apply('dim', "autosaving...")}{outputStyles.S_apply('reset', "")}")
                        babyLLM.saveModel()
                        success = f"autosave successful! saving every {saveModelFreq} steps, the next autosave will be at step {trainingStepCounter+saveModelFreq}..."
                        print(f"{outputStyles.S_apply('dim', success)}{outputStyles.S_apply('reset', "")}")

                    """PRINTING LOSS TO LOGS AND TERMINAL"""
                    terminalPrintStartTime = time.time()
                    if trainingStepCounter == 0:
                        userNote = input("what am i learning today?: ").strip()
                        # scheduledSamplingProb += scheduledSamplingProbIncrement
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get timestamp
                        runStart = f"\n--- {timestamp} ---"
                        runStart += f"\nbabyLLM: what am i learning today?"
                        runStart += f"\nYou: {userNote}\n"
                        print(f"{runStart.strip()}")
                        with open("trainingLogDetail.txt", "a") as logFile:
                            logFile.write(runStart)
                        with open("trainingLog.txt", "a") as logFile:
                            logFile.write(runStart)

                    # Track loss every 1000 steps
                    if trainingStepCounter % printLossFreq == 0:
                        tokenCounts = Counter() # Initialize Counter HERE for each interval
                        if guessedTokenSeq:
                            tokenCounts.update(guessedTokenSeq)
                        topTokens = tokenCounts.most_common(10) # Get top 10 tokens for this interval
                        avgStepDuration = self.totalStepDuration / printLossFreq if printLossFreq > 0 else 0
                        avgSaveDuration = self.totalSaveDuration / printLossFreq if printLossFreq > 0 else 0
                        avgLoadDuration = self.totalLoadDuration / printLossFreq if printLossFreq > 0 else 0
                        avgTerminalPrintDuration = self.totalTerminalPrintDuration / printLossFreq if printLossFreq > 0 else 0
                        avgLogitsDuration = self.totalLogitsDuration / printLossFreq if printLossFreq > 0 else 0
                        avgCombineDuration = self.totalCombineDuration / printLossFreq if printLossFreq > 0 else 0
                        avgGetTokenDuration = self.totalGetTokenDuration / printLossFreq if printLossFreq > 0 else 0
                        avgLoss = self.totalLoss / printLossFreq # True average loss per token
                        avgGradNorm = totalGradNorm / printLossFreq #?????
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get timestamp
                        avgLogitMin = self.totalLogitMin / printLossFreq
                        avgLogitMax = self.totalLogitMax / printLossFreq
                        with torch.no_grad():
                            normWeights = (babyLLM.parallelNeuronLayer.windowWeighting + 0.1)
                            normWeights /= (normWeights.sum() + 0.1)
                            sortedWeights = sorted(
                                zip(allWindowSizes, normWeights.cpu().numpy()),
                                key=lambda x: x[1],
                                reverse=True
                            )
                            weight_str = ",".join(f"W{wsize}:{weight:.5f}" for wsize, weight in sortedWeights)

                        avgDurations = [
                                ("Step", avgStepDuration),
                                ("Save", avgSaveDuration),
                                ("Load", avgLoadDuration),
                                ("Print", avgTerminalPrintDuration),
                                ("Logits", avgLogitsDuration),
                                ("Combine", avgCombineDuration),
                                ("Token", avgGetTokenDuration),
                            ]
                        
                        sortedAvgDurations = sorted(avgDurations, key=lambda item: item[1], reverse=True)
                        durationLog_str = "Durations: "
                        for name, duration in sortedAvgDurations:
                            durationLog_str += f"{name}: {duration*1000:.2f}ms, "
                        durationLog = durationLog_str.rstrip(', ')

                        outputStyles.logTraining(
                            logFilePath=logFilePath,
                            step=i + 1,
                            avgLoss=avgLoss,
                            learningRate=learningRate,
                            logitRange_str = f"{avgLogitMin:.2f}, {avgLogitMax:.2f}",
                            windowWeights_str=weight_str,
                            memoryGates_str=memoryGates_str,
                            gradientNorm_str=f"{avgGradNorm:.3f}",
                            otherInfo="babyLLM.py training",
                            topTokens_str=str(topTokens),
                            durationLog_str=durationLog
                        )

                        if self.totalTokenEvaluations > 0:
                            tokenPerfectRate = (self.perfectTokenCount / self.totalTokenEvaluations) * 100
                            print(f"{outputStyles.S_apply('perfect', f'Token Perfect: {self.perfectTokenCount} / {self.totalTokenEvaluations}')} → {tokenPerfectRate:.2f}%")
                        
                        self.perfectTokenCount = 0
                        self.totalTokenEvaluations = 0

                        #print(durationLog)

                        with open("durationLog.txt", "a") as logFile:
                            logFile.write(durationLog + "\n")
                        
                        self.totalStepDuration = 0
                        self.totalSaveDuration = 0
                        self.totalLoadDuration = 0
                        self.totalTerminalPrintDuration = 0
                        self.totalLogitsDuration = 0
                        self.totalCombineDuration = 0
                        self.totalGetTokenDuration = 0
                        self.totalLogitMin = 0
                        self.totalLogitMax = 0
                        self.totalLoss = 0 # Reset SUM of losses
                        self.totalTokenCount = 0 # Reset token count

                    # Track loss every 100 steps
                    if trainingStepCounter % printLossFreqDetail == 0:
                        tokenCountsDetail = Counter()
                        if guessedTokenSeq:
                            tokenCountsDetail.update(guessedTokenSeq)
                        topTokensDetail = tokenCountsDetail.most_common(10)
                        avgStepDurationDetail = self.totalStepDurationDetail / printLossFreqDetail if printLossFreqDetail > 0 else 0
                        avgSaveDurationDetail = self.totalSaveDurationDetail / printLossFreqDetail if printLossFreqDetail > 0 else 0
                        avgLoadDurationDetail = self.totalLoadDurationDetail / printLossFreqDetail if printLossFreqDetail > 0 else 0
                        avgTerminalPrintDurationDetail = self.totalTerminalPrintDurationDetail / printLossFreqDetail if printLossFreqDetail > 0 else 0
                        avgLogitsDurationDetail = self.totalLogitsDurationDetail / printLossFreqDetail if printLossFreqDetail > 0 else 0
                        avgCombineDurationDetail = self.totalCombineDurationDetail / printLossFreqDetail if printLossFreqDetail > 0 else 0
                        avgGetTokenDurationDetail = self.totalGetTokenDurationDetail / printLossFreqDetail if printLossFreqDetail > 0 else 0
                        avgGradNormDetail = totalGradNormDetail / printLossFreqDetail #?????
                        avgLossDetail = self.totalLossDetail / printLossFreqDetail
                        timestampDetail = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        avgLogitMinDetail = self.totalLogitMinDetail / printLossFreqDetail
                        avgLogitMaxDetail = self.totalLogitMaxDetail / printLossFreqDetail
                        with torch.no_grad():
                            normWeightsDetail = (babyLLM.parallelNeuronLayer.windowWeighting + 0.1)
                            normWeightsDetail /= (normWeightsDetail.sum() + 0.1)
                            sortedWeightsDetail = sorted(
                                zip(allWindowSizes, normWeightsDetail.cpu().numpy()),
                                key=lambda x: x[1],
                                reverse=True
                            )
                            weightDetail_str = ",".join(f"W{wsize}:{weight:.5f}" for wsize, weight in sortedWeightsDetail)

                        avgDurationsDetail = [
                                ("Step", avgStepDurationDetail),
                                ("Save", avgSaveDurationDetail),
                                ("Load", avgLoadDurationDetail),
                                ("Print", avgTerminalPrintDurationDetail),
                                ("Logits", avgLogitsDurationDetail),
                                ("Combine", avgCombineDurationDetail),
                                ("Token", avgGetTokenDurationDetail),
                            ]
                        
                        sortedAvgDurationsDetail = sorted(avgDurationsDetail, key=lambda item: item[1], reverse=True)

                        durationLogDetail_str = "Durations: "
                        for name, duration in sortedAvgDurationsDetail:
                            durationLogDetail_str += f"{name}: {duration*1000:.2f}ms, "

                        durationLogDetail = durationLogDetail_str.rstrip(', ')

                        logitRangeDetail_str = f"{avgLogitMinDetail:.2f},{avgLogitMaxDetail:.2f}"
                        #print(f"DEBUG: logitRange_str before logTraining: '{logitRangeDetail_str}'")
                        outputStyles.logTraining(
                            logFilePath=logFilePath,
                            step=i + 1,
                            avgLoss=avgLossDetail,
                            learningRate=self.learningRate,
                            logitRange_str=logitRangeDetail_str,
                            windowWeights_str=weightDetail_str,
                            memoryGates_str=memoryGates_str,
                            gradientNorm_str=f"{avgGradNormDetail:.3f}",
                            otherInfo="Training",
                            topTokens_str=str(topTokensDetail),
                            durationLog_str=durationLogDetail
                        )

                        #print(durationLogDetail) # Print duration log to terminal
                        with open("durationLogDetail.txt", "a") as logFile:
                            logFile.write(durationLogDetail + "\n")

                        self.totalStepDurationDetail = 0
                        self.totalSaveDurationDetail = 0
                        self.totalLoadDurationDetail = 0
                        self.totalTerminalPrintDurationDetail = 0
                        self.totalLogitsDurationDetail = 0
                        self.totalCombineDurationDetail = 0
                        self.totalGetTokenDurationDetail = 0
                        self.totalLogitMinDetail = 0
                        self.totalLogitMaxDetail = 0
                        self.totalLossDetail = 0 # Reset SUM of losses
                        self.totalTokenCountDetail = 0 # Reset token count

                    """PRINTING GUESSES TO THE TERMINAL"""
                    if trainingStepCounter % printFreq == 0:
                        targetWordSingle = targetSeq[0].replace("Ġ", " ") if targetSeq else "<NO_TARGET>"
                        guessedTokenString = self.getTokenIndexAsString(guessedTokenIndex).replace("Ġ", " ")
                        targetWordSeq = targetSeq[0].replace("Ġ", " ") if targetSeq else "<NO_TARGET>"
                        guessedTokenSeq = [self.getTokenIndexAsString(idx).replace("Ġ", " ") if idx != -1 else "<NO_GUESS>" for idx in predictedTokenIndices]
                        S_arm = []
                        """for i in range(min(3, len(predictedTokenIndices))):
                            guess = self.getTokenIndexAsString(predictedTokenIndices[i])
                            target = targetSeq[i] if i < len(targetSeq) else ""
                            if guess == target:
                                S_arm.append(S_apply("perfect", guess))
                            else:
                                loss_val = losses[i].item() if i < len(losses) else 999.0
                                S_type = S_getStat("loss", loss_val)
                                S_arm.append(S_apply(S_type, guess))"""
                        for i in range(min(3, len(predictedTokenIndices))):
                            lossVal = losses[i].item() if i < len(losses) else 999.0
                            S_type = S_getStat("loss", lossVal)
                            block = S_apply(S_type, "█")  # lil block boi!
                            S_arm.append(block)

                        self.guessHUD.addArm(S_arm)
                        
                        # Get the SEQUENCE of target token strings (for multi-token display)
                        targetSeq = [tok.replace("Ġ", " ") for tok in targetSeq]
                        inputSentenceClean = "".join(inputSeq).replace("Ġ", " ")
                        guessedSeqStr = "".join(guessedTokenSeq[:windowMAX]).lstrip()
                        targetSeqStr = "".join(targetSeq[:windowMAX]).lstrip()

                        isCorrect = guessedSeqStr.strip() == targetSeqStr.strip()

                        isPerfect = isCorrect and loss.item() < 0.01 # Using loss.item() from last loss calculation
                        #print(f"DEBUG: logitRange_str before logTraining: '{logitRange_str}'")
                        outputStyles.colourPrintTraining(
                            step=trainingStepCounter,
                            inputSentence=inputSentenceClean,
                            guessedSeqStr=guessedSeqStr,
                            targetSeqStr=targetSeqStr,
                            loss=loss.item(),
                            isCorrect=isCorrect,
                            isPerfect=isPerfect
                        )
                    
                        terminalPrintEndTime = time.time()
                        terminalPrintDuration = terminalPrintEndTime - terminalPrintStartTime
                        self.totalTerminalPrintDuration += terminalPrintDuration

                        #self.HUD_fixScroll()

                print(f"Epoch {epoch+1}/{epochs} - Loss: {totalLoss / self.totalTokenCount:.4f}") # Final epoch avg loss per token
                #torch.save(self.state_dict(), f"babyLLM_epoch{epoch}.pth")
                self.scheduledSamplingProb = min(self.scheduledSamplingProb + scheduledSamplingProbIncrement, 1.0)
                self.saveModel()

            except KeyboardInterrupt:
                print("\nit's rude to interrupt people.. but, bye bye! :)")
                babyLLM.saveModel()
                break

        print("--- Training Completed! ---")
        
    """saves the model to a file"""    
    def saveModel(self, filePath="babyLLM.pth"):
        saveStartTime = time.time()

        tmpPath = filePath + ".tmp"
        torch.save(self.state_dict(), tmpPath)
        print(f"Model temp file created at {tmpPath}")
        os.replace(tmpPath, filePath)
        print(f"✅ Model successfully saved to {filePath}!")

        saveEndTime = time.time()
        saveDuration = saveEndTime - saveStartTime
        self.totalSaveDuration += saveDuration
        self.totalSaveDurationDetail += saveDuration

    """loads the model from a file"""
    def loadModel(self, filePath = modelPath):
        loadStartTime = time.time()
        try:
            print(f"Loading model from path: {filePath}") 
            self.load_state_dict(torch.load(filePath), strict = saveLock)
            print(f"Model loaded from {filePath}!")
            loadEndTime = time.time()
            loadDuration = loadEndTime - loadStartTime
            self.totalLoadDuration += loadDuration
            self.totalLoadDurationDetail += loadDuration
        except FileNotFoundError:
            print("No saved model found.")


    """this takes the output logits, does temperature scaling and softmax to create a probability distribution over the vocab, 
    and then selects most likely response token"""
    def getResponseFromLogits(self, logits, temperature=None):
        logitsStartTime = time.time()

        if temperature is None:
            temperature = self.temperature
        logits = logits / temperature
        softmaxed = torch.softmax(logits, dim=1)

        topValue, topIndex = torch.max(softmaxed, dim=1)
        guessedTokenIndex = topIndex.item()

        logitsEndTime = time.time()
        logitsDuration = logitsEndTime - logitsStartTime
        self.totalLogitsDuration += logitsDuration
        self.totalLogitsDurationDetail += logitsDuration
        return guessedTokenIndex
    
    """convert token index to string"""
    def getTokenIndexAsString(self, tokenIndex):
        """returns the guessed token as a readable string aka text"""
        #return self.vocab.indexToToken[tokenIndex.__str__()] 
        return self.vocab.indexToToken[int(tokenIndex)]
    
    """generates the chosen next token using getResponseFromLogits"""
    def getNextToken(self, inputSeq, temperature=None):  
        getTokenStartTime = time.time()

        if temperature is None:
            temperature = self.temperature  # Grab from self.temperature (config)
        """returns an integer token index showing the models predicted next token."""
        
        getTokenEndTime = time.time()
        getTokenDuration = getTokenEndTime - getTokenStartTime
        self.totalGetTokenDuration += getTokenDuration
        self.totalGetTokenDurationDetail += getTokenDuration
        return self.getResponseFromLogits(self.forward(inputSeq), temperature)
    
    """combines the parallelNeronLayer output and the multiWindowLayer output into one output"""
    def combineOutputs(self, output1, output2):
        combineStartTime = time.time()

        #print(f"Debug combineOutputs: Shape of output1: {output1.shape}")
        #print(f"Debug combineOutputs: Shape of output2: {output2.shape}")
        output1Flat = output1.squeeze(dim=2) # Remove dimension of shape 1, new shape: [1, 10000]
        #print(f"Debug combineOutputs: Shape of output1Flat: {output1Flat.shape}")
        concatenatedOutput = torch.cat((output1Flat, output2), dim=1) # Concatenate along dim=1 (feature dimension) - 2D tensors
        """linear layer to combine and reduce dimensionality"""
        if not hasattr(self, 'outputCombinationLayer'):
            combined_dim = output1Flat.shape[1] + output2.shape[1] # dim=1 is the feature dimension
            self.outputCombinationLayer = nn.Linear(combined_dim, embedDimension) # Output dimension should be embedDimension
        finalOutput = self.outputCombinationLayer(concatenatedOutput)
        """returns a single combined output tensor of shape (1, embedDimension)."""
        combineEndTime = time.time()
        combineDuration = combineEndTime - combineStartTime
        self.totalCombineDuration += combineDuration
        self.totalCombineDurationDetail += combineDuration
        return finalOutput

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

# Example usage in training loop:
# if step % 1000 == 0:
#     babyllm_diary_entry(parallel_neuron_layer, step)

    
if __name__ == "__main__":
    vocab = VOCAB(vocabSize = vocabSize)
    embedDimension = embedDimension
    numNeurons = numNeurons
    activationFunction = activationFunction

    babyLLM = BABYLLM(vocab = vocab, embedDimension = embedDimension, numNeurons = numNeurons, activationFunction = activationFunction)
    #TESTinputSeq = ["what","will","you","do","out","there","now","?"]
    TESTinputSeq = ["i","love","you","this","is","good","music","is","life",]
    #TESTinputSeq = ["what"] 

    babyLLM.loadModel()
    trainingDataPairs = vocab.genTrainingData(windowMAX)
    babyLLM.trainModel(trainingDataPairs, epochs = epochs)
    

    print("--- BabyLLM TESTING START ---")
    print(f"Vocab size: {len(babyLLM.vocab.vocabList)}")
    print("\n--- BabyLLM TESTING COMPLETED ---")
