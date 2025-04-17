# CHARIS CAT 2025
# BABYLLM - babyLLM.py

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim 
from SCHOOL.staffroom.librarian import VOCAB
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
    def __init__(self, vocab, embedDimension, numNeurons, activationFunction, startIndex):
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
        self.totalTokenEvaluations = 0

        # Hold durations counters in a counter object, this dict ensures values are always defined before print (the value needs to be 0 to ensure a noop)
        self.durationCategories = {
            "Step": 0,
            "Save": 0,
            "Load": 0,
            "Print": 0,
            "Logits": 0,
            "Combine": 0,
            "Token": 0,
        }
        self.duration = Counter(self.durationCategories)
        self.durationDetail = Counter(self.durationCategories)

        self.startIndex = startIndex

        self.hud_height = 5
        self.term_height = shutil.get_terminal_size().lines
        self.hud_start_line = self.term_height - self.hud_height + 1

    """processes input sequence of tokens (str) to generate logits to predict the next token"""
    def forward(self, inputSeq):
        forwardStart = time.time()
        #print(f"Debug: Input to forward: {inputSeq}")

        """convert inputted tokens to indices (batch processing instead of looping)"""
        #print(f"Debug BABYLLM.forward: inputIndices: {inputIndices}")
        inputTimestamp = time.time()
        inputIndices = [self.vocab.tokenToIndex.get(tokenString, self.vocab.tokenToIndex["<UNK>"]) if not isinstance(tokenString, int) else tokenString for tokenString in inputSeq]
        #idxTime = time.time() - inputTimestamp

        """convert indices to embeddings"""
        inputEmbeds = []
        inputIndicesTensor = torch.tensor(inputIndices, device = modelDevice)
        inputEmbedsBatch = self.embedLayer(inputIndicesTensor)
        #inputEmbeds = [embed for embed in inputEmbedsBatch]     # list of (embed_dim,) tensors
        inputEmbeds = inputEmbedsBatch 

        #embedTimestamp = time.time()
        #inputIndicesTensor = torch.tensor(inputIndices, device=modelDevice)
        #inputEmbedsBatch = self.embedLayer(inputIndicesTensor)
        #inputEmbeds = inputEmbedsBatch
        #embedTime = time.time() - embedTimestamp

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
        #if not isinstance(inputEmbeds, list):
        #    inputEmbeds = [inputEmbeds]

        #neuronTimestamp = time.time()
        """PARALLEL NEURON LAYER input/processing (feature extraction)"""
        parallelNeuronOutput = self.parallelNeuronLayer.forward(inputEmbeds, trainingStepCounter = self.trainingStepCounter) 
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

        #print(f"Debug BABYLLM: Shape of lastTokenActivations BEFORE outputLayer: {lastTokenActivations.shape}")

        #outputTimestamp = time.time()
        logits = self.outputLayer.forward(combinedActivations)  
        #outputTime = time.time() - outputTimestamp

        #forwardTotal = time.time() - forwardStart
        #print(f"Debug BABYLLM.forward: probabilityDist shape: {probabilityDist.shape}")
        """returns a logits tensor of shape (1, vocabSize) showing predicted probabilities for the next token"""
        return logits
    
    """computes the cross-entropy loss between the models logits and the target token, essentially checking how good the models prediction was"""
    def computeLoss(self, logits, targetTokenIndex):
        targetTensor = torch.tensor([targetTokenIndex], dtype=torch.long, device = modelDevice)
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
        
        self.stats = Counter({ ###
            "loss": 0,
            "gradNorm": 0,
            "logitMin": 0,
            "logitMax": 0,
            # "scheduledSampling": 0,
            "tokenCount": 0,
        })
        self.statsDetail = Counter({ ###
            "loss": 0,
            "gradNorm": 0,
            "logitMin": 0,
            "logitMax": 0,
            # "scheduledSampling": 0,
            "tokenCount": 0,
        })
        
        #os.system('clear')
        print("babyLLM is heading back to school...")

        self.trainingStepCounter = 1 ###
        tokenCountsDetail = Counter() ###
        tokenCounts = Counter() ###

        """EPOCH LOOP"""
        for epoch in range(epochs):
            print(f"--- Epoch {epoch+1}/{epochs} Started ---")
            
            """TRAINING DATA (batches)"""
            try:
                for i, (inputSeq, targetSeq) in enumerate(trainingDataPairs):
                    stepStartTime = time.time()
                    inputTokenIndices = [vocab.tokenToIndex.get(token, vocab.tokenToIndex["<UNK>"]) for token in inputSeq]
                    targetTokenIndexSeq = [vocab.tokenToIndex.get(target_token, vocab.tokenToIndex["<UNK>"]) for target_token in targetSeq]

                    guessedTokenSeq = []
                    self.resetIfNeeded(context="training")

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
                        #forwardTimestamp = time.time()
                        logits = self.forward(inputSeqPredictions)
                        #forwardTime = time.time() - forwardTimestamp

                        #predictTimestamp = time.time()
                        predictedTokenIndex = self.getResponseFromLogits(logits)
                        #predictTime = time.time() - predictTimestamp

                        logitSeq.append(logits) # Store logits
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
                            
                            #lossTimestamp = time.time()
                            stepLoss = self.computeLoss(logits, targetTokenIndexSeq[j])
                            #lossTime = time.time() - lossTimestamp

                            losses.append(stepLoss) # append loss
                            cumulativeLoss += stepLoss # Cumulative loss (sum)

                    # Average loss over the sequence of predicted tokens
                    
                    if losses:
                        loss = cumulativeLoss / len(losses) # Average loss for THIS SEQUENCE (per token)
                    else:
                        loss = torch.tensor(0.0, device = modelDevice)

                    #backwardTimestamp = time.time()
                    loss.backward()
                    #backwardTime = time.time() - backwardTimestamp

                    #optTimestamp = time.time()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = gradientClipMaxNorm)
                    self.optimizer.step()
                    if modelDevice.type == 'mps':
                        torch.mps.empty_cache()
                    #optTime = time.time() - optTimestamp

                    """CALCULATE GRAD NORM"""
                    gradNorm = 0.0
                    for p in self.parameters():
                        if p.grad is not None:
                            gradNorm += p.grad.data.norm(2).item() ** 2
                    gradNorm = gradNorm ** 0.5
                    statUpdate = {
                        "loss": loss.item(), #cumulativeLoss.item(),
                        "gradNorm": gradNorm,
                        "tokenCount": len(losses),
                    }

                    with torch.no_grad():
                        logitMin = logits.min(dim=-1).values.mean().item()
                        logitMax = logits.max(dim=-1).values.mean().item()
                        statUpdate["logitMin"] = logitMin
                        statUpdate["logitMax"] = logitMax
                        memGatesTensor = self.latestMemGates
                        if memGatesTensor is not None:
                            memoryGates_str = f"Short:{memGatesTensor[0]:.3f}, Long:{memGatesTensor[1]:.3f}, Current:{memGatesTensor[2]:.3f}"
                        else:
                            memoryGates_str = "N/A"

                        self.stats.update(statUpdate)
                        self.statsDetail.update(statUpdate)

                    firstPredictedTokenIndex = predictedTokenIndices[0] if predictedTokenIndices else -1 # Get the first predicted token index for display
                    guessedTokenIndex = firstPredictedTokenIndex # for single token display

                    stepDuration = {"Step": time.time() - stepStartTime}
                    self.duration.update(stepDuration)
                    self.durationDetail.update(stepDuration)
                    guessedTokenSeq = []

                    """SAVE THE MODEL EVERY x STEPS"""
                    if self.trainingStepCounter % saveModelFreq == 0:
                        print(f"{S_output.S_apply('dim', "autosaving...")}{S_output.S_apply('reset', "")}")
                        babyLLM.saveModel()
                        success = f"autosave successful! saving every {saveModelFreq} steps, the next autosave will be at step {self.trainingStepCounter+saveModelFreq}..."
                        print(f"{S_output.S_apply('dim', success)}{S_output.S_apply('reset', "")}")

                    """PRINTING LOSS TO LOGS AND TERMINAL"""
                    terminalPrintStartTime = time.time()
                    if self.trainingStepCounter == 1:
                        # scheduledSamplingProb += scheduledSamplingProbIncrement
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                        runStart = f"\n--- {timestamp} ---\n{babyNote_loadCheckpointCheck}\n{userNote_loadCheckpoint}\n{babyNote_loadCheckpoint}{babyNote_runStart}\n{userNote_runStart}\n"
                        print(runStart)
                        with open(chatLogPath_forHumans, "a") as logFile:
                            logFile.write(runStart)

                        trainingChatLine = f"\n--- {timestamp} --- {babyNote_loadCheckpointCheck} - {userNote_loadCheckpoint} - {babyNote_loadCheckpoint}{babyNote_runStart} - {userNote_runStart}\n"
                        with open(trainingLogPath_100, "a") as logFile:
                            logFile.write(trainingChatLine)
                        with open(trainingLogPath_1000, "a") as logFile:
                            logFile.write(trainingChatLine)
                        with open(chatLogPath_trainingLog, "a") as logFile:
                            logFile.write(trainingChatLine)

                    # Track loss every 1000 steps
                    if self.trainingStepCounter % trainingLogFreq_1000 == 0:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                        # eventually want this at the end only
                        self.duration.update(self.durationCategories) # Force a noop update to ensure we have every category
                        durationLog_1000 = "Durations: " + ", ".join([
                            f"{name}: {(duration * 1000 / trainingLogFreq_1000 if trainingLogFreq_1000 > 0 else 0):.2f}ms"
                            for name, duration in self.duration.most_common() # most_common with no parameter returns everything, already sorted in reverse
                        ])
                        self.duration.clear()

                        with open(durationLogPath_1000, "a") as logFile:
                            logFile.write(durationLog_1000 + "\n")

                        topTokens = tokenCounts.most_common(10)
                        tokenCounts.clear()

                        with torch.no_grad():
                            normWeights = (babyLLM.parallelNeuronLayer.windowWeighting + 0.1)
                            normWeights /= (normWeights.sum() + 0.1)
                            sortedWeights = sorted(
                                zip(allWindowSizes, normWeights.cpu().numpy()),
                                key=lambda x: x[1],
                                reverse=True
                            )
                            weight_str = ", ".join(f"W{wsize}:{weight:.3f}" for wsize, weight in sortedWeights)

                        if self.totalTokenEvaluations > 0:
                            tokenPerfectRate = (self.perfectTokenCount / self.totalTokenEvaluations) * 100
                            print(f"{S_output.S_apply('perfect', f'Token Perfect: {self.perfectTokenCount} / {self.totalTokenEvaluations}')} → {tokenPerfectRate:.2f}%")

                        S_output.logTraining(
                            trainingLogPath_1000=trainingLogPath_1000,
                            trainingStepCounter=self.trainingStepCounter,
                            windowWeights_str=weight_str,
                            memoryGates_str=memoryGates_str,
                            stats=self.stats,
                            freq=trainingLogFreq_1000,
                            otherInfo_str=f"babyLLM.py {trainingLogFreq_1000}",
                            topTokens_str=str(topTokens),
                            #durationLog_str=durationLog
                        )
                        self.stats.clear()
                        
                        self.perfectTokenCount = 0
                        self.totalTokenEvaluations = 0
                        
                    # Track loss every 100 steps
                    if self.trainingStepCounter % trainingLogFreq_100 == 0:
                        topTokensDetail = tokenCountsDetail.most_common(10)
                        tokenCountsDetail.clear()
                        
                        with torch.no_grad():
                            normWeightsDetail = (babyLLM.parallelNeuronLayer.windowWeighting + 0.1)
                            normWeightsDetail /= (normWeightsDetail.sum() + 0.1)
                            sortedWeightsDetail = sorted(
                                zip(allWindowSizes, normWeightsDetail.cpu().numpy()),
                                key=lambda x: x[1],
                                reverse=True
                            )
                            weightDetail_str = ",".join(f"W{wsize}:{weight:.5f}" for wsize, weight in sortedWeightsDetail)
                            
                        self.durationDetail.update(self.durationCategories) # Force a noop update to ensure we have every category

                        #durationLogBabyLLM_inner_100 = (f"inner step {j+1}: forward: {forwardTime*1000:.2f}ms | predict: {predictTime*1000:.2f}ms | loss: {lossTime*1000:.2f}ms")
                        #durationLogBabyLLM_100 = (f"DEBUG: forward() timings: Index: {idxTime*1000:.2f}ms | Embed: {embedTime*1000:.2f}ms | Neuron: {neuronTime*1000:.2f}ms | Memory: {memoryTime*1000:.2f}ms | Output: {outputTime*1000:.2f}ms | Total: {forwardTotal*1000:.2f}ms")
                        durationLog_100 = "Durations: " + ", ".join([
                            f"{name}: {(duration * 1000 / trainingLogFreq_100 if trainingLogFreq_100 > 0 else 0):.2f}ms"
                            for name, duration in self.durationDetail.most_common()
                        ])
                        #durationLogCombined_100 = f"\n--- {timestamp} --- \n{durationLog_100} \n{durationLogBabyLLM_100} \n{durationLogBabyLLM_inner_100}\n"

                        with open(durationLogPath_100, "a") as logFile:
                            logFile.write(durationLog_100 + "\n")

                        self.durationDetail.clear()

                        #print(f"DEBUG: logitRange_str before logTraining: '{logitRangeDetail_str}'")
                        S_output.logTraining(
                            trainingLogPath_1000=trainingLogPath_100,
                            trainingStepCounter=self.trainingStepCounter,
                            freq=trainingLogFreq_100,
                            stats=self.statsDetail,
                            windowWeights_str=weightDetail_str,
                            memoryGates_str=memoryGates_str,
                            otherInfo_str="Training",
                            topTokens_str=str(topTokensDetail),
                        )
                        self.statsDetail.clear()


                    """PRINTING GUESSES TO THE TERMINAL"""
                    if self.trainingStepCounter % printFreq == 0:
                        guessedTokenSeq = [self.getTokenIndexAsString(idx) if idx != -1 else "<UNK>" for idx in predictedTokenIndices]
                        if guessedTokenSeq:
                            tokenCountsDetail.update(guessedTokenSeq)
                            tokenCounts.update(guessedTokenSeq)

                        #print(f"DEBUG: logitRange_str before logTraining: '{logitRange_str}'")
                        S_output.colourPrintTraining(
                            step=self.trainingStepCounter,
                            inputSentence=inputSeq,
                            guessedSeqStr=guessedTokenSeq[:windowMAX],
                            targetSeqStr=targetSeq[:windowMAX],
                            loss=loss.item(),
                        )
                    
                        terminalPrintDuration = {"Print": time.time() - terminalPrintStartTime}
                        self.duration.update(terminalPrintDuration)
                        self.durationDetail.update(terminalPrintDuration)
                        #self.totalTerminalPrintDuration += terminalPrintDuration

                        #self.HUD_fixScroll()

                    self.trainingStepCounter += 1
                    """END OF ONE TURN"""    

                #torch.save(self.state_dict(), f"babyLLM_epoch{epoch}.pth")
                self.scheduledSamplingProb = min(self.scheduledSamplingProb + scheduledSamplingProbIncrement, 1.0)
                self.saveModel()

            except KeyboardInterrupt:
                print("\nit's rude to interrupt people.. but, bye bye! :)")
                babyLLM.saveModel()
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
        self.durationDetail.update(saveDuration)

    """loads the model from a file"""
    def loadModel(self, filePath = modelFilePath):
        loadStartTime = time.time()
        try:
            print(f"loading model from path: {filePath}") 
            self.load_state_dict(torch.load(filePath), strict = saveLock)
            print(f"model loaded from {filePath}!")
            self.resetIfNeeded(context="inference")
            
            loadDuration = {"Load": time.time() - loadStartTime}
            self.duration.update(loadDuration)
            self.durationDetail.update(loadDuration)
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

        guessedTokenIndex = torch.multinomial(softmaxed, 1).item()

        logitsDuration = {"Logits": time.time() - logitsStartTime}
        self.duration.update(logitsDuration)
        self.durationDetail.update(logitsDuration)
        return guessedTokenIndex
    
    def getTokenIndexAsString(self, tokenIndex):
        return self.vocab.indexToToken.get(int(tokenIndex), "<UNK>") # tmp fix for token 1999
    
    def getNextToken(self, inputSeq, temperature=None):  
        getTokenStartTime = time.time()

        if temperature is None:
            temperature = self.temperature
        """returns an integer token index showing the models predicted next token."""
        
        getTokenDuration = {"Token": time.time() - getTokenStartTime}
        self.duration.update(getTokenDuration)
        self.durationDetail.update(getTokenDuration)
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

        combineDuration = {"Combine": time.time() - combineStartTime}
        self.duration.update(combineDuration)
        self.durationDetail.update(combineDuration)
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

    def resetIfNeeded(self, context="inference"):
        """
        Reset memory depending on the context:
        - 'inference': always resets to prevent memory echo
        - 'training': optionally resets every N steps/epochs if CONFIGured
        """
        if context == "inference":
            self.memoryLayer.resetMemory()
            print(f"resetting memory for new conversation...")
        elif context == "training":
            # Only reset memory every N steps if needed
            if hasattr(self, "stepsSinceMemoryReset"):
                self.stepsSinceMemoryReset += 1
            else:
                self.stepsSinceMemoryReset = 1

            if self.stepsSinceMemoryReset >= memoryLength:  # adjust this number if needed
                self.memoryLayer.resetMemory()
                print(f"resetting memory after {memoryLength} steps...")
                self.stepsSinceMemoryReset = 0
    
if __name__ == "__main__":

    embedDimension = embedDimension
    numNeurons = numNeurons
    activationFunction = activationFunction
    startIndex = trainingStartIndex  # default
    vocab = VOCAB(vocabSize = vocabSize)
    babyLLM = BABYLLM(vocab = vocab, embedDimension = embedDimension, numNeurons = numNeurons, activationFunction = activationFunction, startIndex = startIndex)
    babyLLM.loadModel()

    if os.path.exists(stepCheckpointFilePath):
        with open(stepCheckpointFilePath, "r") as f:
            try:
                savedStep = int(f.read().strip())

            except ValueError:
                babyNote_loadCheckpoint = f"{babyName} 'ah, i couldn't load step checkpoint file from {stepCheckpointFilePath}, resetting to 0...' "
                print(babyNote_loadCheckpoint)
                savedStep = 0
    else:
        babyNote_loadCheckpoint = f"{babyName} 'ah, the step checkpoint file {stepCheckpointFilePath} doesn't exist, resetting to 0...' "
        print(babyNote_loadCheckpoint)
        savedStep = 0

    babyNote_loadCheckpointCheck = f"{babyName} 'right, last time i got to step {savedStep}... want to restart from there?' "
    choice = input(babyNote_loadCheckpointCheck + f"\n{userName}: ")

    if choice == "" or choice.startswith("y"):
        babyNote_loadCheckpoint = f"{babyName} 'ok! let's go to step {savedStep}! "
        print(babyNote_loadCheckpoint, end="")
        startIndex = savedStep

    elif choice.startswith("r") or choice in ["random", "i dont care", "i don't care", "idc"]:
        startIndex = random.randint(0, len(vocab.tokens) - windowMAX - 1)
        babyNote_loadCheckpoint = f"{babyName} 'oh, cool! i'll pick a random spot to start from... umm... let's go to step {startIndex}! "
        print(babyNote_loadCheckpoint, end="")

    elif choice.startswith("n") or choice in ["start again", "restart"]:
        babyNote_loadCheckpoint = f"{babyName} 'alright, step {startIndex}, let's go back to the beginning :) "
        print(babyNote_loadCheckpoint, end="")
        startIndex = trainingStartIndex

    elif choice.isdigit():
        babyNote_loadCheckpoint = f"{babyName} 'damn that's specific! heading to step {startIndex}... "
        print(babyNote_loadCheckpoint, end="")
        startIndex = int(choice)

    else:
        babyNote_loadCheckpoint = f"{babyName} 'umm... i don't think i heard you properly, i'll just start from step {startIndex} :) but, "
        print(babyNote_loadCheckpoint, end="")
        startIndex = trainingStartIndex

    babyNote_runStart = f"what am i learning today?'" # no tag of 'babyllm:' because it merges with the end of above message in logs
    userNote_runStart = f"{userName}: '" + input(babyNote_runStart + f"\n{userName}: ").strip().lower() + "'"
    userNote_loadCheckpoint = f"{userName}: '{choice}'"

    #TESTinputSeq = ["what","will","you","do","out","there","now","?"]
    TESTinputSeq = ["i","love","you","this","is","good","music","is","life",]
    #TESTinputSeq = ["what"] 

    trainingDataPairs = vocab.genTrainingData(windowMAX, startIndex = startIndex)
    babyLLM.to(modelDevice)
    babyLLM.trainModel(trainingDataPairs, epochs = epochs)