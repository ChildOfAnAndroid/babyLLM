# CHARIS CAT 2025
# BABYLLM - babyLLM.py

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim 
#from VER1_SCHOOL.staffroom.librarian import LIBRARIAN
from VER1_BRAIN.LAYERS.vocab import VOCAB
from VER1_BRAIN.LAYERS.embed import EMBED
from VER1_BRAIN.LAYERS.interneuronNetwork import INTERNEURON_NETWORK
from VER1_BRAIN.LAYERS.logits import LOGITS
from VER1_BRAIN.LAYERS.memory import MEMORY
#import VER1_BRAIN.LAYERS.S_output as S_output
from VER1_BRAIN.LAYERS.S_output import S_OUTPUT
from VER1_config import *
from datetime import datetime
import random, os, sys, shutil, time
from collections import Counter
from VER1_SCHOOL.staffroom.counsellor import COUNSELLOR

"""this class combines all the core components of the babyLLM:"""
"""EMBEDLAYER: token embedding layer"""
"""PARALLELNEURONLAYER: layer of parallel neurons for feature extraction"""
"""OUTPUTLAYER: output layer to generate logits"""
"""MULTIWINDOWLAYER: (New) layer to incorporate multi-window context"""
"""it also manages training, loss computation, backpropagation, and response generation."""
class BABYLLM(nn.Module):
    def __init__(self, _counsellor, _s_output, _scribe, _librarian, _device = modelDevice):
        super().__init__()
        self.device = _device
        self.counsellor = _counsellor
        self.s_output = _s_output
        self.scribe = _scribe
        self.librarian = _librarian
        optimizerClass = getattr(optim, optimizerName)

        """LAYERS"""
        self.embed = EMBED(_counsellor = self.counsellor, _device = self.device)
        self.interneuronNetwork = INTERNEURON_NETWORK(_counsellor = self.counsellor, _device = self.device)
        self.logits = LOGITS(_counsellor = self.counsellor, _device = self.device)
        self.memory = MEMORY(_counsellor = self.counsellor, _device = self.device)

        """OPTIMIZER - this updates all of the layers learnable parameters"""
        #print("Registered Parameters:")
        #for name, param in BABYLLM.named_parameters(self):
        #    print(name, param.shape)
        self.optimizer = optimizerClass(
            list(self.embed.parameters()) +
            list(self.interneuronNetwork.parameters()) + 
            list(self.logits.parameters()) +
            list(self.memory.parameters()),
            lr=learningRate, weight_decay=0.001
        )

        self.scheduledSamplingProb = 0.0
        self.perfectTokenCount = 0
        self.totalTokenEvaluations = 0

        self.startIndex = 0

    """processes input sequence of tokens (str) to generate logits to predict the next token"""
    def forward(self, inputSeq):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            if debugPrints: print(f"Debug: Input to forward: {inputSeq}")

            """convert inputted tokens to indices (batch processing instead of looping)"""
            ʕっʘ‿ʘʔっ("inputIndices")
            inputIndices = [self.librarian.tokenToIndex.get(tokenString, self.librarian.tokenToIndex["<UNK>"]) if not isinstance(tokenString, int) else tokenString for tokenString in inputSeq]
            if debugPrints: print(f"Debug BABYLLM.forward: inputIndices: {inputIndices}")

            """convert indices to embeddings"""
            ʕっʘ‿ʘʔっ("inputEmbeds")
            inputEmbeds = []
            inputIndicesTensor = torch.tensor(inputIndices, device = modelDevice)
            inputEmbedsBatch = self.embed(inputIndicesTensor)
            inputEmbeds = inputEmbedsBatch 

            """PARALLEL NEURON LAYER input/processing (feature extraction)"""
            ʕっʘ‿ʘʔっ("neuronForward")
            parallelNeuronOutput = self.interneuronNetwork.forward(inputEmbeds) 

            """RESIZE NEURON LAYER TO STANDARD SIZE FOR COMBINED FORWARD PROCESSING"""
            combinedActivationsTensor = torch.mean(parallelNeuronOutput, dim=0, keepdim=True)

            """MEMORY LAYER PROCESSING - NOW PROCESS THE COMBINED ACTIVATIONS"""
            ʕっʘ‿ʘʔっ("memoryForward")
            memoryLayerOutput = self.memory.forward(combinedActivationsTensor)
            self.latestMemGates = self.memory.latestMemoryGates.detach() 
            combinedActivations = memoryLayerOutput

            ʕっʘ‿ʘʔっ("logitsForward")
            logits = self.logits.forward(combinedActivations)  

            """returns a logits tensor of shape (1, vocabSize) showing predicted probabilities for the next token"""
            return logits
    
    """computes the cross-entropy loss between the models logits and the target token, essentially checking how good the models prediction was"""
    def computeLoss(self, logits, targetTokenIndex):
        with self.counsellor.infodump("computeLoss") as ʕっʘ‿ʘʔっ:
            if skipComputeLoss:
                ʕっʘ‿ʘʔっ("skipping loss!")
                loss = torch.tensor([0.1], requires_grad=True, device=modelDevice)  # Constant scalar tensor
            else:
                ʕっʘ‿ʘʔっ("targetTensor")
                targetTensor = torch.tensor([targetTokenIndex], dtype=torch.long, device = modelDevice)
                if debugPrints: print(f"Debug BABYLLM.computeLoss: predictions shape: {logits.shape}\nDebug BABYLLM.computeLoss: predictions (first 10): {logits[:10]}\nDebug BABYLLM.computeLoss: targetTokenIndex: {targetTokenIndex}")
                """Handle cases where logits might be 1D (unsqueeze to make it 2D for cross_entropy)"""
                """SUS!!"""
                if logits.dim() == 1: logits = logits.unsqueeze(0) 
                ʕっʘ‿ʘʔっ("crossEntropy")
                loss = F.cross_entropy(logits, targetTensor) 
            #print(f"Debug BABYLLM.computeLoss: Loss value: {loss.item():.4f}")
            """returns a scalar tensor representing the cross-entropy loss value"""
            return loss
        
    
    """backpropagation and optimization, computes gradients of the loss and uses the optimizer to update the models weights"""
    def backward(self, loss):
        with self.counsellor.infodump("backward") as ʕっʘ‿ʘʔっ:
            ʕっʘ‿ʘʔっ("zero_grad")
            self.optimizer.zero_grad()  # Reset gradients
            for name, p in self.named_parameters():
                if p.grad is None:
                    if debugPrints: print(f"BEFORE backward: No grad for {name}")
                else:
                    if debugPrints: print(f"Grad BEFORE backward for {name} - requires_grad: {p.requires_grad}")
            with torch.autograd.set_detect_anomaly(True):
                ʕっʘ‿ʘʔっ("loss.backward")
                loss.backward()
            for name, p in self.named_parameters():
                if p.grad is None:
                    if debugPrints: print(f"AFTER backward: No grad for {name}")
                else:
                    if debugPrints: print(f"Grad AFTER backward for {name} - requires_grad: {p.requires_grad}")
            for name, p in self.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    print(f"babyLLM.backward - non-finite grad in: {name}") 
                    return
            ʕっʘ‿ʘʔっ("optimizer.step")
            self.optimizer.step()  # Update weights

    """this iterates through training data, performing forward passes, loss computation, backpropagation, and optimization for each step."""
    def trainModel(self, trainingDataPairs, epochs):
        with self.counsellor.infodump("trainModel") as ʕっʘ‿ʘʔっ:
            #print(f"Debug tokenToIndex (First 20): {list(librarian.tokenToIndex.items())[:20]}")
            numTokens = numTokensPerStep
            if isinstance(numTokens, torch.Tensor):
                numTokens = numTokens.item()
            numTokens = int(numTokens)
            
            self.stats = Counter({
                "loss": 0,
                "gradNorm": 0,
                "logitMin": 0,
                "logitMax": 0,
                # "scheduledSampling": 0,
                "tokenCount": 0,
            })
            self.statsDetail = Counter({
                "loss": 0,
                "gradNorm": 0,
                "logitMin": 0,
                "logitMax": 0,
                # "scheduledSampling": 0,
                "tokenCount": 0,
            })
            
            #os.system('clear')
            print("babyLLM is heading back to school...")

            self.trainingStepCounter = 1
            tokenCountsDetail = Counter()
            tokenCounts = Counter()

            """EPOCH LOOP"""
            for epoch in range(epochs):
                ʕっʘ‿ʘʔっ("startEpoch")
                print(f"--- Epoch {epoch+1}/{epochs} Started ---")
                
                """TRAINING DATA (batches)"""
                try:
                    for i, (inputSeq, targetSeq) in enumerate(trainingDataPairs):
                        ʕっʘ‿ʘʔっ("startTrainingStep")
                        inputTokenIndices = [self.librarian.tokenToIndex.get(token, self.librarian.tokenToIndex["<UNK>"]) for token in inputSeq]
                        targetTokenIndexSeq = [self.librarian.tokenToIndex.get(target_token, self.librarian.tokenToIndex["<UNK>"]) for target_token in targetSeq]

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
                            ʕっʘ‿ʘʔっ("logitsForward")
                            try:
                                #with torch.profiler.profile(record_shapes=True) as prof:
                                logits = self.forward(inputSeqPredictions)
                            except RuntimeError as e:
                                print("TUTOR.trainStep.forward failed!", e)
                                return
                            
                            #print(prof.key_averages().table())

                            ʕっʘ‿ʘʔっ("getResponseFromLogits")
                            predictedTokenIndex = self.getResponseFromLogits(logits)

                            logitSeq.append(logits) # Store logits
                            predictedTokenIndices.append(predictedTokenIndex) # Store predicted token index

                            # Decide whether to use teacher forcing or model prediction for next input
                            if False: #scheduledSampling and random.random() < scheduledSamplingProb:
                                # Use model's prediction (scheduled sampling)
                                nextTokenInput = predictedTokenIndex
                            else:
                                # Use teacher forcing (ground truth)
                                if j < len(targetTokenIndexSeq): # Check if target exists for this step
                                    ʕっʘ‿ʘʔっ("using ground truth")
                                    nextTokenInput = targetTokenIndexSeq[j] # Use ground truth target token
                                else:
                                    ʕっʘ‿ʘʔっ("using scheduled sampling")
                                    nextTokenInput = predictedTokenIndex # if no more targets, use prediction

                            inputSeqPredictions.append(nextTokenInput) # Append token index (predicted or ground truth) as next input

                            if j < len(targetTokenIndexSeq): # compute loss only if target exists
                                ʕっʘ‿ʘʔっ("computeLoss")
                                self.totalTokenEvaluations += 1
                                if predictedTokenIndex == targetTokenIndexSeq[j]:
                                    self.perfectTokenCount += 1
                                
                                stepLoss = self.computeLoss(logits, targetTokenIndexSeq[j])

                                losses.append(stepLoss) # append loss
                                cumulativeLoss += stepLoss # Cumulative loss (sum)
                        
                        if losses:
                            loss = cumulativeLoss / len(losses) # Average loss for THIS SEQUENCE (per token)
                        else:
                            loss = torch.tensor(0.0, device = modelDevice)

                        ʕっʘ‿ʘʔっ("loss.backward")
                        if not torch.isfinite(loss): 
                            print("BABYLLM.trainStep.backward !!! Loss is NaN or Inf:", loss)
                            return
                        else: 
                            if debugPrints: print("BABYLLM.trainStep.backward - loss is not NaN or Inf:", loss)
                            
                        try:
                            with torch.profiler.profile(record_shapes=True) as prof:
                            #with torch.mps.profiler.profile(mode='interval', wait_until_completed=False) as prof:
                                loss.backward()
                        except RuntimeError as e:
                            print("BABYLLM.trainStep.backward failed!", e)
                            #
                            return
                        
                        print(prof.key_averages().table())

                        ʕっʘ‿ʘʔっ("clip_grad_norm")
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = gradientClipMaxNorm)
                        ʕっʘ‿ʘʔっ("self.optimizer.step")
                        self.optimizer.step()
                        if modelDevice.type == 'mps':
                            ʕっʘ‿ʘʔっ("emptyCache (mps)")
                            torch.mps.empty_cache()

                        """CALCULATE GRAD NORM"""
                        ʕっʘ‿ʘʔっ("calcGradNorm")
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
                            ʕっʘ‿ʘʔっ("torch.no_grad")
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

                        guessedTokenSeq = []

                        """SAVE THE MODEL EVERY x STEPS"""
                        if self.trainingStepCounter % saveModelFreq == 0:
                            ʕっʘ‿ʘʔっ("autoSaving")
                            print(f"{self.s_output.S_apply('dim', "autosaving...")}{self.s_output.S_apply('reset', "")}")
                            self.saveModel()
                            success = f"autosave successful! saving every {saveModelFreq} steps, the next autosave will be at step {self.trainingStepCounter+saveModelFreq}..."
                            print(f"{self.s_output.S_apply('dim', success)}{self.s_output.S_apply('reset', "")}")

                        """PRINTING LOSS TO VER1_LOGS AND TERMINAL"""
                        terminalPrintStartTime = time.time()
                        if self.trainingStepCounter == 1:
                            ʕっʘ‿ʘʔっ("printing Start Logs")
                            # scheduledSamplingProb += scheduledSamplingProbIncrement
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                            #runStart = f"\n--- {timestamp} ---\n{babyNote_loadCheckpointCheck}\n{userNote_loadCheckpoint}\n{babyNote_loadCheckpoint}{babyNote_runStart}\n{userNote_runStart}\n"
                            runStart = ""
                            print(runStart)
                            with open(chatLogPath_forHumans, "a") as logFile:
                                logFile.write(runStart)

                            trainingChatLine = "ruining my own bot"
                            #trainingChatLine = f"\n--- {timestamp} --- {babyNote_loadCheckpointCheck} - {userNote_loadCheckpoint} - {babyNote_loadCheckpoint}{babyNote_runStart} - {userNote_runStart}\n"
                            with open(trainingLogPath_100, "a") as logFile:
                                logFile.write(trainingChatLine)
                            with open(trainingLogPath_1000, "a") as logFile:
                                logFile.write(trainingChatLine)
                            with open(chatLogPath_trainingLog, "a") as logFile:
                                logFile.write(trainingChatLine)

                        # Track loss every 1000 steps
                        if self.trainingStepCounter % trainingLogFreq_1000 == 0:
                            ʕっʘ‿ʘʔっ("printing trainingLog_1000")
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                            # eventually want this at the end only
                            #self.duration.update(self.durationCategories) # Force a noop update to ensure we have every category
                            #durationLog_1000 = "Durations: " + ", ".join([
                            #    f"{name}: {(duration * 1000 / trainingLogFreq_1000 if trainingLogFreq_1000 > 0 else 0):.2f}ms"
                            #    for name, duration in self.duration.most_common() # most_common with no parameter returns everything, already sorted in reverse
                            #])
                            #self.duration.clear()

                            #with open(durationLogPath_1000, "a") as logFile:
                            #    logFile.write(durationLog_1000 + "\n")

                            topTokens = tokenCounts.most_common(10)
                            tokenCounts.clear()

                            with torch.no_grad():
                                normWeights = (self.interneuronNetwork.cerebellum + 0.1)
                                normWeights /= (normWeights.sum() + 0.1)
                                sortedWeights = sorted(
                                    zip(allWindowSizes, normWeights.cpu().numpy()),
                                    key=lambda x: x[1],
                                    reverse=True
                                )
                                weight_str = ", ".join(f"W{wsize}:{weight:.3f}" for wsize, weight in sortedWeights)

                            if self.totalTokenEvaluations > 0:
                                tokenPerfectRate = (self.perfectTokenCount / self.totalTokenEvaluations) * 100
                                print(f"{self.s_output.S_apply('perfect', f'Token Perfect: {self.perfectTokenCount} / {self.totalTokenEvaluations}')} → {tokenPerfectRate:.2f}%")

                            self.s_output.S_logTraining(
                                _trainingLogPath=trainingLogPath_1000,
                                _trainingStepCounter=self.trainingStepCounter,
                                _INN_cerebellum_str=weight_str,
                                _memoryGates_str=memoryGates_str,
                                _stats=self.stats,
                                _freq=trainingLogFreq_1000,
                                _otherInfo_str=f"self.py {trainingLogFreq_1000}",
                                _topTokens_str=str(topTokens),
                                #durationLog_str=durationLog
                            )
                            self.stats.clear()
                            
                            self.perfectTokenCount = 0
                            self.totalTokenEvaluations = 0
                            
                        # Track loss every 100 steps
                        if self.trainingStepCounter % trainingLogFreq_100 == 0:
                            ʕっʘ‿ʘʔっ("printing trainingLog_100")
                            topTokensDetail = tokenCountsDetail.most_common(10)
                            tokenCountsDetail.clear()
                            
                            with torch.no_grad():
                                normWeightsDetail = (self.interneuronNetwork.cerebellum + 0.1)
                                normWeightsDetail /= (normWeightsDetail.sum() + 0.1)
                                sortedWeightsDetail = sorted(
                                    zip(allWindowSizes, normWeightsDetail.cpu().numpy()),
                                    key=lambda x: x[1],
                                    reverse=True
                                )
                                weightDetail_str = ",".join(f"W{wsize}:{weight:.5f}" for wsize, weight in sortedWeightsDetail)
                                
                            #self.durationDetail.update(self.durationCategories) # Force a noop update to ensure we have every category

                            #durationLogBabyLLM_inner_100 = (f"inner step {j+1}: forward: {forwardTime*1000:.2f}ms | predict: {predictTime*1000:.2f}ms | loss: {lossTime*1000:.2f}ms")
                            #durationLogBabyLLM_100 = (f"DEBUG: forward() timings: Index: {idxTime*1000:.2f}ms | Embed: {embedTime*1000:.2f}ms | Neuron: {neuronTime*1000:.2f}ms | Memory: {memoryTime*1000:.2f}ms | Output: {outputTime*1000:.2f}ms | Total: {forwardTotal*1000:.2f}ms")
                            #durationLog_100 = "Durations: " + ", ".join([
                            #    f"{name}: {(duration * 1000 / trainingLogFreq_100 if trainingLogFreq_100 > 0 else 0):.2f}ms"
                            #    for name, duration in self.durationDetail.most_common()
                            #])
                            #durationLogCombined_100 = f"\n--- {timestamp} --- \n{durationLog_100} \n{durationLogBabyLLM_100} \n{durationLogBabyLLM_inner_100}\n"

                            #with open(durationLogPath_100, "a") as logFile:
                            #    logFile.write(durationLog_100 + "\n")

                            #self.durationDetail.clear()

                            #print(f"DEBUG: logitRange_str before S_logTraining: '{logitRangeDetail_str}'")
                            self.s_output.S_logTraining(
                                _trainingLogPath=trainingLogPath_100,
                                _trainingStepCounter=self.trainingStepCounter,
                                _freq=trainingLogFreq_100,
                                _stats=self.statsDetail,
                                _INN_cerebellum_str=weightDetail_str,
                                _memoryGates_str=memoryGates_str,
                                _otherInfo_str="Training",
                                _topTokens_str=str(topTokensDetail),
                            )
                            self.statsDetail.clear()


                        """PRINTING GUESSES TO THE TERMINAL"""
                        if self.trainingStepCounter % printFreq == 0:
                            ʕっʘ‿ʘʔっ("pritning training step")
                            guessedTokenSeq = [self.getTokenIndexAsString(idx) if idx != -1 else "<UNK>" for idx in predictedTokenIndices]
                            if guessedTokenSeq:
                                tokenCountsDetail.update(guessedTokenSeq)
                                tokenCounts.update(guessedTokenSeq)

                            #print(f"DEBUG: logitRange_str before S_logTraining: '{logitRange_str}'")
                            self.s_output.S_colourPrintTraining(
                                _step=self.trainingStepCounter,
                                _inputSeq=inputSeq,
                                _guessedSeq_str=guessedTokenSeq,
                                _targetSeq_str=targetSeq[:windowMAX],
                                _loss=loss.item(),
                            )
                        
                            #terminalPrintDuration = {"Print": time.time() - terminalPrintStartTime}
                            #self.duration.update(terminalPrintDuration)
                            #self.durationDetail.update(terminalPrintDuration)
                            #self.totalTerminalPrintDuration += terminalPrintDuration

                            #self.HUD_fixScroll()

                        self.trainingStepCounter += 1
                        ʕっʘ‿ʘʔっ("end of one turn")
                        """END OF ONE TURN"""    

                    #torch.save(self.state_dict(), f"babyLLM_epoch{epoch}.pth")
                    self.scheduledSamplingProb = min(self.scheduledSamplingProb + scheduledSamplingProbIncrement, 1.0)
                    self.saveModel()

                except KeyboardInterrupt:
                    ʕっʘ‿ʘʔっ("keyboard interrupt")
                    print("\nit's rude to interrupt people.. but, bye bye! :)")
                    self.saveModel()
                    sys.exit(8)

            print("--- Training Completed! ---")

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
            loadStartTime = time.time()
            try:
                print(f"loading model from path: {filePath}") 
                self.load_state_dict(torch.load(filePath), strict = saveLock)
                print(f"model loaded from {filePath}!")
                self.resetIfNeeded(context="inference")
                
                #loadDuration = {"Load": time.time() - loadStartTime}
                #self.duration.update(loadDuration)
                #self.durationDetail.update(loadDuration)
            except FileNotFoundError:
                print("No saved model found.")

    """this takes the output logits, does temperature scaling and softmax to create a probability distribution over the vocab, 
    and then selects most likely response token"""
    def getResponseFromLogits(self, logits, temperature=temperature):
        with self.counsellor.infodump("getResponseFromLogits") as ʕっʘ‿ʘʔっ:
            logits = logits / temperature
            softmaxed = torch.softmax(logits, dim=1)

            guessedTokenIndex = torch.multinomial(softmaxed, 1).item()
            return guessedTokenIndex
    
    def getTokenIndexAsString(self, tokenIndex):
        with self.counsellor.infodump("getTokenIndexAsString") as ʕっʘ‿ʘʔっ:
            return self.librarian.indexToToken.get(int(tokenIndex), "<UNK>") # tmp fix for token 1999
    
    def getNextToken(self, inputSeq, temperature=None):  
        with self.counsellor.infodump("getNextToken") as ʕっʘ‿ʘʔっ:
            if temperature is None:
                temperature = self.temperature
            """returns an integer token index showing the models predicted next token."""
            return self.getResponseFromLogits(self.forward(inputSeq), temperature)
    
    """combines the parallelNeronLayer output and the multiWindowLayer output into one output"""
    def combineOutputs(self, output1, output2):
        with self.counsellor.infodump("combineOutputs") as ʕっʘ‿ʘʔっ:
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
            return finalOutput

    def babyllm_diary_entry(self, interneuronNetwork, step):
        with self.counsellor.infodump("babyllm_diary_entry") as ʕっʘ‿ʘʔっ:
            # Grab current window weightings
            weights = interneuronNetwork.cerebellum.detach().cpu().numpy()
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

    def resetIfNeeded(self, context="inference"):
        with self.counsellor.infodump("resetIfNeeded") as ʕっʘ‿ʘʔっ:
            """
            Reset memory depending on the context:
            - 'inference': always resets to prevent memory echo
            - 'training': optionally resets every N steps/epochs if CONFIGured
            """
            if context == "inference":
                self.memory.resetMemory()
                print(f"resetting memory for new conversation...")
            elif context == "training":
                # Only reset memory every N steps if needed
                if hasattr(self, "stepsSinceMemoryReset"):
                    self.stepsSinceMemoryReset += 1
                else:
                    self.stepsSinceMemoryReset = 1

                if self.stepsSinceMemoryReset >= memoryLength:  # adjust this number if needed
                    self.memory.resetMemory()
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