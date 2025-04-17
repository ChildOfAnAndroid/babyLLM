# CHARIS CAT 2025 
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# MULTI-TOKEN AUTOREGRESSIVE TRAINING MODULE 
# SCHOOL/staffroom/tutor.py

import random, sys
from collections import Counter
from datetime import datetime
import torch
from config import *

class TUTOR:
    def __init__(self, _counsellor, _s_output, _scribe, _librarian, _device = modelDevice):
        self.counsellor = _counsellor
        self.s_output = _s_output
        self.scribe = _scribe
        self.librarian = _librarian
        self.device = _device

        self.perfectTokens = 0
        self.totalTokenEvaluations = 0
        self.scheduledSamplingProb = 0
        #model.to(self.device)

    def trainStep(self, _inputTokenIndices, _targetTokenIndexSeq, _model):
        with self.counsellor.infodump("trainStep") as ʕっʘ‿ʘʔっ:
            ʕっʘ‿ʘʔっ("_model.optimizer.zero_grad")
            self.model = _model
            self.model.optimizer.zero_grad() # clears gradients last step - needed before any backward
            self.trainingStepCounter += 1
            predictedTokenIndices = [] # this list grows each time a new token is predicted
            inputSeqPredictions = list(_inputTokenIndices)  # Start with input context, create a COPY!
            buffer = torch.zeros(windowMAX, dtype=torch.long, device=self.device) # creates buffer/step instead of recreating tensors inside loop
            buffer[:len(inputSeqPredictions)] = torch.as_tensor(inputSeqPredictions, device=self.device)
            logitSeq = [] # raw output of each prediction
            cumulativeLoss = torch.tensor(0.0, device=self.device) # sum of token losses for THIS sequence - averaged at the end

            for j in range(numTokensPerStep): # Predict multiple tokens in a sequence, one at a time
                ʕっʘ‿ʘʔっ("FORWARD")
                inputTensor = buffer[:len(inputSeqPredictions)] # slices input to only keep relevant part
                try:
                    #with torch.profiler.profile(record_shapes=True) as prof:
                    logits = self.model.forward(inputTensor)
                except RuntimeError as e:
                    print("TUTOR.trainStep.forward failed!", e)
                    return
                
                #print(prof.key_averages().table())

                ʕっʘ‿ʘʔっ("getResponseFromLogits")
                predictedTokenIndex = self.model.getResponseFromLogits(logits)
                ʕっʘ‿ʘʔっ("inputSeqPredictions")
                predictedTokenIndices.append(predictedTokenIndex) # tensor shape [1]
                logitSeq.append(logits)
                nextTokenInput = (predictedTokenIndex 
                    if scheduledSampling and random.random() < self.scheduledSamplingProb # 
                    else _targetTokenIndexSeq[j] if j < len(_targetTokenIndexSeq) #
                    else predictedTokenIndex) #
                inputSeqPredictions.append(nextTokenInput) # multi-token autoregressive generation: append next token to your current input — becomes the prompt for the next token

                ʕっʘ‿ʘʔっ("loop through tokens for this step")
                if j < len(_targetTokenIndexSeq):
                    ʕっʘ‿ʘʔっ("totalTokenCounter")
                    self.totalTokenEvaluations += 1

                    ʕっʘ‿ʘʔっ("computeLoss")
                    stepLoss = self.model.computeLoss(logits, _targetTokenIndexSeq[j])

                    ʕっʘ‿ʘʔっ("appendStepLoss")
                    cumulativeLoss += stepLoss

            ʕっʘ‿ʘʔっ("actions after looping")
            loss = cumulativeLoss / len(_targetTokenIndexSeq) if len(_targetTokenIndexSeq) > 0 else torch.tensor(0.0, device=self.device)
            if hasattr(self.model.interneuronNetwork, "entropyBonus"):
                ʕっʘ‿ʘʔっ("entropyBonus")
                loss = loss - (self.model.interneuronNetwork.entropyBonus * 0.05)

            ʕっʘ‿ʘʔっ("increaseScheduledSamplingProb")
            self.scheduledSamplingProb = min(self.scheduledSamplingProb + scheduledSamplingProbIncrement, 1.0)

            ʕっʘ‿ʘʔっ("backward")
            #if not torch.isfinite(loss): 
                #print("TUTOR.trainStep.backward !!! Loss is NaN or Inf:", loss)
                #return
            #else: 
                #if debugPrints: print("TUTOR.trainStep.backward - loss is not NaN or Inf:", loss)
                
            try:
                with torch.profiler.profile(record_shapes=True) as prof:
                #with torch.mps.profiler.profile(mode='interval', wait_until_completed=False) as prof:
                    self.model.backward(loss)
            except RuntimeError as e:
                print("TUTOR.trainStep.backward failed!", e)
                #
                return

            print(prof.key_averages().table())
            
            ʕっʘ‿ʘʔっ("clip_grad_norm")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = gradientClipMaxNorm)
            ʕっʘ‿ʘʔっ("self.model.optimizer.step")
            self.model.optimizer.step()
            if self.device.type == 'mps':
                ʕっʘ‿ʘʔっ("emptyCache (mps)")
                torch.mps.empty_cache()

            ʕっʘ‿ʘʔっ("finalActions")
            self.model.memory.updateMemoryBuffers()

            return loss, predictedTokenIndices, logitSeq
        
    """this iterates through training data, performing forward passes, loss computation, backpropagation, and optimization for each step."""
    def trainModel(self, _trainingDataPairs, _epochs, _startIndex, _model):
        self.model = _model
        self.startIndex = _startIndex
        torch.autograd.set_detect_anomaly(True)
        with self.counsellor.infodump("trainModel") as ʕっʘ‿ʘʔっ:
            #if debugPrints: print(f"Debug tokenToIndex (First 20): {list(librarian.tokenToIndex.items())[:20]}")
            for name, param in self.model.named_parameters(): print(name, param.device)
            ʕっʘ‿ʘʔっ("COUNTERS INIT")
            self.trainingStepCounter = 0
            self.stats = Counter({"loss": 0, "gradNorm": 0, "logitMin": 0, "logitMax": 0, "scheduledSampling": 0, "tokenCount": 0,})
            self.tokenCounts = Counter()

            ʕっʘ‿ʘʔっ("back to school!")
            print("babyLLM is heading back to school...")

            """EPOCH LOOP"""
            ʕっʘ‿ʘʔっ("epoch♥")
            for epoch in range(_epochs):
                print(f"--- lesson {epoch+1}/{_epochs} started ---")
                """TRAINING DATA (batches)"""
                for i, (_inputSeq, _targetSeq) in enumerate(_trainingDataPairs):
                    ʕっʘ‿ʘʔっ("♥BEFORE TRAINING STEP♥")
                    ʕっʘ‿ʘʔっ("♥tokenToIndex")
                    inputTokenIndices = [self.librarian.tokenToIndex.get(t, self.librarian.tokenToIndex["<UNK>"]) for t in _inputSeq]
                    targetTokenIndexSeq = [self.librarian.tokenToIndex.get(t, self.librarian.tokenToIndex["<UNK>"]) for t in _targetSeq]
                    self.inputSeq = _inputSeq
                    self.targetSeq = _targetSeq

                    ʕっʘ‿ʘʔっ("♥resetMemory")
                    self.model.resetMemory(context="training")

                    ʕっʘ‿ʘʔっ("♥TRAINING STEP♥")
                    ʕっʘ‿ʘʔっ("♥trainStep")
                    result = self.trainStep(inputTokenIndices, targetTokenIndexSeq, self.model)
                    loss, self.predictedTokenIndices, logitSeq = result
                    self.stepLossFloat = loss.detach().cpu().item()

                    ʕっʘ‿ʘʔっ("♥perfectTokens")
                    target = torch.tensor(targetTokenIndexSeq[:numTokensPerStep], device=modelDevice)
                    predicted = torch.tensor(self.predictedTokenIndices, device=modelDevice)

                    correct = (predicted == target).sum().item()
                    self.perfectTokens += correct
                    self.totalTokenEvaluations += len(target)

                    ʕっʘ‿ʘʔっ("♥stats") # CALCULATE BASIC STATS
                    stats = self.getStats(logitSeq)
                    self.stats.update(stats)

                    if self.trainingStepCounter % saveModelFreq == 0:
                        ʕっʘ‿ʘʔっ("♥saveFreq")
                        self.saveFreqActions()

                    if self.trainingStepCounter % printFreq == 0:
                        ʕっʘ‿ʘʔっ("♥printFreq")
                        self.guessedTokenSeq = [self.librarian.indexToToken.get(idx.item(), "<UNK>") for idx in self.predictedTokenIndices]
                        if self.guessedTokenSeq: 
                            self.tokenCounts.update(self.guessedTokenSeq)
                        self.printFreqActions()

                    #if self.trainingStepCounter % trainingLogFreq_1000 == 0:
                        #ʕっʘ‿ʘʔっ("♥trainingLogFreq_1000") # PRINTING LOGS TO TXT AND TERMINAL
                        #self.logFreqActions(_trainingDataPairs)
       
                    # Track loss every 100 steps
                    if self.trainingStepCounter % trainingLogFreq_100 == 0:
                        ʕっʘ‿ʘʔっ("♥logFreq_100")
                        self.logFreqActions(_trainingDataPairs)

                ʕっʘ‿ʘʔっ("♥END TURN♥") # END OF ONE TURN
                ʕっʘ‿ʘʔっ("♥finalSaveBeforeNewEpoch")
                self.model.saveModel(_newStartIndex = self.startIndex)
        print("--- tutoring complete! ---")

    def saveFreqActions(self): 
        #ʕっʘ‿ʘʔっ("♥autoSave") # SAVE THE MODEL EVERY x STEPS
        print(self.s_output.S_apply('dim', 'autosaving...') + self.s_output.S_apply('reset', ''))
        self.model.saveModel(_newStartIndex = self.startIndex)
        p = self.trainingStepCounter + saveModelFreq
        print(self.s_output.S_apply('dim', f"autosave successful! saving every {saveModelFreq} steps, the next autosave will be at step {p}...") + self.s_output.S_apply('reset', ''))
                  
    def printFreqActions(self): 
        #ʕっʘ‿ʘʔっ("♥printGuessesToTerminal♥") # PRINTING TRAINING OUTPUT TO TERMINAL
        #recentLoss = sum(self.recentPrintLosses)/len(self.recentPrintLosses) if self.recentPrintLosses else None
        #ʕっʘ‿ʘʔっ("♥S_output.S_colourPrintTraining")
        self.s_output.S_colourPrintTraining(
            _step = self.trainingStepCounter,
            _inputSeq = self.inputSeq,
            _guessedSeq_str = self.guessedTokenSeq,
            _targetSeq_str = self.targetSeq[:windowMAX],
            _loss = self.stepLossFloat,
            _totalTokenCount = self.tokenCounts)
        #ʕっʘ‿ʘʔっ("♥SCRIBE.maybeCommentOnGuess")
        self.scribe.maybeCommentOnGuess(self.guessedTokenSeq, self.stepLossFloat, "scribe", 0.005)
        
    def logFreqActions(self, _trainingDataPairs): # could also do 10x log freq??
        
        #ʕっʘ‿ʘʔっ("♥calculateTrainingDataRemaining")
        trainingDataRemaining = len(_trainingDataPairs) - self.trainingStepCounter
        trainingDataPercent = (trainingDataRemaining / len(_trainingDataPairs)) * 100
        print(f"step {self.trainingStepCounter} | tokens remaining: {len(_trainingDataPairs) - self.trainingStepCounter} ({trainingDataPercent:.2f}%)")
        
        #ʕっʘ‿ʘʔっ("♥getStringStats")
        stringStats = self.model.getStringStats(self.predictedTokenIndices, self.tokenCounts)
        #ʕっʘ‿ʘʔっ("♥getComplexStats")
        complexStats = self.model.getComplexStats()
        self.stats.update(complexStats)

        #ʕっʘ‿ʘʔっ("♥S_output.S_logTraining")
        self.s_output.S_logTraining(
            _trainingLogPath = trainingLogPath_100,
            _trainingStepCounter = self.trainingStepCounter,
            _freq = trainingLogFreq_1000,
            _stats = self.model.stats,
            _INN_cerebellum_str = stringStats["INN_cerebellum_str"],
            _INN_judgeBias_str = stringStats["INN_judgeBias_str"],
            _INN_credbilityBias_str = stringStats["INN_credibilityBias_str"],
            _topTokens_str = stringStats["topTokens"],
            _otherInfo_str = f"{stringStats['tokenPerfect']} | {stringStats['windowVotes_str']} | TUTOR.py {trainingLogFreq_1000}")
        
        #ʕっʘ‿ʘʔっ("♥finalLogActions")
        self.model.stats.clear()
        self.perfectTokens = 0
        self.totalTokenEvaluations = 0

    def getStats(self, _logitSeq):
        with self.counsellor.infodump("getStats") as ʕっʘ‿ʘʔっ:
            #gradNorm = (sum((p.grad.norm(2)**2 for p in self.parameters() if p.grad is not None)))**0.5
            stats = {}
            if collectStats:
                stats, INN_cerebellum_str, INN_judgeBias_str, INN_credibilityBias_str,  windowVotes_str = self.model.interneuronNetwork.INN_getStats()
                stats["shortDecay"] = torch.sigmoid(self.memory.shortTermDecay)
                stats["longDecay"] = torch.sigmoid(self.memory.longTermDecay)
            else:
                stats = self.model.interneuronNetwork.INN_getStats()

            if _logitSeq:
                stats["logitMin"] = _logitSeq[-1].min(dim=-1).values.mean()
                stats["logitMax"] = _logitSeq[-1].max(dim=-1).values.mean()

            stats["scheduledSampling"] = self.scheduledSamplingProb

            return stats