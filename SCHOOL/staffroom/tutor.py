# CHARIS CAT 2025 
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# MULTI-TOKEN AUTOREGRESSIVE TRAINING MODULE 
# SCHOOL/staffroom/tutor.py

import random, sys
from collections import Counter, defaultdict
from datetime import datetime
import torch
from config import *
import numpy as np
import math
from SCHOOL.staffroom.newsletter import deep_model_summary, STATS

class TUTOR:
    def __init__(self, _counsellor, _calligraphist, _scribe, _librarian, _newsletter, _wobble, _model, _device = modelDevice, _gradientClipMaxNorm = gradientClipMaxNorm, _temperature = temperature, _numTokensPerStep = numTokensPerStep):
        
        self.counsellor = _counsellor
        self.calligraphist = _calligraphist
        self.scribe = _scribe
        self.librarian = _librarian
        self.wobble = _wobble
        self.device = _device
        self.model = _model
        self.newsletter = STATS()
        
        def makeStatRecord():
            return {
                "now": 0.0,
                "prev": 0.0,
                "top": float('-inf'),
                "bot": float('inf'),
                "delta": 0.0,
                "totSum": 0.0,
                "totNum": 0,
                "totAvg": 0.0,
                f"roll{100}": [],
                "rollAvg": 0.0
            }

        self.ʕっෆ‿ෆʔっ = defaultdict(makeStatRecord)

        self.perfectTokens = 0
        self.totalTokenEvaluations = 0
        self.scheduledSamplingRate = scheduledSamplingRate
        self.recentLosses = []
        self.predictedTokenIndices = [] # this list grows each time a new token is predicted
        self.averageRecentLoss = 0
        self.repetitionPenalty = repetitionPenalty
        self.stats = {}
        self.stringStats = {}
        self.trainingStepCounter = 1
        self.gradientClipMaxNorm = _gradientClipMaxNorm
        self.temperature = _temperature
        self.numTokensPerStep = _numTokensPerStep
        self.learningRate = learningRate
        self.rollingAverages = defaultdict(list)
        self.rollingAveragesBufferLen = trainingLogFreq_1000
        #model.to(self.device)

    def trainStep(self, _inputTokenIndices, _targetTokenIndexSeq, _BACKWARDwobbleLoss, _repetitionPenalty):
        with self.counsellor.infodump("trainStep") as ʕっʘ‿ʘʔっ:
            ʕっʘ‿ʘʔっ("_model.optimizer.zero_grad")
            self.repetitionPenalty = _repetitionPenalty
            self.model.optimizer.zero_grad() # clears gradients last step - needed before any backward
            self.trainingStepCounter += 1
            self.predictedTokenIndices = []
            inputSeqPredictions = list(_inputTokenIndices)  # Start with input context, create a COPY!
            buffer = torch.zeros(windowMAX, dtype=torch.long, device=self.device) # creates buffer/step instead of recreating tensors inside loop
            buffer[:len(inputSeqPredictions)] = torch.as_tensor(inputSeqPredictions, device=self.device)
            self.logitSeq = [] # raw output of each prediction
            cumulativeLoss = torch.tensor(0.0, device=self.device) # sum of token losses for THIS sequence - averaged at the end

            for j in range(numTokensPerStep): # Predict multiple tokens in a sequence, one at a time
                ʕっʘ‿ʘʔっ("FORWARD")
                inputTensor = buffer[:len(inputSeqPredictions)] # slices input to only keep relevant part
                try:
                    if forwardProfiler: 
                        with torch.profiler.profile(record_shapes=True) as prof:
                            logits = self.model.forward(inputTensor)
                    else:
                        logits = self.model.forward(inputTensor)
                except RuntimeError as e:
                    print("TUTOR.trainStep.forward failed!", e)
                    return [], []
                
                if forwardProfiler: print(prof.key_averages().table())

                ʕっʘ‿ʘʔっ("getResponseFromLogits")
                predictedTokenIndex = self.model.getResponseFromLogits(logits, _repetitionPenalty = repetitionPenalty)

                ʕっʘ‿ʘʔっ("inputSeqPredictions")
                self.predictedTokenIndices.append(predictedTokenIndex) # tensor shape [1]
                nextTokenInput = (
                    predictedTokenIndex.item() if scheduledSampling and random.random() < self.scheduledSamplingRate
                    else _targetTokenIndexSeq[j] if j < len(_targetTokenIndexSeq)
                    else predictedTokenIndex.item()
                )

                sampledTokens = scheduledSampling and random.random() < self.scheduledSamplingRate
                if sampledTokens:
                    self.stats['sampledTokens'] = self.stats.get('sampledTokens', 0) + 1

                nextTokenInput = (predictedTokenIndex.item() if sampledTokens # .ITEM() REQUIRED!! FOR APPENDING ONLY ONE TOKEN (grids?)
                    else _targetTokenIndexSeq[j] if j < len(_targetTokenIndexSeq)
                    else predictedTokenIndex.item() # .ITEM() REQUIRED!! FOR APPENDING ONLY ONE TOKEN (grids?)
                )
                inputSeqPredictions.append(nextTokenInput) # multi-token autoregressive generation: append next token to your current input — becomes the prompt for the next token

                ʕっʘ‿ʘʔっ("loop through tokens for this step")
                if j < len(_targetTokenIndexSeq):
                    ʕっʘ‿ʘʔっ("totalTokenCounter")
                    self.totalTokenEvaluations += 1

                    ʕっʘ‿ʘʔっ("computeLoss")
                    stepLoss = self.model.computeLoss(logits, _targetTokenIndexSeq[j])

                    ʕっʘ‿ʘʔっ("appendStepLoss")
                    cumulativeLoss += stepLoss

            ʕっʘ‿ʘʔっ("backward")
            BACKWARDloss = cumulativeLoss / len(_targetTokenIndexSeq) if len(_targetTokenIndexSeq) > 0 else torch.tensor(0.0, device=self.device)
            #BACKWARDloss_ = (0.025*self.BACKWARDwobbleLoss)+(0.975*BACKWARDloss)
            if windowEntropyBonus:
                if hasattr(self.model.interneuronNetwork, "entropyBonus"):
                    ʕっʘ‿ʘʔっ("entropyBonus")
                    BACKWARDloss = BACKWARDloss - (self.model.interneuronNetwork.entropyBonus * 0.05)
            if not torch.isfinite(BACKWARDloss): 
                print("TUTOR.trainStep.backward !!! Loss is NaN or Inf:", BACKWARDloss)
                return
            else: 
                if debugPrints: print("TUTOR.trainStep.backward - loss is not NaN or Inf:", BACKWARDloss)
                
            try:
                if profiler: 
                    with torch.profiler.profile(record_shapes=True) as prof:
                        self.model.backward(BACKWARDloss)
                elif mpsProfiler: 
                    with torch.mps.profiler.profile(mode='interval', wait_until_completed=False) as prof:
                        self.model.backward(BACKWARDloss)
                else:
                    self.model.backward(BACKWARDloss)
            except RuntimeError as e:
                print("TUTOR.trainStep.backward failed!", e)
                return [], []

            if profiler: print(prof.key_averages().table())
            
            ʕっʘ‿ʘʔっ("clip_grad_norm")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.gradientClipMaxNorm)
            self.model.optimizer.step()

            ʕっʘ‿ʘʔっ("actions after looping")
            self.stepLossFloat = BACKWARDloss.detach().cpu().numpy()
            self.endTurnActions()
            if self.device.type == 'mps':
                ʕっʘ‿ʘʔっ("emptyCache (mps)")
                torch.mps.empty_cache()

            return self.predictedTokenIndices, self.logitSeq, BACKWARDloss
        
    """this iterates through training data, performing forward passes, loss computation, backpropagation, and optimization for each step."""
    def trainModel(self, _trainingDataPairs, _epochs, _startIndex):
        self.startIndex = _startIndex
        with self.counsellor.infodump("trainModel") as ʕっʘ‿ʘʔっ:
            #if debugPrints: print(f"Debug tokenToIndex (First 20): {list(librarian.tokenToIndex.items())[:20]}")
            for name, param in self.model.named_parameters(): print(name, param.device)
            ʕっʘ‿ʘʔっ("COUNTERS INIT")
            self.trainingStepCounter = 0
            self.stats = Counter({"loss": 0, "gradNorm": 0, "logitMin": 0, "logitMax": 0, "tokenCount": 0})
            self.tokenCounts = Counter()
            self.latestLossDelta = None

            ʕっʘ‿ʘʔっ("back to school!")
            print("babyLLM is heading back to school...")

            """EPOCH LOOP"""
            ʕっʘ‿ʘʔっ("epoch♥")
            for epoch in range(_epochs):
                print(f"--- lesson {epoch+1}/{_epochs} started ---")
                """TRAINING DATA (batches)"""
                for i, (_inputSeq, _targetSeq) in enumerate(_trainingDataPairs):
                    ʕっʘ‿ʘʔっ("♥BEFORE TRAINING STEP")
                    inputTokenIndices, targetTokenIndexSeq = self.startTurnActions(_inputSeq = _inputSeq, _targetSeq = _targetSeq, _lastTurnLossDelta = self.latestLossDelta)
                    ʕっʘ‿ʘʔっ("♥TRAINING STEP")
                    self.predictedTokenIndices, self.logitSeq, BACKWARDloss_ = self.trainStep(_inputTokenIndices = inputTokenIndices, _targetTokenIndexSeq = targetTokenIndexSeq, _BACKWARDwobbleLoss = None, _repetitionPenalty = self.repetitionPenalty)

                    ʕっʘ‿ʘʔっ("♥collectTurnStats")
                    LOGstats, LOGstringStats, self.guessedTokenSeq = self.collectTurnStats(_targetTokenIndexSeq = targetTokenIndexSeq, _predictedTokenIndices = self.predictedTokenIndices, _BACKWARDloss = BACKWARDloss_)

                    if self.trainingStepCounter % saveModelFreq == 0:
                        ʕっʘ‿ʘʔっ("♥saveFreq")
                        self.saveFreqActions()

                    if self.trainingStepCounter % printFreq == 0:
                        ʕっʘ‿ʘʔっ("♥printFreq")
                        self.printFreqActions()
       
                    # Track loss every 100 steps
                    if self.trainingStepCounter % trainingLogFreq_100 == 0:
                        ʕっʘ‿ʘʔっ("♥logFreq_100")
                        self.logFreqActions(_trainingDataPairs, _stats = LOGstats, _stringStats = LOGstringStats)

                    #if self.trainingStepCounter % trainingLogFreq_1000 == 0:
                        #ʕっʘ‿ʘʔっ("♥trainingLogFreq_1000") # PRINTING LOGS TO TXT AND TERMINAL
                        #deep_model_summary(self.model, tracker = self.newsletter, step=self.trainingStepCounter, loss = self.stepLossFloat, calligraphist=self.calligraphist)
                        #self.logFreqActions(_trainingDataPairs)

                    ʕっʘ‿ʘʔっ("♥END TURN♥") # END OF ONE TURN
                    self.latestLossDelta = self.endTurnActions()
                    # < indent (5)
                ʕっʘ‿ʘʔっ("♥finalSaveBeforeNewEpoch")
                self.model.saveModel(_newStartIndex = self.startIndex, _trainingStepCounter = self.trainingStepCounter)
        print("--- tutoring complete! ---")
        return

    def startTurnActions(self, _inputSeq, _targetSeq, _lastTurnLossDelta):
        with self.counsellor.infodump("startTurnActions") as ʕっʘ‿ʘʔっ:
            self.lastTurnLossDelta = _lastTurnLossDelta
            inputTokenIndices = [self.librarian.tokenToIndex.get(t, self.librarian.tokenToIndex["<UNK>"]) for t in _inputSeq]
            targetTokenIndexSeq = [self.librarian.tokenToIndex.get(t, self.librarian.tokenToIndex["<UNK>"]) for t in _targetSeq]
            self.inputSeq = _inputSeq
            self.targetSeq = _targetSeq

            if self.stats["windowEntropy"]:
                self.winEnt = self.stats["windowEntropy"]
            else:
                self.winEnt = 0

            #values = np.array([
            #    self.lastTurnLossDelta, 
            #    self.recentLosses, 
            #    self.perfectTokens, 
            #    self.winEnt
            #], dtype = np.float32)

            #wobbleInputStats_ = torch.tensor(values, device = modelDevice)
            #wobbleLoss, wobbleIncrements, wobbleStats = self.wobble.forward(_wobbleInputStats = wobbleInputStats_, _lastTurnLossDelta = self.lastTurnLossDelta) # CALL WOBBLE.forward HERE!
            #self.wobbleLoss = wobbleLoss
            #self.wobbleIncrements = wobbleIncrements

            ʕっʘ‿ʘʔっ("♥incrementCounters")
            if skipWobble:
                schedIncrement = scheduledSamplingIncrement
                repeatIncrement = repetitionPenaltyIncrement
                LRIncrement = self.learningRate/100000
                clipIncrement = gradClipIncrement
                tempIncrement = temperatureIncrement
                schedUpdate = random.choice([scheduledSamplingIncrement, -scheduledSamplingIncrement, scheduledSamplingIncrement])
                repeatUpdate = random.choice([repetitionPenaltyIncrement, -repetitionPenaltyIncrement, repetitionPenaltyIncrement])
                newLR = random.choice([LRIncrement, -LRIncrement])
                clipUpdate = random.choice([gradClipIncrement, -gradClipIncrement])
                tempUpdate = random.choice([tempIncrement, -tempIncrement])
            elif self.wobbleIncrements is not None:
                schedIncrement = self.wobbleIncrements[0].item()
                repeatIncrement = self.wobbleIncrements[1].item()
                LRIncrement = (self.learningRate + self.wobbleIncrements[2].item())
                clipIncrement = (self.wobbleIncrements[4]).item()
                tempIncrement = (temperature + self.wobbleIncrements[3].item())
                schedUpdate = random.choice([(max(min((schedIncrement), 0.0001), -0.0001)), -1])
                repeatUpdate = random.choice([(max(min((repeatIncrement), 0.0001), -0.0001)), -1])
                newLR = (max(min((random.choice([(max(min((LRIncrement), 0.000000001), -0.000000001)), -0.00000001])), 0.00040), 0.00030))
                clipUpdate = random.choice([((max(min((clipIncrement), 0.0001)), -0.0001)), -1])
                tempUpdate = random.choice([(max(min((tempIncrement), 0.0001)), -0.0001), -0.01])

            self.scheduledSamplingRate = (max(min((self.scheduledSamplingRate + schedUpdate), maxSchedSamp), minSchedSamp))
            self.repetitionPenalty = (max(min((self.repetitionPenalty + repeatUpdate), maxRepPen), minRepPen))
            self.model.setLearningRate(newLR)
            self.gradientClipMaxNorm = (max(min((self.gradientClipMaxNorm + clipUpdate), maxGradClip), minGradClip))
            self.temperature = (max(min((self.temperature + tempUpdate), maxTemp), minTemp))

            if skipMemory:
                ʕっʘ‿ʘʔっ("♥skipMemory")
            else:
                ʕっʘ‿ʘʔっ("resetMemory")
                self.model.resetMemory(context="training")

        return inputTokenIndices, targetTokenIndexSeq, #self.repetitionPenalty, #self.wobbleLoss
    
    def collectTurnStats(self, _targetTokenIndexSeq, _predictedTokenIndices, _BACKWARDloss):
        with self.counsellor.infodump("collectTurnStats") as ʕっʘ‿ʘʔっ:
            ʕっʘ‿ʘʔっ("self.librarian.indexToToken.get(idx.item*())")
            self.averageRecentLoss = sum(self.recentLosses) / len(self.recentLosses) if self.recentLosses else 0.0
            self.guessedTokenSeq = [self.librarian.indexToToken.get(idx.item(), "<UNK>") for idx in self.predictedTokenIndices]
            if self.guessedTokenSeq: 
                self.tokenCounts.update(self.guessedTokenSeq)

            ʕっʘ‿ʘʔっ("SCRIBE.maybeCommentOnGuess")
            self.scribe.maybeCommentOnGuess(self.guessedTokenSeq, self.stepLossFloat, "scribe", 0.00075)

            ʕっʘ‿ʘʔっ("collectStats♥")

            if collectStats:
                ʕっʘ‿ʘʔっ("♥if collectStats♥")

                if token_collectStats:
                    ʕっʘ‿ʘʔっ("♥if token_collectStats♥")
                    self.predictedTokenIndices = _predictedTokenIndices

                    ʕっʘ‿ʘʔっ("♥most common tokens")
                    topTokens = ""
                    topTokens = self.tokenCounts.most_common(10)

                    ʕっʘ‿ʘʔっ("♥calculate perfect tokens")
                    if not _predictedTokenIndices:
                        return self.stats, [], self.guessedTokenSeq
                    target = torch.tensor(_targetTokenIndexSeq[:numTokensPerStep], device=modelDevice)
                    predicted = torch.tensor(self.predictedTokenIndices, device=modelDevice)
                    correct = (predicted == target).sum() # ~~~ if predicted = target, over whole tensor 
                    self.perfectTokens += correct
                    self.totalTokenEvaluations += len(target)

                if static_collectStats:
                    ʕっʘ‿ʘʔっ("♥if static_collectStats")
                    self.stats["scheduledSamplingRate"] = self.scheduledSamplingRate
                    self.stats["repetitionPenalty"] = self.repetitionPenalty
                    self.stats["AvgLoss"] = self.averageRecentLoss
                    self.stats["loss"] = _BACKWARDloss
                    self.stats["temperature"] = self.temperature
                    self.stats["lR"] = self.learningRate
                    self.stats["gradientClip"] = self.gradientClipMaxNorm
                    self.stats["latestLossDelta"] = self.latestLossDelta
                    for statsK, statsV in self.stats.items():
                        if isinstance(statsV, torch.Tensor) and statsV.numel() == 1:
                            statsV = statsV.item()
                        if isinstance(statsV, (float, int)):
                            if statsK not in self.rollingAverages:
                                self.rollingAverages[statsK] = []
                            self.rollingAverages[statsK].append(statsV)
                            if len(self.rollingAverages[statsK]) > self.rollingAveragesBufferLen:
                                self.rollingAverages[statsK].pop(0)

                self.collectAllTimeStats()
 
                if embed_collectStats:
                    ʕっʘ‿ʘʔっ("♥if embed_collectStats")
                    self.stats.update(self.model.embed.getEmbedStats())

                if logit_collectStats:
                    ʕっʘ‿ʘʔっ("♥if logit_collectStats♥")
                    self.stats.update(self.model.logits.getLogitStats())
                    self.stats["logitSeq"] = self.logitSeq
                    if self.stats["logitSeq"]:
                        ʕっʘ‿ʘʔっ("♥logit max & min")
                        self.stats["logitMin"] = self.stats["logitSeq"][-1].min(dim=-1).values.mean()
                        self.stats["logitMax"] = self.stats["logitSeq"][-1].max(dim=-1).values.mean()

                #self.stats.update(self.wobble.getWobbleStats())

                if skipMemory:
                    ʕっʘ‿ʘʔっ("♥skipMemory")
                else:
                    self.model.memory.updateMemoryBuffers()
                    if memory_collectStats:
                        ʕっʘ‿ʘʔっ("♥if memory_collectStats")
                        self.stats.update(self.model.memory.getMemoryStats())

                ʕっʘ‿ʘʔっ("♥INN_collectStats")
                INN_stats, INN_cerebellum_str, INN_judgeBias_str, INN_credibilityBias_str, windowVotes_str = self.model.interneuronNetwork.INN_getStats()
                self.stats.update(INN_stats)
                INN_stringStats = {"INN_cerebellum_str": str(INN_cerebellum_str), "INN_judgeBias_str": str(INN_judgeBias_str), "INN_credibilityBias_str": str(INN_credibilityBias_str), "windowVotes_str": str(windowVotes_str)}
                self.stringStats.update(INN_stringStats)
                self.stringStats.update({"topTokens": str(topTokens)})

        return self.stats, self.stringStats, self.guessedTokenSeq

    def endTurnActions(self):
        with self.counsellor.infodump("endTurnActions") as ʕっʘ‿ʘʔっ:
            #self.lastTurnLossDelta = 0

            ʕっʘ‿ʘʔっ("increment counters")
            self.recentLosses.append(self.stepLossFloat)
            if len(self.recentLosses) > trainingLogFreq_100:
                self.recentLosses.pop(0)

            ʕっʘ‿ʘʔっ("♥calculateLossDelta")
            if self.recentLosses: self.latestLossDelta = self.stepLossFloat - (sum(self.recentLosses) / len(self.recentLosses))
            else: self.latestLossDelta = 0.0
            #self.scheduledSamplingRate = self.scheduledSamplingRate + scheduledSamplingIncrement
            #self.repetitionPenalty = self.repetitionPenalty - repetitionPenaltyIncrement
        
        return self.latestLossDelta

    def saveFreqActions(self): 
        with self.counsellor.infodump("saveFreqActions") as ʕっʘ‿ʘʔっ: # SAVE THE MODEL EVERY x STEPS
            print(self.calligraphist.S_apply('dim', 'autosaving...') + self.calligraphist.S_apply('reset', ''))
            self.model.saveModel(_newStartIndex = self.startIndex, _trainingStepCounter = self.trainingStepCounter)
            p = self.trainingStepCounter + saveModelFreq
            print(self.calligraphist.S_apply('dim', f"autosave successful! saving every {saveModelFreq} steps, the next autosave will be at step {p}...") + self.calligraphist.S_apply('reset', ''))
                    
    def printFreqActions(self): 
        with self.counsellor.infodump("printFreqActions") as ʕっʘ‿ʘʔっ: # PRINTING TRAINING OUTPUT TO TERMINAL
            #recentLoss = sum(self.recentPrintLosses)/len(self.recentPrintLosses) if self.recentPrintLosses else None
            ʕっʘ‿ʘʔっ("calligraphist.S_colourPrintTraining")
            self.calligraphist.S_colourPrintTraining(
                _step = self.trainingStepCounter,
                _inputSeq = self.inputSeq,
                _guessedSeq_str = self.guessedTokenSeq,
                _targetSeq_str = self.targetSeq[:windowMAX],
                _recentLoss = self.averageRecentLoss,
                _loss = self.stepLossFloat,
                _totalTokenCount = self.tokenCounts)
        
    def logFreqActions(self, _trainingDataPairs, _stats, _stringStats): # could also do 10x log freq??
        with self.counsellor.infodump("logFreqActions") as ʕっʘ‿ʘʔっ:
            self.stringStats = _stringStats
            self.stats = _stats
            self.stats.update(self.ʕっෆ‿ෆʔっ)

            ʕっʘ‿ʘʔっ("calculateTrainingDataRemaining")
            trainingDataRemaining = len(_trainingDataPairs) - self.trainingStepCounter
            trainingDataPercent = (trainingDataRemaining / len(_trainingDataPairs)) * 100
            remainingData_str = f"remainingTokens: {len(_trainingDataPairs) - self.trainingStepCounter} ({trainingDataPercent:.2f}%)"

            tokenPerfect_str = ""
            if self.totalTokenEvaluations > 0:
                self.tokenPerfectRate = (self.perfectTokens / self.totalTokenEvaluations) * 100
                statType = self.calligraphist.S_getStat("PT%", self.tokenPerfectRate)
                styledRate = self.calligraphist.S_apply(statType, f"{self.tokenPerfectRate:.2f}%")
                tokenPerfect_str = (f"{self.calligraphist.S_apply('dim', f'perfectTokens: {self.perfectTokens} / {self.totalTokenEvaluations}')} → {styledRate}")

            ʕっʘ‿ʘʔっ("calligraphist.S_logTraining")
            self.calligraphist.S_logTraining(
                _trainingLogPath = trainingLogPath_100,
                _trainingStepCounter = self.trainingStepCounter,
                _stats = self.stats,
                _freq = trainingLogFreq_100,
                _LR = self.learningRate,
                _INN_cerebellum_str = self.stringStats["INN_cerebellum_str"],
                _INN_judgeBias_str = self.stringStats["INN_judgeBias_str"],
                _INN_credbilityBias_str = self.stringStats["INN_credibilityBias_str"],
                _memoryGates_str = "",
                _topTokens_str = self.stringStats["topTokens"],
                _otherInfo_str = f"{tokenPerfect_str} | {self.stringStats['windowVotes_str']} | {remainingData_str} | TUTOR.py {trainingLogFreq_100}")
            
            ʕっʘ‿ʘʔっ("finalLogActions")
            self.calligraphist.refreshStatBands(_rollingAverages = self.rollingAverages)
            self.stats.clear()
            self.stringStats.clear()
            self.tokenPerfectRate = 0
            self.perfectTokens = 0
            self.totalTokenEvaluations = 0

    def collectAllTimeStats(self):
        for key, value in self.stats.items():
            if not isinstance(value, (int, float)):
                continue

            if key not in self.ʕっෆ‿ෆʔっ:
                self.ʕっෆ‿ෆʔっ[key] = value
                self.ʕっෆ‿ෆʔっ[f"{key}_avg"] = value
                self.ʕっෆ‿ෆʔっ[f"{key}_top"] = value
                self.ʕっෆ‿ෆʔっ[f"{key}_bot"] = value
                self.ʕっෆ‿ෆʔっ[f"{key}_count"] = 1
                self.ʕっෆ‿ෆʔっ[f"{key}_sum"] = value
                self.ʕっෆ‿ෆʔっ[f"{key}_roll"] = value
            else:
                if value > self.ʕっෆ‿ෆʔっ[f"{key}_top"]:
                    self.ʕっෆ‿ෆʔっ[f"{key}_top"] = value
                if value < self.ʕっෆ‿ෆʔっ[f"{key}_bot"]:
                    self.ʕっෆ‿ෆʔっ[f"{key}_bot"] = value

                self.ʕっෆ‿ෆʔっ[f"{key}_count"] += 1
                self.ʕっෆ‿ෆʔっ[f"{key}_sum"] += value
                count = self.ʕっෆ‿ෆʔっ[f"{key}_count"]
                total = self.ʕっෆ‿ෆʔっ[f"{key}_sum"]
                self.ʕっෆ‿ෆʔっ[f"{key}_avg"] = total / count

    def collectAllTimeStats(self):
        for key, value in self.stats.items():
            if not isinstance(value, (int, float)): continue  # skip strings, tensors, weird stuff
            """KEYS ETC"""
            ෆ‿ෆ                     = self.ʕっෆ‿ෆʔっ[key]
            ෆ1                      = printFreq
            ෆ10                     = ෆ1 * 10
            ෆ100                    = trainingLogFreq_100
            ෆ1000                   = trainingLogFreq_1000


            """VERY CURRENT"""
            ෆ‿ෆ["prev"]             = ෆ‿ෆ["now"]
            ෆ‿ෆ["now"]              = value

            ෆ‿ෆ[f"roll{ෆ1}Avg"]     = sum(ෆ‿ෆ[f"roll{ෆ1}"]) / len(ෆ‿ෆ[f"roll{ෆ1}"]) if ෆ‿ෆ[f"roll{ෆ1}"] else 0.0
            ෆ‿ෆ[f"roll{ෆ1}△"]           = ෆ‿ෆ["prev"] - ෆ‿ෆ["now"] if ෆ‿ෆ["prev"] else 0.0
            ෆ‿ෆ[f"roll{ෆ1}std"]         = self.stdTest(ෆ‿ෆ[f"roll{ෆ1}"])


            """ROLLING 10"""
            ෆ‿ෆ[f"roll{ෆ10}"].append(value) if len(ෆ‿ෆ[f"roll{ෆ10}"]) > ෆ10: ෆ‿ෆ[f"roll{ෆ10}"].pop(0)

            ෆ‿ෆ[f"roll{ෆ10}Avg"]    = sum(ෆ‿ෆ[f"roll{ෆ10}"]) / len(ෆ‿ෆ[f"roll{ෆ10}"]) if ෆ‿ෆ[f"roll{ෆ10}"] else 0.0
            ෆ‿ෆ[f"roll{ෆ10}△"]      = ෆ‿ෆ["now"] - ෆ‿ෆ[f"roll{ෆ10}Avg"] if ෆ‿ෆ[f"roll{ෆ10}Avg"] else 0.0
            ෆ‿ෆ[f"roll{ෆ10}Std"]    = self.stdTest(ෆ‿ෆ[f"roll{ෆ10}"])


            """ROLLING 100""" # self.recentLosses.append(self.stepLossFloat) if len(self.recentLosses) > trainingLogFreq_100: self.recentLosses.pop(0)
            ෆ‿ෆ[f"roll{ෆ100}"].append(value) if len(ෆ‿ෆ[f"roll{ෆ100}"]) > ෆ100: ෆ‿ෆ[f"roll{ෆ100}"].pop(0)

            ෆ‿ෆ[f"roll{ෆ100}Avg"]   = sum(ෆ‿ෆ[f"roll{ෆ100}"]) / len(ෆ‿ෆ[f"roll{ෆ100}"]) if ෆ‿ෆ[f"roll{ෆ100}"] else 0.0
            ෆ‿ෆ[f"roll{ෆ100}△"]     = ෆ‿ෆ["now"] - ෆ‿ෆ[f"roll{ෆ100}Avg"] if ෆ‿ෆ[f"roll{ෆ100}Avg"] else 0.0 
            ෆ‿ෆ[f"roll{ෆ100}Std"]   = self.stdTest(ෆ‿ෆ[f"roll{ෆ100}"])


            """TOTALS"""
            ෆ‿ෆ["totSum"]          += value
            ෆ‿ෆ["totNum"]          += 1
            ෆ‿ෆ["totAvg"]           = ෆ‿ෆ["totSum"] / ෆ‿ෆ["totNum"]
            ෆ‿ෆ["totAvg△"]          = ෆ‿ෆ["now"] - ෆ‿ෆ["totAvg"] if ෆ‿ෆ["totAvg"] else 0.0
            ෆ‿ෆ[f"totAvgStd"]       = self.stdTest(ෆ‿ෆ[f"roll{ෆ100}"])


            """RECORDS"""
            ෆ‿ෆ["top"]              = max(ෆ‿ෆ["top"], value)
            ෆ‿ෆ["bot"]              = min(ෆ‿ෆ["bot"], value)

    def stdTest(self, values):
        if len(values) <= 1: return 0.0
        avg = sum(values) / len(values)
        variance = sum((x - avg)**2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def modelSummary(self, printLongValues=True, maxShape=10000):
        print("--- MODEL SUMMARY ---")

        for name, module in self.model.named_modules():
            for attr in dir(module):
                if attr.startswith("_"): continue  # skip private attrs
                try:
                    value = getattr(module, attr)
                    full_name = f"{name}.{attr}" if name else attr

                    if isinstance(value, torch.nn.Parameter):
                        data = value.data
                        shape = tuple(data.shape)
                        numel = data.numel()
                        norm = data.norm().item()
                        mean = data.mean().item()
                        std = data.std().item()
                        sparsity = 1 - (data.count_nonzero().item() / numel)

                        summary = f"{full_name:<40} | shape: {shape:<20} | norm: {norm:.4f} | sparsity: {sparsity:.2%} | mean: {mean:.4f} | std: {std:.4f}"
                        print(summary)
                        if printLongValues and numel < maxShape:
                            print("→", data)

                    elif isinstance(value, torch.Tensor) and value.numel() > 0:
                        shape = tuple(value.shape)
                        if value.numel() > maxShape:
                            print(f"{full_name:<40} | tensor shape: {shape} (too big to print)")
                        else:
                            mean = value.mean().item()
                            std = value.std().item()
                            print(f"{full_name:<40} | tensor shape: {shape} | mean: {mean:.4f} | std: {std:.4f}")
                            if printLongValues:
                                print("→", value)

                    elif isinstance(value, (float, int)):
                        print(f"{full_name:<40} = {value}")

                except Exception as e:
                    pass  # some attributes throw when accessed, ignore


