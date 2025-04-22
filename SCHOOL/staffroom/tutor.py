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

def makeStatRecord():
    base = {
        "now": 0.0,
        "prev": 0.0,
        "top": float('-inf'),
        "bot": float('inf'),
        "delta": 0.0,
        "totSum": 0.0,
        "totNum": 0,
        "totAvg": 0.0
    }
    for n in [printFreq, printFreq*10, trainingLogFreq_A, trainingLogFreq_B]:
        base[f"/{n}"] = []

    return base

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

        self.ʕっෆ‿ෆʔっ = defaultdict(makeStatRecord)

        self.perfectTokens = 0
        self.totalTokenEvaluations = 0
        self.scheduledSamplingRate = scheduledSamplingRate
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
        self.rollingAveragesBufferLen = trainingLogFreq_B
        self.memoryLength = memoryLength
        self.cheekyAvgLoss = 0
        self.cheekyTotLoss = 0
        self.cheekyTotPrint = 1
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
            self.stepLossFloat = BACKWARDloss.detach().cpu().numpy().item()
            self.endTurnActions()
            if self.device.type == 'mps':
                ʕっʘ‿ʘʔっ("emptyCache (mps)")
                torch.mps.empty_cache()

            return self.predictedTokenIndices, self.logitSeq
        
    """this iterates through training data, performing forward passes, loss computation, backpropagation, and optimization for each step."""
    def trainModel(self, _trainingDataPairs, _epochs, _startIndex):
        self.startIndex = _startIndex
        self.collectAllTimeStats()
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
                    self.predictedTokenIndices, self.logitSeq = self.trainStep(_inputTokenIndices = inputTokenIndices, _targetTokenIndexSeq = targetTokenIndexSeq, _BACKWARDwobbleLoss = None, _repetitionPenalty = self.repetitionPenalty)
                    """ --- --- -*- BACKWARDS COMPLETE -*- --- --- -*- --- --- -*- --- --- -*- --- --- -*- --- --- -*- --- --- -*- --- --- -*- --- --- -*- --- --- -*- --- --- """
                    ʕっʘ‿ʘʔっ("♥collectTurnStats")
                    _, LOGstringStats, self.guessedTokenSeq = self.collectTurnStats(_targetTokenIndexSeq = targetTokenIndexSeq, _predictedTokenIndices = self.predictedTokenIndices)

                    if self.trainingStepCounter % saveModelFreq == 0:
                        ʕっʘ‿ʘʔっ("♥saveFreq")
                        self.saveFreqActions()

                    if self.trainingStepCounter % printFreq == 0:
                        ʕっʘ‿ʘʔっ("♥printFreq")
                        self.printFreqActions()
       
                    # Track loss every 100 steps
                    if self.trainingStepCounter % trainingLogFreq_A == 0:
                        ʕっʘ‿ʘʔっ("♥logFreq_A")
                        self.logFreqActions(_trainingDataPairs, _stringStats = LOGstringStats)

                    #if self.trainingStepCounter % trainingLogFreq_B == 0:
                        #ʕっʘ‿ʘʔっ("♥trainingLogFreq_B") # PRINTING LOGS TO TXT AND TERMINAL
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
                memIncrement = memoryLengthIncrement
                schedUpdate = random.choice([scheduledSamplingIncrement, -scheduledSamplingIncrement, scheduledSamplingIncrement])
                repeatUpdate = random.choice([repetitionPenaltyIncrement, -repetitionPenaltyIncrement, repetitionPenaltyIncrement])
                newLR = random.choice([LRIncrement, -LRIncrement])
                clipUpdate = random.choice([gradClipIncrement, -gradClipIncrement])
                tempUpdate = random.choice([tempIncrement, -tempIncrement])
                memUpdate = random.choice([memIncrement, -memIncrement])
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
            self.memoryLength = (max(min((self.memoryLength + memUpdate), 100), 1))

            if skipMemory:
                ʕっʘ‿ʘʔっ("♥skipMemory")
            else:
                ʕっʘ‿ʘʔっ("resetMemory")
                self.model.resetMemory(context="training")

        return inputTokenIndices, targetTokenIndexSeq, #self.repetitionPenalty, #self.wobbleLoss

    def endTurnActions(self):
        with self.counsellor.infodump("endTurnActions") as ʕっʘ‿ʘʔっ:
            ʕっʘ‿ʘʔっ("♥getLatestLossDelta")
            lossStats = self.ʕっෆ‿ෆʔっ.get("loss", {})

            rollA_avgKey = f"/BIG{trainingLogFreq_A}_avg"
            rollA_ΔKey = f"/{trainingLogFreq_A}_Δ"

            rollPrint_avgKey = f"/{printFreq}_avg"
            rollPrint_ΔKey = f"/{printFreq}_Δ"

            if rollA_avgKey in lossStats and rollA_ΔKey in lossStats:
                ʕっʘ‿ʘʔっ(f"♥usingLossStats{rollA_ΔKey} for latestLossDelta")
                self.latestLossDelta = lossStats[rollA_ΔKey]
            elif rollPrint_avgKey in lossStats and rollPrint_ΔKey in lossStats:
                ʕっʘ‿ʘʔっ(f"♥usingLossStats{rollPrint_ΔKey} for latestLossDelta")
                self.latestLossDelta = lossStats[rollPrint_ΔKey]
            else:
                ʕっʘ‿ʘʔっ(f"♥using 0.0 for latestLossDelta")
                self.latestLossDelta = 0.0
        
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
            self.cheekyTotPrint += 1
            self.cheekyTotLoss += self.stepLossFloat
            self.cheekyAvgLoss = self.cheekyTotLoss/self.cheekyTotPrint
            ʕっʘ‿ʘʔっ("calligraphist.S_colourPrintTraining")
            self.calligraphist.S_colourPrintTraining(
                _step = self.trainingStepCounter,
                _inputSeq = self.inputSeq,
                _guessedSeq_str = self.guessedTokenSeq,
                _targetSeq_str = self.targetSeq[:windowMAX],
                _recentLoss = self.ʕっෆ‿ෆʔっ.get("loss", {}).get(f"/{trainingLogFreq_A}_avg", 0), # self.stepLossFloat,
                _loss = self.stepLossFloat,
                _totalTokenCount = self.tokenCounts)

        
    def logFreqActions(self, _trainingDataPairs, _stringStats): # could also do 10x log freq??
        with self.counsellor.infodump("logFreqActions") as ʕっʘ‿ʘʔっ:
            self.stringStats = _stringStats
            #self.stats.update(self.ʕっෆ‿ෆʔっ) # SUSSY BUSSY !!!!!!!!!!!!!!!!!!!
            #fullStats = dict(self.stats)
            #fullStats.update(self.ʕっෆ‿ෆʔっ)
            self.cheekyAvgLoss = self.cheekyTotLoss/self.cheekyTotPrint

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
            self.calligraphist.refreshStatBands(_rollingAverages = self.ʕっෆ‿ෆʔっ) #self.rollingAverages)
            self.calligraphist.S_logTraining(
                _trainingLogPath = trainingLogPath_100,
                _trainingStepCounter = self.trainingStepCounter,
                _stats = self.stats,
                _freq = trainingLogFreq_A,
                _LR = self.learningRate,
                _INN_cerebellum_str = self.stringStats["INN_cerebellum_str"],
                _INN_judgeBias_str = self.stringStats["INN_judgeBias_str"],
                _INN_credbilityBias_str = self.stringStats["INN_credibilityBias_str"],
                _memoryGates_str = "",
                _topTokens_str = self.stringStats["topTokens"],
                _otherInfo_str = f"cheekyAvg: {self.cheekyAvgLoss} | {tokenPerfect_str} | {self.stringStats['windowVotes_str']} | {remainingData_str} | TUTOR.py {trainingLogFreq_A}")
            
            ʕっʘ‿ʘʔっ("finalLogActions")
            if debugPrints: # or True:
                for key in self.ʕっෆ‿ෆʔっ:
                    print(key, self.ʕっෆ‿ෆʔっ[key])
            self.stats.clear()
            self.stringStats.clear()
            self.tokenPerfectRate = 0
            self.sampledTokens = 0 
            self.perfectTokens = 0
            self.totalTokenEvaluations = 0
            self.cheekyAvgLoss = 0
            self.cheekyTotLoss = 0
            self.cheekyTotPrint = 0

    def collectTurnStats(self, _targetTokenIndexSeq, _predictedTokenIndices):
        with self.counsellor.infodump("collectTurnStats") as ʕっʘ‿ʘʔっ:
            ʕっʘ‿ʘʔっ("self.librarian.indexToToken.get(idx.item*())")
            lossStats = self.ʕっෆ‿ෆʔっ.get("loss", {})
            rollupA_key = f"/BIG{trainingLogFreq_A}"
            rollupA_avgKey = f"{rollupA_key}_avg"
            rollB_key = f"/{trainingLogFreq_B}"
            rollB_avgKey = f"{rollB_key}_avg"
            rollA_avgKey = f"/{trainingLogFreq_A}_avg"
            rollPrint_avgKey = f"/{printFreq}_avg"

            if rollupA_avgKey in lossStats and rollupA_key in lossStats and len(lossStats[rollupA_key]) > trainingLogFreq_A:
                if debugPrints: print(f"Using {rollupA_avgKey} for averageRecentLoss: {lossStats[rollupA_avgKey]}")
                self.averageRecentLoss = lossStats[rollupA_avgKey]
            elif rollB_avgKey in lossStats and rollB_key in lossStats and len(lossStats[rollB_key]) > trainingLogFreq_B:
                if debugPrints: print(f"Using {rollB_avgKey} for averageRecentLoss: {lossStats[rollB_avgKey]}")
                self.averageRecentLoss = lossStats[rollB_avgKey]
            elif rollA_avgKey in lossStats:
                if debugPrints: print(f"Using {rollA_avgKey} for averageRecentLoss: {lossStats[rollA_avgKey]}")
                self.averageRecentLoss = lossStats[rollA_avgKey]
            elif rollPrint_avgKey in lossStats:
                if debugPrints: print(f"Using {rollPrint_avgKey} for averageRecentLoss: {lossStats[rollPrint_avgKey]}")
                self.averageRecentLoss = lossStats[rollPrint_avgKey]
            else:
                if debugPrints: print(f"Using self.cheekyAvgLoss for averageRecentLoss: {self.cheekyAvgLoss}")
                self.averageRecentLoss = self.cheekyAvgLoss

            self.guessedTokenSeq = [self.librarian.indexToToken.get(idx.item(), "<UNK>") for idx in self.predictedTokenIndices]
            if self.guessedTokenSeq: 
                self.tokenCounts.update(self.guessedTokenSeq)

            ʕっʘ‿ʘʔっ("SCRIBE.maybeCommentOnGuess")
            if self.trainingStepCounter > trainingLogFreq_A:
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
                    self.stats["loss"] = self.stepLossFloat
                    self.stats["temperature"] = self.temperature
                    self.stats["lR"] = self.learningRate
                    self.stats["gradientClip"] = self.gradientClipMaxNorm
                    self.stats["latestLossDelta"] = self.latestLossDelta
                    self.stats["memoryLength"] = self.memoryLength
                    for statsK, statsV in self.stats.items():
                        if isinstance(statsV, torch.Tensor) and statsV.numel() == 1:
                            statsV = statsV.item()
                        if isinstance(statsV, (float, int)):
                            if statsK not in self.rollingAverages:
                                self.rollingAverages[statsK] = []
                            self.rollingAverages[statsK].append(statsV)
                            if len(self.rollingAverages[statsK]) > self.rollingAveragesBufferLen:
                                self.rollingAverages[statsK].pop(0)
 
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
                self.collectAllTimeStats()

        return self.stats, self.stringStats, self.guessedTokenSeq

    def collectAllTimeStats(self):
        for _statKey, _value in self.stats.items():
            if not isinstance(_value, (int, float)):
                if _statKey == "loss":
                    print(f"Loss value is : {_value}, loss value type is {type(_value)}")
                continue  # skip strings, tensors, weird stuff

            """ෆෆෆ^ ♥ KEYS ETC ♥ ^ෆෆෆ"""
            _               = self.ʕっෆ‿ෆʔっ[_statKey]  # this will autoinit with defaultdict
            ෆ‿ෆ             = self.ʕっෆ‿ෆʔっ[_statKey]
            important       = ["loss"]
            rolling         = ["loss", "gradNorm", "scheduledSamplingRate", "sampledTokens", "repetitionPenalty", "temperature"]
            percentiles     = [99.99, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0.01]

            """ ෆෆෆ^ ♥ UPDATE EVERY TURN ♥ ^ෆෆෆ   """
            """ ෆෆෆ^ ♥ turn stats ♥ ^ෆෆෆ  """
            ෆ‿ෆ["prev"]     = ෆ‿ෆ.get("now", 0.0)
            ෆ‿ෆ["now"]      = _value

            """ ෆෆෆ^ ♥ totals ♥ ^ෆෆෆ  """
            ෆ‿ෆ["totSum"]   = ෆ‿ෆ.get("totSum", 0.0)    + _value
            ෆ‿ෆ["totNum"]   = ෆ‿ෆ.get("totNum", 0)      + 1
            ෆ‿ෆ["totAvg"]   = ෆ‿ෆ["totSum"] / ෆ‿ෆ["totNum"]
            ෆ‿ෆ["totAvgΔ"]  = ෆ‿ෆ["now"]    - ෆ‿ෆ["totAvg"]

            """ ෆෆෆ^ ♥ records ♥ ^ෆෆෆ """
            ෆ‿ෆ["top"]      = max(ෆ‿ෆ.get("top", _value), _value)
            ෆ‿ෆ["bot"]      = min(ෆ‿ෆ.get("bot", _value), _value)

            """ ෆෆෆ^ ♥ ROLLING STATS ♥ ^ෆෆෆ   """
            if _statKey in rolling:
                for freq in [printFreq, trainingLogFreq_A]:
                    tag = f"/{freq}"
                    if tag not in ෆ‿ෆ:
                        ෆ‿ෆ[tag] = []
                    if len(ෆ‿ෆ[tag]) >= freq: 
                        ෆ‿ෆ[tag].pop(0)
                    ෆ‿ෆ[tag].append(_value)
                    if ෆ‿ෆ[tag]:
                        self.updateRollingStats(_ෆ‿ෆ = ෆ‿ෆ, _values = ෆ‿ෆ[tag], _freq = freq, _tag = tag, _percentiles = percentiles)

            if _statKey in important and self.trainingStepCounter % trainingLogFreq_A == 0:
                for importantFreq in [trainingLogFreq_A]:
                    importantTag = f"/BIG{importantFreq}"
                    if importantTag not in ෆ‿ෆ:
                        ෆ‿ෆ[importantTag] = []
                    if len(ෆ‿ෆ[importantTag]) >= trainingLogFreq_A:
                        ෆ‿ෆ[importantTag].pop(0)
                    ෆ‿ෆ[importantTag].append(_value)
                    if ෆ‿ෆ[importantTag]:
                        self.updateRollingStats(_ෆ‿ෆ = ෆ‿ෆ, _values = ෆ‿ෆ[importantTag], _freq = importantFreq, _tag = importantTag, _percentiles = percentiles)

    def updateRollingStats(self, _ෆ‿ෆ, _values, _freq, _tag, _percentiles = None):
        average                 = sum(_values) / len(_values)
        _ෆ‿ෆ[f"{_tag}_avg"]     = average

        standardDeviation       = self.stdTest(_values)
        _ෆ‿ෆ[f"{_tag}_std"]     = standardDeviation

        top                     = max(_values)
        _ෆ‿ෆ[f"{_tag}_top"]     = max(top, _ෆ‿ෆ.get(f"{_tag}_top", top))

        bottom                  = min(_values)
        _ෆ‿ෆ[f"{_tag}_bot"]     = min(bottom, _ෆ‿ෆ.get(f"{_tag}_bot", bottom))

        delta                   = _ෆ‿ෆ["now"] - average
        _ෆ‿ෆ[f"{_tag}_Δ"]       = delta

        if _percentiles:
            for p in _percentiles:
                _ෆ‿ෆ[f"{_tag}_p{p}"]  = np.percentile(_values, p)

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


