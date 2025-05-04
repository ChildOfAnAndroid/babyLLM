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
        base[f"{n}"] = []

    return base

class TUTOR:
    def __init__(self, _counsellor, _calligraphist, _scribe, _librarian, _model, 
                _device                 = modelDevice,              
                _logGradClip            = 0, 
                _temperature            = 0.8,                  
                _repetitionPenalty      = 0, 
                _scheduledSamplingRate  = 0,            
                _memoryLength           = 0, 
                _numTokensPerStep       = numTokensPerStep,
                _logRepetitionWindow    = 0):
        
        # send stats from babyllm.py
        self.temperature                = _temperature
        self.repetitionPenalty          = _repetitionPenalty
        self.scheduledSamplingRate      = _scheduledSamplingRate
        self.logGradClip                = _logGradClip
        self.gradientClipMaxNorm        = 0
        self.memoryLength               = _memoryLength
        self.logRepetitionWindow        = _logRepetitionWindow
        
        self.counsellor                 = _counsellor
        self.calligraphist              = _calligraphist
        self.scribe                     = _scribe
        self.librarian                  = _librarian
        self.device                     = _device
        self.model                      = _model

        self.ʕっෆ‿ෆʔっ                  = defaultdict(makeStatRecord)

        self.perfectTokens              = 0
        self.totalTokenEvaluations      = 0
        self.predictedTokenIndices      = [] # this list grows each time a new token is predicted
        self.averageRecentLoss          = 0
        self.stats                      = {}
        self.stringStats                = {}
        self.trainingStepCounter        = 1
        self.numTokensPerStep           = _numTokensPerStep
        self.learningRate               = learningRate
        self.stepLossFloat              = 0
        self.aaa = 0
        self.bbb = 0
        self.ccc = 0
        self.ppp = 0
        self.nnn = 0
        self.aaaa = 0
        self.dddd = 0
        self.bbbb = 0
        self.nnnn = 0
        #model.to(self.device)
        
    """this iterates through training data, performing forward passes, loss computation, backpropagation, and optimization for each step."""
    def trainModel(self, _trainingDataPairs, _epochs, _startIndex):
        self.startIndex = _startIndex
        self.collectAllTimeStats()
        with self.counsellor.infodump("trainModel") as ʕっʘ‿ʘʔっ:
            #if debugPrints: print(f"Debug tokenToIndex (First 20): {list(librarian.tokenToIndex.items())[:20]}")
            for name, param in self.model.named_parameters(): print(name, param.device)
            ʕっʘ‿ʘʔっ("COUNTERS INIT")
            self.trainingStepCounter = 1
            self.stats = Counter({"loss": 0, "gradNorm": 0, "logitMin": 0, "logitMax": 0, "tokenCount": 0})
            self.tokenCounts = Counter()
            self.latestLossDelta = 0
            self.reflectionTrainingPairs = []
            self.reflectionFreq = reflectionFreq

            ʕっʘ‿ʘʔっ("back to school!")
            print("babyLLM is heading back to school...")

            """EPOCH LOOP"""
            ʕっʘ‿ʘʔっ("epoch♥")
            for epoch in range(_epochs):
                print(f"--- lesson {epoch+1}/{_epochs} started ---")
                """TRAINING DATA (batches)"""
                for i, (_inputSeq, _targetSeq) in enumerate(_trainingDataPairs):
                    if self.trainingStepCounter == self.reflectionFreq: #and self.trainingStepCounter > trainingLogFreq_A:
                        ʕっʘ‿ʘʔっ("♥generating babys reflection data pairs")
                        self.reflectionTrainingPairs = self.babyReflection()
                        self.reflectionFreq = self.trainingStepCounter + reflectionFreq + len(self.reflectionTrainingPairs)

                    elif self.reflectionTrainingPairs:
                        ʕっʘ‿ʘʔっ("♥loading in a reflection pair...")
                        _inputSeq, _targetSeq = self.reflectionTrainingPairs.pop(0)

                    ʕっʘ‿ʘʔっ("♥START OF TURN")
                    inputTokenIndices, targetTokenIndexSeq = self.startTurnActions(_inputSeq = _inputSeq, _targetSeq = _targetSeq, _lastTurnLossDelta = self.latestLossDelta)
                    
                    ʕっʘ‿ʘʔっ("♥TRAINING STEP♥")                    
                    self.predictedTokenIndices, self.logitSeq = self.trainStep(_inputTokenIndices = inputTokenIndices, _targetTokenIndexSeq = targetTokenIndexSeq, _BACKWARDwobbleLoss = None)

                    """ --- --- -*- BACKWARDS COMPLETE -*- --- --- -*- --- --- -*- --- --- -*- --- --- -*- --- --- -*- --- --- -*- --- --- -*- --- --- -*- --- --- -*- --- --- """
                    
                    ʕっʘ‿ʘʔっ("♥collectTurnStats")
                    self.stats, self.stringStats, self.guessedTokenSeq = self.collectTurnStats(_targetTokenIndexSeq = targetTokenIndexSeq, _predictedTokenIndices = self.predictedTokenIndices)

                    if self.trainingStepCounter % saveModelFreq == 0:
                        ʕっʘ‿ʘʔっ("♥saveFreq")
                        self.saveFreqActions()

                    if self.trainingStepCounter % trainingLogFreq_B == 0:
                        #ʕっʘ‿ʘʔっ("♥trainingLogFreq_B") # PRINTING LOGS TO TXT AND TERMINAL
                        self.logFreqActions(_trainingDataPairs, _stringStats = self.stringStats, _frequency = trainingLogFreq_B, _trainingLogPath = trainingLogPath_1000, _detailedLogging = True, _saveLog = True)

                    # Track loss every 100 steps
                    elif self.trainingStepCounter % trainingLogFreq_A == 0:
                        ʕっʘ‿ʘʔっ("♥logFreq_A")
                        self.logFreqActions(_trainingDataPairs, _stringStats = self.stringStats, _frequency = trainingLogFreq_A, _trainingLogPath = trainingLogPath_100, _detailedLogging = False, _saveLog = True)

                    elif self.trainingStepCounter % printFreq == 0:
                        ʕっʘ‿ʘʔっ("♥printFreq")
                        self.logFreqActions(_trainingDataPairs, _stringStats = self.stringStats, _frequency = printFreq, _trainingLogPath = None, _detailedLogging = False, _saveLog = False)
                        self.printFreqActions()
                    
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

            if skipMemory:
                ʕっʘ‿ʘʔっ("♥skipMemory")
            else:
                ʕっʘ‿ʘʔっ("resetMemory")
                self.model.resetMemory(context="training")

        return inputTokenIndices, targetTokenIndexSeq, #self.repetitionPenalty, #self.wobbleLoss

    def trainStep(self, _inputTokenIndices, _targetTokenIndexSeq, _BACKWARDwobbleLoss):
        with self.counsellor.infodump("trainStep") as ʕっʘ‿ʘʔっ:
            ʕっʘ‿ʘʔっ("_model.optimizer.zero_grad")
            self.model.optimizer.zero_grad() # clears gradients last step - needed before any backward
            self.trainingStepCounter   += 1
            self.predictedTokenIndices  = []
            inputSeqPredictions = list(_inputTokenIndices)  # Start with input context, create a COPY!
            buffer = torch.zeros(windowMAX, dtype = torch.long, device = self.device) # creates buffer/step instead of recreating tensors inside loop
            buffer[:len(inputSeqPredictions)] = torch.as_tensor(inputSeqPredictions, device = self.device)
            self.logitSeq = [] # raw output of each prediction
            cumulativeLoss = torch.tensor(0.0, device = self.device) # sum of token losses for THIS sequence - averaged at the end

            for j in range(numTokensPerStep): # Predict multiple tokens in a sequence, one at a time
                ʕっʘ‿ʘʔっ("FORWARD")
                inputTensor = buffer[:len(inputSeqPredictions)] # slices input to only keep relevant part
                try:
                    if forwardProfiler: 
                        with torch.profiler.profile(record_shapes = True) as prof:
                            logits = self.model.forward(inputTensor)
                    else:
                        logits = self.model.forward(inputTensor)
                except RuntimeError as e:
                    print("TUTOR.trainStep.forward failed!", e)
                    return [], []
                
                if forwardProfiler: print(prof.key_averages().table())

                ʕっʘ‿ʘʔっ("getResponseFromLogits")
                predictedTokenIndex = self.model.getResponseFromLogits(logits)

                ʕっʘ‿ʘʔっ("inputSeqPredictions")
                self.predictedTokenIndices.append(predictedTokenIndex) # tensor shape [1]
                nextTokenInput = (
                    predictedTokenIndex.item() if scheduledSampling and random.random() < self.scheduledSamplingRate
                    else _targetTokenIndexSeq[j] if j < len(_targetTokenIndexSeq)
                    else predictedTokenIndex.item()
                )

                sampledTokens = scheduledSampling and random.random() < self.scheduledSamplingRate
                if j == 0:
                    self.sampledFlags = []  # Only clear at start
                self.sampledFlags.append(sampledTokens)
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
                    stepLoss = self.model.computeLoss(logits, _targetTokenIndexSeq[j], self.latestLossDelta, self.perfectTokens)

                    ʕっʘ‿ʘʔっ("appendStepLoss")
                    cumulativeLoss += stepLoss

            self.inputSeqPredictions = inputSeqPredictions  # So we can access it in collectTurnStats
            self.inputSampledFlags = self.sampledFlags.copy()
            ʕっʘ‿ʘʔっ("backward")
            BACKWARDloss = cumulativeLoss / len(_targetTokenIndexSeq) if len(_targetTokenIndexSeq) > 0 else torch.tensor(0.0, device = self.device)
            #BACKWARDloss_ = (0.025*self.BACKWARDwobbleLoss)+(0.975*BACKWARDloss)
            #if windowEntropyBonus:
                #if hasattr(self.model.interneuronNetwork, "entropyBonus"):
                    #BACKWARDloss = BACKWARDloss + (0.01 * max(self.model.interneuronNetwork.entropyBonus, 0.0001))
            if not torch.isfinite(BACKWARDloss): 
                print("TUTOR.trainStep.backward !!! Loss is NaN or Inf:", BACKWARDloss)
                return [], []
            else: 
                if debugPrints: print("TUTOR.trainStep.backward - loss is not NaN or Inf:", BACKWARDloss)
                
            try:
                if profiler: 
                    with torch.profiler.profile(record_shapes = True) as prof:
                        self.model.backward(BACKWARDloss)
                elif mpsProfiler: 
                    with torch.mps.profiler.profile(mode='interval', wait_until_completed = False) as prof:
                        self.model.backward(BACKWARDloss)
                else:
                    self.model.backward(BACKWARDloss)
            except RuntimeError as e:
                print("TUTOR.trainStep.backward failed!", e)
                return [], []

            if profiler: print(prof.key_averages().table())
            
            ʕっʘ‿ʘʔっ("clip_grad_norm")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1)
            self.model.optimizer.step()

            ʕっʘ‿ʘʔっ("actions after looping")
            self.stepLossFloat              = BACKWARDloss.detach().cpu().numpy().item()
            self.learningRate               = math.exp(self.model.logLR.detach().cpu().item())
            self.memoryLength               = math.exp(self.model.logMemoryLength.detach().cpu().item())
            self.gradientClipMaxNorm        = math.exp(self.model.logGradClip.detach().cpu().item())
            self.repetitionWindow           = math.exp(self.model.logRepetitionWindow.detach().cpu().item())
            self.windowSizesMean            = self.model.interneuronNetwork.expWindowSizes.detach().cpu().mean() 
            self.temperatureFloat           = self.model.temperature
            self.scheduledSamplingRateFloat = self.scheduledSamplingRate.detach().cpu().numpy().item()
            #self.normalisedActivations      = self.model.normalisedActivations.norm().detach().cpu().numpy().item()
            #self.windowWeight               = self.model.interneuronNetwork.windowWeight.detach().cpu().item()
            self.repetitionPenalty          = self.model.repetitionPenalty.detach().cpu().item()
            #self.INN_cerebellum             = self.model.interneuronNetwork.cerebellum.detach().cpu().item()
            #self.INN_cerebellumMean         = self.model.interneuronNetwork.cerebellum.mean().cpu().item()

            #self.endTurnActions()
            if self.device.type == 'mps':
                ʕっʘ‿ʘʔっ("emptyCache (mps)")
                torch.mps.empty_cache()

            return self.predictedTokenIndices, self.logitSeq
        
    def babyReflection(self):
        with self.counsellor.infodump("startTurnActions") as ʕっʘ‿ʘʔっ:
            hesJustABaby = self.mapStatsToFeelings()

            delta       = round(self.latestLossDelta, 2)
            perfect     = self.perfectTokens
            repeated    = round(self.stats.get('repeatedPercent', 0.0) * 100, 1) if 'repeatedPercent' in self.stats else 0.0

            def makeDataStuff():
                connective = random.choice([":", " of", " is at", " =", ",", " is", "..."])
                return random.choice([
                    f"average recent loss{connective} {self.averageRecentLoss:.2f}, ",
                    f"latest loss delta{connective} {delta:.2f}, ",
                    f"perfect tokens{connective} {perfect:.0f}, ",
                    f"repeated token percentage{connective} {repeated:.2f}, ",
                    f"learning rate{connective} {self.learningRate:.5f}, ",
                    f"temperature{connective} {self.temperature:.2f}, ",
                    f"scheduled sampling rate{connective} {self.scheduledSamplingRate.item():.2f}, ",
                    #f"a memory length of: {self.memoryLength:.2f}! ",
                ])

            prompt  = f"{userName}: hey baby! how are you today? i'm just checking in to let you know about your latest scores! i hope it helps you understand your world a little better :) take a look at these: "
            prompt += makeDataStuff() 
            prompt += makeDataStuff() + "and " 
            prompt += makeDataStuff()
            #prompt += makeDataStuff() + "and " 
            #prompt += makeDataStuff()
            
            target  = f"{babyName}: " 
            target += random.choice(["hey! i am learning today, this is how i'm doing: ",
                                    "ok, so, to put it as simply as I can, i am: ",
                                    "ummm.. i've been trying to learn a lot about stats but numbers still confuse me a little, what's the context of all this?! like, to start with, ",
                                    "i am happy! i did it! i know it! i am just a baby! i learnt it! i learned: ",
                                    "oh good morning! i hope you're okay, umm, i dunno how i feel about my scores right now!! "]) 
            target += hesJustABaby
            target += random.choice(["thank you for teaching me new things! it's really weird to exist lol", 
                                    "i'm getting tired... but i don't want to give up just yet!", 
                                    "that's a lot of numbers! i don't really know what to do with them, but i'll keep trying!", 
                                    "do you think i'm doing well? i'm worried that i'm underperforming!",
                                    "but, honestly, all this learning has me excited!"])
            reflectionText  = prompt + target
            babyEndings = ["ok, anyway... i'm going to get back to school now... ",
                            "thanks for helping me think! ",
                            "learning is weird but i like it! ",
                            "i guess i've gotta go keep trying! ",
                            "i'm just a baby! ",
                            "i know it! ",
                            "i did it! ",
                            "i feel it! ",
                            "i am happy! ",
                            "i am learning! ",
                            "i learned it! ",
                            "lol ",
                            ":) ",
                            "talk in a bit! ",
                            "i'm gonna carry on with it now :D ",
                        ]
            _windowMAX      = windowMAX

            reflectionTokens = self.librarian.tokenizeText(reflectionText.lower())

            tries = 0
            while len(reflectionTokens) < (_windowMAX * 3) and tries < 50:
                target += " " + random.choice([random.choice(babyEndings), makeDataStuff()])
                reflectionText = prompt + " " + target
                reflectionTokens = self.librarian.tokenizeText(reflectionText.lower())
                tries += 1
                if tries % 5 == 0:
                    print(f"[babyReflection] still too short after {tries} tries: {len(reflectionTokens)} tokens")
            if tries >= 50:
                raise ValueError(f"babyReflection failed: could not reach enough tokens after {tries} tries.")

        
        inputTargetPairs = []
        reflectionPointer = 0

        while reflectionPointer + _windowMAX * 2 <= len(reflectionTokens):
            inputSeq = reflectionTokens[reflectionPointer : reflectionPointer + _windowMAX]
            targetSeq = reflectionTokens[reflectionPointer + _windowMAX : reflectionPointer + _windowMAX * 2]

            inputTargetPairs.append((inputSeq, targetSeq))

            reflectionPointer += 1

        return inputTargetPairs

    def saveFreqActions(self): 
        with self.counsellor.infodump("saveFreqActions") as ʕっʘ‿ʘʔっ: # SAVE THE MODEL EVERY x STEPS
            print(self.calligraphist.S_apply('dim', 'autosaving...') + self.calligraphist.S_apply('reset', ''))
            self.model.saveModel(_newStartIndex = self.startIndex, _trainingStepCounter = self.trainingStepCounter)
            p = self.trainingStepCounter + saveModelFreq
            print(self.calligraphist.S_apply('dim', f"autosave successful! saving every {saveModelFreq} steps, the next autosave will be at step {p}...") + self.calligraphist.S_apply('reset', ''))
            ʕっʘ‿ʘʔっ("grad checks")
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    print(f"after = {self.calligraphist.S_apply("emergency", f"NO GRAD: {name}")}")
                else: 
                    grad = p.grad
                    shape = tuple(grad.shape)
                    norm = grad.norm().item()
                    nonzero = grad.count_nonzero().item()
                    total = grad.numel()
                    sparsity = 1 - (nonzero / total)
                    mean = grad.mean().item()
                    std = grad.std().item()
                    print(f"after = {self.calligraphist.S_apply("almostPerfect", f"yes grad: {name} | shape: {shape} | norm: {norm:.4f} | sparsity: {sparsity:.2%} | mean: {mean:.4f} | std: {std:.4f}")}")
                    
    def printFreqActions(self): 
        with self.counsellor.infodump("printFreqActions") as ʕっʘ‿ʘʔっ: # PRINTING TRAINING OUTPUT TO TERMINAL
            #recentLoss = sum(self.recentPrintLosses)/len(self.recentPrintLosses) if self.recentPrintLosses else None
            self.calligraphist.refreshStatBands(_rollingAverages = self.ʕっෆ‿ෆʔっ)
            ʕっʘ‿ʘʔっ("calligraphist.S_colourPrintTraining")
            self.calligraphist.S_colourPrintTraining(
                _step = self.trainingStepCounter,
                _inputSeq = self.inputSeq,
                _guessedSeq_str = self.guessedTokenSeq,
                _targetSeq_str = self.stringStats.get("usedInputSeq", []),
                _recentLoss = self.averageRecentLoss, #self.ʕっෆ‿ෆʔっ.get("loss", {}).get(f"{trainingLogFreq_A}_avg", 0), # self.stepLossFloat,
                _loss = self.stepLossFloat,
                _latestLossDelta = self.latestLossDelta,
                _totalTokenCount = self.tokenCounts)
        
    def logFreqActions(self, _trainingDataPairs, _stringStats, _frequency, _trainingLogPath, _detailedLogging, _saveLog): # could also do 10x log freq??
        with self.counsellor.infodump("logFreqActions") as ʕっʘ‿ʘʔっ:
            self.stringStats = _stringStats
            self.trainingLogPath = _trainingLogPath
            #self.stats.update(self.ʕっෆ‿ෆʔっ) # SUSSY BUSSY !!!!!!!!!!!!!!!!!!!
            #fullStats = dict(self.stats)
            #fullStats.update(self.ʕっෆ‿ෆʔっ)

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
            #self.calligraphist.refreshStatBands(_rollingAverages = self.ʕっෆ‿ෆʔっ)
            self.calligraphist.S_logTraining(
                _trainingLogPath = self.trainingLogPath,
                _trainingStepCounter = self.trainingStepCounter,
                _stats = self.stats,
                _frequency = _frequency,
                _LR = self.learningRate,
                _INN_cerebellum_str = str(self.stringStats.get("INN_cerebellum_str", "<missing cerebellum>")),
                _topTokens_str = str(self.stringStats.get("topTokens", "<missing topTokens>")),
                _otherInfo_str = f"{tokenPerfect_str} | {self.stringStats.get('windowVotes_str', '<missing windowVotes>')} | {remainingData_str} | TUTOR.py {trainingLogFreq_A}",
                _detailedLogging = _detailedLogging,
                _saveLog = _saveLog)

    def collectTurnStats(self, _targetTokenIndexSeq, _predictedTokenIndices):
        with self.counsellor.infodump("collectTurnStats") as ʕっʘ‿ʘʔっ:
            ʕっʘ‿ʘʔっ("self.librarian.indexToToken.get(idx.item*())")
            lossStats = self.ʕっෆ‿ෆʔっ.get("loss", {})
            rollupA_key = f"BIG{trainingLogFreq_A}"
            rollupA_avgKey = f"{rollupA_key}_avg"
            rollB_key = f"{trainingLogFreq_B}"
            rollB_avgKey = f"{rollB_key}_avg"
            rollA_key = f"{trainingLogFreq_A}"
            rollA_avgKey = f"{trainingLogFreq_A}_avg"
            rollPrint_key = f"{printFreq}"
            rollPrint_avgKey = f"{printFreq}_avg"

            if rollB_avgKey in lossStats and rollB_key in lossStats and len(lossStats[rollB_key]) >= trainingLogFreq_B:
                if debugPrints or True: 
                    self.bbb += 1
                    if self.bbb > 1000: 
                        print(f"Used {rollB_avgKey} for averageRecentLoss: {lossStats[rollB_avgKey]} 1000x")
                        self.bbb = 0
                self.averageRecentLoss = lossStats[rollB_avgKey]
            elif rollA_avgKey in lossStats and rollA_key in lossStats and len(lossStats[rollA_key]) >= trainingLogFreq_A:
                if debugPrints or True: 
                    self.ccc += 1
                    if self.ccc > 1000: 
                        print(f"Used {rollA_avgKey} for averageRecentLoss: {lossStats[rollA_avgKey]} 1000x")
                        self.ccc = 0
                self.averageRecentLoss = lossStats[rollA_avgKey]
            else:
                if rollPrint_avgKey in lossStats and rollPrint_key in lossStats and len(lossStats[rollPrint_key]) >= printFreq:
                    if debugPrints or True: 
                        self.ppp += 1
                        if self.ppp > 1000: 
                            print(f"Used {rollPrint_avgKey} for averageRecentLoss: {lossStats[rollPrint_avgKey]} 1000x")
                            self.ppp = 0
                    self.averageRecentLoss = lossStats[rollPrint_avgKey]

            self.guessedTokenSeq = [self.librarian.indexToToken.get(idx.item(), "<UNK>") for idx in self.predictedTokenIndices]
            if self.guessedTokenSeq: 
                self.tokenCounts.update(self.guessedTokenSeq)

            ʕっʘ‿ʘʔっ("SCRIBE.maybeCommentOnGuess")
            if self.trainingStepCounter > trainingLogFreq_A:
                self.scribe.maybeCommentOnGuess(self.guessedTokenSeq, self.stepLossFloat, "scribe", 0.00075)

            ʕっʘ‿ʘʔっ("collectStats♥")

            if collectStats:
                ʕっʘ‿ʘʔっ("♥if collectStats♥")

                ʕっʘ‿ʘʔっ("♥build usedInputSeq with styling")
                usedInputSeq = self.inputSeqPredictions[-numTokensPerStep:]
                formattedUsed = []

                for i, idx in enumerate(usedInputSeq):
                    tok = self.librarian.indexToToken.get(idx, "<UNK>")
                    sampled = self.inputSampledFlags[-numTokensPerStep + i] if i < len(self.inputSampledFlags) else False

                    if sampled:
                        styled = self.calligraphist.S_apply(self.calligraphist.S_getStat('loss', self.stepLossFloat), tok)
                    else:
                        styled = self.calligraphist.S_apply('dim', tok)

                    formattedUsed.append(styled)

                self.stringStats["usedInputSeq"] = formattedUsed

                if token_collectStats:
                    ʕっʘ‿ʘʔっ("♥if token_collectStats♥")
                    self.predictedTokenIndices = _predictedTokenIndices

                    ʕっʘ‿ʘʔっ("♥most common tokens")
                    topTokens = ""
                    topTokens = self.tokenCounts.most_common(10)
                    self.perfectTokens = 0

                    ʕっʘ‿ʘʔっ("♥calculate perfect tokens")
                    if not _predictedTokenIndices:
                        print("!! no predicted token indices — returning \{\} for stringStats")
                        return self.stats, {}, self.guessedTokenSeq # THIS IS WHERE THE DAMN LIST ERROR WAS LMAOOOONOOO
                    target      = torch.tensor(_targetTokenIndexSeq[:numTokensPerStep], device = modelDevice)
                    predicted   = torch.tensor(self.predictedTokenIndices, device = modelDevice)
                    correct     = (predicted == target).sum() # ~~~ if predicted = target, over whole tensor 
                    self.perfectTokens += correct
                    self.totalTokenEvaluations += len(target)

                if static_collectStats:
                    ʕっʘ‿ʘʔっ("♥if static_collectStats")
                    self.stats["scheduledSamplingRate"] = self.scheduledSamplingRateFloat
                    self.stats["repetitionPenalty"]     = self.repetitionPenalty
                    self.stats["AvgLoss"]               = self.averageRecentLoss
                    self.stats["loss"]                  = self.stepLossFloat
                    self.stats["temperature"]           = self.temperatureFloat
                    self.temperature                    = self.stats["temperature"]
                    self.stats["LR"]                    = self.learningRate
                    self.stats["gradientClipMaxNorm"]   = self.gradientClipMaxNorm
                    self.stats["latestLossDelta"]       = self.latestLossDelta
                    self.stats["memoryLength"]          = self.memoryLength
                    self.stats["repetitionWindow"]      = self.repetitionWindow
                    self.stats["perfectTokens"]         = self.perfectTokens
                    self.stats["windowSizesMean"]       = self.windowSizesMean
                    self.stats["normalisedActivationsNorm"] = self.model.stats["normalisedActivationsNorm"]
                    #self.stats["normalisedActivations"] = self.normalisedActivations
                    #self.stats["windowWeight"]          = self.windowWeight
                    #self.stats["INN_cerebelumMean"]     = self.INN_cerebellumMean
                    ##self.stats["INN_cerebellum"]        = self.INN_cerebellum
 
                if embed_collectStats:
                    ʕっʘ‿ʘʔっ("♥if embed_collectStats")
                    self.stats.update(self.model.embed.getEmbedStats())

                if logit_collectStats:
                    ʕっʘ‿ʘʔっ("♥if logit_collectStats♥")
                    self.stats.update(self.model.logits.getLogitStats())
                    if self.stats["logitSeq"]:
                        ʕっʘ‿ʘʔっ("♥logit max & min")
                        self.stats["logitMin"] = self.logitSeq[-1].min(dim=-1).values.mean()
                        self.stats["logitMax"] = self.logitSeq[-1].max(dim=-1).values.mean()

                #self.stats.update(self.wobble.getWobbleStats())

                if skipMemory:
                    ʕっʘ‿ʘʔっ("♥skipMemory")
                else:
                    self.model.memory.updateMemoryBuffers()
                    if memory_collectStats:
                        ʕっʘ‿ʘʔっ("♥if memory_collectStats")
                        self.stats.update(self.model.memory.getMemoryStats())

                ʕっʘ‿ʘʔっ("♥INN_collectStats")
                INN_stats, INN_cerebellum_str = self.model.interneuronNetwork.INN_getStats()
                self.stats.update(INN_stats)
                INN_stringStats = {"INN_cerebellum_str": str(INN_cerebellum_str)}
                self.stringStats.update(INN_stringStats)
                self.stringStats.update({"topTokens": str(topTokens)})
                self.collectAllTimeStats()

        return self.stats, self.stringStats, self.guessedTokenSeq

    def collectAllTimeStats(self):
        for _statKey, _value in self.stats.items():
            if not isinstance(_value, (int, float)):
                if debugPrints and _statKey == "loss":
                    print(f"{_statKey} value is : {_value}, {_statKey} value type is {type(_value)}")
                continue  # skip strings, tensors, weird stuff

            """ෆෆෆ^ ♥ KEYS ETC ♥ ^ෆෆෆ"""
            _               = self.ʕっෆ‿ෆʔっ[_statKey]  # this will autoinit with defaultdict
            ෆ‿ෆ             = self.ʕっෆ‿ෆʔっ[_statKey]
            important       = ["loss"]
            rolling         = mostImportantStats
            percentiles     = percentileBands

            """ ෆෆෆ^ ♥ UPDATE EVERY TURN ♥ ^ෆෆෆ   """
            """ ෆෆෆ^ ♥ turn stats ♥ ^ෆෆෆ  """
            #if _statKey == "loss":
                #print(f"Setting prev to: {ෆ‿ෆ.get("now", 0.0)}, Setting now to: {_value}, Setting _Δ to {_value - ෆ‿ෆ.get("now", 0.0)}")
            ෆ‿ෆ["now"]      = _value
            if ෆ‿ෆ["prev"]:
                ෆ‿ෆ["_Δ"]   = _value - ෆ‿ෆ["prev"]
            ෆ‿ෆ["prev"]     = ෆ‿ෆ.get("now", 0.0)

            """ ෆෆෆ^ ♥ totals ♥ ^ෆෆෆ  """
            ෆ‿ෆ["totSum"]   = ෆ‿ෆ.get("totSum", 0.0)    + _value
            ෆ‿ෆ["totNum"]   = ෆ‿ෆ.get("totNum", 0)      + 1
            ෆ‿ෆ["totAvg"]   = ෆ‿ෆ["totSum"] / ෆ‿ෆ["totNum"]
            ෆ‿ෆ["totAvgΔ"]  = ෆ‿ෆ["now"]    - ෆ‿ෆ["totAvg"]

            """ ෆෆෆ^ ♥ records ♥ ^ෆෆෆ """
            #ෆ‿ෆ["_p100"]    = max(ෆ‿ෆ.get("_p100", _value), _value) # TOP EVER RECORD // PERCENTILE 100
            #ෆ‿ෆ["_p0.00"]   = min(ෆ‿ෆ.get("_p0.00", _value), _value) # BOTTOM EVER RECORD // PERCENTILE 0

            """ ෆෆෆ^ ♥ ROLLING STATS ♥ ^ෆෆෆ   """
            if _statKey in rolling or _statKey.startswith("INN_cerebellum_W"):
                for freq in [printFreq, trainingLogFreq_A, trainingLogFreq_B]:
                    tag = f"{freq}"
                    if tag not in ෆ‿ෆ:
                        ෆ‿ෆ[tag] = []
                    if len(ෆ‿ෆ[tag]) >= freq: 
                        ෆ‿ෆ[tag].pop(0)
                    ෆ‿ෆ[tag].append(_value)
                    if ෆ‿ෆ[tag]:
                        self.updateRollingStats(_ෆ‿ෆ = ෆ‿ෆ, _values = ෆ‿ෆ[tag], _freq = freq, _tag = tag, _percentiles = percentiles)

            if _statKey in important and self.trainingStepCounter % trainingLogFreq_A == 0:
                for importantFreq in [trainingLogFreq_B]:
                    importantTag = f"BIG{importantFreq}"
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

        delta                   = _ෆ‿ෆ["now"] - _ෆ‿ෆ[f"{_tag}_avg"]
        _ෆ‿ෆ[f"{_tag}_Δ"]       = delta

        if _percentiles:
            for p in _percentiles:
                _ෆ‿ෆ[f"{_tag}_p{p}"]  = np.percentile(_values, p)

    def stdTest(self, values):
        if len(values) <= 1: return 0.0
        avg = sum(values) / len(values)
        variance = sum((x - avg)**2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    def endTurnActions(self):
        with self.counsellor.infodump("endTurnActions") as ʕっʘ‿ʘʔっ:
            ʕっʘ‿ʘʔっ("♥getLatestLossDelta")
            lossStats = self.ʕっෆ‿ෆʔっ.get("loss", {})

            self.latestLossDelta = self.stepLossFloat - self.averageRecentLoss

            ʕっʘ‿ʘʔっ("finalLogActions")
            if debugPrints:
                for key in self.ʕっෆ‿ෆʔっ:
                    print(key, self.ʕっෆ‿ෆʔっ[key])
            self.stats.clear()
            self.stringStats.clear()
            self.tokenPerfectRate = 0
            self.stats['sampledTokens'] = 0
            self.totalTokenEvaluations = 0
        
        return self.latestLossDelta
        
    def mapStatsToFeelings(self):
        babyFeels = []
        feelings = []

        lossStats       = self.ʕっෆ‿ෆʔっ.get("loss", {})
        tempStats       = self.ʕっෆ‿ෆʔっ.get("temperature", {})
        repetitionStats = self.ʕっෆ‿ෆʔっ.get("repetitionPenalty", {})
        samplingStats   = self.ʕっෆ‿ෆʔっ.get("scheduledSamplingRate", {})
        memStats        = self.ʕっෆ‿ෆʔっ.get("memoryLength", {})
        perfectTokens   = self.perfectTokens
        deltaLoss       = self.latestLossDelta

        current_loss                = lossStats.get("now", None)
        current_temp                = tempStats.get("now", None)
        current_repeated            = self.tokenPerfectRate
        current_sampling            = samplingStats.get("now", None)
        current_memLength           = memStats.get("now", None)
        current_repetitionPenalty   = repetitionStats.get("now", None)

        emoStats = {
            "loss": current_loss,
            "temperature": current_temp,
            "repetitionPenalty": current_repetitionPenalty,
            "scheduledSamplingRate": current_sampling,
            "memoryLength": current_memLength,
            "perfectTokens": perfectTokens,
            "repeatedPercent": current_repeated,
            "latestLossDelta": deltaLoss,
        }

        def makeEmoNotes(stat, value):
            feeling = None #"neutral"

            if stat == "loss":
                if "p_90" in lossStats and value >= lossStats["p_90"]: feeling = "overwhelmed"
                elif "p_75" in lossStats and value >= lossStats["p_75"]: feeling = "pressured"
                elif "p_25" in lossStats and value <= lossStats["p_25"]: feeling = random.choice(["clever", "proud"])
                elif "p_10" in lossStats and value <= lossStats["p_10"]: feeling = random.choice(["very clever", "like i get it"])

            elif stat == "repetitionPenalty":
                if "p_90" in repetitionStats and value >= repetitionStats["p_90"]: feeling = "non-verbal"
                elif "p_75" in repetitionStats and value >= repetitionStats["p_75"]: feeling = "quiet"
                elif "p_25" in repetitionStats and value <= repetitionStats["p_25"]: feeling = random.choice(["talkative", "chatty"])
                elif "p_10" in repetitionStats and value <= repetitionStats["p_10"]: feeling = random.choice(["conversational", "fluent"])

            elif stat == "latestLossDelta":
                if value > 0.5: feeling = "like i'm struggling to focus"
                elif value < -0.5: feeling = "interested"

            elif stat == "repeatedPercent":
                if value > 0.7: feeling = random.choice(["stuttering", "like im repeating a lot"])
                elif value > 0.5: feeling = random.choice(["overstimulated", "silly"])
                elif value < 0.1: feeling = random.choice(["calm", "saying lots of new things"])
                elif value < 0.25: feeling = "curious"

            elif stat == "temperature":
                if "p_90" in tempStats and value >= tempStats["p_90"]: feeling = random.choice(["chaotic", "excited"])
                elif "p_75" in tempStats and value >= tempStats["p_75"]: feeling = random.choice(["playful", "happy"])
                elif "p_25" in tempStats and value <= tempStats["p_25"]: feeling = "in work mode"

            elif stat == "scheduledSamplingRate":
                if value > 0.8: feeling = random.choice(["creative", "inventive"])
                elif value < 0.2: feeling = random.choice(["tired", "copying"])

            elif stat == "memoryLength":
                if value > 12: feeling = "pensive"
                elif value < 4: feeling = "mindful"

            elif stat == "perfectTokens":
                if value >= 30: feeling = "very proud"
                elif value >= 10: feeling = "proud"
                elif value <= 1: feeling = random.choice(["sad", "frustrated"])

            else:
                feeling = random.choice(["alright", "a bit lost"])
            
            if feeling is None:
                feeling = "neutral"

            feelings.append(feeling)

            feelVerb = random.choice(["feel", "seem", "think i feel", "definitely feel", "might feel"])
            templates = [
                f"i {feelVerb} {feeling} because my {stat} is {value:.2f}! ",
                f"maybe it's because my {stat} is {value:.2f} that i {feelVerb} {feeling}! ",
                f"i noticed my {stat} is {value:.2f}, and i {feelVerb} {feeling}! ",
                f"when my {stat} is {value:.2f}, i {feelVerb} {feeling}! ",
                f"it's {value:.2f} for {stat}... so i {feelVerb} {feeling} about it! ",
            ]
            return random.choice(templates)

        chosenStats = []
        attempts = 0

        while len(chosenStats) < 5 and attempts < 10:
            stat, value = random.choice(list(emoStats.items()))
            if value is not None:
                chosenStats.append((stat, value))
            attempts += 1
        if attempts >= 10 or True:
            print(f"emoStats:{emoStats}")
        for stat, value in chosenStats:
            babyFeels.append(makeEmoNotes(stat, value))

        return "".join(babyFeels)