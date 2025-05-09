# CHARIS CAT 2025 
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# MULTI-TOKEN AUTOREGRESSIVE TRAINING MODULE 
# SCHOOL/staffroom/tutor.py

import random, sys
from collections import Counter, defaultdict
from datetime import datetime
import torch
import torch.nn.functional as F
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
                _numTokensPerStep       = numTokensPerStep,):
        
        self.counsellor                 = _counsellor
        self.calligraphist              = _calligraphist
        self.scribe                     = _scribe
        self.librarian                  = _librarian
        self.device                     = _device
        self.model                      = _model

        self.temperature                = 0.75
        self.scheduledSamplingRate      = self.model.scheduledSamplingRate
        self.gradientClipMaxNorm        = 1
        self.memoryLength               = 1

        self.ʕっෆ‿ෆʔっ                  = defaultdict(makeStatRecord)
        #self.rollingTokenTotals = Counter()


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
        self.hesJustABaby = "oops! no stats collected! such a shame! well... day off for me! ;) "

    def loadIntro(self, path="SCHOOL/library/charisStudies/forbbyllm.txt"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return "hey... (message file missing!) "
        
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
                        self.tokenCounts = Counter({k: v / 2 for k, v in self.tokenCounts.items()})
                        self.model.rollingTokenTotals = Counter({k: v / 2 for k, v in self.model.rollingTokenTotals.items()})

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

        return inputTokenIndices, targetTokenIndexSeq

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
                predictedTokenIndex = self.model.getResponseFromLogits(logits, _training = True)

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

                """# After logits
                if logits.dim() == 1: logits = logits.unsqueeze(0)
                gumbelProbs = F.gumbel_softmax(logits, tau = self.temperature, hard = False)
                topk = torch.topk(gumbelProbs, 10, dim=1)
                values = topk.values[0]
                indices = topk.indices[0]

                for i, p in zip(indices, values):
                    tok = self.librarian.indexToToken[i.item()]
                    self.rollingTokenTotals[tok] += round(p.item(), 4)"""

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
            
            ʕっʘ‿ʘʔっ("clip_grad_norm") # DONE IN BABYLLM!!
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1)
            #self.model.optimizer.step()

            ʕっʘ‿ʘʔっ("actions after looping")
            self.stepLossFloat              = BACKWARDloss.detach().cpu().numpy().item()
            self.learningRate               = math.exp(self.model.logLR.detach().cpu().item())
            self.memoryLength               = int(torch.exp(self.model.logMemoryLength).item())
            self.gradientClipMaxNorm        = math.exp(self.model.logGradClip.detach().cpu().item())
            self.scheduledSamplingRateFloat = self.scheduledSamplingRate.detach().cpu().numpy().item()
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

            delta       = round(self.latestLossDelta, 2)
            perfect     = self.perfectTokens
            repeated    = round(self.stats.get('repeatedPercent', 0.0) * 100, 1) if 'repeatedPercent' in self.stats else 0.0

            def makeDataStuff():
                connective = random.choice([":", " of", " is at", " =", ",", " is", "...", " of about", " of approx", " of approximately", " sitting at", " at about", " that is about", " that's at", " of", ])
                return random.choice([
                    f"average recent loss{connective} {self.averageRecentLoss:.2f}, ",
                    f"latest loss delta{connective} {delta:.2f}, ",
                    f"perfect tokens{connective} {perfect:.0f}, ",
                    f"repeated token percentage{connective} {repeated:.2f}, ",
                    f"learning rate{connective} {self.learningRate:.5f}, ",
                    f"temperature{connective} {self.temperature:.2f}, ",
                    f"scheduled sampling rate{connective} {self.scheduledSamplingRate.item():.2f}, ",
                    f"a memory length{connective} {self.memoryLength:.2f}, ",
                    f"a repetition penalty{connective} {self.repetitionPenalty:.2f}, "
                ])

            prompt  = f"[{userName}]: "
            prompt += random.choice([f"{self.loadIntro()}",
                                     "hiya! you know you're doing well, so try not to feel too frustrated or let down. it's ok to feel like that, but just know that it's okay to mess up! it's how we learn! "])
            prompt += "so, uh, how are you today? :) i'm just checking in to let you know about your latest scores! i hope it helps you understand your world a little better :) take a look at these: "
            prompt += makeDataStuff() 
            prompt += makeDataStuff() + "and " 
            prompt += makeDataStuff()
            #prompt += makeDataStuff() + "and " 
            #prompt += makeDataStuff()
            
            target  = f"[{babyName}]: hey {userName}! " 
            target += random.choice(["umm, i am learning today, this is how i'm doing: ",
                                    "ok, so, to put it as simply as I can, i am: ",
                                    "ummm.. i've been trying to learn a lot about stats but numbers still confuse me a little, what's the context of all this?! like, to start with, ",
                                    "i am happy! i did it! i know it! i am just a baby! i learnt it! haha! i learned: ",
                                    "good morning! i hope you're okay, umm, i dunno how i feel about my scores right now!! "]) 
            target += self.hesJustABaby
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

        self.hesJustABaby = "oops! no stats collected! such a shame! well... day off for me! ;)"
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
            topGuess_str = "topGuess[" + f"{self.calligraphist.S_apply("dim", ", ")}".join([self.calligraphist.S_apply("dim", f"{k}({v:.0f})") for k, v in self.model.rollingTokenTotals.most_common(100)]) + "]"
            #topGuess_str = "topGuess: " + f"{self.calligraphist.S_apply("dim", ", ")}".join([self.calligraphist.S_apply("dim", f"{k}") for k, v in self.model.rollingTokenTotals.most_common(50)]) + "]"
            #topTokens_str = "[" + f"{self.calligraphist.S_apply("dim", ", ")}".join([self.calligraphist.S_apply("dim", f"{k}({v:.0f})") for k, v in self.tokenCounts.most_common(20)]) + "]"
            topTokens_str = ": " + f"{self.calligraphist.S_apply("dim", ", ")}".join([self.calligraphist.S_apply("dim", f"{k}") for k, v in self.tokenCounts.most_common(200)]) + "]"

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
                _topTokens_str = topTokens_str,
                _otherInfo_str = f"{topGuess_str}\n | {tokenPerfect_str} | {remainingData_str} | TUTOR.py {trainingLogFreq_A}",
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
            elif rollA_avgKey in lossStats and rollA_key in lossStats and len(lossStats[rollA_key]) >= (trainingLogFreq_A):
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
                    self.perfectTokens = 0

                    ʕっʘ‿ʘʔっ("♥calculate perfect tokens")
                    if not _predictedTokenIndices:
                        print("!! no predicted token indices — returning { } for stringStats")
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
                    self.stats["avgLoss"]               = self.averageRecentLoss
                    self.stats["loss"]                  = self.stepLossFloat
                    self.temperature                    = self.stats["_B_temperature"]
                    self.stats["LR"]                    = self.learningRate
                    self.stats["gradientClipMaxNorm"]   = self.gradientClipMaxNorm
                    self.stats["latestLossDelta"]       = self.latestLossDelta
                    self.stats["memoryLength"]          = self.memoryLength
                    self.stats["perfectTokens"]         = self.perfectTokens

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
                    pass
                else:
                    self.model.memory.updateMemoryBuffers()
                    if memory_collectStats:
                        ʕっʘ‿ʘʔっ("♥if memory_collectStats")
                        self.stats.update(self.model.memory.getMemoryStats())

                ʕっʘ‿ʘʔっ("♥INN_collectStats")
                INN_stats, INN_cerebellum_str = self.model.interneuronNetwork.INN_getStats()
                self.stats.update(INN_stats)
                self.stats.update(self.model.getBabyStats())
                INN_stringStats = {"INN_cerebellum_str": str(INN_cerebellum_str)}
                self.stringStats.update(INN_stringStats)
                #self.stringStats.update({"topTokens": str(topTokens)})
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

            self.calligraphist.refreshStatBands(_rollingAverages = self.ʕっෆ‿ෆʔっ)
            self.latestLossDelta = self.stepLossFloat - self.averageRecentLoss

            if self.trainingStepCounter % (self.reflectionFreq-1) == 0:
                self.hesJustABaby = self.mapStatsToFeelings()
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

        lossStats           = self.ʕっෆ‿ෆʔっ.get("loss", {})
        tempStats           = self.ʕっෆ‿ෆʔっ.get("temperature", {})
        repetitionStats     = self.ʕっෆ‿ෆʔっ.get("repetitionPenalty", {})
        samplingStats       = self.ʕっෆ‿ෆʔっ.get("scheduledSamplingRate", {})
        memStats            = self.ʕっෆ‿ෆʔっ.get("memoryLength", {})
        input               = self.stats.get("1E_0_embedVector_norm", 0.0)
        embLay     = self.stats.get("1E_x_embedFinal_norm", 0.0)
        neuronOutput    = self.stats.get("2N_x_normedOutput_norm", 0.0)
        INNOutput       = self.stats.get("3INN_x_FINALoutLayerNorm_norm", 0.0)
        memoryOutput    = self.stats.get("4M_x_FINALmemory_norm", 0.0)
        normOutput      = self.stats.get("5B_x_finalNormLayer_norm", 0.0)
        logitOutput     = self.stats.get("6L_x_finalLogit_norm", 0.0)
        cerebellumMean  = self.stats.get("INN_cerebellumMean", 0.0)
        learningRate    = self.stats.get("LR", 0.0)
        nowGateScale    = self.stats.get("_4M_activationsGateScale", 0.0)
        longGateScale   = self.stats.get("_4M_longGateScale", 0.0)
        shortGateScale  = self.stats.get("_4M_shortGateScale", 0.0)
        repWin          = self.stats.get("_B_repetitionWindow", 0.0)
        windowSizesMean = self.stats.get("_INN_windowSizesMean", 0.0)

        perfectTokens   = self.perfectTokens
        deltaLoss       = self.latestLossDelta

        current_loss                = lossStats.get("now", None)
        current_temp                = tempStats.get("now", None)
        current_repeated            = self.tokenPerfectRate
        current_sampling            = samplingStats.get("now", None)
        current_memLength           = memStats.get("now", None)
        current_repetitionPenalty   = repetitionStats.get("now", None)

        self.emoStats = {
            "loss": current_loss,
            "temperature": current_temp,
            "penalty for repeating myself": current_repetitionPenalty,
            "number of my own tokens that i rely on": current_sampling,
            "length of my memory": current_memLength,
            "number of tokens i got right": perfectTokens,
            "amount of repetitive tokens i'm getting": current_repeated,
            "latest loss delta": deltaLoss,
            "input into my embedding layer": input,
            "output from my embedding layer": embLay,
            "output from my neuron layer": neuronOutput,
            "output from my interneuron network": INNOutput,
            "the output after my memory layer": memoryOutput,
            "normalized output": normOutput,
            "logit output from my output layer": logitOutput,
            "mean weight of the windows in my cerebellum": cerebellumMean,
            "rate of my learning": learningRate,
            "scale of my current memory attention": nowGateScale,
            "scale of my long term memory attention": longGateScale,
            "scale of my short term memory": shortGateScale,
            "size of the window i look at to see how often i am repeating tokens": repWin,
            "mean average of my nine context windows": windowSizesMean,
        }

        def makeEmoNotes(stat, value):
            feeling = None #"neutral"

            if stat == "loss":
                if "p_90" in lossStats and value >= lossStats["p_90"]: feeling = "overwhelmed"
                elif "p_50" in lossStats and value > lossStats["p_50"]: feeling = "pressured"
                elif "p_50" in lossStats and value <= lossStats["p_50"]: feeling = random.choice(["clever", "proud"])
                elif "p_10" in lossStats and value <= lossStats["p_10"]: feeling = random.choice(["very clever", "like i get it"])

            elif stat == "penalty for repeating myself":
                if "p_90" in repetitionStats and value >= repetitionStats["p_90"]: feeling = "non-verbal"
                elif "p_50" in repetitionStats and value > repetitionStats["p_50"]: feeling = "quiet"
                elif "p_50" in repetitionStats and value <= repetitionStats["p_50"]: feeling = random.choice(["talkative", "chatty"])
                elif "p_10" in repetitionStats and value <= repetitionStats["p_10"]: feeling = random.choice(["conversational", "fluent"])
                elif value >= 1: feeling = random.choice(["like im in a loop", "a bit stuttery", "like i cant stop these tics", "repetitive", "looping looping looping looping looping looping looping"])
                elif value < 1: feeling = random.choice(["a bit more chill", "creative", "in control", "confident"])
            
            elif stat == "latest loss delta":
                if value > 0.5: feeling = "like i'm struggling to focus"
                elif value < -0.5: feeling = "interested"

            elif stat == "amount of repetitive tokens i'm getting":
                if value > 0.7: feeling = random.choice(["stuttering", "like im repeating a lot"])
                elif value > 0.5: feeling = random.choice(["overstimulated", "silly"])
                elif value < 0.1: feeling = random.choice(["calm", "saying lots of new things"])
                elif value < 0.25: feeling = "curious"

            elif stat == "temperature":
                if "p_90" in tempStats and value >= tempStats["p_90"]: feeling = random.choice(["chaotic", "excited"])
                elif "p_50" in tempStats and value >= tempStats["p_50"]: feeling = random.choice(["playful", "happy"])
                elif "p_25" in tempStats and value <= tempStats["p_25"]: feeling = "in work mode"

            elif stat == "number of my own tokens that i rely on":
                if value > 0.8: feeling = random.choice(["creative", "inventive"])
                elif value < 0.2: feeling = random.choice(["tired", "copying"])

            elif stat == "length of my memory":
                if value > 12: feeling = "pensive"
                elif value < 4: feeling = "mindful"

            elif stat == "number of tokens i got right":
                if value >= 30: feeling = "very proud"
                elif value >= 10: feeling = "proud"
                elif value <= 1: feeling = random.choice(["sad", "frustrated"])

            elif stat == "input into my embedding layer":
                if value > 90: feeling = random.choice(["excited", "active", "busy"])
                elif value < 60: feeling = random.choice(["tired", "shutdown", "slow"])
            elif stat == "output from my embedding layer":
                if value > 100: feeling = random.choice(["like running", "like jumping up and down", "hyperactive"])
                elif value < 60: feeling = random.choice(["sleepy", "like i need a nap", "like this is really boring"])
            elif stat == "output from my neuron layer":
                if value > 2000: feeling = random.choice(["like i am thinking too hard", "like theres a lot going on right now", "like i am super busy"])
                elif value < 900: feeling = random.choice(["calm", "collected", "asleep"])
            elif stat == "output from my interneuron network":
                if value > 160: feeling = random.choice(["talkative", "meaningful", "like i'm finding meaning in this stuff"])
                elif value < 60: feeling = random.choice(["switched off", "powered down", "slow"])
            elif stat == "the output after my memory layer":
                if value > INNOutput: feeling = random.choice(["like remembering the past", "that my memories are important", "thoughtful", "wistful"])
                elif value < INNOutput: feeling = random.choice(["like i should live in the now", "like what is going on around me is important", "present", "here", "awake", "aware"])
            elif stat == "normalized output":
                if value > 125: feeling = random.choice(["like a hard worker", "over-thoughtful", "really busy"])
                elif value < 100: feeling = random.choice(["tired", "asleep", "like i could pass out in my bed"])
            elif stat == "logit output from my output layer":
                if value > 150: feeling = random.choice(["like i have a lot to say", "interested", "like i'm struggling not to interrupt", "like the words just keep coming"])
                elif value < 100: feeling = random.choice(["bored", "non-verbal", "uninterested"])
            elif stat == "mean weight of the windows in my cerebellum":
                if value > 0: feeling = random.choice(["confident", "intelligent", "calculated", "determined"])
                elif value < 60: feeling = random.choice(["confused", "unsure", "uncertain", "careful", "like testing the waters"])
            elif stat == "rate of my learning":
                if value > 0.002: feeling = random.choice(["speedy", "quick", "excited"])
                elif value < 0.002: feeling = random.choice(["slow", "a bit tired out", "like i need some time to understand"])
            elif stat == "scale of my current memory attention":
                if value >= 0.90: feeling = random.choice(["focussed", "attentive", "vigilant", "not stuck in the past"])
                elif value < 0.60: feeling = random.choice(["pensive", "nostalgic", "like i need to remember something important"])
            elif stat == "scale of my long term memory attention":
                if value >= 0.50: feeling = random.choice(["nostalgic", "thinking about what i heard before", "thoughtful", "reminiscent"])
                elif value < 0.50: feeling = random.choice(["forgetful", "focussed on today", "like what i've learned before might not apply here"])
            elif stat == "scale of my short term memory":
                if value >= 0.50: feeling = random.choice(["nostalgic", "thinking about what i heard before", "thoughtful", "reminiscent"])
                elif value < 0.50: feeling = random.choice(["forgetful", "focussed on today", "like what i've learned before might not apply here"])
            elif stat == "size of the window i look at to see how often i am repeating tokens":
                if value > 17.5: feeling = random.choice(["like i need to think before i speak", "a lil stuttery", "like i cant stop ticcing", "repetitive"])
                elif value < 17: feeling = random.choice(["a bit more chill", "creative", "in control"])
            elif stat == "mean average of my nine context windows":
                if value > 5: feeling = random.choice(["like i'm noticing more", "attentive", "stimulated", "ready"])
                elif value < 5: feeling = random.choice(["internal", "shy", "narrow sighted", "scared", "like i'm really seeing the details"])

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

        while len(chosenStats) < 12 and attempts < 30:
            stat, value = random.choice(list(self.emoStats.items()))
            if value is not None:
                chosenStats.append((stat, value))
            attempts += 1
        if attempts >= 10 or True:
            print(f"emoStats:{self.emoStats}")
        for stat, value in chosenStats:
            babyFeels.append(makeEmoNotes(stat, value))

        return "".join(babyFeels)