import time, random, sys
from collections import Counter
from datetime import datetime
import torch
from BRAIN.LAYERS.S_output import *
from SCHOOL.staffroom.counsellor import *
#from BRAIN.LAYERS.memory import *
from SCHOOL.staffroom.HE_IS_SCRIBE import SCRIBE
from config import *

class TUTOR:
    def __init__(self, model, vocab):
        self.counsellor = COUNSELLOR("TUTOR", debug=debugPrints, durations=durationLogging)
        self.s_output = S_OUTPUT()
        self.scribe = SCRIBE()
        vocab = model.vocab
        self.perfectTokenCount = 0
        self.totalTokenEvaluations = 0
        self.perfectTokenCounts_100 = 0
        self.totalTokenEvaluations_100 = 0
        self.recentPrintLosses = []
        self.scheduledSamplingProb = 0
        #model.to(modelDevice)

    def trainStep(self, inputTokenIndices, targetTokenIndexSeq, model):
        with self.counsellor.infodump("trainStep") as ʕっʘ‿ʘʔっ:
            model.optimizer.zero_grad()
            predictedTokenIndices = []
            inputSeqPredictions = list(inputTokenIndices)  # Start with input context, create a COPY!
            losses = []
            logitSeq = []
            cumulativeLoss = 0.0 # Sum of losses for THIS sequence

            for j in range(numTokensPerStep): # Predict multiple tokens in a sequence
                ʕっʘ‿ʘʔっ("FORWARD")
                inputTensor = torch.tensor(inputSeqPredictions, device=modelDevice, dtype=torch.long) # ENSURE THE INPUT TO FORWARD IS A TENSOR
                logits = model.forward(inputTensor)

                ʕっʘ‿ʘʔっ("getResponseFromLogits")
                predictedTokenIndex = model.getResponseFromLogits(logits)
                ʕっʘ‿ʘʔっ("inputSeqPredictions")
                logitSeq.append(logits)
                predictedTokenIndices.append(predictedTokenIndex)
                nextTokenInput = (predictedTokenIndex 
                    if scheduledSampling and random.random() < self.scheduledSamplingProb 
                    else targetTokenIndexSeq[j] if j < len(targetTokenIndexSeq) 
                    else predictedTokenIndex)
                inputSeqPredictions.append(nextTokenInput)

                ʕっʘ‿ʘʔっ("perfectTokenCounter")
                if j < len(targetTokenIndexSeq):
                    self.totalTokenEvaluations += 1
                    self.totalTokenEvaluations_100 += 1
                    if predictedTokenIndex == targetTokenIndexSeq[j]:
                        ʕっʘ‿ʘʔっ("addPerfectToken")
                        self.perfectTokenCount += 1
                        self.perfectTokenCounts_100 += 1

                    ʕっʘ‿ʘʔっ("computeLoss")
                    stepLoss = model.computeLoss(logits, targetTokenIndexSeq[j])

                    ʕっʘ‿ʘʔっ("appendStepLoss")
                    losses.append(stepLoss)
                    cumulativeLoss += stepLoss
                    self.avgLoss_100 += stepLoss
                    self.avgLoss_1000 += stepLoss

            ʕっʘ‿ʘʔっ("loss = cumulativeLoss / len(losses)")
            loss = cumulativeLoss / len(losses) if losses else torch.tensor(0.0, device = modelDevice)

            ʕっʘ‿ʘʔっ("recentPrintLosses")
            self.recentPrintLosses.append(loss.item())
            if len(self.recentPrintLosses) > printFreq: 
                self.recentPrintLosses.pop(0)
            self.latestAverageLoss = sum(self.recentPrintLosses) / len(self.recentPrintLosses)

            ʕっʘ‿ʘʔっ("increaseScheduledSamplingProb")
            self.scheduledSamplingProb = min(self.scheduledSamplingProb + scheduledSamplingProbIncrement, 1.0)

            ʕっʘ‿ʘʔっ("backward")
            if not torch.isfinite(loss): 
                print("TUTOR.trainStep.backward !!! Loss is NaN or Inf:", loss)
                return
            else: 
                if debugPrints: print("TUTOR.trainStep.backward - loss is not NaN or Inf:", loss)
                
            try:
                model.backward(loss)
            except RuntimeError as e:
                print("TUTOR.trainStep.backward failed!", e)
                #
                return
            
            ʕっʘ‿ʘʔっ("clip_grad_norm")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = gradientClipMaxNorm)
            ʕっʘ‿ʘʔっ("model.optimizer.step")
            model.optimizer.step()

            ʕっʘ‿ʘʔっ("finalActions")
            model.memory.updateMemoryBuffers()

            return loss, predictedTokenIndices, logitSeq
        
    """this iterates through training data, performing forward passes, loss computation, backpropagation, and optimization for each step."""
    def trainModel(self, trainingDataPairs, epochs, startIndex, model):
        torch.autograd.set_detect_anomaly(True)
        with self.counsellor.infodump("trainModel") as ʕっʘ‿ʘʔっ:
            vocab = model.vocab
            if statPrints or debugPrints: print(f"Debug tokenToIndex (First 20): {list(vocab.tokenToIndex.items())[:20]}")
            for name, param in model.named_parameters(): print(name, param.device)
            ʕっʘ‿ʘʔっ("COUNTERS INIT")
            self.trainingStepCounter = 1

            self.stats = Counter({"loss": 0, "gradNorm": 0, "logitMin": 0, "logitMax": 0, "scheduledSampling": 0, "tokenCount": 0,})
            self.stats_100 = Counter({"loss": 0, "gradNorm": 0, "logitMin": 0, "logitMax": 0, "scheduledSampling": 0, "tokenCount": 0,})

            self.tokenCounts = Counter()
            self.tokenCounts_100 = Counter()

            self.avgLoss_100 = 0
            self.avgLoss_1000 = 0

            ʕっʘ‿ʘʔっ("back to school!")
            print("babyLLM is heading back to school...")

            """EPOCH LOOP"""
            ʕっʘ‿ʘʔっ("epoch♥")
            for epoch in range(epochs):
                print(f"--- lesson {epoch+1}/{epochs} started ---")
                """TRAINING DATA (batches)"""
                try:
                    ʕっʘ‿ʘʔっ("♥training♥")
                    for i, (inputSeq, targetSeq) in enumerate(trainingDataPairs):
                        ʕっʘ‿ʘʔっ("♥tokenToIndex")
                        inputTokenIndices = [vocab.tokenToIndex.get(t, vocab.tokenToIndex["<UNK>"]) for t in inputSeq]
                        targetTokenIndexSeq = [vocab.tokenToIndex.get(t, vocab.tokenToIndex["<UNK>"]) for t in targetSeq]

                        ʕっʘ‿ʘʔっ("♥resetMemory")
                        model.resetMemory(context="training")

                        ʕっʘ‿ʘʔっ("♥trainStep")
                        result = self.trainStep(inputTokenIndices, targetTokenIndexSeq, model)
                        loss, predictedTokenIndices, logitSeq = result

                        ʕっʘ‿ʘʔっ("♥basicStats") # CALCULATE BASIC STATS
                        basicStats = model.getBasicStats(logitSeq)
                        basicStats["lossTUTOR"] = loss.item()
                        self.stats_100["lossTUTOR"] += loss.item()
                        model.stats.update(basicStats)

                        if self.trainingStepCounter % saveModelFreq == 0:
                            ʕっʘ‿ʘʔっ("♥autoSave") # SAVE THE MODEL EVERY x STEPS
                            print(self.s_output.S_apply('dim', 'autosaving...') + self.s_output.S_apply('reset', ''))
                            model.saveModel()
                            p = self.trainingStepCounter + saveModelFreq
                            print(self.s_output.S_apply('dim', f"autosave successful! saving every {saveModelFreq} steps, the next autosave will be at step {p}...") + self.s_output.S_apply('reset', ''))

                        if self.trainingStepCounter == 1:
                            ʕっʘ‿ʘʔっ("♥bootPrints") # BOOT PRINTS TO TXT AND TERMINAL
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            runStart = f"\n--- {timestamp} ---\n{model.babyNote_loadCheckpointCheck}\n{model.userNote_loadCheckpoint}\n{model.babyNote_loadCheckpoint}{model.babyNote_runStart}\n{model.userNote_runStart}\n"
                            print(runStart)
                            ʕっʘ‿ʘʔっ("♥printStartLogs")
                            with open(chatLogPath_forHumans, "a") as logFile: logFile.write(runStart)
                            trainingChatLine = f"\n--- {timestamp} --- {model.babyNote_loadCheckpointCheck} - {model.userNote_loadCheckpoint} - {model.babyNote_loadCheckpoint}{model.babyNote_runStart} - {model.userNote_runStart}\n"
                            with open(trainingLogPath_100, "a") as logFile: logFile.write(trainingChatLine)
                            with open(trainingLogPath_1000, "a") as logFile: logFile.write(trainingChatLine)
                            with open(chatLogPath_trainingLog, "a") as logFile: logFile.write(trainingChatLine)
                        
                        if self.trainingStepCounter % printFreq == 0:
                            ʕっʘ‿ʘʔっ("♥printGuessesToTerminal♥") # PRINTING TRAINING OUTPUT TO TERMINAL
                            guessedTokenSeq = [model.getTokenIndexAsString(idx) if idx != -1 else "<UNK>" for idx in predictedTokenIndices]
                            if guessedTokenSeq: 
                                self.tokenCounts_100.update(guessedTokenSeq)
                                self.tokenCounts.update(guessedTokenSeq)
                            recentLoss = sum(self.recentPrintLosses)/len(self.recentPrintLosses) if self.recentPrintLosses else None
                            ʕっʘ‿ʘʔっ("♥S_output.S_colourPrintTraining")
                            self.s_output.S_colourPrintTraining(
                                step=self.trainingStepCounter,
                                inputSeq=inputSeq,
                                guessedSeq_str=guessedTokenSeq,
                                targetSeq_str=targetSeq[:windowMAX],
                                loss=loss,
                                recentLoss = recentLoss,
                                totalLoss=loss,
                                totalTokenCount = 0
                            )
                            ʕっʘ‿ʘʔっ("♥SCRIBE.maybeCommentOnGuess")
                            self.scribe.maybeCommentOnGuess(guessedTokenSeq, loss, "scribe", 0.01)

                        if self.trainingStepCounter % trainingLogFreq_1000 == 0:
                            ʕっʘ‿ʘʔっ("♥trainingLogFreq_1000♥") # PRINTING LOGS TO TXT AND TERMINAL
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            ʕっʘ‿ʘʔっ("♥getComplexStats")
                            complexStats = model.getComplexStats()
                            model.stats.update(complexStats)
                            ʕっʘ‿ʘʔっ("♥calculateTrainingDataRemaining")
                            trainingDataRemaining = len(trainingDataPairs) - self.trainingStepCounter
                            trainingDataPercent = (trainingDataRemaining / len(trainingDataPairs)) * 100
                            print(f"step {self.trainingStepCounter} | tokens remaining: {len(trainingDataPairs) - self.trainingStepCounter} ({trainingDataPercent:.2f}%)")

                            #ʕっʘ‿ʘʔっ("♥durationUpdate")
                            #model.duration.update(model.durationCategories) # Force a noop update to ensure we have every category
                            #ʕっʘ‿ʘʔっ("♥durationLog_1000")
                            #durationLog_1000 = "Durations: " + ", ".join([
                            #    f"{name}: {(duration * 1000 / trainingLogFreq_1000 if trainingLogFreq_1000 > 0 else 0):.2f}ms"
                            #    for name, duration in model.duration.most_common() # most_common with no parameter returns everything, already sorted in reverse
                            #])
                            #model.duration.clear()
                            #with open(durationLogPath_1000, "a") as logFile: logFile.write(durationLog_1000 + "\n")

                            self.actualAvgLoss_1000 = self.avgLoss_1000/trainingLogFreq_1000

                            ʕっʘ‿ʘʔっ("♥getStringStats")
                            stringStats = model.getStringStats(predictedTokenIndices, self.tokenCounts, self.tokenCounts_100, logFreq_100=False)

                            ʕっʘ‿ʘʔっ("♥S_output.S_logTraining")
                            self.s_output.S_logTraining(
                                trainingLogPath = trainingLogPath_1000,
                                trainingStepCounter = self.trainingStepCounter,
                                freq = trainingLogFreq_1000,
                                stats = model.stats,
                                INN_cerebellum_str = stringStats["INN_cerebellum_str"],
                                INN_judgeBias_str = stringStats["INN_judgeBias_str"],
                                INN_credbilityBias_str = stringStats["INN_credibilityBias_str"],
                                topTokens_str = stringStats["topTokens"],
                                otherInfo_str = f"avgLoss/{trainingLogFreq_1000}: {self.actualAvgLoss_1000} | {stringStats['tokenPerfect']} | {stringStats['windowVotes_str']} | TUTOR.py {trainingLogFreq_1000}"
                            )
                            ʕっʘ‿ʘʔっ("♥finalLogActions")
                            model.stats.clear()
                            self.avgLoss_1000 = 0
                            self.perfectTokenCount = 0
                            self.totalTokenEvaluations = 0
                            
                            
                        # Track loss every 100 steps
                        if self.trainingStepCounter % trainingLogFreq_100 == 0:
                            ʕっʘ‿ʘʔっ("♥trainingLogFreq_100♥")
                            ʕっʘ‿ʘʔっ("♥getComplexStats")
                            complexStats = model.getComplexStats()
                            ʕっʘ‿ʘʔっ("♥calculateTrainingDataRemaining")
                            trainingDataRemaining = len(trainingDataPairs) - self.trainingStepCounter
                            trainingDataPercent = (trainingDataRemaining / len(trainingDataPairs)) * 100
                            print(f"step {self.trainingStepCounter} | tokens remaining: {len(trainingDataPairs) - self.trainingStepCounter} ({trainingDataPercent:.2f}%)")

                            #ʕっʘ‿ʘʔっ("♥durationUpdate")
                            #model.duration_100.update(model.durationCategories) # Force a noop update to ensure we have every category
                            #durationLogBabyLLM_inner_100 = (f"inner step {j+1}: forward: {forwardTime*1000:.2f}ms | predict: {predictTime*1000:.2f}ms | loss: {lossTime*1000:.2f}ms")
                            #durationLogBabyLLM_100 = (f"DEBUG: forward() timings: Index: {idxTime*1000:.2f}ms | Embed: {embedTime*1000:.2f}ms | Neuron: {neuronTime*1000:.2f}ms | Memory: {memoryTime*1000:.2f}ms | Output: {outputTime*1000:.2f}ms | Total: {forwardTotal*1000:.2f}ms")
                            #ʕっʘ‿ʘʔっ("♥durationLog_100")
                            #durationLog_100 = "Durations: " + ", ".join([
                            #    f"{name}: {(duration * 1000 / trainingLogFreq_100 if trainingLogFreq_100 > 0 else 0):.2f}ms"
                            #    for name, duration in model.duration_100.most_common()
                            #])
                            #durationLogCombined_100 = f"\n--- {timestamp} --- \n{durationLog_100} \n{durationLogBabyLLM_100} \n{durationLogBabyLLM_inner_100}\n"
                            #with open(durationLogPath_100, "a") as logFile: logFile.write(durationLog_100 + "\n")
                            #model.duration_100.clear()

                            self.actualAvgLoss_100 = self.avgLoss_100/trainingLogFreq_100

                            ʕっʘ‿ʘʔっ("♥getStringStats")
                            stringStats = model.getStringStats(predictedTokenIndices, self.tokenCounts, self.tokenCounts_100, logFreq_100 = True)

                            ʕっʘ‿ʘʔっ("♥S_output.S_logTraining")
                            self.s_output.S_logTraining(
                                trainingLogPath = trainingLogPath_100,
                                trainingStepCounter = self.trainingStepCounter,
                                freq = trainingLogFreq_100,
                                stats = model.stats,
                                INN_cerebellum_str = stringStats["INN_cerebellum_str"],
                                INN_judgeBias_str = stringStats["INN_judgeBias_str"],
                                INN_credbilityBias_str = stringStats["INN_credibilityBias_str"],
                                topTokens_str = stringStats["topTokens"],
                                otherInfo_str = f"avgLoss/{trainingLogFreq_100}: {self.actualAvgLoss_100} | {stringStats['tokenPerfect']} | {stringStats['windowVotes_str']} | TUTOR.py {trainingLogFreq_100}"
                            )

                            ʕっʘ‿ʘʔっ("♥finalLogActions")
                            model.stats.clear()
                            self.perfectTokenCounts_100 = 0
                            self.totalTokenEvaluations_100 = 0
                            self.avgLoss_100 = 0

                        ʕっʘ‿ʘʔっ("♥endTurn")
                        self.trainingStepCounter += 1
                        
                        """END OF ONE TURN"""

                    ʕっʘ‿ʘʔっ("♥finalSaveBeforeNewEpoch")
                    model.saveModel()

                except KeyboardInterrupt:
                    ʕっʘ‿ʘʔっ("♥keyboardInterrupt")
                    choice = input("save, cancel (do not save before exit) or interact?" + f"\n{userName}: ").lower()
                    if choice in ("save", "") or choice.startswith("s"): 
                        ʕっʘ‿ʘʔっ("♥choice = s")
                        model.saveModel()
                        print("\nit's rude to interrupt people.. but, bye bye! :)")
                    elif choice == "cancel" or choice.startswith("c"): 
                        ʕっʘ‿ʘʔっ("♥choice = c")
                        print("\nhey! i wanted to remember that! :(")
                    elif choice == "interact" or choice.startswith("i"):
                        ʕっʘ‿ʘʔっ("♥choice = i")
                        model.saveModel()
                        import code
                        print("try:\nbabyLLM.stats\nbabyLLM.scheduledSamplingProb\nbabyLLM.memory.memory\nbabyLLM.interneuronNetwork.cerebellum\nbabyLLM.logits.forward(...)\nUse `exit()` to return to terminal.\n")
                        code.interact(local=locals())
                    else: 
                        ʕっʘ‿ʘʔっ("♥choice = None")
                        model.saveModel()
                        print("\nuhh... i'm confused, but i saved anyway!")

                    sys.exit(8)

            print("--- tutoring complete! ---")
