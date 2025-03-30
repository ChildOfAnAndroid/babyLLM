# CHARIS CAT 2025

import os
import re
import random
import time
from datetime import datetime
from babyLLM import BABYLLM
from vocab import VOCAB
from config import *
import outputStyles
import archive.logHelpers as logHelpers
import torch
from vocab import VOCAB
from collections import Counter

inputFilePath = "data/CHARIS/trainingData.txt"
outputFilePath = "data/CHARIS/talkToYourselfBattle.txt"
guessFilePath = "data/CHARIS/whoIsMoreLikeYou.txt"
splitTextPattern = r'(?<=[.?!,])\s+'
minimumLength = 3
trainingStepCounter = 0
totalLoss = 0
totalLossDetail = 0
totalLogitMinDetail = 0 
totalLogitMaxDetail = 0 
totalLogitMin = 0       
totalLogitMax = 0  

vocab = VOCAB()
babyLLM = BABYLLM(vocab=vocab, embedDimension=embedDimension, numNeurons=numNeurons, activationFunction=activationFunction)
babyLLM.loadModel("babyLLM.pth")

existingLines = set()
if os.path.exists(outputFilePath):
    with open(outputFilePath, 'r', encoding='utf-8') as f:
        for line in f:
            existingLines.add(line.strip().lower())

with open(inputFilePath, 'r', encoding='utf-8') as f:
    rawText = f.read().lower()
prompts = re.split(splitTextPattern, rawText)
prompts = [q.strip() for q in prompts if len(q.strip()) >= minimumLength]

startIndex = random.randint(0, len(prompts) - 1)
indexes = list(range(startIndex, len(prompts))) + list(range(0, startIndex))

charisCheck = input("are you charis, and here to speak to babydroid? (y/n):").strip().lower()
if charisCheck == "y":
    userName = "charis"
    ghostName = "babydroid"
    babyName = "babyllm"
    print(f"i guessed it! it's nice to see you again :)")
else:
    userName = input("oh, sorry! who are you?: ").strip().lower() or "charis"
    ghostName = input("who are you here to talk to?: ").strip().lower() or "babydroid"
    babyName = "babyllm"
    print(f"ok {userName}, let's get started :)")

print(f"\nstarting at {startIndex + 1} of {len(prompts)}")
print("type '!wait' or 'w' to skip, and '!quit' or 'q' to exit.\n")

log = []
context = []
waitTimeSeconds = 20

trainingStepCounter = 0
totalLoss = 0
totalLossDetail = 0

def compareAnswersSimilarity(userAnswerText, ghostAnswerText):
    userAnswerWords = userAnswerText.split()
    ghostAnswerWords = ghostAnswerText.split()
    matches = sum(1 for x, y in zip(userAnswerWords, ghostAnswerWords) if x == y)
    return matches / max(len(userAnswerWords), 1)

def trainOnAnswer(inputText, targetText):
    global trainingStepCounter, totalLoss, totalLossDetail, waitTimeSeconds, totalLogitMinDetail, totalLogitMaxDetail, totalLogitMin, totalLogitMax, totalGradNormDetail, totalGradNorm, totalWindowWeightsDetail, totalWindowWeights, totalMemGatesDetail, totalMemGates

    inputEncoding = vocab.tokenizer.encode(inputText)
    inputTokens = inputEncoding.ids
    targetEncoding = vocab.tokenizer.encode(targetText)
    targetTokens = targetEncoding.ids
    trainingPairs = [(inputTokens, targetTokens)]

    for inputSeq, targetSeq in trainingPairs:
        predictedIndices = []
        inputSeqPredictions = list(inputSeq)
        cumLoss = 0
        logitSeq = []
        memGatesSeq = []

        for j in range(len(targetSeq)):
            logits = babyLLM.forward(inputSeqPredictions)
            logitSeq.append(logits)
            predictedIndex = babyLLM.getResponseFromLogits(logits)
            predictedIndices.append(predictedIndex)
            loss = babyLLM.computeLoss(logits, targetSeq[j])
            babyLLM.backward(loss)
            cumLoss += loss.item()

            inputSeqPredictions.append(targetSeq[j])

        avgLoss = cumLoss / len(targetSeq)
        guessedTokens = [vocab.indexToToken.get(idx, '<UNK>') for idx in predictedIndices]
        targetTokensStr = [vocab.indexToToken.get(idx, '<UNK>') for idx in targetSeq]
        guessedTokensStr = ' '.join(guessedTokens)
        targetTokensStrJoined = ' '.join(targetTokensStr)

        trainingStepCounter += 1
        totalLoss += loss.item()
        totalLossDetail += loss.item()

        with torch.no_grad():
            logitsTensor = torch.cat(logitSeq, dim=0)
            logitMin = logitsTensor.min(dim=-1).values.mean().item()
            logitMax = logitsTensor.max(dim=-1).values.mean().item()
            totalLogitMinDetail += logitMin 
            totalLogitMaxDetail += logitMax 
            totalLogitMin += logitMin 
            totalLogitMax += logitMax

            gradNorm = torch.nn.utils.clip_grad_norm_(babyLLM.parameters(), max_norm = gradientClipMaxNorm, norm_type=2.0).item()
            totalGradNormDetail += gradNorm
            totalGradNorm += gradNorm

            normWeights = (babyLLM.parallelNeuronLayer.windowWeighting + 0.1)
            normWeights /= (normWeights.sum() + 0.1)
            sortedWeights = sorted(
                zip(allWindowSizes, normWeights.cpu().numpy()),
                key=lambda x: x[1],
                reverse=True
            )
            windowWeights_str = "  ".join(f"W{wsize}:{weight:.5f}" for wsize, weight in sortedWeights)
            totalWindowWeightsDetail += normWeights.max().item()
            totalWindowWeights += normWeights.max().item()

            memGatesTensor = babyLLM.memoryLayer.latestMemoryGates.detach()
            if memGatesTensor is not None:
                memGates_str = f"Short:{memGatesTensor[0]:.3f}, Long:{memGatesTensor[1]:.3f}, Current:{memGatesTensor[2]:.3f}"
                totalMemGatesDetail += memGatesTensor.mean().item()
                totalMemGates += memGatesTensor.mean().item()
            else:
                memGates_str = "N/A"

        isCorrect = (guessedTokensStr.strip() == targetTokensStrJoined.strip())
        isPerfect = isCorrect and avgLoss < veryLowLoss

        outputStyles.colourPrintTraining(
            step=trainingStepCounter,
            inputSentence=inputText,
            guessedSeqStr=guessedTokensStr,
            targetSeqStr=targetTokensStrJoined,
            loss=avgLoss,
            isCorrect=isCorrect,
            isPerfect=isPerfect
        )

        similarity = compareAnswersSimilarity(inputText, guessedTokensStr)

        """PRINTING LOSS TO LOGS AND TERMINAL"""
        if trainingStepCounter == 0:
            userNote = input("what am i learning today?").strip()
            scheduledSamplingProb += scheduledSamplingProbIncrement
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            runStart = f"\n--- {timestamp} ---"
            runStart += f"\nbabyLLM: what am i learning today?"
            runStart += f"\nYou: {userNote}\n"
            print(f"{runStart.strip()}")
            with open("trainingLogDetail.txt", "a") as logFile:
                logFile.write(runStart)
            with open("trainingLog.txt", "a") as logFile:
                logFile.write(runStart)

        if trainingStepCounter % printLossFreq == 0:
            avgLoss = totalLoss / printLossFreq
            avgLogitMin = totalLogitMin / printLossFreq
            avgLogitMax = totalLogitMax / printLossFreq
            avgGradNorm = totalGradNorm / printLossFreq
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            avgGradNorm = totalGradNorm / printLossFreq
            avgGuessSimilarity = similarity
            avgWindowWeights = totalWindowWeights / printLossFreq
            avgMemGates = totalMemGates / printLossFreq
            avgGuessSimilarity = similarity

            outputStyles.logTraining(
                logFilePath=logFilePath,
                step=trainingStepCounter,
                avgLoss=avgLoss,
                learningRate=learningRate,
                logitRange_str=f"{avgLogitMin:.2f} → {avgLogitMax:.2f}",
                windowWeights_str=windowWeights_str,
                gradientNorm_str=f"{avgGradNorm:.3f}",
                scheduledSamplingProb_str = "",
                epoch_str = "",
                prompt = "",
                guess = "",
                truth = "",
                memGates_str=memGates_str,
                topTokens_str = "",
                durationLog_str = "",
                #guessSimilarity_str=f"{avgGuessSimilarity:.2f}",
                otherInfo=f"TalkToYourself Training",
            )

            totalLoss = 0
            totalLogitMin = 0
            totalLogitMax = 0
            totalGradNorm = 0
            totalWindowWeights = 0
            totalMemGates = 0

        if trainingStepCounter % printLossFreqDetail == 0:
            avgLossDetail = totalLossDetail / printLossFreqDetail
            avgLogitMinDetail = totalLogitMinDetail / printLossFreqDetail
            avgLogitMaxDetail = totalLogitMaxDetail / printLossFreqDetail
            timestampDetail = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            avgGradNormDetail = totalGradNormDetail / printLossFreq
            avgGuessSimilarityDetail = similarity

            outputStyles.logTraining(
                logFilePath="trainingLogDetail.txt",
                step=trainingStepCounter,
                avgLoss=avgLossDetail,
                learningRate=learningRate,
                logitRange_str=f"{avgLogitMinDetail:.2f} → {avgLogitMaxDetail:.2f}",
                windowWeights_str=windowWeights_str,
                gradientNorm_str=f"{avgGradNormDetail:.3f}",
                scheduledSamplingProb_str = "",
                epoch_str = "",
                prompt = "",
                guess = "",
                truth = "",
                memGates_str=memGates_str,
                topTokens_str = "",
                durationLog_str = "",
                #guessSimilarity_str=f"{avgGuessSimilarityDetail:.2f}",
                otherInfo=f"TalkToYourself Training Detail",
            )
            
            totalLossDetail = 0
            totalLogitMinDetail = 0 
            totalLogitMaxDetail = 0

        """SAVE THE MODEL EVERY x STEPS"""
        if trainingStepCounter % saveModelFreq == 0:
            print(f"{outputStyles.S_apply('dim', "autosaving...")}{outputStyles.S_apply('reset', "")}")
            babyLLM.saveModel()
            success = f"autosave successful! saving every {saveModelFreq} steps, the next autosave will be at step {trainingStepCounter+saveModelFreq}..."
            print(f"{outputStyles.S_apply('dim', success)}{outputStyles.S_apply('reset', "")}")

if __name__ == "__main__":
    for idx in indexes:
        prompt = prompts[idx]
        originalLine = f"[{ghostName}]: {prompt}"
        if originalLine in existingLines:
            context.append(originalLine)
            continue

        print(f"\n[{ghostName}]: {prompt}")
        try:
            print(f"[{userName}] (waiting {waitTimeSeconds}s): ", end='', flush=True)
            start = time.time()
            userInput = None
            while True:
                if time.time() - start + 3 > waitTimeSeconds:
                    print("\n3... ")
                elif time.time() - start + 2 > waitTimeSeconds:
                    print("2... ")
                elif time.time() - start + 1 > waitTimeSeconds:
                    print("1... ")
                elif time.time() - start > waitTimeSeconds:
                    print("too slow! i'll just do it myself!\n")
                    inputText = " ".join(q.split("]: ")[1].strip() for q in context[-windowMAX:] + [originalLine])
                    #trainOnAnswer(inputText, prompt)
                    trainingDataPairs = vocab.genTrainingData(windowMAX)
                    BABYLLM.trainModel(trainingDataPairs, epochs = 1)
                    waitTimeSeconds = max(0, waitTimeSeconds // 2)
                    context.append(originalLine)
                    break
                if os.name == 'nt':
                    import msvcrt
                    if msvcrt.kbhit():
                        userInput = input().strip().lower()
                        break
                else:
                    import select
                    import sys
                    if select.select([sys.stdin], [], [], 1)[0]:
                        userInput = input().strip().lower()
                        break

        except KeyboardInterrupt:
            print("\nit's rude to interrupt people.. but, bye bye! :)")
            babyLLM.saveModel()
            break

        if userInput is None:
            continue
        if userInput in ('!quit', 'q'):
            print(f"bye bye :)")
            break
        elif userInput in ('!wait', 'w'):
            context.append(originalLine)
            continue

        inputText = " ".join(q.split("]: ")[1].strip() for q in context[-windowMAX:] + [originalLine])
        encoding = vocab.tokenizer.encode(inputText)
        inputTokens = encoding.ids
        outputTokens = []
        for _ in range(numTokensPerStep):
            nextToken = babyLLM.getNextToken(inputTokens)
            inputTokens.append(nextToken)
            outputTokens.append(vocab.indexToToken.get(nextToken, '<UNK>'))
        babyGuess = ''.join(outputTokens).replace('Ġ', ' ').strip()
        babyGuess = ' '.join(babyGuess.split())

        similarity = compareAnswersSimilarity(userInput, babyGuess)
        print(f"[{babyName}]: {babyGuess}")
        print(f"[{userName}]: {userInput} (similarity: {similarity:.2f})")

        log.append(f"[{ghostName}]: {prompt}")
        log.append(f"[{userName}]: {userInput}")
        context.append(f"[{ghostName}]: {prompt}")
        context.append(f"[{userName}]: {userInput}")

        with open(guessFilePath, 'a', encoding='utf-8') as g:
            g.write(f"PROMPT: {inputText.strip()}\n")
            g.write(f"USER: {userInput.strip()}\n")
            g.write(f"BABYLLM: {babyGuess}\n")
            g.write(f"SIMILARITY: {similarity:.2f}\n")
            g.write("-" * 40 + "\n")

        trainOnAnswer(inputText, userInput)
        waitTimeSeconds = 20

    if log:
        with open(outputFilePath, 'a', encoding='utf-8') as f:
            for line in log:
                f.write(line + "\n")
        print(f"\nsaved {len(log)//2} message pairs to {outputFilePath}")
    else:
        print("\nnothing saved.")