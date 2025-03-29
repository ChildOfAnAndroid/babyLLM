# CHARIS CAT 2025

import os
import re
import random
import time
from datetime import datetime
from babyLLM import BABYLLM
from vocab import VOCAB
from config import *
from outputStyles import *

inputFilePath = "data/CHARIS/trainingData.txt"
outputFilePath = "data/CHARIS/talkToYourselfBattle.txt"
guessFilePath = "data/CHARIS/whoIsMoreLikeYou.txt"
splitTextPattern = r'(?<=[.?!,])\s+'
minimumLength = 3

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

userName = input("who are you?: ").strip().lower() or "charis"
ghostName = input("who are you talking to? (babydroid?): ").strip().lower() or "babydroid"
babyName = "babyllm"

print(f"\nstarting at {startIndex + 1} of {len(prompts)}")
print("type '!wait' or 'w' to skip, and '!quit' or 'q' to exit.\n")

log = []
context = []
waitTimeSeconds = 20

trainingStepCounter = 0
totalLoss = 0
totalLoss2 = 0


def compareAnswersSimilarity(userAnswerText, ghostAnswerText):
    userAnswerWords = userAnswerText.split()
    ghostAnswerWords = ghostAnswerText.split()
    matches = sum(1 for x, y in zip(userAnswerWords, ghostAnswerWords) if x == y)
    return matches / max(len(userAnswerWords), 1)

def trainOnAnswer(inputText, targetText):
    global trainingStepCounter, totalLoss, totalLoss2, waitTimeSeconds

    inputEncoding = vocab.tokenizer.encode(inputText)
    inputTokens = inputEncoding.ids
    targetEncoding = vocab.tokenizer.encode(targetText)
    targetTokens = targetEncoding.ids
    trainingPairs = [(inputTokens, targetTokens)]

    for inputSeq, targetSeq in trainingPairs:
        predictedIndices = []
        inputSeqPredictions = list(inputSeq)
        currentStepLoss = 0  # Loss for the current training step

        for j in range(len(targetSeq)):
            logits = babyLLM.forward(inputSeqPredictions)
            predictedIndex = babyLLM.getResponseFromLogits(logits)
            predictedIndices.append(predictedIndex)
            loss = babyLLM.computeLoss(logits, targetSeq[j])
            babyLLM.backward(loss)
            currentStepLoss += loss.item()

            inputSeqPredictions.append(targetSeq[j])

        avgLoss = currentStepLoss / len(targetSeq) # Average loss for the current training example
        guessedTokens = [vocab.indexToToken.get(idx, '<UNK>') for idx in predictedIndices]
        targetTokensStr = [vocab.indexToToken.get(idx, '<UNK>') for idx in targetSeq]
        guessedTokensStr = ' '.join(guessedTokens)
        targetTokensStrJoined = ' '.join(targetTokensStr)

        trainingStepCounter += 1
        totalLoss += avgLoss
        totalLoss2 += avgLoss

        isCorrect = (guessedTokensStr.strip() == targetTokensStrJoined.strip()) 
        isPerfect = isCorrect and avgLoss < 0.01 

        colourPrintTraining(
            step=trainingStepCounter,
            inputSentence=inputText,
            guessedSeqStr=guessedTokensStr,
            targetSeqStr=targetTokensStrJoined,
            loss=avgLoss,
            isCorrect=isCorrect,
            isPerfect=isPerfect
        )
        print("\n")

        if trainingStepCounter % printLossFreq == 0:
            avgLoss = totalLoss / printLossFreq
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logOutput = (
                f"{timestamp} | Step {trainingStepCounter:<6} | Avg Loss (last {printLossFreq}): {avgLoss:.4f}"
            )
            print(f"{DIM}DETAIL{RESET} {logOutput}")
            with open("trainingLogDetail.txt", "a") as logFileDetail:
                logFileDetail.write(logOutput + "\n")
            totalLoss = 0

        if trainingStepCounter % printLossFreqDetail == 0:
            avgLoss2 = totalLoss2 / printLossFreqDetail
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logOutput = (
                f"{timestamp} | Step {trainingStepCounter:<6} | Avg Loss (last {printLossFreqDetail}): {avgLoss2:.4f}"
            )
            print(logOutput)
            with open("trainingLog.txt", "a") as logFile:
                logFile.write(logOutput + "\n")
            totalLoss2 = 0


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
            if time.time() - start + 2 > waitTimeSeconds:
                print("2... ")
            if time.time() - start + 1 > waitTimeSeconds:
                print("1... ")
            if time.time() - start > waitTimeSeconds:
                print("too slow! i'll just do it myself!]")
                inputText = " ".join(q.split("]: ")[1].strip() for q in context[-windowMAX:] + [originalLine])
                trainOnAnswer(inputText, prompt)
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