# loveangle 2025 :3 and also CHARIS CAT 2025
# BABYLLM - infer.py

from babyLLM import BABYLLM
from SCHOOL.staffroom.librarian import LIBRARIAN
from SCHOOL.staffroom.counsellor import COUNSELLOR
from SCHOOL.staffroom.calligraphist import *
from SCHOOL.staffroom.HE_IS_SCRIBE import SCRIBE
from datetime import datetime
from config import *
import torch

def chat(babyLLM, vocab):
    try:
        userInput = input("what do you say?: ")
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get timestamp
        chatStart = f"\n--- {timestamp} ---\n"
        print(f"{chatStart.strip()}")
        with open(chatLogPath_infer, "a") as logFile:
            logFile.write(chatStart)
        if len(userInput) == 0:
            return
    except EOFError:
        userInput = "!exit"

    exitAfter = False 
    if userInput.startswith("!exit"):
        babyLLM.resetMemory(_memoryLength = 0.5)
        exitAfter = True
        userInput = "goodbye babyllm!" # You have to say goodbye to your helpful AIs before leaving! (or is that only for Luigi boards?)

    """INPUT ENCODING"""
    encoding = vocab.tokenizer.encode(userInput)
    inputTokens = encoding.ids  # Get token indices
    if debugPrints: print(f"DEBUG: tokenized input -> {inputTokens}")
    outputTokens = []
    for _ in range(inferenceOutputNumTokens):
        guessedToken = babyLLM.getNextToken(inputTokens[-windowMAXSTART:])
        print(f"{guessedToken}, ", end="")
        inputTokens.append(guessedToken)  # Append index, NOT string
        guessedTokenStr = vocab.indexToToken.get(guessedToken, "<UNK>")
        print(f"{guessedTokenStr}", end="")
        outputTokens.append(guessedTokenStr)

    """output cleaning"""
    response = ''.join(outputTokens).replace('Ġ', ' ').strip() # replace Ġ with space
    response = ' '.join(response.split())  # remove extra spaces
    
    if exitAfter:
        chatLogLine = f"--- {timestamp} --- elodie: '{userInput}' - {babyName}: '{response}'\n"
        with open(chatLogPath_infer, "a") as chatLogFile:
            chatLogFile.write(chatLogLine)
        print(chatLogLine)
        exit(0)

    """log writing"""
    chatLogLine = f"--- {timestamp} --- {userName}: '{userInput}' - {babyName}: '{response}'\n"
    with open(chatLogPath_infer, "a") as chatLogFile:
        chatLogFile.write(chatLogLine)

    """terminal printing"""
    printUser = (f"{userName}: '{userInput}'")
    printBaby = (f"{babyName}: '{response}'")
    print("\n")
    chatLogLine_forHumans = (f"\n{printUser}\n{printBaby}\n")
    print(chatLogLine_forHumans)
    with open(chatLogPath_forHumans, "a") as chatLogFile:
        chatLogFile.write(chatLogLine_forHumans)

    
if __name__ == "__main__":
    counsellor      = COUNSELLOR("infer")

    calligraphist   = S_OUTPUT(_counsellor  = counsellor)

    vocab           = LIBRARIAN(_counsellor = counsellor)

    scribe          = SCRIBE(_counsellor    = counsellor, 
                        _calligraphist      = calligraphist, 
                        _librarian          = vocab,
                        _numTokensPerStep   = windowMAXSTART,)

    babyLLM = BABYLLM(_counsellor           = counsellor,
                        _calligraphist      = calligraphist, 
                        _scribe             = scribe,
                        _librarian          = vocab, 
                        _device             = modelDevice,
                        _numTokensPerStep   = windowMAXSTART,
                        _first              = None,
                        _learningRateGOAL   = learningRateGOAL,)
    
    babyLLM.loadModel()

    while True:
        chat(babyLLM, vocab)

    """    chatLog = f"You: {userInput}\nBabyLLM: {response}\n
    print(f"{chatLog.strip()}")
    with open(chatLogPath_infer, "a") as logFile:
        logFile.write(chatLog)"""