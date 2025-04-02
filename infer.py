from babyLLM import BABYLLM
from vocab import VOCAB
from datetime import datetime
from config import *
import torch

def chat(babyLLM, vocab):
    try:
        userInput = input("What do you say? ")
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get timestamp
        chatStart = f"\n--- {timestamp} ---\n"
        print(f"{chatStart.strip()}")
        with open("chatLog.txt", "a") as logFile:
            logFile.write(chatStart)
        if len(userInput) == 0:
            return
    except EOFError:
        userInput = "!exit"
    exitAfter = False 
    if userInput.startswith("!exit"):
        exitAfter = True
        userInput = "good bye !" # You have to say goodbye to your helpful AIs before leaving! (or is that only for Luigi boards?)

    #inputSeq = vocab.nltkTokenizer(userInput)
    #inputSeq = vocab.huggingTokenizer(userInput)
    encoding = vocab.tokenizer.encode(userInput)
    inputTokens = encoding.ids  # Get token indices
    print(f"DEBUG: Tokenized Input -> {inputTokens}")
    outputTokens = []
    for _ in range(windowMAX):
        guessedToken = babyLLM.getNextToken(inputTokens)
        #print(f"DEBUG: Guessed Token Index -> {guessedToken}"
        inputTokens.append(guessedToken)  # Append index, NOT string
        guessedTokenStr = vocab.indexToToken.get(guessedToken, "<UNK>")
        print(f"{guessedTokenStr} ({guessedToken})")
        outputTokens.append(guessedTokenStr)

    response = ''.join(outputTokens).replace('Ġ', ' ').strip() # replace Ġ with space
    response = ' '.join(response.split())  # remove extra spaces
    if exitAfter:
        print(f'Good bye babyLLM! BabyLLM\'s response: "{response}"')
        exit(0)
    #print(f"You said: \"{userInput}\", Full response: {" ".join(output)}")
    chatLog = f"You: {userInput}\nBabyLLM: {response}\n"
    print(f"{chatLog.strip()}")
    with open("chatLog.txt", "a") as logFile:
        logFile.write(chatLog)

    
if __name__ == "__main__":
    vocab = VOCAB()

    babyLLM = BABYLLM(vocab = vocab, embedDimension = embedDimension, numNeurons = numNeurons, activationFunction = activationFunction)
    babyLLM.loadModel()

    while True:
        chat(babyLLM, vocab)
