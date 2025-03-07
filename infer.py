from babyLLM import BABYLLM
from vocab import VOCAB
from config import *
import torch

def chat(babyLLM, vocab):
    inputSentence = input("What do you say? ")
    if len(inputSentence) == 0:
        return
    exitAfter = False 
    if inputSentence.startswith("!exit"):
        exitAfter = True
        inputSentence = "good bye !" # You have to say goodbye to your helpful AIs before leaving! (or is that only for Luigi boards?)

    #inputSeq = vocab.nltkTokenizer(inputSentence)
    inputSeq = vocab.huggingTokenizer(inputSentence)
    output = []
    for _ in range(10):
        guessedToken = babyLLM.getNextToken(inputSeq)
        inputSeq.append(guessedToken)
        output.append(babyLLM.getReadableToken(guessedToken))

    if exitAfter:
        print(f"Good bye babyLLM! Full response: {" ".join(output)}")
        exit(0)
    #print(f"You said: \"{inputSentence}\", Full response: {" ".join(output)}")
    response = ''.join(output).replace('Ġ', ' ').strip() # replace Ġ with space
    response = ' '.join(response.split())  # remove extra spaces
    print(f'You said: "{inputSentence}"\nBabyLLM says: "{response}"')

    
if __name__ == "__main__":
    vocab = VOCAB(vocabSize = vocabSize, vocabPath="vocabCache/vocab_2000")

    babyLLM = BABYLLM(vocab = vocab, embedDimension = embedDimension, numNeurons = numNeurons, activationFunction = activationFunction)
    babyLLM.loadModel("babyLLM.pth")
    while True:
        chat(babyLLM, vocab)