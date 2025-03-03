from babyLLM import BABYLLM
from vocab import VOCAB
from config import *
import torch

def chat(babyLLM, vocab):
    inputSentence = input("What do you say? ")

    if inputSentence.startswith("!exit"):
        exit(0)

    inputSeq = vocab.nltkTokenizer(inputSentence)
    output = []
    for _ in range(10):
        guessedToken = babyLLM.getNextToken(inputSeq)
        inputSeq.append(guessedToken)
        output.append(babyLLM.getReadableToken(guessedToken))

    print(f"You said: \"{inputSentence}\", Full response: {" ".join(output)}")
    
if __name__ == "__main__":
    vocab = VOCAB(vocabSize = vocabSize)

    babyLLM = BABYLLM(vocab = vocab, embedDimension = embedDimension, numNeurons = numNeurons, activationFunction = activationFunction)
    babyLLM.loadModel("babyLLM_epoch0_0.pth")
    while True:
        chat(babyLLM, vocab)