# Charis Cat 2025
from collections import Counter
from config import *

# LOAD TRAINING DATA
def loadTrainingDataFUNK(filepaths):
    loadTrainingData = ""
    for filepaths in filepaths:
        with open(filepaths, 'r', encoding='utf-8') as f:
            text = f.read()
            loadTrainingData += text + " "
    return loadTrainingData

# WHITESPACE TOKENISER
def whitespaceTokenizer(text):
    textLower = text.lower() # lowercase conversion
    tokens = textLower.split()
    return tokens

# TOP 2000 TOKENS
def buildVocab(tokens, vocabSize):
    wordCounts = Counter(tokens)
    mostCommonWords = wordCounts.most_common(vocabSize)
    vocabList = [word for word, count in mostCommonWords]
    return vocabList

# DICTIONARYS
def createTokenToIndex(vocabList):
    tokenToIndex = {token: index for index, token in enumerate(vocabList)}
    return tokenToIndex

def createIndexToToken(vocabList):
    indexToToken = {index: token for index, token in enumerate(vocabList)}
    return indexToToken

# EXAMPLE RUNNING
if __name__ == "__main__":
    #loading data
    dataFilepaths = ["data/CHARIS/CHARISENTIREclean.txt",
                      "data/CHARIS/DISSERTATIONONAI.txt"]
                     #"data/GEEPYENTIRE_1.txt", 
                     #"data/GEEPYENTIRE_2.txt", 
                     #"data/GEEPYENTIRE_3.txt", 
                     #"data/GEEPYENTIRE_4.txt", 
                     #"data/GEEPYENTIRE_5.txt", 
                     #"data/GEEPYENTIRE_6.txt", 
                     #"data/GEEPYENTIRE_7.txt", 
                     #"data/GEEPYENTIRE_8.txt", 
                     #"data/GEEPYENTIRE_9.txt", 
                     #"data/GEEPYENTIRE_10.txt"]
    trainingData = loadTrainingDataFUNK(dataFilepaths)
    print(f"Loaded {len(trainingData)} characters of training data.")

    #tokenisation
    tokens = whitespaceTokenizer(trainingData)
    print(f"Created {len(tokens)} tokens.")
    print(f"First 20 tokens: {tokens[:20]}")

    #build vocabulary
    vocab = buildVocab(tokens, vocabSize)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Vocabulary words: {vocab[:vocabSize]}")

    #create token/index dictionarys
    tokenToIndex = createTokenToIndex(vocab)
    indexToToken = createIndexToToken(vocab)
    print(f"Token to index mapping (first 10): {dict(list(tokenToIndex.items())[:10])}")
    print(f"Index to token mapping (first 10): {dict(list(indexToToken.items())[:10])}")
