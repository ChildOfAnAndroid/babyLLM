# Charis Cat 2025
from collections import Counter
from config import *
from nltk.tokenize import word_tokenize

class VOCAB:
    def __init__(self, vocabSize):
        self.vocabSize = vocabSize
        self.vocabList = []
        self.tokenToIndex = {}
        self.indexToToken = {}
        self.trainingData = self.loadTrainingDataFUNK(dataFilepaths)
        #self.tokens = self.whitespaceTokenizer(self.trainingData)
        self.tokens = self.nltkTokenizer(self.trainingData)
        self.vocabList = self.buildVocab()
        self.tokenToIndex = self.createTokenToIndex()
        self.indexToToken = self.createIndexToToken()

    # LOAD TRAINING DATA
    def loadTrainingDataFUNK(self, filepaths):
        loadTrainingData = ""
        for filepaths in filepaths:
            with open(filepaths, 'r', encoding='utf-8') as f:
                text = f.read()
                loadTrainingData += text + " "
        return loadTrainingData

    # WHITESPACE TOKENISER
    def whitespaceTokenizer(self, text):
        textLower = text.lower() # lowercase conversion
        tokens = textLower.split()
        return tokens
    
    def nltkTokenizer(self, text):
        text_lower = text.lower()
        tokens = word_tokenize(text_lower)
        return tokens

    # TOP 2000 TOKENS
    def buildVocab(self):
        wordCounts = Counter(self.tokens)
        mostCommonWords = wordCounts.most_common(self.vocabSize)
        vocabList = [word for word, count in mostCommonWords]
        return vocabList

    # DICTIONARYS
    def createTokenToIndex(self):
        tokenToIndex = {token: index for index, token in enumerate(self.vocabList)}
        return tokenToIndex

    def createIndexToToken(self):
        indexToToken = {index: token for index, token in enumerate(self.vocabList)}

        #indexToToken = {}
        #for index, token in enumerate(self.vocabList):
        #    indexToToken[index] = token

        return indexToToken
    
    def genTrainingData(self, trainingWindow=2): # Example window size: input sequence of 2 words
        trainingData = []
        for i in range(trainingWindow, len(self.tokens)):
            inputSeq = self.tokens[i-trainingWindow:i]
            target = self.tokens[i]
            # Ensure all tokens in the sequence and target are in our vocabulary
            if all(token in self.vocabList for token in inputSeq) and target in self.vocabList:
                trainingData.append((inputSeq, target))
        return trainingData

# EXAMPLE RUNNING
if __name__ == "__main__":

    vocab = VOCAB(vocabSize = vocabSize)

    #loading data
    #dataFilepaths = ["data/CHARIS/CHARISENTIREclean.txt",
                     #"data/CHARIS/DISSERTATIONONAI.txt"]
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
    #trainingData = loadTrainingDataFUNK(dataFilepaths)
    #print(f"Loaded {len(trainingData)} characters of training data.")

    #tokenisation
    #tokens = whitespaceTokenizer(trainingData)
    #print(f"Created {len(tokens)} tokens.")
    #print(f"First 20 tokens: {tokens[:20]}")

    #build vocabulary
    #vocabList = buildVocab(tokens, vocabSize)
    print(f"Vocabulary size: {len(vocab.vocabList)}")
    print(f"---1701-2000---: {vocab.vocabList[1701:2000]}")
    print(f"---1001-1700---: {vocab.vocabList[301:1700]}")
    print(f"---301-1000---: {vocab.vocabList[301:1000]}")
    print(f"---101-300---: {vocab.vocabList[101:300]}")
    print(f"---Top 100---: {vocab.vocabList[:100]}")

    #create token/index dictionarys
    #tokenToIndex = createTokenToIndex(vocabList)
    #indexToToken = createIndexToToken(vocabList)
    #print(f"Token to index mapping (first 10): {dict(list(tokenToIndex.items())[:10])}")
    #print(f"Index to token mapping (first 10): {dict(list(indexToToken.items())[:10])}")
