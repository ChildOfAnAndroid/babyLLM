# Charis Cat 2025
from collections import Counter
from config import *
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import json

nltk.download('punkt')
nltk.download('wordnet')

class VOCAB:
    def __init__(self, vocabSize):
        self.vocabSize = vocabSize-1
        self.vocabList = []
        self.tokenToIndex = {}
        self.indexToToken = {}
        self.lemmatizer = WordNetLemmatizer()
        self.unkToken = "<UNK>"

        self.vocabCache = "vocabCache" # Directory to store vocab files
        self.vocabFilename = f"vocab_{vocabSize}"
        #self.vocabFilename = f"vocab_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
        self.vocabListFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_list.json")
        self.tokenToIndexFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_to_index.json")
        self.indexToTokenFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_to_token.json")

        if not os.path.exists(self.vocabCache):
            os.makedirs(self.vocabCache)

        if self.loadVocab():
            print(f"Loaded vocabulary from cache directory: {self.vocabCache}")
            self.trainingData = self.loadTrainingDataFUNK(dataFilepaths)
            print(f"Debug: tokenToIndex keys (first 20): {list(self.tokenToIndex.keys())[:20]}")  # 🔥 PRINT DICTIONARY KEYS
            self.tokens = self.nltkTokenizer(self.trainingData)
        else:
            print(f"Building vocabulary from training data (vocab size: {vocabSize})...")
            self.trainingData = self.loadTrainingDataFUNK(dataFilepaths)
            self.initTokens = self.nltkTokenizerInit(self.trainingData)
            self.tokens = self.initTokens
            self.vocabList = self.buildVocab()
            self.tokenToIndex = self.createTokenToIndex()
            self.indexToToken = self.createIndexToToken()
            self.vocabList.append(self.unkToken)  # Add UNK to vocab list
            self.tokenToIndex[self.unkToken] = len(self.vocabList) - 1
            self.indexToToken[len(self.vocabList) - 1] = self.unkToken
            self.tokens = self.nltkTokenizer(self.trainingData) # Tokenize with UNK
            self.saveVocab() # Save vocabulary to files
            print(f"Saved vocabulary to cache directory: {self.vocabCache}")
        print(f"Debug VOCAB.__init__: Length of vocabList AFTER buildVocab: {len(self.vocabList)}")
        print(f"Debug VOCAB.__init__: First 20 tokens in vocabList: {self.vocabList[:20]}")
        # That variable is NOT USED right now, the babyLLM file regenerates it before running the training...
        # That'll speed up the infer loading
        # self.trainingDataGen = self.genTrainingData(trainingWindow)

    # LOAD TRAINING DATA
    def loadTrainingDataFUNK(self, filepaths):
        loadTrainingData = ""
        for filepaths in filepaths:
            with open(filepaths, 'r', encoding='utf-8') as f:
                text = f.read()
                loadTrainingData += text + " "
        return loadTrainingData


    
    def splitPlural(self, token):
        if token.endswith("s") and len(token) > 3:  # Avoid "is", "us", "as"
            print(f"does this ever run? (token ends with s)")
            singular = self.lemmatizer.lemmatize(token)
            
            if singular in self.vocabList:
                print(f"does this ever run? (split plural when singular)")
                # Handle special plural cases
                if token.endswith("ies") and singular.endswith("y"):  # "parties" → ["party", "ies"]
                    return [singular, "ies"]
                elif token.endswith("es") and singular not in ["she", "he"]:  # "boxes" → ["box", "es"]
                    return [singular, "es"]
                else:  # Default case: "cats" → ["cat", "s"]
                    return [singular, "s"]
                
        return [token]

    def nltkTokenizerInit(self, text):
        textLower = text.lower()
        tokens = word_tokenize(textLower)
        initTokens = []
        for token in tokens:
            pluralSplit = self.splitPlural(token)
            initTokens.extend(pluralSplit)

        return initTokens
    
    def nltkTokenizer(self, text):
        textLower = text.lower()
        tokens = word_tokenize(textLower)
        processedTokens = [] # New list to hold processed tokens
        for token in tokens:
            if token in self.vocabList: # Check if token is in vocab
                processedTokens.append(token) # Use token as is if in vocab
            else:
                processedTokens.append(self.unkToken) # Replace with <UNK> if OOV
        return processedTokens # Return processed tokens list


    # TOP 1999 TOKENS + UNK
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
    
    def genTrainingData(self, trainingWindow):
        trainingData = []
        for i in range(trainingWindow, len(self.tokens) - trainingWindow):
            inputSeq = self.tokens[i-trainingWindow:i]
            target = self.tokens[i]
            # Check tokens in the sequence are in our vocabulary
            if all(token in self.vocabList for token in inputSeq) and target in self.vocabList:
                trainingData.append((inputSeq, target))
                #print(f"Debug VOCAB.genTrainingData: Added example: Input Seq: {inputSeq}, Target: {target}") # ADDED
            else:
                print(f"Debug VOCAB.genTrainingData: Skipping UNK example - Input Seq: {inputSeq}, Target: {target}") 
        return trainingData
    
    def loadVocab(self):
        try:
            # Load lists from JSON files
            with open(self.vocabListFile, 'r', encoding='utf-8') as f:
                self.vocabList = json.load(f)
            with open(self.tokenToIndexFile, 'r', encoding='utf-8') as f:
                self.tokenToIndex = json.load(f)
            with open(self.indexToTokenFile, 'r', encoding='utf-8') as f:
                self.indexToToken = json.load(f)

            print("Vocabulary files loaded successfully.")
            return True

        except FileNotFoundError:
            print("Vocabulary files not found. Building new vocabulary.")
            return False
        except json.JSONDecodeError:
            print("Error decoding vocabulary files. Rebuilding vocabulary.")
            return False

    def saveVocab(self):
        with open(self.vocabListFile, 'w', encoding='utf-8') as f:
            json.dump(self.vocabList, f, indent=4)
        with open(self.tokenToIndexFile, 'w', encoding='utf-8') as f:
            json.dump(self.tokenToIndex, f, indent=4)
        with open(self.indexToTokenFile, 'w', encoding='utf-8') as f:
            json.dump(self.indexToToken, f, indent=4)

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
