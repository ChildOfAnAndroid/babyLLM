# Charis Cat 2025
from collections import Counter
from config import *
from transformers import AutoTokenizer, PreTrainedTokenizerFast  # Import PreTrainedTokenizerFast
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import ByteLevel
import os
import re
import json
import random

class VOCAB:
    def __init__(self, vocabSize, vocabPath=None):
        self.vocabSize = vocabSize - 1
        self.vocabList = []
        self.tokenToIndex = {}
        self.indexToToken = {}
        self.unkToken = "<UNK>"

        self.vocabCache = "vocabCache"
        self.vocabFilename = f"vocab_{vocabSize}"
        self.vocabListFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_list.json")
        self.tokenToIndexFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_to_index.json")
        self.indexToTokenFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_to_token.json")

        if vocabPath:
            # Directly load tokenizer from given path (for inference mode)
            tokenizerSavePath = vocabPath
        else:
            # Default path for training mode
            tokenizerSavePath = os.path.join(self.vocabCache, self.vocabFilename)

            if not os.path.exists(self.vocabCache):
                os.makedirs(self.vocabCache)

        if os.path.exists(tokenizerSavePath):
            print("Tokenizer found, loading from disk...")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizerSavePath)
        else:
            if vocabPath:
                raise FileNotFoundError(f"Tokenizer not found at {tokenizerSavePath}. Cannot train tokenizer without data!")
            
            print("Tokenizer not found, training now...")
            tokenizerTrainer = ByteLevelBPETokenizer(lowercase=True)
            tokenizerTrainer.train(
                files=dataFilepaths,
                vocab_size=self.vocabSize,
                min_frequency=20,
            )
            tokenizerTrainer.post_processor = ByteLevel(trim_offsets=True)

            wrappedTokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizerTrainer,
                unk_token = self.unkToken,
                model_max_length=9000000,
            )

            wrappedTokenizer.save_pretrained(tokenizerSavePath)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizerSavePath)

        print(f"Vocab length: {len(self.tokenizer)}")

        if self.loadVocab():
            print(f"Loaded vocab: {self.vocabCache}")
            self.trainingData = self.loadTrainingDataFUNK(dataFilepaths)  # Load text data
            self.tokens = self.huggingTokenizer(self.trainingData)  # Tokenize the text
            print(f"DEBUG: Tokens exist? {hasattr(self, 'tokens')} (Length: {len(self.tokens) if hasattr(self, 'tokens') else 'N/A'})")
            
            print(f"Debug: tokenToIndex keys (first 20): {list(self.tokenToIndex.keys())[:20]}")
        else:
            print(f"Building vocab from scratch (size: {vocabSize})...")
            self.vocabList = list(self.tokenizer.get_vocab().keys())
            self.tokenToIndex = self.tokenizer.get_vocab()
            self.indexToToken = {v: k for k, v in self.tokenToIndex.items()}
            if self.unkToken not in self.tokenToIndex:
                self.vocabList.append(self.unkToken)
                self.tokenToIndex[self.unkToken] = len(self.vocabList) - 1
                self.indexToToken[len(self.vocabList) - 1] = self.unkToken
            
            print(f"Vocab length: {len(self.tokenizer)}")
            self.saveVocab()
            print(f"Saved vocab: {self.vocabCache}")

        print(f"Debug VOCAB.__init__: Length of vocabList AFTER buildVocab: {len(self.vocabList)}")
        print(f"Debug VOCAB.__init__: First 20 tokens in vocabList: {self.vocabList[:20]}")

    # HUGGING FACE TOKENIZER (now uses the newly trained tokenizer)
    def huggingTokenizer(self, text):
        return self.tokenizer.tokenize(text)

    # LOAD TRAINING DATA
    def loadTrainingDataFUNK(self, filepaths, chunk_size=4096):
        loadTrainingData = ""
        for filepath in filepaths:
            with open(filepath, "r", encoding="utf-8") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    loadTrainingData += chunk + " "
        print(f"Loaded {len(loadTrainingData)} characters of training data.")
        loadTrainingData = re.sub(r'\s+', ' ', loadTrainingData)
        return loadTrainingData
    
    # GENERATE TRAINING DATA
    def genTrainingData(self, trainingWindow):
        trainingData = []
        if len(self.tokens) < trainingWindow:
            print("Warning: Not enough tokens to generate training data!")
            return trainingData
            # 🎲 Pick a random starting index
        startIndex = random.randint(0, len(self.tokens) - trainingWindow - 1)
        for i in range(startIndex, len(self.tokens) - trainingWindow):
            inputSeq = self.tokens[i - trainingWindow:i]
            target = self.tokens[i]
            # Check tokens in the sequence are in our vocabulary
            if all(token in self.vocabList for token in inputSeq) and target in self.vocabList:
                trainingData.append((inputSeq, target))
            else:
                print(f"Skipping UNK - Input: {inputSeq}, Target: {target}")
        return trainingData


    # TOP 1999 TOKENS + UNK
    def buildVocab(self):
        tokenCounts = Counter(self.tokens)
        #mostCommonTokens = [token for token, _ in tokenCounts.most_common(self.vocabSize)]
        mostCommonTokens = [token for token, _ in tokenCounts.most_common(self.vocabSize) if token.strip()]
        print("TOP 20 RAW TOKEN COUNTS:", tokenCounts.most_common(20))  # See what it's actually counting
        return mostCommonTokens

    # DICTIONARYS
    def createTokenToIndex(self):
        return {token: index for index, token in enumerate(self.vocabList)}

    def createIndexToToken(self):
        #indexToToken = {}
        #for index, token in enumerate(self.vocabList):
        #    indexToToken[index] = token
        return {index: token for token, index in self.tokenToIndex.items()}

    def saveVocab(self):
        os.makedirs(self.vocabCache, exist_ok=True)  # Ensure directory exists
        with open(self.vocabListFile, "w", encoding="utf-8") as f:
            json.dump(self.vocabList, f, indent=4)
        with open(self.tokenToIndexFile, "w", encoding="utf-8") as f:
            json.dump(self.tokenToIndex, f, indent=4)
        with open(self.indexToTokenFile, "w", encoding="utf-8") as f:
            json.dump(self.indexToToken, f, indent=4)
    
    def loadVocab(self):
        try:
            # Load vocab lists from JSON files
            with open(self.vocabListFile, 'r', encoding='utf-8') as f:
                self.vocabList = json.load(f)
            with open(self.tokenToIndexFile, 'r', encoding='utf-8') as f:
                self.tokenToIndex = json.load(f)
            with open(self.indexToTokenFile, 'r', encoding='utf-8') as f:
                self.indexToToken = json.load(f)

            print("Vocabulary files loaded successfully.")
            return bool(self.vocabList and self.tokenToIndex and self.indexToToken)

        except FileNotFoundError:
            print("Vocabulary files not found. Building new vocabulary.")
            return False
        except json.JSONDecodeError:
            print("Error decoding vocabulary files. Rebuilding vocabulary.")
            return False

# EXAMPLE RUNNING
if __name__ == "__main__":

    vocab = VOCAB(vocabSize = vocabSize)

    #loading data
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

    print(vocab.huggingTokenizer("charis and elodie are very cool, elodie and charis are very suave, sexy bitches, we love these girls and we want to see them living their best lives bruv"))

    #create token/index dictionarys
    #tokenToIndex = createTokenToIndex(vocabList)
    #indexToToken = createIndexToToken(vocabList)
    #print(f"Token to index mapping (first 10): {dict(list(tokenToIndex.items())[:10])}")
    #print(f"Index to token mapping (first 10): {dict(list(indexToToken.items())[:10])}")
