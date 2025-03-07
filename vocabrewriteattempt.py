# Charis Cat 2025
from collections import Counter
from config import *
from transformers import AutoTokenizer
import os
import re
import json

class VOCAB:
    def __init__(self, vocabSize):
        self.vocabSize = vocabSize - 1
        self.vocabList = []
        self.tokenToIndex = {}
        self.indexToToken = {}
        self.unkToken = "<UNK>"

        self.vocabCache = "vocabCache"  # Directory to store vocab files
        self.vocabFilename = f"vocab_{vocabSize}"
        self.vocabListFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_list.json")
        self.tokenToIndexFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_to_index.json")
        self.indexToTokenFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_to_token.json")

        if not os.path.exists(self.vocabCache):
            os.makedirs(self.vocabCache)

        #self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.backend_tokenizer.post_processor = None  # Nukes BPE merge rules
        self.tokenizer.model_max_length = 9000000

        # 🚨 Define special tokens
        custom_tokens = ["charis", "elodie", "froggy", "kevin", "pete", "george", "geepy"]

        # 🚀 Step 1: Register tokens
        self.tokenizer.add_special_tokens({"additional_special_tokens": custom_tokens})

        # 🚀 Step 2: Force single-word tokens
        self.tokenizer.add_tokens(custom_tokens, special_tokens=True)

        # 🚀 Step 3: Save and reload tokenizer
        self.tokenizer.save_pretrained("babyLLM")
        self.tokenizer = AutoTokenizer.from_pretrained("babyLLM")

        # 🔍 Check if it worked
        print("🚨 Custom Tokens Now Exist:", self.tokenizer.convert_tokens_to_ids(custom_tokens))
        print("🔎 Tokenized Output:", self.tokenizer.tokenize("charis and elodie are cool"))

        if self.loadVocab():
            print(f"Loaded vocab: {self.vocabCache}")
            self.trainingData = None
            print(f"Debug: tokenToIndex keys (first 20): {list(self.tokenToIndex.keys())[:20]}")
        else:
            print(f"Building vocab from data (size: {vocabSize})...")
            self.trainingData = self.loadTrainingDataFUNK(dataFilepaths)
            self.tokens = self.huggingTokenizer(self.trainingData) # using huggingface now
            print("TOTAL TOKENS PROCESSED:", len(self.tokens))
            print("🔍 TOKENIZATION PREVIEW:", self.tokens[:100])
            print("⚠️ STANDALONE Ġ COUNT:", self.tokens.count("Ġ"))
            self.vocabList = self.buildVocab()
            self.tokenToIndex = self.createTokenToIndex()
            self.indexToToken = self.createIndexToToken()
            self.vocabList.append(self.unkToken)
            self.tokenToIndex[self.unkToken] = len(self.vocabList) - 1
            self.indexToToken[len(self.vocabList) - 1] = self.unkToken
            self.saveVocab()
            print(f"Saved vocabulary to cache directory: {self.vocabCache}")

        print(f"Debug VOCAB.__init__: Length of vocabList AFTER buildVocab: {len(self.vocabList)}")
        print(f"Debug VOCAB.__init__: First 20 tokens in vocabList: {self.vocabList[:20]}")

    # HUGGING FACE TOKENIZER
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
        
    # WHITESPACE TOKENISER
    #def whitespaceTokenizer(self, text):
    #    textLower = text.lower() # lowercase conversion
    #    tokens = textLower.split()
    #    return tokens

    #NLTK TOKENIZER
    #def nltkTokenizer(self, text):
    #    textLower = text.lower()
    #    tokens = word_tokenize(textLower)
    #    processedTokens = [] # New list to hold processed tokens
    #    for token in tokens:
    #        if token in self.vocabList: # Check if token is in vocab
    #            processedTokens.append(token) # Use token as is if in vocab
    #        else:
    #            processedTokens.append(self.unkToken) # Replace with <UNK> if OOV
    #    return processedTokens # Return processed tokens list

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
