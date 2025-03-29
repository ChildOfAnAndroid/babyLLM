# Charis Cat 2025
from collections import Counter
from config import *
#from transformers import AutoTokenizer, PreTrainedTokenizerFast  # Import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
#from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import ByteLevel
import os
import re
import json
import random
import torch

"""
Handles vocabulary creation, loading, and tokenization.

This class:
- Trains a tokenizer (Byte-Pair Encoding) if no pre-trained tokenizer is found.
- Loads a pretrained tokenizer if its there.
- Builds vocabulary lists and mappings (token to index, index to token).
- Tokenizes text using the pretrained/loaded tokenizer.
- Loads training data.
- Generates training data pairs (input sequence, target token).
- Saves and loads vocabulary data to/from files.
"""
class VOCAB:
    def __init__(self, vocabSize = vocabSize, vocabPath = vocabLoad):
        self.vocabSize = vocabSize - 1 # reduces size by 1 to allow space for UNK token
        self.vocabList = []
        self.tokenToIndex = {}
        self.indexToToken = {}
        self.unkToken = "<UNK>"
        self.vocabCache = "vocabCache"
        self.vocabFilename = f"vocab{vocabSize}_{minTokenFreq}"
        self.vocabListFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_list.json")

        self.tokenToIndexFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_to_index.json")
        self.indexToTokenFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_to_token.json")

        self.tokenizerFilename = "tokenizer.json"
        self.tokenizerPath = os.path.join(self.vocabCache, self.tokenizerFilename)

        if vocabPath:
            """if vocabPath is provided, load a pretrained tokenizer from that path"""
            self.tokenizerPath = vocabPath
        else:
            """if vocabPath not provided (training mode), set tokenizerPath to the default directory"""
            self.tokenizerPath = os.path.join(self.vocabCache, self.tokenizerFilename)
            if not os.path.exists(self.vocabCache):
                os.makedirs(self.vocabCache)

        """check if tokenizer files exist at tokenizerPath. if yes, load it!"""
        if os.path.exists(self.tokenizerPath):
            print("Loading trained tokenizer...")
            self.tokenizer = Tokenizer.from_file(self.tokenizerPath)
        else:
            if not self.tokenizerPath:
                raise FileNotFoundError(f"Tokenizer not found at {self.tokenizerPath}. Cannot train tokenizer without data!")
            
            """call the chosen tokenizer"""
            print("Tokenizer not found, training now...")
            #tokenizerTrainer = ByteLevelBPETokenizer(lowercase=True)
            tokenizerTrainer = Tokenizer(models.BPE(unk_token="<UNK>"))
            tokenizerTrainer.pre_tokenizer = pre_tokenizers.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=self.vocabSize,
                min_frequency=minTokenFreq,
                special_tokens=["<UNK>"]
            )
            print("Processing training data...")
            processed_data = []
            with open(trainingFile, "r", encoding="utf-8") as f:
                processed_data.append(f.read().lower())  # Lowercase each file

            # Train tokenizer on lowercased text (without modifying files)
            print("Training new tokenizer...")
            tokenizerTrainer.train_from_iterator(processed_data, trainer)

            """save this new tokenizer to the save path, where it will be loaded from in future runs to keep vocab consistent"""
            tokenizerTrainer.save(self.tokenizerPath)
            self.tokenizer = tokenizerTrainer

        print(f"Vocab length: {len(self.tokenizer.get_vocab())}")

        if self.loadVocab():
            print(f"Loaded vocab: {self.vocabCache}")
            self.trainingDataPairs = self.loadTrainingData(dataFilepaths)  # Load text data
            self.tokens = self.tokenizeText(self.trainingDataPairs)  # Tokenize the text
            print(f"DEBUG: Tokens exist? {hasattr(self, 'tokens')} (Length: {len(self.tokens) if hasattr(self, 'tokens') else 'N/A'})")
            #print(f"DEBUG: tokenToIndex keys (first 20): {list(self.tokenToIndex.keys())[:20]}")
        else:
            print(f"Building vocab from scratch (size: {vocabSize})...")
            self.buildVocabMap() # Use new method to build vocab mappings
            print(f"Vocab length: {len(self.vocabList)}") # Vocab length from list
            self.saveVocab()
            print(f"Saved vocab data to: {self.vocabCache}")

        #print(f"DEBUG VOCAB.__init__: Length of vocabList AFTER buildVocab: {len(self.vocabList)}")
        #print(f"DEBUG VOCAB.__init__: First 20 tokens in vocabList: {self.vocabList[:20]}")

    def tokenizeText(self, text):
        encoding = self.tokenizer.encode(text)  # Tokenize using encode method
        tokens_str = [self.indexToToken.get(idx, "<UNK>") for idx in encoding.ids]  # Convert IDs back to strings
        #print(f"📝 Tokenizing: {text}")
        #print(f"📌 Token IDs: {encoding.ids}")
        #print(f"📌 Token Strings: {tokens_str}")
        return tokens_str

    def buildVocabMap(self):
        # Load vocab from trained tokenizer
        vocab = self.tokenizer.get_vocab()  # Get vocab dictionary from tokenizer
        self.vocabList = sorted(vocab.keys(), key=lambda x: vocab[x])  # Sort by ID order
        self.tokenToIndex = vocab  # tokenToIndex is the vocab dict itself
        self.indexToToken = {v: k for k, v in vocab.items()}  # FIXED: Direct mapping

        # Ensure UNK token is in vocab
        if self.unkToken not in self.tokenToIndex:
            self.vocabList.append(self.unkToken)
            self.tokenToIndex[self.unkToken] = len(self.vocabList) - 1
            self.indexToToken[len(self.vocabList) - 1] = self.unkToken

        print(f"✅ Final Vocab Size: {len(self.vocabList)} (should be {self.vocabSize})")
        print(f"📌 First 20 Tokens: {self.vocabList[:20]}")


    """HUGGING FACE TOKENIZER"""
    def huggingTokenizer(self, text):
        """uses the trained/loaded tokenizer to convert input text into tokens"""
        return self.tokenizer.tokenize(text)

    """LOAD TRAINING DATA"""
    def loadTrainingData(self, filepaths, chunk_size=loadData_chunkSize):
        """Reads text files in chunks, concatenates the chunks, and removes extra whitespace"""
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
        """returns a single string containing all the loaded and preprocessed training data."""
        return loadTrainingData
    
    """GENERATE TRAINING DATA"""
    def genTrainingData(self, windowMAX, startIndex = trainingStartIndex):
        """generates training data pairs (input sequences and target tokens)"""
        trainingDataPairs = []
        if isinstance(windowMAX, torch.Tensor):
            windowMAX = windowMAX.item()
        else:
            windowMAX = int(windowMAX)
        """allows for a random start in the training data file"""
        if startIndex == 'random':
            startIndex = random.randint(0, len(self.tokens) - windowMAX - 1)
        else:
            startIndex = int(startIndex)
        endIndex = len(self.tokens) - windowMAX
        numTargetTokens = numTokensPerStep
        if isinstance(numTargetTokens, torch.Tensor):
            numTargetTokens = numTargetTokens.item()
        else:
            numTargetTokens = int(numTargetTokens)
        """creates sliding windows from the tokenized training data (`self.tokens`) to form input sequences, using the next token as target."""
        for i in range(startIndex, endIndex):
            inputSeq = self.tokens[i:i + windowMAX] # a list of tokens (str) of length `windowMAX`
            targetTokens = self.tokens[i + windowMAX : i + windowMAX + numTargetTokens] # a single token (str) that follows the input_sequence.
            if len(targetTokens) < numTargetTokens:
                continue
            if all(token in self.vocabList for token in inputSeq) and all(t_token in self.vocabList for t_token in targetTokens):
                trainingDataPairs.append((inputSeq, targetTokens))
            else:
                print(f"Skipping UNK - Input: {inputSeq}, Target: {targetTokens}")
        """returns a list of tuples: (inputSeq, targetToken)"""
        return trainingDataPairs

    """saves vocab data to JSON files in vocabCache directory, meaning it can be reloaded without tokenization"""
    def saveVocab(self):
        os.makedirs(self.vocabCache, exist_ok=True)  # Ensure directory exists
        with open(self.vocabListFile, "w", encoding="utf-8") as f:
            json.dump(self.vocabList, f, indent=4)
        with open(self.tokenToIndexFile, "w", encoding="utf-8") as f:
            json.dump(self.tokenToIndex, f, indent=4)
        with open(self.indexToTokenFile, "w", encoding="utf-8") as f:
            json.dump(self.indexToToken, f, indent=4)

    
    """loads vocab data from JSON files in vocabCache directory"""
    def loadVocab(self):
        try:
            # Load vocab lists from JSON files
            with open(self.vocabListFile, 'r', encoding='utf-8') as f:
                self.vocabList = json.load(f)
            with open(self.tokenToIndexFile, 'r', encoding='utf-8') as f:
                self.tokenToIndex = json.load(f)
            with open(self.indexToTokenFile, 'r', encoding='utf-8') as f:
                self.indexToToken = {int(k): v for k, v in json.load(f).items()}  # Ensure keys are integers!

            print("✅ Vocabulary files loaded successfully.")
            print(f"📌 First 20 Tokens: {self.vocabList[:20]}")
            return bool(self.vocabList and self.tokenToIndex and self.indexToToken)

        except FileNotFoundError:
            print("❌ Vocabulary files not found. Rebuilding vocabulary.")
            return False
        except json.JSONDecodeError:
            print("❌ Error decoding vocabulary files. Rebuilding vocabulary.")
            return False

# EXAMPLE RUNNING
if __name__ == "__main__":

    vocab = VOCAB(vocabSize = vocabSize)

    print(f"Vocabulary size: {len(vocab.vocabList)}")
    print(f"---1701-2000---: {vocab.vocabList[1701:2000]}")
    print(f"---1001-1700---: {vocab.vocabList[301:1700]}")
    print(f"---301-1000---: {vocab.vocabList[301:1000]}")
    print(f"---101-300---: {vocab.vocabList[101:300]}")
    print(f"---Top 100---: {vocab.vocabList[:100]}")

    #print(vocab.huggingTokenizer("charis and elodie are very cool, elodies pretty and charis is very suave, they're sexy bitches, we love these girls and we want to see them living their best lives bruv"))
    sample_text = "charis and elodie are very cool, elodies pretty and charis is very suave, they're sexy bitches, we love these girls and we want to see them living their best lives bruv"
    tokenizedOutput = vocab.tokenizeText(sample_text)
    print(f"Tokenized: {tokenizedOutput}")