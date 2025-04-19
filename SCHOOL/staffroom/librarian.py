# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# VOCAB: TRAINING GENERATION AND TOKENIZATION
# BRAIN/LAYERS/vocab.py

from collections import Counter
from config import *
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, ByteLevelBPETokenizer
from tokenizers.processors import ByteLevel
import os, re, json, random, torch

"""
Handles vocab creation, loading, and tokenization.

This class:
- Trains a tokenizer (Byte-Pair Encoding) if no pre-trained tokenizer is found.
- Loads a pretrained tokenizer if its there.
- Builds vocab lists and mappings (token to index, index to token).
- Tokenizes text using the pretrained/loaded tokenizer.
- Loads training data.
- Generates training data pairs (input sequence, target token).
- Saves and loads vocab data to/from files.
"""
class LIBRARIAN:
    def __init__(self, _counsellor, _vocabSize = vocabSize, _vocabPath = vocabLoad):
        self.v_counsellor = _counsellor
        vocabSize = _vocabSize - 1 # reduces size by 1 to allow space for UNK token
        self.vocabList = []
        self.tokenToIndex = {}
        self.indexToToken = {}
        self.unkToken = "<UNK>"
        self.vocabCache = vocabCachePath
        self.vocabFilename = f"vocab{vocabSize}_{minTokenFreq}"
        self.vocabListFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_list.json")

        self.tokenToIndexFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_to_index.json")
        self.indexToTokenFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_to_token.json")

        self.tokenizerFilename = "tokenizer.json"
        self.tokenizerPath = os.path.join(self.vocabCache, self.tokenizerFilename)

        with self.v_counsellor.infodump("__init__") as ʕっʘ‿ʘʔっ:
            if _vocabPath:
                ʕっʘ‿ʘʔっ("if vocabPath") # if vocabPath is provided, load a pretrained tokenizer from that path
                self.tokenizerPath = _vocabPath
            else:
                ʕっʘ‿ʘʔっ("else tokenizerPath") # if vocabPath not provided (training mode), set tokenizerPath to the default directory
                self.tokenizerPath = os.path.join(self.vocabCache, self.tokenizerFilename)
                if not os.path.exists(self.vocabCache):
                    os.makedirs(self.vocabCache)

            ʕっʘ‿ʘʔっ("does tokenizerPath exist? ") # check if tokenizer files exist at tokenizerPath. if yes, load it!
            if os.path.exists(self.tokenizerPath):
                ʕっʘ‿ʘʔっ("tokenizerPath exists")
                print("loading trained tokenizer...")
                self.tokenizer = Tokenizer.from_file(self.tokenizerPath)
            else:
                if not self.tokenizerPath:
                    ʕっʘ‿ʘʔっ("tokenizerPath doesnt exist")
                    raise FileNotFoundError(f"tokenizer not found at {self.tokenizerPath}")
                
                ʕっʘ‿ʘʔっ("call tokenizer") # call the chosen tokenizer
                tokenizerTrainer = ByteLevelBPETokenizer(lowercase=True)
                tokenizerTrainer = Tokenizer(models.BPE(unk_token="<UNK>"))
                tokenizerTrainer.pre_tokenizer = pre_tokenizers.ByteLevel()
                trainer = trainers.BpeTrainer(
                    vocab_size=vocabSize,
                    min_frequency=minTokenFreq,
                    special_tokens=["<UNK>"]
                )
                ʕっʘ‿ʘʔっ("process training data")
                processed_data = []
                with open(trainingFilePath, "r", encoding="utf-8") as f:
                    processed_data.append(f.read().lower())  # Lowercase each file

                ʕっʘ‿ʘʔっ("train new tokenizer from data") # Train tokenizer on lowercased text (without modifying files)
                tokenizerTrainer.train_from_iterator(processed_data, trainer)

                ʕっʘ‿ʘʔっ("save tokenizer") # save this new tokenizer to the save path, where it will be loaded from in future runs to keep vocab consistent
                tokenizerTrainer.save(self.tokenizerPath)
                self.tokenizer = tokenizerTrainer

            if debugPrints: print(f"vocab length: {len(self.tokenizer.get_vocab())}")

            if self.loadVocab():
                ʕっʘ‿ʘʔっ("loadVocab")
                if debugPrints: print(f"loaded vocab: {self.vocabCache}")
                self.trainingDataPairs = self.loadTrainingData(trainingFilePath_arr)  # Load text data
                self.tokens = self.tokenizeText(self.trainingDataPairs)  # Tokenize the text
                #print(f"DEBUG: tokens exist? {hasattr(self, 'tokens')} (length: {len(self.tokens) if hasattr(self, 'tokens') else 'N/A'})")
                #print(f"DEBUG: tokenToIndex keys (first 20): {list(self.tokenToIndex.keys())[:20]}")
            else:
                ʕっʘ‿ʘʔっ("buildVocab")
                if debugPrints: print(f"building vocab from scratch (size: {vocabSize})...")
                self.buildVocabMap() # Use new method to build vocab mappings
                if debugPrints: print(f"vocab length: {len(self.vocabList)}") # Vocab length from list
                ʕっʘ‿ʘʔっ("saveVocab")
                self.saveVocab()
                print(f"saved vocab data to: {self.vocabCache}")

            #print(f"DEBUG VOCAB.__init__: length of vocabList AFTER buildVocab: {len(self.vocabList)}")
            #print(f"DEBUG VOCAB.__init__: first 20 tokens in vocabList: {self.vocabList[:20]}")

    def tokenizeText(self, _text):
        with self.v_counsellor.infodump("tokenizeText") as ʕっʘ‿ʘʔっ:
            encoding = self.tokenizer.encode(_text)  # Tokenize using encode method
            tokens_str = [self.indexToToken.get(idx, "<UNK>") for idx in encoding.ids]  # Convert IDs back to strings
            #print(f"tokenizing: {text}")
            #print(f"token ids: {encoding.ids}")
            #print(f"token strings: {tokens_str}")
            return tokens_str

    def buildVocabMap(self):
        with self.v_counsellor.infodump("buildVocabMap") as ʕっʘ‿ʘʔっ:
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

            print(f"final vocab size: {len(self.vocabList)} (should be {vocabSize})")
            print(f"first 20 tokens: {self.vocabList[:20]}")


    """HUGGING FACE TOKENIZER"""
    def huggingTokenizer(self, _text):
        with self.v_counsellor.infodump("huggingTokenizer") as ʕっʘ‿ʘʔっ:
            """uses the trained/loaded tokenizer to convert input text into tokens"""
            return self.tokenizer.tokenize(_text)

    """LOAD TRAINING DATA"""
    def loadTrainingData(self, _filepaths, _chunkSize=V_chunkSizeLoadData):
        with self.v_counsellor.infodump("loadTrainingData") as ʕっʘ‿ʘʔっ: #Reads text files in chunks, concatenates the chunks, and removes extra whitespace
            loadTrainingData = ""
            for filepath in _filepaths:
                with open(filepath, "r", encoding="utf-8") as f:
                    while True:
                        chunk = f.read(_chunkSize)
                        if not chunk:
                            break
                        ʕっʘ‿ʘʔっ("loaded a data chunk!")
                        loadTrainingData += chunk + ""
            print(f"loaded {len(loadTrainingData)} characters of training data!")
            loadTrainingData = re.sub(r'\s+', ' ', loadTrainingData)
            """returns a single string containing all the loaded and preprocessed training data."""
            return loadTrainingData
    
    """GENERATE TRAINING DATA"""
    def genTrainingData(self, _windowMAX = windowMAX, _startIndex = trainingStartIndex):
        with self.v_counsellor.infodump("genTrainingData") as ʕっʘ‿ʘʔっ:
            """generates training data pairs (input sequences and target tokens)"""
            trainingDataPairs = []
            if isinstance(_windowMAX, torch.Tensor):
                _windowMAX = _windowMAX
            else:
                _windowMAX = int(_windowMAX)
            """allows for a random start in the training data file"""
            if _startIndex == 'random':
                _startIndex = random.randint(0, len(self.tokens) - _windowMAX - 1)
            else:
                _startIndex = int(_startIndex)
            endIndex = len(self.tokens) - _windowMAX
            numTargetTokens = numTokensPerStep
            if isinstance(numTargetTokens, torch.Tensor):
                numTargetTokens = numTargetTokens
            else:
                numTargetTokens = int(numTargetTokens)
            ʕっʘ‿ʘʔっ("start generating training data")
            for i in range(_startIndex, endIndex):
                ʕっʘ‿ʘʔっ("generate training sequence") # creates sliding windows from the tokenized data to form inputSeq, uses next token as target
                inputSeq = self.tokens[i:i + _windowMAX] # a list of tokens (str) of length `windowMAX`
                targetTokens = self.tokens[i + _windowMAX : i + _windowMAX + numTargetTokens] # a single token (str) that follows the input_sequence.
                if len(targetTokens) < numTargetTokens:
                    continue
                if all(token in self.vocabList for token in inputSeq) and all(t_token in self.vocabList for t_token in targetTokens):
                    trainingDataPairs.append((inputSeq, targetTokens))
                else:
                    print(f"skipping UNK - inputSeq: '{inputSeq}', targetSeq: '{targetTokens}'")
        """returns a list of tuples: (inputSeq, targetToken)"""
        return trainingDataPairs

    """saves vocab data to JSON files in vocabCache directory, meaning it can be reloaded without tokenization"""
    def saveVocab(self):
        with self.v_counsellor.infodump("saveVocab") as ʕっʘ‿ʘʔっ:
            os.makedirs(self.vocabCache, exist_ok=True)  # Ensure directory exists
            with open(self.vocabListFile, "w", encoding="utf-8") as f:
                ʕっʘ‿ʘʔっ("save vocabList")
                json.dump(self.vocabList, f, indent=4)
            with open(self.tokenToIndexFile, "w", encoding="utf-8") as f:
                ʕっʘ‿ʘʔっ("save tokenToIndex")
                json.dump(self.tokenToIndex, f, indent=4)
            with open(self.indexToTokenFile, "w", encoding="utf-8") as f:
                ʕっʘ‿ʘʔっ("save indexToToken")
                json.dump(self.indexToToken, f, indent=4)

    """loads vocab data from JSON files in vocabCache directory"""
    def loadVocab(self):
        with self.v_counsellor.infodump("loadVocab") as ʕっʘ‿ʘʔっ:
            try:
                # Load vocab lists from JSON files
                with open(self.vocabListFile, 'r', encoding='utf-8') as f:
                    ʕっʘ‿ʘʔっ("load vocabList")
                    self.vocabList = json.load(f)
                with open(self.tokenToIndexFile, 'r', encoding='utf-8') as f:
                    ʕっʘ‿ʘʔっ("load tokenToIndex")
                    self.tokenToIndex = json.load(f)
                with open(self.indexToTokenFile, 'r', encoding='utf-8') as f:
                    ʕっʘ‿ʘʔっ("load indexToToken")
                    self.indexToToken = {int(k): v for k, v in json.load(f).items()}  # Ensure keys are integers!

                print("vocab files loaded successfully!")
                if debugPrints: print(f"first 20 tokens: {self.vocabList[:20]}")
                return bool(self.vocabList and self.tokenToIndex and self.indexToToken)

            except FileNotFoundError:
                print("vocab files not found... \nrebuilding vocab...")
                return False
            except json.JSONDecodeError:
                print("error decoding vocab files... \nrebuilding vocab...")
                return False

# EXAMPLE RUNNING
if __name__ == "__main__":

    librarian = LIBRARIAN(vocabSize = vocabSize)

    print(f"--- 1701-2000 ---: {self.vocabList[1701:2000]}")
    print(f"--- 1001-1700 ---: {self.vocabList[301:1700]}")
    print(f"--- 301-1000 ---: {self.vocabList[301:1000]}")
    print(f"--- 101-300 ---: {self.vocabList[101:300]}")
    print(f"--- Top 100 ---: {self.vocabList[:100]}")
    print(f"vocab size: {len(self.vocabList)}")

    #print(vocab.huggingTokenizer("charis and elodie are very cool, elodies pretty and charis is very suave, they're sexy bitches, we love these girls and we want to see them living their best lives bruv"))
    sample_text = "charis and elodie are very cool, elodies pretty and charis is very suave, they're sexy bitches, we love these girls and we want to see them living their best lives bruv"
    tokenizedOutput = self.tokenizeText(sample_text)
    print(f"Tokenized: {tokenizedOutput}")