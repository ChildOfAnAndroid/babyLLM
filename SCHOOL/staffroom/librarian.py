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
from SCHOOL.notebook.tools.genBoi import *

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
    def __init__(self, _counsellor, _vocabSize = vocabSize, _vocabPath = vocabLoad, _baseTokenizerPath = None):
        self.v_counsellor = _counsellor
        self.vocabSize = _vocabSize
        self.vocabSize += 1 # increases size by 1 to allow space for UNK token
        self.vocabList = []
        self.tokenToIndex = {}
        self.indexToToken = {}
        self.unkToken = "<UNK>"
        self.vocabCache = vocabCachePath
        self.vocabFilename = f"vocab{_vocabSize}_{minTokenFreq}"
        self.vocabListFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_list.json")

        self.tokenToIndexFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_to_index.json")
        self.indexToTokenFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_to_token.json")

        self.tokenizerFilename = f"tokenizer_{_vocabSize}.json"
        self.tokenizerPath = os.path.join(self.vocabCache, self.tokenizerFilename)
        self.baseTokenizerPath = _baseTokenizerPath #optional

        with self.v_counsellor.infodump("__init__") as ʕっʘ‿ʘʔっ:
            if _vocabPath:
                ʕっʘ‿ʘʔっ("using provided vocabPath...") # if vocabPath is provided, load a pretrained tokenizer from that path
                self.tokenizerPath = _vocabPath
            else:
                ʕっʘ‿ʘʔっ("creating provided vocabPath...") # if vocabPath not provided (training mode), set tokenizerPath to the default directory
                self.tokenizerPath = os.path.join(self.vocabCache, self.tokenizerFilename)
                if not os.path.exists(self.vocabCache):
                    os.makedirs(self.vocabCache)

            if os.path.exists(self.tokenizerPath):
                ʕっʘ‿ʘʔっ("loading existing tokenizer...")
                print("loading existing tokenizer...")
                self.tokenizer = Tokenizer.from_file(self.tokenizerPath)
            else:
                ʕっʘ‿ʘʔっ("training new tokenizer...")
                if _baseTokenizerPath:
                    tokenizerModel = Tokenizer.from_file(self.baseTokenizerPath) # load existing tokenizer
                    trainer = trainers.BpeTrainer(vocab_size = vocabSize, min_frequency = minTokenFreq, special_tokens = ["<UNK>"])
                else:
                    tokenizerModel = Tokenizer(models.BPE(unk_token="<UNK>"))
                    tokenizerModel.pre_tokenizer = pre_tokenizers.ByteLevel()
                    trainer = trainers.BpeTrainer(vocab_size = vocabSize, min_frequency = minTokenFreq, special_tokens = ["<UNK>"])
            with open(trainingFilePath, "r", encoding="utf-8") as f:
                training_data = [f.read().lower()]
            tokenizerModel.train_from_iterator(training_data, trainer)
            tokenizerModel.save(self.tokenizerPath)
            self.tokenizer = tokenizerModel

            if self.baseTokenizerPath: # compare with previous tokenizer if provided
                try:
                    baseTokenizer = Tokenizer.from_file(self.baseTokenizerPath)
                    oldVocab = set(baseTokenizer.get_vocab().keys())
                    newVocab = set(tokenizerModel.get_vocab().keys())
                    addedTokens = sorted(list(newVocab - oldVocab))
                    print(f"\nvocab expansion!")
                    print(f"previous vocab: {len(oldVocab)}")
                    print(f"new vocab: {len(newVocab)}")
                    print(f"{len(addedTokens)} new tokens added.")
                    print(f"example: {addedTokens[:20]}")
                except Exception as e:
                    print(f"couldnt compare with old tokenizer: {e}")
            if self.baseTokenizerPath:
                baseTokenizer = Tokenizer.from_file(self.baseTokenizerPath)
                baseVocab = baseTokenizer.get_vocab()
                newVocab = self.tokenizer.get_vocab()

                changed = []
                for token in baseVocab:
                    oldIndex = baseVocab[token]
                    newIndex = newVocab.get(token, -1)
                    if oldIndex != newIndex:
                        changed.append((token, oldIndex, newIndex))

                if changed:
                    print(f"noooo! {len(changed)} tokens have changed positions!")
                    for tok, old, new in changed[:2000]:
                        print(f"token: {tok} - was {old}, now {new}")
            else:
                print("both tokenizers match! :D")


            self.trainingDataPairs = self.loadTrainingData(trainingFilePath_arr)
            self.tokens = self.tokenizeText(self.trainingDataPairs)

            if debugPrints: print(f"vocab length: {len(self.tokenizer.get_vocab())}")

            if self.loadVocab():
                ʕっʘ‿ʘʔっ("loaded existing vocab")
                if debugPrints:
                    print(f"loaded vocab from {self.vocabCache}")
                self.trainingDataPairs = self.loadTrainingData(trainingFilePath_arr)
                self.tokens = self.tokenizeText(self.trainingDataPairs)
            else:
                ʕっʘ‿ʘʔっ("building vocab from scratch")
                self.buildVocabMap()
                if debugPrints:
                    print(f"vocab length is {len(self.vocabList)}")
                self.saveVocab()
                print(f"saved vocab data to {self.vocabCache}")

    def tokenizeText(self, _text):
        with self.v_counsellor.infodump("tokenizeText") as ʕっʘ‿ʘʔっ:
            encoding = self.tokenizer.encode(_text)
            if debugPrints:
                print(f"tokenizing: {_text}")
                print(f"token ids: {encoding.ids}")
            return [self.indexToToken.get(idx, self.unkToken) for idx in encoding.ids] # Convert indexs back to strings

    def buildVocabMap(self):
        with self.v_counsellor.infodump("buildVocabMap") as ʕっʘ‿ʘʔっ:
            ʕっʘ‿ʘʔっ("getting vocab dictionary from tokenizer...")
            invVocab = self.tokenizer.get_vocab()
            ʕっʘ‿ʘʔっ("ordering by index...")
            sortedTokens = sorted(invVocab.items(), key=lambda item: item[1])  # sort by index
            self.vocabList = [token for token, idx in sortedTokens]
            ʕっʘ‿ʘʔっ("mapping vocab dicts...")
            self.tokenToIndex = {token: idx for token, idx in sortedTokens}
            self.indexToToken = {idx: token for token, idx in sortedTokens}
            ʕっʘ‿ʘʔっ("ensuring <UNK> is in the vocab...")
            if self.unkToken not in self.tokenToIndex:
                self.vocabList.append(self.unkToken)
                unk_index = len(self.vocabList) - 1
                self.tokenToIndex[self.unkToken] = unk_index
                self.indexToToken[unk_index] = self.unkToken
            print(f"final vocab size: {len(self.vocabList)}")
            print(f"first 20 tokens: {self.vocabList[:20]}")

    def huggingTokenizer(self, _text): return self.tokenizer.tokenize(_text)

    def loadTrainingData(self, _filepaths, _chunkSize=V_chunkSizeLoadData):
        with self.v_counsellor.infodump("loadTrainingData") as ʕっʘ‿ʘʔっ:
            result = ""
            for path in _filepaths:
                with open(path, "r", encoding="utf-8") as f:
                    while True:
                        chunk = f.read(_chunkSize)
                        if not chunk: break
                        result += chunk
            result = re.sub(r'\s+', ' ', result)
            print(f"loaded {len(result)} characters of training data!")
            return result

    def genTrainingData(self, _windowMAX=windowMAX, _startIndex=trainingStartIndex, _trainingDataPairNumber=trainingDataPairNumber):
        with self.v_counsellor.infodump("genTrainingData") as ʕっʘ‿ʘʔっ:
            trainingDataPairs = []
            count = 0
            tokens = self.tokens
            ʕっʘ‿ʘʔっ("check if windowMax is tensor?")
            if isinstance(_windowMAX, torch.Tensor): _windowMAX = _windowMAX.item()
            
            ʕっʘ‿ʘʔっ("allows for random start")
            if _startIndex == 'random':
                _startIndex = random.randint(0, len(tokens) - _windowMAX - 1)

            end = len(tokens) - _windowMAX

            ʕっʘ‿ʘʔっ("generate training pairs")
            for i in range(_startIndex, end):
                inputSeq = tokens[i:i+_windowMAX]
                target = tokens[i+_windowMAX:i+_windowMAX+_windowMAX]
                if len(target) < _windowMAX: continue
                if all(t in self.vocabList for t in inputSeq + target):
                    trainingDataPairs.append((inputSeq, target))
                    count += 1
                    if count >= _trainingDataPairNumber:
                        break
                    if count % 10000 == 0:
                        print(f"{makeDatBoi()} {babyName}: generated {count}x trainingDataPairs!")
                else:
                    print(f"skipping <UNK> - inputSeq: {inputSeq}, target: {target}")
            return trainingDataPairs

    def saveVocab(self):
        with self.v_counsellor.infodump("saveVocab") as ʕっʘ‿ʘʔっ:
            os.makedirs(self.vocabCache, exist_ok = True)  # Ensure directory exists
            with open(self.vocabListFile, "w", encoding="utf-8") as f:
                ʕっʘ‿ʘʔっ("save vocabList")
                json.dump(self.vocabList, f, indent = 4)
            with open(self.tokenToIndexFile, "w", encoding="utf-8") as f:
                ʕっʘ‿ʘʔっ("save tokenToIndex")
                json.dump(self.tokenToIndex, f, indent = 4)
            with open(self.indexToTokenFile, "w", encoding="utf-8") as f:
                ʕっʘ‿ʘʔっ("save indexToToken")
                json.dump(self.indexToToken, f, indent = 4)

    def loadVocab(self):
        with self.v_counsellor.infodump("loadVocab") as ʕっʘ‿ʘʔっ:
            try:
                with open(self.vocabListFile, 'r', encoding='utf-8') as f:
                    ʕっʘ‿ʘʔっ("load vocabList")
                    self.vocabList = json.load(f)
                with open(self.tokenToIndexFile, 'r', encoding='utf-8') as f:
                    ʕっʘ‿ʘʔっ("load tokenToIndex")
                    self.tokenToIndex = json.load(f)
                with open(self.indexToTokenFile, 'r', encoding='utf-8') as f:
                    ʕっʘ‿ʘʔっ("load indexToToken")
                    self.indexToToken = {int(k): v for k, v in json.load(f).items()} # ensures that keys are integers!
                print("vocab files loaded successfully!")
                return bool(self.vocabList and self.tokenToIndex and self.indexToToken)
            except (FileNotFoundError, json.JSONDecodeError):
                print("vocab files not found or invalid... rebuilding vocab...")
                return False

if __name__ == "__main__":
    counsellor = type("Dummy", (), {"infodump": lambda self, label: open(os.devnull, 'w')})()
    librarian = LIBRARIAN(_counsellor = counsellor, _vocabSize = 4200, _baseTokenizerPath = "BRAIN/vocabCache/tokenizer_2000.json")
    print(f"--- 2000-{vocabSize} ---: {librarian.vocabList[2000:vocabSize]}")
    print(f"--- 1701-2000 ---: {librarian.vocabList[1701:2000]}")
    print(f"--- 1001-1700 ---: {librarian.vocabList[301:1700]}")
    print(f"--- 301-1000 ---: {librarian.vocabList[301:1000]}")
    print(f"--- 101-300 ---: {librarian.vocabList[101:300]}")
    print(f"--- Top 100 ---: {librarian.vocabList[:100]}")
    print(f"vocab size: {len(librarian.vocabList)}")
    print(f"Top 20 tokens: {librarian.vocabList[:20]}")

    #print(vocab.huggingTokenizer("charis and elodie are very cool, elodies pretty and charis is very suave, they're sexy bitches, we love these girls and we want to see them living their best lives bruv"))
    sample_text = "charis and elodie are very cool, elodies pretty and charis is very suave, they're sexy bitches, we love these girls and we want to see them living their best lives bruv"
    tokenizedOutput = librarian.tokenizeText(sample_text)
    print(f"Tokenized: {tokenizedOutput}")