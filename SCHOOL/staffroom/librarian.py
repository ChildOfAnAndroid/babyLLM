# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# VOCAB: TRAINING GENERATION AND TOKENIZATION
# BRAIN/LAYERS/vocab.py

from collections import Counter
from config import *
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, ByteLevelBPETokenizer
from tokenizers.processors import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import os, re, json, random, torch, io
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
    def __init__(self, _counsellor, _vocabSize = vocabSize, _vocabPath = None, _baseTokenizerPath = None, _forceRetrain = False):
        self.v_counsellor = _counsellor
        self.vocabSize = _vocabSize
        self.unkToken = "<UNK>"
        self.vocabCache = vocabCachePath
        self.vocabFilename = f"vocab{_vocabSize}_{minTokenFreq}"

        self.tokenizerFilename = f"tokenizer_{_vocabSize}.json"
        self.tokenizerPath = _vocabPath or os.path.join(self.vocabCache, self.tokenizerFilename)
        self.tokenizerLockFile = os.path.join(self.vocabCache, f"{self.tokenizerFilename}.lock")

        self.vocabListFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_list.json")
        self.tokenToIndexFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_to_index.json")
        self.indexToTokenFile = os.path.join(self.vocabCache, f"{self.vocabFilename}_to_token.json")

        self.vocabList = []
        self.tokenToIndex = {}
        self.indexToToken = {}

        self.baseTokenizerPath = _baseTokenizerPath

        os.makedirs(self.vocabCache, exist_ok = True)

        with self.v_counsellor.infodump("__init__") as ʕっʘ‿ʘʔっ:

            shouldTrain = _forceRetrain or not os.path.exists(self.tokenizerPath) or not os.path.exists(self.tokenizerLockFile)

            if shouldTrain:
                if debugPrints: ʕっʘ‿ʘʔっ("TRAINING NEW TOKENIZER")
                print("training new tokenizer...")
                tokenizerModel = Tokenizer(models.BPE(unk_token = self.unkToken))
                tokenizerModel.pre_tokenizer = pre_tokenizers.ByteLevel()
                trainer = trainers.BpeTrainer(
                    vocab_size = self.vocabSize,
                    min_frequency = minTokenFreq,
                    special_tokens=[self.unkToken]
                )

                with open(trainingFilePath, "r", encoding="utf-8") as f:
                    training_data = [f.read().lower()]
                tokenizerModel.train_from_iterator(training_data, trainer)
                tokenizerModel.save(self.tokenizerPath)

                with open(self.tokenizerLockFile, "w") as f:
                    f.write("LOCKED") # avoid retraining by accident lol

            if debugPrints and not shouldTrain:
                ʕっʘ‿ʘʔっ("LOADING EXISTING TOKENIZER")
                print("loading existing tokenizer...")

            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizerPath, unk_token=self.unkToken, add_prefix_space=True,)
            # byte-level decoder so decoded text is plain
            self.tokenizer.backend_tokenizer.decoder = ByteLevelDecoder()

            self.buildVocabMap()

            if self.loadVocab():
                if debugPrints: ʕっʘ‿ʘʔっ("loaded vocab from files...")
                self.trainingDataPairs = self.loadTrainingData(trainingFilePath_arr)
                self.tokens = self.tokenizeText(self.trainingDataPairs)
            else:
                if debugPrints: ʕっʘ‿ʘʔっ("building vocab from tokenizer...")
                self.buildVocabMap()
                self.saveVocab()
                print(f"saved vocab data to {self.vocabCache}!")

    def tokenizeText(self, _text):
        with self.v_counsellor.infodump("tokenizeText") as ʕっʘ‿ʘʔっ:
            ids = self.tokenizer.encode(_text)
            if debugPrints:
                print(f"tokenizing: {_text}")
                print(f"token ids: {ids}")
            return [self.indexToToken.get(idx, self.unkToken) for idx in ids]  # Convert indexs back to strings
        
    def decodeIDs(self, _ids):
        with self.v_counsellor.infodump("decodeIDs") as ʕっʘ‿ʘʔっ:
            decoded = self.tokenizer.decode(_ids).lstrip()
            if debugPrints:
                print(f"decoding: {_ids}")
                print(f"decoded: {decoded}")
            return decoded
        
    def buildVocabMap(self):
        with self.v_counsellor.infodump("buildVocabMap") as ʕっʘ‿ʘʔっ:
            if debugPrints: ʕっʘ‿ʘʔっ("getting vocab dictionary from tokenizer...")
            invVocab = self.tokenizer.get_vocab()
            if debugPrints: ʕっʘ‿ʘʔっ("ordering by index...")
            sortedTokens = sorted(invVocab.items(), key = lambda item: item[1])  # sort by index
            self.vocabList = [token for token, idx in sortedTokens]
            if debugPrints: ʕっʘ‿ʘʔっ("mapping vocab dicts...")
            self.tokenToIndex = {token: idx for token, idx in sortedTokens}
            self.indexToToken = {idx: token for token, idx in sortedTokens}
            if debugPrints: ʕっʘ‿ʘʔっ("ensuring <UNK> is in the vocab...")
            if self.unkToken not in self.tokenToIndex:
                self.vocabList.append(self.unkToken)
                unk_index = len(self.vocabList) - 1
                self.tokenToIndex[self.unkToken] = unk_index
                self.indexToToken[unk_index] = self.unkToken
            print(f"final vocab size: {len(self.vocabList)}")
            print(f"first 20 tokens: {self.vocabList[:20]}")

    def huggingTokenizer(self, _text): return self.tokenizer.tokenize(_text)

    def loadTrainingData(self, _filepaths, _chunkSize = V_chunkSizeLoadData, _dataCharactersToLoad = 900000):
        with self.v_counsellor.infodump("loadTrainingData") as ʕっʘ‿ʘʔっ:
            buffer = io.StringIO()
            loadedChars = 0
            for path in _filepaths:
                with open(path, "r", encoding="utf-8") as f:
                    while loadedChars < _dataCharactersToLoad:
                        readSize = min(_chunkSize, _dataCharactersToLoad - loadedChars)
                        chunk = f.read(readSize)
                        if not chunk:
                            break
                        buffer.write(chunk)
                        loadedChars += len(chunk)
                if loadedChars >= _dataCharactersToLoad:
                    break

            result = re.sub(r'[ \t]+', ' ', buffer.getvalue())
            print(f"loaded {len(result)} characters of training data!")
            return result
        
    def add_training_text(self, text):
        """append new training text to the internal buffer."""
        self.tokens.extend(text)

    def genTrainingData(self, _windowMAX = numTokensPerStepSTART, _startIndex = trainingStartIndex, _trainingDataPairNumber = 1, _stride = trainingDataStride, _tokens=None):
        with self.v_counsellor.infodump("genTrainingData") as ʕっʘ‿ʘʔっ:
            count = 0
            tokens = _tokens if _tokens is not None else self.tokens
            if debugPrints: ʕっʘ‿ʘʔっ("check if windowMax is tensor?")
            if isinstance(_windowMAX, torch.Tensor):
                _windowMAX = _windowMAX.item()
            
            if debugPrints: ʕっʘ‿ʘʔっ("allows for random start")
            if _startIndex == 'random':
                _startIndex = random.randint(0, len(tokens) - _windowMAX - 1)

            end = len(tokens) - _windowMAX

            i = _startIndex
            while count < _trainingDataPairNumber and i < len(tokens) - _windowMAX:
                if debugPrints: ʕっʘ‿ʘʔっ("generate training pairs")
                inputSeq = tokens[i:i + _windowMAX]
                target = tokens[i + _windowMAX:i + _windowMAX + _windowMAX]
                if len(target) < _windowMAX:
                    i += int(_stride)
                    continue
                if all(t in self.vocabList for t in inputSeq + target):
                    yield (inputSeq, target)
                    count += 1
                    if count % 1000 == 0:
                        print(f"{makeDatBoi()} {babyName}: generated {count}x trainingDataPairs!")
                else:
                    print(f"skipping <UNK> - inputSeq: {inputSeq}, target: {target}")
                i += int(_stride)

    def saveVocab(self):
        with self.v_counsellor.infodump("saveVocab") as ʕっʘ‿ʘʔっ:
            os.makedirs(self.vocabCache, exist_ok = True)  # Ensure directory exists
            with open(self.vocabListFile, "w", encoding="utf-8") as f:
                if debugPrints: ʕっʘ‿ʘʔっ("save vocabList")
                json.dump(self.vocabList, f, indent = 4)
            with open(self.tokenToIndexFile, "w", encoding="utf-8") as f:
                if debugPrints: ʕっʘ‿ʘʔっ("save tokenToIndex")
                json.dump(self.tokenToIndex, f, indent = 4)
            with open(self.indexToTokenFile, "w", encoding="utf-8") as f:
                if debugPrints: ʕっʘ‿ʘʔっ("save indexToToken")
                json.dump(self.indexToToken, f, indent = 4)

    def loadVocab(self):
        with self.v_counsellor.infodump("loadVocab") as ʕっʘ‿ʘʔっ:
            try:
                with open(self.vocabListFile, 'r', encoding='utf-8') as f:
                    if debugPrints: ʕっʘ‿ʘʔっ("load vocabList")
                    self.vocabList = json.load(f)
                with open(self.tokenToIndexFile, 'r', encoding='utf-8') as f:
                    if debugPrints: ʕっʘ‿ʘʔっ("load tokenToIndex")
                    self.tokenToIndex = json.load(f)
                with open(self.indexToTokenFile, 'r', encoding='utf-8') as f:
                    if debugPrints: ʕっʘ‿ʘʔっ("load indexToToken")
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
    sampleText = "ʕっʘ‿ʘʔっ ʕっʘ‿ʘʔっ ʕっʘ‿ʘʔっ charis and elodie are very cool, ʕっʘ‿ʘʔっ ʕっʘ‿ʘʔっ ʕっʘ‿ʘʔっ elodies pretty and charis is very suave, they're sexy bitches, we love these girls and we want to see them living their best lives bruv"
    tokenizedOutput = librarian.tokenizeText(sampleText)
    print(f"sample text: {sampleText}")
    print(f"tokenized: {tokenizedOutput}")