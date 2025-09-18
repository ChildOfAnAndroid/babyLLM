# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# VOCAB: TRAINING GENERATION AND TOKENIZATION
# brain/LAYERS/vocab.py
# v4.15

from collections import Counter
from config import *
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, ByteLevelBPETokenizer
from tokenizers.processors import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import os, re, json, random, torch, io
from SHKAIRA.notebook.tools.genBoi import *
from collections import defaultdict
from textCleaningTool import clean_text
import csv

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

        # small LRU-ish cache for per-file tokens to avoid re-tokenizing
        # during autonomy; keep bounded to prevent memory growth
        self._file_token_cache: dict[str, list[str]] = {}
        # rolling dynamic tokens from live training buffer (bounded)
        self._dynamic_tokens: list[str] = []
        self._dynamic_tokens_max = 20000  # ~lightweight ring buffer

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
                # Do NOT pre-load full training corpora into memory.
                # Maintain an empty base token buffer; call sites should
                # pass tokens explicitly or use tokens_from_file.
                self.tokens: list[str] = []
            else:
                if debugPrints: ʕっʘ‿ʘʔっ("building vocab from tokenizer...")
                self.buildVocabMap()
                self.saveVocab()
                print(f"saved vocab data to {self.vocabCache}!")
                self.tokens: list[str] = []

    def tokenizeText(self, _text):
        with self.v_counsellor.infodump("tokenizeText") as ʕっʘ‿ʘʔっ:
            ids = self.tokenizer.encode(_text)
            if debugPrints:
                print(f"tokenizing: {_text}")
                print(f"token ids: {ids}")
            return [self.indexToToken.get(idx, self.unkToken) for idx in ids]  # Convert indexs back to strings
        
    def decodeIDs(self, _ids):
        with self.v_counsellor.infodump("decodeIDs") as ʕっʘ‿ʘʔっ:
            decoded = self.tokenizer.decode(_ids)
            # keep leading spaces so tokens reflect their true form (e.g. " charis")
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

    def loadSingleFile(self, path: str, ftype: str = "text", *, max_chars: int = 900000, strategy: str = "head") -> str:
        """Load and clean a single file according to its type.

        Supported types: 'text', 'json', 'discord_json', 'discord_txt',
        'reddit_post', 'reddit_comment'. Returns a cleaned lowercased string
        (possibly truncated to max_chars).
        """
        raw: str = ""
        try:
            typ = (ftype or "text").lower()
            if typ == "discord_json":
                # Stream JSON array of strings to avoid loading whole file
                raw = self._load_json_array_windowed(path, max_chars=max_chars, strategy=strategy)
            elif typ == "discord_txt":
                # Efficient windowed read: head/tail/random chunk from file
                size = os.path.getsize(path)
                # Read slightly extra to reduce broken-line artifacts
                extra = 4096
                read_len = min(max_chars + extra, size)
                if strategy == "tail":
                    start = max(0, size - read_len)
                elif strategy == "random" and size > read_len:
                    start = random.randint(0, size - read_len)
                else:
                    start = 0
                with open(path, "rb") as fb:
                    fb.seek(start)
                    chunk = fb.read(read_len)
                text = chunk.decode("utf-8", errors="ignore")
                # Keep full lines: drop first/last partials
                lines = text.splitlines()
                if lines:
                    if start > 0 and len(lines) > 1:
                        lines = lines[1:]  # drop likely-partial first line
                    joined = "\n".join(lines)
                    if not joined.strip():
                        # flat chunk; reflow
                        raw = self._reflow_to_lines(text, max_chars)
                    elif len(joined) > max_chars:
                        # reflow to keep line sizes reasonable and within budget
                        raw = self._reflow_to_lines(joined[: max_chars + 512], max_chars)
                    else:
                        raw = joined
                else:
                    # no natural line breaks: reflow to pseudo-lines
                    raw = self._reflow_to_lines(text, max_chars)
            elif typ == "json":
                # Try streaming if it's a JSON array of strings; otherwise fallback
                streamed = self._load_json_array_windowed(path, max_chars=max_chars, strategy=strategy, soft=True)
                if streamed is not None:
                    raw = streamed
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            raw = "\n".join(map(str, data))
                        elif isinstance(data, dict):
                            raw = json.dumps(data, ensure_ascii=False)
                        else:
                            raw = str(data)
            elif typ in ("reddit_post", "reddit_comment"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    rdr = csv.DictReader(f)
                    bodies = []
                    for row in rdr:
                        body = (row.get('body', '') or '').strip()
                        if body:
                            bodies.append(body)
                    raw = "\n".join(bodies)
            else:
                # Plain text: head/tail/random window
                size = os.path.getsize(path)
                extra = 4096
                read_len = min(max_chars + extra, size)
                if strategy == "tail":
                    start = max(0, size - read_len)
                elif strategy == "random" and size > read_len:
                    start = random.randint(0, size - read_len)
                else:
                    start = 0
                with open(path, "rb") as fb:
                    fb.seek(start)
                    chunk = fb.read(read_len)
                text = chunk.decode("utf-8", errors="ignore")
                # Keep whole lines when possible
                lines = text.splitlines()
                if lines:
                    if start > 0 and len(lines) > 1:
                        lines = lines[1:]
                    joined = "\n".join(lines)
                    if not joined.strip():
                        raw = self._reflow_to_lines(text, max_chars)
                    elif len(joined) > max_chars:
                        raw = self._reflow_to_lines(joined[: max_chars + 512], max_chars)
                    else:
                        raw = joined
                else:
                    raw = self._reflow_to_lines(text, max_chars)
        except Exception as e:
            print(f"[LIBRARIAN.loadSingleFile] failed to read {path}: {e}")
            return ""

        if not raw:
            return ""
        raw = raw[:max_chars]
        try:
            return clean_text(raw)
        except Exception:
            # Fallback: minimal clean
            return re.sub(r'[ \t]+', ' ', raw).strip().lower()

    def _reflow_to_lines(self, text: str, max_chars: int, max_line_len: int = 240, min_line_len: int = 20) -> str:
        """Reflow long, newline-free text into short lines for snippet mining.

        Produces lines in [min_line_len, max_line_len] where possible and caps
        the total size to max_chars. Keeps whitespace simple and avoids giant lines.
        """
        t = (text or "").strip()
        if not t:
            return ""
        # Tokenize on whitespace; keep it simple and fast
        words = re.findall(r"\S+", t)
        if not words:
            return t[:max_chars]
        out = []
        cur = []
        cur_len = 0
        budget = max_chars
        for w in words:
            wlen = len(w) if not cur else len(w) + 1
            if cur_len + wlen > max_line_len and cur_len >= min_line_len:
                line = " ".join(cur)
                out.append(line)
                budget -= len(line) + 1
                if budget <= 0:
                    break
                cur = [w]
                cur_len = len(w)
            else:
                cur.append(w)
                cur_len += wlen
        if cur and budget > 0:
            line = " ".join(cur)
            out.append(line)
        joined = "\n".join(out)
        return joined[:max_chars]

    def _iter_json_array_items(self, path: str):
        """Incrementally iterate items of a top-level JSON array without loading all.

        Yields each parsed JSON value. Designed for arrays of strings but works
        for generic JSON values. Keeps the internal buffer bounded.
        """
        dec = json.JSONDecoder()
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            buf = ""
            in_array = False
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                buf += chunk
                while True:
                    if not in_array:
                        i = buf.find("[")
                        if i == -1:
                            # keep a small tail to catch '[' spanning chunks
                            buf = buf[-2:]
                            break
                        in_array = True
                        buf = buf[i + 1:]
                    # skip whitespace and commas
                    m = re.match(r"\s*(,)?\s*", buf)
                    if m:
                        buf = buf[m.end():]
                    if not buf:
                        break
                    if buf[0] == ']':
                        return
                    try:
                        val, idx = dec.raw_decode(buf)
                    except json.JSONDecodeError:
                        # Need more data; trim buffer growth
                        if len(buf) > 1_000_000:
                            buf = buf[-4096:]
                        break
                    yield val
                    buf = buf[idx:]

    def _load_json_array_windowed(self, path: str, *, max_chars: int, strategy: str, soft: bool = False) -> str | None:
        """Load a window from a JSON array of strings without full load.

        - strategy: 'head'|'tail'|'random'
        - max_chars: character budget for joined result
        - soft=True returns None if the structure doesn't look like array-of-strings
        """
        try:
            # First few items to determine structure and to build head/tail
            items_iter = self._iter_json_array_items(path)
            if strategy not in ("head", "tail", "random"):
                strategy = "head"

            # Helper: append while respecting budget
            def join_with_budget(lines: list[str]) -> str:
                out = []
                total = 0
                for ln in lines:
                    if not isinstance(ln, str):
                        ln = str(ln)
                    if total + len(ln) + 1 > max_chars:
                        if total == 0 and len(ln) >= 1:
                            out.append(ln[: max_chars])
                            total = max_chars
                        break
                    out.append(ln)
                    total += len(ln) + 1
                return "\n".join(out)

            if strategy == "head":
                acc: list[str] = []
                total = 0
                seen_any = False
                for it in items_iter:
                    seen_any = True
                    s = it if isinstance(it, str) else str(it)
                    if total + len(s) + 1 > max_chars:
                        break
                    acc.append(s)
                    total += len(s) + 1
                if not seen_any and soft:
                    return None
                return join_with_budget(acc)

            if strategy == "tail":
                from collections import deque
                dq: deque[str] = deque()
                total = 0
                seen_any = False
                for it in items_iter:
                    seen_any = True
                    s = it if isinstance(it, str) else str(it)
                    dq.append(s)
                    total += len(s) + 1
                    while total > max_chars and dq:
                        left = dq.popleft()
                        total -= len(left) + 1
                if not seen_any and soft:
                    return None
                return "\n".join(list(dq))[:max_chars]

            # random reservoir sample
            k = max(8, min(1000, max_chars // 120))
            sample: list[str] = []
            n = 0
            for it in items_iter:
                s = it if isinstance(it, str) else str(it)
                if n < k:
                    sample.append(s)
                else:
                    j = random.randint(0, n)
                    if j < k:
                        sample[j] = s
                n += 1
            if n == 0 and soft:
                return None
            random.shuffle(sample)
            return join_with_budget(sample)
        except Exception:
            if soft:
                return None
            raise

    def tokens_from_file(self, path: str, ftype: str = "text"):
        key = f"{ftype}::{path}"
        if key in self._file_token_cache:
            return self._file_token_cache[key]
        txt = self.loadSingleFile(path, ftype)
        toks = self.tokenizeText(txt) if txt else []
        self._file_token_cache[key] = toks
        # Prevent unbounded growth: drop an arbitrary cached entry if over limit
        if len(self._file_token_cache) > 16:
            try:
                self._file_token_cache.pop(next(iter(self._file_token_cache)))
            except Exception:
                pass
        return toks
        
    def add_training_text(self, text):
        """Append new training text into a bounded internal token buffer.

        Accepts either a raw string (will tokenize) or an iterable of token
        strings. Keeps a lightweight rolling window to avoid memory growth.
        """
        try:
            if isinstance(text, str):
                toks = self.tokenizeText(text)
            else:
                toks = list(text)
            if not toks:
                return 0
            self._dynamic_tokens.extend(toks)
            # Clamp to ring buffer size
            if len(self._dynamic_tokens) > self._dynamic_tokens_max:
                overflow = len(self._dynamic_tokens) - self._dynamic_tokens_max
                if overflow > 0:
                    del self._dynamic_tokens[:overflow]
            return len(toks)
        except Exception:
            return 0

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

    def genTrainingData_weighted(self, _windowMAX, _trainingDataPairNumber):
        # 1. Create a "pool" of training pairs for each data source type
        source_pools = defaultdict(list)
        print("Generating training pairs from each data source...")
        
        # This assumes trainingFilePath_dict_weighted is available to the librarian
        for source_info in trainingFilePath_dict_weighted:
            source_type = source_info['type']
            source_path = source_info['in']
            
            # Load and tokenize text for this specific file
            # (You'd need a way to load single files, modifying your loadTrainingData)
            text = self.loadSingleFile(source_path)
            tokens = self.tokenizeText(text)

            # Generate all possible pairs from this source
            i = 0
            while i < len(tokens) - (_windowMAX * 2):
                inputSeq = tokens[i : i + _windowMAX]
                targetSeq = tokens[i + _windowMAX : i + _windowMAX * 2]
                source_pools[source_type].append((inputSeq, targetSeq))
                i += 1 # Using a stride of 1 here to get all pairs

        # 2. Create the final list by sampling from pools based on weights
        final_training_pairs = []
        source_weights = {info['type']: info['weight'] for info in trainingFilePath_dict_weighted}
        
        # Normalize weights to get probabilities
        total_weight = sum(source_weights.values())
        source_probs = {source: weight / total_weight for source, weight in source_weights.items()}
        
        source_types = list(source_probs.keys())
        source_p_values = list(source_probs.values())

        print("Sampling from source pools to build final training set...")
        for _ in range(_trainingDataPairNumber):
            # Choose a source type based on its weight
            chosen_source = random.choices(source_types, weights=source_p_values, k=1)[0]
            
            # Pick a random training pair from that source's pool
            if source_pools[chosen_source]:
                pair = random.choice(source_pools[chosen_source])
                final_training_pairs.append(pair)
                
        random.shuffle(final_training_pairs) # Final shuffle of the sampled pairs
        return final_training_pairs

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

# __main__ test harness removed (legacy)
