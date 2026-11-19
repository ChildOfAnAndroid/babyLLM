# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ ---
# VOCAB: TRAINING GENERATION AND TOKENIZATION
# brain/LAYERS/vocab.py
# v1.1

import csv
import itertools
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from collections import defaultdict, deque
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
from numpy.lib.format import open_memmap
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
)
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast

from config import *
from SHKAIRA.notebook.tools.genBoi import *
from textCleaningTool import clean_text
from utils.text_rendering import to_british_english
from utils.icharis2_ingest import read_training_manifest
import CONFIG_trainingData as curriculum

_BRACKETED_CHILD_HANDLE = re.compile(
    r"\[\[\s*(?:childofagamingdroid|child of an android|childofanandroid|childo|coaa)\s*\]\]",
    re.IGNORECASE,
)
_BRACKETED_SIMPLE_LABEL = re.compile(
    r"\[\[\s*([a-z0-9 _.'-]{1,48})\s*\]\]", re.IGNORECASE
)


# Keep authored email above the explicit quoted/forwarded thread boundary.
_EMAIL_THREAD = re.compile(
    r"(?im)^\s*(?:on [^\n]{1,200} wrote:|[- ]*forwarded message[- ]*|"
    r"[- ]+original message[- ]*|from:[^\n]+\n(?:sent|date):)"
)


def _normalize_bracketed_handles_for_training(text: str) -> str:
    # Keep one canonical identity in training data instead of literal [[...]] handles.
    text = _BRACKETED_CHILD_HANDLE.sub(" charis ", text)
    return _BRACKETED_SIMPLE_LABEL.sub(lambda m: f" {m.group(1).strip()} ", text)


class _MemmapTokenSequence:
    """Lightweight list-like wrapper that chains multiple array-like segments."""

    def __init__(self, arrays: Sequence[Sequence]):
        self._segments: list[tuple[int, int, Sequence]] = []
        self._length = 0
        dtype = None
        sample_value = None
        for arr in arrays:
            if arr is None:
                continue
            try:
                seg_len = int(len(arr))
            except TypeError:
                continue
            if seg_len <= 0:
                continue
            start = self._length
            end = start + seg_len
            self._segments.append((start, end, arr))
            self._length = end
            if dtype is None and hasattr(arr, "dtype"):
                dtype = arr.dtype
            if sample_value is None and seg_len:
                try:
                    sample_value = arr[0]
                except Exception:
                    sample_value = None
        self.dtype = dtype
        if dtype is not None:
            self.is_numeric = bool(np.issubdtype(dtype, np.integer))
        else:
            self.is_numeric = isinstance(sample_value, (int, np.integer))

    def __len__(self) -> int:
        return self._length

    def _locate(self, idx: int) -> tuple[Sequence, int]:
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError("memmap index out of range")
        for start, end, arr in self._segments:
            if idx < end:
                return arr, idx - start
        raise IndexError("memmap index lookup failed")

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(self._length)
            if step != 1:
                return [self[i] for i in range(start, stop, step)]
            if start >= stop:
                return []
            out: list = []
            idx = start
            while idx < stop:
                arr, local_idx = self._locate(idx)
                seg_len = len(arr)
                take = min(stop - idx, seg_len - local_idx)
                view = arr[local_idx : local_idx + take]
                if hasattr(view, "tolist"):
                    out.extend(view.tolist())
                else:
                    out.extend(list(view))
                idx += take
            return out
        arr, local_idx = self._locate(int(item))
        val = arr[local_idx]
        return val.item() if hasattr(val, "item") else val

    def flush(self) -> None:
        for _, _, arr in self._segments:
            if hasattr(arr, "flush"):
                arr.flush()

    def __iter__(self) -> Iterator:
        for _, _, arr in self._segments:
            for val in arr:
                yield val.item() if hasattr(val, "item") else val



class ProvenancedTrainingPair(tuple):
    """Two-item training pair retaining target-token provenance out of band.

    Existing readers can still unpack (input_seq, target_seq) unchanged.
    The metadata describes TARGET positions, not decoded text or model guesses.
    """

    def __new__(cls, input_seq, target_seq, target_provenance):
        pair = super().__new__(cls, (input_seq, target_seq))
        pair.target_provenance = tuple(target_provenance)
        return pair


class _TrainingPairStream:
    """Re-iterable wrapper around a streaming training pair generator."""

    def __init__(
        self,
        factory: Callable[[], Iterator[tuple[list, list]]],
        length: int | None = None,
        description: str | None = None,
    ) -> None:
        self._factory = factory
        self._length = length
        self.description = description or ""

    def __iter__(self) -> Iterator[tuple[list, list]]:
        return self._factory()

    def __len__(self) -> int:
        if self._length is None:
            raise TypeError("stream length is unknown")
        return self._length

    def __repr__(self) -> str:
        desc = f" {self.description}" if self.description else ""
        length = self._length if self._length is not None else "?"
        return f"<TrainingPairStream{desc} len={length}>"


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
    def __init__(
        self,
        _counsellor,
        _vocabSize=vocabSize,
        _vocabPath=None,
        _baseTokenizerPath=None,
        _forceRetrain=False,
    ):
        self.v_counsellor = _counsellor
        self.vocabSize = _vocabSize
        self.unkToken = "<UNK>"
        self.vocabCache = vocabCachePath
        self.vocabFilename = f"vocab{_vocabSize}_{minTokenFreq}"

        self.tokenizerFilename = f"tokenizer_{_vocabSize}.json"
        self.tokenizerPath = _vocabPath or os.path.join(
            self.vocabCache, self.tokenizerFilename
        )
        self.tokenizerLockFile = os.path.join(
            self.vocabCache, f"{self.tokenizerFilename}.lock"
        )

        self.vocabListFile = os.path.join(
            self.vocabCache, f"{self.vocabFilename}_list.json"
        )
        self.tokenToIndexFile = os.path.join(
            self.vocabCache, f"{self.vocabFilename}_to_index.json"
        )
        self.indexToTokenFile = os.path.join(
            self.vocabCache, f"{self.vocabFilename}_to_token.json"
        )

        self.vocabList = []
        self.tokenToIndex = {}
        self.indexToToken = {}

        self.baseTokenizerPath = _baseTokenizerPath

        os.makedirs(self.vocabCache, exist_ok=True)

        # small LRU-ish cache for per-file tokens to avoid re-tokenizing
        # during autonomy; keep bounded to prevent memory growth
        self._file_token_cache: dict[str, list[str]] = {}
        # rolling dynamic tokens from live training buffer (bounded)
        self._dynamic_tokens_max = 20000  # ~lightweight ring buffer
        self._dynamic_tokens: deque[str] = deque(maxlen=self._dynamic_tokens_max)
        self._tokens_are_indices = False
        self._token_stream_factories: list[Callable[[], Iterator]] = []
        self._training_token_count_estimate: int | None = None
        self._icharis2_snapshot = None
        self._icharis2_version = None
        self._icharis2_publish_after = 0.0
        self._icharis2_sampled = set()
        self._icharis2_anniversary_seen = set()
        self._icharis2_anniversary_day = None

        with self.v_counsellor.infodump("__init__") as ʕっʘ‿ʘʔっ:
            shouldTrain = (
                _forceRetrain
                or not os.path.exists(self.tokenizerPath)
                or not os.path.exists(self.tokenizerLockFile)
            )

            if shouldTrain:
                if debugPrints:
                    ʕっʘ‿ʘʔっ("TRAINING NEW TOKENIZER")
                print("training new tokenizer...")
                tokenizerModel = Tokenizer(models.BPE(unk_token=self.unkToken))
                tokenizerModel.pre_tokenizer = pre_tokenizers.ByteLevel()
                trainer = trainers.BpeTrainer(
                    vocab_size=self.vocabSize,
                    min_frequency=minTokenFreq,
                    special_tokens=[self.unkToken],
                )

                with open(trainingFilePath, "r", encoding="utf-8") as f:
                    training_data = [f.read().lower()]
                tokenizerModel.train_from_iterator(training_data, trainer)
                tokenizerModel.save(self.tokenizerPath)

                with open(self.tokenizerLockFile, "w") as f:
                    f.write("LOCKED")  # avoid retraining by accident lol

            if debugPrints and not shouldTrain:
                ʕっʘ‿ʘʔっ("LOADING EXISTING TOKENIZER")
                print("loading existing tokenizer...")

            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=self.tokenizerPath,
                unk_token=self.unkToken,
                add_prefix_space=True,
            )
            # byte-level decoder so decoded text is plain
            self.tokenizer.backend_tokenizer.decoder = ByteLevelDecoder()

            self.buildVocabMap()

            if self.loadVocab():
                if debugPrints:
                    ʕっʘ‿ʘʔっ("loaded vocab from files...")
                # Do NOT pre-load full training corpora into memory.
                # Maintain an empty base token buffer; call sites should
                # pass tokens explicitly or use tokens_from_file.
                self.tokens: list[str] = []
            else:
                if debugPrints:
                    ʕっʘ‿ʘʔっ("building vocab from tokenizer...")
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
            return [
                self.indexToToken.get(idx, self.unkToken) for idx in ids
            ]  # Convert indexs back to strings

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
            if debugPrints:
                ʕっʘ‿ʘʔっ("getting vocab dictionary from tokenizer...")
            invVocab = self.tokenizer.get_vocab()
            if debugPrints:
                ʕっʘ‿ʘʔっ("ordering by index...")
            sortedTokens = sorted(
                invVocab.items(), key=lambda item: item[1]
            )  # sort by index
            self.vocabList = [token for token, idx in sortedTokens]
            if debugPrints:
                ʕっʘ‿ʘʔっ("mapping vocab dicts...")
            self.tokenToIndex = {token: idx for token, idx in sortedTokens}
            self.indexToToken = {idx: token for token, idx in sortedTokens}
            if debugPrints:
                ʕっʘ‿ʘʔっ("ensuring <UNK> is in the vocab...")
            if self.unkToken not in self.tokenToIndex:
                self.vocabList.append(self.unkToken)
                unk_index = len(self.vocabList) - 1
                self.tokenToIndex[self.unkToken] = unk_index
                self.indexToToken[unk_index] = self.unkToken
            print(f"final vocab size: {len(self.vocabList)}")
            print(f"first 20 tokens: {self.vocabList[:20]}")



    def loadTrainingData(
        self, _filepaths, _chunkSize=V_chunkSizeLoadData, _dataCharactersToLoad=900000
    ):
        with self.v_counsellor.infodump("loadTrainingData") as ʕっʘ‿ʘʔっ:
            self._token_stream_factories = []
            memmaps: list[np.memmap] = []
            text_factories: list[Callable[[], Iterator]] = []
            loaded_chars_estimate = 0

            for path in _filepaths:
                if not path:
                    continue
                ext = os.path.splitext(path)[1].lower()
                if ext == ".npy":
                    try:
                        mmap_arr = open_memmap(path, mode="r")
                    except Exception as e:
                        print(f"[WARN] Failed to memory-map {path}: {e}")
                    else:
                        memmaps.append(mmap_arr)
                        self._token_stream_factories.append(
                            lambda arr=mmap_arr: self._iter_memmap_tokens(arr)
                        )
                        try:
                            loaded_chars_estimate += int(len(mmap_arr))
                        except Exception:
                            pass
                    continue

                remaining = None
                if _dataCharactersToLoad is not None:
                    remaining = max(0, _dataCharactersToLoad - loaded_chars_estimate)
                    if remaining <= 0:
                        break
                    try:
                        file_size = os.path.getsize(path)
                    except OSError:
                        file_size = remaining
                    max_chars = min(remaining, file_size) if file_size else remaining
                else:
                    max_chars = None

                try:
                    factory = self._make_text_stream_factory(
                        path,
                        chunk_size=_chunkSize,
                        max_chars=max_chars,
                    )
                except FileNotFoundError:
                    print(f"[WARN] Training data file not found: {path}")
                    continue
                except Exception as e:
                    print(f"[WARN] Failed to prepare stream for {path}: {e}")
                    continue

                text_factories.append(factory)
                if max_chars is not None:
                    loaded_chars_estimate += int(max_chars)

            if text_factories:
                self._token_stream_factories.extend(text_factories)

            if memmaps:
                sequence = _MemmapTokenSequence(memmaps)
                self.tokens = sequence
                self._tokens_are_indices = sequence.is_numeric
                self._training_token_count_estimate = len(sequence)
                print(
                    f"memory-mapped {len(sequence)} training items from {len(memmaps)} file(s)"
                )
                return sequence

            # No memmaps: fall back to streaming text tokens without loading entire file
            self.tokens = []
            if self._token_stream_factories:
                self._tokens_are_indices = True
                self._training_token_count_estimate = None
                print(
                    f"prepared streaming token generators for {len(self._token_stream_factories)} text file(s)"
                )
                return ""

            print("[WARN] No training data sources available to load")
            return ""

    def _iter_memmap_tokens(
        self, mmap_arr: np.memmap, block_size: int = 131072
    ) -> Iterator[int]:
        block_size = max(1, int(block_size))
        try:
            length = int(len(mmap_arr))
        except Exception:
            length = int(mmap_arr.shape[0])
        for start in range(0, length, block_size):
            block = mmap_arr[start : start + block_size]
            for val in block:
                yield val.item() if hasattr(val, "item") else val

    def _make_text_stream_factory(
        self,
        path: str,
        *,
        chunk_size: int,
        max_chars: int | None,
        overlap: int = 256,
    ) -> Callable[[], Iterator[int]]:
        chunk_size = max(1, int(chunk_size))
        overlap = max(0, int(overlap))

        def factory() -> Iterator[int]:
            return self._iter_text_file_tokens(
                path,
                chunk_size=chunk_size,
                max_chars=max_chars,
                overlap=overlap,
            )

        return factory

    def _iter_text_file_tokens(
        self,
        path: str,
        *,
        chunk_size: int,
        max_chars: int | None,
        overlap: int,
    ) -> Iterator[int]:
        remaining = max_chars if max_chars is not None else None
        tail = ""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    if remaining is not None and remaining <= 0:
                        break
                    to_read = (
                        chunk_size if remaining is None else min(chunk_size, remaining)
                    )
                    chunk = f.read(to_read)
                    if not chunk:
                        break
                    if remaining is not None:
                        remaining -= len(chunk)
                    text = tail + chunk
                    if not text:
                        continue
                    text = _normalize_bracketed_handles_for_training(text)
                    tokens = self.tokenizer.encode(text)
                    if not tokens:
                        tail = text[-overlap:] if overlap else ""
                        continue
                    if overlap:
                        tail_text = text[-overlap:]
                        tail_tokens = (
                            self.tokenizer.encode(tail_text) if tail_text else []
                        )
                        emit_count = len(tokens) - len(tail_tokens)
                        if emit_count > 0:
                            for tok in tokens[:emit_count]:
                                yield tok
                        tail = tail_text
                    else:
                        for tok in tokens:
                            yield tok
                        tail = ""
                if tail:
                    for tok in self.tokenizer.encode(tail):
                        yield tok
        except FileNotFoundError:
            print(f"[WARN] Training data file not found during stream: {path}")
        except Exception as e:
            print(f"[WARN] Streaming read failed for {path}: {e}")

    def training_token_count_estimate(self) -> int:
        if self.tokens is not None:
            try:
                length = len(self.tokens)
                if length:
                    return length
            except Exception:
                pass
        if self._training_token_count_estimate is not None:
            return int(self._training_token_count_estimate)
        return len(self._dynamic_tokens)

    def _estimate_pair_count(
        self,
        length_estimate: int | None,
        dynamic_tokens: Sequence,
        *,
        window: int,
        stride: int,
        limit: int | None,
        start: int,
    ) -> int | None:
        if length_estimate is None:
            return None
        total = max(0, int(length_estimate)) + (
            len(dynamic_tokens) if dynamic_tokens else 0
        )
        required = window * 2
        if total < start + required:
            return 0
        available = total - (start + required) + stride
        if available < 0:
            positions = 0
        else:
            positions = available // stride
        positions = max(0, positions)
        if limit is not None:
            positions = min(positions, limit)
        return positions

    def _make_pair_iterator_factory(
        self,
        token_iterator_factory: Callable[[], Iterator],
        *,
        window: int,
        stride: int,
        limit: int | None,
        tokens_are_indices: bool,
        token_lookup: dict,
        start: int,
        token_provenance: Sequence | None = None,
    ) -> Callable[[], Iterator[tuple[list, list]]]:
        window = max(1, int(window))
        stride = max(1, int(stride))

        def make_pair(input_seq, target_seq, position):
            if token_provenance is None:
                return (input_seq, target_seq)
            first_target = start + position * stride + window
            metadata = token_provenance[first_target : first_target + window]
            if len(metadata) != len(target_seq):
                raise ValueError("Training token provenance is not aligned to targets")
            return ProvenancedTrainingPair(input_seq, target_seq, metadata)

        def factory() -> Iterator[tuple[list, list]]:
            iterator = token_iterator_factory()
            if start > 0:
                iterator = itertools.islice(iterator, start, None)
            buffer: deque = deque(maxlen=window * 2 + stride)
            produced = 0
            for token in iterator:
                buffer.append(token)
                if len(buffer) < window * 2:
                    continue
                while len(buffer) >= window * 2:
                    input_seq = list(itertools.islice(buffer, 0, window))
                    target_seq = list(itertools.islice(buffer, window, 2 * window))
                    if len(target_seq) < window:
                        break
                    if tokens_are_indices or (
                        input_seq and isinstance(input_seq[0], (int, np.integer))
                    ):
                        yield make_pair(input_seq, target_seq, produced)
                    elif all(
                        t in token_lookup
                        for t in itertools.chain(input_seq, target_seq)
                    ):
                        yield make_pair(input_seq, target_seq, produced)
                    else:
                        print(
                            f"skipping <UNK> - inputSeq sample: {input_seq[:4]}, target sample: {target_seq[:4]}"
                        )
                    produced += 1
                    if limit is not None and produced >= limit:
                        return
                    for _ in range(stride):
                        if buffer:
                            buffer.popleft()
                    if len(buffer) < window * 2:
                        break

        return factory

    def loadSingleFile(
        self,
        path: str,
        ftype: str = "text",
        *,
        max_chars: int = 900000,
        strategy: str = "head",
    ) -> str:
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
                raw = self._load_json_array_windowed(
                    path, max_chars=max_chars, strategy=strategy
                )
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
                        raw = self._reflow_to_lines(
                            joined[: max_chars + 512], max_chars
                        )
                    else:
                        raw = joined
                else:
                    # no natural line breaks: reflow to pseudo-lines
                    raw = self._reflow_to_lines(text, max_chars)
            elif typ == "json":
                # Try streaming if it's a JSON array of strings; otherwise fallback
                streamed = self._load_json_array_windowed(
                    path, max_chars=max_chars, strategy=strategy, soft=True
                )
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
                        body = (row.get("body", "") or "").strip()
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
                        raw = self._reflow_to_lines(
                            joined[: max_chars + 512], max_chars
                        )
                    else:
                        raw = joined
                else:
                    raw = self._reflow_to_lines(text, max_chars)
        except Exception as e:
            print(f"[LIBRARIAN.loadSingleFile] failed to read {path}: {e}")
            return ""

        if not raw:
            return ""

        # Provenance guard for archive/log files: downweight bot-self output to 1%
        # so historical logs don't re-enter and dominate without the live 1% rule.
        is_chat_log = "discordlog" in str(path).lower() or "twitchlog" in str(path).lower()
        if is_chat_log and raw:
            filtered_lines = []
            for line in raw.splitlines():
                line_str = str(line or "").strip()
                match = re.match(r"^\s*([^:\n]{1,64})\s*:\s*(.+)$", line_str)
                if match:
                    speaker = match.group(1).strip().lower()
                    if "(babyllm" in speaker or speaker in {"babyllm", "baby", "skunkllm"}:
                        if random.random() < 0.01:
                            filtered_lines.append(line_str)
                        continue
                filtered_lines.append(line_str)
            raw = "\n".join(filtered_lines)

        raw = raw[:max_chars]
        try:
            return clean_text(raw)
        except Exception:
            # Fallback: minimal clean
            return re.sub(r"[ \t]+", " ", raw).strip().lower()

    def sampleTrainingSources(self, count: int = 3) -> list[dict]:
        """Sample configured training sources according to their weights."""
        sources = [
            entry
            for entry in (trainingFilePath_dict_weighted or [])
            if isinstance(entry, dict) and entry.get("in")
        ]
        if not sources or count <= 0:
            return []
        weights = [max(0.0, float(entry.get("weight", 1.0) or 1.0)) for entry in sources]
        if not any(weights):
            weights = [1.0] * len(sources)

        sampled: list[dict] = []
        remaining_sources = list(sources)
        remaining_weights = list(weights)
        while remaining_sources and len(sampled) < count:
            index = random.choices(
                range(len(remaining_sources)), weights=remaining_weights, k=1
            )[0]
            sampled.append(remaining_sources.pop(index))
            remaining_weights.pop(index)
        return sampled

    def loadTrainingSource(
        self, entry: dict, *, max_chars: int = 150_000, retries: int = 5
    ) -> str:
        """Load one configured source, retrying through other canonical sources."""
        candidates = [entry]
        candidates.extend(
            candidate
            for candidate in self.sampleTrainingSources(
                len(trainingFilePath_dict_weighted or [])
            )
            if candidate != entry
        )
        seen_paths = set()
        attempts = 0
        for candidate in candidates:
            path = candidate.get("in") if isinstance(candidate, dict) else None
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            attempts += 1
            if attempts > max(1, retries):
                break
            source_type = str(candidate.get("type", "text") or "text").lower()
            strategy = (
                "random"
                if source_type in ("text", "discord_txt", "discord_json", "json")
                else "head"
            )
            text = self.loadSingleFile(
                path,
                source_type,
                max_chars=max_chars,
                strategy=strategy,
            )
            if text:
                return text.strip().lower()
        return ""

    def _reflow_to_lines(
        self, text: str, max_chars: int, max_line_len: int = 240, min_line_len: int = 20
    ) -> str:
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
        for generic JSON values. Keeps the internal buffer bounded while
        avoiding costly string copies on every element.
        """
        decoder = json.JSONDecoder()
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            buf = ""
            pos = 0
            in_array = False
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                if pos:
                    buf = buf[pos:]
                    pos = 0
                buf += chunk
                while True:
                    if not in_array:
                        idx = buf.find("[", pos)
                        if idx == -1:
                            # keep a small tail to catch '[' spanning chunks
                            buf = buf[-2:]
                            pos = 0
                            break
                        in_array = True
                        pos = idx + 1
                    # skip whitespace
                    length = len(buf)
                    while pos < length and buf[pos].isspace():
                        pos += 1
                    if pos >= length:
                        break
                    if buf[pos] == ",":
                        pos += 1
                        continue
                    if buf[pos] == "]":
                        return
                    try:
                        value, next_pos = decoder.raw_decode(buf, pos)
                    except json.JSONDecodeError:
                        # Need more data; trim buffer growth if necessary
                        if length - pos > 1_000_000:
                            buf = buf[-4096:]
                            pos = 0
                        break
                    yield value
                    pos = next_pos
                    if pos > 32768:
                        buf = buf[pos:]
                        pos = 0

    def _load_json_array_windowed(
        self, path: str, *, max_chars: int, strategy: str, soft: bool = False
    ) -> str | None:
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
                            out.append(ln[:max_chars])
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


    def _sequence_is_indices(self, seq) -> bool:
        try:
            sample = seq[0]
        except Exception:
            return False
        if isinstance(sample, (list, tuple)) and sample:
            sample = sample[0]
        return isinstance(sample, (int, np.integer))

    @staticmethod
    def prepare_training_message(text, *, source_type=None):
        if source_type == "email_parser":
            text = _EMAIL_THREAD.split(text, maxsplit=1)[0]
            text = "\n".join(line for line in text.split("\n") if not line.lstrip().startswith(">"))
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return to_british_english(clean_text(text, random_variations=False)).strip()

    def refresh_icharis2_training(self):
        """Request the canonical export periodically and adopt atomic changes."""
        if not curriculum.icharis2_user_text:
            return
        path = Path(curriculum.icharis2_pipeline_manifest_path).expanduser().absolute()
        now = monotonic()
        if now >= self._icharis2_publish_after:
            self._icharis2_publish_after = now + curriculum.icharis2_refresh_seconds
            try:
                subprocess.run(
                    [sys.executable, "-m", "icharis2.scripts.reports.export_babyllm_training_manifest",
                     "--output", str(path), "--limit", str(curriculum.icharis2_pipeline_limit)],
                    cwd=curriculum.icharis2_repository,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                    timeout=30,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                detail = getattr(exc, "stderr", None)
                reason = detail.strip().splitlines()[-1] if detail and detail.strip() else str(exc)
                print(f"[TRAINING][ICHARIS2] Export refresh failed: {reason}")
        try:
            stat = path.stat()
        except OSError:
            return
        version = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if version == self._icharis2_version:
            return
        snapshot = read_training_manifest(path, limit=curriculum.icharis2_pipeline_limit)
        if snapshot is None:
            return

        for key in ("items", "anniversary_items"):
            records = []
            seen = set()
            for record in snapshot[key]:
                if key == "items" and record.get("timestamp", "")[:10] > datetime.now(timezone.utc).date().isoformat():
                    continue
                text = self.prepare_training_message(record["text"], source_type=record.get("source_type"))
                if not text or text in seen:
                    continue
                seen.add(text)
                records.append({**record, "text": text})
            snapshot[key] = records
        self._icharis2_snapshot = snapshot
        self._icharis2_version = version
        self._icharis2_sampled.intersection_update(record["content_sha256"] for record in snapshot["items"])
        counts = defaultdict(int)
        for record in snapshot["items"]:
            counts[record.get("source_type", "unknown")] += 1
        print(f"[TRAINING][ICHARIS2] Adopted {snapshot['generated_at']}: {dict(counts)}")

    def has_fresh_icharis2_training(self, *, max_chars):
        records = self._icharis2_snapshot["items"] if self._icharis2_snapshot else []
        source_count = len({record.get("source_type") for record in records})
        budget = max_chars // max(1, source_count)
        return any(
            len(record["text"]) <= budget
            and record["content_sha256"] not in self._icharis2_sampled
            for record in records
        )

    def sample_archive_training_text(self, *, max_chars):
        """Sample the separately configured archive corpora for one lesson."""
        chunks = []
        remaining = max_chars
        for source in self.sampleTrainingSources(2):
            text = self.loadTrainingSource(source, max_chars=remaining)
            if text:
                chunks.append(text)
                remaining -= len(text) + 1
            if remaining <= 0:
                break
        return "\n".join(chunks)

    def sample_icharis2_training(self, *, max_chars):
        """Choose whole records across sources, prioritizing unsampled content."""
        snapshot = self._icharis2_snapshot
        if not snapshot or max_chars <= 0:
            return []
        by_source = defaultdict(list)
        for record in snapshot["items"]:
            by_source[record.get("source_type", "unknown")].append(record)
        groups = list(by_source.values())
        random.shuffle(groups)
        for records in groups:
            random.shuffle(records)
            records.sort(key=lambda record: record["content_sha256"] in self._icharis2_sampled)

        messages = []
        source_budgets = [max_chars // len(groups) for _ in groups]
        for group in itertools.zip_longest(*groups):
            for index, record in enumerate(group):
                if record is None:
                    continue
                text = record["text"]
                if len(text) > source_budgets[index]:
                    continue
                messages.append(text)
                self._icharis2_sampled.add(record["content_sha256"])
                source_budgets[index] -= len(text)
        return messages

    def due_icharis2_memories(self, now, *, limit):
        snapshot = self._icharis2_snapshot
        if not snapshot or limit <= 0:
            return []
        day = now.date().isoformat()
        if snapshot["anniversary_day"] != day:
            return []
        if self._icharis2_anniversary_day != day:
            self._icharis2_anniversary_day = day
            self._icharis2_anniversary_seen.clear()
        now_minute = now.hour * 60 + now.minute
        messages = []
        for record in snapshot["anniversary_items"]:
            key = record["content_sha256"]
            if key in self._icharis2_anniversary_seen:
                continue
            try:
                timestamp = datetime.fromisoformat(record["timestamp"]).astimezone(now.tzinfo)
            except (KeyError, TypeError, ValueError):
                continue
            minute = timestamp.hour * 60 + timestamp.minute
            if not 0 <= now_minute - minute <= 5:
                continue
            if len(record["text"]) > 5000:
                continue
            self._icharis2_anniversary_seen.add(key)
            messages.append(record["text"])
            if len(messages) >= limit:
                break
        return messages

    def tokenize_training_text(self, text, *, prepared=False, provenance=None):
        """Tokenize complete messages, optionally returning token-aligned origins.

        provenance contains one (is_self, feedback_signal_or_None) entry per
        source message. Feedback labels attach to the self-message ending token,
        so calibration uses Baby's own representation, not the human's reply.
        """
        explicit = isinstance(text, (list, tuple))
        messages = text if explicit else [text]
        eos_available = enable_train_append_eos and eos_replacement_token_str in self.tokenToIndex
        sos_available = enable_train_prepend_sos and sos_replacement_token_str in self.tokenToIndex
        if provenance is not None and (not explicit or len(provenance) != len(messages)):
            raise ValueError("Message provenance must match the explicit message list")
        tokens = []
        token_origins = [] if provenance is not None else None
        seen = set()
        for message_index, message in enumerate(messages):
            cleaned = message if prepared else self.prepare_training_message(str(message or ""))
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            append_eos = eos_available and (explicit or random.random() < eos_append_probability)
            message_tokens = self.tokenize_training_message(cleaned, append_eos=append_eos)
            if not message_tokens:
                continue
            if tokens and explicit and not eos_available:
                newline = self.tokenizeText("\n")
                tokens.extend(newline)
                if token_origins is not None:
                    token_origins.extend([(False, None)] * len(newline))
            if sos_available and random.random() < sos_prepend_probability:
                tokens.append(sos_replacement_token_str)
                if token_origins is not None:
                    token_origins.append((bool(provenance[message_index][0]), None))
            tokens.extend(message_tokens)
            if token_origins is not None:
                is_self, feedback = provenance[message_index]
                token_origins.extend([(bool(is_self), None)] * len(message_tokens))
                if is_self and feedback is not None:
                    token_origins[-1] = (True, int(feedback))
        return (tokens, token_origins) if token_origins is not None else tokens

    def tokenize_training_message(self, text, *, append_eos=True):
        """Tokenize one complete message, preserving its internal BPE newlines.

        The reserved EOS token belongs at the message boundary, never at a
        paragraph break. Callers handling non-message corpora can opt out.
        """
        tokens = self.tokenizeText(text) if isinstance(text, str) else list(text)
        if not tokens:
            return []
        if append_eos:
            from config import eos_replacement_token_str

            if (
                eos_replacement_token_str
                and eos_replacement_token_str in self.tokenToIndex
                and tokens[-1] != eos_replacement_token_str
            ):
                tokens.append(eos_replacement_token_str)
        return tokens

    def add_training_text(self, text):
        """Append a bounded token window; each string represents one message."""
        try:
            if isinstance(text, str):
                toks = self.tokenize_training_text([text])
            else:
                # Explicit token sequences have their own boundary policy.
                toks = list(text)
            if not toks:
                return 0
            self._dynamic_tokens.extend(toks)
            added = len(toks)
            if self._training_token_count_estimate is not None:
                self._training_token_count_estimate += added
            return added
        except Exception:
            return 0

    def genTrainingData(
        self,
        _windowMAX=numTokensPerStepSTART,
        _startIndex=trainingStartIndex,
        _trainingDataPairNumber=1,
        _stride=trainingDataStride,
        _tokens=None,
        _token_provenance=None,
    ):
        if _token_provenance is not None:
            if _tokens is None or len(_tokens) != len(_token_provenance):
                raise ValueError("Training-token provenance must match explicit tokens")
        with self.v_counsellor.infodump("genTrainingData") as ʕっʘ‿ʘʔっ:
            if isinstance(_windowMAX, torch.Tensor):
                _windowMAX = int(_windowMAX.item())
            window = max(1, int(_windowMAX))
            stride = max(1, int(_stride) if _stride else 1)
            limit = (
                int(_trainingDataPairNumber)
                if _trainingDataPairNumber and _trainingDataPairNumber > 0
                else None
            )

            tokens_are_indices = getattr(self, "_tokens_are_indices", False)
            description = "tokens"

            if _tokens is not None:
                base_tokens = _tokens
                tokens_are_indices = tokens_are_indices or self._sequence_is_indices(
                    base_tokens
                )

                def base_iter_factory() -> Iterator:
                    return iter(base_tokens)

                try:
                    length_estimate = len(base_tokens)  # type: ignore[arg-type]
                except Exception:
                    length_estimate = None
                dynamic_snapshot: list = []
                description = "custom"
            elif self._token_stream_factories:
                factories = list(self._token_stream_factories)

                def base_iter_factory() -> Iterator:
                    def chained() -> Iterator:
                        for factory in factories:
                            yield from factory()

                    return chained()

                length_estimate = self.training_token_count_estimate()
                tokens_are_indices = True
                dynamic_snapshot = list(self._dynamic_tokens)
                description = "stream"
            else:
                base_tokens = self.tokens if self.tokens is not None else []
                tokens_are_indices = tokens_are_indices or self._sequence_is_indices(
                    base_tokens
                )

                def base_iter_factory() -> Iterator:
                    return iter(base_tokens)

                try:
                    length_estimate = len(base_tokens)  # type: ignore[arg-type]
                except Exception:
                    length_estimate = None
                dynamic_snapshot = list(self._dynamic_tokens)

            if _tokens is not None:
                dynamic_snapshot = []

            start_index = _startIndex
            if isinstance(start_index, torch.Tensor):
                start_index = int(start_index.item())
            if start_index == "random":
                estimate = (
                    length_estimate
                    if length_estimate is not None
                    else self.training_token_count_estimate()
                )
                if estimate and estimate > window * 2:
                    start_index = random.randint(0, max(0, estimate - window * 2))
                else:
                    start_index = 0
            else:
                try:
                    start_index = int(start_index)
                except Exception:
                    start_index = 0

            token_lookup = self.tokenToIndex

            def token_iterator_factory() -> Iterator:
                def iterator() -> Iterator:
                    for val in base_iter_factory():
                        yield val
                    if dynamic_snapshot:
                        for val in dynamic_snapshot:
                            yield val

                return iterator()

            pair_factory = self._make_pair_iterator_factory(
                token_iterator_factory,
                window=window,
                stride=stride,
                limit=limit,
                tokens_are_indices=tokens_are_indices,
                token_lookup=token_lookup,
                start=start_index,
                token_provenance=_token_provenance,
            )

            estimated_pairs = self._estimate_pair_count(
                length_estimate,
                dynamic_snapshot,
                window=window,
                stride=stride,
                limit=limit,
                start=start_index,
            )

            return _TrainingPairStream(
                pair_factory, estimated_pairs, description=description
            )



    def saveVocab(self):
        with self.v_counsellor.infodump("saveVocab") as ʕっʘ‿ʘʔっ:
            os.makedirs(self.vocabCache, exist_ok=True)  # Ensure directory exists
            with open(self.vocabListFile, "w", encoding="utf-8") as f:
                if debugPrints:
                    ʕっʘ‿ʘʔっ("save vocabList")
                json.dump(self.vocabList, f, indent=4)
            with open(self.tokenToIndexFile, "w", encoding="utf-8") as f:
                if debugPrints:
                    ʕっʘ‿ʘʔっ("save tokenToIndex")
                json.dump(self.tokenToIndex, f, indent=4)
            with open(self.indexToTokenFile, "w", encoding="utf-8") as f:
                if debugPrints:
                    ʕっʘ‿ʘʔっ("save indexToToken")
                json.dump(self.indexToToken, f, indent=4)

    def loadVocab(self):
        with self.v_counsellor.infodump("loadVocab") as ʕっʘ‿ʘʔっ:
            try:
                with open(self.vocabListFile, "r", encoding="utf-8") as f:
                    if debugPrints:
                        ʕっʘ‿ʘʔっ("load vocabList")
                    self.vocabList = json.load(f)
                with open(self.tokenToIndexFile, "r", encoding="utf-8") as f:
                    if debugPrints:
                        ʕっʘ‿ʘʔっ("load tokenToIndex")
                    self.tokenToIndex = json.load(f)
                with open(self.indexToTokenFile, "r", encoding="utf-8") as f:
                    if debugPrints:
                        ʕっʘ‿ʘʔっ("load indexToToken")
                    self.indexToToken = {
                        int(k): v for k, v in json.load(f).items()
                    }  # ensures that keys are integers!
                print("vocab files loaded successfully!")
                return bool(self.vocabList and self.tokenToIndex and self.indexToToken)
            except (FileNotFoundError, json.JSONDecodeError):
                print("vocab files not found or invalid... rebuilding vocab...")
                return False


# __main__ test harness removed (legacy)
