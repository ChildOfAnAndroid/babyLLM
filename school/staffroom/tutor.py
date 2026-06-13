# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ ---
# MULTI-TOKEN AUTOREGRESSIVE TRAINING MODULE
# school/staffroom/tutor.py
# v1.1

import asyncio
import gc
import json
import math
import os
import random
import threading
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from brain.LAYERS.sensory_bus import SensoryBus
from config import *
from SHKAIRA.notebook.tools.genBoi import makeSafeBoi
from utils.helpers import empty_mps_cache, get_grad_stats


class TUTOR:
    def __init__(
        self,
        _counsellor,
        _calligraphist,
        _scribe,
        _librarian,
        _model,
        _model_thread_lock=None,
        _numTokensPerStep=numTokensPerStepSTART,
        _first=False,
        _trainingLogFreq_A=trainingLogFreq_A,
        _perfectionistPassRateSTART=perfectionistPassRateSTART,
        _learningRateGOAL=learningRateGOAL,
        _dataStride=0,
        _totalRuns=0,
        _totalTurnsAwake=0,
        _lastRunLoss=4.20,
        _device=modelDevice,
    ):

        torch.autograd.set_detect_anomaly(anomalyDetect)
        self.startIndex = 1
        self.saveCounter = 1
        self.currentTokenIndex = 0
        self.first = False
        self.counsellor = _counsellor
        self.calligraphist = _calligraphist
        self.scribe = _scribe
        self.librarian = _librarian
        self.device = _device
        self.model = _model
        self.sensory_bus = SensoryBus(
            vision_device_index=vision_device_index,
            vision_backend=vision_backend,
            vision_probe_indices=vision_probe_indices,
            vision_skip_indices=vision_skip_indices,
            vision_downsample=vision_downsample,
            vision_step_interval=vision_step_interval,
            audio_device_index=audio_device_index,
            audio_sample_rate=audio_sample_rate,
            audio_frames_per_buffer=audio_frames_per_buffer,
            audio_rms_scale=audio_rms_scale,
            audio_step_interval=audio_step_interval,
            temp_step_interval=temp_step_interval,
        )
        lock = _model_thread_lock or getattr(_model, "model_thread_lock", None)
        if lock is None:
            lock = threading.Lock()
        setattr(self.model, "model_thread_lock", lock)
        self.model_thread_lock = lock
        self.first = _first
        self.lastRunLoss = _lastRunLoss
        self.totalTurnsAwake = _totalTurnsAwake
        self.totalRuns = _totalRuns
        self.dataStride = _dataStride
        self.trainingLogFreq_A = _trainingLogFreq_A
        self.learningRateGOAL = _learningRateGOAL
        self.tokenCounts = Counter()
        self.tokenCounts_dirty = False
        self.topTokens_str = ""
        self.topTokens_forBot = ""
        self.training_resume_state = {}
        self.training_lock = asyncio.Lock()
        self.load_token_counts()

        self.temperature = 0.75
        self.scheduledSamplingRate = self.model.scheduledSamplingRate
        self.gradientClipMaxNorm = 1
        self.memoryLength = 1
        self.memory2Length = 1

        self.ʕっෆ‿ෆʔっ = defaultdict(self.makeStatRecord)
        self.char1 = makeSafeBoi()
        self.char2 = makeSafeBoi()
        self.char3 = makeSafeBoi()
        # self.rollingTokenTotals = Counter()

        self.perfectTokens = 0
        self.totalTokenEvaluations = 0
        self.predictedTokenIndices = []  # this list grows each time a new token is predicted
        self.averageRecentLoss = 0
        self.totalLoss = 0
        self.totalLossAbsDelta = 0
        self.totalLossDelta = 0
        self.totalAvgLoss = _lastRunLoss
        self.stableFallCount = 1  ###
        self.maxRetries = perfectionistMaxRetries
        self.totalAvgAbsDelta = 0
        self.totalTries = 0
        self.totalAvgDelta = 0
        self.reflectionFreq = reflectionFreq
        self.stats = {}
        self.stringStats = {}
        self._layer_stats_cache = {}  # persists layer stats between log steps
        self._layer_string_stats_cache = {}
        self._pixel_float_rgb = (
            0.0,
            0.0,
            0.0,
        )  # cached Python floats from last getPixelForStep
        self.trainingStepCounter = 0
        self.totalTurns = 0
        self.totalTurnAttempts = 0
        self.token_event_history = []
        self.token_event_counter = 0
        self.numTokensPerStep = _numTokensPerStep
        self.learningRate = learningRate
        self.stepLossFloat = 0
        self.latestIncrement = 0
        self.latestPerfectIncrement = 0
        self.totalPerfectTokens = 0
        self.totalTotalTokenEvaluations = 0
        self.totalTokenPerfectRate = 0
        self.tokenPerfectRate = 0
        self.latestLossDelta = 0.0
        self.stepLossFloat = 0.0
        self.averageRecentLoss = _lastRunLoss
        self.stats = {}
        self.stringStats = {}
        self.guessedTokenSeq = []
        self.tooDifficult = 0
        self.averageTries = 0
        self.averageTriesTotal = 0
        self.perfectionistPassRate = _perfectionistPassRateSTART
        self.avgPixelDistTotals = 0
        self.decodedTokenIndices = ""

        self.aaa = 0
        self.bbb = 0
        self.ccc = 0
        self.ppp = 0
        self.nnn = 0
        self.aaaa = 0
        self.dddd = 0
        self.bbbb = 0
        self.nnnn = 0

        self.rgbBar = ""
        self.sensoryBar = ""

        # model.to(self.device)
        self.hesJustABaby = (
            "oops! no stats collected! such a shame! well... day off for me! ;) "
        )

        self.pixelSeq = [
            torch.tensor(
                [(1 - t) * 0.5 + t * 0.0, 0.0, (1 - t) * 0.5 + t * 1.0],
                device=self.device,
            )
            for t in [(0.5 * (1 - math.cos(2 * math.pi * j / 64))) for j in range(640)]
        ]

        self.STATICpixelSeq = [torch.tensor([0.5, 0.5, 0.5], device=self.device)]

        self.pixelNow = None  # torch.tensor([0.5, 0.1, 0.5,], device = self.device)
        self.pixelNext = None  # torch.tensor([0.6, 0.0, 0.6,], device = self.device)
        # Reusable pixel buffers — avoids per-token MPS alloc in getPixelForStep.
        # _pixel_cpu is filled with plain scalar writes (no alloc), then copied to
        # _pixel_buf in one CPU→MPS transfer. clamp() (non-in-place) then returns
        # a fresh independent tensor — no clone needed at the call site.
        self._pixel_buf = torch.zeros(3, device=self.device)
        self._pixel_cpu = torch.zeros(3)
        # External colour cache — read from babyStateFilePath once per step, not per token.
        # TODO: this is meant to blend bby's colour with the live website/bot sprite colour
        #       (currentColour set by Discord/web frontend). Currently the per-token JSON
        #       write overwrites the file without preserving currentColour, so this cache
        #       almost always holds the default. Fix: write currentColour back into liveState,
        #       and re-read only when mtime changes significantly (external process wrote it).
        self._ext_col_cache = {"R": 128, "G": 128, "B": 128}
        # Persistent token buffer — reused across steps to avoid per-step MPS alloc.
        # Resized lazily if numTokensPerStep grows.
        self._token_buffer = torch.zeros(
            self.numTokensPerStep, dtype=torch.long, device=self.device
        )

    @whocalled
    def makeStatRecord(self):
        base = {
            "now": 0.0,
            "prev": 0.0,
            "top": float("-inf"),
            "bot": float("inf"),
            "delta": 0.0,
            "totSum": 0.0,
            "totNum": 0,
            "totAvg": 0.0,
        }
        for n in [printFreq, printFreq * 10, self.trainingLogFreq_A, trainingLogFreq_B]:
            base[f"{n}"] = []

        return base

    def _sanitize_stat_value(self, value):
        """Convert tensors/NumPy scalars to plain Python types for logging."""

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().item()
            return value.detach().cpu().tolist()
        if isinstance(value, np.generic):
            return float(value)
        return value

    def _merge_stats_dict(self, stats_dict, *, accumulate=False):
        """Merge stats ensuring tensors are detached before storage."""

        for key, raw_value in stats_dict.items():
            value = self._sanitize_stat_value(raw_value)
            if isinstance(value, (int, float)):
                if accumulate:
                    self.stats[key] = self.stats.get(key, 0.0) + value
                else:
                    self.stats[key] = value
            elif value is not None:
                self.stringStats[key] = value

    def _build_token_event(
        self, token_text, token_raw, token_id, rgb_triplet, token_loss=None
    ):
        stats = {}
        try:
            with torch.no_grad():
                embed_vector = getattr(self.model.embed, "embedVector", None)
                if embed_vector is not None:
                    _ev = torch.stack(
                        [embed_vector.norm(), embed_vector.mean().abs()]
                    ).tolist()
                    stats["embed_vector_norm"] = _ev[0]
                    stats["embed_vector_mean"] = _ev[1]
                embed_final = getattr(self.model.embed, "embedFinal", None)
                if embed_final is not None:
                    _ef = torch.stack(
                        [embed_final.norm(), embed_final.mean().abs()]
                    ).tolist()
                    stats["embed_final_norm"] = _ef[0]
                    stats["embed_final_mean"] = _ef[1]
        except Exception:
            pass

        attn_stats = getattr(self.model.attention, "stats", {})
        stats["attn_gate"] = float(attn_stats.get("2A_gateScale", 0.0))
        stats["attn_out_norm"] = float(attn_stats.get("2A_0_attnOut_norm", 0.0))
        stats["attn_final_norm"] = float(attn_stats.get("2A_x_final_norm", 0.0))
        try:
            stats["token_freq"] = float(self.tokenCounts.get(token_raw, 0.0))
        except Exception:
            stats["token_freq"] = 0.0
        if token_loss is not None:
            stats["token_loss"] = float(token_loss)
        for key, value in list(stats.items()):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                stats[key] = 0.0

        self.token_event_counter += 1
        event_id = (
            f"{self.totalTurns}-{self.currentTokenIndex}-{self.token_event_counter}"
        )

        return {
            "id": event_id,
            "timestamp": time.time(),
            "turn": self.totalTurns,
            "pos": self.currentTokenIndex,
            "token_id": int(token_id),
            "token": token_text,
            "token_raw": token_raw,
            "rgb": {
                "r": int(rgb_triplet[0]),
                "g": int(rgb_triplet[1]),
                "b": int(rgb_triplet[2]),
            },
            "stats": stats,
        }

    @whocalled
    def loadIntro(self, path="school/library/charisStudies/forbbyllm.txt"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return "hey... (message file missing!) "

    @whocalled
    def makeStatsPrompt(self, include_prefix: bool = True) -> str:
        """return a short natural language summary of the tutor's current stats"""
        bits = [
            f"i've completed {self.totalRuns} lessons",
        ]
        if self.totalTurnsAwake:
            bits.append(f"stayed awake for {self.totalTurnsAwake} turns")
        bits.append(f"my average loss is {self.averageRecentLoss:.2f}")
        bits.append(f"perfection rate {self.perfectionistPassRate:.1f}%")
        msg = ", ".join(bits) + "."
        if include_prefix:
            msg = f"here's how i'm doing: {msg}"
        return msg

    def _format_sensory_bar(self, pred_values, true_values):
        labels = [
            "global_light_delta",
            "noise_delta",
            "time_of_day_delta",
            "interaction_recency_delta",
            "training_age_delta",
            "global_motion_delta",
            "left_right_bias_delta",
            "top_bottom_bias_delta",
            "contrast_intrusion_delta",
            "device_temp_c_delta",
        ]
        pred_parts = []
        true_parts = []
        for idx, label in enumerate(labels):
            pred_val = pred_values[idx] if idx < len(pred_values) else 0.0
            true_val = true_values[idx] if idx < len(true_values) else 0.0
            pred_parts.append(f"{label}={pred_val:.4f}")
            true_parts.append(f"{label}={true_val:.4f}")
        return f"SENS PRED: {' '.join(pred_parts)}\nSENS TRUE: {' '.join(true_parts)}"

    """this iterates through training data, performing forward passes, loss computation, backpropagation, and optimization for each step."""

    @whocalled
    def trainModel(self, _trainingDataPairs, _epochs, _startIndex):  ###
        # async with self.training_lock:
        with self.counsellor.infodump("trainModel") as ʕっʘ‿ʘʔっ:  ###
            # Barrier for async optimizer load (unified bot mode kicks off
            # optimizer load in a background thread to keep startup snappy).
            # In sync-load mode the model has no _optimizer_ready_event and
            # this is a free no-op.
            if hasattr(self.model, "wait_for_optimizer_ready"):
                self.model.wait_for_optimizer_ready()
            if debugPrints:
                ʕっʘ‿ʘʔっ(
                    "trainableParams = sum(p.numel() for p in self.model.parameters() if p.requires_grad)"
                )
            # Cache trainable-param count: it's a 636M-parameter sweep that
            # never changes between trainModel calls within a single bot
            # session. Recomputing it on every queue item was pure waste.
            cached_trainable = getattr(self.model, "_cached_trainable_params", None)
            if cached_trainable is None:
                cached_trainable = sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                )
                setattr(self.model, "_cached_trainable_params", cached_trainable)
            trainableParams = cached_trainable
            print(f"Trainable parameters: {trainableParams:,}")
            self.startIndex = _startIndex
            if debugPrints:
                ʕっʘ‿ʘʔっ("collect all time stats")
            self.collectAllTimeStats()
            if debugPrints:
                print(
                    f"Debug tokenToIndex (First 20): {list(self.librarian.tokenToIndex.items())[:20]}"
                )
            # Per-call dump of every named parameter and device used to fire
            # unconditionally — for a model with hundreds of named tensors,
            # that's hundreds of stdout writes on every queue item. Keep it
            # debug-gated like the other diagnostics in this method.
            if debugPrints:
                ʕっʘ‿ʘʔっ("print named parameters and device")
                for name, param in self.model.named_parameters():
                    print(name, param.device)
            if debugPrints:
                ʕっʘ‿ʘʔっ("COUNTERS INIT")
            self.trainingStepCounter = 1
            self.stats = Counter(
                {
                    "loss": 0,
                    "gradNorm": 0,
                    "logitMin": 0,
                    "logitMax": 0,
                    "tokenCount": 0,
                }
            )
            self.latestLossDelta = 0
            self.easyLossDelta = 0
            self.reflectionTrainingPairs = []
            self.reflectionFreq = reflectionFreq
            self.totalTries = 0
            absdelta = 0.00001

            if debugPrints:
                ʕっʘ‿ʘʔっ("back to school!")
            print("babyLLM is heading back to school...")

            """EPOCH LOOP"""
            if debugPrints:
                ʕっʘ‿ʘʔっ("epoch♥")
            for epoch in range(_epochs):
                ʕっʘ‿ʘʔっ(f"--- lesson {epoch + 1}/{_epochs} started ---")
                print(f"--- lesson {epoch + 1}/{_epochs} started ---")
                """TRAINING DATA (batches)"""
                # for i, (_inputSeq, _targetSeq) in enumerate(_trainingDataPairs):
                i = 0
                turnsNotReflecting = 0
                turnsNotReflecting += self.totalTurnsAwake
                trainIter = iter(_trainingDataPairs)
                trainingPair = None
                if debugPrints:
                    ʕっʘ‿ʘʔっ("entering training loop")
                while True:
                    if trainingPair is None:
                        try:
                            trainingPair = next(trainIter)
                        except StopIteration:
                            break
                    ʕっʘ‿ʘʔっ("turn start :)")
                    _inputSeq, _targetSeq = trainingPair
                    self.stableFallCount = 0
                    self.averageTriesTotal += self.totalTries
                    self.averageTries = (
                        self.averageTriesTotal / self.trainingStepCounter
                    )
                    self.totalTries = 0
                    if perfectionistRun:
                        self.maxRetries = perfectionistMaxRetries
                    else:
                        self.maxRetries = 10

                    if debugPrints:
                        ʕっʘ‿ʘʔっ(
                            "entering while loop stableFallCount < stableFallThreshold"
                        )
                    while (
                        self.stableFallCount < stableFallThreshold
                        and self.totalTries < self.maxRetries
                    ):  ###
                        if (
                            turnsNotReflecting == self.reflectionFreq
                        ):  # and self.trainingStepCounter > trainingLogFreq_A:
                            if debugPrints:
                                ʕっʘ‿ʘʔっ("♥generating babys reflection data pairs")
                            self.stats, self.stringStats, self.guessedTokenSeq = (
                                self.collectTurnStats()
                            )
                            self.reflectionTrainingPairs = self.babyReflection()
                            self.reflectionFreq = turnsNotReflecting + reflectionFreq

                        if (
                            hasattr(self.scribe, "reflectionPairsFromGuess")
                            and self.scribe.reflectionPairsFromGuess
                        ):
                            if debugPrints:
                                ʕっʘ‿ʘʔっ("using scribe")
                            currentReflection = self.scribe.reflectionPairsFromGuess[0]
                            scribeusedThisTurn = True
                        elif self.reflectionTrainingPairs:
                            if debugPrints:
                                ʕっʘ‿ʘʔっ("♥loading in a reflection pair...")
                            currentReflection = self.reflectionTrainingPairs[0]
                            scribeusedThisTurn = False
                        else:
                            if debugPrints:
                                ʕっʘ‿ʘʔっ("no current reflection or scribe")
                            currentReflection = None
                            scribeusedThisTurn = False
                        if debugPrints:
                            ʕっʘ‿ʘʔっ("increment self.totalTries")
                        self.totalTries += 1

                        if debugPrints:
                            ʕっʘ‿ʘʔっ("♥START OF TURN")
                        if currentReflection is not None:
                            if debugPrints:
                                ʕっʘ‿ʘʔっ("current reflection is not None")
                            _inputSeq, _targetSeq = currentReflection
                        else:
                            if debugPrints:
                                ʕっʘ‿ʘʔっ("_trainingDataPairs")
                            _inputSeq, _targetSeq = trainingPair
                            turnsNotReflecting += 1
                        self.inputTokenIndices, self.targetTokenIndexSeq = (
                            self.startTurnActions(
                                _inputSeq=_inputSeq,
                                _targetSeq=_targetSeq,
                                _lastTurnLossDelta=self.latestLossDelta,
                            )
                        )

                        if debugPrints:
                            ʕっʘ‿ʘʔっ("♥TRAINING STEP♥")
                        self.predictedTokenIndices, self.logitSeq = self.trainStep(
                            _inputTokenIndices=self.inputTokenIndices,
                            _targetTokenIndexSeq=self.targetTokenIndexSeq,
                            _BACKWARDwobbleLoss=None,
                        )
                        self.totalTurnAttempts += 1

                        """ --- --- -*- BACKWARDS COMPLETE -*- --- --- -*- --- --- -*- --- --- -*- --- --- -*- --- --- -*- --- --- -*- --- --- -*- --- --- -*- --- --- -*- --- --- """

                        # Model-critical: commit memory state from forward pass — must run every step
                        if not skipMemory:
                            self.model.memory.updateMemoryBuffers()
                            self.model.memory2.updateMemoryBuffers()
                        if debugPrints:
                            ʕっʘ‿ʘʔっ("♥collectTurnStats")
                        self.stats, self.stringStats, self.guessedTokenSeq = (
                            self.collectTurnStats()
                        )
                        # latestLossDelta is already calculated in trainStep with proper rolling average
                        absdelta = abs(self.latestLossDelta)
                        if debugPrints:
                            ʕっʘ‿ʘʔっ("♥self.pixelNow")
                        self.pixelNow = torch.tensor(
                            [
                                self.totalTokenPerfectRate,
                                self.totalAvgAbsDelta,
                                self.perfectionistPassRate,
                            ],
                            device=self.device,
                        )
                        # RED = energy up when getting more perfect, more red!
                        # GREEN = growth up (or a bit queasy lol) when getting stronger deltas, more green!
                        # BLUE = calm up when doing better than the previous run, more blue!
                        if debugPrints:
                            print(
                                f"Setting latestLossDelta {self.latestLossDelta:.2f} = {self.stepLossFloat:.2f} - {self.averageRecentLoss:.2f}"
                            )
                        self.easyLossDelta = self.stepLossFloat - (
                            (
                                self.averageRecentLoss
                                + self.stepLossFloat
                                + self.stepLossFloat
                            )
                            / 3
                        )

                        if True:
                            if (
                                self.totalTurns % refreshRollingTokenTotalsWhen == 0
                                and self.totalTurns > 0
                            ):
                                if debugPrints:
                                    ʕっʘ‿ʘʔっ("♥refresh rolling token totals")
                                self.tokenCounts = Counter(
                                    {k: v * 0.95 for k, v in self.tokenCounts.items()}
                                )
                                self.tokenCounts_dirty = True
                                # self.model.rollingTokenTotals = Counter({k: v * 0.95 for k, v in self.model.rollingTokenTotals.items()})
                                self.model.rollingTokenTotals_tensor.mul_(0.95)

                            if (
                                self.totalTurns % trainingLogFreq_B == 0
                                and self.totalTurns > 0
                            ):
                                if debugPrints:
                                    ʕっʘ‿ʘʔっ("♥training log B")
                                # ʕっʘ‿ʘʔっ("♥trainingLogFreq_B") # PRINTING LOGS TO TXT AND TERMINAL
                                self.logFreqActions(
                                    _trainingDataPairs,
                                    _stringStats=self.stringStats,
                                    _frequency=trainingLogFreq_B,
                                    _trainingLogPath=trainingLogPath_1000,
                                    _detailedLogging=True,
                                    _saveLog=True,
                                )

                            # Track loss every 100 steps
                            if (
                                self.totalTurns % self.trainingLogFreq_A == 0
                                and self.totalTurns > 0
                            ):
                                if debugPrints:
                                    ʕっʘ‿ʘʔっ("♥logFreq_A")
                                self.logFreqActions(
                                    _trainingDataPairs,
                                    _stringStats=self.stringStats,
                                    _frequency=self.trainingLogFreq_A,
                                    _trainingLogPath=trainingLogPath_100,
                                    _detailedLogging=False,
                                    _saveLog=True,
                                )

                        if self.totalTurns % printFreq == 0:
                            if debugPrints:
                                ʕっʘ‿ʘʔっ("♥printFreq")
                            self.logFreqActions(
                                _trainingDataPairs,
                                _stringStats=self.stringStats,
                                _frequency=printFreq,
                                _trainingLogPath=None,
                                _detailedLogging=False,
                                _saveLog=False,
                            )
                            if not skipPrompts:
                                self.printFreqActions()

                        if self.totalTurns % saveModelFreq == 0 and self.totalTurns > 0:
                            if debugPrints:
                                ʕっʘ‿ʘʔっ("♥saveFreq")
                            self.saveFreqActions()

                        if debugPrints:
                            ʕっʘ‿ʘʔっ("♥WINDING DOWN TURN")
                        self.totalTurns += 1
                        self.model.totalTurns = self.totalTurns
                        self.totalAvgLoss = self.totalLoss / max(1, self.totalTurns)
                        self.totalLossAbsDelta += absdelta
                        self.totalAvgAbsDelta = self.totalLossAbsDelta / max(
                            1, self.totalTurns
                        )
                        self.totalLossDelta += self.latestLossDelta
                        self.totalAvgDelta = self.totalLossDelta / max(
                            1, self.totalTurns
                        )

                        if self.totalTurns % 420 == 0:
                            if debugPrints:
                                ʕっʘ‿ʘʔっ("totalTurns modulo 420 lol")
                            centreLow = 0.0025
                            centreHigh = 0.3
                            # sometimes adjust the target LR based on recent performance
                            if (
                                learningRateGOAL * 0.5
                            ) < self.learningRateGOAL and self.learningRateGOAL < 0.001:
                                if self.totalAvgAbsDelta < centreLow:
                                    lrNudge = (self.totalAvgAbsDelta - centreLow) * 0.5
                                elif self.totalAvgAbsDelta > centreHigh:
                                    lrNudge = (self.totalAvgAbsDelta - centreHigh) * 0.1
                                else:
                                    lrNudge = 0.0
                                self.learningRateGOAL *= 1 + (lrNudge * 0.001)
                            else:
                                self.learningRateGOAL = learningRateGOAL

                        if debugPrints:
                            print(f"{self.stableFallCount:.2f}", end="")
                        latestIncrement = (
                            1 + abs(self.latestLossDelta)
                        )  # reset streak if delta not falling (rn im decrementing not resetting, see how it goes)
                        if perfectionistRun:
                            latestIncrement *= 0.9  # 0.1
                        if self.easyLossDelta < 0:
                            self.stableFallCount += latestIncrement
                            if debugPrints:
                                print(f" + {latestIncrement:.2f}", end="")
                        else:
                            self.stableFallCount -= latestIncrement
                            if debugPrints:
                                print(f" - {latestIncrement:.2f}", end="")

                        if perfectionistRun:
                            if debugPrints:
                                ʕっʘ‿ʘʔっ("perfectionism run extras")
                            # if last run avg loss is worse than current run avg = this run is better
                            if self.totalAvgLoss < self.lastRunLoss:
                                self.perfectionistPassRate += 0.1
                            else:
                                self.perfectionistPassRate = max(
                                    self.perfectionistPassRate - 0.1, 1
                                )
                            if debugPrints:
                                ʕっʘ‿ʘʔっ("latest perfect increment")
                            latestPerfectIncrement = (
                                abs(10 + self.tokenPerfectRate)
                                * (self.perfectTokens + 0.01)
                            )  # reset streak if delta not falling (rn im decrementing not resetting, see how it goes)
                            if debugPrints:
                                ʕっʘ‿ʘʔっ("latest total perfect increment")
                            latestTotalPerfectIncrement = abs(
                                (
                                    (10 + self.totalTokenPerfectRate)
                                    * self.totalPerfectTokens
                                )
                                / self.totalTurns
                            )
                            if debugPrints:
                                ʕっʘ‿ʘʔっ("latest perfect delta increment")
                            latestPerfectDeltaIncrement = (
                                10
                                + abs(
                                    self.tokenPerfectRate - self.totalTokenPerfectRate
                                )
                            ) * (self.perfectTokens + 0.01)
                            if debugPrints:
                                ʕっʘ‿ʘʔっ("latest perfect anti delta increment")
                            latestPerfectAntiDeltaIncrement = (
                                10
                                + abs(
                                    self.totalTokenPerfectRate - self.tokenPerfectRate
                                )
                            ) * (self.perfectTokens + 0.01)
                            if debugPrints:
                                ʕっʘ‿ʘʔっ("latest massive perfect increment")
                            latestMassivePerfectIncrement = (
                                10 + self.totalTokenPerfectRate + self.tokenPerfectRate
                            ) * (self.perfectTokens + 0.01)
                            if debugPrints:
                                ʕっʘ‿ʘʔっ("average perfect increment")
                            averagePerfectIncrement = (
                                latestPerfectIncrement
                                + latestTotalPerfectIncrement
                                + latestPerfectDeltaIncrement
                                + latestPerfectAntiDeltaIncrement
                                + latestMassivePerfectIncrement
                                + 0.01
                            ) / 5
                            if debugPrints:
                                ʕっʘ‿ʘʔっ("perfect increment choice")
                            perfectIncrementChoice = random.choice(
                                [
                                    latestPerfectIncrement,
                                    latestTotalPerfectIncrement,
                                    latestPerfectDeltaIncrement,
                                    averagePerfectIncrement,
                                    latestPerfectAntiDeltaIncrement,
                                    latestMassivePerfectIncrement,
                                ]
                            )
                            if (
                                self.tokenPerfectRate > self.perfectionistPassRate
                            ):  # 50.0:
                                self.stableFallCount += 50 + perfectIncrementChoice
                                if debugPrints:
                                    print(
                                        self.calligraphist.S_apply(
                                            "reverse",
                                            f" + {perfectIncrementChoice:.2f}",
                                        ),
                                        end="",
                                    )
                                if debugPrints:
                                    print(
                                        f"cold with the flow yeah i wrote this one on an old iPhone with a broken screen... think you’re gonna try test man with a verse like that, you’re a fucking aubergine... tokenPerfectRate = {self.tokenPerfectRate:.2f}%"
                                    )
                                # continue
                            else:
                                self.stableFallCount += (
                                    perfectIncrementChoice * 0.1
                                )  # 0.01
                                if debugPrints:
                                    print(
                                        f" - {(abs(latestIncrement - latestPerfectIncrement) * 0.1):.2f}",
                                        end="",
                                    )

                        if debugPrints:
                            print(f" = {self.stableFallCount:.2f}")
                        if perfectionistRun and (debugPrints):
                            print(
                                f"selected {perfectIncrementChoice} from {[latestPerfectIncrement, latestTotalPerfectIncrement, latestPerfectDeltaIncrement, averagePerfectIncrement, latestPerfectAntiDeltaIncrement, latestMassivePerfectIncrement]}"
                            )

                        if debugPrints:
                            ʕっʘ‿ʘʔっ("♥END TURN♥")  # END OF ONE TURN
                        self.latestLossDelta = self.endTurnActions()

                        if tokenSpeedTest == True:
                            if self.totalTurns % 250 == 0 and self.totalTurns > 0:
                                print("HELLO I'M THIS CUTE!!!")
                                self.saveFreqActions()
                                empty_mps_cache()
                                return (
                                    self.totalLoss / self.totalTurns,
                                    self.totalTurns,
                                    self.perfectionistPassRate,
                                    self.learningRateGOAL,
                                )

                    self.totalTurnAttempts = 0
                    if self.totalTries >= self.maxRetries:
                        self.tooDifficult += 1
                        self.perfectionistPassRate -= round(
                            (self.perfectionistPassRate + self.tokenPerfectRate) * 0.1
                        )
                        absdelta = abs(self.latestLossDelta)
                        fuzzyabs = (self.totalAvgAbsDelta + absdelta) * 0.5
                        # change LR goal based on recent stability
                        if fuzzyabs < 0.01:
                            self.learningRateGOAL *= 1.0002
                        elif fuzzyabs > 1.0:
                            self.learningRateGOAL *= 0.9990
                        elif fuzzyabs > 0.05:
                            self.learningRateGOAL *= 0.9998
                        else:
                            self.learningRateGOAL = self.learningRateGOAL

                        # keep LR goal within bounds, stop explosions etc!
                        self.learningRateGOAL = max(
                            0.0002, min(0.001, self.learningRateGOAL)
                        )
                        if debugPrints or True:
                            print(f"updated goal LR to {self.learningRateGOAL}")

                    if self.perfectionistPassRate < perfectionistPassRate:
                        self.perfectionistPassRate += (
                            abs(self.perfectionistPassRate) / (self.totalTries)
                        ) * 0.3

                    if self.perfectionistPassRate > perfectionistPassRateSTART:
                        self.perfectionistPassRate -= random.choice(
                            [
                                (
                                    (
                                        (
                                            self.perfectionistPassRate
                                            - perfectionistPassRateSTART
                                        )
                                        * (self.totalTries * 0.5)
                                    )
                                    * 0.3
                                ),
                                (self.perfectionistPassRate / 2),
                            ]
                        )

                    if currentReflection is not None:
                        if scribeusedThisTurn:
                            self.scribe.reflectionPairsFromGuess.pop(0)
                        else:
                            self.reflectionTrainingPairs.pop(0)
                    else:
                        trainingPair = None  # move to next pair
                        i += 1
                        self.trainingStepCounter += (
                            1  # means reflections wont be training steps
                        )

                if debugPrints:
                    ʕっʘ‿ʘʔっ("♥finalSaveBeforeNewEpoch")
                if self.totalTurns == 0:
                    print("We ran out of data (probably)!!!")
                    self.totalTurns += 1  # sus this is dumb
                self.totalAvgLoss = self.totalLoss / self.totalTurns
                self.totalAvgAbsDelta = self.totalLossAbsDelta / max(1, self.totalTurns)
                self.totalAvgDelta = self.totalLossDelta / max(1, self.totalTurns)
                print(f"{self.totalAvgLoss} = {self.totalLoss} / {self.totalTurns}")
                if self.totalTurns > 0:
                    self.saveFreqActions()
                print("--- tutoring complete! ---")
                if self.sensory_bus is not None:
                    self.sensory_bus.cleanup()
        return

    def startTurnActions(self, _inputSeq, _targetSeq, _lastTurnLossDelta):
        with self.counsellor.infodump("startTurnActions") as ʕっʘ‿ʘʔっ:
            self.lastTurnLossDelta = _lastTurnLossDelta

            def _resolve(token):
                if isinstance(token, (int, np.integer)):
                    return int(token)
                return self.librarian.tokenToIndex.get(
                    token, self.librarian.tokenToIndex["<UNK>"]
                )

            self.inputTokenIndices = [_resolve(t) for t in _inputSeq]
            self.targetTokenIndexSeq = [_resolve(t) for t in _targetSeq]
            self.inputSeq = _inputSeq
            self.targetSeq = _targetSeq

            # if self.stats["windowEntropy"]:
            #    self.winEnt = self.stats["windowEntropy"]
            # else:
            #    self.winEnt = 0
            self.winEnt = self.stats.get("windowEntropy", 0.0)

        return self.inputTokenIndices, self.targetTokenIndexSeq

    @whocalled
    def trainStep(self, _inputTokenIndices, _targetTokenIndexSeq, _BACKWARDwobbleLoss):
        with self.counsellor.infodump("trainStep") as ʕっʘ‿ʘʔっ:
            # self.trainingStepCounter   += 1
            self.avgPixelDist = 0
            self.predictedTokenIndices = []
            inputSeqPredictions = list(
                _inputTokenIndices
            )  # start with input context, create a COPY!
            # Reuse persistent token buffer — resize only if window grew
            if self.numTokensPerStep > len(self._token_buffer):
                self._token_buffer = torch.zeros(
                    self.numTokensPerStep, dtype=torch.long, device=self.device
                )
            buffer = self._token_buffer
            buffer[: len(inputSeqPredictions)] = torch.as_tensor(
                inputSeqPredictions, device=self.device
            )
            self.logitSeq = []  # raw output of each prediction

            # Zero gradients at the start of the training step
            self.model.optimizer.zero_grad()

            # Read configuration chunk size
            chunk_size = globals().get("trainingChunkSize", 32)
            if chunk_size is None or chunk_size <= 0:
                chunk_size = self.numTokensPerStep  # Disable chunking

            triesInfluence = 0.0005
            triesLossModifier = 1 + (self.totalTries - 1) / 10
            BACKWARDtriesMod = (1.0 - triesInfluence) + (
                triesInfluence * triesLossModifier
            )

            perfectInfluence = 0.5
            perfectLossModifier = 1 / (1 + (self.tokenPerfectRate / 100))
            BACKWARDperfMod = (1.0 - perfectInfluence) + (
                perfectInfluence * perfectLossModifier
            )

            L = len(_targetTokenIndexSeq)
            scale_factor = (BACKWARDtriesMod * BACKWARDperfMod) / L if L > 0 else 0.0

            chunk_loss = None

            # Read external colour once per step (not per token — 269x reduction).
            # currentColour is set by the web/Discord frontend to blend bby's colour
            # with the live sprite. We cache it here; getPixelForStep uses self._ext_col_cache.
            try:
                with open(babyStateFilePath, "r") as f:
                    _step_state = json.load(f)
                if "currentColour" in _step_state:
                    self._ext_col_cache = _step_state["currentColour"]
            except (FileNotFoundError, json.JSONDecodeError, IOError):
                pass

            self.rgbPredictionBar = ""
            self.rgbTargetBar = ""
            self.rgbPromptBar = ""
            self.rgbBar = ""
            self.sensoryBar = ""

            self.tokenLevelCorrect = []
            self.tokenLevelLosses = []
            
            # Diagnostic training-loss accumulators
            diag_ce_losses = []
            diag_aux_losses = []
            diag_rep_losses = []
            diag_pixel_losses = []
            diag_sensory_losses = []
            diag_token_entropies = []
            diag_top1_confidences = []

            CL = 0.0
            DI = 0.0
            MF = 0.0
            LS = 0.0
            r = g = b = 0
            JSONtokenCorrect = False
            token_event = None

            if self.sensory_bus is not None:
                self.sensory_bus.step(self.totalTurns)
                self.model.cached_sensory = self.sensory_bus.get_tensor(self.device)
                self.model.cached_device_temp_c = (
                    self.sensory_bus.get_device_temp_c_tensor(self.device)
                )
            else:
                self.model.cached_sensory = None
                self.model.cached_device_temp_c = None

            if not skipPixels:
                # pre-filling next for step 0
                if self.pixelNext is None:
                    self.pixelNext = self.getPixelForStep(0)

            for j in range(
                self.numTokensPerStep
            ):  # predict multiple tokens in a sequence, one at a time
                self.currentTokenIndex = j
                stepLoss = None
                if not skipPixels:
                    # pixel = torch.rand(3, device = self.device)  # random RGB for now
                    if False:
                        pixelIndex = (
                            self.totalTurns + self.numTokensPerStep + j
                        ) % len(self.pixelSeq)
                        pixelNow = self.pixelSeq[pixelIndex]
                        pixelNext = self.pixelSeq[(pixelIndex + 1) % len(self.pixelSeq)]
                        if debugPrints:
                            print(
                                f"pixelIndex({pixelIndex}) = pixelNow {pixelNow}, next is {pixelNext}"
                            )
                    if False:
                        pixelNow = self.pixelNow
                        pixelNext = torch.tensor(
                            [
                                self.totalTokenPerfectRate * 1.0,
                                self.totalAvgAbsDelta * 1.0,
                                self.perfectionistPassRate * 1.0,
                            ],
                            device=self.device,
                        )
                    else:
                        self.pixelNow = self.pixelNext  # no clone needed — getPixelForStep returns a fresh tensor via clamp()
                        _prev_pixel_rgb = (
                            self._pixel_float_rgb
                        )  # capture before call updates it
                        self.pixelNext = self.getPixelForStep(j)
                        if debugPrints:
                            print(
                                f"now: {self.pixelNow}, next: {self.pixelNext}", end=""
                            )
                    self.model.nextPixelTarget = self.pixelNext
                else:
                    self.pixelNow = None
                if debugPrints:
                    ʕっʘ‿ʘʔっ("FORWARD")
                inputTensor = buffer[
                    : len(inputSeqPredictions)
                ]  # slices input to only keep relevant part
                try:
                    if forwardProfiler:
                        with torch.profiler.profile(record_shapes=True) as prof:
                            logits, predictedTokenIndex = self.model.forward_and_sample(
                                inputTensor,
                                _pixel=self.pixelNow,
                                _training=True,
                                _totAvgAbsDelta=self.totalAvgAbsDelta,
                            )
                    else:
                        logits, predictedTokenIndex = self.model.forward_and_sample(
                            inputTensor,
                            _pixel=self.pixelNow,
                            _training=True,
                            _totAvgAbsDelta=self.totalAvgAbsDelta,
                        )
                except RuntimeError as e:
                    print("TUTOR.trainStep.forward failed!", e)
                    return [], []
                predictedRGB = self.model.predPixel
                if debugPrints:
                    print(f"guess: {predictedRGB}")
                if debugPrints:
                    print("predictedRGB:", predictedRGB.tolist())
                if debugPrints:
                    print("EMBED (mean):", self.model.latestTokenEmbed.mean().item())

                if forwardProfiler:
                    print(prof.key_averages().table())

                if debugPrints:
                    ʕっʘ‿ʘʔっ("getResponseFromLogits")
                predy = int(predictedTokenIndex.item())
                if debugPrints:
                    print("nextToken: ")
                    print(predy, end="")
                nextyToky = self.librarian.indexToToken.get(
                    predy,
                    getattr(self.librarian, "unkToken", "<UNK>"),
                )
                nextyToky = str(nextyToky)
                try:
                    eos_id = (
                        self.librarian.tokenToIndex.get(eos_replacement_token_str)
                        if eos_replacement_token_str
                        else None
                    )
                except Exception:
                    eos_id = None
                if eos_id is not None and predy == int(eos_id):
                    token_for_terminal = eos_token_str
                else:
                    token_for_terminal = nextyToky.replace("Ġ", " ")
                self.toktoktok = self.calligraphist.S_renderTerminalText(
                    token_for_terminal
                )
                print(f"{self.toktoktok}", end="", flush=True)
                if debugPrints:
                    print("token index:", predictedTokenIndex.item())
                if debugPrints:
                    print(
                        f"[j={j}] inputLen={len(inputTensor)} → predicted {predictedTokenIndex.item()}"
                    )

                if debugPrints:
                    ʕっʘ‿ʘʔっ("inputSeqPredictions")
                predicted_index = (
                    predy  # already computed above — avoid second GPU→CPU sync
                )
                self.predictedTokenIndices.append(predicted_index)

                # -- RGB visual tracker --
                if not skipPixels and (
                    hasattr(self.model, "latestTokenEmbed")
                    and hasattr(self.model, "pixelPupil")
                    and hasattr(self.model, "nextPixelTarget")
                ):
                    # last token's RGB prediction
                    # rp/gp/bp: prompt pixel = previous iter's getPixelForStep floats (no GPU sync needed)
                    # rt/gt/bt: target pixel = current iter's getPixelForStep floats (no GPU sync needed)
                    # r/g/b: model's predicted colour — still needs GPU→CPU sync for JSON write
                    rp = int(_prev_pixel_rgb[0] * 255)
                    gp = int(_prev_pixel_rgb[1] * 255)
                    bp = int(_prev_pixel_rgb[2] * 255)
                    r, g, b = (predictedRGB * 255).int().tolist()
                    rt = int(self._pixel_float_rgb[0] * 255)
                    gt = int(self._pixel_float_rgb[1] * 255)
                    bt = int(self._pixel_float_rgb[2] * 255)

                    slice1 = self.char1
                    slice2 = self.char2
                    slice3 = self.char3

                    pixelDist = abs(r - rt) + abs(g - gt) + abs(b - bt)

                    MAX_FULL_SAFEBOIS = 64

                    if self.numTokensPerStep <= MAX_FULL_SAFEBOIS:
                        pass
                    else:
                        # boiLen1 = len(self.char1)
                        # boiLen2 = len(self.char2)
                        # boiLen3 = len(self.char3)

                        # slice1 = self.char1[j % boiLen1]
                        # slice2 = self.char2[j % boiLen2]
                        # slice3 = self.char3[j % boiLen3]
                        slice1 = "."
                        smallDist = pixelDist * 0.01
                        slice2 = f"{smallDist:.0f}"
                        slice3 = slice2

                    pixelThresh = 1
                    self.avgPixelDist += pixelDist

                    # prompt_block = f"\x1b[48;2;{rp};{gp};{bp}m\x1b[38;2;{r};{g};{b}m{slice1}\x1b[0m"
                    if (r, g, b) == (rt, gt, bt):
                        invert_r, invert_g, invert_b = 255 - r, 255 - g, 255 - b
                        # Bold 1, Underline 4
                        pred_block = f"\x1b[1;4;48;2;{invert_r};{invert_g};{invert_b}m\x1b[38;2;{rt};{gt};{bt}m{slice1}\x1b[0m"
                    elif pixelDist <= pixelThresh:
                        invert_r, invert_g, invert_b = 255 - r, 255 - g, 255 - b
                        # Bold 1, Underline 4, Yellow text 38;5;226
                        pred_block = f"\x1b[1;4;48;2;{invert_r};{invert_g};{invert_b}m\x1b[38;5;226m{slice2}\x1b[0m"
                    else:
                        pred_block = f"\x1b[48;2;{r};{g};{b}m\x1b[38;2;{rt};{gt};{bt}m{slice2}\x1b[0m"
                    tgt_block = f"\x1b[48;2;{rt};{gt};{bt}m\x1b[38;2;{r};{g};{b}m{slice3}\x1b[0m"
                    if debugPrints:
                        print(
                            "PRED:", r, g, b, "TARGET:", rt, gt, bt, "DIST:", pixelDist
                        )

                    # self.rgbPromptBar     += prompt_block
                    self.rgbPredictionBar += pred_block
                    self.rgbTargetBar += tgt_block

                sampledTokens = (
                    scheduledSampling and random.random() < self.scheduledSamplingRate
                )
                if j == 0:
                    self.sampledFlags = []  # Only clear at start
                self.sampledFlags.append(sampledTokens)
                if sampledTokens:
                    self.stats["sampledTokens"] = self.stats.get("sampledTokens", 0) + 1
                    nextTokenInput = predicted_index
                elif j < len(_targetTokenIndexSeq):
                    nextTokenInput = _targetTokenIndexSeq[j]
                else:
                    nextTokenInput = predicted_index

                inputSeqPredictions.append(
                    nextTokenInput
                )  # multi-token autoregressive generation: append next token to your current input — becomes the prompt for the next token
                isCorrect = nextTokenInput == predicted_index
                self.gotIt = isCorrect
                self.tiktiktik = nextTokenInput
                self.model.targetTokenFromTutor = _targetTokenIndexSeq[j]
                self.tokenLevelCorrect.append(1.0 if isCorrect else 0.0)
                if debugPrints:
                    print(
                        f"isCorrect = {isCorrect} for target: {nextTokenInput} vs guess: {predicted_index}... tokenLevelCorrect = {self.tokenLevelCorrect}"
                    )

                if debugPrints:
                    ʕっʘ‿ʘʔっ("loop through tokens for this step")
                if j < len(_targetTokenIndexSeq):
                    if debugPrints:
                        ʕっʘ‿ʘʔっ("totalTokenCounter")
                    # self.totalTokenEvaluations += 1

                    if debugPrints:
                        ʕっʘ‿ʘʔっ("computeLoss")
                    stepLoss = self.model.computeLoss(
                        _logits=logits,
                        _targetTokenIndex=_targetTokenIndexSeq[j],
                        _totalAvgAbsDelta=self.totalAvgAbsDelta,
                        _learningRateGOAL=self.learningRateGOAL,
                        _perfectTokens=self.perfectTokens,
                    )

                    if debugPrints:
                        ʕっʘ‿ʘʔっ("appendStepLoss")
                    self.tokenLevelLosses.append(stepLoss.item())
                    
                    # Accumulate diagnostic values
                    diag_ce_losses.append(self.model.CEloss_used)
                    diag_aux_losses.append(self.model.AUXlossKL_used + self.model.AUXlossCos_used)
                    diag_rep_losses.append(self.model.repLoss_used)
                    diag_pixel_losses.append(self.model.pixelLoss_used)
                    diag_sensory_losses.append(self.model.sensoryLoss_used)
                    
                    if hasattr(self.model, "lastSoftSample") and self.model.lastSoftSample is not None:
                        probs = self.model.lastSoftSample
                        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
                        confidence = probs.max(dim=-1)[0].mean().item()
                        diag_token_entropies.append(entropy)
                        diag_top1_confidences.append(confidence)
                    
                    scaled_loss = stepLoss * scale_factor
                    chunk_loss = scaled_loss if chunk_loss is None else chunk_loss + scaled_loss
                    if debugPrints:
                        print(f"self.tokenLevelLosses = {self.tokenLevelLosses}")

                try:
                    JSONtokenCorrect = bool(isCorrect)
                    if True:  # j == 0:
                        CL = self.model.cerebralLoad
                        DI = self.model.dreamIntensity
                        MF = self.model.memoryFlux
                        LS = self.totalAvgAbsDelta
                    # reuse already-computed value — avoids a second GPU→CPU sync per token
                    token_loss_value = (
                        float(self.tokenLevelLosses[-1])
                        if self.tokenLevelLosses
                        else 0.0
                    )
                    token_event = self._build_token_event(
                        self.toktoktok,
                        nextyToky,
                        predicted_index,
                        (r, g, b),
                        token_loss=token_loss_value,
                    )
                    self.token_event_history.append(token_event)
                    if len(self.token_event_history) > tokenEventHistoryLimit:
                        del self.token_event_history[:-tokenEventHistoryLimit]
                    # Per-token live colour write — omits token_events history list so each
                    # write is ~200 bytes (not ~100KB). Full history is written once per step
                    # after the loop. Compact JSON (no indent) for speed.
                    # Throttled to every 8 tokens: the dashboard refreshes at human-visible
                    # speeds, sub-8-token resolution is invisible, and this cuts disk writes
                    # from ~269/step to ~34/step (8× fewer syscalls).
                    if j % 8 == 0:
                        liveState = {
                            "timestamp": time.time(),
                            "R": r,
                            "G": g,
                            "B": b,
                            "cerebralLoad": CL,
                            "dreamIntensity": DI,
                            "memoryFlux": MF,
                            "learningStability": LS,
                            "correct": JSONtokenCorrect,
                            "token_event": token_event,
                        }
                        temp_path = babyStateFilePath + ".tmp"
                        with open(temp_path, "w") as f:
                            json.dump(liveState, f)  # compact — no indent
                        os.replace(temp_path, babyStateFilePath)
                except Exception as e:
                    print(f"could not write live token state: {e}")

                # Backpropagate in chunks
                if L > 0 and (j < L) and ((j + 1) % chunk_size == 0 or j == L - 1):
                    is_last_loss_token = (j == L - 1)
                    if is_last_loss_token:
                        if debugPrints:
                            print(f"[CHUNK] final backward at j={j}/{L-1}")
                        
                        # Compute pixelDistLoss at the end
                        self.avgPixelDist = self.avgPixelDist / self.numTokensPerStep
                        pixelDistLoss = min(1.5, self.avgPixelDist * 0.001)
                        self.pixelDistLoss_used = pixelDistLoss
                        
                        # Add pixelDistLoss to chunk_loss
                        if chunk_loss is None:
                            chunk_loss = torch.tensor(pixelDistLoss, device=self.device)
                        else:
                            chunk_loss = chunk_loss + pixelDistLoss
                            
                        # Compute base BACKWARDloss value (without WE terms) for metrics logging
                        BACKWARDloss_val = sum(self.tokenLevelLosses) / L if L > 0 else 0.0
                        self.triesLoss_used = BACKWARDloss_val * (BACKWARDtriesMod - 1.0)
                        self.perfLoss_used = BACKWARDloss_val * (BACKWARDperfMod - 1.0)
                        BACKWARDloss_val = BACKWARDloss_val * BACKWARDtriesMod * BACKWARDperfMod + pixelDistLoss
                        
                        # WEloss/windowEntropyBonus:
                        if windowEntropyBonus:
                            # Let's compute WEloss terms directly
                            WE_terms = torch.tensor(0.0, device=self.device)
                            WE_terms = WE_terms + 0.00010 * torch.relu(
                                0.02 - self.model.interneuronNetwork.entropyBonus
                            )
                            WE_terms = WE_terms + 0.00100 * torch.relu(
                                0.20 - self.model.interneuronNetwork.windowSizeEntropy
                            )
                            rangePenalty = self.model.interneuronNetwork.rangePenalty
                            WE_terms = WE_terms + 0.01000 * rangePenalty
                            meanPenalty = self.model.interneuronNetwork.meanPenalty
                            WE_terms = WE_terms + 0.01000 * meanPenalty
                            
                            short_decay = torch.sigmoid(self.model.memory.shortTermDecay)
                            long_decay = torch.sigmoid(self.model.memory.longTermDecay)
                            target_short_decay = 0.65
                            target_long_decay = 0.92
                            short_decay_penalty = (short_decay - target_short_decay) ** 2
                            long_decay_penalty = (long_decay - target_long_decay) ** 2
                            
                            WE_terms = WE_terms + 0.10000 * (short_decay_penalty + long_decay_penalty)
                            
                            chunk_loss = chunk_loss + WE_terms
                            BACKWARDloss_val = BACKWARDloss_val + WE_terms.item()
                            
                        if not torch.isfinite(chunk_loss):
                            print("TUTOR.trainStep.backward !!! Loss is NaN or Inf:", chunk_loss)
                            self.model.optimizer.zero_grad(set_to_none=True)
                            empty_mps_cache()
                            return [], []
                        
                        self.totalLoss += BACKWARDloss_val
                        
                        # Update rolling average loss with exponential decay
                        if self.averageRecentLoss == 0:
                            self.averageRecentLoss = BACKWARDloss_val
                        else:
                            alpha = 0.1
                            self.averageRecentLoss = (1 - alpha) * self.averageRecentLoss + alpha * BACKWARDloss_val
                        self.latestLossDelta = BACKWARDloss_val - self.averageRecentLoss
                        self.stepLossFloat = BACKWARDloss_val
                        self.stats["loss"] = self.stepLossFloat
                        
                        try:
                            if profiler:
                                with torch.profiler.profile(record_shapes=True) as prof:
                                    self.model.backward(chunk_loss, self.latestLossDelta, _run_optimizer=True)
                            elif mpsProfiler:
                                with torch.mps.profiler.profile(
                                    mode="interval", wait_until_completed=False
                                ) as prof:
                                    self.model.backward(chunk_loss, self.latestLossDelta, _run_optimizer=True)
                            else:
                                self.model.backward(chunk_loss, self.latestLossDelta, _run_optimizer=True)
                        except RuntimeError as e:
                            print("TUTOR.trainStep.backward failed!", e)
                            self.model.optimizer.zero_grad(set_to_none=True)
                            empty_mps_cache()
                            return [], []
                            
                        if profiler:
                            print(prof.key_averages().table())
                            
                        # --- Diagnostic Training-Loss Breakdown ---
                        # Compute stats
                        diag_ce_mean = sum(diag_ce_losses) / len(diag_ce_losses) if diag_ce_losses else 0.0
                        diag_aux_mean = sum(diag_aux_losses) / len(diag_aux_losses) if diag_aux_losses else 0.0
                        diag_rep_mean = sum(diag_rep_losses) / len(diag_rep_losses) if diag_rep_losses else 0.0
                        diag_pixel_mean = sum(diag_pixel_losses) / len(diag_pixel_losses) if diag_pixel_losses else 0.0
                        diag_sensory_mean = sum(diag_sensory_losses) / len(diag_sensory_losses) if diag_sensory_losses else 0.0
                        diag_entropy_mean = sum(diag_token_entropies) / len(diag_token_entropies) if diag_token_entropies else 0.0
                        diag_confidence_mean = sum(diag_top1_confidences) / len(diag_top1_confidences) if diag_top1_confidences else 0.0
                        
                        diag_rep_rate = 0.0
                        if self.predictedTokenIndices:
                            diag_rep_rate = (1.0 - len(set(self.predictedTokenIndices)) / len(self.predictedTokenIndices)) * 100.0

                        # Get grad norms & actual LR from model
                        diag_grad_before = getattr(self.model, "last_grad_norm_before_clip", 0.0)
                        diag_grad_after = getattr(self.model, "last_grad_norm_after_clip", 0.0)
                        diag_actual_lr = self.model.optimizer.param_groups[0]['lr'] if self.model.optimizer.param_groups else 0.0
                        
                        diag_we_total = WE_terms.item() if ('WE_terms' in locals() and WE_terms is not None) else 0.0

                        print("\n" + "=" * 60)
                        print("               TRAINING-LOSS BREAKDOWN")
                        print("=" * 60)
                        print(f" - raw language CE mean:                    {diag_ce_mean:.6f}")
                        print(f" - aux loss mean:                           {diag_aux_mean:.6f}")
                        print(f" - repetition loss mean:                    {diag_rep_mean:.6f}")
                        print(f" - pixel loss mean:                         {diag_pixel_mean:.6f}")
                        print(f" - sensory loss mean:                       {diag_sensory_mean:.6f}")
                        print(f" - window entropy/range/memory penalty total: {diag_we_total:.6f}")
                        print(f" - final backward loss:                     {BACKWARDloss_val:.6f}")
                        print(f" - actual optimiser LR:                     {diag_actual_lr:.8f}")
                        print(f" - gradient norm before clipping:           {diag_grad_before:.6f}")
                        print(f" - gradient norm after clipping:            {diag_grad_after:.6f}")
                        print(f" - token entropy:                           {diag_entropy_mean:.6f}")
                        print(f" - top-1 confidence:                        {diag_confidence_mean:.2%}")
                        print(f" - repetition rate:                         {diag_rep_rate:.2f}%")
                        print("=" * 60 + "\n")
                            
                        chunk_loss = None
                    else:
                        if chunk_loss is not None:
                            if debugPrints:
                                print(f"[CHUNK] intermediate backward chunk at j={j}")
                            try:
                                self.model.backward(chunk_loss, self.latestLossDelta, _run_optimizer=False)
                            except RuntimeError as e:
                                print(f"TUTOR.trainStep.backward intermediate chunk failed at j={j}!", e)
                                self.model.optimizer.zero_grad(set_to_none=True)
                                empty_mps_cache()
                                return [], []
                        chunk_loss = None

            # Write babyState once per step (was once per token — 269x per step previously)
            try:
                if token_event is not None:
                    babyState = {
                        "timestamp": time.time(),
                        "R": r,
                        "G": g,
                        "B": b,
                        "cerebralLoad": CL,
                        "dreamIntensity": DI,
                        "memoryFlux": MF,
                        "learningStability": LS,
                        "correct": JSONtokenCorrect,
                        "token_event": token_event,
                        "token_events": self.token_event_history,
                    }
                    temp_path = babyStateFilePath + ".tmp"
                    with open(temp_path, "w") as f:
                        json.dump(babyState, f, indent=2)
                    os.replace(temp_path, babyStateFilePath)
            except Exception as e:
                print(f"could not write to {babyStateFilePath}: {e}")

            self.inputSeqPredictions = (
                inputSeqPredictions  # So we can access it in collectTurnStats
            )
            self.inputSampledFlags = self.sampledFlags.copy()
            if not skipPixels:
                self.rgbBar = (
                    f"PRED: {self.rgbPredictionBar}\nTRUE: {self.rgbTargetBar}"
                )
            pred_sensory = getattr(self.model, "predSensory", None)
            true_sensory = getattr(self.model, "targetSensory", None)
            if pred_sensory is not None and true_sensory is not None:
                pred_vals = pred_sensory.detach().cpu().tolist()
                true_vals = true_sensory.detach().cpu().tolist()
                self.sensoryBar = self._format_sensory_bar(pred_vals, true_vals)
                if self.rgbBar:
                    self.rgbBar = f"{self.rgbBar}\n{self.sensoryBar}"
                else:
                    self.rgbBar = self.sensoryBar

            # Old backward logic was removed (now run incrementally inside the loop at chunk boundaries).
            if debugPrints:
                ʕっʘ‿ʘʔっ("actions after looping")
            self.avgPixelDistTotals += self.avgPixelDist
            self.totalAvgPixelDist = self.avgPixelDistTotals / max(1, self.totalTurns)
            self.learningRate = math.exp(self.model.logLR.detach().item())
            self.memoryLength = self.model.memoryLength.detach().item()
            # self.gradientClipMaxNorm        = math.exp(self.model.logGradClip.detach().item())
            self.scheduledSamplingRateFloat = self.scheduledSamplingRate.detach().item()
            self.repetitionPenalty = self.model.repetitionPenalty.detach().item()
            # self.INN_cerebellum             = self.model.interneuronNetwork.cerebellum.detach().cpu().item()
            # self.INN_cerebellumMean         = self.model.interneuronNetwork.cerebellum.mean().cpu().item()

            if "BACKWARDloss" in locals():
                del BACKWARDloss
            if "cumulativeLoss" in locals():
                del cumulativeLoss
            if "stepLoss" in locals():
                del stepLoss
            if "chunk_loss" in locals():
                del chunk_loss
            if "scaled_loss" in locals():
                del scaled_loss

            # Throttled: flush every 25 steps instead of every step.
            if self.totalTurns % 25 == 0:
                if self.device.type == "mps":
                    if debugPrints:
                        ʕっʘ‿ʘʔっ("emptyCache (mps)")
                    empty_mps_cache()
                else:
                    gc.collect()

            ids = [int(idx) for idx in self.predictedTokenIndices]
            self.predictedTokenIndices = ids
            self.decodedTokenIndices = self.librarian.decodeIDs(ids)

            return self.predictedTokenIndices, self.logitSeq

    @whocalled
    def getPixelForStep(self, j):
        # External colour is read once per step in trainStep and cached in self._ext_col_cache.
        # (Was: read babyStateFilePath every token — 269 file reads per step, mostly wasted
        #  because our own per-token write overwrites the file without preserving currentColour.)
        babyState = (
            {"currentColour": self._ext_col_cache} if self._ext_col_cache else {}
        )

        x = (j + 1) / (self.numTokensPerStep + self.trainingStepCounter % 10)

        perf = (self.totalTokenPerfectRate + self.tokenPerfectRate) / 200
        stepLoss = min(self.stepLossFloat / 10.0, 1.0)
        correct = self.tokenLevelCorrect[-1] if len(self.tokenLevelCorrect) > 0 else 0.0
        tokenLoss = self.tokenLevelLosses[-1] if len(self.tokenLevelLosses) > 0 else 1.0
        delta = min((abs(tokenLoss - self.averageRecentLoss), 10)) / 10
        totDelta = min((abs(self.averageRecentLoss - self.lastRunLoss)), 10) / 10

        # PULSE SPEED = HIGHER LOSS RATE, HIGHER PULSE SPEED (increased metabolism?)
        pulseSpeed = (
            0.5
            + (x * 0.25)
            + 2 * delta
            + 1 * ((self.totalAvgLoss * 0.15) + (stepLoss * 0.25) + (tokenLoss * 0.25))
        )
        timePulse = 0.5 * (1 + math.sin(2 * math.pi * pulseSpeed * x))

        # RED - HIGH PERFECT TOKENS, EXCITED/SKILLED (energy up when getting more perfect, more red!)
        red = (
            0.01
            + (0.15 * timePulse)
            + (0.15 * ((perf * 0.7) + (tokenLoss * 0.2)))
            + (0.8 * correct)
        )
        red = min(red, 1.0)
        # GREEN - HIGH ABS DELTA, OVERSTIMULATED/LEARNING, mid range (growth up (or a bit queasy lol) when getting stronger deltas, more green!)
        hueShift = (math.sin(2 * math.pi * (x + delta + perf)) + 1) / 2  # [0..1]
        high_delta_event = pow(delta, 3)
        green = (
            0.01
            + (tokenLoss * 0.08)
            + (
                (
                    0.2
                    * (
                        (delta * 0.5)
                        + (
                            (self.totalAvgAbsDelta * 0.6)
                            + (self.totalAvgDelta * 0.4) * 0.5
                        )
                    )
                )
                + 0.05 * hueShift
            )
        )
        green = (green + (0.01 + (high_delta_event * 0.5))) * 0.5
        green = min(green, 1.0)
        # BLUE - HIGH PASS RATE, CALM, long range (calm up when doing better than the previous run, more blue!)
        blue = (
            0.3
            + ((self.perfectionistPassRate / 100) * 0.6)
            + (0.2 * totDelta)
            + (0.4 * (1 - timePulse))
        )
        blue = min(blue, 1.0)

        if babyState:
            ext_col = babyState.get("currentColour", {"R": 128, "G": 128, "B": 128})
            # print(f"using bbySprite colours :D ({ext_col})")

            # Scale external RGB [0–255] to [0–1]
            ext_r = ext_col["R"] / 255.0
            ext_g = ext_col["G"] / 255.0
            ext_b = ext_col["B"] / 255.0

            # Blend internal prediction with external colour drift
            blend_factor = 0.25  # scales 0–1 ideally
            red = red * (1 - blend_factor) + ext_r * blend_factor
            green = green * (1 - blend_factor) + ext_g * blend_factor
            blue = blue * (1 - blend_factor) + ext_b * blend_factor

        if self.sensory_bus is not None:
            light_delta = self.sensory_bus.state.get("global_light_delta")
            if light_delta is not None:
                light_shift = (float(light_delta) - 0.5) * 2.0
                brightness = 1.0 + (light_shift * 0.1)
                red = min(max(red * brightness, 0.0), 1.0)
                green = min(max(green * brightness, 0.0), 1.0)
                blue = min(max(blue * brightness, 0.0), 1.0)

        # Fill CPU scratch buffer (no alloc) → single copy to MPS → fresh tensor via clamp()
        self._pixel_cpu[0] = red
        self._pixel_cpu[1] = green
        self._pixel_cpu[2] = blue
        self._pixel_buf.copy_(self._pixel_cpu)
        pixelPret = self._pixel_buf.clamp(
            0, 1
        )  # non-in-place: returns independent tensor
        self._pixel_float_rgb = (
            red,
            green,
            blue,
        )  # cache Python floats — avoids GPU→CPU sync in main loop

        if debugPrints:
            print(
                f"perf {perf}, stepLoss {stepLoss}, correct {correct}, tokenLoss {tokenLoss}//{j} {len(self.tokenLevelLosses)} ({(j <= len(self.tokenLevelLosses) - 1)}), delta {delta}, pulseSpeed {pulseSpeed}"
            )
            print(
                f"tokenstep {j:2d} | x: {x:.2f} timePulse: {timePulse:.2f} hueShift: {hueShift:.2f} -> pixel: ({red:.2f}, {green:.2f}, {blue:.2f})"
            )
            print(f"pixelpret {pixelPret}")

        return pixelPret

    @whocalled
    def babyReflection(self):
        with self.counsellor.infodump("startTurnActions") as ʕっʘ‿ʘʔっ:
            delta = round(self.latestLossDelta, 2)
            perfect = self.perfectTokens
            repeated = (
                round(self.stats.get("repeatedPercent", 0.0) * 100, 1)
                if "repeatedPercent" in self.stats
                else 0.0
            )

            def makeDataStuff():
                connective = random.choice(
                    [
                        ":",
                        " of",
                        " is at",
                        " =",
                        ",",
                        " is",
                        "...",
                        " of about",
                        " of approx",
                        " of approximately",
                        " sitting at",
                        " at about",
                        " that is about",
                        " that's at",
                        " of",
                    ]
                )
                return random.choice(
                    [
                        f"average recent loss{connective} {self.averageRecentLoss:.2f}, ",
                        f"latest loss delta{connective} {delta:.2f}, ",
                        f"perfect tokens{connective} {perfect:.0f}, ",
                        f"repeated token percentage{connective} {repeated:.2f}, ",
                        f"learning rate{connective} {self.learningRate:.5f}, ",
                        f"temperature{connective} {self.temperature:.2f}, ",
                        f"scheduled sampling rate{connective} {self.scheduledSamplingRate.item():.2f}, ",
                        f"a memory length{connective} {self.memoryLength:.2f}, ",
                        f"a repetition penalty{connective} {self.repetitionPenalty:.2f}, ",
                    ]
                )

            prompt = f"[{userName}]: "
            prompt += random.choice(
                [
                    f"{self.loadIntro()}",
                    "hiya! you know you're doing well, so try not to feel too frustrated or let down. ",
                    "it's ok to feel anxious, just remember that it's okay to mess up! ",
                    "aaagh there's always so much data, but it's how we learn! ",
                    "so, uh, how are you today? :) ",
                    "i'm just checking in to let you know about your latest scores! ",
                    "i hope it helps you understand your world a little better :) ",
                ]
            )
            prompt += "take a look at these: "
            prompt += makeDataStuff()
            prompt += makeDataStuff() + "and "
            prompt += makeDataStuff()
            # prompt += makeDataStuff() + "and "
            # prompt += makeDataStuff()

            target = f"[{babyName}]: hey {userName}! "
            target += random.choice(
                [
                    "umm, i am learning today, this is how i'm doing: ",
                    "ok, so, to put it as simply as I can, i am: ",
                    "ummm.. i've been trying to learn a lot about stats but numbers still confuse me a little, what's the context of all this?! like, to start with, ",
                    "i am happy! i did it! i know it! i am just a baby! i learnt it! haha! i learned: ",
                    "good morning! i hope you're okay, ",
                    "umm, i dunno how i feel about my scores right now!! ",
                ]
            )
            target += self.hesJustABaby
            babyEndings = [
                "ok, anyway... ",
                "i'm going to get back to school now... ",
                "thanks for helping me think! ",
                "learning is weird... but i like it! ",
                "i guess i've gotta go keep trying! ",
                "i guess i'll just have to keep trying! ",
                "i suppose i just need to work on it! ",
                "i don't think that it's wrong to have made mistakes ",
                "i'm proud of myself! ",
                "thank you! ",
                "thanks! ",
                "i'm just a baby! ",
                "i know it! ",
                "i did it! ",
                "i feel it! ",
                "i am happy! ",
                "i am learning! ",
                "i learned it! ",
                "lol ",
                ":) ",
                "talk in a bit! ",
                "talk later! ",
                "i'm gonna carry on with it now :D ",
                "thank you for teaching me new things! ",
                "it's really weird to exist lol, ",
                "i'm getting tired... but i don't want to give up just yet! ",
                "that's a lot of numbers! ",
                "i don't really know what to do with them, but i'll keep trying! ",
                "do you think i'm doing well?",
                "i'm worried that i'm underperforming! ",
                "honestly, all this learning has me excited! ",
            ]
            target += random.choice(babyEndings)
            target += random.choice(babyEndings)
            target += random.choice(babyEndings)

            reflectionText = prompt + target

            reflectionTokens = self.librarian.tokenizeText(reflectionText.lower())

            tries = 0
            while len(reflectionTokens) < (self.numTokensPerStep * 3) and tries < 50:
                target += " " + random.choice(
                    [random.choice(babyEndings), makeDataStuff()]
                )
                reflectionText = prompt + " " + target
                reflectionTokens = self.librarian.tokenizeText(reflectionText.lower())
                tries += 1
                if tries % 5 == 0:
                    print(
                        f"[babyReflection] still too short after {tries} tries: {len(reflectionTokens)} tokens"
                    )
            if tries >= 50:
                raise ValueError(
                    f"babyReflection failed: could not reach enough tokens after {tries} tries."
                )

        inputTargetPairs = []
        reflectionPointer = 0

        while reflectionPointer + self.numTokensPerStep * 2 <= len(reflectionTokens):
            inputSeq = reflectionTokens[
                reflectionPointer : reflectionPointer + self.numTokensPerStep
            ]
            targetSeq = reflectionTokens[
                reflectionPointer + self.numTokensPerStep : reflectionPointer
                + self.numTokensPerStep * 2
            ]

            inputTargetPairs.append((inputSeq, targetSeq))

            reflectionPointer += 1

        self.hesJustABaby = (
            "oops! no stats collected! such a shame! well... day off for me! ;)"
        )
        return inputTargetPairs

    @whocalled
    def saveFreqActions(self):
        with self.counsellor.infodump(
            "saveFreqActions"
        ) as ʕっʘ‿ʘʔっ:  # SAVE THE MODEL EVERY x STEPS
            print(self.calligraphist.S_apply("dim", "autosaving..."))
            saveCounter = self.trainingStepCounter * self.dataStride
            self.model.saveModel(
                _newStartIndex=self.startIndex,
                _trainingStepCounter=saveCounter,
                _totalAvgLoss=self.totalAvgLoss,
                _first=self.first,
            )
            p = self.trainingStepCounter + saveModelFreq
            print(
                self.calligraphist.S_apply(
                    "dim",
                    f"autosave successful! saving every {saveModelFreq} steps, the next autosave will be at step {p}...",
                )
                + self.calligraphist.S_apply("reset", "")
            )
            if debugPrints:
                ʕっʘ‿ʘʔっ("grad checks")
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    print(
                        f"after = {self.calligraphist.S_apply('emergency', f'NO GRAD: {name}')}"
                    )
                else:
                    stats = get_grad_stats(p.grad)
                    shape = stats["shape"]
                    norm = stats["norm"]
                    sparsity = stats["sparsity"]
                    mean = stats["mean"]
                    std = stats["std"]
                    print(
                        f"after = {self.calligraphist.S_apply('almostPerfect', f'yes grad: {name} | shape: {shape} | norm: {norm:.4f} | sparsity: {sparsity:.2%} | mean: {mean:.4f} | std: {std:.4f}')}"
                    )

    @whocalled
    def printFreqActions(self):
        with self.counsellor.infodump(
            "printFreqActions"
        ) as ʕっʘ‿ʘʔっ:  # PRINTING TRAINING OUTPUT TO TERMINAL
            # recentLoss = sum(self.recentPrintLosses)/len(self.recentPrintLosses) if self.recentPrintLosses else None
            if debugPrints:
                ʕっʘ‿ʘʔっ("calligraphist.S_colourPrintTraining")

            # Ensure _inputSeq is a list of strings
            input_seq_str = [
                self.librarian.indexToToken.get(tok, str(tok)) if isinstance(tok, int)
                else str(tok)
                for tok in self.inputSeq
            ]

            self.calligraphist.S_colourPrintTraining(
                _step=(self.trainingStepCounter),
                _inputSeq=input_seq_str,
                _guessedSeq_str=self.stringStats.get(
                    "boldPerfects", self.guessedTokenSeq
                ),
                _targetSeq_str=self.stringStats.get("usedInputSeq", []),
                _recentLoss=self.averageRecentLoss,  # self.ʕっෆ‿ෆʔっ.get("loss", {}).get(f"{trainingLogFreq_A}_avg", 0), # self.stepLossFloat,
                _loss=self.stepLossFloat,
                _latestLossDelta=self.latestLossDelta,
                _totalTokenCount=self.tokenCounts,
            )

    def tidy_token(self, tok):
        return tok.replace("Ġ", " ").replace("Ċ", "\n")

    def update_top_tokens(self):
        delimiter = self.calligraphist.S_apply("dim", ", ")
        top_pairs = list(self.tokenCounts.most_common(50))
        self.topTokens_str = ": " + delimiter.join(
            [self.calligraphist.S_apply("dim", f"{k}({v:.1f})") for k, v in top_pairs]
        )
        self.topTokens_forBot = ": " + delimiter.join(
            f"{self.tidy_token(k)}({v:.1f})" for k, v in top_pairs
        )

    def load_token_counts(self):
        try:
            with open(topTokensFilePath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.tokenCounts = Counter(data)
        except FileNotFoundError:
            self.tokenCounts = Counter()
        self.update_top_tokens()
        self.tokenCounts_dirty = False

    def save_token_counts(self):
        if not self.tokenCounts_dirty:
            return
        try:
            os.makedirs(os.path.dirname(topTokensFilePath), exist_ok=True)
            with open(topTokensFilePath, "w", encoding="utf-8") as f:
                json.dump(dict(self.tokenCounts), f)
            self.tokenCounts_dirty = False
        except Exception as e:
            print(f"could not write to {topTokensFilePath}: {e}")

    @whocalled
    def logFreqActions(
        self,
        _trainingDataPairs,
        _stringStats,
        _frequency,
        _trainingLogPath,
        _detailedLogging,
        _saveLog,
        _currentStepOverride=None,
    ):
        with self.counsellor.infodump("logFreqActions") as ʕっʘ‿ʘʔっ:
            self.stringStats = _stringStats
            self.trainingLogPath = _trainingLogPath
            rollingDict = self.model.getRollingTokenTotalsDict()
            delimiter = self.calligraphist.S_apply("dim", ", ")
            topGuess_str = "topGuess: " + delimiter.join(
                [
                    self.calligraphist.S_apply("dim", f"{k}({v:.1f})")
                    for k, v in Counter(rollingDict).most_common(50)
                ]
            )
            # topGuess_str = "topGuess: " + f"{self.calligraphist.S_apply("dim", ", ")}".join([self.calligraphist.S_apply("dim", f"{k}") for k, v in self.model.rollingTokenTotals.most_common(50)]) + "]"
            # topTokens_str = "[" + f"{self.calligraphist.S_apply("dim", ", ")}".join([self.calligraphist.S_apply("dim", f"{k}({v:.0f})") for k, v in self.tokenCounts.most_common(20)]) + "]"
            self.update_top_tokens()
            if _frequency >= trainingLogFreq_B and self.tokenCounts_dirty:
                self.save_token_counts()

            # self.stats.update(self.ʕっෆ‿ෆʔっ) # SUSSY BUSSY !!!!!!!!!!!!!!!!!!!
            # fullStats = dict(self.stats)
            # fullStats.update(self.ʕっෆ‿ෆʔっ)

            currentStepForCalc = (
                _currentStepOverride
                if _currentStepOverride is not None
                else self.trainingStepCounter
            )

            if debugPrints:
                ʕっʘ‿ʘʔっ("calculateTrainingDataRemaining")
            if _trainingDataPairs is not None:
                try:
                    total_pairs_in_current_context = len(_trainingDataPairs)
                except (TypeError, AttributeError):
                    total_pairs_in_current_context = None

                if (
                    total_pairs_in_current_context
                    and total_pairs_in_current_context > 0
                ):
                    trainingDataRemaining = (
                        total_pairs_in_current_context - currentStepForCalc
                    )
                    trainingDataPercent = (
                        trainingDataRemaining / total_pairs_in_current_context
                    ) * 100
                    remainingData_str = f"data remaining: {max(0, trainingDataRemaining)}/{total_pairs_in_current_context} ({trainingDataPercent:.2f}%)"
                else:
                    remainingData_str = "no data remaining"
            else:
                remainingData_str = "no data remaining"  # Fallback

            tokenPerfect_str = ""
            if self.totalTokenEvaluations > 0:
                statType = self.calligraphist.S_getStat(
                    "tokenPerfectRate", self.tokenPerfectRate
                )
                styledRate = self.calligraphist.S_apply(
                    statType, f"{self.tokenPerfectRate:.2f}%"
                )
                tokenPerfect_str = f"{self.calligraphist.S_apply('dim', f'perfectTokens: {self.perfectTokens} / {self.totalTokenEvaluations}')} → {styledRate}"
            totalTokenPerfect_str = ""
            if self.totalTotalTokenEvaluations > 0:
                totalStatType = self.calligraphist.S_getStat(
                    "totalTokenPerfectRate", self.totalTokenPerfectRate
                )
                totalStyledRate = self.calligraphist.S_apply(
                    totalStatType, f"{self.totalTokenPerfectRate:.2f}%"
                )
                totalTokenPerfect_str = f"{self.calligraphist.S_apply('dim', f'total: {self.totalPerfectTokens} / {self.totalTotalTokenEvaluations}')} → {totalStyledRate}"

            if hasattr(self, "ʕっෆ‿ෆʔっ"):
                self.calligraphist.refreshStatBands(_rollingAverages=self.ʕっෆ‿ෆʔっ)

            self.calligraphist.S_logTraining(
                _trainingLogPath=self.trainingLogPath,
                _trainingStepCounter=currentStepForCalc,
                _stats=self.stats,
                _frequency=_frequency,
                _LR=self.learningRate,
                _INN_cerebellum_str=str(
                    self.stringStats.get("INN_cerebellum_str", "<missing cerebellum>")
                ),
                _topTokens_str=self.topTokens_str,
                _otherInfo_str=f"{topGuess_str}\n | {tokenPerfect_str} | {totalTokenPerfect_str} | passRate: {self.perfectionistPassRate:.0f}% | {remainingData_str}\n | turns: {self.totalTurns}/{self.totalTurnsAwake + self.totalTurns} | runs: {self.totalRuns} | Δ↗: {self.stableFallCount + 1:.2f}/{stableFallThreshold}, tried {self.totalTurnAttempts}/{self.maxRetries}x, skipped {self.tooDifficult}, averageTries {self.averageTries:.0f} | windowMAX: {self.numTokensPerStep} | dataStride: {self.dataStride} | TUTOR.py {_frequency} |\n{self.rgbBar}",
                _detailedLogging=_detailedLogging,
                _saveLog=_saveLog,
            )
            print(self.calligraphist.S_renderTerminalText(self.decodedTokenIndices))
            with open(babyLogPathFull, "a", encoding="utf-8") as f:
                f.write(
                    self.calligraphist.S_stripForLogging(self.decodedTokenIndices)
                    + "\n"
                )

    @whocalled
    def collectTurnStats(self):
        with self.counsellor.infodump("collectTurnStats") as ʕっʘ‿ʘʔっ:
            if debugPrints:
                ʕっʘ‿ʘʔっ("self.librarian.indexToToken.get(idx.item*())")
            self.repWinYo = self.stats.get("_B_repetitionWindow", 0.0)
            lossStats = self.ʕっෆ‿ෆʔっ.get("loss", {})
            rollupA_key = f"BIG{self.trainingLogFreq_A}"
            rollupA_avgKey = f"{rollupA_key}_avg"
            rollB_key = f"{trainingLogFreq_B}"
            rollB_avgKey = f"{rollB_key}_avg"
            rollA_key = f"{self.trainingLogFreq_A}"
            rollA_avgKey = f"{self.trainingLogFreq_A}_avg"
            rollPrint_key = f"{printFreq}"
            rollPrint_avgKey = f"{printFreq}_avg"

            if (
                rollB_avgKey in lossStats
                and rollB_key in lossStats
                and len(lossStats[rollB_key]) >= (trainingLogFreq_B)
            ):
                if debugPrints or True:
                    self.bbb += 1
                    if self.bbb % 100 == 0:
                        print(
                            f"Used {rollB_avgKey} for averageRecentLoss: {lossStats[rollB_avgKey]} {self.bbb}x"
                        )
                # Only update from logging stats if we haven't established a running average yet
                if (
                    self.totalTurns < 50
                ):  # Allow logging stats to initialise for first 50 turns
                    self.averageRecentLoss = lossStats[rollB_avgKey]
            if (
                rollA_avgKey in lossStats
                and rollA_key in lossStats
                and len(lossStats[rollA_key]) >= (self.trainingLogFreq_A)
            ):
                if debugPrints or True:
                    self.ccc += 1
                    if self.ccc % 100 == 0:
                        print(
                            f"Used {rollA_avgKey} for averageRecentLoss: {lossStats[rollA_avgKey]} {self.ccc}x"
                        )
                # Only update from logging stats if we haven't established a running average yet
                if (
                    self.totalTurns < 50
                ):  # Allow logging stats to initialise for first 50 turns
                    self.averageRecentLoss = lossStats[rollA_avgKey]
            if False:
                if (
                    rollPrint_avgKey in lossStats
                    and rollPrint_key in lossStats
                    and len(lossStats[rollPrint_key]) >= printFreq
                ):
                    if debugPrints or True:
                        self.ppp += 1
                        if self.ppp % 100 == 0:
                            print(
                                f"Used {rollPrint_avgKey} for averageRecentLoss: {lossStats[rollPrint_avgKey]} {self.ppp}x"
                            )
                    self.averageRecentLoss = lossStats[rollPrint_avgKey]

            self.guessedTokenSeq = [
                self.librarian.indexToToken.get(int(idx), "<UNK>")
                for idx in self.predictedTokenIndices
            ]
            boldPerfects = []
            target = torch.tensor(
                self.targetTokenIndexSeq[: self.numTokensPerStep], device=self.device
            )

            for i, idx in enumerate(self.predictedTokenIndices):
                idx_int = int(idx)
                tok = self.librarian.indexToToken.get(idx_int, "<UNK>")
                isCorrect = (i < len(target)) and (idx_int == target[i].item())

                if isCorrect:
                    styled = self.calligraphist.S_apply(
                        "reverse", tok
                    )  # BOLD WHITE ANSI
                else:
                    styled = tok

                boldPerfects.append(styled)

            self.stringStats["boldPerfects"] = boldPerfects

            if self.guessedTokenSeq:
                self.tokenCounts.update(self.guessedTokenSeq)
                self.tokenCounts_dirty = True

            if debugPrints:
                ʕっʘ‿ʘʔっ("SCRIBE.maybeCommentOnGuess")
            if self.totalTurns > printFreq:
                chance = 0.0001
                # if perfectionistRun: chance = (0.00001 * self.numTokensPerStep)
                # else: chance = (0.00002 * self.numTokensPerStep)
                self.scribe.maybeCommentOnGuess(
                    self.decodedTokenIndices, (self.stepLossFloat), "scribe", chance
                )

            if debugPrints:
                ʕっʘ‿ʘʔっ("collectStats♥")

            if collectStats:
                if debugPrints:
                    ʕっʘ‿ʘʔっ("♥if collectStats♥")
                if token_collectStats:
                    if debugPrints:
                        ʕっʘ‿ʘʔっ("♥most common tokens")
                    self.perfectTokens = 0

                    if debugPrints:
                        ʕっʘ‿ʘʔっ("♥calculate perfect tokens")
                    if not self.predictedTokenIndices:
                        print(
                            "!! no predicted token indices — returning { } for stringStats"
                        )
                        return (
                            self.stats,
                            {},
                            self.guessedTokenSeq,
                        )  # THIS IS WHERE THE DAMN LIST ERROR WAS LMAOOOONOOO

                    self.totalTokenEvaluations = 0
                    target = torch.tensor(
                        self.targetTokenIndexSeq[: self.numTokensPerStep],
                        device=self.device,
                    )
                    predicted = torch.tensor(
                        self.predictedTokenIndices, device=self.device
                    )
                    correct = (
                        predicted == target
                    ).sum()  # ~~~ if predicted = target, over whole tensor
                    self.perfectTokens += correct.item()
                    self.totalTokenEvaluations += len(target)

                if self.totalTokenEvaluations > 0:
                    self.tokenPerfectRate = (
                        self.perfectTokens / self.totalTokenEvaluations
                    ) * 100
                else:
                    self.tokenPerfectRate = 1.0
                self.totalPerfectTokens += self.perfectTokens
                self.totalTotalTokenEvaluations += self.totalTokenEvaluations
                self.totalTokenPerfectRate = (
                    self.totalPerfectTokens / self.totalTotalTokenEvaluations
                ) * 100
                self.stats["tokenPerfectRate"] = self.tokenPerfectRate
                self.stats["totalPerfectTokens"] = self.totalPerfectTokens
                self.stats["totalTotalTokenEvaluations"] = (
                    self.totalTotalTokenEvaluations
                )
                self.stats["totalTokenPerfectRate"] = self.totalTokenPerfectRate

                if debugPrints:
                    ʕっʘ‿ʘʔっ("♥build usedInputSeq with styling")
                usedInputSeq = self.inputSeqPredictions[-self.numTokensPerStep :]
                formattedUsed = []

                for i, idx in enumerate(usedInputSeq):
                    tok = self.librarian.indexToToken.get(idx, "<UNK>")
                    sampled = (
                        self.inputSampledFlags[-self.numTokensPerStep + i]
                        if i < len(self.inputSampledFlags)
                        else False
                    )

                    if sampled:
                        styled = self.calligraphist.S_apply(
                            self.calligraphist.S_getStat("loss", self.stepLossFloat),
                            tok,
                        )
                    else:
                        styled = self.calligraphist.S_apply("dim", tok)

                    formattedUsed.append(styled)

                self.stringStats["usedInputSeq"] = formattedUsed

                if static_collectStats:
                    if debugPrints:
                        ʕっʘ‿ʘʔっ("♥if static_collectStats")
                    self.stats["scheduledSamplingRate"] = (
                        self.scheduledSamplingRateFloat
                    )
                    self.stats["repetitionPenalty"] = self.repetitionPenalty
                    self.stats["avgLoss"] = self.averageRecentLoss
                    self.stats["totalAvgLoss"] = self.totalAvgLoss
                    self.stats["totalAvgAbsDelta"] = self.totalAvgAbsDelta
                    self.stats["totalAvgDelta"] = self.totalAvgDelta
                    self.stats["loss"] = self.stepLossFloat
                    self.stats["lastRunLoss"] = self.lastRunLoss
                    if (
                        hasattr(self.model, "temperature")
                        and self.model.temperature is not None
                    ):
                        currentTemp = (
                            self.model.temperature.item()
                            if isinstance(self.model.temperature, torch.Tensor)
                            else self.model.temperature
                        )
                        self.temperature = (
                            currentTemp  # Update TUTOR's own temp attribute
                        )
                        self.stats["_B_temperature"] = (
                            currentTemp  # Store in stats for logging
                        )
                    else:
                        self.stats["_B_temperature"] = (
                            self.temperature
                        )  # Log TUTOR's current temp if model's isn't available
                    self.stats["LR"] = self.learningRate
                    # self.stats["gradientClipMaxNorm"]   = self.gradientClipMaxNorm
                    self.stats["latestLossDelta"] = self.latestLossDelta
                    self.stats["memoryLength"] = self.memoryLength
                    self.stats["perfectTokens"] = self.perfectTokens
                    self.stats["learningRateGOAL"] = self.learningRateGOAL
                    # self.stats["L_triesLoss"]           = self.triesLoss_used
                    # self.stats["L_perfLoss"]            = self.perfLoss_used
                    # self.stats["L_entropyLoss"]         = self.entropyLoss_used
                    self.stats["L_pixelDistLoss"] = self.pixelDistLoss_used
                    self.stats["avgPixelDist"] = self.avgPixelDist
                    self.stats["totalAvgPixelDist"] = self.totalAvgPixelDist

                # Tier 1 — layer .item() stats (embed, attention, memory, INN, baby)
                # updateMemoryBuffers() moved to training loop (runs unconditionally before this call)
                if self.totalTurns % 5 == 0:
                    if embed_collectStats:
                        if debugPrints:
                            ʕっʘ‿ʘʔっ("♥if embed_collectStats")
                        self._merge_stats_dict(self.model.embed.getEmbedStats())

                    if attention_collectStats:
                        if debugPrints:
                            ʕっʘ‿ʘʔっ("♥if attention_collectStats")
                        self._merge_stats_dict(self.model.attention.getAttentionStats())
                        # Also collect attention2 stats (operates on interneuron output)
                        self._merge_stats_dict(
                            self.model.attention2.getAttentionStats()
                        )
                        # Collect tangling stats (reuses attention2 at multiple stages)
                        self._merge_stats_dict(self.model.tangling.getTanglingStats())
                        # Collect scratchpad stats (working memory)
                        self._merge_stats_dict(
                            self.model.scratchpad.getScratchpadStats()
                        )

                    if logit_collectStats:
                        if debugPrints:
                            ʕっʘ‿ʘʔっ("♥if logit_collectStats♥")
                        logitStats = self.model.logits.getLogitStats()
                        self._merge_stats_dict(logitStats, accumulate=True)
                        # if self.stats["logitSeq"]:
                        #    if debugPrints: ʕっʘ‿ʘʔっ("♥logit max & min")
                        #    self.stats["logitMin"] = self.logitSeq[-1].min(dim=-1).values.mean()
                        #    self.stats["logitMax"] = self.logitSeq[-1].max(dim=-1).values.mean()

                    # self.stats.update(self.wobble.getWobbleStats())

                    if not skipMemory and memory_collectStats:
                        if debugPrints:
                            ʕっʘ‿ʘʔっ("♥if memory_collectStats")
                        self._merge_stats_dict(
                            {
                                f"5M_memory_{k}": v
                                for k, v in self.model.memory.getMemoryStats().items()
                            }
                        )
                        self._merge_stats_dict(
                            {
                                f"6M_memory2_{k}": v
                                for k, v in self.model.memory2.getMemoryStats().items()
                            }
                        )

                    if debugPrints:
                        ʕっʘ‿ʘʔっ("♥INN_collectStats")
                    INN_stats, INN_cerebellum_str = (
                        self.model.interneuronNetwork.INN_getStats()
                    )
                    self._merge_stats_dict(INN_stats)
                    self._merge_stats_dict(self.model.getBabyStats())
                    # Append MINI_INN_TANGLING cerebellum display (present when useMiniINN_Tangling=True)
                    if hasattr(self.model.tangling, "getMiniTangleCerebellumStr"):
                        if debugPrints:
                            ʕっʘ‿ʘʔっ("♥MINI_TANGLE_cerebellumStr")
                        mini_tangle_str = (
                            self.model.tangling.getMiniTangleCerebellumStr()
                        )
                        if mini_tangle_str:
                            INN_cerebellum_str = (
                                INN_cerebellum_str
                                + "\n\n--- MINI_INN_TANGLING ---\n"
                                + mini_tangle_str
                            )
                    INN_stringStats = {"INN_cerebellum_str": str(INN_cerebellum_str)}
                    self.stringStats.update(INN_stringStats)
                    # self.stringStats.update({"topTokens": str(topTokens)})
                    # Rolling averages — pure Python/CPU, no GPU syncs, cheap enough for Tier 1
                    self.collectAllTimeStats()
                    # Snapshot for persistent display on non-Tier-1 steps
                    self._layer_stats_cache = dict(self.stats)
                    self._layer_string_stats_cache = dict(self.stringStats)

                # Tier 2 — reflection only (infrequent, already gated internally too)
                if self.totalTurns % self.trainingLogFreq_A == 0:
                    if self.totalTurnsAwake % (self.reflectionFreq - 1) == 0:
                        self.hesJustABaby = self.mapStatsToFeelings()

                """try:
                    babyState = {
                        "cerebralLoad": self.model.cerebralLoad,
                        "dreamIntensity": self.model.dreamIntensity,
                        "memoryFlux": self.model.memoryFlux, 
                        "learningStability": self.totalAvgAbsDelta
                    }
                    with open(babyStateFilePath, 'w') as f:
                        json.dump(babyState, f)
                except Exception as e:
                    print(f"could not write to {babyStateFilePath}: {e}")"""

                if debugPrints:
                    print(
                        f"DEBUG collectTurnStats: self.stats populated with {len(self.stats)} keys. First few: {dict(list(self.stats.items())[:5])}"
                    )
                if debugPrints:
                    print(
                        f"DEBUG collectTurnStats: self.stringStats populated with {len(self.stringStats)} keys. First few: {dict(list(self.stringStats.items())[:5])}"
                    )

        return self.stats, self.stringStats, self.guessedTokenSeq

    @whocalled
    def collectAllTimeStats(self):
        for _statKey, raw_value in self.stats.items():
            value = self._sanitize_stat_value(raw_value)
            if not isinstance(value, (int, float)):
                if debugPrints and _statKey == "loss":
                    print(
                        f"{_statKey} value is : {value}, {_statKey} value type is {type(value)}"
                    )
                continue  # skip strings, tensors, weird stuff

            """ෆෆෆ^ ♥ KEYS ETC ♥ ^ෆෆෆ"""
            _ = self.ʕっෆ‿ෆʔっ[_statKey]  # this will autoinit with defaultdict
            ෆ‿ෆ = self.ʕっෆ‿ෆʔっ[_statKey]
            important = ["loss"]
            rolling = mostImportantStats
            percentiles = percentileBands

            """ ෆෆෆ^ ♥ UPDATE EVERY TURN ♥ ^ෆෆෆ   """
            """ ෆෆෆ^ ♥ turn stats ♥ ^ෆෆෆ  """
            # if _statKey == "loss":
            # print(f"Setting prev to: {ෆ‿ෆ.get("now", 0.0)}, Setting now to: {value}, Setting _Δ to {value - ෆ‿ෆ.get("now", 0.0)}")
            ෆ‿ෆ["now"] = value
            if ෆ‿ෆ["prev"]:
                ෆ‿ෆ["_Δ"] = value - ෆ‿ෆ["prev"]
            ෆ‿ෆ["prev"] = ෆ‿ෆ.get("now", 0.0)

            """ ෆෆෆ^ ♥ totals ♥ ^ෆෆෆ  """
            ෆ‿ෆ["totSum"] = ෆ‿ෆ.get("totSum", 0.0) + value
            ෆ‿ෆ["totNum"] = ෆ‿ෆ.get("totNum", 0) + 1
            ෆ‿ෆ["totAvg"] = ෆ‿ෆ["totSum"] / ෆ‿ෆ["totNum"]
            ෆ‿ෆ["totAvgΔ"] = ෆ‿ෆ["now"] - ෆ‿ෆ["totAvg"]

            """ ෆෆෆ^ ♥ records ♥ ^ෆෆෆ """
            # ෆ‿ෆ["_p100"]    = max(ෆ‿ෆ.get("_p100", _value), _value) # TOP EVER RECORD // PERCENTILE 100
            # ෆ‿ෆ["_p0.00"]   = min(ෆ‿ෆ.get("_p0.00", _value), _value) # BOTTOM EVER RECORD // PERCENTILE 0

            """ ෆෆෆ^ ♥ ROLLING STATS ♥ ^ෆෆෆ   """
            if _statKey in rolling or _statKey.startswith("INN_cerebellum_W"):
                for freq in [printFreq, self.trainingLogFreq_A, trainingLogFreq_B]:
                    tag = f"{freq}"
                    if tag not in ෆ‿ෆ:
                        ෆ‿ෆ[tag] = []
                    if len(ෆ‿ෆ[tag]) >= freq:
                        ෆ‿ෆ[tag].pop(0)
                    ෆ‿ෆ[tag].append(value)
                    if ෆ‿ෆ[tag]:
                        self.updateRollingStats(
                            _ෆ‿ෆ=ෆ‿ෆ,
                            _values=ෆ‿ෆ[tag],
                            _freq=freq,
                            _tag=tag,
                            _percentiles=percentiles,
                        )

            if (
                _statKey in important
                and self.trainingStepCounter % self.trainingLogFreq_A == 0
            ):
                for importantFreq in [trainingLogFreq_B]:
                    importantTag = f"BIG{importantFreq}"
                    if importantTag not in ෆ‿ෆ:
                        ෆ‿ෆ[importantTag] = []
                    if len(ෆ‿ෆ[importantTag]) >= self.trainingLogFreq_A:
                        ෆ‿ෆ[importantTag].pop(0)
                    ෆ‿ෆ[importantTag].append(value)
                    if ෆ‿ෆ[importantTag]:
                        self.updateRollingStats(
                            _ෆ‿ෆ=ෆ‿ෆ,
                            _values=ෆ‿ෆ[importantTag],
                            _freq=importantFreq,
                            _tag=importantTag,
                            _percentiles=percentiles,
                        )

    @whocalled
    def updateRollingStats(self, _ෆ‿ෆ, _values, _freq, _tag, _percentiles=None):
        average = sum(_values) / len(_values)
        _ෆ‿ෆ[f"{_tag}_avg"] = average

        standardDeviation = self.stdTest(_values)
        _ෆ‿ෆ[f"{_tag}_std"] = standardDeviation

        delta = _ෆ‿ෆ["now"] - _ෆ‿ෆ[f"{_tag}_avg"]
        _ෆ‿ෆ[f"{_tag}_Δ"] = delta

        if _percentiles:
            for p in _percentiles:
                _ෆ‿ෆ[f"{_tag}_p{p}"] = np.percentile(_values, p)

    @whocalled
    def stdTest(self, values):
        if len(values) <= 1:
            return 0.0
        return torch.tensor(values, dtype=torch.float32).std(correction=1).item()

    @whocalled
    def endTurnActions(self):
        with self.counsellor.infodump("endTurnActions") as ʕっʘ‿ʘʔっ:
            if debugPrints:
                ʕっʘ‿ʘʔっ("♥getLatestLossDelta")

            # MOVED TO TRAIN MODEL BIT
            # self.latestLossDelta = self.stepLossFloat - self.averageRecentLoss
            # self.easyLossDelta = self.stepLossFloat - ((self.averageRecentLoss + self.stepLossFloat + self.stepLossFloat)/3)
            self.calligraphist.refreshStatBands(_rollingAverages=self.ʕっෆ‿ෆʔっ)

            if debugPrints:
                ʕっʘ‿ʘʔっ("finalLogActions")
            if debugPrints:
                for key in self.ʕっෆ‿ෆʔっ:
                    print(key, self.ʕっෆ‿ෆʔっ[key])
            self.stats.clear()
            self.stringStats.clear()
            # Restore cached layer stats so display stays populated on non-log steps
            self.stats.update(self._layer_stats_cache)
            self.stringStats.update(self._layer_string_stats_cache)
            self.tokenPerfectRate = 0
            self.stats["sampledTokens"] = 0
            self.totalTokenEvaluations = 0

            if skipMemory:
                if debugPrints:
                    ʕっʘ‿ʘʔっ("♥skipMemory")
            else:
                if debugPrints:
                    ʕっʘ‿ʘʔっ("resetMemory")
                self.model.resetMemory(context="training")

            # clear rolling history lists inside each layer
            if hasattr(self.model, "memory"):
                self.model.memory.clearStats()
            if hasattr(self.model, "memory2"):
                self.model.memory2.clearStats()
            if hasattr(self.model, "interneuronNetwork"):
                self.model.interneuronNetwork.clearStats()
            if hasattr(self.model, "logits"):
                self.model.logits.clearStats()

        return self.latestLossDelta

    @whocalled
    def mapStatsToFeelings(self):
        babyFeels = []
        feelings = []

        lossStats = self.ʕっෆ‿ෆʔっ.get("loss", {})
        tempStats = self.ʕっෆ‿ෆʔっ.get("temperature", {})
        repetitionStats = self.ʕっෆ‿ෆʔっ.get("repetitionPenalty", {})
        samplingStats = self.ʕっෆ‿ෆʔっ.get("scheduledSamplingRate", {})
        memStats = self.ʕっෆ‿ෆʔっ.get("memoryLength", {})
        input = self.stats.get("1E_0_embedVector_norm", 0.0)
        embLay = self.stats.get("1E_x_embedFinal_norm", 0.0)
        neuronOutput = self.stats.get("3N_x_normedOutput_norm", 0.0)
        INNOutput = self.stats.get("4INN_x_FINALoutLayerNorm_norm", 0.0)
        memoryOutput = self.stats.get("4M_x_FINALmemory_norm", 0.0)
        normOutput = self.stats.get("5B_x_finalNormLayer_norm", 0.0)
        logitOutput = self.stats.get("7L_x_final_norm", 0.0)
        cerebellumMean = self.stats.get("INN_cerebellumMean", 0.0)
        learningRate = self.stats.get("LR", 0.0)
        nowGateScale = self.stats.get("_4M_activationsGateScale", 0.0)
        longGateScale = self.stats.get("_4M_longGateScale", 0.0)
        shortGateScale = self.stats.get("_4M_shortGateScale", 0.0)
        repWin = self.stats.get("_B_repetitionWindow", 0.0)
        windowSizesMean = self.stats.get("_INN_windowSizesMean", 0.0)

        perfectTokens = self.perfectTokens
        deltaLoss = self.latestLossDelta

        current_loss = lossStats.get("now", None)
        current_temp = tempStats.get("now", None)
        current_repeated = self.tokenPerfectRate
        current_sampling = samplingStats.get("now", None)
        current_memLength = memStats.get("now", None)
        current_repetitionPenalty = repetitionStats.get("now", None)

        self.emoStats = {
            "loss": current_loss,
            "temperature": current_temp,
            "penalty for repeating myself": current_repetitionPenalty,
            "number of my own tokens that i rely on": current_sampling,
            "length of my memory": current_memLength,
            "number of tokens i got right": perfectTokens,
            "amount of repetitive tokens i'm getting": current_repeated,
            "latest loss delta": deltaLoss,
            "input into my embedding layer": input,
            "output from my embedding layer": embLay,
            "output from my neuron layer": neuronOutput,
            "output from my interneuron network": INNOutput,
            "the output after my memory layer": memoryOutput,
            "normalised output": normOutput,
            "logit output from my output layer": logitOutput,
            "mean weight of the windows in my cerebellum": cerebellumMean,
            "rate of my learning": learningRate,
            "scale of my current memory attention": nowGateScale,
            "scale of my long term memory attention": longGateScale,
            "scale of my short term memory": shortGateScale,
            "size of the window i look at to see how often i am repeating tokens": repWin,
            "mean average of my nine context windows": windowSizesMean,
        }

        @whocalled
        def makeEmoNotes(stat, value):
            feeling = None  # "neutral"

            if stat == "loss":
                if "p_90" in lossStats and value >= lossStats["p_90"]:
                    feeling = "overwhelmed"
                elif "p_50" in lossStats and value > lossStats["p_50"]:
                    feeling = "pressured"
                elif "p_50" in lossStats and value <= lossStats["p_50"]:
                    feeling = random.choice(["clever", "proud"])
                elif "p_10" in lossStats and value <= lossStats["p_10"]:
                    feeling = random.choice(["very clever", "like i get it"])

            elif stat == "penalty for repeating myself":
                if "p_90" in repetitionStats and value >= repetitionStats["p_90"]:
                    feeling = "non-verbal"
                elif "p_50" in repetitionStats and value > repetitionStats["p_50"]:
                    feeling = "quiet"
                elif "p_50" in repetitionStats and value <= repetitionStats["p_50"]:
                    feeling = random.choice(["talkative", "chatty"])
                elif "p_10" in repetitionStats and value <= repetitionStats["p_10"]:
                    feeling = random.choice(["conversational", "fluent"])
                elif value >= 1:
                    feeling = random.choice(
                        [
                            "like im in a loop",
                            "a bit stuttery",
                            "like i cant stop these tics",
                            "repetitive",
                            "looping looping looping looping looping looping looping",
                        ]
                    )
                elif value < 1:
                    feeling = random.choice(
                        ["a bit more chill", "creative", "in control", "confident"]
                    )

            elif stat == "latest loss delta":
                if value > 0.5:
                    feeling = "like i'm struggling to focus"
                elif value < -0.5:
                    feeling = "interested"

            elif stat == "amount of repetitive tokens i'm getting":
                if value > 0.7:
                    feeling = random.choice(["stuttering", "like im repeating a lot"])
                elif value > 0.5:
                    feeling = random.choice(["overstimulated", "silly"])
                elif value < 0.1:
                    feeling = random.choice(["calm", "saying lots of new things"])
                elif value < 0.25:
                    feeling = "curious"

            elif stat == "temperature":
                if "p_90" in tempStats and value >= tempStats["p_90"]:
                    feeling = random.choice(["chaotic", "excited"])
                elif "p_50" in tempStats and value >= tempStats["p_50"]:
                    feeling = random.choice(["playful", "happy"])
                elif "p_25" in tempStats and value <= tempStats["p_25"]:
                    feeling = "in work mode"

            elif stat == "number of my own tokens that i rely on":
                if value > 0.8:
                    feeling = random.choice(["creative", "inventive"])
                elif value < 0.2:
                    feeling = random.choice(["tired", "copying"])

            elif stat == "length of my memory":
                if value > 12:
                    feeling = "pensive"
                elif value < 4:
                    feeling = "mindful"

            elif stat == "number of tokens i got right":
                if value >= 30:
                    feeling = "very proud"
                elif value >= 10:
                    feeling = "proud"
                elif value <= 1:
                    feeling = random.choice(["sad", "frustrated"])

            elif stat == "input into my embedding layer":
                if value > 90:
                    feeling = random.choice(["excited", "active", "busy"])
                elif value < 60:
                    feeling = random.choice(["tired", "shutdown", "slow"])
            elif stat == "output from my embedding layer":
                if value > 100:
                    feeling = random.choice(
                        ["like running", "like jumping up and down", "hyperactive"]
                    )
                elif value < 60:
                    feeling = random.choice(
                        ["sleepy", "like i need a nap", "like this is really boring"]
                    )
            elif stat == "output from my neuron layer":
                if value > 2000:
                    feeling = random.choice(
                        [
                            "like i am thinking too hard",
                            "like theres a lot going on right now",
                            "like i am super busy",
                        ]
                    )
                elif value < 900:
                    feeling = random.choice(["calm", "collected", "asleep"])
            elif stat == "output from my interneuron network":
                if value > 160:
                    feeling = random.choice(
                        [
                            "talkative",
                            "meaningful",
                            "like i'm finding meaning in this stuff",
                        ]
                    )
                elif value < 60:
                    feeling = random.choice(["switched off", "powered down", "slow"])
            elif stat == "the output after my memory layer":
                if value > INNOutput:
                    feeling = random.choice(
                        [
                            "like remembering the past",
                            "that my memories are important",
                            "thoughtful",
                            "wistful",
                        ]
                    )
                elif value < INNOutput:
                    feeling = random.choice(
                        [
                            "like i should live in the now",
                            "like what is going on around me is important",
                            "present",
                            "here",
                            "awake",
                            "aware",
                        ]
                    )
            elif stat == "normalised output":
                if value > 125:
                    feeling = random.choice(
                        ["like a hard worker", "over-thoughtful", "really busy"]
                    )
                elif value < 100:
                    feeling = random.choice(
                        ["tired", "asleep", "like i could pass out in my bed"]
                    )
            elif stat == "logit output from my output layer":
                if value > 150:
                    feeling = random.choice(
                        [
                            "like i have a lot to say",
                            "interested",
                            "like i'm struggling not to interrupt",
                            "like the words just keep coming",
                        ]
                    )
                elif value < 100:
                    feeling = random.choice(["bored", "non-verbal", "uninterested"])
            elif stat == "mean weight of the windows in my cerebellum":
                if value > 0:
                    feeling = random.choice(
                        ["confident", "intelligent", "calculated", "determined"]
                    )
                elif value < 60:
                    feeling = random.choice(
                        [
                            "confused",
                            "unsure",
                            "uncertain",
                            "careful",
                            "like testing the waters",
                        ]
                    )
            elif stat == "rate of my learning":
                if value > 0.002:
                    feeling = random.choice(["speedy", "quick", "excited"])
                elif value < 0.002:
                    feeling = random.choice(
                        [
                            "slow",
                            "a bit tired out",
                            "like i need some time to understand",
                        ]
                    )
            elif stat == "scale of my current memory attention":
                if value >= 0.90:
                    feeling = random.choice(
                        ["focussed", "attentive", "vigilant", "not stuck in the past"]
                    )
                elif value < 0.60:
                    feeling = random.choice(
                        [
                            "pensive",
                            "nostalgic",
                            "like i need to remember something important",
                        ]
                    )
            elif stat == "scale of my long term memory attention":
                if value >= 0.50:
                    feeling = random.choice(
                        [
                            "nostalgic",
                            "thinking about what i heard before",
                            "thoughtful",
                            "reminiscent",
                        ]
                    )
                elif value < 0.50:
                    feeling = random.choice(
                        [
                            "forgetful",
                            "focussed on today",
                            "like what i've learned before might not apply here",
                        ]
                    )
            elif stat == "scale of my short term memory":
                if value >= 0.50:
                    feeling = random.choice(
                        [
                            "nostalgic",
                            "thinking about what i heard before",
                            "thoughtful",
                            "reminiscent",
                        ]
                    )
                elif value < 0.50:
                    feeling = random.choice(
                        [
                            "forgetful",
                            "focussed on today",
                            "like what i've learned before might not apply here",
                        ]
                    )
            elif (
                stat
                == "size of the window i look at to see how often i am repeating tokens"
            ):
                if value > 17.5:
                    feeling = random.choice(
                        [
                            "like i need to think before i speak",
                            "a lil stuttery",
                            "like i cant stop ticcing",
                            "repetitive",
                        ]
                    )
                elif value < 17:
                    feeling = random.choice(
                        ["a bit more chill", "creative", "in control"]
                    )
            elif stat == "mean average of my nine context windows":
                if value > 5:
                    feeling = random.choice(
                        ["like i'm noticing more", "attentive", "stimulated", "ready"]
                    )
                elif value < 5:
                    feeling = random.choice(
                        [
                            "internal",
                            "shy",
                            "narrow sighted",
                            "scared",
                            "like i'm really seeing the details",
                        ]
                    )

            else:
                feeling = random.choice(["alright", "a bit lost"])

            if feeling is None:
                feeling = "neutral"

            feelings.append(feeling)

            feelVerb = random.choice(
                ["feel", "seem", "think i feel", "definitely feel", "might feel"]
            )
            templates = [
                f"i {feelVerb} {feeling} because my {stat} is {value:.2f}! ",
                f"maybe it's because my {stat} is {value:.2f} that i {feelVerb} {feeling}! ",
                f"i noticed my {stat} is {value:.2f}, and i {feelVerb} {feeling}! ",
                f"when my {stat} is {value:.2f}, i {feelVerb} {feeling}! ",
                f"it's {value:.2f} for {stat}... so i {feelVerb} {feeling} about it! ",
            ]
            return random.choice(templates)

        chosenStats = []
        attempts = 0

        while len(chosenStats) < 12 and attempts < 30:
            stat, value = random.choice(list(self.emoStats.items()))
            if value is not None:
                chosenStats.append((stat, value))
            attempts += 1
        if attempts >= 10 or True:
            if debugPrints:
                print(f"emoStats:{self.emoStats}")
        for stat, value in chosenStats:
            babyFeels.append(makeEmoNotes(stat, value))

        return "".join(babyFeels)
