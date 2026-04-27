# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ ---
# MEMORY LAYER // brain/LAYERS/memory.py
# v1.2

from collections import deque

import torch
import torch.nn as nn

from config import *

"""this makes a rolling buffer of past activations"""


class MEMORY(nn.Module):
    def __init__(self, _counsellor, _numTokensPerStep, _device=modelDevice):
        super().__init__()
        self.device = _device
        self.counsellor = _counsellor
        self.numTokensPerStep = _numTokensPerStep

        # learnable decay rates and gates
        self.shortTermDecay = nn.Parameter(torch.tensor(0.7, device=self.device))
        self.longTermDecay = nn.Parameter(torch.tensor(0.95, device=self.device))

        self.inputReducer = nn.Linear(
            numNeurons, embedDimension, device=self.device
        )  # 10k → 1k
        self.gateLayer2 = nn.Linear(
            embedDimension, 4 * numNeurons, device=self.device
        )  # 1k → 40k

        self.memoryProjector = nn.Linear(numNeurons, embedDimension, device=self.device)
        self.memoryInfluence2 = torch.nn.Sequential(
            nn.Linear(embedDimension, 512, device=self.device),  # bottleneck layer
            nn.GELU(),  # smoother activation
            nn.LayerNorm(512, device=self.device),  # mid normalisation
            nn.Linear(512, numNeurons, device=self.device),  # expand back
            nn.LayerNorm(numNeurons, device=self.device),  # final safety net
        )

        # Output normalization and gating (for safe integration with rest of architecture)
        self.output_norm = nn.LayerNorm(numNeurons, device=self.device)
        gate_init = (
            -4.0
        )  # sigmoid(-4.0) ≈ 0.018, starts small to preserve existing training
        self.output_gate = nn.Parameter(torch.tensor(gate_init, device=self.device))

        # buffers to store memory (outside gradient)
        self.register_buffer(
            "shortTermMemory", torch.zeros(1, numNeurons, device=self.device)
        )
        self.register_buffer(
            "longTermMemory", torch.zeros(1, numNeurons, device=self.device)
        )

        # stats
        self.stats = {}
        self._history_attrs = [
            "shortGateScaleHistory",
            "longGateScaleHistory",
            "activationsGateScaleHistory",
            "gateLayer2History",
            "gateLayer2MaxHistory",
            "gateLayer2MinHistory",
            "reducedInputHistory",
            "reducedInputMaxHistory",
            "reducedInputMinHistory",
            "rawActivationsHistory",
            "rawActivationsMaxHistory",
            "rawActivationsMinHistory",
            "shortTermMemoryHistory",
            "shortTermMemoryMaxHistory",
            "shortTermMemoryMinHistory",
            "longTermMemoryHistory",
            "longTermMemoryMaxHistory",
            "longTermMemoryMinHistory",
            "FINALmemoryHistory",
            "FINALmemoryMaxHistory",
            "FINALmemoryMinHistory",
            "memGateScaleHistory",
            "projectedMemoryHistory",
            "projectedMemoryMaxHistory",
            "projectedMemoryMinHistory",
            "memoryGateHistory",
            "memoryGateMaxHistory",
            "memoryGateMinHistory",
            "mixedEmbedHistory",
            "mixedEmbedMaxHistory",
            "mixedEmbedMinHistory",
            "gateLayer2NormHistory",
            "reducedInputNormHistory",
            "rawActivationsNormHistory",
            "shortTermMemoryNormHistory",
            "longTermMemoryNormHistory",
            "FINALmemoryNormHistory",
            "projectedMemoryNormHistory",
            "memoryGateNormHistory",
            "mixedEmbedNormHistory",
        ]
        self._init_history_buffers()

        self.register_buffer(
            "reducedInputBuf", torch.zeros(1, embedDimension, device=self.device)
        )
        self.register_buffer(
            "gateLogitsBuf", torch.zeros(4, numNeurons, device=self.device)
        )

    def _init_history_buffers(self):
        maxlen = max(1, self.numTokensPerStep)
        for attr in self._history_attrs:
            setattr(self, attr, deque(maxlen=maxlen))

    def _reset_history_buffers(self):
        """Clear history buffers once stats have been aggregated."""
        for attr in self._history_attrs:
            getattr(self, attr).clear()

    def _history_mean(self, history, offset=0.0):
        if not history:
            return offset
        tensor = torch.as_tensor(history, dtype=torch.float32, device=self.device)
        return tensor.mean().item() + offset

    @whocalled
    def forward(self, _activationsTensor):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            self.activationsTensor = _activationsTensor

            if debugPrints:
                ʕっʘ‿ʘʔっ("shortTermDecay")
            shortDecay = torch.sigmoid(self.shortTermDecay)
            with torch.no_grad():
                self.shortTermDecay.clamp_(
                    -5, 5
                )  # keeps sigmoid ~[0.0067, 0.9933], so memory doesnt vanish or freeze forever
            if debugPrints:
                ʕっʘ‿ʘʔっ("longTermDecay")
            longDecay = torch.sigmoid(self.longTermDecay)
            with torch.no_grad():
                self.longTermDecay.clamp_(
                    -5, 5
                )  # prevents total decay collapse (was the root cause of the Feb 26 NaN event)

            if debugPrints:
                ʕっʘ‿ʘʔっ("newShortTermMemory")
            # Clamp input activations to prevent unbounded accumulation.
            # NOTE: must check isfinite() FIRST — NaN > 1000.0 == False, so the norm
            # check alone silently passes NaN straight through.
            safe_activations = _activationsTensor
            if not torch.isfinite(safe_activations).all():
                safe_activations = torch.zeros_like(
                    safe_activations
                )  # NaN input → zero memory update
            else:
                act_norm = safe_activations.norm()
                if act_norm > 1000.0:  # Emergency clamp if activations are exploding
                    safe_activations = safe_activations / act_norm * 1000.0

            newShort = (shortDecay * self.shortTermMemory) + (
                (1 - shortDecay) * safe_activations
            )
            if debugPrints:
                ʕっʘ‿ʘʔっ("newLongTermMemory")
            newLong = (longDecay * self.longTermMemory) + (
                (1 - longDecay) * safe_activations
            )

            if debugPrints:
                ʕっʘ‿ʘʔっ("self.inputReducer")
            reducedInput = self.inputReducer(_activationsTensor)
            # self.reducedInputBuf.copy_(reducedInput.detach())  # ????? SUS ?? store for stats without affecting autograd

            # unified gate logits -> shape: [1, 4 * numNeurons]
            if debugPrints:
                ʕっʘ‿ʘʔっ("self.gateLater2")
            # gateLogits = self.gateLayer2(reducedInput)
            gateLogits = self.gateLayer2(reducedInput).squeeze(
                0
            )  # removes batch dim, gives [40000]

            # check before reshaping
            if gateLogits.shape[0] != 4 * numNeurons:
                raise ValueError(
                    f"gateLogits shape mismatch: expected {(4 * numNeurons,)}, got {gateLogits.shape}"
                )

            gateLogits = gateLogits.view(4, numNeurons)
            # self.gateLogitsBuf.copy_(gateLogits.detach())  # ??? SUS ??? keep detached version for logging
            if debugPrints:
                ʕっʘ‿ʘʔっ("clamp gatelayer2 -> gateLogits")
            # clamp extreme values while preserving gradient flow
            gateLogits = gateLogits.clamp(-30, 30)

            # softmax across sources (dim=0), sum to 1 per neuron
            if debugPrints:
                ʕっʘ‿ʘʔっ("softmax gateLogits")
            gateWeights = torch.softmax(gateLogits, dim=0)

            try:
                shortGateScale, longGateScale, actGateScale, memGateScale = (
                    gateWeights.unbind()
                )
            except Exception as e:
                raise RuntimeError(f"unpacc gate weights failed: {gateWeights}") from e

            # shortGateScale, longGateScale, actGateScale, memGateScale = gateWeights

            if debugPrints:
                ʕっʘ‿ʘʔっ("firstGatedMemory")
            firstGatedMemory = (
                (shortGateScale * newShort)
                + (longGateScale * newLong)
                + (actGateScale * _activationsTensor)
            )

            if debugPrints:
                ʕっʘ‿ʘʔっ("self.memoryProjector")
            projectedMemory = self.memoryProjector(firstGatedMemory)
            if debugPrints:
                ʕっʘ‿ʘʔっ("mix embeds")
            # mixedEmbed = self.reducedInputBuf + projectedMemory
            mixedEmbed = reducedInput + projectedMemory
            if debugPrints:
                ʕっʘ‿ʘʔっ("self.memoryInfluence2")
            memoryGate = self.memoryInfluence2(mixedEmbed)

            if debugPrints:
                ʕっʘ‿ʘʔっ("self.gatedMemory")
            self.gatedMemory = (
                (shortGateScale * newShort)
                + (longGateScale * newLong)
                + (actGateScale * _activationsTensor)
                + (memGateScale * memoryGate)
            )

            # Apply output normalization and gating with residual connection
            if debugPrints:
                ʕっʘ‿ʘʔっ("output norm + gate + residual")
            normalized_memory = self.output_norm(self.gatedMemory)
            gated = torch.sigmoid(self.output_gate) * normalized_memory
            self.FINALmemory = self.gatedMemory + gated  # residual connection!

            if debugPrints:
                ʕっʘ‿ʘʔっ("shortGateScale stats")
            _gate_stats = torch.stack(
                [
                    shortGateScale.mean(),
                    longGateScale.mean(),
                    actGateScale.mean(),
                    memGateScale.mean(),
                ]
            ).tolist()
            self.short_used = _gate_stats[0]
            self.shortGateScaleHistory.append(_gate_stats[0])  # 1
            self.shortDecay_used = shortDecay.detach().item()
            self.long_used = _gate_stats[1]
            self.longGateScaleHistory.append(_gate_stats[1])  # 2
            self.longDecay_used = longDecay.detach().item()
            self.act_used = _gate_stats[2]
            self.activationsGateScaleHistory.append(_gate_stats[2])  # 0
            self.mem_used = _gate_stats[3]
            self.memGateScaleHistory.append(_gate_stats[3])  # 7

            if debugPrints:
                ʕっʘ‿ʘʔっ("raw activation stats")
            _raw_stats = torch.stack(
                [
                    _activationsTensor.norm(),
                    _activationsTensor.mean(),
                    _activationsTensor.max(),
                    _activationsTensor.min(),
                ]
            ).tolist()
            self.rawActivationsNormHistory.append(_raw_stats[0])  # 0
            self.rawActivationsHistory.append(_raw_stats[1])  # 0
            self.rawActivationsMaxHistory.append(_raw_stats[2])  # 0
            self.rawActivationsMinHistory.append(_raw_stats[3])  # 0

            """if debugPrints: ʕっʘ‿ʘʔっ("STM stats")
            self.shortTermMemoryNormHistory.append(self.shortTermMemory.norm().item()) # 1
            self.shortTermMemoryHistory.append(self.shortTermMemory.mean().item()) # 1
            self.shortTermMemoryMaxHistory.append(self.shortTermMemory.max().item()) # 1
            self.shortTermMemoryMinHistory.append(self.shortTermMemory.min().item()) # 1

            if debugPrints: ʕっʘ‿ʘʔっ("LTM stats")
            self.longTermMemoryNormHistory.append(self.longTermMemory.norm().item()) # 2
            self.longTermMemoryHistory.append(self.longTermMemory.mean().item()) # 2
            self.longTermMemoryMaxHistory.append(self.longTermMemory.max().item()) # 2
            self.longTermMemoryMinHistory.append(self.longTermMemory.min().item()) # 2

            if debugPrints: ʕっʘ‿ʘʔっ("reduced Input stats")
            self.reducedInputNormHistory.append(self.reducedInputBuf.norm().item()) # 3
            self.reducedInputHistory.append(self.reducedInputBuf.mean().item()) # 3
            self.reducedInputMaxHistory.append(self.reducedInputBuf.max().item()) # 3
            self.reducedInputMinHistory.append(self.reducedInputBuf.min().item()) # 3

            if debugPrints: ʕっʘ‿ʘʔっ("gate layer 2 stats")
            self.gateLayer2NormHistory.append(self.gateLayer2.weight.norm().item()) # 4
            self.gateLayer2History.append(self.gateLayer2.weight.mean().item()) # 4
            self.gateLayer2MaxHistory.append(self.gateLayer2.weight.max().item()) # 4
            self.gateLayer2MinHistory.append(self.gateLayer2.weight.min().item()) # 4

            if debugPrints: ʕっʘ‿ʘʔっ("projected memory stats")
            self.projectedMemoryNormHistory.append(projectedMemory.norm().item()) # 5
            self.projectedMemoryHistory.append(projectedMemory.mean().item()) # 5
            self.projectedMemoryMaxHistory.append(projectedMemory.max().item()) # 5
            self.projectedMemoryMinHistory.append(projectedMemory.min().item()) # 5

            if debugPrints: ʕっʘ‿ʘʔっ("mixed embed stats")
            self.mixedEmbedNormHistory.append(mixedEmbed.norm().item()) # 6
            self.mixedEmbedHistory.append(mixedEmbed.mean().item()) # 6
            self.mixedEmbedMaxHistory.append(mixedEmbed.max().item()) # 6
            self.mixedEmbedMinHistory.append(mixedEmbed.min().item()) # 6

            if debugPrints: ʕっʘ‿ʘʔっ("memory gate stats")
            self.memoryGateNormHistory.append(memoryGate.norm().item()) # 7
            self.memoryGateHistory.append(memoryGate.mean().item()) # 7
            self.memoryGateMaxHistory.append(memoryGate.max().item()) # 7
            self.memoryGateMinHistory.append(memoryGate.min().item()) # 7"""

            if debugPrints:
                ʕっʘ‿ʘʔっ("final memory stats")
            _fin_stats = torch.stack(
                [
                    self.FINALmemory.norm(),
                    self.FINALmemory.mean(),
                    self.FINALmemory.max(),
                    self.FINALmemory.min(),
                ]
            ).tolist()
            self.FINALmemoryNormHistory.append(_fin_stats[0])
            self.FINALmemoryHistory.append(_fin_stats[1])
            self.FINALmemoryMaxHistory.append(_fin_stats[2])
            self.FINALmemoryMinHistory.append(_fin_stats[3])

            if len(self.shortGateScaleHistory) >= self.numTokensPerStep:
                if debugPrints:
                    ʕっʘ‿ʘʔっ(
                        "updateStats if short gate scale history >= self.numTokensPerStep"
                    )
                statShortDecay = torch.sigmoid(self.shortTermDecay).item()
                statLongDecay = torch.sigmoid(self.longTermDecay).item()
                _flush_means = (
                    torch.stack(
                        [
                            torch.as_tensor(
                                list(self.rawActivationsNormHistory),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            torch.as_tensor(
                                list(self.rawActivationsHistory),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            torch.as_tensor(
                                list(self.rawActivationsMaxHistory),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            torch.as_tensor(
                                list(self.rawActivationsMinHistory),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            torch.as_tensor(
                                list(self.FINALmemoryNormHistory),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            torch.as_tensor(
                                list(self.FINALmemoryHistory),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            torch.as_tensor(
                                list(self.FINALmemoryMaxHistory),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            torch.as_tensor(
                                list(self.FINALmemoryMinHistory),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            torch.as_tensor(
                                list(self.shortGateScaleHistory),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            torch.as_tensor(
                                list(self.longGateScaleHistory),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            torch.as_tensor(
                                list(self.activationsGateScaleHistory),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            torch.as_tensor(
                                list(self.memGateScaleHistory),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                        ]
                    )
                    .mean(dim=1)
                    .add_(0.001)
                    .tolist()
                )
                (
                    raw_norm,
                    raw_mean,
                    raw_max,
                    raw_min,
                    final_norm,
                    final_mean,
                    final_max,
                    final_min,
                    short_gate,
                    long_gate,
                    act_gate,
                    mem_gate,
                ) = _flush_means
                self.stats = {
                    "4M_0_rawActs_norm": raw_norm,
                    "4M_0_rawActs_mean": raw_mean,
                    "4M_0_rawActs_max": raw_max,
                    "4M_0_rawActs_min": raw_min,
                    # "4M_1_STM_norm": sum(self.shortTermMemoryNormHistory) / len(self.shortTermMemoryNormHistory),
                    # "4M_1_STM_mean": sum(self.shortTermMemoryHistory) / len(self.shortTermMemoryHistory),
                    # "4M_1_STM_max": sum(self.shortTermMemoryMaxHistory) / len(self.shortTermMemoryMaxHistory),
                    # "4M_1_STM_min": sum(self.shortTermMemoryMinHistory) / len(self.shortTermMemoryMinHistory),
                    # "4M_2_LTM_norm": sum(self.longTermMemoryNormHistory) / len(self.longTermMemoryNormHistory),
                    # "4M_2_LTM_mean": sum(self.longTermMemoryHistory) / len(self.longTermMemoryHistory),
                    # "4M_2_LTM_max": sum(self.longTermMemoryMaxHistory) / len(self.longTermMemoryMaxHistory),
                    # "4M_2_LTM_min": sum(self.longTermMemoryMinHistory) / len(self.longTermMemoryMinHistory),
                    # "4M_3_reducedInput_norm": sum(self.reducedInputNormHistory) / len(self.reducedInputNormHistory),
                    # "4M_3_reducedInput_mean": sum(self.reducedInputHistory) / len(self.reducedInputHistory),
                    # "4M_3_reducedInput_max": sum(self.reducedInputMaxHistory) / len(self.reducedInputMaxHistory),
                    # "4M_3_reducedInput_min": sum(self.reducedInputMinHistory) / len(self.reducedInputMinHistory),
                    # "4M_4_gateLayer_norm": sum(self.gateLayer2NormHistory) / len(self.gateLayer2NormHistory),
                    # "4M_4_gateLayer_mean": sum(self.gateLayer2History) / len(self.gateLayer2History),
                    # "4M_4_gateLayer_max": sum(self.gateLayer2MaxHistory) / len(self.gateLayer2MaxHistory),
                    # "4M_4_gateLayer_min": sum(self.gateLayer2MinHistory) / len(self.gateLayer2MinHistory),
                    # "4M_5_projected_norm": sum(self.projectedMemoryNormHistory) / len(self.projectedMemoryNormHistory),
                    # "4M_5_projected_mean": sum(self.projectedMemoryHistory) / len(self.projectedMemoryHistory),
                    # "4M_5_projected_max": sum(self.projectedMemoryMaxHistory) / len(self.projectedMemoryMaxHistory),
                    # "4M_5_projected_min": sum(self.projectedMemoryMinHistory) / len(self.projectedMemoryMinHistory),
                    # "4M_6_mixedEmbed_norm": sum(self.mixedEmbedNormHistory) / len(self.mixedEmbedNormHistory),
                    # "4M_6_mixedEmbed_mean": sum(self.mixedEmbedHistory) / len(self.mixedEmbedHistory),
                    # "4M_6_mixedEmbed_max": sum(self.mixedEmbedMaxHistory) / len(self.mixedEmbedMaxHistory),
                    # "4M_6_mixedEmbed_min": sum(self.mixedEmbedMinHistory) / len(self.mixedEmbedMinHistory),
                    # "4M_7_memoryGate_norm": sum(self.memoryGateNormHistory) / len(self.memoryGateNormHistory),
                    # "4M_7_memoryGate_mean": sum(self.memoryGateHistory) / len(self.memoryGateHistory),
                    # "4M_7_memoryGate_max": sum(self.memoryGateMaxHistory) / len(self.memoryGateMaxHistory),
                    # "4M_7_memoryGate_min": sum(self.memoryGateMinHistory) / len(self.memoryGateMinHistory),
                    "4M_x_FINAL_norm": final_norm,
                    "4M_x_FINAL_mean": final_mean,
                    "4M_x_FINAL_max": final_max,
                    "4M_x_FINAL_min": final_min,
                    "4M_1_shortGateScale": short_gate,
                    "4M_2_longGateScale": long_gate,
                    "4M_0_actGateScale": act_gate,
                    "4M_7_memoryGateScale": mem_gate,
                    "4M_1_shortDecay": statShortDecay,
                    "4M_1_longDecay": statLongDecay,
                }

                self._reset_history_buffers()

            if debugPrints:
                ʕっʘ‿ʘʔっ("store computed memories for after backward")
            self.newShort = newShort
            self.newLong = newLong

            if debugPrints:
                ʕっʘ‿ʘʔっ("return finalmemory")
            return self.FINALmemory

    @whocalled
    def updateMemoryBuffers(self):
        with self.counsellor.infodump("updateMemoryBuffers") as ʕっʘ‿ʘʔっ:
            new_short = getattr(self, "newShort", None)
            new_long = getattr(self, "newLong", None)
            with torch.no_grad():
                if new_short is not None:
                    if debugPrints:
                        ʕっʘ‿ʘʔっ("self.shortTermMemory.copy_")
                    if torch.isfinite(new_short).all():
                        self.shortTermMemory.copy_(new_short.detach())
                    else:
                        print(
                            "⚠️ [MEMORY] NaN in new_short — skipping buffer update to protect shortTermMemory"
                        )
                if new_long is not None:
                    if debugPrints:
                        ʕっʘ‿ʘʔっ("self.longTermMemory.copy_")
                    if torch.isfinite(new_long).all():
                        self.longTermMemory.copy_(new_long.detach())
                    else:
                        print(
                            "⚠️ [MEMORY] NaN in new_long — skipping buffer update to protect longTermMemory"
                        )

            self.newShort = None
            self.newLong = None
            if hasattr(self, "activationsTensor"):
                self.activationsTensor = None

    @whocalled
    def resetMemory(self, _memoryLength):
        with self.counsellor.infodump("resetMemory") as ʕっʘ‿ʘʔっ:
            with torch.no_grad():  # Ensure all ops here are outside graph
                # Detach _resetStrength if it's a tensor from a graph
                current_reset_strength_tensor = (
                    _memoryLength.detach()
                    if isinstance(_memoryLength, torch.Tensor)
                    else torch.tensor(float(_memoryLength), device=self.device)
                )
                reset_strength_scalar = (
                    current_reset_strength_tensor.item()
                    if current_reset_strength_tensor.numel() == 1
                    else float(current_reset_strength_tensor)
                )  # Fallback if not scalar
                reset_strength_scalar = max(0.0, min(1.0, reset_strength_scalar))
                keep_factor = 1.0 - reset_strength_scalar
                if debugPrints:
                    print(
                        f"resetting memory from {self.shortTermMemory.mean().item():.10f}",
                        end="",
                    )
                self.shortTermMemory.mul_(
                    keep_factor
                )  # In-place multiplication with a float
                # self.longTermMemory.mul_(keep_factor)
                if debugPrints:
                    print(f" to {self.shortTermMemory.mean().item():.10f}!")
                # self.longTermDecay += 0.1
                # self.shortTermDecay -= 0.001
                # self.longTermMemory.zero_() #retaining long term cause, yk, long term! i felt mean!
            # clear any pending buffers
            # self.newShort = None
            # self.newLong = None
            # self.activationsTensor = None

    @whocalled
    def getMemoryStats(self):
        return self.stats

    @whocalled
    def clearStats(self):
        self.stats = {}
        self._reset_history_buffers()
        with torch.no_grad():
            self.reducedInputBuf.zero_()
            self.gateLogitsBuf.zero_()
        self.newShort = None
        self.newLong = None
        if hasattr(self, "activationsTensor"):
            self.activationsTensor = None
