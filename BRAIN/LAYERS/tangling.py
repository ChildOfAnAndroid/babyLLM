# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ ---
# DIMENSION-ADAPTIVE TANGLING LAYER // brain/LAYERS/tangling.py
# v1.0  — TANGLING: reuses attention2 weights at multiple pipeline stages
# v2.0  — MINI_INN_TANGLING: lightweight causal window mixer, no attention2 needed

import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

"""
Dimension-Adaptive Tangling: Reuses attention2's 400M params throughout the network
- Projects any representation to 10k-dim
- Applies shared attention2 refinement
- Projects back to original dimension
- Residual connection preserves gradients
"""


class TANGLING(nn.Module):
    def __init__(self, _counsellor, _attention2_reference, _device=modelDevice):
        super().__init__()
        self.counsellor = _counsellor
        self.device = _device
        self.stats = {}

        # Reference to existing attention2 (NO new params here - we reuse!)
        self.attention2 = _attention2_reference

        # Shared projection layers (used at all tangling points)
        self.project_up = nn.Linear(
            embedDimension, numNeurons, device=self.device
        )  # 1024 → 10000
        self.project_down = nn.Linear(
            numNeurons, embedDimension, device=self.device
        )  # 10000 → 1024

        # Learnable gates to control how much tangling to apply at each stage
        # Initialize very small (sigmoid ≈ 0.01) so he learns whether to use it
        gate_init = math.log(0.01 / 0.99)  # logit for 0.01
        self.embed_tangle_gate = nn.Parameter(
            torch.tensor(gate_init, device=self.device)
        )
        self.neuron_tangle_gate = nn.Parameter(
            torch.tensor(gate_init, device=self.device)
        )  # Now learnable!
        self.memory_tangle_gate = nn.Parameter(
            torch.tensor(gate_init, device=self.device)
        )

        # Initialize projection weights very small so minimal impact initially
        with torch.no_grad():
            self.project_up.weight.fill_(1e-5)
            self.project_up.bias.zero_()
            self.project_down.weight.fill_(1e-5)
            self.project_down.bias.zero_()

        # Track which stage we're at for stats
        self.current_stage = "unknown"

    @whocalled
    def refine(self, _input, stage_name="unknown"):
        """
        Refine any representation by projecting to 10k-dim and applying attention2

        Args:
            _input: Input tensor (any dimension)
            stage_name: "embed", "neuron", or "memory" for stats tracking

        Returns:
            Gated refinement (caller should add to input as residual)
        """
        with self.counsellor.infodump("refine") as ʕっʘ‿ʘʔっ:
            self.current_stage = stage_name
            original = _input
            original_dim = _input.shape[-1]

            # Project to 10k-dim if needed
            if original_dim == numNeurons:
                # Already 10k-dim (neurons/INN output)
                x_10k = _input
                if debugPrints:
                    ʕっʘ‿ʘʔっ(f"tangling {stage_name}: already 10k-dim")
            else:
                # Need to project up (embeddings or memory)
                x_10k = self.project_up(_input)
                if debugPrints:
                    ʕっʘ‿ʘʔっ(
                        f"tangling {stage_name}: projected {original_dim} → {numNeurons}"
                    )

            # Apply attention2 refinement (REUSING existing weights!)
            refined_10k = self.attention2(x_10k)
            if debugPrints:
                ʕっʘ‿ʘʔっ(f"tangling {stage_name}: attention2 refinement applied")

            # Project back to original dimension if needed
            if original_dim == numNeurons:
                refined = refined_10k
            else:
                refined = self.project_down(refined_10k)
                if debugPrints:
                    ʕっʘ‿ʘʔっ(
                        f"tangling {stage_name}: projected {numNeurons} → {original_dim}"
                    )

            # Apply stage-specific gating
            if stage_name == "embed":
                gate = torch.sigmoid(self.embed_tangle_gate)
            elif stage_name == "memory":
                gate = torch.sigmoid(self.memory_tangle_gate)
            else:
                gate = torch.sigmoid(
                    self.neuron_tangle_gate
                )  # Now gated like the others!

            gated_refinement = refined * gate

            # Collect stats
            try:
                with torch.no_grad():
                    self.stats[f"TANGLE_{stage_name}_gate"] = gate.item()
                    self.stats[f"TANGLE_{stage_name}_refinement_norm"] = (
                        gated_refinement.norm().item()
                    )
            except:
                pass

            # Return just the refinement (caller will add residual)
            return gated_refinement

    @whocalled
    def getTanglingStats(self):
        with self.counsellor.infodump("getTanglingStats") as ʕっʘ‿ʘʔっ:
            return self.stats


# ─────────────────────────────────────────────────────────────────────────────
# MINI_INN_TANGLING  v1.0
# ─────────────────────────────────────────────────────────────────────────────
"""
Lightweight Temporal-Window Tangling — a drop-in replacement for TANGLING.

Philosophy: Instead of borrowing attention2 (O(seq²) compute, 400M param module),
each stage gets its own small architecture tailored to its dimension:

  "embed"  (seq_len, 1024) — full window-mixing path:
    LayerNorm  →  project(1024 → MINI_NEURONS=256)  →  GELU
    →  causal stackedWindowMeans (per-position, same cumsum trick as INN)
    →  cerebellum soft-blend over windows  →  internal residual
    →  project(256 → 1024)  →  out LayerNorm
    →  learnable gate (starts ≈0.01)  →  residual in caller

  "neuron" / "memory"  (1, 10000) — lightweight MLP path:
    LayerNorm  →  project(10000 → MINI_NEURONS_LARGE=128)  →  GELU
    →  project(128 → 10000)  →  out LayerNorm
    →  learnable gate (starts ≈0.01)  →  residual in caller
    (window mixing is a no-op for seq_len=1; MLP provides per-area tangling)

~0.5M params (embed) + ~2.5M params each (neuron, memory) = ~5.5M total.
Zero quadratic compute.
"""


class MINI_INN_TANGLING(nn.Module):
    MINI_NEURONS = 256  # width of the window-mixing bottleneck (embed stage)
    MINI_NEURONS_LARGE = 128  # width of the MLP bottleneck (neuron + memory stages)

    def __init__(
        self, _counsellor, _numTokensPerStep, _calligraphist=None, _device=modelDevice
    ):
        super().__init__()
        self.counsellor = _counsellor
        self.calligraphist = _calligraphist
        self.device = _device
        self.numTokensPerStep = _numTokensPerStep
        self.stats = {}

        mn = self.MINI_NEURONS
        mn_large = self.MINI_NEURONS_LARGE
        gate_init = math.log(0.01 / 0.99)  # sigmoid ≈ 0.01 — starts almost off

        # ── embed stage: full window-mixing path (1024-dim) ──────────────────
        for stage in ("embed",):
            setattr(self, f"{stage}_norm", nn.LayerNorm(embedDimension, device=_device))
            setattr(
                self,
                f"{stage}_proj_down",
                nn.Linear(embedDimension, mn, device=_device, bias=False),
            )
            setattr(
                self,
                f"{stage}_proj_up",
                nn.Linear(mn, embedDimension, device=_device, bias=False),
            )
            # Output LayerNorm — stabilises proj_up output before the gate,
            # mirroring refinement2's final nn.LayerNorm in the INN.
            setattr(
                self, f"{stage}_out_norm", nn.LayerNorm(embedDimension, device=_device)
            )
            # Cerebellum: one learnable weight per window (only embed uses windows)
            setattr(
                self,
                f"{stage}_cerebellum",
                nn.Parameter(torch.ones(len(allWindowSizes_new), device=_device)),
            )
            setattr(
                self,
                f"{stage}_gate",
                nn.Parameter(torch.tensor(gate_init, device=_device)),
            )

        # ── neuron + memory stages: lightweight MLP path (10000-dim) ─────────
        # Window mixing is a no-op for seq_len=1; a bottleneck MLP still gives
        # each brain area its own learnable tangling transform.
        for stage in ("neuron", "memory"):
            setattr(self, f"{stage}_norm", nn.LayerNorm(numNeurons, device=_device))
            setattr(
                self,
                f"{stage}_proj_down",
                nn.Linear(numNeurons, mn_large, device=_device, bias=False),
            )
            setattr(
                self,
                f"{stage}_proj_up",
                nn.Linear(mn_large, numNeurons, device=_device, bias=False),
            )
            setattr(self, f"{stage}_out_norm", nn.LayerNorm(numNeurons, device=_device))
            # No cerebellum for neuron/memory — no windows, no temporal blending
            setattr(
                self,
                f"{stage}_gate",
                nn.Parameter(torch.tensor(gate_init, device=_device)),
            )

        # ── Shared learnable window sizes (same temporal scales for all stages) ──
        self.logWindowSizes = nn.Parameter(
            torch.log(
                torch.tensor(allWindowSizes_new, dtype=torch.float32, device=_device)
            )
        )
        self.windowFractionality = nn.Parameter(
            torch.full((len(allWindowSizes_new),), -6.0, device=_device)
        )
        # Softmax sharpness for window competition — parameterised as softplus(param) + 1e-3.
        # This replaces the old exp(param) temperature: division by temperature is replaced by
        # multiplication by sharpness, so the denominator can never reach zero.
        # Init -0.436: softplus(-0.436) + 1e-3 ≈ 0.5 sharpness (= 1 / old_temp_2.0), same start.
        self.window_softmax_temperature = nn.Parameter(
            torch.tensor(-0.436, device=_device)
        )

        # ── History buffers for stats ──
        self._history_attrs = [
            "embed_gate_hist",
            "embed_norm_hist",
            "neuron_gate_hist",
            "neuron_norm_hist",
            "memory_gate_hist",
            "memory_norm_hist",
        ]
        self._init_history_buffers()

        # ── Initialise projections very small so initial impact is negligible ──
        with torch.no_grad():
            for stage in ("embed", "neuron", "memory"):
                getattr(self, f"{stage}_proj_down").weight.fill_(1e-4)
                getattr(self, f"{stage}_proj_up").weight.fill_(1e-4)

        if debugPrints:
            print(
                f"[MINI_INN_TANGLING] init: mn={mn}, mn_large={mn_large}, "
                f"num_windows={len(allWindowSizes_new)}, "
                f"numTokensPerStep={_numTokensPerStep}, device={_device}"
            )

    # ── History helpers (mirrors INN / NEURON pattern) ────────────────────────

    def _init_history_buffers(self):
        maxlen = max(1, self.numTokensPerStep)
        for attr in self._history_attrs:
            setattr(self, attr, deque(maxlen=maxlen))

    def _clear_histories(self):
        for attr in self._history_attrs:
            getattr(self, attr).clear()

    def _history_mean(self, history, offset=0.0):
        if not history:
            return offset
        tensor = torch.as_tensor(list(history), dtype=torch.float32, device=self.device)
        return tensor.mean().item() + offset

    # ── Internal: window tensor construction ──────────────────────────────────

    def _build_window_tensor(self):
        """Compute float+int blended window sizes. Returns (num_windows, 1).
        Also stores floatWindowSizes_used and windowTensor_used (detached) for
        calligraphist display, mirroring INN's self.floatWindowSizes_used pattern."""
        fractionality = torch.sigmoid(self.windowFractionality)
        minW = 0.1
        maxW = float(self.numTokensPerStep)
        clamped = torch.tanh(self.logWindowSizes / 2) * 1.5
        scaled = (torch.tanh(clamped) + 1) / 2
        floatW = minW + (maxW - minW) * scaled
        intW = torch.round(floatW).clamp(min=1.0)
        wt = ((1 - fractionality) * intW + fractionality * floatW).unsqueeze(
            1
        )  # (W, 1)
        # Store for stats display (detached — no GPU sync in the forward path)
        self.floatWindowSizes_used = floatW.detach()
        self.windowTensor_used = wt.squeeze(1).detach()
        return wt

    # ── Internal: causal per-position window means ────────────────────────────

    def _causal_window_means(
        self, x: torch.Tensor, windowTensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Per-position causal window means — the INN's stackedWindowMeans extended
        to return a mean for *every* sequence position, not just the final one.

        Args:
            x:            (seq_len, dim) — mini-neuron activations
            windowTensor: (num_windows, 1) — float window sizes

        Returns:
            (num_windows, seq_len, dim) — each position × each window × dim
        """
        with self.counsellor.infodump("_causal_window_means") as ʕっʘ‿ʘʔっ:
            seq_len, dim = x.shape
            maxLen = self.numTokensPerStep

            if debugPrints:
                ʕっʘ‿ʘʔっ(f"seq_len={seq_len}, dim={dim}, maxLen={maxLen}")

            # Align to maxLen: pad zeros at front if short, drop oldest if over
            tail = x[-maxLen:]  # at most maxLen tokens, shape (actual_len, dim)
            actual_len = tail.shape[0]
            offset = maxLen - actual_len  # zero-padded positions at front

            if debugPrints:
                ʕっʘ‿ʘʔっ("pad to maxLen")
            padded = torch.zeros(maxLen, dim, device=self.device)
            padded[-actual_len:] = tail

            if debugPrints:
                ʕっʘ‿ʘʔっ("cumsum")
            cumsum = F.pad(padded, (0, 0, 1, 0)).cumsum(dim=0)  # (maxLen+1, dim)

            # Integer window sizes: (num_windows,)
            window_sizes = torch.ceil(windowTensor.squeeze(1)).clamp(1, maxLen).long()
            num_windows = window_sizes.shape[0]

            # Cumsum end-indices for each token in the actual sequence: (actual_len,)
            pos = torch.arange(actual_len, device=self.device)
            end_idx = offset + pos + 1  # one past each token in the padded cumsum

            if debugPrints:
                ʕっʘ‿ʘʔっ("gather start indices")
            # Start indices per window × token: (num_windows, actual_len)
            start_idx = (end_idx.unsqueeze(0) - window_sizes.unsqueeze(1)).clamp(min=0)

            # Counts: (num_windows, actual_len)
            counts = (end_idx.unsqueeze(0) - start_idx).float().clamp(min=1.0)

            if debugPrints:
                ʕっʘ‿ʘʔっ("gather cumsum end/start and compute means")
            sum_end = cumsum[end_idx]  # (actual_len, dim)
            flat_start = start_idx.reshape(-1)  # (num_windows * actual_len,)
            sum_start = cumsum[flat_start].reshape(num_windows, actual_len, dim)

            means = (sum_end.unsqueeze(0) - sum_start) / counts.unsqueeze(
                -1
            )  # (W, actual_len, dim)

        return means

    # ── Public interface (mirrors TANGLING.refine exactly) ────────────────────

    @whocalled
    def refine(self, _input, stage_name="unknown"):
        """
        Refine a stage representation using the appropriate architecture:
          - "embed"  → causal window mixing (1024-dim, full seq_len)
          - "neuron" → lightweight bottleneck MLP (10000-dim, single token)
          - "memory" → lightweight bottleneck MLP (10000-dim, single token)

        Args:
            _input:     tensor from the pipeline — shape varies by stage
            stage_name: "embed", "neuron", or "memory"

        Returns:
            Gated refinement — caller adds as residual.
        """
        with self.counsellor.infodump("refine") as ʕっʘ‿ʘʔっ:
            if debugPrints:
                ʕっʘ‿ʘʔっ(f"stage={stage_name}, input shape={_input.shape}")

            norm_layer = getattr(self, f"{stage_name}_norm")
            proj_down = getattr(self, f"{stage_name}_proj_down")
            proj_up = getattr(self, f"{stage_name}_proj_up")
            out_norm = getattr(self, f"{stage_name}_out_norm")
            gate_param = getattr(self, f"{stage_name}_gate")

            if stage_name in ("neuron", "memory"):
                # ── Lightweight MLP path (10000-dim, single token) ──────────
                # Window mixing is a no-op for seq_len=1; the bottleneck MLP
                # still gives each brain area its own learnable tangling transform.
                if debugPrints:
                    ʕっʘ‿ʘʔっ(
                        f"MLP path: LN → {numNeurons}→{self.MINI_NEURONS_LARGE} → GELU → {numNeurons} → out_norm → gate"
                    )
                x = norm_layer(_input)  # (1, 10000) pre-norm
                mlp = F.gelu(proj_down(x))  # (1, MINI_NEURONS_LARGE)
                out = out_norm(proj_up(mlp))  # (1, 10000) output
                gate = torch.sigmoid(gate_param)
                gated = out * gate

                if debugPrints:
                    ʕっʘ‿ʘʔっ(
                        f"gate={gate.item():.4f}, gated norm={gated.norm().item():.4f}"
                    )

                try:
                    with torch.no_grad():
                        gate_val = gate.item()
                        norm_val = gated.norm().item()
                        getattr(self, f"{stage_name}_gate_hist").append(gate_val)
                        getattr(self, f"{stage_name}_norm_hist").append(norm_val)
                        self.stats[f"MINI_TANGLE_{stage_name}_gate"] = gate_val
                        self.stats[f"MINI_TANGLE_{stage_name}_norm"] = norm_val
                except Exception:
                    pass

                return gated

            # ── Window-mixing path (embed stage, full seq_len, 1024-dim) ─────
            cerebellum = getattr(self, f"{stage_name}_cerebellum")

            # 1. Pre-norm on input (standard pre-LN pattern)
            if debugPrints:
                ʕっʘ‿ʘʔっ("layernorm")
            x = norm_layer(_input)

            # 2. Project to mini-neuron space and activate
            if debugPrints:
                ʕっʘ‿ʘʔっ(f"proj_down {embedDimension}→{self.MINI_NEURONS} + GELU")
            x = F.gelu(proj_down(x))  # (seq_len, MINI_NEURONS)

            # 3. Build window tensor and compute causal per-position means
            if debugPrints:
                ʕっʘ‿ʘʔっ("build_window_tensor")
            windowTensor = self._build_window_tensor()  # (W, 1)

            if debugPrints:
                ʕっʘ‿ʘʔっ("causal_window_means")
            window_means = self._causal_window_means(
                x, windowTensor
            )  # (W, seq_len, mn)

            # 4. Cerebellum: learned softmax blend over windows
            if debugPrints:
                ʕっʘ‿ʘʔっ("cerebellum softmax blend")
            # softplus(param) + 1e-3 is always > 1e-3 and always finite — no division needed.
            # param → -∞: sharpness → 1e-3 (near-uniform, fully collaborative)
            # param → +∞: sharpness → param (linear growth, arbitrarily competitive)
            sharpness = F.softplus(self.window_softmax_temperature) + 1e-3
            log_w = torch.log(torch.sigmoid(cerebellum) + 1e-12)
            soft_weights = F.softmax(log_w * sharpness, dim=0)  # (W,)

            # Store for stats (detached; no GPU→CPU sync here)
            setattr(self, f"{stage_name}_cerebellumSoft", soft_weights.detach())

            blended = (window_means * soft_weights.unsqueeze(1).unsqueeze(2)).sum(dim=0)
            # shape: (seq_len, MINI_NEURONS)

            # 4b. Internal residual: mini-neuron activations flow through unchanged.
            #     Mirrors INN: FINALout = combinedActivationsTensor + refinedActivations
            if debugPrints:
                ʕっʘ‿ʘʔっ("internal residual (blended + x)")
            blended = blended + x  # (seq_len, MINI_NEURONS)

            # 5. Project back to embedDimension + output LayerNorm
            if debugPrints:
                ʕっʘ‿ʘʔっ(f"proj_up {self.MINI_NEURONS}→{embedDimension} + out_norm")
            out = out_norm(proj_up(blended))  # (seq_len, embedDimension)

            # 6. Learnable gate (sigmoid, starts ≈ 0.01)
            gate = torch.sigmoid(gate_param)
            gated = out * gate

            if debugPrints:
                ʕっʘ‿ʘʔっ(f"gate={gate.item():.4f}, gated shape={gated.shape}")

            # ── Stats collection (deferred to step end — no GPU→CPU sync here) ──
            try:
                with torch.no_grad():
                    gate_val = gate.item()
                    norm_val = gated.norm().item()
                    getattr(self, f"{stage_name}_gate_hist").append(gate_val)
                    getattr(self, f"{stage_name}_norm_hist").append(norm_val)
                    self.stats[f"MINI_TANGLE_{stage_name}_gate"] = gate_val
                    self.stats[f"MINI_TANGLE_{stage_name}_norm"] = norm_val
            except Exception:
                pass

        return gated

    @whocalled
    def getTanglingStats(self):
        with self.counsellor.infodump("getTanglingStats") as ʕっʘ‿ʘʔっ:
            # Add rolling window weight stats for each active stage
            if debugPrints:
                ʕっʘ‿ʘʔっ("collecting cerebellum per-stage stats")
            try:
                with torch.no_grad():
                    # Only embed has a cerebellum (neuron/memory use MLP, no windows)
                    for stage in ("embed",):
                        soft = getattr(self, f"{stage}_cerebellumSoft", None)
                        raw = getattr(self, f"{stage}_cerebellum", None)
                        if soft is not None and raw is not None:
                            for i, (s, r) in enumerate(zip(soft, raw)):
                                self.stats[f"MINI_TANGLE_{stage}_W{i}_soft"] = s.item()
                                self.stats[f"MINI_TANGLE_{stage}_W{i}_raw"] = r.item()
            except Exception:
                pass
            return self.stats

    @whocalled
    def getMiniTangleCerebellumStr(self):
        """
        Returns a calligraphist-formatted string showing the cerebellum window
        weights for the embed tangling stage (the only stage with windows).
        Neuron and memory stages use a plain MLP — no cerebellum to display.
        Mirrors INN_getStats()'s INN_cerebellum_str — call from collectTurnStats
        alongside INN_getStats() to display in the training log.
        Gated by collectStats and INN_cerebellumStats (shared flags with INN).
        """
        with self.counsellor.infodump("getMiniTangleCerebellumStr") as ʕっʘ‿ʘʔっ:
            if not (collectStats and INN_cerebellumStats):
                return ""
            if self.calligraphist is None:
                return "<MINI_TANGLE: no calligraphist>"

            float_windows = getattr(self, "floatWindowSizes_used", torch.empty(0))
            tensor_windows = getattr(self, "windowTensor_used", torch.empty(0))
            if float_windows.numel() == 0:
                return "<MINI_TANGLE: no forward pass yet>"

            parts = []
            try:
                with torch.no_grad():
                    # embed: full cerebellum window display
                    for stage in ("embed",):
                        soft = getattr(self, f"{stage}_cerebellumSoft", None)
                        raw = getattr(self, f"{stage}_cerebellum", None)
                        if soft is None or raw is None:
                            parts.append(f"[{stage}: waiting for first forward pass]")
                            continue
                        if debugPrints:
                            ʕっʘ‿ʘʔっ(f"♥{stage} cerebellumString")
                        stage_str = self.calligraphist.S_formatWindowBiasTriplets(
                            label=f"MINI_TANGLE_{stage}",
                            rawTensor=raw,
                            softTensor=soft,
                            windowSizes=float_windows,
                            windowTensor=tensor_windows,
                            per_window_style=True,
                        )
                        parts.append(stage_str)

                    # neuron + memory: compact gate/norm summary (no windows, no cerebellum)
                    for stage in ("neuron", "memory"):
                        gate_val = self.stats.get(f"MINI_TANGLE_{stage}_gate")
                        norm_val = self.stats.get(f"MINI_TANGLE_{stage}_norm")
                        if gate_val is not None:
                            parts.append(
                                f"MINI_TANGLE_{stage}   gate={gate_val:.4f}   norm={norm_val:.4f}"
                            )
                        else:
                            parts.append(f"[{stage}: waiting for first forward pass]")

            except Exception as e:
                return f"<ERR in getMiniTangleCerebellumStr: {e}>"

            return "\n".join(parts) if parts else "<MINI_TANGLE: no data>"

    @whocalled
    def clearStats(self):
        with self.counsellor.infodump("clearStats") as ʕっʘ‿ʘʔっ:
            self._clear_histories()
