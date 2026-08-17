# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ ---
# BABYLLM // babyLLM.py
# v1.12

import math
import os
import threading
import time
from collections import deque
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from adan_pytorch import Adan

# This might need some fixing when pulling the package!
from sophia.sophia import SophiaG

from brain.LAYERS.attention import GATED_MHA
from brain.LAYERS.embed import EMBED
from brain.LAYERS.interneuronNetwork import INTERNEURON_NETWORK
from brain.LAYERS.logits import LOGITS
from brain.LAYERS.memory import MEMORY
from brain.LAYERS.scratchpad import SCRATCHPAD
from brain.LAYERS.tangling import MINI_INN_TANGLING, TANGLING

# Creative modules removed
# from brain.LAYERS.sensoryWobble import WOBBLE
from config import *
from secret import *
from utils.helpers import clamp_param, debug_print, get_grad_stats

GRAD_SNAPSHOT_LIMIT = 8


def log_param_to_length(log_param):
    return torch.sigmoid((1 - torch.exp(log_param)) * 0.1)


"""this class combines all the core components of the babyLLM:"""
"""EMBED: token embedding layer"""
"""INTERNEURON_NETWORK: layer of parallel neurons for feature extraction"""
"""LOGITS: output layer to generate logits"""
"""it also manages training, loss computation, backpropagation, and response generation."""


class BABYLLM(nn.Module):
    def __init__(
        self,
        _counsellor,
        _calligraphist,
        _scribe,
        _librarian,
        _numTokensPerStep,
        _learningRateGOAL=learningRateGOAL,
        _device=modelDevice,
        _first=True,
        _model_thread_lock=None,
    ):
        super().__init__()
        self.device = _device
        self.counsellor = _counsellor
        self.calligraphist = _calligraphist
        self.scribe = _scribe
        self.librarian = _librarian
        self.model_thread_lock = _model_thread_lock or threading.Lock()
        self.numTokensPerStep = _numTokensPerStep
        # self.wobble = _wobble

        # MUST BE ON SELF - ONLY ACCESSED IN THIS CLASS AND NOT NN.PARAMS
        self.totalTokenEvaluations = 0
        self.learningRateGOAL = _learningRateGOAL
        self.latestLossDelta = 0
        self.totalTokenEvaluations_A = 0
        self.recentGeneratedTokens = []  # used for repetition penalty
        self.lastLossBaby = 0
        self.computeLossCount = 0
        self.repeatedPercent = 0
        self.normalisedActivations = 0
        self.rollingTokenTotals_tensor = torch.zeros(
            len(self.librarian.vocabList), device=self.device
        )
        self.gumBellend = 0
        self.last_forward_had_nonfinite = False
        self.nonfinite_forward_count = 0
        self.nonfinite_forward_last_log = 0.0
        self.nonfinite_recovery_count = 0
        self.pixelLoss_used = 0
        self.PIXELloss = 0
        self.CEloss_used = 0.0
        self.lrSoftClamp_used = 0.0
        self.tempSoftClamp_used = 0.0
        self.repPenSoftClamp_used = 0.0
        self.repLoss_used = 0.0
        self.targetTokenFromTutor = None
        self.last_grad_norm_before_clip = 0.0
        self.last_grad_norm_after_clip = 0.0

        # BABY EXCITEMENT MICROSCOPE
        # Pure telemetry: no parameters, no graph hooks, no loss changes.
        # Token CE records accumulate across gradient-accumulation chunks and
        # are cleared only after the optimiser step they belong to.
        self._excite_threshold = 1.0
        self._excite_token_losses = deque(maxlen=512)
        self._excite_last_blend = None

        self.stats = {}
        history_maxlen = max(1, self.numTokensPerStep)
        self.normalisedHistory = deque(maxlen=history_maxlen)
        self.INNOutputHistory = deque(maxlen=history_maxlen)
        self.memoryOutputHistory = deque(maxlen=history_maxlen)
        self.totalTurns = 1
        self.memory2OutputHistory = deque(maxlen=history_maxlen)
        self.penalisedOutputHistory = deque(maxlen=history_maxlen)
        self.inputEmbedsHistory = deque(maxlen=history_maxlen)
        self.FINALlogitsHistory = deque(maxlen=history_maxlen)
        self.charEmbedHistory = deque(maxlen=history_maxlen)
        self.predPixel = torch.tensor([0.0, 0.0, 0.0], device=self.device)

        self.cerebralLoad = 0.0
        self.dreamIntensity = 0.0
        self.memoryFlux = 0.0
        self.learningStability = 0.0
        self.AUXlossCos_used = 0.0
        self.AUXlossKL_used = 0.0
        self.lastSoftSample = None
        self._lastSoftSample_for_loss = None

        # NEW MINI LAYER
        self.char_vocab_size = 256
        self.char_embed_dim = 128
        self.char_embed = nn.Embedding(
            self.char_vocab_size, self.char_embed_dim, padding_idx=0, device=self.device
        )
        self.char_projector = nn.Linear(
            self.char_embed_dim, embedDimension, device=self.device
        )

        # B. Build the high-speed lookup table (THIS IS THE NEW PART)
        print("ʕっ•ᴥ•ʔっ Pre-calculating character-byte lookup table...")

        # Find the longest token (in bytes) to know how wide our table needs to be
        max_bytes = 0
        for i in range(self.librarian.vocabSize):
            s = self.librarian.indexToToken.get(i, "<UNK>")
            max_bytes = max(max_bytes, len(s.encode("utf-8")))

        print(
            f"Longest token is {max_bytes} bytes. Creating lookup table [shape: {self.librarian.vocabSize}, {max_bytes}]"
        )

        # This tensor (e.g., [4200, 16]) will store the byte-IDs for every token.
        # It's LONG (integers)
        char_lookup_data = torch.zeros(
            (self.librarian.vocabSize, max_bytes), dtype=torch.long
        )

        # This tensor (e.g., [4200, 16]) will store the *mask* to ignore padding.
        # It's FLOAT
        char_mask_data = torch.zeros(
            (self.librarian.vocabSize, max_bytes), dtype=torch.float
        )

        for i in range(self.librarian.vocabSize):
            s = self.librarian.indexToToken.get(i, "<UNK>")
            byte_ids = list(s.encode("utf-8"))
            if not byte_ids:
                byte_ids = [0]  # Handle empty

            length = len(byte_ids)
            char_lookup_data[i, :length] = torch.tensor(byte_ids, dtype=torch.long)
            char_mask_data[i, :length] = 1.0

        # C. Register these tables as **BUFFERS** (non-trainable data)
        # This is the fix. They are no longer nn.Parameters.
        self.register_buffer("char_lookup_data", char_lookup_data)
        self.register_buffer("char_mask_data", char_mask_data)

        print("...Character lookup table created successfully.")

        """CEREBRAL LAYERS // brain"""
        self.embed = EMBED(_counsellor=self.counsellor, _device=self.device)
        self.attention = GATED_MHA(
            _counsellor=self.counsellor, _device=self.device, _stat_prefix="2A"
        )
        self.attention2 = GATED_MHA(
            _counsellor=self.counsellor,
            _device=self.device,
            _embed_dim=numNeurons,
            _stat_prefix="4A_1",
        )
        self.interneuronNetwork = INTERNEURON_NETWORK(
            _model=BABYLLM,
            _counsellor=self.counsellor,
            _calligraphist=self.calligraphist,
            _device=self.device,
            _numTokensPerStep=self.numTokensPerStep,
        )
        self.logits = LOGITS(
            _counsellor=self.counsellor,
            _device=self.device,
            _numTokensPerStep=self.numTokensPerStep,
        )
        self.memory = MEMORY(
            _counsellor=self.counsellor,
            _device=self.device,
            _numTokensPerStep=self.numTokensPerStep,
        )
        self.memory2 = MEMORY(
            _counsellor=self.counsellor,
            _device=self.device,
            _numTokensPerStep=self.numTokensPerStep,
        )
        self.pixelPupil = PIXEL(embedDimension, embedDimension, 3, _device=self.device)

        # Tangling: causal window mixer (MINI_INN_TANGLING) or attention2-reuse (TANGLING)
        if useMiniINN_Tangling:
            self.tangling = MINI_INN_TANGLING(
                _counsellor=self.counsellor,
                _numTokensPerStep=self.numTokensPerStep,
                _calligraphist=self.calligraphist,
                _device=self.device,
            )
        else:
            self.tangling = TANGLING(
                _counsellor=self.counsellor,
                _attention2_reference=self.attention2,
                _device=self.device,
            )
        self.scratchpad = SCRATCHPAD(_counsellor=self.counsellor, _device=self.device)

        # Creative modules removed

        """LEARNABLE LEARNING PARAMETERS"""
        self.repetitionPenalty = nn.Parameter(torch.tensor(1.0, device=self.device))
        self.logTemp = nn.Parameter(torch.tensor(math.log(0.8), device=self.device))
        self.logLR = nn.Parameter(torch.tensor(math.log(1e-4), device=self.device))
        self.logGradClip = nn.Parameter(torch.tensor(math.log(1.0), device=self.device))
        self.scheduledSamplingRate = nn.Parameter(torch.tensor(0.2, device=self.device))
        self.logMemoryLength = nn.Parameter(
            torch.tensor(math.log(memoryLengthGOAL), device=self.device)
        )
        self.logMemory2Length = nn.Parameter(
            torch.tensor(math.log(memoryLengthGOAL), device=self.device)
        )
        self.logRepetitionWindow = nn.Parameter(
            torch.tensor(math.log(repetitionWindowGOAL), device=self.device)
        )
        self.inputBlend = nn.Parameter(torch.ones(3, device=self.device))
        self.charBlendWeight = nn.Parameter(torch.zeros(1, device=self.device))
        self.memoryLength = log_param_to_length(self.logMemoryLength)
        self.memory2Length = log_param_to_length(self.logMemory2Length)

        self.embed.pixelEmbed.requires_grad_(False)

        self.sensory_dim = 9
        self.sensory_pred_dim = self.sensory_dim + 1
        self.sensory_scale = nn.Parameter(
            torch.ones(self.sensory_dim, device=self.device)
        )
        self.sensory_bias = nn.Parameter(
            torch.zeros(self.sensory_dim, device=self.device)
        )
        self.sensoryEmbed = nn.Sequential(
            nn.Linear(self.sensory_dim, embedDimension // 2),
            nn.Tanh(),
            nn.Linear(embedDimension // 2, embedDimension),
        )
        # Output normalization (matches MEMORY/SCRATCHPAD pattern, separate so checkpoint-compatible)
        self.sensoryEmbed_norm = nn.LayerNorm(embedDimension, device=self.device)
        self.sensoryPupil = nn.Linear(
            embedDimension, self.sensory_pred_dim, device=self.device
        )
        sensory_gate_init = math.log(0.01 / 0.99)
        self.sensory_gate = nn.Parameter(
            torch.tensor(sensory_gate_init, device=self.device)
        )
        self.sensory_gate_used = 0.0
        self.cached_sensory = None
        self.cached_device_temp_c = None
        self.latest_sensory_vector = None
        self.latest_device_temp_c = None
        self.predSensory = None
        self.targetSensory = None
        self.sensoryLoss_used = 0.0
        self.latestTokenEmbed_raw = None
        self.prevSensoryPredEmbed_raw = None
        self.nextSensoryPredEmbed_raw = None
        self.sensory_temp_scale = 1.0

        self.temperature_scale = nn.Parameter(torch.tensor(1.0, device=self.device))
        self.temperature_bias = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.temperature_vector = nn.Parameter(
            torch.zeros(embedDimension, device=self.device)
        )

        with torch.no_grad():
            self.sensory_scale.fill_(1e-5)
            self.sensory_bias.zero_()
            for layer in self.sensoryEmbed:
                if isinstance(layer, nn.Linear):
                    layer.weight.fill_(1e-5)
                    if layer.bias is not None:
                        layer.bias.zero_()
            self.temperature_scale.fill_(1e-5)
            self.temperature_bias.zero_()
            self.temperature_vector.fill_(1e-5)
        # === AR TEMPORAL TANGLE ==========================================
        # Static/horizon Baby remains intact.
        #
        # Rolling chronology enters through the EXISTING character language,
        # tangles with EXISTING position, compresses through a shared 512-D
        # dialect, then fans into individually gated existing neurons.
        self.ar_width = 512

        self.ar_throat = nn.Linear(
            embedDimension,
            self.ar_width,
            bias=False,
            device=self.device,
        )
        self.ar_expand = nn.Linear(
            self.ar_width,
            numNeurons,
            bias=False,
            device=self.device,
        )
        self.ar_neuron_gate = nn.ParameterList(
            [
                nn.Parameter(
                    torch.full(
                        (numNeurons,),
                        -4.5,
                        device=self.device,
                    )
                )
            ]
        )

        self._ar_braid_last = None
        self._ar_raw_last = None
        self._ar_applied_last = None
        # === END AR TEMPORAL TANGLE =====================================

        """self.transformer_block = nn.TransformerEncoderLayer(
            d_model=embedDimension, 
            nhead=8,  # A reasonable number of attention heads
            dim_feedforward=embedDimension * 4, # Standard practice
            dropout=0.1,
            activation='gelu',
            batch_first=True # IMPORTANT
        ).to(self.device)"""

        """stuff"""
        # self.gradientClipMaxNorm = torch.exp(self.logGradClip)
        self.temperature = None

        """OPTIMIZER - this updates all of the layers learnable parameters"""
        debug_print("registered parameters: ")
        for name, param in BABYLLM.named_parameters(self):
            debug_print(name, param.shape)

        # baseOptim = optim.RAdam(self.parameters(), lr = learningRate)
        # baseOptim = torch_optimizer.Lion(self.parameters(), lr = 1e-4)
        # self.optimizer = optim.Lookahead(baseOptim)
        # self.optimizer = baseOptim

        if optimizerName == "Adan":
            self.optimizer = Adan(
                self.parameters(),
                lr=learningRate,
                betas=(0.98, 0.92, 0.99),
                eps=1e-6,
                weight_decay=0.05,
            )
        elif optimizerName == "Sophia":
            self.optimizer = SophiaG(
                self.parameters(),
                lr=learningRate,  # start slightly lower LR than AdamW, Sophia can be aggressive
                betas=(0.965, 0.99),
                rho=0.04,
                weight_decay=0.05,
            )
        else:
            optimizerClass = getattr(optim, optimizerName)
            optimizer_kwargs = {"lr": learningRate, "weight_decay": 0.05}
            if getattr(self.device, "type", None) == "cuda":
                optimizer_kwargs["fused"] = True
            try:
                self.optimizer = optimizerClass(self.parameters(), **optimizer_kwargs)
            except TypeError:
                optimizer_kwargs.pop("fused", None)
                self.optimizer = optimizerClass(self.parameters(), **optimizer_kwargs)
        # print("!!! RUNNING WITH SGD OPTIMIZER !!!")
        # self.optimizer = optim.SGD(self.parameters(), lr=learningRate, momentum=0.9)

        for name, param in self.named_parameters():
            debug_print(f"{name}: requires_grad={param.requires_grad}")

        # self.to(self.device)
        self.statsCategories = {
            "loss": 0,
            "gradNorm": 0,
            "logitMin": 0,
            "logitMax": 0,
            "scheduledSamplingRate": 0,
            "tokenCount": 0,
            "memoryGateShort": 0,
            "memoryGateLong": 0,
            "memoryGateCurrent": 0,
            "shortDecay": 0,
            "longDecay": 0,
        }

    def _snapshot_gradients(self, limit: int = GRAD_SNAPSHOT_LIMIT):
        snapshot = []
        for name, param in self.named_parameters():
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            try:
                stats = get_grad_stats(grad)
            except Exception:
                continue
            snapshot.append((name, stats))
            if len(snapshot) >= limit:
                break
        return snapshot

    def _excite_token_name(self, token_id):
        try:
            token = self.librarian.indexToToken.get(int(token_id), f"<{int(token_id)}>")
            return repr(str(token).replace("\n", "\\n"))
        except Exception:
            return repr(f"<{token_id}>")

    def _excite_current_grad_norm(self):
        """Measure current global L2 grad norm WITHOUT modifying gradients."""
        total_sq = 0.0
        with torch.no_grad():
            for p in self.parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if not torch.isfinite(g).all():
                    return float("inf")
                n = float(g.norm(2).item())
                total_sq += n * n
        return total_sq ** 0.5

    def _print_excitement_report(self, preclip_norm, clip_value):
        """Rare-event microscope. Called BEFORE clip_grad_norm_."""
        try:
            print("\n" + "=" * 74)
            print("🚨 BABY EXCITEMENT — PRE-CLIP MICROSCOPE")
            print("=" * 74)
            print(
                f"global preclip={preclip_norm:.6f} | "
                f"clip target={float(clip_value):.6f} | "
                f"would scale≈{min(1.0, float(clip_value) / max(preclip_norm, 1e-12)):.6f}"
            )

            # A. WHO IS SHOUTING? — top individual trainable parameters.
            param_rows = []
            with torch.no_grad():
                for name, p in self.named_parameters():
                    if p.grad is None:
                        continue
                    g = p.grad.detach()
                    if not torch.isfinite(g).all():
                        norm = float("inf")
                        mx = float("inf")
                    else:
                        norm = float(g.norm().item())
                        mx = float(g.abs().max().item())
                    param_rows.append((norm, mx, name, p.numel()))

            param_rows.sort(key=lambda row: row[0], reverse=True)

            print("\nTOP INDIVIDUAL PARAMETER GRADIENTS")
            for norm, mx, name, numel in param_rows[:15]:
                pct = (norm / preclip_norm * 100.0) if preclip_norm > 0 else 0.0
                print(
                    f"  {name:<52} "
                    f"norm={norm:>10.6f}  max={mx:>10.6f}  "
                    f"{pct:>6.2f}%  n={numel}"
                )

            # B. EXACT INPUT-BLEND ARGUMENT.
            print("\nINPUT NEGOTIATION")
            blend_names = ("token", "position", "rgb/state")
            blend_grad = getattr(self.inputBlend, "grad", None)
            blend_now = self._excite_last_blend

            if blend_grad is not None:
                vals = blend_grad.detach().flatten().cpu().tolist()
                for i, label in enumerate(blend_names):
                    current = (
                        float(blend_now[i])
                        if blend_now is not None and i < len(blend_now)
                        else float("nan")
                    )
                    grad = float(vals[i]) if i < len(vals) else float("nan")
                    desire = "↑ MORE" if grad < 0 else "↓ LESS" if grad > 0 else "·"
                    # Gradient descent moves opposite the gradient.
                    print(
                        f"  {label:<10} blend={current:>9.6f}  "
                        f"dL/dw={grad:>+11.6f}  optimiser tendency: {desire}"
                    )

            char_grad = getattr(self.charBlendWeight, "grad", None)
            if char_grad is not None:
                grad = float(char_grad.detach().flatten()[0].item())
                current = (
                    float(blend_now[3])
                    if blend_now is not None and len(blend_now) > 3
                    else float("nan")
                )
                desire = "↑ MORE" if grad < 0 else "↓ LESS" if grad > 0 else "·"
                print(
                    f"  {'char':<10} blend={current:>9.6f}  "
                    f"dL/dw={grad:>+11.6f}  optimiser tendency: {desire}"
                )

            # C. Small learned controls: automatically discover them rather
            # than maintaining a brittle hard-coded list.
            controls = []
            with torch.no_grad():
                for name, p in self.named_parameters():
                    if p.grad is None or p.numel() > 16:
                        continue
                    g = p.grad.detach()
                    if not torch.isfinite(g).all():
                        norm = float("inf")
                    else:
                        norm = float(g.norm().item())
                    controls.append((norm, name, p.detach(), g))

            controls.sort(key=lambda row: row[0], reverse=True)

            print("\nSMALL / SCALAR CONTROLS")
            for norm, name, value, grad in controls[:15]:
                v = value.flatten().cpu().tolist()
                g = grad.flatten().cpu().tolist()
                vtxt = ",".join(f"{float(x):+.4g}" for x in v[:8])
                gtxt = ",".join(f"{float(x):+.4g}" for x in g[:8])
                if len(v) > 8:
                    vtxt += ",…"
                    gtxt += ",…"
                print(
                    f"  {name:<42} norm={norm:>9.6f}  "
                    f"value=[{vtxt}] grad=[{gtxt}]"
                )

            # D. Which token predictions contributed the largest raw CE to
            # THIS accumulated optimiser step?
            print("\nWORST RAW TOKEN CE IN THIS OPTIMISER STEP")
            records = sorted(
                list(self._excite_token_losses),
                key=lambda r: r["ce"],
                reverse=True,
            )
            for rec in records[:12]:
                tail = " ".join(
                    self._excite_token_name(t)
                    for t in rec.get("context_tail", [])
                )
                print(
                    f"  pos={rec['position']:<4} "
                    f"target={self._excite_token_name(rec['target']):<22} "
                    f"CE={rec['ce']:>8.4f}"
                )
                if tail:
                    print(f"       context tail: {tail}")

            if records:
                ces = [r["ce"] for r in records]
                print(
                    f"  recorded={len(records)} | "
                    f"CE mean={sum(ces)/len(ces):.4f} | "
                    f"max={max(ces):.4f}"
                )

            # E. What did the world look like while this argument happened?
            print("\nCURRENT SENSORY STATE")
            sensory_names = (
                "global_light",
                "noise",
                "time_of_day",
                "interaction_recency",
                "training_age",
                "global_motion",
                "left_right_bias",
                "top_bottom_bias",
                "contrast_intrusion",
            )
            sensory = getattr(self, "latest_sensory_vector", None)
            if sensory is not None:
                vals = sensory.detach().flatten().cpu().tolist()
                for name, val in zip(sensory_names, vals):
                    print(f"  {name:<24} {float(val):.6f}")
            else:
                print("  (no current sensory vector)")

            temp = getattr(self, "latest_device_temp_c", None)
            if temp is not None:
                try:
                    print(f"  {'device_temp':<24} {float(temp.detach().item()):.6f}")
                except Exception:
                    pass

            print("=" * 74 + "\n")

        except Exception as exc:
            # A diagnostic must NEVER get to break training.
            print(f"[BABY EXCITEMENT microscope failed safely: {exc!r}]")

    def _sensory_nudge(self, vector, index: int, max_pct: float = 0.02) -> float:
        try:
            if vector is None:
                return 1.0
            val = vector[index]
            if torch.is_tensor(val):
                if not torch.isfinite(val):
                    return 1.0
                val = float(val.detach().item())
            else:
                val = float(val)
            shift = (val - 0.5) * 2.0
            shift = max(-1.0, min(1.0, shift))
            max_pct = float(max_pct)
            factor = 1.0 + (max_pct * shift)
            lower = 1.0 - max_pct
            upper = 1.0 + max_pct
            return max(lower, min(upper, factor))
        except Exception:
            return 1.0

    def _encode_char_tokens(self, token_indices):
        padded_bytes = F.embedding(token_indices, self.char_lookup_data)
        byte_mask = F.embedding(token_indices, self.char_mask_data)
        embedded_chars = self.char_embed(padded_bytes) * byte_mask.unsqueeze(-1)
        char_vectors = embedded_chars.sum(dim=1) / byte_mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1.0)
        return self.char_projector(char_vectors)

    @whocalled
    def forward(
        self,
        _inputSeq=None,
        _pixel=None,
        _use_lock: bool = True,
        _arInputSeq=None,
    ):
        with (
            self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ
        ):  # processes input sequence of tokens (str) to generate logits to predict the next token
            lock_ctx = self.model_thread_lock if _use_lock else nullcontext()
            with lock_ctx:
                if debugPrints:
                    tensor_snitch(self, "babyllm forward start")
                    tensor_snitch(self.memory, "babyllm forward start")
                    tensor_snitch(self.memory2, "babyllm forward start")
                    tensor_snitch(self.embed, "babyllm forward start")
                    tensor_snitch(self.interneuronNetwork, "babyllm forward start")
                    tensor_snitch(self.logits, "babyllm forward start")
                self.pixel = _pixel
                self.prevSensoryPredEmbed_raw = self.nextSensoryPredEmbed_raw
                self.nextSensoryPredEmbed_raw = None
                sensory_source = self.cached_sensory
                motion_scale = self._sensory_nudge(sensory_source, 5, 0.02)
                attention_scale = self._sensory_nudge(sensory_source, 8, 0.02)
                self.sensory_temp_scale = motion_scale
                self.attention.gate_nudge = attention_scale
                self.attention2.gate_nudge = attention_scale
                self.temperature = torch.exp(self.logTemp) * motion_scale
                self.interneuronNetwork.temperature = self.temperature
                if self.cached_sensory is not None:
                    sensory_vector = self.cached_sensory
                    sensory_scale_used = (
                        torch.sigmoid(self.sensory_scale) * 10.0
                    )  # [0, 10]
                    sensory_bias_used = (
                        torch.sigmoid(self.sensory_bias) - 0.5
                    ) * 10.0  # [-5, 5]
                    sensory_adjusted = (
                        sensory_vector * sensory_scale_used
                    ) + sensory_bias_used
                    sensory_embed = self.sensoryEmbed(sensory_adjusted)
                    sensory_embed = self.sensoryEmbed_norm(
                        sensory_embed
                    )  # Normalize before gating (matches SCRATCHPAD)
                    sensory_embed = (
                        torch.tanh(sensory_embed) * 10.0
                    )  # [-10, 10] bounded output
                    gate = torch.sigmoid(self.sensory_gate)
                    self.latest_sensory_vector = sensory_vector.detach()
                    self.sensory_gate_used = gate.detach().item()
                else:
                    sensory_embed = 0.0
                    gate = 0.0
                    self.latest_sensory_vector = None
                    self.sensory_gate_used = 0.0

                if self.cached_device_temp_c is not None:
                    temp_value = self.cached_device_temp_c
                    temp_scale_used = (
                        torch.sigmoid(self.temperature_scale) * 5.0
                    )  # [0, 5]
                    temp_bias_used = (
                        torch.sigmoid(self.temperature_bias) - 0.5
                    ) * 5.0  # [-2.5, 2.5]
                    temp_vec_used = (
                        torch.sigmoid(self.temperature_vector) * 5.0
                    )  # [0, 5]
                    temp_scaled = (temp_value * temp_scale_used) + temp_bias_used
                    temp_embed = temp_scaled * temp_vec_used
                    if torch.is_tensor(sensory_embed):
                        sensory_embed = sensory_embed + temp_embed
                    else:
                        sensory_embed = temp_embed
                    self.latest_device_temp_c = temp_value.detach()
                else:
                    self.latest_device_temp_c = None

                if debugPrints:
                    ʕっʘ‿ʘʔっ("B0: inputEmbeds")  # convert indices to embeddings
                tokenEmbed = self.embed(_tokenIndex=_inputSeq)
                seq_len = tokenEmbed.shape[0]  # e.g., 1024

                # --- START: NEW **SUPER-FAST** CHARACTER EMBEDDING LOGIC ---
                if debugPrints:
                    ʕっʘ‿ʘʔっ("B0.5: charEmbeds (Super-Fast)")

                charEmbed = self._encode_char_tokens(_inputSeq)

                char_scale = self._sensory_nudge(sensory_source, 7)
                if char_scale != 1.0:
                    charEmbed = charEmbed * char_scale

                self.charEmbedHistory.append(charEmbed.norm().item())
                # --- END: SUPER-FAST CHARACTER EMBEDDING LOGIC ---

                # --- 6. BLEND (Same as before) ---
                pos_indices = torch.arange(seq_len, device=tokenEmbed.device)
                posEmbed = self.embed.posEmbedding(pos_indices)
                posEmbed = self.embed.posDropout(
                    posEmbed * self.embed.scale
                )  # [seq_len, embed_dim]
                pos_scale = self._sensory_nudge(sensory_source, 6)
                if pos_scale != 1.0:
                    posEmbed = posEmbed * pos_scale

                # === AR TEMPORAL TANGLE: rolling CHAR × POSITION ==========
                self._ar_braid_last = None
                self._ar_raw_last = None
                self._ar_applied_last = None
                ar_applied = None

                # During training Tutor supplies an independent rolling
                # chronology. During normal generation the ordinary context
                # already rolls, so use that as the temporal source.
                ar_source_seq = (
                    _inputSeq
                    if _arInputSeq is None
                    else _arInputSeq
                )

                if ar_source_seq is not None:
                    if torch.is_tensor(ar_source_seq):
                        # Tutor mutates its rolling buffer before chunk backward,
                        # therefore forward owns an immutable index snapshot.
                        ar_input = (
                            ar_source_seq.detach()
                            .to(
                                device=tokenEmbed.device,
                                dtype=torch.long,
                            )
                            .flatten()
                            .clone()
                        )
                    else:
                        ar_input = torch.as_tensor(
                            ar_source_seq,
                            device=tokenEmbed.device,
                            dtype=torch.long,
                        ).flatten().clone()

                    if ar_input.numel() > 0:
                        rollingChar = self._encode_char_tokens(ar_input)

                        if char_scale != 1.0:
                            rollingChar = rollingChar * char_scale

                        ar_len = min(
                            int(rollingChar.shape[0]),
                            int(charEmbed.shape[0]),
                            int(posEmbed.shape[0]),
                        )

                        if ar_len > 0:
                            rolling_now = rollingChar[-ar_len:]
                            static_now = charEmbed[-ar_len:]
                            ar_pos = posEmbed[-ar_len:]

                            # Two views through ONE shared temporal dialect:
                            #
                            # 1. rolling chronology itself
                            # 2. what chronology knows that frozen Baby does not
                            #
                            # In live generation #2 naturally becomes ~zero,
                            # while #1 remains available.
                            ar_roll_norm = F.layer_norm(
                                rolling_now,
                                (rolling_now.shape[-1],),
                            )

                            ar_delta = rolling_now - static_now
                            ar_delta_norm = F.layer_norm(
                                ar_delta,
                                (ar_delta.shape[-1],),
                            )

                            ar_pos_norm = F.layer_norm(
                                ar_pos,
                                (ar_pos.shape[-1],),
                            )

                            # SAME 1024→512 throat for all three.
                            ar_roll_small = self.ar_throat(
                                ar_roll_norm
                            )
                            ar_delta_small = self.ar_throat(
                                ar_delta_norm
                            )
                            ar_pos_small = self.ar_throat(
                                ar_pos_norm
                            )

                            # Direct chronology + divergence,
                            # then positional multiplicative tangle.
                            ar_temporal_small = torch.tanh(
                                ar_roll_small + ar_delta_small
                            )

                            ar_braid = (
                                ar_temporal_small
                                * torch.tanh(ar_pos_small)
                            ).mean(dim=0, keepdim=True)

                            ar_braid = F.layer_norm(
                                ar_braid,
                                (self.ar_width,),
                            )

                            ar_neurons = self.ar_expand(
                                ar_braid
                            )

                            ar_gate = torch.sigmoid(
                                self.ar_neuron_gate[0]
                            ).unsqueeze(0)

                            ar_applied = ar_gate * ar_neurons

                            self._ar_braid_last = ar_braid.detach()
                            self._ar_raw_last = ar_neurons.detach()
                            self._ar_applied_last = (
                                ar_applied.detach()
                            )
                # === END AR TEMPORAL TANGLE ================================

                all_blend_weights = torch.cat(
                    [self.inputBlend, self.charBlendWeight], dim=0
                )
                blend = F.softmax(all_blend_weights, dim=0)
                noise_scale = self._sensory_nudge(sensory_source, 1, 0.02)
                if noise_scale != 1.0:
                    blend = blend.clone()
                    blend[2] = blend[2] * noise_scale
                    blend = blend / blend.sum().clamp_min(1e-6)

                # Snapshot current blend for rare excitement reports.
                # Token position/context belongs to TUTOR, which actually
                # knows which target this forward pass is predicting.
                self._excite_last_blend = blend.detach().cpu().tolist()

                if not skipPixels and (_pixel is not None):
                    legacy_rgb_embed = self.embed(_pixel=_pixel)
                    rgbEmbed = (1 - gate) * legacy_rgb_embed + gate * (
                        legacy_rgb_embed + sensory_embed
                    )
                    debug_print("tokenEmbed:", tokenEmbed.shape)
                    debug_print("posEmbed:", posEmbed.shape)
                    debug_print("rgbEmbed:", rgbEmbed.shape)
                    debug_print("charEmbed:", charEmbed.shape)

                    inputEmbeds = (
                        blend[0] * tokenEmbed
                        + blend[1] * posEmbed
                        + blend[2] * rgbEmbed
                        + blend[3] * charEmbed
                    )
                else:
                    inputEmbeds = (
                        blend[0] * tokenEmbed
                        + blend[1] * posEmbed
                        + blend[3] * charEmbed
                    )
                # --- END: BLEND ---

                token_embed_for_pixel = inputEmbeds
                self.latestTokenEmbed_raw = token_embed_for_pixel
                self.latestTokenEmbed = token_embed_for_pixel.detach()
                self.nextSensoryPredEmbed_raw = token_embed_for_pixel.detach()
                if (
                    hasattr(self, "pixelPupil")
                    and len(self.latestTokenEmbed.shape) == 1
                ):
                    debug_print(
                        f"[DEBUG] latestTokenEmbed is 1D with shape {self.latestTokenEmbed.shape}"
                    )

                inputEmbeds = self.attention(inputEmbeds)

                # TANGLING STAGE 1: Refine embeddings (1024-dim) via attention2
                if enableTangling:
                    inputEmbeds = inputEmbeds + self.tangling.refine(
                        inputEmbeds, stage_name="embed"
                    )

                debug_print(
                    f"Debug BABYLLM.forward: inputEmbeds requires_grad: {inputEmbeds.requires_grad} [EXPECTED: TRUE]"
                )

                if debugPrints:
                    ʕっʘ‿ʘʔっ("B1: interneuronNetworkOutput")

                if True:
                    interneuron_output = self.interneuronNetwork.forward(inputEmbeds)
                    INNOutput = interneuron_output + self.attention2(interneuron_output)

                    # Extra chronology only; established neuron state survives.
                    if ar_applied is not None:
                        INNOutput = INNOutput + ar_applied

                    # TANGLING STAGE 2: Refine neurons (10k-dim) via attention2
                    if enableTangling:
                        INNOutput = INNOutput + self.tangling.refine(
                            INNOutput, stage_name="neuron"
                        )

                    # SCRATCH PAD: Working memory before permanent storage
                    INNOutput = INNOutput + self.scratchpad(INNOutput)

                    debug_print(
                        f"Debug BABYLLM.forward: interneuronNetworkOutput length: {len(INNOutput)}"
                    )
                    debug_print(
                        "combinedActivationsTensor.requires_grad:",
                        INNOutput.requires_grad,
                    )
                    debug_print("combinedActivationsTensor.grad_fn:", INNOutput.grad_fn)

                    if debugPrints:
                        ʕっʘ‿ʘʔっ("B2: memoryOutput")
                    if skipMemory:
                        debug_print("skipping memory layer...")
                        memoryOutput = INNOutput
                    else:
                        memoryOutput = self.memory.forward(INNOutput) + INNOutput

                        # TANGLING STAGE 3: Refine memory output (1024-dim) via attention2
                        if enableTangling:
                            memoryOutput = memoryOutput + self.tangling.refine(
                                memoryOutput, stage_name="memory"
                            )

                        memory2Input = (INNOutput * 0.5) + (memoryOutput * 0.5)
                        memory2Output = (
                            self.memory2.forward(memory2Input) + memory2Input
                        )

                    if debugPrints:
                        ʕっʘ‿ʘʔっ("B3: logits.forward BEFORE penalty")
                    logitsBeforePenalty = self.logits.forward(memory2Output)
                    debug_print(
                        "combinedActivations.requires_grad:", memoryOutput.requires_grad
                    )

                if debugPrints:
                    ʕっʘ‿ʘʔっ("B4: applyRepetitionPenalty to logits")
                if not torch.isfinite(self.logRepetitionWindow):
                    print("logRepetitionWindow has gone non-finite. Resetting.")
                    self.logRepetitionWindow.data = torch.tensor(
                        math.log(repetitionWindowGOAL), device=self.device
                    )
                if self.logRepetitionWindow > math.log(windowMAXSTART):
                    print(
                        "logRepetitionWindow is higher than windowMAXSTART. Resetting."
                    )
                    self.logRepetitionWindow.data = torch.tensor(
                        math.log(repetitionWindowGOAL), device=self.device
                    )
                penalisedLogits = self.applyRepetitionPenalty(
                    logitsBeforePenalty, _inputSeq
                )

                debug_print(
                    "before memory output requires_grad?",
                    self.memory.longTermMemory.requires_grad,
                )
                debug_print(
                    "before cerebellum requires_grad?",
                    self.interneuronNetwork.cerebellum.requires_grad,
                )
                debug_print(
                    "before logRepetitionWindow requires_grad?",
                    self.logRepetitionWindow.requires_grad,
                )
                debug_print(
                    "before logMemoryLength requires_grad?",
                    self.logMemoryLength.requires_grad,
                )
                if skipFINALlogitNorm:
                    if debugPrints:
                        ʕっʘ‿ʘʔっ("Bx: logits.forward")
                    FINALlogits = penalisedLogits
                else:
                    FINALlogits = penalisedLogits

                debug_print(
                    "AFTER logMemoryLength requires_grad?",
                    self.logMemoryLength.requires_grad,
                )
                debug_print(
                    "AFTER logRepetitionWindow requires_grad?",
                    self.logRepetitionWindow.requires_grad,
                )
                debug_print(
                    "AFTER cerebellum requires_grad?",
                    self.interneuronNetwork.cerebellum.requires_grad,
                )
                debug_print(
                    "AFTER memory output requires_grad?",
                    self.memory.longTermMemory.requires_grad,
                )

                if True:
                    if debugPrints:
                        ʕっʘ‿ʘʔっ("stats collection!")

                    # --- START: MODIFIED STATS LOGIC ---
                    # Get blend weights (this variable is now available everywhere)
                    blend_vals_detached = blend.detach().cpu().tolist()

                    self.FINALlogitsHistory.append(FINALlogits.norm().item())

                    if len(self.FINALlogitsHistory) >= self.numTokensPerStep:
                        self.forwardStats = {
                            "7B_x_FINALlogits_norm": sum(self.FINALlogitsHistory)
                            / len(self.FINALlogitsHistory),
                            # --- ADDING YOUR NEW STATS ---
                            # 1. The average norm of the mini-layer's *output*
                            "B_charEmbed_OUT_norm": sum(self.charEmbedHistory)
                            / len(self.charEmbedHistory),
                            # 2. The norm of the mini-layer's *embedding weights*
                            "B_charEmbed_W_norm": self.char_embed.weight.norm().item(),
                            # 3. The norm of the mini-layer's *projector weights*
                            "B_charProj_W_norm": self.char_projector.weight.norm().item(),
                        }

                        if self._ar_braid_last is not None:
                            with torch.no_grad():
                                _ar_gates = torch.sigmoid(
                                    self.ar_neuron_gate[0]
                                )
                                _ar_values = torch.stack(
                                    [
                                        _ar_gates.mean(),
                                        _ar_gates.max(),
                                        self._ar_braid_last.norm(),
                                        self._ar_raw_last.norm(),
                                        self._ar_applied_last.norm(),
                                    ]
                                ).tolist()

                            self.forwardStats["AR_gate_mean"] = _ar_values[0]
                            self.forwardStats["AR_gate_max"] = _ar_values[1]
                            self.forwardStats["AR_braid_norm"] = _ar_values[2]
                            self.forwardStats["AR_raw_neuron_norm"] = _ar_values[3]
                            self.forwardStats["AR_applied_neuron_norm"] = _ar_values[4]

                        self.forwardStats["B_sensory_gate"] = self.sensory_gate_used
                        _fwd_stats = torch.stack(
                            [
                                (torch.sigmoid(self.sensory_scale) * 10.0).mean(),
                                (
                                    (torch.sigmoid(self.sensory_bias) - 0.5) * 10.0
                                ).mean(),
                                self.sensoryEmbed[0].weight.norm(),
                                self.sensoryPupil.weight.norm(),
                                torch.sigmoid(self.temperature_scale) * 5.0,
                                (torch.sigmoid(self.temperature_bias) - 0.5) * 5.0,
                                (torch.sigmoid(self.temperature_vector) * 5.0).norm(),
                            ]
                        ).tolist()
                        self.forwardStats["B_sensory_scale_mean"] = _fwd_stats[0]
                        self.forwardStats["B_sensory_bias_mean"] = _fwd_stats[1]
                        self.forwardStats["B_sensory_embed_w_norm"] = _fwd_stats[2]
                        self.forwardStats["B_sensory_pupil_w_norm"] = _fwd_stats[3]
                        self.forwardStats["B_temp_scale"] = _fwd_stats[4]
                        self.forwardStats["B_temp_bias"] = _fwd_stats[5]
                        self.forwardStats["B_temp_vec_norm"] = _fwd_stats[6]
                        if self.latest_sensory_vector is not None:
                            _sens_keys = [
                                "S_global_light_delta",
                                "S_noise_delta",
                                "S_time_of_day_delta",
                                "S_interaction_recency_delta",
                                "S_training_age_delta",
                                "S_global_motion_delta",
                                "S_left_right_bias_delta",
                                "S_top_bottom_bias_delta",
                                "S_contrast_intrusion_delta",
                            ]
                            for _k, _v in zip(
                                _sens_keys, self.latest_sensory_vector.tolist()
                            ):
                                self.forwardStats[_k] = _v
                        if self.latest_device_temp_c is not None:
                            self.forwardStats["S_device_temp_c_delta"] = (
                                self.latest_device_temp_c.item()
                            )

                        # Log blend weights
                        self.forwardStats["B_blendToken"] = blend_vals_detached[0]
                        self.forwardStats["B_blendPos"] = blend_vals_detached[1]

                        if not skipPixels and (_pixel is not None):
                            self.forwardStats["B_blendPixel"] = blend_vals_detached[2]
                            self.forwardStats["B_blendChar"] = blend_vals_detached[3]
                            debug_print(
                                f"token {blend_vals_detached[0]:.2f}, pos {blend_vals_detached[1]:.2f}, pixel {blend_vals_detached[2]:.2f}, char {blend_vals_detached[3]:.2f}"
                            )
                        else:
                            self.forwardStats["B_blendPixel"] = (
                                0.0  # Log 0 since it wasn't used
                            )
                            self.forwardStats["B_blendChar"] = blend_vals_detached[3]
                            debug_print(
                                f"token {blend_vals_detached[0]:.2f}, pos {blend_vals_detached[1]:.2f}, char {blend_vals_detached[3]:.2f} (no pixel)"
                            )
                        # --- END: MODIFIED STATS LOGIC ---

                        # Collect tangling and scratchpad stats
                        if enableTangling:
                            tangling_stats = self.tangling.getTanglingStats()
                            self.forwardStats.update(tangling_stats)

                        scratchpad_stats = self.scratchpad.getScratchpadStats()
                        self.forwardStats.update(scratchpad_stats)

                        self.stats.update(self.forwardStats)

                        self.inputEmbedsHistory.clear()
                        self.INNOutputHistory.clear()
                        self.memoryOutputHistory.clear()
                        self.memory2OutputHistory.clear()
                        self.penalisedOutputHistory.clear()
                        self.FINALlogitsHistory.clear()
                        self.normalisedHistory.clear()
                        self.charEmbedHistory.clear()  # <-- NEW: Clear the new history

                """returns a logits tensor of shape (1, vocabSize) showing predicted probabilities for the next token"""
                if debugPrints:
                    tensor_snitch(self, "babyllm forward end")
                    tensor_snitch(self.memory, "babyllm forward end")
                    tensor_snitch(self.memory2, "babyllm forward end")
                    tensor_snitch(self.embed, "babyllm forward end")
                    tensor_snitch(self.interneuronNetwork, "babyllm forward end")
                    tensor_snitch(self.logits, "babyllm forward end")

            # Safety check for numerical stability
            if not torch.isfinite(FINALlogits).all():
                self.last_forward_had_nonfinite = True
                self.nonfinite_forward_count += 1
                self._recover_after_nonfinite_forward()
                now = time.time()
                if (now - self.nonfinite_forward_last_log) >= 2.0:
                    print(
                        "⚠️ Non-finite logits detected in forward pass! "
                        f"(count={self.nonfinite_forward_count}, recoveries={self.nonfinite_recovery_count})"
                    )
                    try:
                        logit_norm = FINALlogits.norm()
                        print(f"   Logit norm: {logit_norm.item()}")
                    except Exception:
                        print("   Logit norm: [unavailable]")
                    self.nonfinite_forward_last_log = now
                FINALlogits = torch.nan_to_num(
                    FINALlogits, nan=0.0, posinf=80.0, neginf=-80.0
                )
            else:
                self.last_forward_had_nonfinite = False

            return FINALlogits

    def _recover_after_nonfinite_forward(self):
        with torch.no_grad():
            self.nonfinite_recovery_count += 1
            try:
                self.logTemp.data.clamp_(math.log(0.1), math.log(5.0))
            except Exception:
                pass
            try:
                self.logLR.data.clamp_(math.log(1e-6), math.log(1e-2))
            except Exception:
                pass
            try:
                self.logGradClip.data.clamp_(math.log(0.1), math.log(50.0))
            except Exception:
                pass
            try:
                max_window = max(1.0, float(self.numTokensPerStep))
                self.logRepetitionWindow.data.clamp_(
                    math.log(1.0), math.log(max_window)
                )
            except Exception:
                pass
            try:
                self.repetitionPenalty.data.clamp_(0.5, 3.0)
            except Exception:
                pass
            try:
                self.memory.resetMemory(1.0)
            except Exception:
                pass
            try:
                self.memory2.resetMemory(1.0)
            except Exception:
                pass

    """computes the cross-entropy loss between the models logits and the target token, essentially checking how good the models prediction was"""

    @whocalled
    def computeLoss(
        self,
        _logits,
        _targetTokenIndex,
        _totalAvgAbsDelta=1,
        _learningRateGOAL=learningRateGOAL,
        _perfectTokens=0,
        _training=False,
    ):
        with self.counsellor.infodump("computeLoss") as ʕっʘ‿ʘʔっ:
            self.perfectTokens = _perfectTokens
            self.totalAvgAbsDelta = _totalAvgAbsDelta
            self.learningRateGOAL = _learningRateGOAL
            if skipComputeLoss:
                if debugPrints:
                    ʕっʘ‿ʘʔっ("skipping loss!")
                return torch.tensor(
                    [0.1], requires_grad=True, device=self.device
                )  # Constant scalar tensor

            if debugPrints:
                ʕっʘ‿ʘʔっ("targetTensor")
            targetTensor = torch.tensor(
                [_targetTokenIndex], dtype=torch.long, device=self.device
            )

            debug_print(f"logits shape: {_logits.shape} | target: {_targetTokenIndex}")
            if _logits.dim() == 1:
                _logits = _logits.unsqueeze(0)  # ensure logits are at least 2d

            if debugPrints:
                ʕっʘ‿ʘʔっ("cross Entropy Loss")
            loss = F.cross_entropy(_logits, targetTensor)
            loss_value = loss.detach().item()
            self.CEloss_used = loss_value

            if not torch.isfinite(loss):
                print("NaN/Inf loss detected — logits:", _logits)
                return torch.tensor(10.0, device=self.device, requires_grad=True)

            debug_print(
                f"crossentropy raw loss: {F.cross_entropy(_logits, targetTensor)}"
            )

            self.CELossDelta = loss_value - (
                (self.lastLossBaby) if self.lastLossBaby is not None else 0
            )

            debug_print(f"{self.lastLossBaby:0.1f}", end=", ")  # take delta

            # regulate the learned LR, temperature, repetition penalty (etc) towards target values
            lrSoftClamp = 0.0015 * (self.logLR - math.log(learningRateGOAL)).pow(2)
            # lrSoftClamp = (self.totalAvgAbsDelta ** 1.5) * (self.logLR - math.log(self.learningRateGOAL)).pow(2)
            tempSoftClamp = (loss_value * temperatureSoftClampStrength) * (
                self.logTemp - math.log(temperatureGOAL)
            ).pow(2)
            repetitionPenaltySoftClamp = 0.04 * (
                self.repetitionPenalty - repetitionPenaltyGOAL
            ).pow(2)

            # Creative gate regulation removed

            loss += lrSoftClamp  # use .detach() to avoid .backward()
            self.lrSoftClamp_used = lrSoftClamp.detach().item()
            loss += tempSoftClamp
            self.tempSoftClamp_used = tempSoftClamp.detach().item()
            loss += repetitionPenaltySoftClamp
            self.repPenSoftClamp_used = repetitionPenaltySoftClamp.detach().item()

            self.lastLossBaby = loss.item()
            FINALloss = loss
            debug_print(f"{FINALloss} + loss")

            soft_sample = getattr(self, "_lastSoftSample_for_loss", None)
            if soft_sample is None:
                soft_sample = self.lastSoftSample

            if soft_sample is not None and not skipAuxLoss:
                target = F.one_hot(targetTensor, num_classes=_logits.shape[1]).float()
                # Clamp to avoid log(0) producing -inf and destabilising KL
                eps = 1e-8
                safe_probs = soft_sample.clamp(min=eps)
                kl_loss = F.kl_div(safe_probs.log(), target, reduction="batchmean")
                AUXloss_kl = kl_loss * 0.01
                self.AUXlossKL_used = AUXloss_kl.detach().item()
                # AUXloss = auxLoss * torch.sigmoid(loss - auxLoss) # low weight for anti-dominatrix
                # Ensure cosine similarity is well-defined (avoid zero-norm vectors)
                safe_probs_norm = safe_probs / safe_probs.norm(
                    dim=-1, keepdim=True
                ).clamp_min(eps)
                target_norm = target / target.norm(dim=-1, keepdim=True).clamp_min(eps)
                cosSim = (safe_probs_norm * target_norm).sum(dim=-1)
                AUXloss_cos = 1.0 - cosSim.mean()
                self.AUXlossCos_used = AUXloss_cos.detach().item()
                AUXloss = AUXloss_cos + AUXloss_kl
                debug_print(f"{AUXloss} + aux")
            else:
                self.AUXlossKL_used = 0.0
                self.AUXlossCos_used = 0.0
                AUXloss = 0

            if soft_sample is not None:
                token_freqs = soft_sample.mean(dim=0)
                repLoss_raw = (token_freqs**2).mean()
                repLoss = repLoss_raw * 100.0
                self.repLoss_used = repLoss.detach().item()
                FINALloss += repLoss
                debug_print(f"{FINALloss} repLoss ({repLoss}) + final")
            else:
                self.repLoss_used = 0.0

            if not skipPixels and (
                self.nextPixelTarget is not None and hasattr(self, "pixelPupil")
            ):
                if debugPrints:
                    ʕっʘ‿ʘʔっ(
                        "RGB regression loss with creative synesthetic enhancement"
                    )

                # Handle different tensor shapes properly
                token_embed_for_pixel = getattr(self, "latestTokenEmbed_raw", None)
                if token_embed_for_pixel is None:
                    token_embed_for_pixel = getattr(self, "latestTokenEmbed", None)
                if token_embed_for_pixel is None:
                    debug_print(
                        "latestTokenEmbed is None; using zero embedding for pixel loss"
                    )
                    token_embed_for_pixel = torch.zeros(
                        self.pixelPupil.linear1.in_features, device=self.device
                    )
                else:
                    debug_print(
                        f"latestTokenEmbed is {token_embed_for_pixel} ({token_embed_for_pixel.shape})"
                    )

                if len(token_embed_for_pixel.shape) == 1:
                    # Already 1D embedding - use directly
                    embedding = token_embed_for_pixel
                elif len(token_embed_for_pixel.shape) == 2:
                    # 2D tensor [seq_len, embed_dim] - take the last token
                    embedding = token_embed_for_pixel[-1]
                else:
                    # Unexpected shape - flatten and take appropriate slice
                    embedding = token_embed_for_pixel.flatten()
                    if embedding.size(0) > self.pixelPupil.linear1.in_features:
                        embedding = embedding[: self.pixelPupil.linear1.in_features]

                # Debug tensor shapes before pixelPupil
                debug_print(
                    f"[DEBUG] About to pass embedding to pixelPupil: shape={embedding.shape}, dtype={embedding.dtype}"
                )
                debug_print(
                    f"[DEBUG] pixelPupil.linear1 expects input size: {self.pixelPupil.linear1.in_features}"
                )

                # Ensure embedding is the right shape - should be [embedDimension] for single token
                # Fix potential batch dimension issues
                if len(embedding.shape) == 0:
                    raise RuntimeError(
                        f"Embedding is a scalar! Original latestTokenEmbed shape: {self.latestTokenEmbed.shape}"
                    )
                elif len(embedding.shape) == 2:
                    debug_print(
                        f"[DEBUG] Embedding has 2D shape {embedding.shape}, taking mean across sequence dimension"
                    )
                    embedding = embedding.mean(
                        dim=0
                    )  # Average across sequence if needed
                elif len(embedding.shape) > 2:
                    debug_print(
                        f"[DEBUG] Embedding has unexpected shape {embedding.shape}, flattening to expected size..."
                    )
                    embedding = embedding.flatten()
                    if embedding.size(0) != self.pixelPupil.linear1.in_features:
                        debug_print(
                            f"[DEBUG] Flattened size {embedding.size(0)} doesn't match expected {self.pixelPupil.linear1.in_features}, truncating/padding"
                        )
                        if embedding.size(0) > self.pixelPupil.linear1.in_features:
                            embedding = embedding[: self.pixelPupil.linear1.in_features]
                        else:
                            padding_size = (
                                self.pixelPupil.linear1.in_features - embedding.size(0)
                            )
                            embedding = torch.cat(
                                [
                                    embedding,
                                    torch.zeros(padding_size, device=embedding.device),
                                ]
                            )

                # Ensure we have the right final shape
                if embedding.size(0) != self.pixelPupil.linear1.in_features:
                    raise RuntimeError(
                        f"Embedding final size {embedding.size(0)} doesn't match pixelPupil input size {self.pixelPupil.linear1.in_features}"
                    )

                # Base pixel prediction
                base_predicted_rgb = self.pixelPupil(embedding)

                # Synesthetic enhancement removed; use base prediction
                predictedRGB = base_predicted_rgb
                if debugPrints:
                    debug_print("Using base pixel prediction (no synesthesia)")

                self.predPixel = predictedRGB
                rgbLoss = F.mse_loss(self.predPixel, self.nextPixelTarget)
                # Weight pixel loss safely; avoid 0/0 when both rgbLoss and loss are ~0
                eps = 1e-8
                pixelWeight = rgbLoss / (rgbLoss + loss + eps)
                if not torch.isfinite(pixelWeight):
                    pixelWeight = torch.tensor(0.0, device=self.device)
                self.PIXELloss = max(min((pixelWeight * 1), 1), -1)
                if debugPrints:
                    self.print_rgb_block(self.pixel, "prompt")
                if debugPrints:
                    self.print_rgb_block(predictedRGB, "guess")
                if debugPrints:
                    self.print_rgb_block(self.nextPixelTarget, "truth")
                debug_print(f"{rgbLoss} + rgb")
                debug_print(f"{self.PIXELloss} + pixel")
            else:
                FINALloss = loss
                debug_print(f"{FINALloss} + final")

            # tempSoftClamp = 0.4 * (self.logTemp - math.log(0.5)).pow(2)

            # more tokens (better) > perfTokens > less tokens (worse)
            # HIGHER NUMBER > 2 > LOWER NUMBER
            # 0.3x > 2 > 1.3x

            # worse (explore) > latestlossdelta > better (stay still)
            # POSITIVE NUMBER > 0 > NEGATIVE NUMBER
            # +4 Delta (worse) > 0 > -4 Delta (better)
            # [0-25]x0.1 > 0 > [0-1]
            # 0-2.5 > 0 > 0-1
            if self.cached_sensory is not None and hasattr(self, "sensoryPupil"):
                token_embed_for_sensory = getattr(
                    self, "prevSensoryPredEmbed_raw", None
                )
                if token_embed_for_sensory is None:
                    self.sensoryLoss_used = 0.0
                    self.predSensory = None
                    self.targetSensory = None
                else:
                    if len(token_embed_for_sensory.shape) == 1:
                        embedding = token_embed_for_sensory
                    elif len(token_embed_for_sensory.shape) == 2:
                        embedding = token_embed_for_sensory[-1]
                    else:
                        embedding = token_embed_for_sensory.flatten()
                    if embedding.size(0) != self.sensoryPupil.in_features:
                        if embedding.size(0) > self.sensoryPupil.in_features:
                            embedding = embedding[: self.sensoryPupil.in_features]
                        else:
                            padding_size = (
                                self.sensoryPupil.in_features - embedding.size(0)
                            )
                            embedding = torch.cat(
                                [
                                    embedding,
                                    torch.zeros(padding_size, device=embedding.device),
                                ]
                            )

                    sensory_logits = self.sensoryPupil(embedding)
                    # Plain sigmoid: avoids the double-saturating softsign*3 that
                    # was slamming outputs to ~0.04 / ~0.95 with vanishing gradients.
                    # Logits near zero now produce predictions near 0.5 (the true
                    # stable-environment target), and gradients remain strong.
                    sensory_pred = torch.sigmoid(sensory_logits)
                    # --- Diagnostic: record raw logit stats for dashboard ---
                    # If predictions are still ~0.95/~0.05 after the activation fix,
                    # large values here confirm the checkpoint weights are the culprit.
                    with torch.no_grad():
                        self._sensory_logits_mean   = sensory_logits.mean().item()
                        self._sensory_logits_std    = sensory_logits.std().item() if sensory_logits.numel() > 1 else 0.0
                        self._sensory_logits_absmax = sensory_logits.abs().max().item()
                    if self.cached_device_temp_c is not None:
                        sensory_target = torch.cat(
                            [self.cached_sensory, self.cached_device_temp_c.view(1)],
                            dim=0,
                        )
                    else:
                        sensory_target = torch.cat(
                            [
                                self.cached_sensory,
                                torch.zeros(1, device=self.cached_sensory.device),
                            ],
                            dim=0,
                        )

                    sensory_loss = F.mse_loss(sensory_pred, sensory_target)
                    if not torch.isfinite(sensory_loss):
                        sensory_loss = torch.tensor(0.0, device=self.device)
                    sensory_weight = torch.sigmoid(self.sensory_gate).detach()
                    sensory_loss_scaled = (sensory_loss * 0.1) * (sensory_weight * 0.01)
                    FINALloss += sensory_loss_scaled
                    self.sensoryLoss_used = sensory_loss_scaled.detach().item()
                    self.predSensory = sensory_pred.detach()
                    self.targetSensory = sensory_target.detach()
                    debug_print(
                        f"{sensory_loss} sensory ({sensory_loss_scaled}) + final"
                    )
            else:
                self.sensoryLoss_used = 0.0
                self.predSensory = None
                self.targetSensory = None

            # Detach the token embedding once it's no longer needed for gradient computation
            if self.latestTokenEmbed is not None:
                self.latestTokenEmbed = self.latestTokenEmbed.detach()
            if self.latestTokenEmbed_raw is not None:
                self.latestTokenEmbed_raw = self.latestTokenEmbed_raw.detach()
            if self.prevSensoryPredEmbed_raw is not None:
                self.prevSensoryPredEmbed_raw = self.prevSensoryPredEmbed_raw.detach()
            if self.nextSensoryPredEmbed_raw is not None:
                self.nextSensoryPredEmbed_raw = self.nextSensoryPredEmbed_raw.detach()
            if not skipPixels and (
                self.nextPixelTarget is not None and hasattr(self, "pixelPupil")
            ):
                FINALloss += self.PIXELloss * 0.5
                self.pixelLoss_used = self.PIXELloss * 0.5
                debug_print(f"{FINALloss} pixel + final")

            if soft_sample is not None and not skipAuxLoss:
                if torch.isnan(AUXloss) or not torch.isfinite(AUXloss):
                    print("AUXloss contains NaN!")
                    AUXloss = torch.tensor(0.0, device=self.device)
                FINALloss += AUXloss
                debug_print(f"{FINALloss} aux ({AUXloss}) + final")
            debug_print(
                f"[LOSS DEBUG] requires_grad: {loss.requires_grad} | value: {loss.detach().cpu().item():.4f}"
            )

            # Drop references to the computation graph so successive calls do not accumulate memory.
            self._lastSoftSample_for_loss = None
            if soft_sample is not None:
                self.lastSoftSample = soft_sample.detach()

            if not torch.isfinite(FINALloss):
                print(
                    "computeLoss produced non-finite FINALloss; resetting to fallback."
                )
                FINALloss = torch.tensor(10.0, device=self.device, requires_grad=True)

            return FINALloss

    """backpropagation and optimization, computes gradients of the loss and uses the optimizer to update the models weights"""

    @whocalled
    def backward(self, _loss, _lossDelta, _run_optimizer=True):
        with self.counsellor.infodump("backward") as ʕっʘ‿ʘʔっ:
            collect_grad_stats = self.totalTurns % 100 == 0
            grad_snapshot = None

            if debugPrints:
                tensor_snitch(self, "babyllm backward start")
                tensor_snitch(self.memory, "babyllm backward start")
                tensor_snitch(self.memory2, "babyllm backward start")
                tensor_snitch(self.embed, "babyllm backward start")
                tensor_snitch(self.interneuronNetwork, "babyllm backward start")
                tensor_snitch(self.logits, "babyllm backward start")
                ʕっʘ‿ʘʔっ("print named parameters")
                printTensorAttrs(self, name="babyllm")
                printTensorAttrs(self.memory, name="memory")
                printTensorAttrs(self.memory2, name="memory2")
                printTensorAttrs(self.embed, name="embed")
                printTensorAttrs(self.interneuronNetwork, name="interneuronNetwork")
                printTensorAttrs(self.logits, name="logits")
                for name, p in self.named_parameters():
                    if p.grad is None:
                        ʕっʘ‿ʘʔっ("print no grads")
                        print(
                            f"before = {self.calligraphist.S_apply('dim', f'no grad: {name}')}"
                        )
                    else:
                        if debugPrints:
                            ʕっʘ‿ʘʔっ("set yes grads")
                        stats = get_grad_stats(p.grad)
                        shape = stats["shape"]
                        norm = stats["norm"]
                        sparsity = stats["sparsity"]
                        mean = stats["mean"]
                        std = stats["std"]
                        if debugPrints:
                            ʕっʘ‿ʘʔっ("print yes grads")
                        print(
                            f"before = {self.calligraphist.S_apply('almostPerfect', f'yes grad: {name} | shape: {shape} | norm: {norm:.4f} | sparsity: {sparsity:.2%} | mean: {mean:.4f} | std: {std:.4f}')}"
                        )
                        debug_print("Loss:", _loss.item())

            # If the forward pass had non-finite logits, nan_to_num patched the *value*
            # but NaN is already baked into the computation graph — backward() would
            # propagate NaN gradients to every parameter and destroy all weights in one
            # optimizer step.  Skip the entire backward instead.
            if self.last_forward_had_nonfinite:
                print(
                    "⚠️ [BACKWARD SKIP] forward pass had non-finite logits — skipping backward + step to protect weights"
                )
                self.optimizer.zero_grad()
                self.last_forward_had_nonfinite = False
                return False

            if debugPrints:
                ʕっʘ‿ʘʔっ("loss.backward")
            debug_print(f"windowMAX: {self.numTokensPerStep}")
            _loss.backward()
            debug_print("Logit weights grad norm:", self.logits.l_weights.grad.norm())
            debug_print(
                "LogWindowSizes grad norm:",
                self.interneuronNetwork.logWindowSizes.grad.norm(),
            )
            debug_print(
                "Cerebellum grad norm:", self.interneuronNetwork.cerebellum.grad.norm()
            )
            debug_print(
                "Repetition penalty grad norm:", self.repetitionPenalty.grad.norm()
            )
            # print(next(self.parameters()).grad)

            # Early return if we are accumulating gradients and not stepping the optimizer yet
            if not _run_optimizer:
                return True

            if diagnoseLogitHead:
                def _grad_norm_or_zero(param):
                    grad = getattr(param, "grad", None)
                    if grad is None:
                        return 0.0
                    if not torch.isfinite(grad).all():
                        return float("inf")
                    return float(grad.detach().norm().item())

                with torch.no_grad():
                    self.logits.grad_stats = {
                        "grad_norm_logits.l_weights": _grad_norm_or_zero(self.logits.l_weights),
                        "grad_norm_logits.l_bias": _grad_norm_or_zero(self.logits.l_bias),
                    }
                    if hasattr(self.logits, "logitNorm"):
                        if hasattr(self.logits.logitNorm, "weight") and self.logits.logitNorm.weight is not None:
                            self.logits.grad_stats["grad_norm_logits.logitNorm.weight"] = _grad_norm_or_zero(self.logits.logitNorm.weight)
                        if hasattr(self.logits.logitNorm, "bias") and self.logits.logitNorm.bias is not None:
                            self.logits.grad_stats["grad_norm_logits.logitNorm.bias"] = _grad_norm_or_zero(self.logits.logitNorm.bias)

            if diagnoseGradientSources:
                with torch.no_grad():
                    groups = {
                        "embed": [],
                        "attention": [],
                        "attention2": [],
                        "interneuronNetwork.neurons": [],
                        "interneuronNetwork.refinement2": [],
                        "interneuronNetwork.window/cerebellum": [],
                        "memory": [],
                        "memory2": [],
                        "scratchpad": [],
                        "tangling": [],
                        "logits": [],
                        "pixelPupil": [],
                        "sensoryPupil": [],
                        "sensoryEmbed": [],
                        "scalar/control": [],
                        "other": []
                    }

                    control_param_names = {
                        "logLR", "logGradClip", "scheduledSamplingRate", "temperature", "repetitionPenalty",
                        "logMemoryLength", "logMemory2Length", "logRepetitionWindow", "sensory_scale", "sensory_bias",
                        "temp_scale", "temp_bias", "logWindowSizes", "windowFractionality", "cerebellum",
                        "windowFractionality_short", "cerebellum_short"
                    }

                    for name, p in self.named_parameters():
                        if p.grad is None:
                            continue
                        
                        group = "other"
                        if name.startswith("ar_"):
                            group = "AR temporal tangle"
                        elif any(k in name for k in control_param_names) or p.numel() == 1:
                            group = "scalar/control"
                        elif name.startswith("embed."):
                            group = "embed"
                        elif name.startswith("attention2."):
                            group = "attention2"
                        elif name.startswith("attention."):
                            group = "attention"
                        elif name.startswith("interneuronNetwork.neurons."):
                            group = "interneuronNetwork.neurons"
                        elif name.startswith("interneuronNetwork.refinement2."):
                            group = "interneuronNetwork.refinement2"
                        elif name.startswith("interneuronNetwork.") and any(x in name for x in ["window", "cerebellum", "Window", "Cerebellum"]):
                            group = "interneuronNetwork.window/cerebellum"
                        elif name.startswith("interneuronNetwork."):
                            if "neurons" in name:
                                group = "interneuronNetwork.neurons"
                            elif "refinement2" in name:
                                group = "interneuronNetwork.refinement2"
                            else:
                                group = "interneuronNetwork.window/cerebellum"
                        elif name.startswith("memory2."):
                            group = "memory2"
                        elif name.startswith("memory."):
                            group = "memory"
                        elif name.startswith("scratchpad."):
                            group = "scratchpad"
                        elif name.startswith("tangling."):
                            group = "tangling"
                        elif name.startswith("logits."):
                            group = "logits"
                        elif name.startswith("pixelPupil."):
                            group = "pixelPupil"
                        elif name.startswith("sensoryPupil."):
                            group = "sensoryPupil"
                        elif name.startswith("sensoryEmbed."):
                            group = "sensoryEmbed"
                        else:
                            parts = name.split(".")
                            if parts:
                                group = parts[0]
                            else:
                                group = "other"

                        if group not in groups:
                            groups[group] = []
                        groups[group].append(p.grad.data)

                    total_sq_norm = 0.0
                    group_l2_norms = {}
                    for group_name, grads in groups.items():
                        if not grads:
                            continue
                        sq_sum = 0.0
                        for g in grads:
                            sq_sum += float(g.norm(2).item()) ** 2
                        total_sq_norm += sq_sum
                        group_l2_norms[group_name] = sq_sum ** 0.5

                    global_l2_norm = total_sq_norm ** 0.5

                    sorted_groups = sorted(group_l2_norms.items(), key=lambda x: x[1], reverse=True)

                    self.gradient_leaderboard = []
                    for group_name, norm_val in sorted_groups[:20]:
                        pct = (norm_val / global_l2_norm * 100.0) if global_l2_norm > 0 else 0.0
                        self.gradient_leaderboard.append((group_name, float(norm_val), float(pct)))

            # --- MOVE GRAD SNAPSHOT/REPORTING HERE (after backward, before zero_grad) ---
            if collect_grad_stats:
                grad_snapshot = self._snapshot_gradients()
                grad_total_norm = None
                if grad_snapshot:
                    grad_log_output = ["\n--- Gradient Snapshot (pre-zero_grad) ---"]
                    for name, stats in grad_snapshot:
                        norm_val = stats["norm"]
                        sparsity_val = stats["sparsity"]
                        mean_val = stats["mean"]
                        std_val = stats["std"]
                        norm_style = self.calligraphist.S_getStat(
                            f"{name}_norm", norm_val
                        )
                        sparsity_style = self.calligraphist.S_getStat(
                            f"{name}_sparsity", sparsity_val
                        )
                        mean_style = self.calligraphist.S_getStat(
                            f"{name}_mean", mean_val
                        )
                        std_style = self.calligraphist.S_getStat(f"{name}_std", std_val)
                        grad_log_output.append(
                            f"{name:<50} | "
                            f"norm: {self.calligraphist.S_apply(norm_style, f'{norm_val:.6f}')} | "
                            f"sparsity: {self.calligraphist.S_apply(sparsity_style, f'{sparsity_val:.6%}')} | "
                            f"mean: {self.calligraphist.S_apply(mean_style, f'{mean_val:.6f}')} | "
                            f"std: {self.calligraphist.S_apply(std_style, f'{std_val:.6f}')}"
                        )
                    print("\n".join(grad_log_output))
                else:
                    print(
                        "\n--- Gradient Snapshot (pre-zero_grad) ---\n(no gradients recorded)"
                    )

            if debugPrints:
                if debugPrints:
                    ʕっʘ‿ʘʔっ("print named parameters")
                printTensorAttrs(self, name="babyllm")
                printTensorAttrs(self.memory, name="memory")
                printTensorAttrs(self.memory2, name="memory2")
                printTensorAttrs(self.embed, name="embed")
                printTensorAttrs(self.interneuronNetwork, name="interneuronNetwork")
                printTensorAttrs(self.logits, name="logits")
                for name, p in self.named_parameters():
                    if p.grad is None:
                        if debugPrints:
                            ʕっʘ‿ʘʔっ("print no grads")
                        print(
                            f"after = {self.calligraphist.S_apply('emergency', f'NO GRAD: {name}')}"
                        )
                    else:
                        if debugPrints:
                            ʕっʘ‿ʘʔっ("set yes grads")
                        stats = get_grad_stats(p.grad)
                        shape = stats["shape"]
                        norm = stats["norm"]
                        sparsity = stats["sparsity"]
                        mean = stats["mean"]
                        std = stats["std"]
                        if debugPrints:
                            ʕっʘ‿ʘʔっ("print yes grads")
                        print(
                            f"after = {self.calligraphist.S_apply('almostPerfect', f'yes grad: {name} | shape: {shape} | norm: {norm:.4f} | sparsity: {sparsity:.2%} | mean: {mean:.4f} | std: {std:.4f}')}"
                        )
            if debugPrints:
                ʕっʘ‿ʘʔっ("torch.no_grad")
            with torch.no_grad():  # RESET LEARNABLE PARAMETERS
                # self.logLR.data.fill_(math.log(0.00035))  # Learning rate back to 1e-4
                if debugPrints:
                    ʕっʘ‿ʘʔっ("fill scheduledSamplingRate")
                # self.scheduledSamplingRate.data.fill_(0.02)  # Scheduled sampling full (no scheduled sampling yet)
                # self.temperature.data.fill_(math.exp(self.logTemp))  # Temperature normal
                # self.repetitionPenalty.data.fill_(1.0)  # Repetition penalty normal
                # self.logMemoryLength.data.fill_(math.log(5))  # Memory length default
                # self.logRepetitionWindow.data.fill_(math.log(16))  # Repetition window default
                # self.interneuronNetwork.logWindowSizes.data.copy_(
                #    torch.log(torch.tensor(allWindowSizes_new, dtype = torch.float32, device = self.device))
                # )
                # for module in self.interneuronNetwork.windowMeta:
                #    if isinstance(module, torch.nn.Linear):
            #        module.reset_parameters()

            if True:
                if debugPrints:
                    ʕっʘ‿ʘʔっ("torch.no_grad")
                if debugPrints:
                    ʕっʘ‿ʘʔっ("clamp logLR")
                clamp_param(
                    self.logLR, math.log(0.0001), math.log(0.001)
                )  # CLAMP IT! IN MEMORY OF THE AMAZING 1.00 SELF LEARNED LOSS RUN OF 27-APRIL-2025! - you certainly dropped the delta! you win!
                if debugPrints:
                    ʕっʘ‿ʘʔっ("set self.memoryLength")
                self.memoryLength = torch.sigmoid(
                    (self.totalTurns - torch.exp(self.logMemoryLength)) * 0.5
                )
                if debugPrints:
                    ʕっʘ‿ʘʔっ("set self.memoryLength2")
                self.memory2Length = torch.sigmoid(
                    (self.totalTurns - torch.exp(self.logMemory2Length)) * 0.5
                )
                mem_source = (
                    self.latest_sensory_vector
                    if self.latest_sensory_vector is not None
                    else self.cached_sensory
                )
                mem_scale = self._sensory_nudge(mem_source, 3, 0.02)
                if mem_scale != 1.0:
                    self.memoryLength = (self.memoryLength * mem_scale).clamp(0.0, 1.0)
                    self.memory2Length = (self.memory2Length * mem_scale).clamp(
                        0.0, 1.0
                    )
                if debugPrints:
                    ʕっʘ‿ʘʔっ("set learnedLR")
                learnedLR = torch.exp(self.logLR).item()
                for g in self.optimizer.param_groups:
                    if debugPrints:
                        ʕっʘ‿ʘʔっ("update self.optimizer.param_groups")
                    g["lr"] = learnedLR  # send the learned LR to the optimizer
                # self.gradientClipMaxNorm = torch.exp(self.logGradClip).item()
                # self.repetitionWindow = torch.exp(self.logRepetitionWindow).item()
                # self.logLR.data.fill_(self.logLR+0.000001) # increment LR manually (break grid)

            if debugPrints:
                ʕっʘ‿ʘʔっ("clip_grad_norm")
            with torch.no_grad():
                base_clip = 1.5  # was 5.0 — restored to safer range
                sensitivity = 0.5  # was 2.5 — less reactive to loss swings

                lossDelta_tensor = torch.tensor(_lossDelta, device=self.device)
                adjustment = lossDelta_tensor * sensitivity
                clipValue = (base_clip + adjustment).clamp(
                    min=0.5, max=2.0
                )  # max was 10.0; back to ~2.0

            # Rare-event excitement microscope.
            # Measure first, while ALL gradients are still exactly pre-clip.
            _excite_preclip = self._excite_current_grad_norm()
            if (
                math.isfinite(_excite_preclip)
                and _excite_preclip > self._excite_threshold
            ):
                self._print_excitement_report(
                    _excite_preclip,
                    clipValue.item(),
                )

            # --- EMBED GRAD DIAGNOSTIC (pre-clip) ---
            for _name, _p in self.embed.named_parameters():
                if _p.grad is not None:
                    _g = _p.grad.detach()
                    print(f"EMBED PARAM GRAD {_name}: norm={_g.norm().item():.6f} max={_g.abs().max().item():.6f} mean={_g.abs().mean().item():.6f} nonzero={_g.count_nonzero().item()} numel={_g.numel()}")

            # Clip gradients BEFORE the lock to prevent NaNs
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), max_norm=clipValue.item()
            )
            self.gradientClipMaxNorm = clipValue.item()
            self.last_grad_norm_before_clip = total_grad_norm.item() if hasattr(total_grad_norm, "item") else float(total_grad_norm)
            
            # Compute gradient norm after clipping
            total_grad_norm_after = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm_after += param_norm.item() ** 2
            self.last_grad_norm_after_clip = total_grad_norm_after ** 0.5

            # clip_grad_norm_ returns inf when any gradient contains NaN — it does NOT
            # zero them out.  Check here so Adan never sees NaN inputs (which would corrupt
            # all three momentum buffers and every parameter in one step).
            if not torch.isfinite(total_grad_norm):
                nan_params = [
                    n
                    for n, p in self.named_parameters()
                    if p.grad is not None and not torch.isfinite(p.grad).all()
                ]
                print(
                    f"⚠️ [STEP SKIP] NaN/Inf in gradients ({len(nan_params)} params) — "
                    f"skipping optimizer.step() to protect weights. "
                    f"First offenders: {nan_params[:5]}"
                )
                self.optimizer.zero_grad()
                return False

            if debugPrints:
                ʕっʘ‿ʘʔっ(
                    "optimizer.step"
                )  # Acquire the lock only for the weight update step
            with self.model_thread_lock:
                self.optimizer.step()
                with torch.no_grad():
                    inn = self.interneuronNetwork

                    # Replaces the old per-token clamps from NEURON/INN forward().
                    weight_norm = inn.neurons.n_weights.norm(dim=1, keepdim=True)
                    inn.neurons.n_weights.div_(
                        weight_norm.clamp(min=1.0, max=100.0)
                    )
                    inn.windowFractionality.clamp_(-3.0, 3.0)
                    inn.cerebellum.clamp_(0.01, 0.99)
                    inn.windowFractionality_short.clamp_(-3.0, 3.0)
                    inn.cerebellum_short.clamp_(0.01, 0.99)
            self.optimizer.zero_grad()

            # These token records belong only to the optimiser step that just
            # completed. Clearing here preserves gradient-accumulation chunks.
            self._excite_token_losses.clear()

            # CRITICAL: Clamp sensory/temperature parameters to prevent explosion
            with torch.no_grad():
                # Emergency reset if parameters have exploded beyond recovery
                if self.sensory_scale.abs().max() > 100.0:
                    print(
                        f"⚠️ EMERGENCY: sensory_scale exploded to {self.sensory_scale.abs().max():.0f}, resetting!"
                    )
                    self.sensory_scale.fill_(1e-5)
                if self.sensory_bias.abs().max() > 100.0:
                    print(
                        f"⚠️ EMERGENCY: sensory_bias exploded to {self.sensory_bias.abs().max():.0f}, resetting!"
                    )
                    self.sensory_bias.zero_()
                if self.temperature_scale.abs() > 100.0:
                    print(
                        f"⚠️ EMERGENCY: temperature_scale exploded to {self.temperature_scale.abs():.0f}, resetting!"
                    )
                    self.temperature_scale.fill_(1e-5)
                if self.temperature_bias.abs() > 100.0:
                    print(
                        f"⚠️ EMERGENCY: temperature_bias exploded to {self.temperature_bias.abs():.0f}, resetting!"
                    )
                    self.temperature_bias.zero_()
                if self.temperature_vector.abs().max() > 100.0:
                    print(
                        f"⚠️ EMERGENCY: temperature_vector exploded to {self.temperature_vector.abs().max():.0f}, resetting!"
                    )
                    self.temperature_vector.fill_(1e-5)

                # Normal clamping for ongoing training
                self.sensory_scale.clamp_(-10.0, 10.0)
                self.sensory_bias.clamp_(-10.0, 10.0)
                self.temperature_scale.clamp_(-10.0, 10.0)
                self.temperature_bias.clamp_(-10.0, 10.0)
                self.temperature_vector.clamp_(-10.0, 10.0)

            if collect_grad_stats:
                grad_snapshot = self._snapshot_gradients()
                grad_total_norm = float(total_grad_norm)
            else:
                grad_total_norm = None

            if debugPrints:
                ʕっʘ‿ʘʔっ("torch.exp(self.logRepetionWindow)")
            repWindow = torch.exp(self.logRepetitionWindow)
            if debugPrints:
                ʕっʘ‿ʘʔっ("set self.repetitionWindow")
            self.repetitionWindow = repWindow / (
                1 + repWindow / self.numTokensPerStep
            )  # asymptotes near windowMAX

            if debugPrints:
                ʕっʘ‿ʘʔっ("set backwardStats")
            if True:
                _bwd_params = torch.stack(
                    [
                        torch.exp(self.logMemoryLength),
                        torch.exp(self.logMemory2Length),
                        self.repetitionWindow,
                        torch.exp(self.logTemp),
                    ]
                ).tolist()
                self.backwardStats = {
                    "B_floatMemoryLength": _bwd_params[0],
                    "B_floatMemory2Length": _bwd_params[1],
                    # "B_expWindow": repWindow.item(),
                    "B_repetitionWindow": _bwd_params[2],
                    "B_temperature": _bwd_params[3],
                    "L_CEloss": self.CEloss_used,
                    "L_PIXELloss": self.PIXELloss.detach().item() if torch.is_tensor(self.PIXELloss) else self.PIXELloss,
                    "L_PIXELloss_scaled": self.pixelLoss_used.detach().item() if torch.is_tensor(self.pixelLoss_used) else self.pixelLoss_used,
                    "L_AUXlossCos": self.AUXlossCos_used,
                    "L_AUXlossKL": self.AUXlossKL_used,
                    "L_LRclamp": self.lrSoftClamp_used,
                    "L_tempClamp": self.tempSoftClamp_used,
                    "L_repPenClamp": self.repPenSoftClamp_used,
                    "L_repLoss": self.repLoss_used,
                    "L_sensoryLoss": self.sensoryLoss_used,
                    # Raw sensory-logit diagnostics — healthy range is roughly ±2.
                    # Values ≫ 3 mean the sigmoid is saturated and predictions will
                    # still read ~0.95 / ~0.05 regardless of the activation fix.
                    "L_sens_logit_mean":   getattr(self, "_sensory_logits_mean",   0.0),
                    "L_sens_logit_std":    getattr(self, "_sensory_logits_std",    0.0),
                    "L_sens_logit_absmax": getattr(self, "_sensory_logits_absmax", 0.0),
                    "L_sensPupil_w_norm":  self.sensoryPupil.weight.norm().item(),
                    "L_sensPupil_b_norm":  self.sensoryPupil.bias.norm().item() if self.sensoryPupil.bias is not None else 0.0,
                    "B_gradClip": self.gradientClipMaxNorm,
                }
                if debugPrints:
                    ʕっʘ‿ʘʔっ("update self.stats with self.backwardStats")
                self.stats.update(self.backwardStats)

            if collect_grad_stats:
                if grad_snapshot:
                    grad_log_output = ["\n--- Gradient Snapshot ---"]
                    for name, stats in grad_snapshot:
                        norm_val = stats["norm"]
                        sparsity_val = stats["sparsity"]
                        mean_val = stats["mean"]
                        std_val = stats["std"]
                        norm_style = self.calligraphist.S_getStat(
                            f"{name}_norm", norm_val
                        )
                        sparsity_style = self.calligraphist.S_getStat(
                            f"{name}_sparsity", sparsity_val
                        )
                        mean_style = self.calligraphist.S_getStat(
                            f"{name}_mean", mean_val
                        )
                        std_style = self.calligraphist.S_getStat(f"{name}_std", std_val)
                        grad_log_output.append(
                            f"{name:<50} | "
                            f"norm: {self.calligraphist.S_apply(norm_style, f'{norm_val:.6f}')} | "
                            f"sparsity: {self.calligraphist.S_apply(sparsity_style, f'{sparsity_val:.6%}')} | "
                            f"mean: {self.calligraphist.S_apply(mean_style, f'{mean_val:.6f}')} | "
                            f"std: {self.calligraphist.S_apply(std_style, f'{std_val:.6f}')}"
                        )
                    if grad_total_norm is not None:
                        grad_log_output.append(
                            f"total grad norm: {grad_total_norm:.6f}"
                        )
                    print("\n".join(grad_log_output))
                else:
                    print("\n--- Gradient Snapshot ---\n(no gradients recorded)")
            # self.log_all_learnable_params(prefix="BACKWARD_")
            self.pixelLoss_used = 0
            if hasattr(self, "predPixel") and self.predPixel is not None:
                self.predPixel = self.predPixel.detach()
            if torch.is_tensor(self.PIXELloss):
                self.PIXELloss = self.PIXELloss.detach().item()

            # with torch.no_grad(): # FORCE RESET THE MEMORY GATES IF OVER USING LONG
            # self.memory.currentGate.data = self.memory.currentGate.data.abs()
            # self.memory.shortGate.data = self.memory.shortGate.data.abs()

            if debugPrints:
                tensor_snitch(self, "babyllm backward end")
                tensor_snitch(self.memory, "babyllm backward end")
                tensor_snitch(self.memory2, "babyllm backward end")
                tensor_snitch(self.embed, "babyllm backward end")
                tensor_snitch(self.interneuronNetwork, "babyllm backward end")
                tensor_snitch(self.logits, "babyllm backward end")

    @whocalled
    def getResponseFromLogits(
        self, _logits, _training=False, _totAvgAbsDelta=0.0, _use_lock: bool = True
    ):
        with self.counsellor.infodump("getResponseFromLogits") as ʕっʘ‿ʘʔっ:
            lock_ctx = self.model_thread_lock if _use_lock else nullcontext()
            with lock_ctx:
                # Ensure incoming logits are finite
                if not torch.isfinite(_logits).all():
                    _logits = torch.nan_to_num(
                        _logits, nan=0.0, posinf=1e3, neginf=-1e3
                    )

                # Clamp temperature to a safe, non-zero range
                raw_temp = torch.exp(self.logTemp)
                safe_temp = raw_temp.clamp(min=0.1, max=5.0)
                temp_scale = getattr(self, "sensory_temp_scale", 1.0)
                safe_temp = (safe_temp * temp_scale).clamp(min=0.1, max=5.0)
                # Keep attrs up-to-date for any downstream consumers
                self.temperature = safe_temp
                self.interneuronNetwork.temperature = safe_temp

                # Scale logits and sanitize again to avoid inf/NaN after division
                logits_scaled = _logits / safe_temp
                logits_scaled = torch.nan_to_num(
                    logits_scaled, nan=0.0, posinf=1e3, neginf=-1e3
                )
                # Optional safety clamp to keep within softmax-stable range
                logits_scaled = logits_scaled.clamp(min=-80.0, max=80.0)

                if logits_scaled.dim() == 1:
                    logits_scaled = logits_scaled.unsqueeze(0)

                # Gumbel-Softmax (robust to tiny tau)
                try:
                    tau = float(safe_temp.detach().cpu().item())
                    tau = max(tau, 1e-2)
                    base_probs = F.gumbel_softmax(logits_scaled, tau=tau, hard=False)
                    assert torch.isfinite(base_probs).all(), (
                        "gumbelProbs has NaN or Inf!"
                    )
                except Exception as e:
                    self.gumBellend += 1
                    debug_print(f"Gumbel softmax failed: {e}. Falling back to softmax.")
                    base_probs = F.softmax(logits_scaled, dim=-1)
                # Clamp and renormalise to avoid zeros that cause log(0) downstream
                eps = 1e-8
                base_probs = torch.nan_to_num(base_probs, nan=0.0)
                base_probs = base_probs.clamp(min=eps)
                base_probs = base_probs / base_probs.sum(
                    dim=-1, keepdim=True
                ).clamp_min(eps)

                if _training:
                    self._lastSoftSample_for_loss = base_probs
                self.lastSoftSample = base_probs.detach()

                if _training:
                    with torch.no_grad():
                        # Existing creativity metrics
                        a = self.memory.FINALmemory
                        b = self.memory2.FINALmemory
                        if a.dim() == 2 and a.size(0) > 1:
                            a = a.mean(dim=0, keepdim=True)
                        if b.dim() == 2 and b.size(0) > 1:
                            b = b.mean(dim=0, keepdim=True)
                        denom = (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(eps)
                        cos_val = (a * b).sum(dim=-1) / denom
                        cos_val = torch.nan_to_num(cos_val, nan=0.0)
                        self.memoryFlux = (1 - cos_val).item()
                        self.cerebralLoad = (
                            self.interneuronNetwork.cerebellum.std().item()
                        )
                        self.learningStability = _totAvgAbsDelta
                        self.dreamIntensity = (
                            (self.memoryFlux * 2.0)
                            + (self.cerebralLoad * 5.0)
                            + (self.learningStability * 1.0)
                        )

                # Simplified sampling without creative modules
                augmented_probs = base_probs.clone()
                top_p = 0.92
                sorted_probs, sorted_indices = torch.sort(
                    augmented_probs, descending=True
                )
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                augmented_probs[indices_to_remove] = 0

                if _training:
                    # take the argmax from the *augmented* distribution - can still influence the training choice
                    responseFromLogits = augmented_probs.argmax(dim=1, keepdim=True)
                else:
                    if torch.sum(augmented_probs) > 0:
                        responseFromLogits = torch.multinomial(
                            augmented_probs, num_samples=1
                        )
                    else:
                        responseFromLogits = torch.topk(base_probs, 1).indices

                repWindow = torch.exp(self.logRepetitionWindow).item()
                effective_repWindow = repWindow / (
                    1 + repWindow / self.numTokensPerStep
                )
                self.recentGeneratedTokens.append(responseFromLogits.item())
                if len(self.recentGeneratedTokens) > int(effective_repWindow):
                    self.recentGeneratedTokens.pop(0)

                return responseFromLogits

    def forward_and_sample(
        self,
        _inputSeq,
        _pixel=None,
        _training=False,
        _totAvgAbsDelta=0.0,
        _arInputSeq=None,
    ):
        """Run ``forward`` and ``getResponseFromLogits`` while holding the model lock once."""

        with self.model_thread_lock:
            logits = self.forward(
                _inputSeq,
                _pixel=_pixel,
                _use_lock=False,
                _arInputSeq=_arInputSeq,
            )
            response = self.getResponseFromLogits(
                logits,
                _training=_training,
                _totAvgAbsDelta=_totAvgAbsDelta,
                _use_lock=False,
            )
        return logits, response

    def _trace_clone_value(self, value):
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, deque):
            return deque(list(value), maxlen=value.maxlen)
        if isinstance(value, list):
            return list(value)
        if isinstance(value, dict):
            return dict(value)
        return value

    def _trace_scalar(self, value, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        if torch.is_tensor(value):
            if value.numel() == 0:
                return float(default)
            detached = value.detach()
            return (
                float(detached.mean().item())
                if detached.numel() > 1
                else float(detached.item())
            )
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _trace_tensor_norm(self, tensor) -> float:
        if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
            return 0.0
        return float(tensor.detach().norm().item())

    def _trace_last_token_norm(self, tensor) -> float:
        if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
            return 0.0
        detached = tensor.detach()
        if detached.dim() == 1:
            focus = detached
        elif detached.dim() == 2:
            focus = detached[-1]
        elif detached.dim() == 3:
            focus = detached[0, -1]
        else:
            focus = detached.reshape(-1)
        return float(focus.norm().item())

    def _trace_per_token_norms(self, tensor):
        if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
            return []
        detached = tensor.detach()
        if detached.dim() == 1:
            return [float(detached.norm().item())]
        if detached.dim() == 3:
            detached = detached[0]
        if detached.dim() != 2:
            return [float(detached.reshape(-1).norm().item())]
        return [float(row.norm().item()) for row in detached]

    def _trace_snapshot_histories(self, owner, attr_names):
        snapshot = {}
        for attr in attr_names:
            value = getattr(owner, attr, None)
            if isinstance(value, deque):
                snapshot[attr] = (list(value), value.maxlen)
        return snapshot

    def _trace_restore_histories(self, owner, snapshot):
        for attr, (items, maxlen) in (snapshot or {}).items():
            setattr(owner, attr, deque(items, maxlen=maxlen))

    def _trace_snapshot_module_state(
        self, module, *, buffer_names=None, attr_names=None
    ):
        buffer_names = buffer_names or []
        attr_names = attr_names or []
        state = {
            "stats": dict(getattr(module, "stats", {}) or {}),
            "histories": self._trace_snapshot_histories(
                module, getattr(module, "_history_attrs", [])
            ),
            "buffers": {},
            "attrs": {},
        }
        for name in buffer_names:
            if hasattr(module, name):
                state["buffers"][name] = self._trace_clone_value(getattr(module, name))
        for name in attr_names:
            state["attrs"][name] = self._trace_clone_value(getattr(module, name, None))
        return state

    def _trace_restore_module_state(self, module, state):
        if not state:
            return
        if hasattr(module, "stats"):
            module.stats = dict(state.get("stats", {}) or {})
        self._trace_restore_histories(module, state.get("histories"))
        with torch.no_grad():
            for name, value in state.get("buffers", {}).items():
                current = getattr(module, name, None)
                if (
                    torch.is_tensor(current)
                    and torch.is_tensor(value)
                    and current.shape == value.shape
                ):
                    current.copy_(value.to(current.device))
                else:
                    setattr(module, name, self._trace_clone_value(value))
            for name, value in state.get("attrs", {}).items():
                setattr(module, name, self._trace_clone_value(value))

    def trace_forward(
        self,
        _inputSeq,
        _pixel=None,
        top_k: int = 5,
        include_distribution: bool = False,
        include_vectors: bool = False,
    ):
        """Run a no-grad forward trace and restore mutable state afterwards."""

        if _inputSeq is None:
            raise ValueError("trace_forward needs input token ids")

        with self.model_thread_lock:
            with torch.no_grad():
                input_seq = _inputSeq.to(device=self.device, dtype=torch.long).flatten()
                if input_seq.numel() == 0:
                    raise ValueError("trace_forward needs at least one token")

                was_training = self.training
                model_history_names = [
                    "inputEmbedsHistory",
                    "INNOutputHistory",
                    "memoryOutputHistory",
                    "memory2OutputHistory",
                    "penalisedOutputHistory",
                    "FINALlogitsHistory",
                    "normalisedHistory",
                    "charEmbedHistory",
                ]
                model_snapshot = {
                    "stats": dict(self.stats or {}),
                    "forwardStats": dict(getattr(self, "forwardStats", {}) or {}),
                    "histories": self._trace_snapshot_histories(
                        self, model_history_names
                    ),
                    "recentGeneratedTokens": list(self.recentGeneratedTokens),
                    "attrs": {
                        name: self._trace_clone_value(getattr(self, name, None))
                        for name in [
                            "prevSensoryPredEmbed_raw",
                            "nextSensoryPredEmbed_raw",
                            "latestTokenEmbed_raw",
                            "latestTokenEmbed",
                            "predSensory",
                            "targetSensory",
                            "latest_sensory_vector",
                            "latest_device_temp_c",
                            "sensory_gate_used",
                            "sensory_temp_scale",
                            "temperature",
                            "last_forward_had_nonfinite",
                            "nonfinite_forward_count",
                            "nonfinite_forward_last_log",
                            "nonfinite_recovery_count",
                        ]
                    },
                }
                attention_snapshot = self._trace_snapshot_module_state(
                    self.attention,
                    attr_names=["gate_nudge"],
                )
                attention2_snapshot = self._trace_snapshot_module_state(
                    self.attention2,
                    attr_names=["gate_nudge"],
                )
                inn_snapshot = self._trace_snapshot_module_state(
                    self.interneuronNetwork,
                    attr_names=[
                        "temperature",
                        "entropyBonus",
                        "windowTensor_used",
                        "floatWindowSizes_used",
                        "windowTensor_short_used",
                        "floatWindowSizes_short_used",
                        "short_gate_used",
                        "softmax_temp_used",
                        "softmax_temp_short_used",
                        "cerebellumSoft",
                        "cerebellumSoft_short",
                        "windowSizeEntropy",
                        "windowEntropy",
                        "rangePenalty",
                        "meanPenalty",
                        "windowWeightSpread",
                    ],
                )
                neuron_snapshot = self._trace_snapshot_module_state(
                    self.interneuronNetwork.neurons
                )
                memory_snapshot = self._trace_snapshot_module_state(
                    self.memory,
                    buffer_names=["shortTermMemory", "longTermMemory"],
                    attr_names=[
                        "newShort",
                        "newLong",
                        "activationsTensor",
                        "gatedMemory",
                        "FINALmemory",
                        "short_used",
                        "long_used",
                        "act_used",
                        "mem_used",
                        "shortDecay_used",
                        "longDecay_used",
                    ],
                )
                memory2_snapshot = self._trace_snapshot_module_state(
                    self.memory2,
                    buffer_names=["shortTermMemory", "longTermMemory"],
                    attr_names=[
                        "newShort",
                        "newLong",
                        "activationsTensor",
                        "gatedMemory",
                        "FINALmemory",
                        "short_used",
                        "long_used",
                        "act_used",
                        "mem_used",
                        "shortDecay_used",
                        "longDecay_used",
                    ],
                )
                scratchpad_snapshot = self._trace_snapshot_module_state(
                    self.scratchpad,
                    buffer_names=["buffer", "slot_usage"],
                )
                tangling_snapshot = self._trace_snapshot_module_state(
                    self.tangling,
                    attr_names=[
                        "current_stage",
                        "floatWindowSizes_used",
                        "windowTensor_used",
                    ],
                )
                logits_snapshot = self._trace_snapshot_module_state(self.logits)

                try:
                    self.eval()

                    sensory_source = self.cached_sensory
                    motion_scale = self._sensory_nudge(sensory_source, 5, 0.02)
                    attention_scale = self._sensory_nudge(sensory_source, 8, 0.02)
                    char_scale = self._sensory_nudge(sensory_source, 7)
                    pos_scale = self._sensory_nudge(sensory_source, 6)
                    noise_scale = self._sensory_nudge(sensory_source, 1, 0.02)

                    self.sensory_temp_scale = motion_scale
                    self.attention.gate_nudge = attention_scale
                    self.attention2.gate_nudge = attention_scale
                    self.temperature = torch.exp(self.logTemp) * motion_scale
                    self.interneuronNetwork.temperature = self.temperature

                    if self.cached_sensory is not None:
                        sensory_vector = self.cached_sensory
                        sensory_scale_used = torch.sigmoid(self.sensory_scale) * 10.0
                        sensory_bias_used = (
                            torch.sigmoid(self.sensory_bias) - 0.5
                        ) * 10.0
                        sensory_adjusted = (
                            sensory_vector * sensory_scale_used
                        ) + sensory_bias_used
                        sensory_embed = self.sensoryEmbed(sensory_adjusted)
                        sensory_embed = self.sensoryEmbed_norm(sensory_embed)
                        sensory_embed = torch.tanh(sensory_embed) * 10.0
                        gate = torch.sigmoid(self.sensory_gate)
                    else:
                        sensory_embed = 0.0
                        gate = 0.0

                    if self.cached_device_temp_c is not None:
                        temp_value = self.cached_device_temp_c
                        temp_scale_used = torch.sigmoid(self.temperature_scale) * 5.0
                        temp_bias_used = (
                            torch.sigmoid(self.temperature_bias) - 0.5
                        ) * 5.0
                        temp_vec_used = torch.sigmoid(self.temperature_vector) * 5.0
                        temp_scaled = (temp_value * temp_scale_used) + temp_bias_used
                        temp_embed = temp_scaled * temp_vec_used
                        if torch.is_tensor(sensory_embed):
                            sensory_embed = sensory_embed + temp_embed
                        else:
                            sensory_embed = temp_embed

                    tokenEmbed = self.embed(_tokenIndex=input_seq)
                    seq_len = int(tokenEmbed.shape[0])

                    padded_byte_tensor = F.embedding(input_seq, self.char_lookup_data)
                    attention_mask = F.embedding(input_seq, self.char_mask_data)
                    embedded_chars = self.char_embed(padded_byte_tensor)
                    embedded_chars = embedded_chars * attention_mask.unsqueeze(-1)
                    summed_vectors = embedded_chars.sum(dim=1)
                    real_lengths = attention_mask.sum(dim=1, keepdim=True).clamp(
                        min=1.0
                    )
                    char_vector_batch = summed_vectors / real_lengths
                    charEmbed = self.char_projector(char_vector_batch)
                    if char_scale != 1.0:
                        charEmbed = charEmbed * char_scale

                    pos_indices = torch.arange(seq_len, device=tokenEmbed.device)
                    posEmbed = self.embed.posEmbedding(pos_indices)
                    posEmbed = self.embed.posDropout(posEmbed * self.embed.scale)
                    if pos_scale != 1.0:
                        posEmbed = posEmbed * pos_scale

                    all_blend_weights = torch.cat(
                        [self.inputBlend, self.charBlendWeight], dim=0
                    )
                    blend = F.softmax(all_blend_weights, dim=0)
                    if noise_scale != 1.0:
                        blend = blend.clone()
                        blend[2] = blend[2] * noise_scale
                        blend = blend / blend.sum().clamp_min(1e-6)

                    if not skipPixels and (_pixel is not None):
                        legacy_rgb_embed = self.embed(_pixel=_pixel)
                        rgbEmbed = (1 - gate) * legacy_rgb_embed + gate * (
                            legacy_rgb_embed + sensory_embed
                        )
                        blendedInput = (
                            blend[0] * tokenEmbed
                            + blend[1] * posEmbed
                            + blend[2] * rgbEmbed
                            + blend[3] * charEmbed
                        )
                        rgb_sequence_norm = self._trace_tensor_norm(rgbEmbed)
                        rgb_last_norm = self._trace_last_token_norm(rgbEmbed)
                    else:
                        rgbEmbed = None
                        blendedInput = (
                            blend[0] * tokenEmbed
                            + blend[1] * posEmbed
                            + blend[3] * charEmbed
                        )
                        rgb_sequence_norm = 0.0
                        rgb_last_norm = 0.0

                    attention1Output = self.attention(blendedInput)
                    if enableTangling:
                        tangle_embed = self.tangling.refine(
                            attention1Output, stage_name="embed"
                        )
                        post_embed = attention1Output + tangle_embed
                    else:
                        tangle_embed = None
                        post_embed = attention1Output

                    inn_core = self.interneuronNetwork.forward(post_embed)
                    attention2_add = self.attention2(inn_core)
                    inn_after_attention2 = inn_core + attention2_add

                    if enableTangling:
                        tangle_neuron = self.tangling.refine(
                            inn_after_attention2, stage_name="neuron"
                        )
                        inn_after_tangle = inn_after_attention2 + tangle_neuron
                    else:
                        tangle_neuron = None
                        inn_after_tangle = inn_after_attention2

                    scratch_add = self.scratchpad(inn_after_tangle)
                    inn_after_scratch = inn_after_tangle + scratch_add

                    if skipMemory:
                        memory_base = inn_after_scratch
                        memory_out = inn_after_scratch
                        memory_tangle = None
                        memory2_input = inn_after_scratch
                        memory2_base = inn_after_scratch
                        memory2_out = inn_after_scratch
                    else:
                        memory_base = self.memory.forward(inn_after_scratch)
                        memory_out = memory_base + inn_after_scratch
                        if enableTangling:
                            memory_tangle = self.tangling.refine(
                                memory_out, stage_name="memory"
                            )
                            memory_out = memory_out + memory_tangle
                        else:
                            memory_tangle = None
                        memory2_input = (inn_after_scratch * 0.5) + (memory_out * 0.5)
                        memory2_base = self.memory2.forward(memory2_input)
                        memory2_out = memory2_base + memory2_input

                    scaledActsTensor = memory2_out + self.logits.activationNorm(
                        memory2_out
                    )
                    rawLogitOutput = (
                        scaledActsTensor @ self.logits.l_weights
                    ) + self.logits.l_bias
                    logitsBeforePenalty = self.logits.forward(memory2_out)
                    finalLogits = self.applyRepetitionPenalty(
                        logitsBeforePenalty, input_seq
                    )
                    finalLogits = torch.nan_to_num(
                        finalLogits, nan=0.0, posinf=80.0, neginf=-80.0
                    )

                    logits_view = (
                        finalLogits[-1] if finalLogits.dim() > 1 else finalLogits
                    )
                    raw_temp = torch.exp(self.logTemp)
                    safe_temp = raw_temp.clamp(min=0.1, max=5.0)
                    safe_temp = (safe_temp * motion_scale).clamp(min=0.1, max=5.0)
                    scaled_logits = torch.nan_to_num(
                        logits_view / safe_temp, nan=0.0, posinf=1e3, neginf=-1e3
                    )
                    scaled_logits = scaled_logits.clamp(min=-80.0, max=80.0)
                    probs = F.softmax(scaled_logits, dim=-1)

                    top_count = max(1, min(int(top_k), int(probs.shape[-1])))
                    top_probs, top_idx = torch.topk(probs, top_count)
                    top_predictions = []
                    for prob, idx in zip(top_probs.tolist(), top_idx.tolist()):
                        idx = int(idx)
                        top_predictions.append(
                            {
                                "token_id": idx,
                                "prob": float(prob),
                                "logit": float(logits_view[idx].item()),
                                "scaled_logit": float(scaled_logits[idx].item()),
                            }
                        )

                    eos_id = None
                    eos_prob = None
                    eos_rank = None
                    eos_logit = None
                    if eos_replacement_token_str:
                        eos_id = self.librarian.tokenToIndex.get(
                            eos_replacement_token_str
                        )
                    if eos_id is not None and 0 <= int(eos_id) < int(probs.shape[-1]):
                        eos_id = int(eos_id)
                        eos_prob = float(probs[eos_id].item())
                        eos_logit = float(logits_view[eos_id].item())
                        eos_rank = int((probs > probs[eos_id]).sum().item()) + 1

                    active_blend = {
                        "token": float(blend[0].item()),
                        "pos": float(blend[1].item()),
                        "char": float(blend[3].item()) if blend.numel() > 3 else 0.0,
                        "pixel": float(blend[2].item()) if blend.numel() > 2 else 0.0,
                    }
                    if rgbEmbed is None:
                        active_total = (
                            active_blend["token"]
                            + active_blend["pos"]
                            + active_blend["char"]
                        )
                        if active_total > 0:
                            active_blend["token"] /= active_total
                            active_blend["pos"] /= active_total
                            active_blend["char"] /= active_total
                        active_blend["pixel"] = 0.0
                    else:
                        active_total = sum(active_blend.values())
                        if active_total > 0:
                            for key in active_blend:
                                active_blend[key] /= active_total

                    def _window_pairs(size_tensor, weight_tensor):
                        if not torch.is_tensor(size_tensor) or not torch.is_tensor(
                            weight_tensor
                        ):
                            return []
                        if size_tensor.numel() == 0 or weight_tensor.numel() == 0:
                            return []
                        count = min(
                            int(size_tensor.numel()), int(weight_tensor.numel())
                        )
                        pairs = []
                        for idx in range(count):
                            pairs.append(
                                {
                                    "size": float(size_tensor[idx].item()),
                                    "weight": float(weight_tensor[idx].item()),
                                }
                            )
                        return pairs

                    trace = {
                        "sequence_length": seq_len,
                        "input_ids": [int(idx) for idx in input_seq.tolist()],
                        "decoded_prompt": self.librarian.decodeIDs(
                            [int(idx) for idx in input_seq.tolist()]
                        ),
                        "temperature": float(safe_temp.item()),
                        "pixel_active": rgbEmbed is not None,
                        "blend": {
                            "token": float(blend[0].item()),
                            "pos": float(blend[1].item()),
                            "pixel": float(blend[2].item())
                            if blend.numel() > 2
                            else 0.0,
                            "char": float(blend[3].item())
                            if blend.numel() > 3
                            else 0.0,
                        },
                        "active_blend": active_blend,
                        "sensory": {
                            "gate": self._trace_scalar(gate),
                            "motion_scale": float(motion_scale),
                            "attention_scale": float(attention_scale),
                            "char_scale": float(char_scale),
                            "pos_scale": float(pos_scale),
                            "noise_scale": float(noise_scale),
                        },
                        "gates": {
                            "attention1": self._trace_scalar(
                                torch.sigmoid(self.attention.logit_gate)
                                * attention_scale
                            ),
                            "attention2": self._trace_scalar(
                                torch.sigmoid(self.attention2.logit_gate)
                                * attention_scale
                            ),
                            "memory1_short": self._trace_scalar(
                                getattr(self.memory, "short_used", 0.0)
                            ),
                            "memory1_long": self._trace_scalar(
                                getattr(self.memory, "long_used", 0.0)
                            ),
                            "memory1_act": self._trace_scalar(
                                getattr(self.memory, "act_used", 0.0)
                            ),
                            "memory1_mem": self._trace_scalar(
                                getattr(self.memory, "mem_used", 0.0)
                            ),
                            "memory2_short": self._trace_scalar(
                                getattr(self.memory2, "short_used", 0.0)
                            ),
                            "memory2_long": self._trace_scalar(
                                getattr(self.memory2, "long_used", 0.0)
                            ),
                            "memory2_act": self._trace_scalar(
                                getattr(self.memory2, "act_used", 0.0)
                            ),
                            "memory2_mem": self._trace_scalar(
                                getattr(self.memory2, "mem_used", 0.0)
                            ),
                            "scratch_write": float(
                                self.scratchpad.stats.get("SCRATCH_write_amount", 0.0)
                            ),
                            "scratch_erase": float(
                                self.scratchpad.stats.get("SCRATCH_erase_amount", 0.0)
                            ),
                            "scratch_read": float(
                                self.scratchpad.stats.get("SCRATCH_read_amount", 0.0)
                            ),
                            "inn_short_window_gate": self._trace_scalar(
                                getattr(self.interneuronNetwork, "short_gate_used", 0.0)
                            ),
                        },
                        "memory": {
                            "memory1_short_decay": self._trace_scalar(
                                getattr(self.memory, "shortDecay_used", 0.0)
                            ),
                            "memory1_long_decay": self._trace_scalar(
                                getattr(self.memory, "longDecay_used", 0.0)
                            ),
                            "memory2_short_decay": self._trace_scalar(
                                getattr(self.memory2, "shortDecay_used", 0.0)
                            ),
                            "memory2_long_decay": self._trace_scalar(
                                getattr(self.memory2, "longDecay_used", 0.0)
                            ),
                            "memory1_pending_short_norm": self._trace_tensor_norm(
                                getattr(self.memory, "newShort", None)
                            ),
                            "memory1_pending_long_norm": self._trace_tensor_norm(
                                getattr(self.memory, "newLong", None)
                            ),
                            "memory2_pending_short_norm": self._trace_tensor_norm(
                                getattr(self.memory2, "newShort", None)
                            ),
                            "memory2_pending_long_norm": self._trace_tensor_norm(
                                getattr(self.memory2, "newLong", None)
                            ),
                        },
                        "scratchpad": {
                            "write_strength": float(
                                self.scratchpad.stats.get("SCRATCH_write_strength", 0.0)
                            ),
                            "erase_strength": float(
                                self.scratchpad.stats.get("SCRATCH_erase_strength", 0.0)
                            ),
                            "buffer_norm": float(
                                self.scratchpad.stats.get("SCRATCH_buffer_norm", 0.0)
                            ),
                            "retrieved_norm": float(
                                self.scratchpad.stats.get("SCRATCH_retrieved_norm", 0.0)
                            ),
                            "integrated_norm": float(
                                self.scratchpad.stats.get(
                                    "SCRATCH_integrated_norm", 0.0
                                )
                            ),
                            "slot_usage_max": float(
                                self.scratchpad.stats.get("SCRATCH_slot_usage_max", 0.0)
                            ),
                            "slot_usage_mean": float(
                                self.scratchpad.stats.get(
                                    "SCRATCH_slot_usage_mean", 0.0
                                )
                            ),
                        },
                        "stages": {
                            "token_embed": {
                                "sequence_norm": self._trace_tensor_norm(tokenEmbed),
                                "last_token_norm": self._trace_last_token_norm(
                                    tokenEmbed
                                ),
                            },
                            "pos_embed": {
                                "sequence_norm": self._trace_tensor_norm(posEmbed),
                                "last_token_norm": self._trace_last_token_norm(
                                    posEmbed
                                ),
                            },
                            "char_embed": {
                                "sequence_norm": self._trace_tensor_norm(charEmbed),
                                "last_token_norm": self._trace_last_token_norm(
                                    charEmbed
                                ),
                            },
                            "pixel_embed": {
                                "sequence_norm": rgb_sequence_norm,
                                "last_token_norm": rgb_last_norm,
                            },
                            "blended_input": {
                                "sequence_norm": self._trace_tensor_norm(blendedInput),
                                "last_token_norm": self._trace_last_token_norm(
                                    blendedInput
                                ),
                            },
                            "attention1": {
                                "sequence_norm": self._trace_tensor_norm(
                                    attention1Output
                                ),
                                "last_token_norm": self._trace_last_token_norm(
                                    attention1Output
                                ),
                            },
                            "tangle_embed": {
                                "sequence_norm": self._trace_tensor_norm(tangle_embed),
                                "last_token_norm": self._trace_last_token_norm(
                                    tangle_embed
                                ),
                            },
                            "inn_core": {
                                "sequence_norm": self._trace_tensor_norm(inn_core),
                                "last_token_norm": self._trace_last_token_norm(
                                    inn_core
                                ),
                            },
                            "attention2_add": {
                                "sequence_norm": self._trace_tensor_norm(
                                    attention2_add
                                ),
                                "last_token_norm": self._trace_last_token_norm(
                                    attention2_add
                                ),
                            },
                            "inn_after_attention2": {
                                "sequence_norm": self._trace_tensor_norm(
                                    inn_after_attention2
                                ),
                                "last_token_norm": self._trace_last_token_norm(
                                    inn_after_attention2
                                ),
                            },
                            "tangle_neuron": {
                                "sequence_norm": self._trace_tensor_norm(tangle_neuron),
                                "last_token_norm": self._trace_last_token_norm(
                                    tangle_neuron
                                ),
                            },
                            "scratchpad_add": {
                                "sequence_norm": self._trace_tensor_norm(scratch_add),
                                "last_token_norm": self._trace_last_token_norm(
                                    scratch_add
                                ),
                            },
                            "inn_after_scratch": {
                                "sequence_norm": self._trace_tensor_norm(
                                    inn_after_scratch
                                ),
                                "last_token_norm": self._trace_last_token_norm(
                                    inn_after_scratch
                                ),
                            },
                            "memory1_base": {
                                "sequence_norm": self._trace_tensor_norm(memory_base),
                                "last_token_norm": self._trace_last_token_norm(
                                    memory_base
                                ),
                            },
                            "memory1_out": {
                                "sequence_norm": self._trace_tensor_norm(memory_out),
                                "last_token_norm": self._trace_last_token_norm(
                                    memory_out
                                ),
                            },
                            "tangle_memory": {
                                "sequence_norm": self._trace_tensor_norm(memory_tangle),
                                "last_token_norm": self._trace_last_token_norm(
                                    memory_tangle
                                ),
                            },
                            "memory2_input": {
                                "sequence_norm": self._trace_tensor_norm(memory2_input),
                                "last_token_norm": self._trace_last_token_norm(
                                    memory2_input
                                ),
                            },
                            "memory2_base": {
                                "sequence_norm": self._trace_tensor_norm(memory2_base),
                                "last_token_norm": self._trace_last_token_norm(
                                    memory2_base
                                ),
                            },
                            "memory2_out": {
                                "sequence_norm": self._trace_tensor_norm(memory2_out),
                                "last_token_norm": self._trace_last_token_norm(
                                    memory2_out
                                ),
                            },
                            "logits_pre_penalty": {
                                "sequence_norm": self._trace_tensor_norm(
                                    logitsBeforePenalty
                                ),
                                "last_token_norm": self._trace_last_token_norm(
                                    logitsBeforePenalty
                                ),
                            },
                            "final_logits": {
                                "sequence_norm": self._trace_tensor_norm(finalLogits),
                                "last_token_norm": self._trace_last_token_norm(
                                    finalLogits
                                ),
                            },
                        },
                        "per_token": {
                            "token_embed": self._trace_per_token_norms(tokenEmbed),
                            "pos_embed": self._trace_per_token_norms(posEmbed),
                            "char_embed": self._trace_per_token_norms(charEmbed),
                            "blended_input": self._trace_per_token_norms(blendedInput),
                            "attention1": self._trace_per_token_norms(attention1Output),
                        },
                        "inn": {
                            "window_entropy": self._trace_scalar(
                                getattr(
                                    self.interneuronNetwork, "windowSizeEntropy", 0.0
                                )
                            ),
                            "window_spread": self._trace_scalar(
                                getattr(
                                    self.interneuronNetwork, "windowWeightSpread", 0.0
                                )
                            ),
                            "range_penalty": self._trace_scalar(
                                getattr(self.interneuronNetwork, "rangePenalty", 0.0)
                            ),
                            "mean_penalty": self._trace_scalar(
                                getattr(self.interneuronNetwork, "meanPenalty", 0.0)
                            ),
                        },
                        "windows": {
                            "long": _window_pairs(
                                getattr(
                                    self.interneuronNetwork,
                                    "floatWindowSizes_used",
                                    None,
                                ),
                                getattr(
                                    self.interneuronNetwork, "cerebellumSoft", None
                                ),
                            ),
                            "short": _window_pairs(
                                getattr(
                                    self.interneuronNetwork,
                                    "floatWindowSizes_short_used",
                                    None,
                                ),
                                getattr(
                                    self.interneuronNetwork,
                                    "cerebellumSoft_short",
                                    None,
                                ),
                            ),
                        },
                        "top_predictions": top_predictions,
                        "eos": {
                            "token_id": eos_id,
                            "rank": eos_rank,
                            "prob": eos_prob,
                            "logit": eos_logit,
                        },
                    }
                    if include_distribution:
                        trace["distribution"] = {
                            "probs": probs.detach().cpu(),
                            "scaled_logits": scaled_logits.detach().cpu(),
                        }
                    if include_vectors:

                        def _last_row_cpu(tensor):
                            if not torch.is_tensor(tensor) or tensor.numel() == 0:
                                return None
                            detached = tensor.detach()
                            if detached.dim() == 1:
                                row = detached
                            elif detached.dim() == 2:
                                row = detached[-1]
                            elif detached.dim() == 3:
                                row = detached[0, -1]
                            else:
                                row = detached.reshape(-1)
                            return row.cpu()

                        trace["vectors"] = {
                            "memory2_last": _last_row_cpu(memory2_out),
                            "scaled_acts_last": _last_row_cpu(scaledActsTensor),
                            "raw_logits_last": _last_row_cpu(rawLogitOutput),
                        }
                    return trace
                finally:
                    self.stats = dict(model_snapshot.get("stats", {}) or {})
                    self.forwardStats = dict(
                        model_snapshot.get("forwardStats", {}) or {}
                    )
                    self.recentGeneratedTokens = list(
                        model_snapshot.get("recentGeneratedTokens", [])
                    )
                    self._trace_restore_histories(self, model_snapshot.get("histories"))
                    for name, value in model_snapshot.get("attrs", {}).items():
                        setattr(self, name, self._trace_clone_value(value))
                    self._trace_restore_module_state(self.attention, attention_snapshot)
                    self._trace_restore_module_state(
                        self.attention2, attention2_snapshot
                    )
                    self._trace_restore_module_state(
                        self.interneuronNetwork, inn_snapshot
                    )
                    self._trace_restore_module_state(
                        self.interneuronNetwork.neurons, neuron_snapshot
                    )
                    self._trace_restore_module_state(self.memory, memory_snapshot)
                    self._trace_restore_module_state(self.memory2, memory2_snapshot)
                    self._trace_restore_module_state(
                        self.scratchpad, scratchpad_snapshot
                    )
                    self._trace_restore_module_state(self.tangling, tangling_snapshot)
                    self._trace_restore_module_state(self.logits, logits_snapshot)
                    if was_training:
                        self.train()
                    else:
                        self.eval()

    @whocalled
    def applyRepetitionPenalty(self, _logits, _contextTokens=None):
        with self.counsellor.infodump("applyRepetitionPenalty") as ʕっʘ‿ʘʔっ:
            if not self.recentGeneratedTokens:
                if _contextTokens is None:
                    if debugPrints:
                        ʕっʘ‿ʘʔっ("no recent tokens or context, returning _logits")
                    return _logits
                if debugPrints:
                    ʕっʘ‿ʘʔっ("using context tokens for repetition penalty")
                recentTokens = _contextTokens[-int(self.numTokensPerStep) :].detach()
                recentTokens = recentTokens.to(self.device)
                if recentTokens.dtype != torch.long:
                    recentTokens = recentTokens.long()
                recentTokens = recentTokens.reshape(-1)
            else:
                recentTokens = torch.tensor(
                    self.recentGeneratedTokens, device=self.device, dtype=torch.long
                )
                recentTokens = recentTokens.reshape(-1)

            if debugPrints:
                ʕっʘ‿ʘʔっ("repWindow = torch.exp(self.logRepetitionWindow)")
            repWindow = torch.exp(self.logRepetitionWindow)
            repWindow = repWindow / (1 + repWindow / self.numTokensPerStep)
            if debugPrints:
                ʕっʘ‿ʘʔっ("penalty = self.repetitionPenalty")
            penalty = self.repetitionPenalty
            if penalty < 0.0:
                new_value = repetitionPenaltyGOAL
                self.repetitionPenalty.data.copy_(new_value)
                penalty = self.repetitionPenalty

            if isinstance(recentTokens, list):
                if debugPrints:
                    ʕっʘ‿ʘʔっ("recentTokens list -> tensor")
                recentTokens = torch.tensor(
                    recentTokens, device=self.device, dtype=torch.long
                )
                recentTokens = recentTokens.reshape(-1)
            if debugPrints:
                ʕっʘ‿ʘʔっ("vocabSize = _logits.shape[1]")
            vocabSize = _logits.shape[1]

            if debugPrints:
                ʕっʘ‿ʘʔっ("positions = torch.arange(len(recentTokens)).float()")
            positions = torch.arange(len(recentTokens), device=self.device).float()
            if debugPrints:
                ʕっʘ‿ʘʔっ("windowCenter")
            windowCenter = len(recentTokens) - 0.5  # so token 0 gets proper suppression
            if debugPrints:
                ʕっʘ‿ʘʔっ(
                    "softMask = torch.sigmoid((positions - (windowCenter - repWindow)) * 0.5)"
                )
            # softMask = torch.sigmoid((positions - (windowCenter - repWindow)) * 0.5)
            distance_from_window_start = positions - (len(recentTokens) - repWindow)
            relative_position_in_window = distance_from_window_start / repWindow
            softMask = torch.clamp(relative_position_in_window, 0.0, 1.0)

            if debugPrints:
                ʕっʘ‿ʘʔっ("computing weighted frequencies")
            softMask = softMask.to(dtype=_logits.dtype)
            weightedFreqs = torch.zeros(
                vocabSize, device=_logits.device, dtype=_logits.dtype
            )
            if recentTokens.numel() > 0:
                weightedFreqs.index_add_(0, recentTokens, softMask)
            weightedFreqs = weightedFreqs.view(1, -1)

            if debugPrints:
                ʕっʘ‿ʘʔっ("setting penalty to 0 for target token!")
            if self.targetTokenFromTutor is not None:
                weightedFreqs[0, self.targetTokenFromTutor] = 0.0

        return _logits - (weightedFreqs * penalty)

    @whocalled
    def saveModel(
        self,
        _trainingStepCounter,
        _totalAvgLoss,
        _first,
        filePath=modelFilePath,
        _newStartIndex=trainingStartIndex,
    ):
        with self.counsellor.infodump("saveModel") as ʕっʘ‿ʘʔっ:
            # Refuse to save if any parameter OR memory buffer contains NaN/Inf.
            # A corrupted checkpoint is worse than no save — it would overwrite the last clean one.
            nan_params = [
                (n, tuple(p.shape))
                for n, p in self.named_parameters()
                if torch.is_floating_point(p) and not torch.isfinite(p).all()
            ]
            if nan_params:
                print(
                    f"⚠️ [SAVE BLOCKED] {len(nan_params)} NaN/Inf parameter(s) detected — refusing to overwrite checkpoint!"
                )
                print(f"   First offenders: {[n for n, _ in nan_params[:5]]}")
                return
            # Also check memory buffers (these are not nn.Parameter so named_parameters() misses them)
            mem_bufs = {
                "memory.shortTermMemory": self.memory.shortTermMemory,
                "memory.longTermMemory": self.memory.longTermMemory,
                "memory2.shortTermMemory": self.memory2.shortTermMemory,
                "memory2.longTermMemory": self.memory2.longTermMemory,
            }
            nan_bufs = [k for k, v in mem_bufs.items() if not torch.isfinite(v).all()]
            if nan_bufs:
                print(
                    f"⚠️ [SAVE BLOCKED] NaN/Inf in memory buffers {nan_bufs} — refusing to overwrite checkpoint!"
                )
                return
            with open(stepCheckpointFilePath, "w") as f:
                if debugPrints or True:
                    print(
                        f"HELLO I AM SAVEMODEL STEPCOUNTER IS {_trainingStepCounter} AND START INDEX IS {_newStartIndex} I SHOULD WRITE {str(_trainingStepCounter + _newStartIndex)} to {stepCheckpointFilePath}"
                    )
                f.write(
                    str(_trainingStepCounter + _newStartIndex)
                )  # THIS ISNT REAL, FIX LATER, MAYBE MOVE SAVE AND LOAD TO WAKEUP?
            with open(lossCheckpointAppendFilePath, "a") as f:
                if debugPrints or True:
                    print(
                        f"hi :) i am saveModel... avgLoss is: {_totalAvgLoss}, so... i'm writing {str(_totalAvgLoss)} to {lossCheckpointAppendFilePath}!"
                    )
                f.write(str(_totalAvgLoss))
            with open(lossCheckpointFilePath, "w") as f:
                if debugPrints or True:
                    print(
                        f"HELLO I AM SAVEMODEL AVGLOSS IS {_totalAvgLoss} I SHOULD WRITE {str(_totalAvgLoss)} to {lossCheckpointFilePath}"
                    )
                f.write(str(_totalAvgLoss))
            tmpPath = filePath + ".tmp"
            torch.save(self.state_dict(), tmpPath)
            print(f"model temp file created at {tmpPath}...")
            # save optimizer to a separate file (if present)
            if hasattr(self, "optimizer") and self.optimizer is not None:
                optimPath = filePath + ".optim"
                tmpOptimPath = optimPath + ".tmp"
                torch.save(self.optimizer.state_dict(), tmpOptimPath)
                print(f"optimizer saved to {optimPath}")
                os.replace(tmpOptimPath, optimPath)
            os.replace(tmpPath, filePath)
            print(f"model successfully saved to {filePath}!")
            # (existing model and optimizer saving)
            memory_buffers_state = {
                "memory1_short": self.memory.shortTermMemory.detach().cpu(),
                "memory1_long": self.memory.longTermMemory.detach().cpu(),
                "memory2_short": self.memory2.shortTermMemory.detach().cpu(),
                "memory2_long": self.memory2.longTermMemory.detach().cpu(),
            }
            buffers_path = filePath + ".membuff"
            tmp_buffers_path = buffers_path + ".tmp"
            torch.save(memory_buffers_state, tmp_buffers_path)
            print(f"Memory buffers temp file created at {tmp_buffers_path}...")
            os.replace(tmp_buffers_path, buffers_path)
            print(f"Memory buffers successfully saved to {buffers_path}!")

    @whocalled
    def loadModel(self, filePath=modelFilePath, *, async_optimizer: bool = False):
        """Load model + optimizer + memory buffers from a checkpoint.

        Set ``async_optimizer=True`` to defer optimizer state loading to a
        background daemon thread. The model is ready as soon as this method
        returns; the optimizer becomes ready some time later. Any code that
        needs the optimizer (i.e. before calling ``optimizer.step()`` or
        anything that mutates optimizer state) must call
        ``self.wait_for_optimizer_ready()`` first. This is the safe path for
        bot modes (twitch / discord / unified) where startup latency matters
        more than first-training-step latency.
        """
        with self.counsellor.infodump("loadModel") as ʕっʘ‿ʘʔっ:
            try:
                if debugPrints:
                    ʕっʘ‿ʘʔっ("update logarithmic parameters")
                repWindow = torch.exp(self.logRepetitionWindow)
                self.repetitionWindow = repWindow / (
                    1 + repWindow / self.numTokensPerStep
                )  # asymptotes near windowMAX
                self.temperature = torch.exp(
                    self.logTemp
                )  # TORCH.exp keeps gradient path!
                self.interneuronNetwork.temperature = self.temperature
                print(f"loading model from path: {filePath}")
                state_dict = torch.load(filePath, map_location=self.device)
                state_dict = self._upgrade_sensory_state_dict(state_dict)
                self.load_state_dict(state_dict, strict=saveStrict)
                # try loading optimizer separately
                if hasattr(self, "optimizer"):
                    optimPath = filePath + ".optim"
                    if os.path.exists(optimPath):
                        if async_optimizer:
                            self._start_async_optimizer_load(optimPath)
                        else:
                            self._load_optimizer_state(optimPath)
                print(f"model loaded from {filePath}!")
                self.to(self.device)
                print(f"device set to {self.device}!")
                # self.resetMemory(context="inference", _memoryLength = self.memoryLength)
                # (existing model and optimizer loading)
                buffers_path = filePath + ".membuff"
                if os.path.exists(buffers_path):
                    try:
                        memory_buffers_state = torch.load(
                            buffers_path, map_location=self.device
                        )  # Load to current device
                        # Guard: if any buffer is NaN/Inf, refuse to load and zero everything.
                        nan_bufs = [
                            k
                            for k, v in memory_buffers_state.items()
                            if isinstance(v, torch.Tensor)
                            and not torch.isfinite(v).all()
                        ]
                        if nan_bufs:
                            print(
                                f"⚠️ [MEMBUFF] NaN/Inf detected in {nan_bufs} — refusing to load corrupted memory buffers, initializing to zeros."
                            )
                            self.memory.shortTermMemory.zero_()
                            self.memory.longTermMemory.zero_()
                            self.memory2.shortTermMemory.zero_()
                            self.memory2.longTermMemory.zero_()
                        else:
                            self.memory.shortTermMemory.data.copy_(
                                memory_buffers_state["memory1_short"]
                            )
                            self.memory.longTermMemory.data.copy_(
                                memory_buffers_state["memory1_long"]
                            )
                            self.memory2.shortTermMemory.data.copy_(
                                memory_buffers_state["memory2_short"]
                            )
                            self.memory2.longTermMemory.data.copy_(
                                memory_buffers_state["memory2_long"]
                            )
                            print(f"Memory buffers restored from {buffers_path}")
                    except Exception as e:
                        print(
                            f"Failed to load memory buffers: {e}. Initializing to zeros."
                        )
                        # Ensure they are zeroed if loading fails
                        self.memory.shortTermMemory.zero_()
                        self.memory.longTermMemory.zero_()
                        self.memory2.shortTermMemory.zero_()
                        self.memory2.longTermMemory.zero_()
                else:
                    print(
                        f"No memory buffer file found at {buffers_path}. Initializing to zeros."
                    )
                    # Ensure they are zeroed if file not found
                    self.memory.shortTermMemory.zero_()
                    self.memory.longTermMemory.zero_()
                    self.memory2.shortTermMemory.zero_()
                    self.memory2.longTermMemory.zero_()
                self.memory.to(self.device)
                self.memory2.to(self.device)
                print(f"memory device set to {self.device}!")

            except FileNotFoundError:
                print("no saved model found")

    def _load_optimizer_state(self, optimPath):
        """Synchronous optimizer load. Re-uploads tensors to self.device because
        torch.load(..., map_location=device) drops them on the right device but
        the load_state_dict copy can land on CPU when the optimizer was created
        before any tensors moved to the GPU."""
        try:
            loaded_optim_state = torch.load(
                optimPath,
                map_location=self.device,
            )

            current_optim_state = self.optimizer.state_dict()
            loaded_groups = loaded_optim_state.get(
                "param_groups", []
            )
            current_groups = current_optim_state.get(
                "param_groups", []
            )

            shape_changed = (
                len(loaded_groups) == len(current_groups)
                and any(
                    len(old_group.get("params", []))
                    != len(new_group.get("params", []))
                    for old_group, new_group
                    in zip(loaded_groups, current_groups)
                )
            )

            if shape_changed:
                if any(
                    len(old_group.get("params", []))
                    > len(new_group.get("params", []))
                    for old_group, new_group
                    in zip(loaded_groups, current_groups)
                ):
                    raise RuntimeError(
                        "Saved optimizer has more parameters "
                        "than current Baby; refusing unsafe remap."
                    )

                names = [
                    name for name, _ in self.named_parameters()
                ]

                expected_tail = [
                    "ar_throat.weight",
                    "ar_expand.weight",
                    "ar_neuron_gate.0",
                ]

                added = sum(
                    len(new_group.get("params", []))
                    - len(old_group.get("params", []))
                    for old_group, new_group
                    in zip(loaded_groups, current_groups)
                )

                if (
                    added != 3
                    or names[-3:] != expected_tail
                ):
                    raise RuntimeError(
                        "Optimizer layout changed for something "
                        "other than known AR512 append; refusing. "
                        f"added={added}, tail={names[-6:]}"
                    )

                old_state = loaded_optim_state.get(
                    "state", {}
                )
                remapped = {}

                for old_group, new_group in zip(
                    loaded_groups,
                    current_groups,
                ):
                    old_ids = list(
                        old_group.get("params", [])
                    )
                    new_ids = list(
                        new_group.get("params", [])
                    )

                    for old_id, new_id in zip(
                        old_ids,
                        new_ids[:len(old_ids)],
                    ):
                        if old_id in old_state:
                            remapped[new_id] = old_state[old_id]

                    old_group["params"] = new_ids

                loaded_optim_state["state"] = remapped

                print(
                    "[AR TANGLE] optimizer checkpoint safely "
                    "extended by 3 new parameter tensors; "
                    "all existing optimizer state preserved."
                )

            self.optimizer.load_state_dict(
                loaded_optim_state
            )
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            print(f"optimizer restored from {optimPath}")
        except Exception as e:
            print(f"failed to load optimizer: {e}")
            raise RuntimeError(
                f"Refusing to continue with newly initialized optimizer state: {e}"
            ) from e

    def _start_async_optimizer_load(self, optimPath):
        """Kick off optimizer loading in a daemon thread. The 5GB optim file
        I/O overlaps with Discord/Twitch/Web bot startup so the user-visible
        latency drops to whatever the slower of the two is. Code that needs
        the optimizer (e.g. tutor.trainModel) must call
        wait_for_optimizer_ready() first; we set that as a barrier in the
        training entry points."""
        import threading

        if not hasattr(self, "_optimizer_ready_event"):
            self._optimizer_ready_event = threading.Event()
        else:
            self._optimizer_ready_event.clear()
        self._optimizer_load_error = None

        def _worker():
            try:
                self._load_optimizer_state(optimPath)
            except Exception as exc:
                self._optimizer_load_error = exc
                print(f"[ASYNC OPTIM] load failed: {exc}")
            finally:
                self._optimizer_ready_event.set()

        thread = threading.Thread(
            target=_worker, name="async-optim-load", daemon=True
        )
        thread.start()
        print(
            f"[ASYNC OPTIM] optimizer load started in background thread "
            f"(file: {optimPath})"
        )

    def wait_for_optimizer_ready(self, timeout: float | None = None) -> bool:
        """Block until async optimizer load finishes. Returns True if ready,
        False if a timeout was hit. Cheap when load was synchronous (the event
        attribute won't exist) — returns True immediately."""
        event = getattr(self, "_optimizer_ready_event", None)
        if event is None:
            return True
        if event.is_set():
            ready = True
        else:
            import time

            t0 = time.perf_counter()
            ready = event.wait(timeout=timeout)
            if ready:
                elapsed = time.perf_counter() - t0
                if elapsed > 0.05:
                    print(f"[ASYNC OPTIM] waited {elapsed:.2f}s for optimizer ready")
        if ready and self._optimizer_load_error is not None:
            raise RuntimeError(
                "Optimizer checkpoint restoration failed; training is blocked "
                "to protect learned state."
            ) from self._optimizer_load_error
        return ready

    def _upgrade_sensory_state_dict(self, state_dict):
        """Expand sensory tensors when older checkpoints have smaller dims."""

        target_dim = int(self.sensory_dim)
        target_pred_dim = int(self.sensory_pred_dim)

        def _expand_1d(name, target_len, fill_value):
            tensor = state_dict.get(name)
            if tensor is None or tensor.dim() != 1 or tensor.numel() == target_len:
                return
            new_tensor = tensor.new_full((target_len,), fill_value)
            copy_len = min(tensor.numel(), target_len)
            new_tensor[:copy_len] = tensor[:copy_len]
            state_dict[name] = new_tensor

        def _expand_2d_cols(name, target_cols, fill_value):
            tensor = state_dict.get(name)
            if tensor is None or tensor.dim() != 2 or tensor.shape[1] == target_cols:
                return
            new_tensor = tensor.new_full((tensor.shape[0], target_cols), fill_value)
            copy_cols = min(tensor.shape[1], target_cols)
            new_tensor[:, :copy_cols] = tensor[:, :copy_cols]
            state_dict[name] = new_tensor

        def _expand_2d_rows(name, target_rows, fill_value):
            tensor = state_dict.get(name)
            if tensor is None or tensor.dim() != 2 or tensor.shape[0] == target_rows:
                return
            new_tensor = tensor.new_full((target_rows, tensor.shape[1]), fill_value)
            copy_rows = min(tensor.shape[0], target_rows)
            new_tensor[:copy_rows] = tensor[:copy_rows]
            state_dict[name] = new_tensor

        _expand_1d("sensory_scale", target_dim, 1e-5)
        _expand_1d("sensory_bias", target_dim, 0.0)
        _expand_2d_cols("sensoryEmbed.0.weight", target_dim, 1e-5)
        _expand_2d_rows("sensoryPupil.weight", target_pred_dim, 1e-5)
        _expand_1d("sensoryPupil.bias", target_pred_dim, 0.0)

        return state_dict

    @whocalled
    def resetMemory(self, context="inference"):
        with self.counsellor.infodump("resetMemory") as ʕっʘ‿ʘʔっ:
            """Reset memory depending on the context: inference always resets, training resets every n turns"""
            self.memoryLength = torch.sigmoid(
                (self.totalTurns - torch.exp(self.logMemoryLength)) * 0.1
            )
            self.memory2Length = torch.sigmoid(
                (self.totalTurns - torch.exp(self.logMemory2Length)) * 0.1
            )
            # print(f"resetting memory... (learned mem length: {self.memoryLength})")
            # self.memory.resetMemory(_memoryLength = self.memoryLength)
            # self.memory2.resetMemory(_memoryLength = self.memoryLength)
            if context == "inference":
                if debugPrints:
                    ʕっʘ‿ʘʔっ("context = inference")
                self.memory.resetMemory(self.memoryLength)
                self.memory2.resetMemory(self.memory2Length)
                print("resetting memory for new conversation...")
            elif context == "training":
                if debugPrints:
                    ʕっʘ‿ʘʔっ("context = training")
                if hasattr(self, "stepsSinceMemoryReset"):
                    self.stepsSinceMemoryReset += 1
                else:
                    self.stepsSinceMemoryReset = 1
                if hasattr(self, "stepsSinceMemory2Reset"):
                    self.stepsSinceMemory2Reset += 1
                else:
                    self.stepsSinceMemory2Reset = 1
                if self.stepsSinceMemoryReset > 3:
                    debug_print(
                        f"resetting memory1 after {self.stepsSinceMemoryReset} steps... (learned mem length: {torch.exp(self.logMemoryLength)} ({self.memoryLength}))"
                    )
                    self.memory.resetMemory(_memoryLength=self.memoryLength)
                    self.stepsSinceMemoryReset = 0
                if self.stepsSinceMemory2Reset > 3:
                    debug_print(
                        f"resetting memory2 after {self.stepsSinceMemory2Reset} steps... (learned mem length: {torch.exp(self.logMemory2Length)} ({self.memory2Length}))"
                    )
                    self.memory2.resetMemory(_memoryLength=self.memory2Length)
                    self.stepsSinceMemory2Reset = 0



    @whocalled
    def print_rgb_block(self, rgb_tensor, label="RGB"):
        # rgb_tensor = rgb_tensor.detach().cpu().clamp(0, 1).numpy()
        rgb_tensor = rgb_tensor.detach().cpu().numpy()

        # If it's a 1D array, convert it to shape (1, 3)
        if rgb_tensor.ndim == 1:
            print("DIM1 reshape")
            rgb_tensor = rgb_tensor.reshape(1, 3)

        for i, rgb in enumerate(rgb_tensor):
            r, g, b = (rgb * 255).astype(int)
            print(f"{label}[{i}]: \x1b[48;2;{r};{g};{b}m     \x1b[0m  ({r}, {g}, {b})")

    def getRollingTokenTotalsDict(self):
        counts = self.rollingTokenTotals_tensor.detach().cpu()
        non_zero = torch.nonzero(counts).squeeze()
        if non_zero.numel() == 0:
            return {}
        if non_zero.dim() == 0:
            non_zero = non_zero.unsqueeze(0)
        return {
            self.librarian.indexToToken[int(i)]: float(counts[int(i)]) for i in non_zero
        }



    """def log_all_learnable_params(self, prefix="PARAM_"):
        Logs all learnable scalar parameters and basic stats for tensors in self.stats dict.
        Also ensures mostImportantStats includes new param keys matching include_patterns.
        for name, param in self.named_parameters():
            if param.requires_grad:
                if param.numel() == 1:
                    self.stats[f"{prefix}{name}"] = param.item()
                else:
                    self.stats[f"{prefix}{name}_mean"] = param.data.mean().item()
                    self.stats[f"{prefix}{name}_norm"] = param.data.norm().item()

        new_keys = [k for k in self.stats if k.startswith(prefix)]
        for key in new_keys:
            if re.search(pat, key, re.IGNORECASE):
                if key not in mostImportantStats:
                    mostImportantStats.append(key)"""

    @whocalled
    def getBabyStats(self):
        # Creative modules removed; return core stats only
        return dict(self.stats)


class PIXEL(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = 3,
        *,
        output_mode: str = "sigmoid",
        use_layernorm: bool = True,
        res_scale_init: float = 0.5,
        _device=modelDevice,
    ):
        super().__init__()
        self.device = _device
        self.linear1 = nn.Linear(in_features, hidden_features, device=self.device)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_features, out_features, device=self.device)

        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln = nn.LayerNorm(hidden_features, device=self.device)

        self.alpha = nn.Parameter(torch.tensor(res_scale_init, device=self.device))
        self.beta = nn.Parameter(torch.tensor(res_scale_init, device=self.device))

        self.register_buffer(
            "inv_sqrt2", torch.tensor(1 / math.sqrt(2), device=self.device)
        )

        assert output_mode in {"sigmoid", "clamp", "raw"}
        self.output_mode = output_mode

    def forward(self, x):
        x_res = x
        x = self.linear1(x)
        x = x + self.alpha * x_res

        x_res_gelu = x
        x = self.gelu(x)
        x = x + self.beta * x_res_gelu
        x = x * self.inv_sqrt2

        if self.use_layernorm:
            x = self.ln(x)

        logits = self.linear2(x)

        if self.output_mode == "sigmoid":
            return torch.sigmoid(logits)
        elif self.output_mode == "clamp":
            return torch.clamp(logits, 0.0, 1.0)
        else:
            return logits


if __name__ == "__main__":
    exit(0)
