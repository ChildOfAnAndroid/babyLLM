# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ ---
# INTERNEURON NETWORK & NEURONS
# brain/LAYERS/interneuronNetwork.py
# v1.3

import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gelu

from config import *
from school.staffroom.counsellor import COUNSELLOR
from utils.helpers import clamp_param


class NEURON(nn.Module):
    def __init__(self, _counsellor, _numTokensPerStep, _device=modelDevice):
        super().__init__()
        self.device = _device
        self.n_counsellor = _counsellor
        self.numTokensPerStep = _numTokensPerStep

        # SELF ALLOWED - nn.parameter!
        self.inputNorm = nn.LayerNorm(
            embedDimension, elementwise_affine=True, device=self.device
        )
        self.n_weights = nn.Parameter(
            torch.randn(numNeurons, embedDimension, device=self.device) * 0.01
        )
        self.n_biases = nn.Parameter(torch.zeros(numNeurons, device=self.device))
        self.activation_gain = nn.Parameter(torch.ones(1, device=self.device))
        self.n_counsellor = COUNSELLOR(
            "NEURON", _debug=debugPrints, _durations=durationLogging
        )

        self.stats = {}
        self._history_attrs = [
            "rawInputHistory",
            "rawInputNormHistory",
            "rawInputHistory_tokens",
            "rawInputHistory_neurons",
            "normedInputHistory",
            "normedInputNormHistory",
            "normedInputHistory_tokens",
            "normedInputHistory_neurons",
            "rawOutputHistory",
            "rawOutputNormHistory",
            "rawOutputHistory_tokens",
            "rawOutputHistory_neurons",
            "activatedOutputHistory",
            "activatedOutputNormHistory",
            "activatedOutputHistory_tokens",
            "activatedOutputHistory_neurons",
            "activatedOutputHistory_std_token",
            "activatedOutputHistory_std_neuron",
            "activatedOutputHistory_saturation",
            "activatedOutputHistory_min",
            "activatedOutputHistory_max",
        ]
        self._init_history_buffers()

        # self.normedOutputHistory = []
        # self.normedOutputHistory_tokens = []
        # self.normedOutputHistory_neurons = []

        # MUST NOT BE ON SELF - global parameters that may be used by backward pass
        # numNeuron, embedDimension, activationFunction, e

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
        tensor = torch.as_tensor(history, dtype=torch.float32, device=self.device)
        return tensor.mean().item() + offset

    @whocalled
    def forward(self, _inputEmbeds):  # embed: (batch_size, embed_size)
        with self.n_counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            inputEmbeds = _inputEmbeds
            with torch.no_grad():
                weightNorm = self.n_weights.norm(dim=1, keepdim=True)
                clippedWeights = self.n_weights / weightNorm.clamp(min=1.0, max=100.0)
                self.n_weights.data = clippedWeights

            if skipNeuron:
                if debugPrints:
                    ʕっʘ‿ʘʔっ("skipping forward")
                activations = _inputEmbeds.mean(
                    dim=-1
                )  # Mean across embed_dim, shape (sequence_length,)
                return activations[-1].unsqueeze(
                    0
                )  # Take LAST activation, unsqueeze to (1, ) for batch dim

            if debugPrints:
                ʕっʘ‿ʘʔっ("inputNorm")
            normedInput = self.inputNorm(_inputEmbeds)

            if debugPrints:
                ʕっʘ‿ʘʔっ(
                    "computeBatchedDotProduct+bias"
                )  # Compute batched dot product + bias: (batch_size, num_neurons)
            # rawOutput = torch.matmul(normedInput, self.n_weights.T) + self.n_biases  # shape: (seq_len, numNeurons)
            # Use a tensor for scaling to avoid device placement issues on MPS
            scale_value = math.sqrt(embedDimension)
            scale = torch.tensor(scale_value, device=normedInput.device)
            rawOutput = (
                torch.matmul(normedInput, self.n_weights.T) + self.n_biases
            ) / scale

            if debugPrints:
                ʕっʘ‿ʘʔっ(
                    "activationFunction"
                )  # magic activation function applied to this weighted sum, which outputs a single number from the neuron
            # activated = activationFunction(rawOutput)
            gained_output = (
                rawOutput * self.activation_gain
            ) + rawOutput  # SHOULD I PUT THIS NEW PASSTHROUGH HERE?
            activated = gelu(gained_output) + gained_output

            if debugPrints:
                ʕっʘ‿ʘʔっ("device check")
            if debugPrints:
                print("Device check:")
            if debugPrints:
                print("inputEmbeds:", _inputEmbeds.device)

            if True:
                if debugPrints:
                    ʕっʘ‿ʘʔっ("raw input history append")
                try:
                    _in_stats = torch.stack(
                        [inputEmbeds.norm(), inputEmbeds.mean()]
                    ).tolist()
                    self.rawInputNormHistory.append(_in_stats[0])
                    self.rawInputHistory.append(_in_stats[1])
                except Exception:
                    # If tensor operations hang, use safe defaults
                    self.rawInputNormHistory.append(1.0)
                    self.rawInputHistory.append(0.0)
                # self.rawInputHistory_tokens.append(inputEmbeds.norm(dim = 1).mean().item())
                # self.rawInputHistory_neurons.append(inputEmbeds.norm(dim = 0).mean().item())

                if debugPrints:
                    ʕっʘ‿ʘʔっ("activated output history append")
                try:
                    _out_stats = torch.stack(
                        [activated.norm(), activated.mean()]
                    ).tolist()
                    self.activatedOutputNormHistory.append(_out_stats[0])
                    self.activatedOutputHistory.append(_out_stats[1])
                    # self.activatedOutputHistory_tokens.append(activated.norm(dim = 1).mean().item())
                    # self.activatedOutputHistory_neurons.append(activated.norm(dim = 0).mean().item())
                except Exception:
                    # If tensor operations hang, use safe defaults
                    self.activatedOutputNormHistory.append(1.0)
                    self.activatedOutputHistory.append(0.0)

                history_length = len(self.activatedOutputHistory)
                if history_length >= self.numTokensPerStep:
                    if debugPrints:
                        ʕっʘ‿ʘʔっ("if len >= windowMAX, add to self.stats")
                    _flush_means = (
                        torch.stack(
                            [
                                torch.as_tensor(
                                    list(self.rawInputNormHistory),
                                    dtype=torch.float32,
                                    device=self.device,
                                ),
                                torch.as_tensor(
                                    list(self.rawInputHistory),
                                    dtype=torch.float32,
                                    device=self.device,
                                ),
                                torch.as_tensor(
                                    list(self.activatedOutputNormHistory),
                                    dtype=torch.float32,
                                    device=self.device,
                                ),
                                torch.as_tensor(
                                    list(self.activatedOutputHistory),
                                    dtype=torch.float32,
                                    device=self.device,
                                ),
                            ]
                        )
                        .mean(dim=1)
                        .tolist()
                    )
                    raw_input_norm, raw_input_mean, act_out_norm, act_out_mean = (
                        _flush_means
                    )
                    self.stats = {
                        "3N_0_rawInput_norm": raw_input_norm,
                        "3N_0_rawInput_mean": raw_input_mean,
                        # "3N_0_rawInput_norm_token": sum(self.rawInputHistory_tokens) / len(self.rawInputHistory_tokens),
                        # "3N_0_rawInput_norm_neuron": sum(self.rawInputHistory_neurons) / len(self.rawInputHistory_neurons),
                        # "3N_1_normedInput_norm": sum(self.normedInputNormHistory) / len(self.normedInputNormHistory),
                        # "3N_1_normedInput_mean": sum(self.normedInputHistory) / len(self.normedInputHistory),
                        # "3N_1_normedInput_norm_token": sum(self.normedInputHistory_tokens) / len(self.normedInputHistory_tokens),
                        # "3N_1_normedInput_norm_neuron": sum(self.normedInputHistory_neurons) / len(self.normedInputHistory_neurons),
                        # "3N_2_rawOutput_norm": sum(self.rawOutputNormHistory) / len(self.rawOutputNormHistory),
                        # "3N_2_rawOutput_mean": sum(self.rawOutputHistory) / len(self.rawOutputHistory),
                        # "3N_2_rawOutput_norm_token": sum(self.rawOutputHistory_tokens) / len(self.rawOutputHistory_tokens),
                        # "3N_2_rawOutput_norm_neuron": sum(self.rawOutputHistory_neurons) / len(self.rawOutputHistory_neurons),
                        "3N_x_actOut_norm": act_out_norm,
                        "3N_x_actOut_mean": act_out_mean,
                        # "3N_x_actOut_norm_token": sum(self.activatedOutputHistory_tokens) / len(self.activatedOutputHistory_tokens),
                        # "3N_x_actOut_norm_neuron": sum(self.activatedOutputHistory_neurons) / len(self.activatedOutputHistory_neurons),
                        # "3N_x_actOut_std_token": sum(self.activatedOutputHistory_std_token) / len(self.activatedOutputHistory_std_token),
                        # "3N_x_actOut_std_neuron": sum(self.activatedOutputHistory_std_neuron) / len(self.activatedOutputHistory_std_neuron),
                        # "3N_x_actOut_saturation": sum(self.activatedOutputHistory_saturation) / len(self.activatedOutputHistory_saturation),
                        # "3N_x_actOut_min": sum(self.activatedOutputHistory_min) / len(self.activatedOutputHistory_min),
                        # "3N_x_actOut_max": sum(self.activatedOutputHistory_max) / len(self.activatedOutputHistory_max),
                        # "3N_x_normedOutput_norm": sum(self.normedOutputHistory) / len(self.normedOutputHistory),
                        # "3N_x_normedOutput_norm_token": sum(self.normedOutputHistory_tokens) / len(self.normedOutputHistory_tokens),
                        # "3N_x_normedOutput_norm_neuron": sum(self.normedOutputHistory_neurons) / len(self.normedOutputHistory_neurons),
                    }

                    if debugPrints:
                        ʕっʘ‿ʘʔっ("clear stats")
                    self._clear_histories()

        return activated

    @whocalled
    def getStats(self):
        return self.stats


"""layer that applies the same set of neurons to each token embedding independently. - no sequence awareness!"""


class INTERNEURON_NETWORK(nn.Module):
    def __init__(
        self,
        _model,
        _counsellor,
        _calligraphist,
        _numTokensPerStep,
        _device=modelDevice,
    ):
        super().__init__()
        # self.inn_counsellor = COUNSELLOR("3INN", debug = debugPrints, durations = durationLogging)
        self.model = _model
        self.inn_counsellor = _counsellor
        self.device = _device
        self.calligraphist = _calligraphist
        self.numTokensPerStep = _numTokensPerStep
        self.entropyBonus = 0
        self.temperature = temperatureGOAL

        self.stats = {}
        self._history_attrs = [
            "activationsHistory",
            "activationsHistory_token",
            "activationsHistory_neuron",
            "normedMeanInputHistory",
            "normedMeanInputHistory_token",
            "normedMeanInputHistory_neuron",
            "combHistory",
            "combHistory_token",
            "combHistory_neuron",
            "refHistory",
            "refHistory_token",
            "refHistory_neuron",
            "scaledHistory",
            "scaledHistory_token",
            "scaledHistory_neuron",
            "combiOutHistory",
            "combiOutHistory_token",
            "combiOutHistory_neuron",
            "logitHistory",
            "combiScaleHistory",
        ]
        self._init_history_buffers()

        # SELF ALLOWED - nn.parameter!
        self.neurons = NEURON(
            _counsellor=self.inn_counsellor, _numTokensPerStep=self.numTokensPerStep
        )

        # === LONG-RANGE WINDOWS === (original system, untouched)
        self.cerebellum = nn.Parameter(
            torch.ones(len(allWindowSizes_new), device=self.device)
        )  # Window weighting layer - preserves learned weights when loading
        self.windowFractionality = nn.Parameter(
            torch.full((len(allWindowSizes_new),), -6.0, device=self.device)
        )
        self.logWindowSizes = nn.Parameter(
            torch.log(
                torch.tensor(
                    allWindowSizes_new, dtype=torch.float32, device=self.device
                )
            )
        )  # one tensor per window size!

        # === SHORT-RANGE WINDOWS === (new parallel system for finer temporal resolution)
        self.cerebellum_short = nn.Parameter(
            torch.ones(len(allWindowSizes_short), device=self.device)
        )
        self.windowFractionality_short = nn.Parameter(
            torch.full((len(allWindowSizes_short),), -6.0, device=self.device)
        )

        # Initialize with inverse-transformed values so they map to desired window sizes
        # After tanh scaling, these will give approximately: [1, 2, 4, 6, 8, 12, 16, 24, 48] tokens
        logWindowSizes_short_init = torch.tensor(
            [
                -1.5767,
                -1.5415,
                -1.4734,
                -1.4081,
                -1.3453,
                -1.2261,
                -1.1142,
                -0.9075,
                -0.3688,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.logWindowSizes_short = nn.Parameter(logWindowSizes_short_init)

        # Learned gate to blend long and short window views (starts small ~0.01)
        self.short_window_gate = nn.Parameter(
            torch.tensor(-4.595, device=self.device)
        )  # sigmoid ≈ 0.01

        # Learnable softmax temperature for window competition
        # Temperature = exp(param), so:
        #   param = 0.0 → temp = 1.0 (neutral)
        #   param > 0 → temp > 1.0 (collaborative, flat distribution)
        #   param < 0 → temp < 1.0 (competitive, sharp distribution)
        # Long windows start at 2.0 (collaborative)
        # Short windows start at 0.5 (competitive, so they differentiate faster)
        self.window_softmax_temperature = nn.Parameter(
            torch.tensor(0.693, device=self.device)
        )  # exp(0.693) ≈ 2.0
        self.window_softmax_temperature_short = nn.Parameter(
            torch.tensor(-0.69, device=self.device)
        )  # exp(-0.69) ≈ 0.5

        self.refinement2 = torch.nn.Sequential(
            nn.Linear(numNeurons, 512, device=self.device),  # bottleneck layer
            nn.GELU(),  # smoother activation
            nn.LayerNorm(512, device=self.device),  # mid normalisation
            nn.Linear(512, numNeurons, device=self.device),  # expand back
            nn.LayerNorm(numNeurons, device=self.device),  # final safety net
        )

        self._cerebellum_temperature = nn.Parameter(
            torch.tensor(0.0, device=self.device)
        )

    @property
    def cerebellum_temperature(self):
        return F.softplus(self._cerebellum_temperature) + 1e-6

        # MUST NOT BE ON SELF - global parameters that may be used by backward pass
        # numNeurons, embedDimension, activationFunction, allWindowSizes_new, etc

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
        tensor = torch.as_tensor(history, dtype=torch.float32, device=self.device)
        return tensor.mean().item() + offset

    @whocalled
    def forward(self, _inputEmbeds):
        with self.inn_counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            if debugPrints:
                ʕっʘ‿ʘʔっ("INN1: neuronActivationsPerToken")
            neuronActsPerToken = self.neurons(_inputEmbeds)

            if debugPrints:
                ʕっʘ‿ʘʔっ("compute fresh floatWindowSizes & fractionality")
            # learnable fractionality, allows it to decide how descrite the windows should be
            fractionality = torch.sigmoid(self.windowFractionality)  # (numWindows,)
            clamp_param(self.windowFractionality, -3.0, 3.0)

            minWindowSize = 0.1
            maxWindowSize = float(self.numTokensPerStep)

            clampedLogWindowSizes = (
                torch.tanh(self.logWindowSizes / 2) * 1.5
            )  # output ∈ [-1.5, 1.5]
            scaledTanh = (torch.tanh(clampedLogWindowSizes) + 1) / 2  # ∈ [0, 1]
            floatWindowSizes = (
                minWindowSize + (maxWindowSize - minWindowSize) * scaledTanh
            )
            intWindowSizes = torch.round(floatWindowSizes).clamp(min=1.0)

            if debugPrints:
                ʕっʘ‿ʘʔっ(
                    "blend int+float using fractionality"
                )  # straight-through estimator: gradient flows only through floatWindow
            windowTensor = (
                1 - fractionality
            ) * intWindowSizes + fractionality * floatWindowSizes
            windowTensor = windowTensor.unsqueeze(1)  # (numWindows, 1)

            # Store for stats
            self.windowTensor_used = windowTensor.squeeze(1).detach()
            self.floatWindowSizes_used = floatWindowSizes.detach()

            # entropy of window usage (soft histogram keeps gradients)
            window_sizes = windowTensor.squeeze(1)
            num_bins = 10
            bin_centers = torch.linspace(
                1.0, float(self.numTokensPerStep), steps=num_bins, device=self.device
            )
            bin_width = (float(self.numTokensPerStep) - 1.0) / max(num_bins - 1, 1)
            sigma = max(bin_width, 1e-3)
            dist = window_sizes.unsqueeze(1) - bin_centers.unsqueeze(0)
            weights = torch.exp(-0.5 * (dist / sigma) ** 2)
            soft_counts = weights.sum(dim=0)
            probs = soft_counts / (soft_counts.sum() + 1e-12)
            window_range = torch.max(windowTensor) - torch.min(windowTensor)
            desired_range = self.numTokensPerStep - 1.0  # ideally full spread
            self.rangePenalty = torch.relu(desired_range - window_range)

            std_window_size = torch.std(windowTensor)
            self.meanPenalty = torch.relu(0.1 - std_window_size)

            self.windowSizeEntropy = -torch.sum(probs * torch.log(probs + 1e-12))

            if debugPrints:
                ʕっʘ‿ʘʔっ("INN2: stackedWindowMeans")
            windowMeanStack = self.stackedWindowMeans(neuronActsPerToken, windowTensor)

            if debugPrints:
                ʕっʘ‿ʘʔっ("softmax weights from cerebellum")
            # Add small noise during training to break symmetry, but preserve learned weights
            cerebellum_with_noise = self.cerebellum
            if self.training:  # Only add noise during training, not inference
                noise = (
                    torch.randn_like(self.cerebellum) * 0.005
                )  # Very weak noise to gently break symmetry
                cerebellum_with_noise = self.cerebellum + noise

            temperature_for_sigmoid = (self.cerebellum_temperature * 0.5) + (
                self.temperature * 0.5
            )
            temperature_for_sigmoid = temperature_for_sigmoid.clamp(
                min=0.1
            )  # floor: prevents division by near-zero

            cerebellum_scaled = cerebellum_with_noise / temperature_for_sigmoid

            sigmoidWeights = torch.sigmoid(
                cerebellum_scaled
            )  # reduced from 10 to 2 for better gradient flow
            clamp_param(self.cerebellum, 0.01, 0.99)

            # Apply learnable softmax temperature (controls competition vs collaboration)
            # Temperature = exp(param), can be any positive value
            # temp < 1.0 (param < 0) = competitive/sharp, temp = 1.0 (param = 0) = neutral, temp > 1.0 (param > 0) = collaborative/flat
            softmax_temp = torch.exp(self.window_softmax_temperature).clamp(
                min=0.01
            )  # floor: exp(-Inf)→0 → softmax(all -Inf) = NaN
            log_weights = torch.log(sigmoidWeights + 1e-12)
            self.cerebellumSoft = F.softmax(log_weights / softmax_temp, dim=0)

            variance = torch.var(self.cerebellumSoft)
            self.windowWeightSpread = variance

            weightedWindowStack = windowMeanStack * self.cerebellumSoft.unsqueeze(1)

            if debugPrints:
                ʕっʘ‿ʘʔっ("entropy reward")
            self.windowEntropy = -torch.sum(
                self.cerebellumSoft * torch.log(self.cerebellumSoft + 1e-12)
            )
            self.entropyBonus = self.windowEntropy

            if debugPrints:
                ʕっʘ‿ʘʔっ("weightedWindowStack.sum (long windows)")
            combinedActivationsTensor_long = weightedWindowStack.sum(
                dim=0, keepdim=True
            )

            # === SHORT-RANGE WINDOW COMPUTATION === (parallel system)
            if debugPrints:
                ʕっʘ‿ʘʔっ("SHORT WINDOWS: compute window sizes")
            fractionality_short = torch.sigmoid(self.windowFractionality_short)
            clamp_param(self.windowFractionality_short, -3.0, 3.0)

            minWindowSize_short = 0.1
            maxWindowSize_short = float(windowShortMAX)  # 132 tokens max

            clampedLogWindowSizes_short = (
                torch.tanh(self.logWindowSizes_short / 2) * 1.5
            )
            scaledTanh_short = (torch.tanh(clampedLogWindowSizes_short) + 1) / 2
            floatWindowSizes_short = (
                minWindowSize_short
                + (maxWindowSize_short - minWindowSize_short) * scaledTanh_short
            )
            intWindowSizes_short = torch.round(floatWindowSizes_short).clamp(min=1.0)

            windowTensor_short = (
                1 - fractionality_short
            ) * intWindowSizes_short + fractionality_short * floatWindowSizes_short
            windowTensor_short = windowTensor_short.unsqueeze(1)

            # Store for stats
            self.windowTensor_short_used = windowTensor_short.squeeze(1).detach()
            self.floatWindowSizes_short_used = floatWindowSizes_short.detach()

            if debugPrints:
                ʕっʘ‿ʘʔっ("SHORT WINDOWS: stackedWindowMeans")
            windowMeanStack_short = self.stackedWindowMeans(
                neuronActsPerToken, windowTensor_short
            )

            if debugPrints:
                ʕっʘ‿ʘʔっ("SHORT WINDOWS: cerebellum weights")
            cerebellum_short_with_noise = self.cerebellum_short
            if self.training:
                noise_short = torch.randn_like(self.cerebellum_short) * 0.005
                cerebellum_short_with_noise = self.cerebellum_short + noise_short

            cerebellum_short_scaled = (
                cerebellum_short_with_noise / temperature_for_sigmoid
            )
            sigmoidWeights_short = torch.sigmoid(cerebellum_short_scaled)
            clamp_param(self.cerebellum_short, 0.01, 0.99)

            # Apply learnable softmax temperature for short windows
            # Starts at 0.5 (competitive) to encourage differentiation
            softmax_temp_short = torch.exp(self.window_softmax_temperature_short).clamp(
                min=0.01
            )  # floor: same NaN risk as long-window temp
            log_weights_short = torch.log(sigmoidWeights_short + 1e-12)
            self.cerebellumSoft_short = F.softmax(
                log_weights_short / softmax_temp_short, dim=0
            )

            weightedWindowStack_short = (
                windowMeanStack_short * self.cerebellumSoft_short.unsqueeze(1)
            )

            if debugPrints:
                ʕっʘ‿ʘʔっ("SHORT WINDOWS: sum weighted windows")
            combinedActivationsTensor_short = weightedWindowStack_short.sum(
                dim=0, keepdim=True
            )

            # === BLEND LONG AND SHORT WINDOW VIEWS ===
            if debugPrints:
                ʕっʘ‿ʘʔっ("BLEND: combine long + short windows")
            short_gate = torch.sigmoid(self.short_window_gate)
            combinedActivationsTensor = combinedActivationsTensor_long + (
                short_gate * combinedActivationsTensor_short
            )

            # Store values for stats
            self.short_gate_used = short_gate.detach()
            self.softmax_temp_used = softmax_temp.detach()
            self.softmax_temp_short_used = softmax_temp_short.detach()

            ʕっʘ‿ʘʔっ(self.refinement2)
            refinedActivations = self.refinement2(combinedActivationsTensor)

            # --- RESIDUAL CONNECTION ---
            FINALout = combinedActivationsTensor + refinedActivations

            # --- logging
            if debugPrints:
                ʕっʘ‿ʘʔっ("get inn stats no grad")
            try:
                # Use torch.no_grad() and add timeout handling for tensor operations
                with torch.no_grad():
                    # Check if tensors are valid before computing norms
                    if (
                        neuronActsPerToken.numel() > 0
                        and not torch.isnan(neuronActsPerToken).any()
                    ):
                        norm_val = neuronActsPerToken.norm().item()
                        if not math.isnan(norm_val) and not math.isinf(norm_val):
                            self.activationsHistory.append(norm_val)
                        else:
                            self.activationsHistory.append(1.0)
                    else:
                        self.activationsHistory.append(1.0)

                    # Same checks for combinedActivationsTensor
                    if (
                        combinedActivationsTensor.numel() > 0
                        and not torch.isnan(combinedActivationsTensor).any()
                    ):
                        comb_norm = combinedActivationsTensor.norm().item()
                        if not math.isnan(comb_norm) and not math.isinf(comb_norm):
                            self.combHistory.append(comb_norm)
                        else:
                            self.combHistory.append(1.0)
                    else:
                        self.combHistory.append(1.0)

                    # Same checks for refinedActivations
                    if (
                        refinedActivations.numel() > 0
                        and not torch.isnan(refinedActivations).any()
                    ):
                        ref_norm = refinedActivations.norm().item()
                        if not math.isnan(ref_norm) and not math.isinf(ref_norm):
                            self.refHistory.append(ref_norm)
                        else:
                            self.refHistory.append(1.0)
                    else:
                        self.refHistory.append(1.0)

            except Exception as e:
                # If tensor operations hang, use safe defaults
                print(f"[INN_LOGGING] Error in tensor norm calculations: {e}")
                self.activationsHistory.append(1.0)
                self.combHistory.append(1.0)
                self.refHistory.append(1.0)

            with torch.no_grad():
                # acts = neuronActsPerToken
                # comb = combinedActivationsTensor
                # ref = refinedActivations

                if debugPrints:
                    ʕっʘ‿ʘʔっ("get act norms")
                # norms = [
                #    acts.norm(),
                #    acts.norm(dim = 1).mean(),
                #    acts.norm(dim = 0).mean(),
                # ]
                # if debugPrints: ʕっʘ‿ʘʔっ("get comb norms")
                # norms += [
                #    comb.norm(),
                #    comb.norm(dim = 0).mean(),
                # ]
                # if debugPrints: ʕっʘ‿ʘʔっ("get ref norms")
                # norms += [
                # ref.norm(),
                # ref.norm(dim = 0).mean(),
                # ]
                # if debugPrints: print(f"{norms}")

                # if debugPrints: ʕっʘ‿ʘʔっ("norms to cpu")
                # norms_cpu = [x.cpu().item() for x in norms]
                # (
                #    acts_norm,
                #    #acts_token_norm,
                #    #acts_neuron_norm,
                # comb_norm,
                # comb_neuron_norm,
                #    ref_norm,
                # ref_neuron_norm,
                # ) = norms_cpu

        if len(self.combHistory) >= self.numTokensPerStep:
            if debugPrints:
                ʕっʘ‿ʘʔっ("add to self.stats")
            raw_acts_mean = self._history_mean(self.activationsHistory)
            refined_mean = self._history_mean(self.refHistory)
            self.stats = {
                "4INN_0_rawActs_norm": raw_acts_mean,
                # "4INN_0_rawActs_norm_token": sum(self.activationsHistory_token) / len(self.activationsHistory_token),
                # "4INN_0_rawActs_norm_neuron": sum(self.activationsHistory_neuron) / len(self.activationsHistory_neuron),
                # "4INN_2_combinedActs_norm": sum(self.combHistory) / len(self.combHistory),
                # "4INN_2_combinedActs_norm_neuron": sum(self.combHistory_neuron) / len(self.combHistory_neuron),
                "4INN_x_refinedActs_norm": refined_mean,
                # "4INN_3_refinedActs_norm_neuron": sum(self.refHistory_neuron) / len(self.refHistory_neuron),
                "4INN_windowSizesMean": floatWindowSizes.mean().item(),
                "4INN_windowEntropy": self.windowSizeEntropy,
            }
            if debugPrints:
                print(f"{self.stats}")

            if debugPrints:
                ʕっʘ‿ʘʔっ("clearstats")
            self._clear_histories()

        return FINALout

    @whocalled
    def stackedWindowMeans(
        self, activations: torch.Tensor, windowTensor: torch.Tensor
    ) -> torch.Tensor:
        """Fully vectorized, loop-free window mean calculation. Expects: windowTensor already built."""
        with self.inn_counsellor.infodump("stackedWindowMeans") as ʕっʘ‿ʘʔっ:
            # neuronActsPerToken = activations
            if debugPrints:
                ʕっʘ‿ʘʔっ("activations.shape")
            if len(activations.shape) == 1:
                activations = activations.unsqueeze(0)
            seqLen, embedDim = activations.shape

            # PADDING ENSURES UNDERLYING DATA HAS CORRECT TENSOR/VECTOR SHAPE FOR THE MASK
            maxLen = self.numTokensPerStep
            if debugPrints:
                ʕっʘ‿ʘʔっ("pad activations to maxLen")
            padded = torch.zeros((maxLen, embedDim), device=self.device)
            tail = activations[-maxLen:]
            padded[-tail.shape[0] :] = tail

            if debugPrints:
                ʕっʘ‿ʘʔっ("cumulative sum")
            cumsum = F.pad(padded, (0, 0, 1, 0)).cumsum(dim=0)

            windowSizes = torch.ceil(windowTensor.squeeze(1)).clamp(1, maxLen).long()
            start_idx = maxLen - windowSizes

            if debugPrints:
                ʕっʘ‿ʘʔっ("gather sums")
            sums = cumsum[maxLen].unsqueeze(0) - cumsum[start_idx]
            means = sums / windowTensor

            return means

    @whocalled
    def INN_getStats(self):
        with self.inn_counsellor.infodump("INN_getStats") as ʕっʘ‿ʘʔっ:
            INN_cerebellum_str = ""
            if collectStats and n_collectStats:
                if debugPrints:
                    ʕっʘ‿ʘʔっ("torch.no_grad♥")

                if INN_cerebellumStats:
                    if debugPrints:
                        ʕっʘ‿ʘʔっ("♥getCerebellumStats")

                    # --- RESILIENCE UPGRADE ---
                    # Safely get the attributes. If they don't exist (because forward failed),
                    # getattr will provide an empty tensor as a default.
                    float_windows = getattr(
                        self, "floatWindowSizes_used", torch.empty(0)
                    )
                    soft_weights = getattr(self, "cerebellumSoft", torch.empty(0))
                    tensor_windows = getattr(self, "windowTensor_used", torch.empty(0))

                    # Check if we have all the data we need before proceeding.
                    if all(
                        t.numel() > 0
                        for t in [
                            float_windows,
                            self.cerebellum,
                            soft_weights,
                            tensor_windows,
                        ]
                    ):
                        INN_cerebellumStats_fullValues = zip(
                            float_windows, self.cerebellum, soft_weights, tensor_windows
                        )
                        for w, raw, soft, tensor in INN_cerebellumStats_fullValues:
                            w_val, raw_val, soft_val, tensor_val = torch.stack(
                                [w, raw, soft, tensor]
                            ).tolist()
                            self.stats[f"INN_cerebellum_W{int(w_val)}_float"] = w_val
                            self.stats[f"INN_cerebellum_W{int(w_val)}"] = raw_val
                            self.stats[f"INN_cerebellum_W{int(w_val)}_softMax"] = (
                                soft_val
                            )
                            self.stats[f"INN_cerebellum_W{int(w_val)}_tensor"] = (
                                tensor_val
                            )
                        # if debugPrints: print(f"cerebellum: {self.cerebellum}, soft: {self.cerebellumSoft} mean: {self.stats['4INN_cerebellumMean']} std: {self.stats['4INN_cerebellumStd']}")
                        if debugPrints:
                            ʕっʘ‿ʘʔっ("♥cerebellumString")
                        INN_cerebellum_str = (
                            self.calligraphist.S_formatWindowBiasTriplets(
                                label="INN_cerebellum",
                                rawTensor=self.cerebellum,
                                softTensor=self.cerebellumSoft,
                                windowSizes=self.floatWindowSizes_used,
                                windowTensor=self.windowTensor_used,
                                per_window_style=True,
                            )
                        )
                        if debugPrints:
                            print(f"{INN_cerebellum_str}")
                    else:
                        if debugPrints:
                            print(
                                "[INN_getStats] Missing one or more required tensors from forward pass. Skipping cerebellum stats."
                            )
                        INN_cerebellum_str = "<cerebellum data unavailable>"
                    # --- END RESILIENCE UPGRADE ---

                    # === SHORT WINDOW STATS ===
                    if debugPrints:
                        ʕっʘ‿ʘʔっ("♥shortWindowStats")
                    float_windows_short = getattr(
                        self, "floatWindowSizes_short_used", torch.empty(0)
                    )
                    soft_weights_short = getattr(
                        self, "cerebellumSoft_short", torch.empty(0)
                    )
                    tensor_windows_short = getattr(
                        self, "windowTensor_short_used", torch.empty(0)
                    )
                    short_gate = getattr(self, "short_gate_used", torch.tensor(0.0))

                    if all(
                        t.numel() > 0
                        for t in [
                            float_windows_short,
                            self.cerebellum_short,
                            soft_weights_short,
                            tensor_windows_short,
                        ]
                    ):
                        # Add short window individual stats
                        for w, raw, soft, tensor in zip(
                            float_windows_short,
                            self.cerebellum_short,
                            soft_weights_short,
                            tensor_windows_short,
                        ):
                            w_val, raw_val, soft_val, tensor_val = torch.stack(
                                [w, raw, soft, tensor]
                            ).tolist()
                            self.stats[f"INN_cerebellum_SHORT_W{int(w_val)}_float"] = (
                                w_val
                            )
                            self.stats[f"INN_cerebellum_SHORT_W{int(w_val)}"] = raw_val
                            self.stats[
                                f"INN_cerebellum_SHORT_W{int(w_val)}_softMax"
                            ] = soft_val
                            self.stats[f"INN_cerebellum_SHORT_W{int(w_val)}_tensor"] = (
                                tensor_val
                            )

                        # Add summary stats
                        _short_summary = torch.stack(
                            [short_gate, float_windows_short.mean()]
                        ).tolist()
                        self.stats["4INN_short_gate"] = _short_summary[0]
                        self.stats["4INN_short_windowSizesMean"] = _short_summary[1]

                        # Add softmax temperature stats
                        softmax_temp = getattr(
                            self, "softmax_temp_used", torch.tensor(1.0)
                        )
                        softmax_temp_short = getattr(
                            self, "softmax_temp_short_used", torch.tensor(1.0)
                        )
                        _temps = torch.stack(
                            [softmax_temp, softmax_temp_short]
                        ).tolist()
                        self.stats["4INN_window_softmax_temp"] = _temps[0]
                        self.stats["4INN_window_softmax_temp_short"] = _temps[1]

                        if debugPrints:
                            ʕっʘ‿ʘʔっ("♥shortCerebellumString")
                        INN_cerebellum_short_str = (
                            self.calligraphist.S_formatWindowBiasTriplets(
                                label="INN_cerebellum_SHORT",
                                rawTensor=self.cerebellum_short,
                                softTensor=self.cerebellumSoft_short,
                                windowSizes=self.floatWindowSizes_short_used,
                                windowTensor=self.windowTensor_short_used,
                                per_window_style=True,
                            )
                        )
                        if debugPrints:
                            print(f"{INN_cerebellum_short_str}")

                        # Combine both cerebellum strings for display
                        INN_cerebellum_str = (
                            INN_cerebellum_str
                            + "\n--- SHORT WINDOWS ---\n"
                            + INN_cerebellum_short_str
                        )
                    else:
                        if debugPrints:
                            print(
                                "[INN_getStats] Missing short window tensors from forward pass."
                            )

            if debugPrints:
                ʕっʘ‿ʘʔっ("update self.stats")
            # self.stats.update({f"{k}": v for k, v in self.neurons.getStats().items()})
        return self.stats, INN_cerebellum_str

    @whocalled
    def clearStats(self):
        self._init_history_buffers()
        if hasattr(self, "neurons"):
            self.neurons._init_history_buffers()
