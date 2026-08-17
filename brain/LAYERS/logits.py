# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ ---
# OUTPUT LAYER FOR LOGIT PREDICTION
# brain/LAYERS/logits.py
# v1.1

import torch
import torch.nn as nn

from config import *
from utils.helpers import clamp_param, init_history_buffers

"""final layer, maps neuron activations to logits for each token in the vocab"""


class LOGITS(nn.Module):
    def __init__(self, _counsellor, _device, _numTokensPerStep):
        super().__init__()
        self.device = _device
        self.counsellor = _counsellor
        self.numTokensPerStep = _numTokensPerStep
        self.lastSavedWeights = 0  # for stats

        self.l_weights = nn.Parameter(
            torch.randn(numNeurons, vocabSize, device=self.device)
        )  # this is set to move the NEURON ACTIVATIONS (10000) onto VOCAB SIZE (2000)
        self.l_bias = nn.Parameter(torch.zeros(vocabSize, device=self.device))
        self.activationNorm = nn.LayerNorm(numNeurons, device=self.device)
        self.rawActivationsScale = nn.Parameter(torch.tensor(0.5))
        self.normedActivationsScale = nn.Parameter(torch.tensor(0.5))

        self.logitNorm = nn.LayerNorm(vocabSize, device=self.device)

        self.grad_stats = {}
        self.stats = {}
        self._history_attrs = [
            "tensorNormHist",
            "normedNormHist",
            "activNormHist",
            "logitNormHist",
            "normLayerNormHist",
            "finalNormHist",
            "tensorHist",
            "normedHist",
            "activHist",
            "logitHist",
            "normLayerHist",
            "finalHist",
            "tensorMinHist",
            "normedMinHist",
            "activMinHist",
            "logitMinHist",
            "normLayerMinHist",
            "finalMinHist",
            "tensorMaxHist",
            "normedMaxHist",
            "activMaxHist",
            "logitMaxHist",
            "normLayerMaxHist",
            "finalMaxHist",
        ]
        if diagnoseLogitHead:
            self._history_attrs.extend([
                "rawLogitOutputNormHist", "rawLogitOutputMeanHist", "rawLogitOutputStdHist", "rawLogitOutputMinHist", "rawLogitOutputMaxHist",
                "finalLogitNormHist", "finalLogitMeanHist", "finalLogitStdHist", "finalLogitMinHist", "finalLogitMaxHist",
                "rawToFinalNormRatioHist", "rawToFinalStdRatioHist"
            ])
        init_history_buffers(self, self._history_attrs, self.numTokensPerStep)



    @whocalled
    def forward(self, _meanActivationsTensor):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            # <- = from
            # INN? -> L1 -> L2 -> L3 -> L4 -> L5 -> L6 -> *
            """imports the activations from interneuronNetwork, assuming that is is a tensor"""
            if debugPrints:
                ʕっʘ‿ʘʔっ(
                    "L1: activationsTensor"
                )  # <- INN? no? seems to come from babyLLM? maybe through babyLLM?
            actsTensor = _meanActivationsTensor  # _1
            # rawActScale = torch.sigmoid(self.rawActivationsScale)
            # normActScale = torch.sigmoid(self.normedActivationsScale)
            if debugPrints:
                ʕっʘ‿ʘʔっ("L2: normedActivationsTensor")  # <- L1
            normedActsTensor = self.activationNorm(actsTensor)  # _2
            # scaledActs = (actsTensor * rawActScale + normedActsTensor * normActScale)
            scaledActs = (
                actsTensor + normedActsTensor
            )  # direct pass through skipping scaling
            # scaledActs = scaledActs.clamp(-10, 10)

            if debugPrints:
                print(
                    f"Debug logits: activations shape before @ weights: {scaledActs.shape}"
                )
            if debugPrints:
                print(f"Debug logits: weights shape: {self.l_weights.shape}")

            if debugPrints:
                ʕっʘ‿ʘʔっ("L3: scaledActivations")  # <- L1 + L2
            # logitOutput = (scaledActs @ self.l_weights / (numNeurons ** 0.5)) + self.l_bias
            logitOutput = (scaledActs @ self.l_weights) + self.l_bias
            # logitOutput = logitOutput.clamp(-60, 60) # DO NOT CLAMP, softens too much!
            logitNormed = self.logitNorm(logitOutput)  # + logitOutput  # softly smooth
            finalLogit = logitNormed

            if debugPrints:
                print(
                    f"Debug logits: logitOutput shape AFTER @ weights: {logitOutput.shape}"
                )

            if debugPrints:
                ʕっʘ‿ʘʔっ("clamp scalar parameters")
            clamp_param(self.rawActivationsScale, 0, 0.75)
            clamp_param(self.normedActivationsScale, 0, 0.75)

            if debugPrints:
                ʕっʘ‿ʘʔっ("append rolling self.stats")
            _a_stats = torch.stack(
                [
                    actsTensor.norm(),
                    actsTensor.mean(),
                    actsTensor.min(),
                    actsTensor.max(),
                ]
            ).tolist()
            _f_stats = torch.stack(
                [
                    finalLogit.norm(),
                    finalLogit.mean(),
                    finalLogit.min(),
                    finalLogit.max(),
                ]
            ).tolist()
            self.tensorNormHist.append(_a_stats[0])
            self.finalNormHist.append(_f_stats[0])
            self.tensorHist.append(_a_stats[1])
            # self.normedHist.append(normedActsTensor.mean().item())
            # self.activHist.append(scaledActs.mean().item())
            # self.logitHist.append(logitOutput.mean().item())
            # self.normLayerHist.append(logitNormed.mean().item())
            self.finalHist.append(_f_stats[1])
            self.tensorMinHist.append(_a_stats[2])
            # self.normedMinHist.append(normedActsTensor.min().item())
            # self.activMinHist.append(scaledActs.min().item())
            # self.logitMinHist.append(logitOutput.min().item())
            # self.normLayerMinHist.append(logitNormed.min().item())
            self.finalMinHist.append(_f_stats[2])
            self.tensorMaxHist.append(_a_stats[3])
            # self.normedMaxHist.append(normedActsTensor.max().item())
            # self.activMaxHist.append(scaledActs.max().item())
            # self.logitMaxHist.append(logitOutput.max().item())
            # self.normLayerMaxHist.append(logitNormed.max().item())
            self.finalMaxHist.append(_f_stats[3])

            if diagnoseLogitHead:
                with torch.no_grad():
                    lo_norm = float(logitOutput.detach().norm().item())
                    lo_mean = float(logitOutput.detach().mean().item())
                    lo_std = float(logitOutput.detach().std().item())
                    lo_min = float(logitOutput.detach().min().item())
                    lo_max = float(logitOutput.detach().max().item())

                    fl_norm = float(finalLogit.detach().norm().item())
                    fl_mean = float(finalLogit.detach().mean().item())
                    fl_std = float(finalLogit.detach().std().item())
                    fl_min = float(finalLogit.detach().min().item())
                    fl_max = float(finalLogit.detach().max().item())

                    ratio_norm = lo_norm / fl_norm if fl_norm != 0 else 0.0
                    ratio_std = lo_std / fl_std if fl_std != 0 else 0.0

                    self.rawLogitOutputNormHist.append(lo_norm)
                    self.rawLogitOutputMeanHist.append(lo_mean)
                    self.rawLogitOutputStdHist.append(lo_std)
                    self.rawLogitOutputMinHist.append(lo_min)
                    self.rawLogitOutputMaxHist.append(lo_max)

                    self.finalLogitNormHist.append(fl_norm)
                    self.finalLogitMeanHist.append(fl_mean)
                    self.finalLogitStdHist.append(fl_std)
                    self.finalLogitMinHist.append(fl_min)
                    self.finalLogitMaxHist.append(fl_max)

                    self.rawToFinalNormRatioHist.append(ratio_norm)
                    self.rawToFinalStdRatioHist.append(ratio_std)

            if len(self.tensorHist) >= self.numTokensPerStep:
                if debugPrints:
                    ʕっʘ‿ʘʔっ("clear rolling self.stats at end of window")
                _flush_means = (
                    torch.stack(
                        [
                            torch.as_tensor(
                                list(self.tensorNormHist),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            torch.as_tensor(
                                list(self.finalNormHist),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            torch.as_tensor(
                                list(self.tensorHist),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            torch.as_tensor(
                                list(self.finalHist),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            torch.as_tensor(
                                list(self.tensorMinHist),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            torch.as_tensor(
                                list(self.finalMinHist),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            torch.as_tensor(
                                list(self.tensorMaxHist),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            torch.as_tensor(
                                list(self.finalMaxHist),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                        ]
                    )
                    .mean(dim=1)
                    .tolist()
                )
                (
                    acts_norm,
                    final_norm,
                    acts_mean,
                    final_mean,
                    acts_min,
                    final_min,
                    acts_max,
                    final_max,
                ) = _flush_means
                self.stats = {
                    "7L_0_actsTensor_norm": acts_norm,
                    # "7L_1_normActsTensor_norm": sum(self.normedNormHist) / len(self.normedNormHist),
                    # "7L_2_scaledActsTensor_norm": sum(self.activNormHist) / len(self.activNormHist),
                    # "7L_3_out_norm": sum(self.logitNormHist) / len(self.logitNormHist),
                    # "7L_4_outNorm_norm": sum(self.normLayerNormHist) / len(self.normLayerNormHist),
                    "7L_x_final_norm": final_norm,
                    "7L_0_actsTensor_mean": acts_mean,
                    # "7L_1_normActsTensor_mean": sum(self.normedHist) / len(self.normedHist),
                    # "7L_2_scaledActsTensor_mean": sum(self.activHist) / len(self.activHist),
                    # "7L_3_out_mean": sum(self.logitHist) / len(self.logitHist),
                    # "7L_4_outNorm_mean": sum(self.normLayerHist) / len(self.normLayerHist),
                    "7L_x_final_mean": final_mean,
                    "7L_0_actsTensor_min": acts_min,
                    # "7L_1_normActsTensor_min": sum(self.normedMinHist) / len(self.normedMinHist),
                    # "7L_2_scaledActsTensor_min": sum(self.activMinHist) / len(self.activMinHist),
                    # "7L_3_out_min": sum(self.logitMinHist) / len(self.logitMinHist),
                    # "7L_4_outNorm_min": sum(self.normLayerMinHist) / len(self.normLayerMinHist),
                    "7L_x_final_min": final_min,
                    "7L_0_actsTensor_max": acts_max,
                    # "7L_1_normActsTensor_max": sum(self.normedMaxHist) / len(self.normedMaxHist),
                    # "7L_2_scaledActsTensor_max": sum(self.activMaxHist) / len(self.activMaxHist),
                    # "7L_3_out_max": sum(self.logitMaxHist) / len(self.logitMaxHist),
                    # "7L_4_outNorm_max": sum(self.normLayerMaxHist) / len(self.normLayerMaxHist),
                    "7L_x_final_max": final_max,
                }

                if diagnoseLogitHead:
                    def get_mean(hist):
                        return sum(hist) / len(hist) if hist else 0.0

                    self.stats.update({
                        "7L_3_rawLogitOutput_norm": get_mean(self.rawLogitOutputNormHist),
                        "7L_3_rawLogitOutput_mean": get_mean(self.rawLogitOutputMeanHist),
                        "7L_3_rawLogitOutput_std": get_mean(self.rawLogitOutputStdHist),
                        "7L_3_rawLogitOutput_min": get_mean(self.rawLogitOutputMinHist),
                        "7L_3_rawLogitOutput_max": get_mean(self.rawLogitOutputMaxHist),

                        "7L_4_finalLogit_norm": get_mean(self.finalLogitNormHist),
                        "7L_4_finalLogit_mean": get_mean(self.finalLogitMeanHist),
                        "7L_4_finalLogit_std": get_mean(self.finalLogitStdHist),
                        "7L_4_finalLogit_min": get_mean(self.finalLogitMinHist),
                        "7L_4_finalLogit_max": get_mean(self.finalLogitMaxHist),

                        "7L_6_raw_to_final_norm_ratio": get_mean(self.rawToFinalNormRatioHist),
                        "7L_6_raw_to_final_std_ratio": get_mean(self.rawToFinalStdRatioHist),
                    })

                for attr in self._history_attrs:
                    getattr(self, attr).clear()

            # with torch.no_grad():
            # topValues, topIndices = torch.topk(finalLogit, 5)
            # self.stats["7L_topLogits"] = topValues.tolist()
            # self.stats["7L_topIndices"] = topIndices.tolist()
            # self.stats["7L_logitMax"] = finalLogit.max().item()
            # self.stats["7L_logitMin"] = finalLogit.min().item()
            # self.stats["7L_logitMean"] = finalLogit.mean().item()
            # self.stats["7L_logitStd"] = finalLogit.std().item()
            # self.stats["7L_0_actsTensor_scale"] = rawScale.item()
            # self.stats["7L_1_normActsTensor_scale"] = normedScale.item()
            # self.stats["7L_3_outSigmoid_scale"] = outScale.detach().item()
            # self.stats["7L_4_outNormSigmoid_scale"] = normOutScale.detach().item()
            # if debugPrints: print("activation norm:", scaledActs.norm().item())
            if debugPrints:
                print("weight norm mean:", self.l_weights.norm(dim=0).mean().item())
            if debugPrints:
                print("weight norm max:", self.l_weights.norm(dim=0).max().item())

            # return logits (not softmax) for better gradient computation in cross-entropy loss
            return finalLogit  # L6 ->

    @whocalled
    def getLogitStats(self):
        with self.counsellor.infodump("getLogitStats") as ʕっʘ‿ʘʔっ:
            if diagnoseLogitHead:
                with torch.no_grad():
                    w = self.logitNorm.weight.detach()
                    self.stats["7L_5_logitNorm_weight_norm"] = float(w.norm().item())
                    self.stats["7L_5_logitNorm_weight_mean"] = float(w.mean().item())
                    self.stats["7L_5_logitNorm_weight_std"] = float(w.std().item())
                    self.stats["7L_5_logitNorm_weight_min"] = float(w.min().item())
                    self.stats["7L_5_logitNorm_weight_max"] = float(w.max().item())
                    b = self.logitNorm.bias.detach()
                    self.stats["7L_5_logitNorm_bias_norm"] = float(b.norm().item())
                    self.stats["7L_5_logitNorm_bias_mean"] = float(b.mean().item())
                    self.stats["7L_5_logitNorm_bias_std"] = float(b.std().item())
                    self.stats["7L_5_logitNorm_bias_min"] = float(b.min().item())
                    self.stats["7L_5_logitNorm_bias_max"] = float(b.max().item())
                if self.grad_stats:
                    self.stats.update(self.grad_stats)

        return self.stats

    @whocalled
    def clearStats(self):
        init_history_buffers(self, self._history_attrs, self.numTokensPerStep)
