# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ ---
# EMBEDDING LAYER // brain/LAYERS/embed.py
# v1.3

import math

import torch
import torch.nn as nn

from config import *
from utils.helpers import clamp_param

"""creates an embedding layer for each word in the vocabulary"""


class EMBED(nn.Module):
    def __init__(self, _counsellor, _device=modelDevice):
        super().__init__()
        self.counsellor = _counsellor
        self.device = _device
        self.stats = {}

        """creates the embedding weights matrix with random numbers initially"""
        self.e_weights = nn.Parameter(
            torch.randn(vocabSize, embedDimension, device=self.device)
        )  # [2000,]
        self.embedNorm = nn.LayerNorm(embedDimension, device=self.device)
        self.weightsScale = nn.Parameter(torch.tensor(0.5))
        self.normScale = nn.Parameter(torch.tensor(0.5))
        self.lastSavedEmbeds = (
            self.e_weights.detach().clone()
        )  # THIS IS INITIALISED ONCE, FOR STATS, DOES NOT BREAK GRAPH CONFIRMED!!

        # Output normalization to prevent explosion after scale multiplication
        self.finalNorm = nn.LayerNorm(embedDimension, device=self.device)

        self.pixelEmbed = nn.Linear(3, embedDimension, device=self.device)

        self.maxPosLen = 2048
        self.posEmbedding = nn.Embedding(
            self.maxPosLen, embedDimension, device=self.device
        )
        self.dropout = nn.Dropout(p=embedDropoutProb)
        self.posDropout = nn.Dropout(p=embedDropoutProb)
        self.scale = math.sqrt(embedDimension)

    """looks up and returns the embedding vector for a specific token index"""

    @whocalled
    def forward(self, _tokenIndex=None, _pixel=None):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            if not skipPixels and (_pixel is not None):
                if debugPrints:
                    ʕっʘ‿ʘʔっ("E0_pixelInjected")
                if _pixel.dim() == 1:  # [3]
                    if debugPrints:
                        ʕっʘ‿ʘʔっ("pixel.dim == 1")
                    self.embedVector = self.pixelEmbed(_pixel.unsqueeze(0)).squeeze(
                        0
                    )  # [embedDimension]
                elif _pixel.dim() == 2:  # [seq_len, 3]
                    if debugPrints:
                        ʕっʘ‿ʘʔっ("pixel.dim == 2")
                    self.embedVector = self.pixelEmbed(
                        _pixel
                    )  # [seq_len, embedDimension]
                else:
                    raise ValueError(f"Pixel input has wrong shape: {_pixel.shape}")
            else:
                if debugPrints:
                    ʕっʘ‿ʘʔっ(
                        "E0_embedVector"
                    )  # <- vocab???? base token indexes seem to come in here so... from tutor??
                self.embedVector = self.e_weights[_tokenIndex]
            if debugPrints:
                ʕっʘ‿ʘʔっ("E1_embedNormed")  # <- E1
            # EXPERIMENT: complementary affine split.
            # Learned gamma and beta both remain active and trainable, but
            # never act on the same feature coordinate.
            _affine_idx = torch.arange(
                self.embedNorm.weight.numel(),
                device=self.embedNorm.weight.device,
            )
            # EXPERIMENT: shared beta detour.
            #
            # Gamma and beta are once again fully shared across ALL
            # dimensions and BOTH token/RGB invocations.
            #
            # Beta no longer enters as LayerNorm's direct additive bias.
            # Instead it perturbs the representation BEFORE normalization,
            # forcing its influence through cross-feature mean/variance
            # geometry before learned gamma acts on the result.
            _beta_shifted = self.embedVector + self.embedNorm.bias

            self.embedNormed = torch.nn.functional.layer_norm(
                _beta_shifted,
                self.embedNorm.normalized_shape,
                weight=self.embedNorm.weight,
                bias=None,
                eps=self.embedNorm.eps,
            )

            combined = (
                self.embedVector + self.embedNormed
            )  # direct passthrough instead of scaling cause he abuses them lol, -0.005 scale... wtf is that!?
            # Final normalization of the residual embedding path.
            normalized = self.finalNorm(combined)

            # LOOKING GLASS: observe normalization geometry only.
            # LayerNorm operates across the final feature dimension, so record
            # per-vector feature std rather than one global sequence std.
            with torch.no_grad():
                kind = (
                    "pixel"
                    if (not skipPixels and _pixel is not None)
                    else "token"
                )

                def _std_profile(tensor):
                    std = tensor.detach().float().std(dim=-1, unbiased=False)
                    return {
                        "min": float(std.min().item()),
                        "mean": float(std.mean().item()),
                        "max": float(std.max().item()),
                    }

                trace = {
                    "raw": _std_profile(self.embedVector),
                    "norm1": _std_profile(self.embedNormed),
                    "combined": _std_profile(combined),
                    "norm2": _std_profile(normalized),
                }

                if not hasattr(self, "_looking_glass"):
                    self._looking_glass = {}
                self._looking_glass[kind] = trace

            self.embedFinal = self.embedVector + normalized
            self.embedFinal = self.dropout(self.embedFinal)
            clamp_param(self.weightsScale, -10, 10)
            clamp_param(self.normScale, -10, 10)
            return self.embedFinal  # E3 -> N??

    @whocalled
    def getEmbedStats(self):
        with self.counsellor.infodump("getEmbedStats") as ʕっʘ‿ʘʔっ:
            if debugPrints:
                ʕっʘ‿ʘʔっ("with torch.no_grad")
            with torch.no_grad():
                self.stats = {}

                # Latest detached token/pixel normalization geometry.
                looking_glass = getattr(self, "_looking_glass", {})
                for kind in ("token", "pixel"):
                    trace = looking_glass.get(kind)
                    if trace:
                        print(
                            f"[EMBED LOOKING GLASS {kind}] "
                            f"raw={trace['raw']['mean']:.6g} "
                            f"norm1={trace['norm1']['mean']:.6g} "
                            f"combined={trace['combined']['mean']:.6g} "
                            f"[{trace['combined']['min']:.6g}.."
                            f"{trace['combined']['max']:.6g}] "
                            f"norm2={trace['norm2']['mean']:.6g}"
                        )
                # DEEP LOOKING GLASS: inspect the learned LayerNorm affine
                # parameters and estimate their inverse-std amplification.
                # Telemetry only; detached values do not participate in gradients.
                def _param_profile(param):
                    t = param.detach().float()
                    return {
                        "mean": float(t.mean().item()),
                        "std": float(t.std(unbiased=False).item()),
                        "min": float(t.min().item()),
                        "max": float(t.max().item()),
                        "norm": float(t.norm().item()),
                        "absmean": float(t.abs().mean().item()),
                        "absmax": float(t.abs().max().item()),
                    }

                param_glass = {
                    "embedNorm.weight": _param_profile(self.embedNorm.weight),
                    "embedNorm.bias": _param_profile(self.embedNorm.bias),
                    "finalNorm.weight": _param_profile(self.finalNorm.weight),
                    "finalNorm.bias": _param_profile(self.finalNorm.bias),
                }

                for name, profile in param_glass.items():
                    print(
                        f"[EMBED PARAM GLASS {name}] "
                        f"mean={profile['mean']:.6g} "
                        f"std={profile['std']:.6g} "
                        f"min={profile['min']:.6g} "
                        f"max={profile['max']:.6g} "
                        f"norm={profile['norm']:.6g} "
                        f"absmean={profile['absmean']:.6g} "
                        f"absmax={profile['absmax']:.6g}"
                    )

                # Approximate |gamma| / input_std. This is NOT the complete
                # LayerNorm Jacobian; it is a useful scale diagnostic.
                embed_gamma = param_glass["embedNorm.weight"]
                final_gamma = param_glass["finalNorm.weight"]

                for kind in ("token", "pixel"):
                    trace = looking_glass.get(kind)
                    if not trace:
                        continue

                    raw_mean_std = max(trace["raw"]["mean"], 1e-12)
                    raw_min_std = max(trace["raw"]["min"], 1e-12)
                    combined_mean_std = max(
                        trace["combined"]["mean"], 1e-12
                    )
                    combined_min_std = max(
                        trace["combined"]["min"], 1e-12
                    )

                    print(
                        f"[EMBED SCALE GLASS {kind}] "
                        f"raw_std={raw_mean_std:.6g} "
                        f"embed_gamma_absmean="
                        f"{embed_gamma['absmean']:.6g} "
                        f"embed_gamma_absmax="
                        f"{embed_gamma['absmax']:.6g} "
                        f"embed_gain~="
                        f"{embed_gamma['absmean'] / raw_mean_std:.6g} "
                        f"embed_gain_upper~="
                        f"{embed_gamma['absmax'] / raw_min_std:.6g} "
                        f"combined_std={combined_mean_std:.6g} "
                        f"final_gamma_absmean="
                        f"{final_gamma['absmean']:.6g} "
                        f"final_gamma_absmax="
                        f"{final_gamma['absmax']:.6g} "
                        f"final_gain~="
                        f"{final_gamma['absmean'] / combined_mean_std:.6g} "
                        f"final_gain_upper~="
                        f"{final_gamma['absmax'] / combined_min_std:.6g}"
                    )

                if debugPrints:
                    ʕっʘ‿ʘʔっ("embedNorms = torch.norm(self.e_weights, dim = 1)")
                # embedNorms = torch.norm(self.e_weights, dim = 1)
                if debugPrints:
                    ʕっʘ‿ʘʔっ("embedNorms Stats")
                # self.stats["1E_weightNormMean"] = embedNorms.mean().item()
                # self.stats["1E_weightNormStd"] = embedNorms.std().item()
                # self.stats["1E_weightNormMax"] = embedNorms.max().item()

                if debugPrints:
                    ʕっʘ‿ʘʔっ("vectorNorm stats")
                try:
                    self.stats["1E_0_vector_norm"] = self.embedVector.norm().item()
                    # self.stats["1E_1_normed_norm"] = self.embedNormed.norm().item()
                    self.stats["1E_0_vector_mean"] = self.embedVector.mean().item()
                    # self.stats["1E_1_normed_mean"] = self.embedNormed.mean().item()
                    self.stats["1E_x_final_norm"] = self.embedFinal.norm().item()
                    self.stats["1E_x_final_mean"] = self.embedFinal.mean().item()
                    ###self.stats["1E_1_pixelEmbed_norm"] = self.pixelEmbed.norm().item()###
                    ###self.stats["1E_1_pixelEmbed_mean"] = self.pixelEmbed.weight.mean().item()###
                    # self.stats["1E_0_vector_scale"] = self.weightsScale.norm().item()
                    # self.stats["1E_1_normed_scale"] = self.normScale.norm().item()
                    pos_emb_row_norm = (
                        self.posEmbedding.weight.norm(dim=1).mean().item()
                    )
                    self.stats["1E_1_posEmbWeight_norm"] = pos_emb_row_norm
                    self.stats["1E_1_posEmbWeight_mean"] = (
                        self.posEmbedding.weight.mean().item()
                    )
                except Exception:
                    # If tensor operations hang or fail, use safe defaults
                    self.stats["1E_0_vector_norm"] = 1.0
                    self.stats["1E_0_vector_mean"] = 0.0
                    self.stats["1E_x_final_norm"] = 1.0
                    self.stats["1E_x_final_mean"] = 0.0
                    self.stats["1E_1_posEmbWeight_norm"] = 1.0
                    self.stats["1E_1_posEmbWeight_mean"] = 0.0

                # dimMean = self.e_weights.detach().clone().mean(dim = 0)
                # self.stats["1E_dimMean"] = dimMean
                # dimSparsity = (dimMean.abs() < 1e-4).float().mean().item()
                # self.stats["1E_dimSparsity"] = dimSparsity

                # Drift since last save
                # drift = torch.norm(self.e_weights - self.lastSavedEmbeds).item()
                # self.stats["1E_drift"] = drift
                # self.lastSavedEmbeds = self.e_weights.detach().clone()

                return self.stats


