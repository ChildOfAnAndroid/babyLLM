# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ ---
# GATED MULTI-HEAD ATTENTION LAYER // brain/LAYERS/attention.py

import torch
import torch.nn as nn
from config import *

"""multi-head self attention with a learnable gate so existing training isn't thrown away"""
class GATED_MHA(nn.Module):
    def __init__(self, _counsellor, _num_heads: int = 16, _device = modelDevice, _numTokensPerStep = numTokensPerStepSTART):
        super().__init__()
        self.counsellor = _counsellor
        self.device = _device
        self.num_heads = _num_heads
        self.numTokensPerStep = _numTokensPerStep
        self.attn = nn.MultiheadAttention(embed_dim=embedDimension,
                                          num_heads=_num_heads,
                                          batch_first=True,
                                          device=self.device)
        # start almost closed so behaviour initially matches pre-attention training
        self.logit_gate = nn.Parameter(torch.tensor(-16.0, device=self.device))
        self.norm = nn.LayerNorm(embedDimension, device=self.device)
        self.stats = {}
        self.attnOutNormHist = []
        self.gatedNormHist = []
        self.finalNormHist = []
        self.gateHist = []

    @whocalled
    def forward(self, _embeds: torch.Tensor):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            if _embeds.dim() == 2:
                embeds = _embeds.unsqueeze(0)  # [1, seq, dim]
            else:
                embeds = _embeds
            attn_out, _ = self.attn(embeds, embeds, embeds, need_weights=False)
            attn_out = attn_out.squeeze(0)
            gate = torch.sigmoid(self.logit_gate)
            gated = gate * attn_out
            output = self.norm(_embeds + gated)

            self.attnOutNormHist.append(attn_out.norm().item())
            self.gatedNormHist.append(gated.norm().item())
            self.finalNormHist.append(output.norm().item())
            self.gateHist.append(gate.item())

            if len(self.attnOutNormHist) >= self.numTokensPerStep:
                self.stats = {
                    "5A_0_attnOut_norm": sum(self.attnOutNormHist) / len(self.attnOutNormHist),
                    "5A_1_gated_norm":   sum(self.gatedNormHist) / len(self.gatedNormHist),
                    "5A_x_final_norm":   sum(self.finalNormHist) / len(self.finalNormHist),
                    "5A_gateScale":      sum(self.gateHist) / len(self.gateHist),
                }
                self.attnOutNormHist = []
                self.gatedNormHist = []
                self.finalNormHist = []
                self.gateHist = []

            return output

    @whocalled
    def getAttentionStats(self):
        return self.stats
