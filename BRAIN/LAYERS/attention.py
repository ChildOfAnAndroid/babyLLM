# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ ---
# GATED MULTI-HEAD ATTENTION LAYER // brain/LAYERS/attention.py

import torch
import torch.nn as nn
from config import *

"""multi-head self attention with a learnable gate so existing training isn't thrown away"""
class GATED_MHA(nn.Module):
    def __init__(self, _counsellor, _num_heads: int = 16, _device = modelDevice):
        super().__init__()
        self.counsellor = _counsellor
        self.device = _device
        self.num_heads = _num_heads
        self.attn = nn.MultiheadAttention(embed_dim=embedDimension,
                                          num_heads=_num_heads,
                                          batch_first=True,
                                          device=self.device)
        # start almost closed so behaviour initially matches pre-attention training
        self.logit_gate = nn.Parameter(torch.tensor(-8.0, device=self.device))
        self.norm = nn.LayerNorm(embedDimension, device=self.device)
        self.stats = {}

    @whocalled
    def forward(self, _embeds: torch.Tensor):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            if _embeds.dim() == 2:
                embeds = _embeds.unsqueeze(0)  # [1, seq, dim]
            else:
                embeds = _embeds
            attn_out, _ = self.attn(
                embeds,
                embeds,
                embeds,
                need_weights=False,
                is_causal=True,
            )
            attn_out = attn_out.squeeze(0)
            gate = torch.sigmoid(self.logit_gate)
            gated = gate * attn_out
            out = self.norm(_embeds + gated)

            # collect stats
            self.stats = {
                "2A_0_attnOut_norm": attn_out.norm().item(),
                "2A_0_attnOut_mean": attn_out.mean().item(),
                "2A_1_gated_norm": gated.norm().item(),
                "2A_1_gated_mean": gated.mean().item(),
                "2A_x_final_norm": out.norm().item(),
                "2A_x_final_mean": out.mean().item(),
                "2A_gateScale": gate.item(),
            }

            return out

    @whocalled
    def getAttnStats(self):
        with self.counsellor.infodump("getAttnStats") as ʕっʘ‿ʘʔっ:
            return self.stats