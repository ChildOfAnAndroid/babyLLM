# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ ---
# GATED MULTI-HEAD ATTENTION LAYER // brain/LAYERS/attention.py
# v1.5

import math
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
            original_dim = _embeds.dim()
            if original_dim == 1:
                embeds = _embeds.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
            elif original_dim == 2:
                embeds = _embeds.unsqueeze(0)  # [1, seq, dim]
            else:
                embeds = _embeds  # already [batch, seq, dim]
            seq_len = embeds.size(1)
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=embeds.device),
                diagonal=1,
            )
            attn_out, _ = self.attn(
                embeds,
                embeds,
                embeds,
                need_weights=False,
                attn_mask=causal_mask,
            )
            # embeddings are scaled by sqrt(embedDimension) in the embed layer
            # which causes the raw attention output to grow very large.  Counter
            # this by normalising with the same scale (and sequence length) so
            # the attention statistics remain comparable to other layers.
            attn_out = attn_out / math.sqrt(embedDimension * seq_len)
            if original_dim <= 2:
                attn_out = attn_out.squeeze(0)
                if original_dim == 1:
                    attn_out = attn_out.squeeze(0)  # [dim]
            gate = torch.sigmoid(self.logit_gate)
            gated = gate * attn_out
            out = self.norm(_embeds + gated)

            # collect stats in the same format as other layers so values are
            # directly comparable
            try:
                attn_norm = attn_out.norm().item()
                gated_norm = gated.norm().item()
                final_norm = out.norm().item()
                attn_mean = attn_out.mean().item()
                gated_mean = gated.mean().item()
                final_mean = out.mean().item()
                gate_item = gate.item()
                self.stats = {
                    "2A_0_attnOut_norm": attn_norm,
                    "2A_0_attnOut_mean": attn_mean,
                    "2A_1_gated_norm": gated_norm,
                    "2A_1_gated_mean": gated_mean,
                    "2A_x_final_norm": final_norm,
                    "2A_x_final_mean": final_mean,
                    "2A_gateScale": gate_item,
                }
            except Exception:
                # If tensor operations hang or fail, use safe defaults
                self.stats = {
                    "2A_0_attnOut_norm": 0.0,
                    "2A_0_attnOut_mean": 0.0,
                    "2A_1_gated_norm": 0.0,
                    "2A_1_gated_mean": 0.0,
                    "2A_x_final_norm": 0.0,
                    "2A_x_final_mean": 0.0,
                    "2A_gateScale": 0.5,
                }

            return out

    @whocalled
    def getAttentionStats(self):
        with self.counsellor.infodump("getAttentionStats") as ʕっʘ‿ʘʔっ:
            return self.stats
