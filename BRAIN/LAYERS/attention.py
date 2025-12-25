# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ ---
# GATED MULTI-HEAD ATTENTION LAYER // brain/LAYERS/attention.py
# v1.3

import math
import torch
import torch.nn as nn
from config import *

"""multi-head self attention with a learnable gate so existing training isn't thrown away"""
class GATED_MHA(nn.Module):
    def __init__(self, _counsellor, _num_heads: int = 16, _device = modelDevice, _embed_dim: int = None, _stat_prefix: str = "2A"):
        super().__init__()
        self.counsellor = _counsellor
        self.device = _device
        self.num_heads = _num_heads
        self.stat_prefix = _stat_prefix

        # use the provided _embed_dim, or default to embedDimension from config
        current_embed_dim = _embed_dim if _embed_dim is not None else embedDimension

        self.attn = nn.MultiheadAttention(embed_dim=current_embed_dim, num_heads=_num_heads, batch_first=True, device=self.device)
        self.logit_gate = nn.Parameter(torch.tensor(-8.0, device=self.device))
        self.norm = nn.LayerNorm(current_embed_dim, device=self.device)
        self.stats = {}
        self._scale = math.sqrt(current_embed_dim)

        # Pre-calculate the causal mask one time
        # We use maxPosLen from your embed layer (2048) as a safe max
        max_len = 2048 
        mask = torch.triu(torch.ones((max_len, max_len), dtype=torch.bool, device=self.device), diagonal=1,)
        # Register it as a non-trainable buffer
        self.register_buffer("causal_mask_buffer", mask)

    @whocalled
    def forward(self, _embeds: torch.Tensor):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            original_dim = _embeds.dim()
            if original_dim == 1: embeds = _embeds.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
            elif original_dim == 2: embeds = _embeds.unsqueeze(0)  # [1, seq, dim]
            else: embeds = _embeds  # already [batch, seq, dim]
            seq_len = embeds.size(1)
            # Instead of creating a new mask, just slice your pre-built one!
            causal_mask = self.causal_mask_buffer[0:seq_len, 0:seq_len]
            attn_out, _ = self.attn(embeds, embeds, embeds, need_weights=False, attn_mask=causal_mask,)
            # embeddings are scaled by sqrt(embedDimension) in the embed layer, so normalising with the same scale (and sequence length)
            attn_out = attn_out / (self._scale * math.sqrt(seq_len))
            if original_dim <= 2:
                attn_out = attn_out.squeeze(0)
                if original_dim == 1: attn_out = attn_out.squeeze(0)  # [dim]
            gate = torch.sigmoid(self.logit_gate)
            gated = gate * attn_out
            out = self.norm(_embeds + gated)

            try:
                attn_norm = attn_out.norm().item()
                gated_norm = gated.norm().item()
                final_norm = out.norm().item()
                attn_mean = attn_out.mean().item()
                gated_mean = gated.mean().item()
                final_mean = out.mean().item()
                gate_item = gate.item()
                self.stats = {
                    f"{self.stat_prefix}_0_attnOut_norm": attn_norm,
                    f"{self.stat_prefix}_0_attnOut_mean": attn_mean,
                    f"{self.stat_prefix}_1_gated_norm": gated_norm,
                    f"{self.stat_prefix}_1_gated_mean": gated_mean,
                    f"{self.stat_prefix}_x_final_norm": final_norm,
                    f"{self.stat_prefix}_x_final_mean": final_mean,
                    f"{self.stat_prefix}_gateScale": gate_item,
                }
            except (RuntimeError, ValueError):
                self.stats = {
                    f"{self.stat_prefix}_0_attnOut_norm": 0.0,
                    f"{self.stat_prefix}_0_attnOut_mean": 0.0,
                    f"{self.stat_prefix}_1_gated_norm": 0.0,
                    f"{self.stat_prefix}_1_gated_mean": 0.0,
                    f"{self.stat_prefix}_x_final_norm": 0.0,
                    f"{self.stat_prefix}_x_final_mean": 0.0,
                    f"{self.stat_prefix}_gateScale": 0.5,
                }

            return out
        
    @whocalled
    def getAttentionStats(self):
        with self.counsellor.infodump("getAttentionStats") as ʕっʘ‿ʘʔっ:
            return self.stats
