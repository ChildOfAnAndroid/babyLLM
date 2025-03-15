from config import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuron import NEURON

class TINYATTENTIONLAYER(nn.Module):
    def __init__(self, embedDimension = embedDimension, numHeads = numHeads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embedDimension, num_heads=numHeads, batch_first=True)
        self.norm = nn.LayerNorm(embedDimension)
        self.ffn = nn.Sequential(
            nn.Linear(embedDimension, embedDimension * 2),
            nn.ReLU(),
            nn.Linear(embedDimension * 2, embedDimension)
        )

    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        attnOutput = self.norm(attended + x)
        attnOutput = self.ffn(attnOutput) + attnOutput

        return attnOutput  # Remove batch dim on outpu