# CHARIS CAT 2025
# BABYLLM - embedLayer.py

import torch
import torch.nn as nn
from VER1_config import *

"""creates an embedding layer for each word in the vocabulary"""
class EMBEDLAYER(nn.Module):
    def __init__(self, vocabSize, embedDimension):
        super().__init__()
        self.vocabSize = vocabSize
        self.embedDimension = embedDimension
        """creates the embedding weights matrix with random numbers initially"""
        self.weights = nn.Parameter(torch.randn(vocabSize, embedDimension, device = modelDevice))
        self.weights.data *= 0.01 # makes the layer numbers smaller for more stable training

    """looks up and returns the embedding vector for a specifc token index"""
    def forward(self, tokenIndex):
        tokenIndex = tokenIndex.to(self.weights.device)
        embedVector = self.weights[tokenIndex] 
        return embedVector 
    
if __name__ == "__main__":
    TESTtokenIndex = 500

    # 32 (embedDimension) x 2000 (vocab) = 64,000 in embed layer
    embedLayer = EMBEDLAYER(vocabSize, embedDimension) 
    embedVector = embedLayer.forward(TESTtokenIndex)

    print(f"--- EMBEDDING LAYER TESTING START ---")
    print(f"embedding layer weights shape: {embedLayer.weights.shape}") # Check shape of weight matrix
    print(f"embedding vector for token index {TESTtokenIndex}:")
    print(embedVector)
    print(f"embedding vector shape: {embedVector.shape}")
    print(f"--- EMBEDDING LAYER TESTING COMPLETE ---")