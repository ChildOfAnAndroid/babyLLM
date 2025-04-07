# CHARIS CAT 2025
# BABYLLM - embedLayer.py

import torch
import torch.nn as nn
from config import *

"""creates an embedding layer for each word in the vocabulary"""
class EMBEDLAYER(nn.Module):
    def __init__(self, vocabSize, embedDimension):
        super().__init__()
        self.vocabSize = vocabSize
        self.embedDimension = embedDimension
        """creates the embedding weights matrix with random numbers initially"""
        self.weights = nn.Parameter(torch.randn(vocabSize, embedDimension))
        self.weights.data *= 0.01 # makes the layer numbers smaller for more stable training

    def forward(self, tokenIndex):
        """looks up and returns the embedding vector for a specifc token index"""
        embedVector = self.weights[tokenIndex] 
        return embedVector 
    
if __name__ == "__main__":
    TESTtokenIndex = 500

    # 32 (embedDimension) x 2000 (vocab) = 64,000 in embed layer
    embedLayer = EMBEDLAYER(vocabSize, embedDimension) 
    embedVector = embedLayer.forward(TESTtokenIndex)

    print(f"--- EMBEDDING LAYER TESTING START ---")
    print(f"Embedding Layer Weights Shape: {embedLayer.weights.shape}") # Check shape of weight matrix
    print(f"Embedding Vector for token index {TESTtokenIndex}:")
    print(embedVector)
    print(f"Embedding Vector Shape: {embedVector.shape}")
    print(f"--- EMBEDDING LAYER TESTING COMPLETE ---")