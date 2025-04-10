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
        self.weights = nn.Parameter(torch.randn(vocabSize, embedDimension, device = modelDevice))
        self.weights.data *= 0.01 # makes the layer numbers smaller for more stable training

    """looks up and returns the embedding vector for a specifc token index"""
    def forward(self, tokenIndex):
        tokenIndex = tokenIndex.to(self.weights.device)
        embedVector = self.weights[tokenIndex] 
        return embedVector 
    
    def getEmbeddingStats(self):
        with torch.no_grad():
            stats = {}
            embedNorms = torch.norm(self.weights, dim=1)
            #stats["embedNormMean"] = embedNorms.mean().item()
            #stats["embedNormStd"] = embedNorms.std().item()
            stats["embedNormMax"] = embedNorms.max().item()

            #dimMean = self.weights.mean(dim=0)
            #dimSparsity = (dimMean.abs() < 1e-5).float().mean().item()
            #stats["embedDimSparsity"] = dimSparsity

            # Drift since last save
            #drift = torch.norm(self.weights - self.lastSavedEmbeds).item()
            #stats["embeddingDrift"] = drift
            #self.lastSavedEmbeds = self.weights.clone().detach()

            return stats
        
    #def cosineSimilarity(self, idx1, idx2):
    #    e1 = self.weights[idx1]
    #    e2 = self.weights[idx2]
    #    return torch.nn.functional.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()

    
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