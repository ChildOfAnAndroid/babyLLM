# CHARIS CAT 2025
# BABYLLM - embed.py

import torch
import torch.nn as nn
from config import *
from SCHOOL.staffroom.counsellor import *

"""creates an embedding layer for each word in the vocabulary"""
class EMBED(nn.Module):
    def __init__(self):
        super().__init__()
        """creates the embedding weights matrix with random numbers initially"""
        self.e_weights = nn.Parameter(torch.randn(vocabSize, embedDimension, device = modelDevice))
        self.counsellor = COUNSELLOR("EMBED", debug=debugPrints, durations=durationLogging)

    """looks up and returns the embedding vector for a specifc token index"""
    def forward(self, tokenIndex):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            ʕっʘ‿ʘʔっ("tokenIndex.to(self.e_weights.device)")
            #tokenIndex = tokenIndex.to(self.e_weights.device)
            ʕっʘ‿ʘʔっ("self.e_weights[tokenIndex]")
            embedVector = self.e_weights[tokenIndex] 
            return embedVector 
    
    def getEmbeddingStats(self):
        with self.counsellor.infodump("getEmbeddingStats") as ʕっʘ‿ʘʔっ:
            with torch.no_grad():
                stats = {}
                embedNorms = torch.norm(self.e_weights, dim=1)
                stats["embedNormMean"] = embedNorms.mean()
                stats["embedNormStd"] = embedNorms.std()
                stats["embedNormMax"] = embedNorms.max()

                dimMean = self.e_weights.mean(dim=0)
                dimSparsity = (dimMean.abs() < 1e-5).float().mean()
                stats["embedDimSparsity"] = dimSparsity

                # Drift since last save
                drift = torch.norm(self.e_weights - self.lastSavedEmbeds)
                stats["embeddingDrift"] = drift
                self.lastSavedEmbeds = self.e_weights

                return stats
        
    def cosineSimilarity(self, idx1, idx2):
        e1 = self.e_weights[idx1]
        e2 = self.e_weights[idx2]
        return torch.nn.functional.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0))

    
if __name__ == "__main__":
    TESTtokenIndex = 500

    # 32 (embedDimension) x 2000 (vocab) = 64,000 in embed layer
    embed = EMBED(vocabSize, embedDimension) 
    embedVector = embed.forward(TESTtokenIndex)

    print(f"--- EMBEDDING LAYER TESTING START ---")
    print(f"embedding layer weights shape: {embed.weights.shape}") # Check shape of weight matrix
    print(f"embedding vector for token index {TESTtokenIndex}:")
    print(embedVector)
    print(f"embedding vector shape: {embedVector.shape}")
    print(f"--- EMBEDDING LAYER TESTING COMPLETE ---")