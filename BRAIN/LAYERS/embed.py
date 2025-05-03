# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# EMBEDDING LAYER // BRAIN/LAYERS/embed.py

import torch
import torch.nn as nn
from config import *

"""creates an embedding layer for each word in the vocabulary"""
class EMBED(nn.Module):
    def __init__(self, _counsellor, _device = modelDevice):
        super().__init__()
        self.counsellor = _counsellor
        self.device = _device

        """creates the embedding weights matrix with random numbers initially"""
        self.e_weights = nn.Parameter(torch.randn(vocabSize, embedDimension, device = self.device)) # [2000,]
        self.embedNorm = nn.LayerNorm(embedDimension, device = self.device)
        self.weightsScale = nn.Parameter(torch.tensor(0.5)) 
        self.normScale = nn.Parameter(torch.tensor(0.5)) 
        self.lastSavedEmbeds = self.e_weights.detach().clone() # THIS IS INITIALISED ONCE, FOR STATS, DOES NOT BREAK GRAPH CONFIRMED!!

    """looks up and returns the embedding vector for a specifc token index"""
    def forward(self, _tokenIndex):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            ʕっʘ‿ʘʔっ("self.e_weights[tokenIndex]")
            self.embedVector = self.e_weights[_tokenIndex] 
            self.embedNormed = self.embedNorm(self.embedVector)
            self.finalEmbed = (self.embedVector * (min(self.weightsScale, 1.1)) + (self.embedNormed * self.normScale) 
            return self.finalEmbed 
    
    def getEmbedStats(self):
        with self.counsellor.infodump("getEmbedStats") as ʕっʘ‿ʘʔっ:
            with torch.no_grad():
                stats = {}
                embedNorms = torch.norm(self.e_weights, dim = 1)
                stats["embedNormMean"] = embedNorms.mean()
                stats["embedNormStd"] = embedNorms.std()
                stats["embedNormMax"] = embedNorms.max()

                stats["embedVector"] = self.embedVector.norm().item()
                stats["embedNormed"] = self.embedNormed.norm().item()
                stats["embedFinal"] = self.finalEmbed.norm().item()
                stats["embedVectorScale"] = self.weightsScale.norm().item()
                stats["embedNormedScale"] = self.normScale.norm().item()

                dimMean = self.e_weights.mean(dim = 0)
                stats["embedDimensionMean"] = dimMean
                dimSparsity = (dimMean.abs() < 1e-4).float().mean()
                stats["embedDimensionSparsity"] = dimSparsity

                # Drift since last save
                drift = torch.norm(self.e_weights - self.lastSavedEmbeds)
                stats["embeddingDrift"] = drift
                self.lastSavedEmbeds = self.e_weights.detach().clone()

                return stats
        
    def cosineSimilarity(self, _idx1, _idx2):
        e1 = self.e_weights[_idx1]
        e2 = self.e_weights[_idx2]
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