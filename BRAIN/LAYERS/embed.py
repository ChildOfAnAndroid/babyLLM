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
        self.stats = {}

        """creates the embedding weights matrix with random numbers initially"""
        self.e_weights = nn.Parameter(torch.randn(vocabSize, embedDimension, device = self.device)) # [2000,]
        self.embedNorm = nn.LayerNorm(embedDimension, device = self.device)
        self.weightsScale = nn.Parameter(torch.tensor(0.5)) 
        self.normScale = nn.Parameter(torch.tensor(0.5)) 
        self.lastSavedEmbeds = self.e_weights.detach().clone() # THIS IS INITIALISED ONCE, FOR STATS, DOES NOT BREAK GRAPH CONFIRMED!!

    """looks up and returns the embedding vector for a specifc token index"""
    @whocalled
    def forward(self, _tokenIndex):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            ʕっʘ‿ʘʔっ("E0_embedVector") # <- vocab???? base token indexes seem to come in here so... from tutor??
            self.embedVector = self.e_weights[_tokenIndex] 
            ʕっʘ‿ʘʔっ("E1_embedNormed") # <- E1
            self.embedNormed = self.embedNorm(self.embedVector)
            ʕっʘ‿ʘʔっ("Ex_embedFinal") # <- E2
            self.embedFinal = (self.embedVector * self.weightsScale) + (self.embedNormed * self.normScale) 
            return self.embedFinal # E3 -> N??
    
    def getEmbedStats(self):
        with self.counsellor.infodump("getEmbedStats") as ʕっʘ‿ʘʔっ:
            with torch.no_grad():
                self.stats = {}
                embedNorms = torch.norm(self.e_weights, dim = 1)
                self.stats["embedNormMean"] = embedNorms.mean()
                self.stats["embedNormStd"] = embedNorms.std()
                self.stats["embedNormMax"] = embedNorms.max()

                self.stats["1E_0_embedVector_norm"] = self.embedVector.norm().item()
                self.stats["1E_1_embedNormed_norm"] = self.embedNormed.norm().item()
                self.stats["1E_x_embedFinal_norm"] = self.embedFinal.norm().item()
                self.stats["1E_0_embedVector_scale"] = self.weightsScale.norm().item()
                self.stats["1E_1_embedNormed_scale"] = self.normScale.norm().item()

                dimMean = self.e_weights.mean(dim = 0)
                self.stats["embedDimensionMean"] = dimMean
                dimSparsity = (dimMean.abs() < 1e-4).float().mean()
                self.stats["embedDimensionSparsity"] = dimSparsity

                # Drift since last save
                drift = torch.norm(self.e_weights - self.lastSavedEmbeds)
                self.stats["embeddingDrift"] = drift
                self.lastSavedEmbeds = self.e_weights.detach().clone()

                return self.stats
        
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