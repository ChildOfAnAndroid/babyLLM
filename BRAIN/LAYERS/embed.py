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

        self.pixelEmbed = nn.Linear(3, embedDimension, device = self.device)

        self.maxPosLen = 2048
        self.posEmbedding = nn.Embedding(self.maxPosLen, embedDimension, device = self.device)

    """looks up and returns the embedding vector for a specifc token index"""
    @whocalled
    def forward(self, _tokenIndex = None, _pixel = None):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            if not skipPixels and (_pixel is not None):
                if debugPrints: ʕっʘ‿ʘʔっ("E0_pixelInjected")
                if _pixel.dim() == 1:  # [3]
                    if debugPrints: ʕっʘ‿ʘʔっ("pixel.dim == 1")
                    self.embedVector = self.pixelEmbed(_pixel.unsqueeze(0)).squeeze(0)  # [embedDimension]
                elif _pixel.dim() == 2:  # [seq_len, 3]
                    if debugPrints: ʕっʘ‿ʘʔっ("pixel.dim == 2")
                    self.embedVector = self.pixelEmbed(_pixel)  # [seq_len, embedDimension]
                else:
                    raise ValueError(f"Pixel input has wrong shape: {_pixel.shape}")
            else:
                if debugPrints: ʕっʘ‿ʘʔっ("E0_embedVector") # <- vocab???? base token indexes seem to come in here so... from tutor??
                self.embedVector = self.e_weights[_tokenIndex] 
            if debugPrints: ʕっʘ‿ʘʔっ("E1_embedNormed") # <- E1
            self.embedNormed = self.embedNorm(self.embedVector)
            if debugPrints: ʕっʘ‿ʘʔっ("Ex_embedFinal") # <- E2
            self.embedFinal = (self.embedVector * self.weightsScale) + (self.embedNormed * self.normScale) 
            with torch.no_grad():
                self.weightsScale.data.clamp_(-10, 10)
                self.normScale.data.clamp_(-10, 10)
            return self.embedFinal # E3 -> N??
    
    @whocalled
    def getEmbedStats(self):
        with self.counsellor.infodump("getEmbedStats") as ʕっʘ‿ʘʔっ:
            if debugPrints: ʕっʘ‿ʘʔっ("with torch.no_grad")
            with torch.no_grad():
                self.stats = {}
                if debugPrints: ʕっʘ‿ʘʔっ("embedNorms = torch.norm(self.e_weights, dim = 1)")
                #embedNorms = torch.norm(self.e_weights, dim = 1)
                if debugPrints: ʕっʘ‿ʘʔっ("embedNorms Stats")
                #self.stats["1E_weightNormMean"] = embedNorms.mean().item()
                #self.stats["1E_weightNormStd"] = embedNorms.std().item()
                #self.stats["1E_weightNormMax"] = embedNorms.max().item()

                if debugPrints: ʕっʘ‿ʘʔっ("vectorNorm stats")
                self.stats["1E_0_vector_norm"] = self.embedVector.norm().item()
                #self.stats["1E_1_normed_norm"] = self.embedNormed.norm().item()
                self.stats["1E_0_vector_mean"] = self.embedVector.mean().item()
                #self.stats["1E_1_normed_mean"] = self.embedNormed.mean().item()
                self.stats["1E_x_final_norm"] = self.embedFinal.norm().item()
                self.stats["1E_x_final_mean"] = self.embedFinal.mean().item()
                ###self.stats["1E_1_pixelEmbed_norm"] = self.pixelEmbed.norm().item()###
                ###self.stats["1E_1_pixelEmbed_mean"] = self.pixelEmbed.weight.mean().item()###
                #self.stats["1E_0_vector_scale"] = self.weightsScale.norm().item()
                #self.stats["1E_1_normed_scale"] = self.normScale.norm().item()
                self.stats["1E_1_posEmbWeight_norm"] = self.posEmbedding.weight.norm().item()
                self.stats["1E_1_posEmbWeight_mean"] = self.posEmbedding.weight.mean().item()

                #dimMean = self.e_weights.detach().clone().mean(dim = 0)
                #self.stats["1E_dimMean"] = dimMean
                #dimSparsity = (dimMean.abs() < 1e-4).float().mean().item()
                #self.stats["1E_dimSparsity"] = dimSparsity
 
                # Drift since last save
                #drift = torch.norm(self.e_weights - self.lastSavedEmbeds).item()
                #self.stats["1E_drift"] = drift
                #self.lastSavedEmbeds = self.e_weights.detach().clone()

                return self.stats
    
    @whocalled
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