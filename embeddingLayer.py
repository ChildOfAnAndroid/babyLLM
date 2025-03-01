# CHARIS CAT 2025

import torch
from config import *

class EMBEDLAYER:
    def __init__(self, vocabSize, embedDimension):
        self.vocabSize = vocabSize
        self.embedDimension = embedDimension # keep in self for later use

        self.weights = torch.randn(vocabSize, embedDimension)
        return

    def forward(self, tokenIndex):
        # load current weights from the embed vector
        embedVector = self.weights[tokenIndex] 

        return embedVector 
    
if __name__ == "__main__":
    TESTtokenIndex = 500

    # 32 (embedDimension) x 2000 (vocab) = 64,000 in embed layer
    embedLayer = EMBEDLAYER(vocabSize, embedDimension) 
    embedVector = embedLayer.forward(TESTtokenIndex)

    print(f"Embedding Layer Weights Shape: {embedLayer.weights.shape}") # Check shape of weight matrix
    print(f"Embedding Vector for token index {TESTtokenIndex}:")
    print(embedVector) # Print embedding vector
    print(f"Embedding Vector Shape: {embedVector.shape}")