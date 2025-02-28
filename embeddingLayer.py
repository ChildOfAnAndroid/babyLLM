# CHARIS CAT 2025

import torch
from config import *

class EMBEDLAYER:
    def __init__(self, vocabSize, embedData):
        self.vocabSize = vocabSize
        self.embedData = embedData

        self.weights = torch.randn(vocabSize, embedData)

    def forward(self, tokenIndex):
        embedVector = self.weights[tokenIndex]

        return embedVector
    
if __name__ == "__main__":
    embedLayer = EMBEDLAYER(vocabSize, embedDimension)
    TESTtokenIndex = 500 
    embedVector = embedLayer.forward(TESTtokenIndex)

    print(f"Embedding Layer Weights Shape: {embedLayer.weights.shape}") # Check shape of weight matrix
    print(f"Embedding Vector for token index {TESTtokenIndex}:")
    print(embedVector) # Print embedding vector
    print(f"Embedding Vector Shape: {embedVector.shape}")