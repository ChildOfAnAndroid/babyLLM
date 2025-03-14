import torch
import torch.nn as nn
from config import *

#class MULTIWINDOWLAYER(nn.Module):






"""This layer captures multiple windows of input embedding information, and then combines these into a single context vector."""
class MULTIWINDOWLAYER(nn.Module):

    def __init__(self, embedDimension, windowSizes = [window1, window2, window3]):
        super().__init__()
        self.embedDimension = embedDimension
        self.windowSizes = windowSizes

    """FORWARD PASS"""
    def forward(self, inputEmbeds):
        """processes the input embeddings through the sliding windows"""
        windowContextVectors = []
        #print(f"Debug MULTIWINDOWLAYER.forward: Input inputEmbedsList length: {len(inputEmbedsList) if isinstance(inputEmbedsList, list) else 'Not a list'}")
        for windowSize in self.windowSizes:
            windowEmbeds = self.createWindows(inputEmbeds, windowSize)
            #print(f"Debug MULTIWINDOWLAYER.forward: After createWindows (windowSize={windowSize}), windowEmbeds length: {len(windowEmbeds)}")
            windowPooledVectors = self.processWindows(windowEmbeds)
            #print(f"Debug MULTIWINDOWLAYER.forward: After processWindows (windowSize={windowSize}), windowPooledVectors is now a LIST")
            """if it isnt a tensor, this extends it to create a tensor"""
            if not (isinstance(windowPooledVectors, torch.Tensor) and windowPooledVectors.numel() == 0):
                windowContextVectors.extend(windowPooledVectors)
            else:
                #print(f"Debug MULTIWINDOWLAYER.forward: windowPooledVectors is EMPTY TENSOR - NOT APPENDING to windowContextVectors")
                pass
        """combines the output of the windows into a single vector"""
        combinedContextVector = self.combineWindowVectors(windowContextVectors)
        #print(f"Debug MULTIWINDOWLAYER.forward: After combineWindowVectors, combinedContextVector shape: {combinedContextVector.shape}")
        return combinedContextVector
    
    """CREATE MULTIPLE SLIDING WINDOWS"""
    def createWindows(self, inputEmbeds, windowSize):
        """for each token in the input sequence, this extracts a window of embeddings centered around that token, of a specified size"""
        windowEmbeds = []
        seqLen = len(inputEmbeds)
        """handles empty lists"""
        if not inputEmbeds:
            print("Debug createWindows: inputEmbedsList is EMPTY! Returning empty list.")
            return []
        """this iterates through token positions, and ensures that the loop always runs at least once"""
        for i in range(max(1, seqLen)):
            """find start and handle negative start"""
            startIndex = max(0, i - windowSize // 2)
            endIndex = min(seqLen, i + windowSize - windowSize // 2)
            """extract window of embeddings (might be shorter than windowSize if near edges)"""
            window = list(inputEmbeds[startIndex:endIndex]) # slice as a list
            #print(f"Debug createWindows (i={i}): Type of window: {type(window)}")
            """DEBUG PRINTS"""
            if isinstance(window, list):
                if window:
                    #print(f"Debug createWindows (i={i}): Shape of first element in window: {window[0].shape}")
                    pass
                else:
                    #print(f"Debug createWindows (i={i}): window is an EMPTY LIST!")
                    pass
            elif isinstance(window, torch.Tensor):
                #print(f"Debug createWindows (i={i}): Shape of window (Tensor): {window.shape}")
                pass
            else:
                #print(f"Debug createWindows (i={i}): window is of UNEXPECTED TYPE: {type(window)}")
                pass
            """ZERO PADDING - ensures all windows are the correct size, even at the edges of the data"""
            paddingNeededStart = max(0, windowSize // 2 - i)
            paddingNeededEnd = max(0, (i + windowSize - windowSize // 2) - seqLen)
            paddingStart = [torch.zeros(self.embedDimension) for _ in range(paddingNeededStart)] # Create zero padding vectors at start
            paddingEnd = [torch.zeros(self.embedDimension) for _ in range(paddingNeededEnd)] # Create zero padding vectors at end
            """DEBUG PRINTS"""
            #print(f"Debug createWindows (i={i}): Type of paddingStart: {type(paddingStart)}") # DEBUG - TYPE OF paddingStart
            #print(f"Debug createWindows (i={i}): Type of window: {type(window)}") # DEBUG - TYPE OF window
            #print(f"Debug createWindows (i={i}): Type of paddingEnd: {type(paddingEnd)}") # DEBUG - TYPE OF paddingEnd
            paddedWindow = paddingStart + window + paddingEnd # <--- SUSPECT LINE - + OPERATOR AND POTENTIAL NoneType!
            windowEmbeds.append(torch.stack(paddedWindow))
        """this returns a list of tensors, where each tensor is a padded window for each token in the input sequence"""
        """each window tensor will have a shape of (windowSize, embedDimension)"""
        return windowEmbeds

    """MEAN POOLING EACH WINDOW SEPARATELY"""
    def processWindows(self, windowEmbeds):
        """this mean pools each window tensor in the 'window embeds' list along dim 0, reducing each to a pooled vector showing the average"""
        windowPooledVectors = []
        #print(f"Debug processWindows: windowEmbeds length: {len(windowEmbeds)}")
        # handle cases where the window is empty/no token
        if not windowEmbeds:
            #print("Debug processWindows: windowEmbeds is EMPTY! Returning empty tensor.") # DEBUG PRINT
            return torch.empty((0, self.embedDimension))
        for window in windowEmbeds:
            # mean pool embeddings for each window
            pooledVector = torch.mean(window, dim = 0)
            windowPooledVectors.append(pooledVector)
        #print(f"Debug processWindows: windowPooledVectors length before stack: {len(windowPooledVectors)}")
        """returns a list of pooled context vectors, where each tensor has shape (embedDimension)"""
        return windowPooledVectors

    """COMBINING THE WINDOWS INTO ONE LAYER"""
    def combineWindowVectors(self, windowContextVectors):
        #print(f"Debug MULTIWINDOWLAYER.combineWindowVectors: Input windowContextVectors length: {len(windowContextVectors)}")
        if windowContextVectors:
            #print(f"Debug MULTIWINDOWLAYER.combineWindowVectors: Shape of first element in windowContextVectors: {windowContextVectors[0].shape}")
            pass
        """this takes a list of window context vectors, and concatenates them into a single vector"""
        concatenatedVectors = torch.cat(windowContextVectors, dim = 0)
        #print(f"Debug MULTIWINDOWLAYER.combineWindowVectors: After torch.cat, concatenatedVectors shape: {concatenatedVectors.shape}") 
        """this creates a 2D linear layer, reducing dimensionality for future calcs"""
        if not hasattr(self, 'combinationLayer'):
            self.combinationLayer = nn.Linear(concatenatedVectors.shape[0], self.embedDimension)
        combinedVector = self.combinationLayer(concatenatedVectors)
        combinedVector = combinedVector.unsqueeze(0) # Ensure output is 2D
        #print(f"Debug MULTIWINDOWLAYER.combineWindowVectors: After combinationLayer, combinedVector shape: {combinedVector.shape}")
        """returns a single combined context vector of shape (1, embedDimension)"""
        return combinedVector
