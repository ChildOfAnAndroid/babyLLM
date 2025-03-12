import torch
import torch.nn as nn
from config import *

class MULTIWINDOWLAYER(nn.Module):

    def __init__(self, embedDimension, windowSizes = [window1, window2, window3]):
        super().__init__()
        self.embedDimension = embedDimension
        self.windowSizes = windowSizes

    # FORWARD PASS
    def forward(self, inputEmbedsList):
        windowContextVectors = []
        print(f"Debug MULTIWINDOWLAYER.forward: Input inputEmbedsList length: {len(inputEmbedsList) if isinstance(inputEmbedsList, list) else 'Not a list'}")
        for windowSize in self.windowSizes:
            windowEmbeds = self.createWindows(inputEmbedsList, windowSize)
            print(f"Debug MULTIWINDOWLAYER.forward: After createWindows (windowSize={windowSize}), windowEmbeds length: {len(windowEmbeds)}")
            windowPooledVectors = self.processWindows(windowEmbeds)
            print(f"Debug MULTIWINDOWLAYER.forward: After processWindows (windowSize={windowSize}), windowPooledVectors is now a LIST")
            if not (isinstance(windowPooledVectors, torch.Tensor) and windowPooledVectors.numel() == 0):
                windowContextVectors.extend(windowPooledVectors)
            else:
                print(f"Debug MULTIWINDOWLAYER.forward: windowPooledVectors is EMPTY TENSOR - NOT APPENDING to windowContextVectors")
        combinedContextVector = self.combineWindowVectors(windowContextVectors)
        print(f"Debug MULTIWINDOWLAYER.forward: After combineWindowVectors, combinedContextVector shape: {combinedContextVector.shape}")
        return combinedContextVector
    
    # CREATE MULTIPLE SLIDING WINDOWS
    def createWindows(self, inputEmbedsList, windowSize):
        windowEmbeds = []
        seqLen = len(inputEmbedsList)
        # handles empty lists
        if not inputEmbedsList:
            print("Debug createWindows: inputEmbedsList is EMPTY! Returning empty list.")
            return []
        # always runs at least once if sequence isnt empty
        for i in range(max(1, seqLen)):
            # find start and handle negative start
            startIndex = max(0, i - windowSize // 2)
            endIndex = min(seqLen, i + windowSize - windowSize // 2)
            # extract window (might be shorter than windowSize)
            window = list(inputEmbedsList[startIndex:endIndex])
            print(f"Debug createWindows (i={i}): Type of window: {type(window)}")
            if isinstance(window, list):
                if window:
                    print(f"Debug createWindows (i={i}): Shape of first element in window: {window[0].shape}")
                else:
                    print(f"Debug createWindows (i={i}): window is an EMPTY LIST!")
            elif isinstance(window, torch.Tensor):
                print(f"Debug createWindows (i={i}): Shape of window (Tensor): {window.shape}")
            else:
                print(f"Debug createWindows (i={i}): window is of UNEXPECTED TYPE: {type(window)}")
            # ZERO PADDING
            paddingNeededStart = max(0, windowSize // 2 - i)
            paddingNeededEnd = max(0, (i + windowSize - windowSize // 2) - seqLen)
            paddingStart = [torch.zeros(self.embedDimension) for _ in range(paddingNeededStart)] # Create zero padding vectors at start
            paddingEnd = [torch.zeros(self.embedDimension) for _ in range(paddingNeededEnd)] # Create zero padding vectors at end
            # --- DEBUG PRINTS - INSPECT PADDING LISTS BEFORE CONCATENATION ---
            print(f"Debug createWindows (i={i}): Type of paddingStart: {type(paddingStart)}") # DEBUG - TYPE OF paddingStart
            print(f"Debug createWindows (i={i}): Type of window: {type(window)}") # DEBUG - TYPE OF window
            print(f"Debug createWindows (i={i}): Type of paddingEnd: {type(paddingEnd)}") # DEBUG - TYPE OF paddingEnd
            paddedWindow = paddingStart + window + paddingEnd # <--- SUSPECT LINE - + OPERATOR AND POTENTIAL NoneType!
            windowEmbeds.append(torch.stack(paddedWindow))
        return windowEmbeds

    # MEAN POOLING EACH WINDOW SEPARATELY
    def processWindows(self, windowEmbeds):
        windowPooledVectors = []
        print(f"Debug processWindows: windowEmbeds length: {len(windowEmbeds)}")
        # handle cases where the window is empty/no token
        if not windowEmbeds:
            print("Debug processWindows: windowEmbeds is EMPTY! Returning empty tensor.") # DEBUG PRINT
            return torch.empty((0, self.embedDimension))
        for window in windowEmbeds:
            # mean pool embeddings for each window
            pooledVector = torch.mean(window, dim = 0)
            windowPooledVectors.append(pooledVector)
        print(f"Debug processWindows: windowPooledVectors length before stack: {len(windowPooledVectors)}")
        return windowPooledVectors

    # COMBINING THE WINDOWS INTO ONE LAYER
    def combineWindowVectors(self, windowContextVectors):
        print(f"Debug MULTIWINDOWLAYER.combineWindowVectors: Input windowContextVectors length: {len(windowContextVectors)}") # ADDED
        if windowContextVectors:
            print(f"Debug MULTIWINDOWLAYER.combineWindowVectors: Shape of first element in windowContextVectors: {windowContextVectors[0].shape}") # ADDED
        concatenatedVectors = torch.cat(windowContextVectors, dim = 0)
        print(f"Debug MULTIWINDOWLAYER.combineWindowVectors: After torch.cat, concatenatedVectors shape: {concatenatedVectors.shape}") # ADDED
        # this linear layer reduces dimensionality for future calcs
        if not hasattr(self, 'combinationLayer'):
            self.combinationLayer = nn.Linear(concatenatedVectors.shape[0], self.embedDimension)
        combinedVector = self.combinationLayer(concatenatedVectors)
        combinedVector = combinedVector.unsqueeze(0) # ADD THIS LINE - Ensure output is 2D
        print(f"Debug MULTIWINDOWLAYER.combineWindowVectors: After combinationLayer, combinedVector shape: {combinedVector.shape}") # ADDED
        return combinedVector
