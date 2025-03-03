# CHARIS CAT 2025

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim 
from vocab import VOCAB
from embedLayer import EMBEDLAYER
from transformerLayer import TRANSFORMERLAYER
from outputLayer import OUTPUTLAYER
from neuron import NEURON
from config import *

class BABYLLM(nn.Module):
    def __init__(self, vocab, embedDimension, numNeurons, activationFunction):
        super().__init__()
        self.vocabSize = vocabSize
        self.vocab = vocab
        self.embedDimension = embedDimension
        self.numNeurons = numNeurons
        self.activationFunction = activationFunction

        self.embedLayer = EMBEDLAYER(vocabSize, self.embedDimension)
        self.transformerLayer = TRANSFORMERLAYER(numNeurons = self.numNeurons, embedDimension = self.embedDimension, activationFunction = self.activationFunction)
        self.outputLayer = OUTPUTLAYER(numNeurons = self.numNeurons, vocabSize = self.vocabSize)
        self.optimizer = optim.Adam(
                        list(self.embedLayer.parameters()) +
                        list(self.transformerLayer.parameters()) + # ADDED transformerLayer parameters
                        list(self.outputLayer.parameters()),
                        lr=0.0001)

    def forward(self, inputSeq):
        #print(f"Debug: Input to forward: {inputSeq}")

        # Convert tokens to indices (batch processing instead of looping)
        inputIndices = [self.vocab.tokenToIndex.get(token, self.vocab.tokenToIndex["<UNK>"]) if not isinstance(token, int) else token for token in inputSeq]
        #print(f"Debug BABYLLM.forward: inputIndices: {inputIndices}")

        # Convert indices to embeddings
        inputEmbeds = self.embedLayer.forward(torch.tensor(inputIndices))
        #print(f"Debug BABYLLM.forward: inputEmbeds shape: {inputEmbeds.shape}")
        #print(f"Debug BABYLLM.forward: inputEmbeds (first few):\n{inputEmbeds[:5]}")

        # Process embeddings through Transformer Layer
        transformerOutput = self.transformerLayer.forward(inputEmbeds) 
        #print(f"Debug BABYLLM.forward: transformerOutput length: {len(transformerOutput)}") # ADDED - should be same as input seq len

        # Take last token's activations
        #lastTokenActivations = transformerOutput[-1]  
        combinedActivations = torch.mean(transformerOutput, dim=0, keepdim=True)

        #print(f"Debug BABYLLM: Shape of lastTokenActivations BEFORE outputLayer: {lastTokenActivations.shape}")

        # Convert activations to probability distribution
        logits = self.outputLayer.forward(combinedActivations)  
        #print(f"Debug BABYLLM.forward: probabilityDist shape: {probabilityDist.shape}") # ADDED

        return logits
    
    def computeLoss(self, logits, targetTokenIndex):
        self.target = torch.tensor([targetTokenIndex], dtype=torch.long)
        #print(f"Debug BABYLLM.computeLoss: predictions shape: {logits.shape}") # ADDED
        #print(f"Debug BABYLLM.computeLoss: predictions (first 10): {logits[:10]}") # ADDED
        #print(f"Debug BABYLLM.computeLoss: targetTokenIndex: {targetTokenIndex}") # ADDED
        #print(f"Debug BABYLLM.computeLoss: self.target: {self.target}") # ADDED

        if logits.dim() == 1: 
            logits = logits.unsqueeze(0) 

        self.loss = F.cross_entropy(logits, self.target) 
        #print(f"Debug BABYLLM.computeLoss: Loss value: {self.loss.item():.4f}") # ADDED
        return self.loss
    
    def backward(self, loss):
        self.optimizer.zero_grad()  # Reset gradients
        loss.backward()
        self.optimizer.step()  # Update weights

    def train(self, trainingData, epochs):
        babyLLM.loadModel()
        print(f"Debug tokenToIndex (First 20): {list(vocab.tokenToIndex.items())[:20]}")
        print("--- Training Started ---")

        for epoch in range(epochs):
            print(f"--- Epoch {epoch+1}/{epochs} Started ---")
            totalLoss = 0

            for i, (inputSeq, target) in enumerate(trainingData):
                print(f" Processing training example {i+1}/{len(trainingData)}...")
                print(f" Training on: {inputSeq} -> {target}")

                inputTokenIndices = [vocab.tokenToIndex.get(token, vocab.tokenToIndex["<UNK>"]) for token in inputSeq]
                targetTokenIndex = vocab.tokenToIndex.get(target, vocab.tokenToIndex["<UNK>"])

                if isinstance(target, int):
                    targetTokenIndex = target
                else:
                    targetTokenIndex = vocab.tokenToIndex.get(target, vocab.tokenToIndex["<UNK>"])

                self.optimizer.zero_grad()
                logits = self.forward(inputTokenIndices)
                self.getResponseFromLogits(logits)
                loss = self.computeLoss(logits, targetTokenIndex)
                loss.backward()
                self.optimizer.step()
                totalLoss += loss.item()
                print(f"    Example {i+1}/{len(trainingData)} Loss: {loss.item():.4f}")
                if i > 0 and int(i % (len(trainingData) / 700)) == 0:
                    self.saveModel(f"babyLLM_epoch{epoch}_{int(i / (len(trainingData) / 700))}.pth")
                    self.saveModel()

            print(f"Epoch {epoch+1}/{epochs} - Loss: {totalLoss:.4f}")
            #torch.save(self.state_dict(), f"babyLLM_epoch{epoch}.pth")

        babyLLM.saveModel()
        print("--- Training Completed ---")
        
    def saveModel(self, filePath="babyLLM.pth"):
        torch.save(self.state_dict(), filePath)
        print(f"✅ Model saved to {filePath}!")

    def loadModel(self, filePath="babyLLM.pth"):
        try:
            self.load_state_dict(torch.load(filePath))
            print(f"🔄 Model loaded from {filePath}!")
        except FileNotFoundError:
            print("⚠ No saved model found.")

    def getResponseFromLogits(self, logits):
        softmaxed = torch.softmax(logits, dim=1)

        topProb = 0
        tokenGuessed = None
        for tokenIndex in range(vocabSize):
            prob = softmaxed[0][tokenIndex].item()
            if prob > topProb or tokenGuessed is None:
                tokenGuessed = tokenIndex
                topProb = prob
        
        print(f"(probability {softmaxed[0][tokenGuessed].item()}) Got word ---> \"{self.getReadableToken(tokenGuessed)}\" ")

        return tokenGuessed
    
    def getReadableToken(self, token):
        return self.vocab.indexToToken[token.__str__()]
    
    def getNextToken(self, inputSeq):
        return self.getResponseFromLogits(self.forward(inputSeq))
    
if __name__ == "__main__":
    vocab = VOCAB(vocabSize = vocabSize)
    embedDimension = embedDimension
    numNeurons = numNeurons
    activationFunction = activationFunction

    babyLLM = BABYLLM(vocab = vocab, embedDimension = embedDimension, numNeurons = numNeurons, activationFunction = activationFunction)
    #TESTinputSeq = ["what","will","you","do","out","there","now","?"]
    TESTinputSeq = ["i","love","you","this","is","good","music","is","life",]
    #TESTinputSeq = ["what"]
    probabilityDist = babyLLM.forward(TESTinputSeq)
    #TESTtrainingData = [
    #(["i", "love"], "you"),
    #(["this", "is"], "good"),
    #(["music", "is"], "life")]

    trainingData = vocab.genTrainingData(trainingWindow)
    babyLLM.train(trainingData, epochs = epochs)

    print("--- BabyLLM Forward Pass Testing ---")
    print(f"Vocab size: {len(babyLLM.vocab.vocabList)}")
    print(f"Probability Distribution (first 100):")
    print(probabilityDist[:100]) # Print first 10 probabilities
    print(f"Probability Distribution Shape: {probabilityDist.shape}") # Check shape (should be vocabulary_size)
    print(f"Sum of Probabilities: {torch.sum(probabilityDist).item():.4f}") # Check if probabilities sum to ~1.0
    print("\n--- Baby LLM Forward Pass Testing Completed ---")
