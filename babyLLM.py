# CHARIS CAT 2025

import torch
import torch.nn.functional as F
import torch.optim as optim 
from vocab import VOCAB
from embedLayer import EMBEDLAYER
from transformerLayer import TRANSFORMERLAYER
from outputLayer import OUTPUTLAYER
from neuron import NEURON
from config import *

class BABYLLM:
    def __init__(self, vocab, embedDimension, numNeurons, activationFunction):
        self.vocab = vocab
        self.vocabSize = vocabSize
        self.embedDimension = embedDimension
        self.numNeurons = numNeurons
        self.activationFunction = activationFunction

        self.embedLayer = EMBEDLAYER(vocabSize, self.embedDimension)
        self.transformerLayer = TRANSFORMERLAYER(numNeurons = self.numNeurons, embedDimension = self.embedDimension, activationFunction = self.activationFunction)
        self.outputLayer = OUTPUTLAYER(numNeurons = self.numNeurons, vocabSize = self.vocabSize)
        self.optimize = optim.Adam(
                        list(self.embedLayer.parameters()) +
                        list(self.transformerLayer.parameters()) + # ADDED transformerLayer parameters
                        list(self.outputLayer.parameters()),
                        lr=0.001)

    def forward(self, inputSeq):
        # Convert input tokens into embedding vectors
        #inputEmbeds = self.embedLayer.forward(inputSeq) 
        inputEmbeds = [] # creates list of each embed vector
        for token in inputSeq:
            tokenIndex = vocab.tokenToIndex[token]
            embedVector = self.embedLayer.forward(tokenIndex)
            inputEmbeds.append(embedVector)

        # Process embeddings through Transformer Layer
        self.transformerOutput = self.transformerLayer.forward(inputEmbeds) 

        # Convert transformer activations to probability distribution
         #probabilityDist = self.outputLayer.forward(transformerOutput)
        self.transformerOutputTensor = torch.cat(self.transformerOutput, dim=0).view(1, -1)
        self.probabilityDist = self.outputLayer.forward(self.transformerOutputTensor)

        return self.probabilityDist
    
    def computeLoss(self, predictions, targetTokenIndex):
        self.target = torch.tensor([targetTokenIndex], dtype=torch.long)  # Convert target to tensor
        self.loss = F.cross_entropy(predictions, self.target)  # Compute loss
        return self.loss
    
    def backward(self, loss):
        self.optimize.zero_grad()  # Reset gradients
        loss.backward()  # Compute gradients
        self.optimize.step()  # Update weights

    def train(self, trainingData, epochs):

        for epoch in range(epochs):
            totalLoss = 0

            for inputSeq, target in trainingData:
                # Convert input words to token indices
                inputTokenIndices = [vocab.tokenToIndex[word] for word in inputSeq]
                targetTokenIndex = vocab.tokenToIndex[target]
                predictions = self.forward(inputTokenIndices)
                loss = self.computeLoss(predictions, targetTokenIndex)

                self.backward(loss)
                totalLoss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {totalLoss:.4f}")
    
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

    trainingData = vocab.genTrainingData(trainingWindow=2)
    babyLLM.train(trainingData, epochs = epochs)

    print("--- BabyLLM Forward Pass Testing ---")
    print(f"Vocab size: {len(babyLLM.vocab.vocabList)}")
    print(f"Probability Distribution (first 100):")
    print(probabilityDist[:100]) # Print first 10 probabilities
    print(f"Probability Distribution Shape: {probabilityDist.shape}") # Check shape (should be vocabulary_size)
    print(f"Sum of Probabilities: {torch.sum(probabilityDist).item():.4f}") # Check if probabilities sum to ~1.0
    print("\n--- Baby LLM Forward Pass Testing Completed ---")
