# CHARIS CAT 2025

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim 
from vocab import VOCAB
from embedLayer import EMBEDLAYER
from parallelNeuronLayer import PARALLELNEURONLAYER
from outputLayer import OUTPUTLAYER
from neuron import NEURON
from multiWindowLayer import MULTIWINDOWLAYER
from config import *
from datetime import datetime

class BABYLLM(nn.Module):
    def __init__(self, vocab, embedDimension, numNeurons, activationFunction):
        super().__init__()
        self.vocabSize = vocabSize
        self.vocab = vocab
        self.embedDimension = embedDimension
        self.numNeurons = numNeurons
        self.learningRate = learningRate
        self.temperature = temperature
        self.activationFunction = activationFunction
        optimizerClass = getattr(optim, optimizerName)
        
        self.embedLayer = EMBEDLAYER(vocabSize, self.embedDimension)
        self.parallelNeuronLayer = PARALLELNEURONLAYER(numNeurons = self.numNeurons, embedDimension = self.embedDimension, activationFunction = self.activationFunction)
        self.outputLayer = OUTPUTLAYER(numNeurons = self.numNeurons, vocabSize = self.vocabSize)
        self.multiWindowLayer = MULTIWINDOWLAYER(embedDimension = self.embedDimension, windowSizes = [window1, window2, window3])
        self.optimizer = optimizerClass(
            list(self.embedLayer.parameters()) +
            list(self.parallelNeuronLayer.parameters()) + 
            list(self.outputLayer.parameters()),
            lr=self.learningRate, weight_decay=0.001
        )

    def forward(self, inputSeq):
        #print(f"Debug: Input to forward: {inputSeq}")

        # Convert tokens to indices (batch processing instead of looping)
        inputIndices = [self.vocab.tokenToIndex.get(token, self.vocab.tokenToIndex["<UNK>"]) if not isinstance(token, int) else token for token in inputSeq]
        #print(f"Debug BABYLLM.forward: inputIndices: {inputIndices}")

        # Convert indices to embeddings
        #inputEmbeds = self.embedLayer.forward(torch.tensor(inputIndices))
        inputEmbeds = []
        for index in inputIndices:
            embedVector = self.embedLayer.forward(torch.tensor(index))
            inputEmbeds.append(embedVector)
        #print(f"Debug BABYLLM.forward: inputEmbeds shape: {inputEmbeds.shape}")
        #print(f"Debug BABYLLM.forward: inputEmbeds (first few):\n{inputEmbeds[:5]}")

        print(f"Debug BABYLLM.forward: Length of inputEmbeds list: {len(inputEmbeds)}") # DEBUG - LENGTH OF INPUTEMBEDS LIST
        if inputEmbeds: # Check if inputEmbeds is not empty
            print(f"Debug BABYLLM.forward: Type of first element in inputEmbeds: {type(inputEmbeds[0])}") # DEBUG - TYPE OF FIRST ELEMENT
            print(f"Debug BABYLLM.forward: Shape of first element in inputEmbeds: {inputEmbeds[0].shape}") # DEBUG - SHAPE OF FIRST ELEMENT
            print(f"Debug BABYLLM.forward: Shapes of first 5 elements in inputEmbeds: {[embed.shape for embed in inputEmbeds[:min(5, len(inputEmbeds))] ]}... (first 5)") # DEBUG - SHAPES OF FIRST 5 ELEMENTS
        else:
            print(f"Debug BABYLLM.forward: inputEmbeds list is EMPTY!")

        # Process embeddings through Transformer Layer
        transformerOutput = self.parallelNeuronLayer.forward(inputEmbeds) 
        #print(f"Debug BABYLLM.forward: transformerOutput length: {len(transformerOutput)}") # ADDED - should be same as input seq len

        # make sure inputEmbeds is a LIST of tensors
        if not isinstance(inputEmbeds, list):
            inputEmbeds = [inputEmbeds] 

        # multi window input/processing
        contextVectors_multiWindow = self.multiWindowLayer.forward(inputEmbeds)

        # Take last token's activations
        #lastTokenActivations = transformerOutput[-1]  

        # combine activation vectors
        combinedActivations = torch.mean(transformerOutput, dim=0, keepdim=True)
        combinedActivations_multiWindow = self.combineOutputs(combinedActivations, contextVectors_multiWindow)
        #print(f"Debug BABYLLM: Shape of lastTokenActivations BEFORE outputLayer: {lastTokenActivations.shape}")

        # Convert activations to probability distribution
        logits = self.outputLayer.forward(combinedActivations_multiWindow)  
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
                inputTokenIndices = [vocab.tokenToIndex.get(token, vocab.tokenToIndex["<UNK>"]) for token in inputSeq]
                targetTokenIndex = vocab.tokenToIndex.get(target, vocab.tokenToIndex["<UNK>"])

                if isinstance(target, int):
                    targetTokenIndex = target
                else:
                    targetTokenIndex = vocab.tokenToIndex.get(target, vocab.tokenToIndex["<UNK>"])

                self.optimizer.zero_grad()
                logits = self.forward(inputTokenIndices)
                guessedTokenIndex = self.getResponseFromLogits(logits)
                loss = self.computeLoss(logits, targetTokenIndex)
                loss.backward()
                self.optimizer.step()
                totalLoss += loss.item()
                # Track loss every 1000 steps
                if (i + 1) % printLossFreq == 0:  
                    avg_loss = totalLoss / 1000  # Compute average loss
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get timestamp
                    lossLog = f"{timestamp} | Context: {trainingWindow} | LR: {learningRate} | Step {i+1} | Avg Loss: {avg_loss:.4f}\\n"
                    print(f"🔥 {lossLog.strip()}")
                    with open("trainingLog.txt", "a") as log_file:
                        log_file.write(lossLog)
                    totalLoss = 0
                
                """PRINTING GUESSES TO THE TERMINAL"""
                if (i + 1) % printFreq == 0:  
                    inputSentence = "".join(inputSeq).replace("Ġ", " ").lstrip()
                    targetWord = target.replace("Ġ", "")
                    guessedTokenString = self.getTokenIndexAsString(guessedTokenIndex).replace("Ġ", "")
                    isCorrect = (targetWord == guessedTokenString)
                    isPerfect = isCorrect and loss.item() == 0.01
                    self.lowLoss = lowLoss
                    self.veryLowLoss = veryLowLoss
                    #print(f"DEBUG -> Step {i+1}: Target='{targetWord}', Guess='{guessedTokenString}', Loss={loss.item():.4f}, isCorrect={isCorrect}, isPerfect={isPerfect}")
                    if isPerfect:
                        formattedWords = f"{GOLD} Step {i+1}: {inputSentence}{RESET}{DIM} → {RESET}{GOLD}{guessedTokenString}{RESET}{DIM}[!] {RESET}{GOLD}{targetWord}{RESET}{DIM} | {RESET}{GOLD}Loss: {loss.item():.3f} {RESET}"
                    elif isCorrect and loss.item() < self.veryLowLoss:  # correct, very low loss
                        formattedWords = f"{DIM}Step {i+1}: {RESET}{PURPLE}{inputSentence}{RESET}{DIM} → {RESET}{PURPLE}{guessedTokenString}{RESET}{DIM}[!] {RESET}{PURPLE}{targetWord}{RESET}{DIM} | {RESET}{PURPLE}Loss: {loss.item():.3f}{RESET}"
                    elif isCorrect and loss.item() < self.lowLoss:  # correct, low loss
                        formattedWords = f"{DIM}Step {i+1}: {RESET}{LIGHT_PURPLE}{inputSentence}{RESET}{DIM} → {RESET}{PURPLE}{guessedTokenString}{RESET}{DIM}[!] {RESET}{PURPLE}{targetWord}{RESET}{DIM} | {RESET}{LIGHT_PURPLE}Loss: {loss.item():.3f}{RESET}"  
                    elif loss.item() > 30.0:  # super high loss
                        formattedWords = f"{DIM}Step {i+1}: {inputSentence} → {guessedTokenString}[?] {targetWord} | {RESET}{FLASHING_RED}Loss: {loss.item():.3f}{RESET}"  
                    elif loss.item() > 10.0:  # high loss
                        formattedWords = f"{DIM}Step {i+1}: {inputSentence} → {guessedTokenString}[?] {targetWord} | {RESET}{RED}Loss: {loss.item():.3f}{RESET}"  
                    elif loss.item() > 5.0:  # pretty high loss
                        formattedWords = f"{DIM}Step {i+1}: {inputSentence} → {guessedTokenString}[?] {targetWord} | {RESET}{ORANGE}Loss: {loss.item():.3f}{RESET}"  
                    elif loss.item() < self.veryLowLoss:  # incorrect, very low loss
                        formattedWords = f"{DIM}Step {i+1}: {inputSentence} → {guessedTokenString}[?] {targetWord} | {RESET}{PURPLE}Loss: {loss.item():.3f}{RESET}"  
                    elif loss.item() < self.lowLoss:  # incorrect, low loss
                        formattedWords = f"{DIM}Step {i+1}: {inputSentence} → {guessedTokenString}[?] {targetWord} | {RESET}{PURPLE}Loss: {loss.item():.3f}{RESET}"  
                    elif isCorrect:  # correct, normal loss
                        formattedWords = f"{DIM}Step {i+1}: {RESET}{LIGHT_PURPLE}{inputSentence}{RESET}{DIM} → {RESET}{PURPLE}{guessedTokenString}{RESET}{DIM}[!]  {RESET}{PURPLE}{targetWord}{RESET} {DIM}| Loss: {loss.item():.3f}{RESET}"  
                    else:  # default
                        formattedWords = f"{DIM}Step {i+1}: {inputSentence} → {guessedTokenString}[?] {targetWord} | Loss: {loss.item():.3f}{RESET}"  
  
                print(formattedWords)
                #print(f"Loss debug: {loss.item()} (Raw) | Rounded: {round(loss.item(), 6)}")

                if i > 0 and int(i % 250) == 0:
                    # self.saveModel(f"babyLLM_epoch{epoch}_{int(i / (len(trainingData) / 2000))}.pth")
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

    def getResponseFromLogits(self, logits, temperature=None):
        if temperature is None:
            temperature = self.temperature
        logits = logits / temperature
        softmaxed = torch.softmax(logits, dim=1)  # Convert to probabilities after scaling
        topProb = 0
        guessedTokenIndex = None
        for tokenIndex in range(vocabSize):
            prob = softmaxed[0][tokenIndex].item()
            if prob > topProb or guessedTokenIndex is None:
                guessedTokenIndex = tokenIndex
                topProb = prob
        return guessedTokenIndex
    
    def getTokenIndexAsString(self, tokenIndex):
        return self.vocab.indexToToken[tokenIndex] # i dont really know why this is a string. but right now it doesnt work if i change it (take the .__str__() out)
    
    def getNextToken(self, inputSeq, temperature=None):  
        if temperature is None:
            temperature = self.temperature  # Grab from self.temperature (config)
        return self.getResponseFromLogits(self.forward(inputSeq), temperature)
    
    def combineOutputs(self, output1, output2):
        print(f"Debug combineOutputs: Shape of output1: {output1.shape}")
        print(f"Debug combineOutputs: Shape of output2: {output2.shape}")
        output1Flat = output1.squeeze(dim=2) # Shape: [1, 10000]
        print(f"Debug combineOutputs: Shape of output1_flattened: {output1Flat.shape}")
        # output2 is already [1, 32]
        concatenatedOutput = torch.cat((output1Flat, output2), dim=1) # Concatenate along dim=1 (feature dimension) - 2D tensors
        if not hasattr(self, 'outputCombinationLayer'):
            combined_dim = output1Flat.shape[1] + output2.shape[1] # dim=1 is the feature dimension
            self.outputCombinationLayer = nn.Linear(combined_dim, embedDimension) # Output dimension should be embedDimension
        finalOutput = self.outputCombinationLayer(concatenatedOutput)
        return finalOutput
    
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
