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
from tinyAttentionLayer import TINYATTENTIONLAYER
from config import *
from datetime import datetime

"""this class combines all the core components of the babyLLM:"""
"""EMBEDLAYER: token embedding layer"""
"""PARALLELNEURONLAYER: layer of parallel neurons for feature extraction"""
"""OUTPUTLAYER: output layer to generate logits"""
"""MULTIWINDOWLAYER: (New) layer to incorporate multi-window context"""
"""it also manages training, loss computation, backpropagation, and response generation."""
class BABYLLM(nn.Module):
    def __init__(self, vocab, embedDimension, numNeurons, activationFunction):
        super().__init__()
        """CONFIG"""
        self.vocabSize = vocabSize
        self.vocab = vocab
        self.embedDimension = embedDimension
        self.numNeurons = numNeurons
        self.learningRate = learningRate
        self.temperature = temperature
        self.activationFunction = activationFunction
        optimizerClass = getattr(optim, optimizerName)

        """LAYERS"""
        self.embedLayer = EMBEDLAYER(vocabSize, self.embedDimension)
        self.parallelNeuronLayer = PARALLELNEURONLAYER(numNeurons = self.numNeurons, embedDimension = self.embedDimension, activationFunction = self.activationFunction)
        self.outputLayer = OUTPUTLAYER(numNeurons = self.numNeurons, vocabSize = self.vocabSize)
        #self.multiWindowLayer = MULTIWINDOWLAYER(embedDimension = self.embedDimension, windowSizes = [window1, window2, window3])

        """OPTIMIZER - this updates all of the layers learnable parameters"""
        #print("Registered Parameters:")
        #for name, param in BABYLLM.named_parameters(self):
        #    print(name, param.shape)
        self.optimizer = optimizerClass(
            list(self.embedLayer.parameters()) +
            list(self.parallelNeuronLayer.parameters()) + 
            list(self.outputLayer.parameters()),
            lr=self.learningRate, weight_decay=0.001
        )

    def forward(self, inputSeq):
        """processes input sequence of tokens (str) to generate logits to predict the next token"""
        #print(f"Debug: Input to forward: {inputSeq}")

        """convert inputted tokens to indices (batch processing instead of looping)"""
        inputIndices = [self.vocab.tokenToIndex.get(tokenString, self.vocab.tokenToIndex["<UNK>"]) if not isinstance(tokenString, int) else tokenString for tokenString in inputSeq]
        #print(f"Debug BABYLLM.forward: inputIndices: {inputIndices}")

        """convert indices to embeddings"""
        inputEmbeds = []
        for tokenIndex in inputIndices:
            """get embedding vector for each tokenIndex from embedding layer (32 dim x each token x each neuron)"""
            embedVector = self.embedLayer.forward(torch.tensor(tokenIndex))
            inputEmbeds.append(embedVector)

        """DEBUG PRINTS"""
        if inputEmbeds: # Check if inputEmbeds is not empty
            #print(f"Debug BABYLLM.forward: Type of first element in inputEmbeds: {type(inputEmbeds[0])}")
            #print(f"Debug BABYLLM.forward: Shape of first element in inputEmbeds: {inputEmbeds[0].shape}")
            #print(f"Debug BABYLLM.forward: Shapes of first 5 elements in inputEmbeds: {[embed.shape for embed in inputEmbeds[:min(5, len(inputEmbeds))] ]}... (first 5)")
            pass
        else:
            #print(f"Debug BABYLLM.forward: inputEmbeds list is EMPTY!")
            pass

        """PARALLEL NEURON LAYER input/processing (feature extraction)"""
        parallelNeuronOutput = self.parallelNeuronLayer.forward(inputEmbeds) 
        #print(f"Debug BABYLLM.forward: parallelNeuronOutput length: {len(parallelNeuronOutput)}") # ADDED - should be same as input seq len

        """make sure inputEmbeds is a LIST of tensors"""
        if not isinstance(inputEmbeds, list):
            inputEmbeds = [inputEmbeds] 

        """MULTI WINDOW LAYER input/processing (context)"""
        #contextVectors_multiWindow = self.multiWindowLayer.forward(inputEmbeds)

        """COMBINE ACTIVATIONS"""
        """takes the mean of the transformer output activations across the sequence dimension"""
        combinedActivations = torch.mean(parallelNeuronOutput, dim=0, keepdim=True)
        #combinedActivations_multiWindow = self.combineOutputs(combinedActivations, contextVectors_multiWindow)
        #print(f"Debug BABYLLM: Shape of lastTokenActivations BEFORE outputLayer: {lastTokenActivations.shape}")

        # Convert activations to probability distribution
        logits = self.outputLayer.forward(combinedActivations)  
        #logits = self.outputLayer.forward(combinedActivations_multiWindow)  
        #print(f"Debug BABYLLM.forward: probabilityDist shape: {probabilityDist.shape}")
        """returns a logits tensor of shape (1, vocabSize) showing predicted probabilities for the next token"""
        return logits
    
    """computes the cross-entropy loss between the models logits and the target token, essentially checking how good the models prediction was"""
    def computeLoss(self, logits, targetTokenIndex):
        self.targetTokenIndex = torch.tensor([targetTokenIndex], dtype=torch.long)
        #print(f"Debug BABYLLM.computeLoss: predictions shape: {logits.shape}")
        #print(f"Debug BABYLLM.computeLoss: predictions (first 10): {logits[:10]}")
        #print(f"Debug BABYLLM.computeLoss: targetTokenIndex: {targetTokenIndex}")
        #print(f"Debug BABYLLM.computeLoss: self.target: {self.target}")
        """Handle cases where logits might be 1D (unsqueeze to make it 2D for cross_entropy)"""
        """SUS!!"""
        if logits.dim() == 1: 
            logits = logits.unsqueeze(0) 

        self.loss = F.cross_entropy(logits, self.targetTokenIndex) 
        #print(f"Debug BABYLLM.computeLoss: Loss value: {self.loss.item():.4f}")
        """returns a scalar tensor representing the cross-entropy loss value"""
        return self.loss
    
    """backpropagation and optimization, computes gradients of the loss and uses the optimizer to update the models weights"""
    def backward(self, loss):
        self.optimizer.zero_grad()  # Reset gradients
        loss.backward()
        self.optimizer.step()  # Update weights

    """this iterates through training data, performing forward passes, loss computation, backpropagation, and optimization for each step."""
    def train(self, trainingDataPairs, epochs):
        #self.combinationLayer = nn.Linear((self.numNeurons * 5), self.numNeurons)
        babyLLM.loadModel()
        print(f"Debug tokenToIndex (First 20): {list(vocab.tokenToIndex.items())[:20]}")
        print("--- Training Started ---")

        """EPOCH LOOP"""
        for epoch in range(epochs):
            print(f"--- Epoch {epoch+1}/{epochs} Started ---")
            totalLoss = 0 # this is total loss PER 1000 STEPS
            totalLoss2 = 0 # this is total loss PER 10 STEPS

            """TRAINING DATA (batches)"""
            for i, (inputSeq, target) in enumerate(trainingDataPairs):
                inputTokenIndices = [vocab.tokenToIndex.get(token, vocab.tokenToIndex["<UNK>"]) for token in inputSeq]
                targetTokenIndex = vocab.tokenToIndex.get(target, vocab.tokenToIndex["<UNK>"])
                """handles cases where the target might already be an index, or converts to an index"""
                """SUS!!!"""
                if isinstance(target, int):
                    targetTokenIndex = target
                else:
                    targetTokenIndex = vocab.tokenToIndex.get(target, vocab.tokenToIndex["<UNK>"])

                """TRAINING STEP"""
                self.optimizer.zero_grad() # reset gradients from previous step
                logits = self.forward(inputTokenIndices)
                guessedTokenIndex = self.getResponseFromLogits(logits)
                loss = self.computeLoss(logits, targetTokenIndex)
                loss.backward()
                self.optimizer.step()
                totalLoss += loss.item()
                totalLoss2 += loss.item()

                """PRINTING LOSS TO LOGS AND TERMINAL"""
                if i == 1:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get timestamp
                    runStart = f"\n--- {timestamp} ---\n"
                    print(f" {runStart.strip()}")
                    with open("trainingLogDetail.txt", "a") as logFile:
                        logFile.write(runStart)
                    with open("trainingLog.txt", "a") as logFile:
                        logFile.write(runStart)

                # Track loss every 1000 steps
                if (i + 1) % printLossFreq == 0:  
                    avgLoss = totalLoss / printLossFreq  # Compute average loss
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get timestamp
                    lossLog = f"{timestamp} | Context: {[allWindowSizes]} | LR: {learningRate:.5f} | Step {i+1} | Avg Loss: {avgLoss:.4f}\n"
                    print(f" {lossLog.strip()}")
                    with open("trainingLog.txt", "a") as logFile:
                        logFile.write(lossLog)
                    totalLoss = 0

                # Track loss every 10 steps
                if (i + 1) % printLossFreq2 == 0:  
                    avgLoss2 = totalLoss2 / printLossFreq2  # Compute average loss
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get timestamp
                    lossLog = f"{timestamp} | Context: {[allWindowSizes]} | LR: {learningRate:.5f} | Step {i+1} | Avg Loss: {avgLoss2:.4f}\n"
                    print(f" {lossLog.strip()}")
                    with open("trainingLogDetail.txt", "a") as logFile:
                        logFile.write(lossLog)
                    totalLoss2 = 0
                
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
                    elif loss.item() > superHighLoss:  # super high loss
                        formattedWords = f"{DIM}Step {i+1}: {inputSentence} → {guessedTokenString}[?] {targetWord} | {RESET}{FLASHING_RED}Loss: {loss.item():.3f}{RESET}"  
                    elif loss.item() > highLoss:  # high loss
                        formattedWords = f"{DIM}Step {i+1}: {inputSentence} → {guessedTokenString}[?] {targetWord} | {RESET}{RED}Loss: {loss.item():.3f}{RESET}"  
                    elif loss.item() > prettyHighLoss:  # pretty high loss
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

                """SAVE THE MODEL EVERY x STEPS"""
                if i > 0 and int(i % saveModelFreq) == 0:
                    # self.saveModel(f"babyLLM_epoch{epoch}_{int(i / (len(trainingDataPairs) / 2000))}.pth")
                    self.saveModel()

            print(f"Epoch {epoch+1}/{epochs} - Loss: {totalLoss:.4f}")
            #torch.save(self.state_dict(), f"babyLLM_epoch{epoch}.pth")

        babyLLM.saveModel()
        print("--- Training Completed ---")
        
    """saves the model to a file"""    
    def saveModel(self, filePath="babyLLM.pth"):
        torch.save(self.state_dict(), filePath)
        print(f"✅ Model saved to {filePath}!")

    """loads the model from a file"""
    def loadModel(self, filePath = modelPath):
        try:
            print(f"Loading model from path: {filePath}") 
            self.load_state_dict(torch.load(filePath), strict = saveLock)
            print(f"Model loaded from {filePath}!")
        except FileNotFoundError:
            print("No saved model found.")

    """this takes the output logits, does temperature scaling and softmax to create a probability distribution over the vocab, 
    and then selects most likely response token"""
    def getResponseFromLogits(self, logits, temperature=None):
        if temperature is None:
            temperature = self.temperature
        logits = logits / temperature
        softmaxed = torch.softmax(logits, dim=1)  # apply softmax to logits to convert to probabilities (along vocab dimension - dim=1)
        topProb = topP
        guessedTokenIndex = None
        for tokenIndex in range(vocabSize):
            prob = softmaxed[0][tokenIndex].item()
            if prob > topProb or guessedTokenIndex is None:
                guessedTokenIndex = tokenIndex
                topProb = prob
        """return the token index with the highest probability of being next"""
        return guessedTokenIndex
    
    """convert token index to string"""
    def getTokenIndexAsString(self, tokenIndex):
        """returns the guessed token as a readable string aka text"""
        return self.vocab.indexToToken[tokenIndex.__str__()] 
    
    """generates the chosen next token using getResponseFromLogits"""
    def getNextToken(self, inputSeq, temperature=None):  
        if temperature is None:
            temperature = self.temperature  # Grab from self.temperature (config)
        """returns an integer token index showing the models predicted next token."""
        return self.getResponseFromLogits(self.forward(inputSeq), temperature)
    
    """combines the parallelNeronLayer output and the multiWindowLayer output into one output"""
    def combineOutputs(self, output1, output2):
        #print(f"Debug combineOutputs: Shape of output1: {output1.shape}")
        #print(f"Debug combineOutputs: Shape of output2: {output2.shape}")
        output1Flat = output1.squeeze(dim=2) # Remove dimension of shape 1, new shape: [1, 10000]
        #print(f"Debug combineOutputs: Shape of output1Flat: {output1Flat.shape}")
        concatenatedOutput = torch.cat((output1Flat, output2), dim=1) # Concatenate along dim=1 (feature dimension) - 2D tensors
        """linear layer to combine and reduce dimensionality"""
        if not hasattr(self, 'outputCombinationLayer'):
            combined_dim = output1Flat.shape[1] + output2.shape[1] # dim=1 is the feature dimension
            self.outputCombinationLayer = nn.Linear(combined_dim, embedDimension) # Output dimension should be embedDimension
        finalOutput = self.outputCombinationLayer(concatenatedOutput)
        """returns a single combined output tensor of shape (1, embedDimension)."""
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

    trainingDataPairs = vocab.genTrainingData(windowMAX)
    babyLLM.train(trainingDataPairs, epochs = epochs)

    print("--- BabyLLM TESTING START ---")
    print(f"Vocab size: {len(babyLLM.vocab.vocabList)}")
    print("\n--- BabyLLM TESTING COMPLETED ---")
