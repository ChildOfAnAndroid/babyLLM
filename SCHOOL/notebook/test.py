trainingLogFreq_1000 = 10

from collections import Counter
import random
stats = Counter({
    "Loss": 0,
    "Grad Norm": 0,
    "Logit Min": 0,
    "Logit Max": 0,
})

for _ in range(10):
    for _ in range(10):
        stats.update({
            "Loss": random.random(),
            "Grad Norm": random.random(),
        })
    avgStats = {}
    for key, value in stats.items():
        avgStats[key] = value / trainingLogFreq_1000 if trainingLogFreq_1000 > 0 else 0

    print(" ".join([f"{key}: {(value/trainingLogFreq_1000 if trainingLogFreq_1000 > 0 else 0):.2f}" for key, value in stats.most_common()]))

    stats.clear()

"""    import code
    vars = globals().copy()
    vars.update(locals())
    code.interact(local=vars)"""

"""for attr in dir(babyLLM):
    if not attr.startswith("__"):
        print(attr, "→", getattr(babyLLM, attr))"""

"""geepy thingy test"""

"""def deep_inspect(obj, prefix="babyLLM", depth=0, max_depth=2):
        indent = "    " * depth
        if depth > max_depth:
            return

        print(f"{indent}inspecting... {prefix} ({type(obj).__name__})")

        for attr in dir(obj):
            if attr.startswith("__"):
                continue

            try:
                value = getattr(obj, attr)
                if callable(value):
                    print(f"{indent}  {attr} → <function>")
                elif isinstance(value, (int, float, str, bool, torch.Tensor)):
                    val_str = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
                    print(f"{indent}  {attr} → {val_str}")
                else:
                    print(f"{indent}  {attr} → {type(value).__name__}")
                    babyLLM.deep_inspect(value, prefix=f"{prefix}.{attr}", depth=depth+1, max_depth=max_depth)
            except Exception as e:
                print(f"{indent}  {attr} → <error reading value: {e}>")"""

"""def HUD_fixScroll(self):
    height = shutil.get_terminal_size().lines
    #reserved_hud_lines = 5
    #training_lines_height = height - reserved_hud_lines

    #sys.stdout.write("\033[?25l\033[H\033[2J")  # Hide cursor, clear, move to top
    #sys.stdout.flush()

    # You should print training lines here *before* calling this if you want control

    # Move to bottom section and draw HUD
    training_lines_height = training_lines_height
    sys.stdout.write(f"\033[{training_lines_height + 1};0H")  # Move to HUD zone
    sys.stdout.flush()

    self.printHUD(
        windowWeights=(F.softmax(self.parallelNeuronLayer.windowWeighting, dim=0) + 0.1).detach().cpu().numpy(),
        guessHUD=self.guessHUD
    )

    sys.stdout.write(f"\033[{height};0H")  # Move cursor just above HUD for next cycle
    sys.stdout.flush()"""
"""
class EMBED(nn.Module):
    self.weights = nn.Parameter(torch.randn(vocabSize, embedDimension, device = modelDevice))

class NEURON(nn.Module):
        self.n_weights = nn.Parameter(torch.randn(numNeurons, embedDimension, device = modelDevice) * 0.01)
        self.n_biases = nn.Parameter(torch.zeros(numNeurons, device = modelDevice))

class INTERNEURON_NETWORK(nn.Module):
        self.cerebellum = nn.Parameter(torch.ones(len(allWindowSizes_new), device = modelDevice)) # THIS WAS THE WINDOW WEIGHTING LAYER
        self.windowCombos = nn.ModuleList([nn.Linear(numNeurons, numNeurons, device = modelDevice) for _ in range(len(allWindowSizes_new))])
        self.queryProj = nn.Linear(numNeurons, embedDimension, bias=True, device=modelDevice)
        self.keyProj = nn.Linear(numNeurons, embedDimension, bias=True, device=modelDevice)
        self.judgeBias = nn.Parameter(torch.zeros(len(allWindowSizes_new), device = modelDevice))
        self.credibilityBias = nn.Parameter(torch.zeros(len(allWindowSizes_new), device = modelDevice))

class LOGITS(nn.Module):
        self.weights = nn.Parameter(torch.randn(numNeurons, vocabSize, device = modelDevice)) # this is set to move the NEURON ACTIVATIONS (10000) onto VOCAB SIZE (2000)
        self.bias = nn.Parameter(torch.zeros(vocabSize, device = modelDevice))

class MEMORY(nn.Module):
        self.shortTermDecay = nn.Parameter(torch.tensor(0.7, device = modelDevice))
        self.longTermDecay = nn.Parameter(torch.tensor(0.95, device = modelDevice))
        self.shortTermMemory = torch.zeros(numNeurons, device = modelDevice)
        self.longTermMemory = torch.zeros(numNeurons, device = modelDevice)
        self.shortGate = nn.Parameter(torch.tensor(0.25, device = modelDevice))  
        self.longGate = nn.Parameter(torch.tensor(0.25, device = modelDevice))
        self.currentGate = nn.Parameter(torch.tensor(0.5, device = modelDevice))

class BABYLLM(nn.Module):
        self.embed = EMBED(vocabSize, embedDimension)
        self.interneuronNetwork = INTERNEURON_NETWORK()
        self.logits = LOGITS(numNeurons = numNeurons, vocabSize = vocabSize)
        self.memory = MEMORY(numNeurons = numNeurons)
        self.optimizer = optimizerClass(
            list(self.embed.parameters()) +
            list(self.interneuronNetwork.parameters()) + 
            list(self.logits.parameters()) +
            list(self.memory.parameters()),
            lr=learningRate, weight_decay=0.001
        )

EMBED
def forward(self, tokenIndex):
    with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
        ʕっʘ‿ʘʔっ("tokenIndex.to(self.weights.device)")
        tokenIndex = tokenIndex.to(self.weights.device)
        ʕっʘ‿ʘʔっ("self.weights[tokenIndex]")
        embedVector = self.weights[tokenIndex] 
        return embedVector 

NEURON
def forward(self, inputEmbeds):  # embed: (batch_size, embed_size)
    with self.n_counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:

        ʕっʘ‿ʘʔっ("computeBatchedDotProduct+bias") # Compute batched dot product + bias: (batch_size, num_neurons)
        output = torch.matmul(inputEmbeds, self.n_weights.T) + self.n_biases   

        ʕっʘ‿ʘʔっ("activationFunction") # magic activation function applied to this weighted sum, which outputs a single number from the neuron
        output = activationFunction(output)
        return torch.clamp(output, -5, 5)
    
INTERNEURON_NETWORK
def forward(self, inputEmbeds):  
    with self.inn_counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
    # --- iterates through input embeddings, applies all neurons in parallel for each, produces a vector of neuron outputs
        ʕっʘ‿ʘʔっ("localParamInit") # AVOIDING SELF - parameters only used in this function and never passed
        tinyWindowCount = 0
        # --- DO NOT TAKE ANYTHING TO SELF PAST HERE, IT SHOULD ALL PASS THROUGH BACKWARD WITHOUT SAVING! --- #
        ʕっʘ‿ʘʔっ("CALL NEURON FORWARD")
        neuronActivations = self.neurons(inputEmbeds)
        ʕっʘ‿ʘʔっ("windowOutputs") # --- combine activations into their own learnable layer
        windowOutputs = []
        if debugPrints: 
            for name, param in self.named_parameters():
                print(name, param.requires_grad, param.grad is not None)
        for windowSize in allWindowSizes_new:
            if inputEmbeds.shape[0] < windowSize: 
                ʕっʘ‿ʘʔっ("not enough tokens for window (neurons.n_weights.mean)") # --- Not enough tokens for this window; use a zero vector
                tinyWindowCount += 1
                #summary = torch.zeros_like(numNeurons, device = modelDevice)
                summary = neuronActivations.mean(dim=0) * 0  # KEEPS GRADIENTS FLOWING EVEN WHEN ZERO - shape: [numNeurons], safe to stack
            else: # Mean pooling over the last 'windowSize' token activations
                ʕっʘ‿ʘʔっ("mean pooling all over all tokens (torch.mean)") # --- MEAN IS OVER WINDOW SIZE
                summary = torch.mean(neuronActivations[-windowSize:], dim=0)
            ʕっʘ‿ʘʔっ("append window summaries")
            if debugPrints: 
                for name, param in self.named_parameters():
                    print(name, param.requires_grad, param.grad is not None)
            windowOutputs.append(summary)

        ʕっʘ‿ʘʔっ("windowOutputTensor") # Stack summaries into a tensor of shape (num_windows, numNeurons)
        windowOutputsTensor = torch.stack(windowOutputs, dim=0)  # shape: (32, numNeurons)

        # Project summaries to queries and keys for attention scoring
        ʕっʘ‿ʘʔっ("cerebellumSoft")
        self.cerebellumSoft = F.softmax(self.cerebellum, dim=0) # THIS WAS THE WINDOW WEIGHTING LAYER
        ʕっʘ‿ʘʔっ("query")
        query = self.queryProj(windowOutputsTensor) + self.judgeBias.unsqueeze(1) + self.cerebellum.unsqueeze(1)  # shape: (32, numNeurons)
        ʕっʘ‿ʘʔっ("key")
        key = self.keyProj(windowOutputsTensor) + self.credibilityBias.unsqueeze(1) + self.cerebellum.unsqueeze(1)   # shape: (32, numNeurons)
        # Compute attention scores between every pair of windows (32x32 matrix)

        ʕっʘ‿ʘʔっ("scores")
        #scores = torch.matmul(query, key.T) / math.sqrt(embedDimension)  # shape: (32, 32)
        scores = torch.matmul(query, key.T) / temperature

        ʕっʘ‿ʘʔっ("selfScores & peerScores") # separate self scores (diagonal) and peer scores (off-diagonals)
        selfScores = torch.diag(scores) # self score for window i: scores[i, i] (shape: (32,))
        peerScores = scores.sum(dim=0) - selfScores # peer scores for window j: sum of scores[i, j] for all i != j (shape: (32,))
        ʕっʘ‿ʘʔっ("combinedScores")
        combinedScores = selfScores + peerScores # shape: (32,)
        softCombinedScores = F.softmax(combinedScores, dim=0)
        attentionWindowWeights = softCombinedScores # shape: (32,), sum of weights = 1
        if statPrints or debugPrints: 
            ʕっʘ‿ʘʔっ("♥getWindowEntropy")
            print(f"attentionWindowWeights: {attentionWindowWeights}")
            windowEntropy = -torch.sum(attentionWindowWeights * torch.log(attentionWindowWeights + 1e-12)).item()
            if statPrints or debugPrints: print(windowEntropy)

        ʕっʘ‿ʘʔっ("weightedWindows") # Weight each window's output (summary) by its soft combined scores (attention weight) and sum
        weightedWindows = windowOutputsTensor * attentionWindowWeights.unsqueeze(1)  # shape: (32, numNeurons)
        ʕっʘ‿ʘʔっ("windowContextVector")
        windowContextVector = weightedWindows.sum(dim=0, keepdim=True)   # shape: (1, numNeurons)

        ʕっʘ‿ʘʔっ("finalActions")
        if tinyWindowCount > 0: print(f"saw {neuronActivations.shape[0]} tokens; created {tinyWindowCount} empty windows.")
        

        return windowContextVector
        
LOGITS
def forward(self, meanActivationsTensor):
    with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
        imports the activations from interneuronNetwork, assuming that is is a tensor
        activationsTensor = meanActivationsTensor
        activationsTensor = activationsTensor.to(modelDevice)
        if debugPrints: print(f"Debug logits: activationsTensor shape before @ weights: {activationsTensor.shape}")
        if debugPrints: print(f"Debug logits: weights shape: {self.weights.shape}")
        return logits (not softmax) for better gradient computation in cross-entropy loss
        logitOutput = activationsTensor @ self.weights + self.bias
        return logitOutput

MEMORY
def forward(self, combinedActivationsTensor):
    with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
        ʕっʘ‿ʘʔっ("tensorsToDevice")
        device = self.shortTermMemory.device
        combinedActivationsTensor = combinedActivationsTensor.to(device)

        ʕっʘ‿ʘʔっ("sigmoid gate decays") # make sure decay values stay within [0, 1] range
        shortDecay = torch.sigmoid(self.shortTermDecay) 
        longDecay = torch.sigmoid(self.longTermDecay)

        ʕっʘ‿ʘʔっ("updateMemories") # update memories with learned decay rates
        newShortTermMemory = (shortDecay * self.shortTermMemory) + ((1 - shortDecay) * combinedActivationsTensor)
        newLongTermMemory = (longDecay * self.longTermMemory) + ((1 - longDecay) * combinedActivationsTensor)

        ʕっʘ‿ʘʔっ("copyLongTermMemories")
        oldLongTermMemory = self.longTermMemory.clone()
        newLongTermMemory = (longDecay * oldLongTermMemory) + ((1 - longDecay) * combinedActivationsTensor)
        self.longTermMemory = newLongTermMemory

        ʕっʘ‿ʘʔっ("copyShortTermMemories")
        oldShortTermMemory = self.shortTermMemory.clone()
        newShortTermMemory = (shortDecay * oldShortTermMemory) + ((1 - longDecay) * combinedActivationsTensor)
        self.shortTermMemory = newShortTermMemory

        ʕっʘ‿ʘʔっ("logGateSizes") # log the memory gate sizes
        gateSum = self.shortGate + self.longGate + self.currentGate + 1e-9
        self.latestMemoryGates = torch.stack([
        self.shortGate / gateSum,
        self.longGate / gateSum,
        self.currentGate / gateSum])

        ʕっʘ‿ʘʔっ("blendMemories") # blend memories using weighted sum of the memories, using gates as weights
        blendedActivations = (
            self.shortGate * self.shortTermMemory) + (
            self.longGate * self.longTermMemory) + (
            self.currentGate * combinedActivationsTensor)

        return blendedActivations
    
BABYLLM
def forward(self, inputSeq):
    with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ: # processes input sequence of tokens (str) to generate logits to predict the next token
        if debugPrints: print(f"Debug: Input to forward: {inputSeq}")

        ʕっʘ‿ʘʔっ("inputIndices") # convert inputted tokens to indices (batch processing instead of looping)
        inputIndices = [vocab.tokenToIndex.get(tokenString, vocab.tokenToIndex["<UNK>"]) if not isinstance(tokenString, int) else tokenString for tokenString in inputSeq]

        ʕっʘ‿ʘʔっ("inputEmbeds") # convert indices to embeddings
        inputEmbeds = []
        inputIndicesTensor = torch.tensor(inputIndices, device = modelDevice)
        inputEmbeds = self.embed(inputIndicesTensor)

        ʕっʘ‿ʘʔっ("interneuronNetworkOutput") # PARALLEL NEURON LAYER input/processing (feature extraction)
        interneuronNetworkOutput = self.interneuronNetwork.forward(inputEmbeds) 
        if debugPrints: print(f"Debug BABYLLM.forward: interneuronNetworkOutput length: {len(interneuronNetworkOutput)}") 

        ʕっʘ‿ʘʔっ("combinedActivationsTensor") # RESIZE NEURON LAYER TO STANDARD SIZE FOR COMBINED FORWARD PROCESSING
        #combinedActivationsTensor = torch.mean(interneuronNetworkOutput, dim=0, keepdim=True)
        combinedActivationsTensor = interneuronNetworkOutput
        if debugPrints: print("combinedActivationsTensor.requires_grad:", combinedActivationsTensor.requires_grad)
        if debugPrints: print("combinedActivationsTensor.grad_fn:", combinedActivationsTensor.grad_fn)

        ʕっʘ‿ʘʔっ("memoryLayer") # MEMORY LAYER PROCESSING - NOW PROCESS THE COMBINED ACTIVATIONS
        memoryOutput = self.memory.forward(combinedActivationsTensor)
        latestMemGates = self.memory.latestMemoryGates
        combinedActivations = memoryOutput

        ʕっʘ‿ʘʔっ("logits.forward")
        logits = self.logits.forward(combinedActivations)  

        returns a logits tensor of shape (1, vocabSize) showing predicted probabilities for the next token
        return logits, interneuronNetworkOutput, inputEmbeds, latestMemGates, self.memory.longTermMemory, self.memory.shortTermMemory"""