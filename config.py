#from torch import relu 
from torch.nn.functional import leaky_relu

"""BASIC CONFIG"""
vocabSize = 2000 # maximum vocabulary size
embedDimension = 32 # dimensionality of token embeddings
numNeurons = 10000 # number of neurons in the parallel neuron layer
epochs = 20 # number of training epochs
trainingWindow = 7 # context window size (number of input tokens) for training - 8 is too high rn
temperature = 0.7 # temperature for softmax in response generation (controls randomness)
saveModelFreq = 250 # saves the model every x number of turns
topP = 0 # top P (probability), default is 0

"""OPTIMIZER"""
optimizerName = "AdamW" # Adam with the weights decoupled, helps avoid erasing learning by overfitting etc.
#optimizerName = "Adam" # good for initial fast training, likely to do overfitting stuff
learningRate = 0.0002

"""ACTIVATION FUNCTION"""
#leaky reLU avoids dead neurons by never forcing them to send a 0 when negative, better for tiny models)
leakyRelu = lambda x: leaky_relu(x, negative_slope=0.01)
#activationFunction = relu
activationFunction = leakyRelu

"""MULTI WINDOW CONTEXT SIZES"""
window1 = 3
window2 = trainingWindow
window3 = 11

"""VISUALISATIONS"""
printFreq = 1 # how often to print training progress to the terminal
printLossFreq = 1000 # how often to save average loss to a text file
LIGHT_PURPLE = "\033[94m"  # light purple
PURPLE = "\033[38;5;225m" # purple
RESET = "\033[0m"    # normal terminal
BLUE = "\033[34m"    # blue
GOLD = "\033[93m"    # gold
DIM = "\033[2m"      # dimmed terminal
RED = "\033[38;5;124m"
FLASHING_RED = "\033[5;91m"
ORANGE = "\033[38;5;52m"

"""visualisation colour boundaries"""
lowLoss = 1
veryLowLoss = 0.5
prettyHighLoss = 5.0
highLoss = 10.0
superHighLoss = 30.0

"""TOKENIZER"""
minTokenFreq = 20 # the amount of repeats of a token needed to create a split during tokenizer training

"""TRAINING DATA"""
dataFilepaths = ["data/CHARIS/trainingData.txt"]
trainingStartIndex = 'random' # start training at a random point in the file
#trainingStartIndex = 0 # start training at the beginning of the file
loadData_chunkSize = 4096

rawDataFilepaths = [
    #("text", "data/CHARIS/miniTraining.txt"), # i am happy! i did it! i know it!
    #("text", "data/CHARIS/mixedwrittenanddefs.txt"),
    #("text", "data/CHARIS/lineSortedData.txt"),
    #("text", "data/CHARIS/shortestwrittenexamples.txt"),
    #("text", "data/CHARIS/shorterwrittenexamples.txt"),
    #("text", "data/CHARIS/writtenexamples.txt"),
    #("text", "data/CHARIS/longerwrittenexamples.txt"),
    #("text", "data/CHARIS/longestwrittenexamples.txt"),
    #("text", "data/CHARIS/DISSERTATIONONAI.txt"), # existential openAI forums comments
    #("text", "data/CHARIS/charisGPT.txt"), # weird fake sentences
    #("json", "data/CHARIS/discord.json"), # discord message history
    ("text", "data/CHARIS/shitpoems.txt"),
    #("json", "data/CHARIS/CHARIShtmlExtract.txt"), # chatgpt history charis side only
    #("reddit_post", "data/CHARIS/reddit_posts.csv"), # reddit posts
    #("reddit_comment", "data/CHARIS/reddit_comments.csv"), # reddit comments
    #("text", "data/CHARIS/old_fb_messages_extract.txt"), # old account facebook messages charis side only
]

outputFile = "data/CHARIS/trainingData.txt" # output path for fully processed training data

dataFiles = [{"type": ftype, "in": fname, "out": outputFile} for ftype, fname in rawDataFilepaths] # Convert to dictionary format when needed

modelPath = "babyLLM.pth" # where your currently trained saved boi is :)