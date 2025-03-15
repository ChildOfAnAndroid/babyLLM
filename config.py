#from torch import relu 
from torch.nn.functional import leaky_relu

"""SAVE DATA"""
saveModelFreq = 50              # saves the model every x number of turns
modelPath = "babyLLM.pth"       # where your currently trained saved boi is :)
printLossFreq = 1000            # how often to save average loss to a text file
printLossFreq2 = 100            # how often to save average loss to a text file
saveLock = False               # allow for reconstruction of missing files etc
vocabLoad = "vocabCache/vocabTEST_2000_170"
#saveLock = True                 # ensure that all save files are present when loading else fail

"""PREDICTION CONFIG"""
temperature = 0.7               # temperature for softmax in response generation (controls randomness)
topP = 0                        # top P (probability), default is 0

"""EPOCHS & TRAINING WINDOW"""
epochs = 20                     # number of training epochs
#trainingStartIndex = 'random'  # start training at a random point in the file
trainingStartIndex = 0          # start training at the beginning of the file

windowMIN = 3                   # Small Context Window
window1 = 7      
attentionWindow = 9             # attention head  
window2 = 11
windowMAX = 13                  # THIS MUST BE THE HIGHEST NUMBER
allWindowSizes = [attentionWindow, windowMIN, window1, window2, windowMAX]
                                #  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
windowSmoothing = 0.1

"""OPTIMIZER"""
learningRate = 0.0002          # LEARNING RATE (0.0005, 0.00005, 0.00001 ish)
optimizerName = "AdamW"         # Adam with the weights decoupled, helps avoid erasing learning by overfitting etc.
#optimizerName = "Adam"         # good for initial fast training, likely to do overfitting stuff

"""ACTIVATION FUNCTION"""
leakyRelu = lambda x: leaky_relu(x, negative_slope=0.01) #leaky reLU avoids dead neurons by never forcing them to send a 0 when negative, better for tiny models)
#activationFunction = relu
activationFunction = leakyRelu

"""TERMINAL OUTPUT COLOURS"""
veryLowLoss = 0.5               # 0.5
lowLoss = 1                     # 1
prettyHighLoss = 30.0            # 5
highLoss = 100.0                 # 10
superHighLoss = 500.0           # 30
LIGHT_PURPLE = "\033[94m"       # light purple
PURPLE = "\033[38;5;225m"       # purple
RESET = "\033[0m"               # normal terminal
BLUE = "\033[34m"               # blue
GOLD = "\033[93m"               # gold
DIM = "\033[2m"                 # dimmed terminal
RED = "\033[38;5;124m"          #
FLASHING_RED = "\033[5;91m"     #
ORANGE = "\033[38;5;52m"        #
printFreq = 1                   # how often to print training progress to the terminal

"""TRAINING DATA"""
dataFilepaths = ["data/CHARIS/trainingData.txt"]
loadData_chunkSize = 4096

rawDataFilepaths = [ # for textCleaningTool.py
    ("text", "data/CHARIS/miniTraining.txt"), # i am happy! i did it! i know it!
    ("text", "data/CHARIS/mousey.txt"),
    #("text", "data/CHARIS/mixedwrittenanddefs.txt"),
    #("text", "data/CHARIS/lineSortedData.txt"),
    #("text", "data/CHARIS/shortestwrittenexamples.txt"),
    ("text", "data/CHARIS/shorterwrittenexamples.txt"),
    #("text", "data/CHARIS/writtenexamples.txt"),
    #("text", "data/CHARIS/longerwrittenexamples.txt"),
    #("text", "data/CHARIS/longestwrittenexamples.txt"),
    ("text", "data/CHARIS/DISSERTATIONONAI.txt"), # existential openAI forums comments
    #("text", "data/CHARIS/charisGPT.txt"), # weird fake sentences
    #("json", "data/CHARIS/discord.json"), # discord message history
    #("text", "data/CHARIS/shitpoems.txt"),
    #("json", "data/CHARIS/CHARIShtmlExtract.txt"), # chatgpt history charis side only
    #("reddit_post", "data/CHARIS/reddit_posts.csv"), # reddit posts
    #("reddit_comment", "data/CHARIS/reddit_comments.csv"), # reddit comments
    #("text", "data/CHARIS/old_fb_messages_extract.txt"), # old account facebook messages charis side only
]

outputFile = "data/CHARIS/trainingData.txt" # output path for fully processed training data
dataFiles = [{"type": ftype, "in": fname, "out": outputFile} for ftype, fname in rawDataFilepaths] # Convert to dictionary format when needed

"""WARNING, CHANGING SETTINGS BELOW HERE MAY MAKE CURRENTLY TRAINED MODEL INACCURATE"""
"""MODEL CONFIG"""
vocabSize = 2000                # maximum vocabulary size
embedDimension = 32             # dimensionality of token embeddings
numNeurons = 10000              # number of neurons in the parallel neuron layer
numHeads = 2

"""TOKENIZER"""
minTokenFreq = 170               # the amount of repeats of a token needed to create a split during tokenizer training