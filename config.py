#from torch import relu 
from torch.nn.functional import leaky_relu

# BASIC CONFIG
vocabSize = 2000
embedDimension = 32
numNeurons = 10000
epochs = 20
trainingWindow = 8
temperature = 0.7

# OPTIMIZER
optimizerName = "AdamW" # Adam with the weights decoupled, helps avoid erasing learning by overfitting etc.
#optimizerName = "Adam" # good for initial fast training, likely to do overfitting stuff
learningRate = 0.0005

# ACTIVATION FUNCTION
#leaky reLU avoids dead neurons by never forcing them to send a 0 when negative, better for tiny models)
leakyRelu = lambda x: leaky_relu(x, negative_slope=0.01)
#activationFunction = relu
activationFunction = leakyRelu

# VISUALISATIONS
printFreq = 1
printLossFreq = 1000
LIGHT_PURPLE = "\033[94m"  # purple
PURPLE = "\033[38;5;225m" # light purple (yes i know they're marked backwards)
RESET = "\033[0m"    # normal temrinal
BLUE = "\033[34m"    # blue
GOLD = "\033[93m"    # gold
DIM = "\033[2m"      # dimmed terminal
RED = "\033[38;5;124m"
FLASHING_RED = "\033[5;91m"
ORANGE = "\033[38;5;52m"

# visualisation colour boundaries for low and very low loss
lowLoss = 1
veryLowLoss = 0.5

# TRAINING DATA
dataFilepaths = ["data/CHARIS/trainingData.txt"]

rawDataFilepaths = [
    #("text", "data/CHARIS/miniTraining.txt"), # i am happy! i did it! i know it!
    ("text", "data/CHARIS/mixedwrittenanddefs.txt"),
    #("text", "data/CHARIS/shortestwrittenexamples.txt"),
    #("text", "data/CHARIS/shorterwrittenexamples.txt"),
    #("text", "data/CHARIS/writtenexamples.txt"),
    #("text", "data/CHARIS/longerwrittenexamples.txt"),
    #("text", "data/CHARIS/longestwrittenexamples.txt"),
    #("text", "data/CHARIS/DISSERTATIONONAI.txt"),
    #("text", "data/CHARIS/charisGPT.txt"), # weird fake sentences
    #("json", "data/CHARIS/discord.json"), # discord message history
    #("json", "data/CHARIS/CHARIShtmlExtract.txt"), # chatgpt history charis side only
    #("reddit_post", "data/CHARIS/reddit_posts.csv"), # reddit posts
    #("reddit_comment", "data/CHARIS/reddit_comments.csv"), # reddit comments
    #("text", "data/CHARIS/old_fb_messages_extract.txt"), # old account facebook messages charis side only
]

outputFile = "data/CHARIS/trainingData.txt"

# Convert to dictionary format when needed
dataFiles = [{"type": ftype, "in": fname, "out": outputFile} for ftype, fname in rawDataFilepaths]