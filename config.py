#from torch import relu 
from torch.nn.functional import leaky_relu

# BASIC CONFIG
vocabSize = 2000
embedDimension = 32
numNeurons = 10000
epochs = 20
trainingWindow = 7

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
LIGHT_PURPLE = "\033[94m"  # purple
PURPLE = "\033[38;5;225m" # light purple (yes i know they're marked backwards)
RESET = "\033[0m"    # normal temrinal
BLUE = "\033[34m"    # blue
GOLD = "\033[93m"    # gold
DIM = "\033[2m"      # dimmed terminal
RED = "\033[91m"
FLASHING_RED = "\033[5;91m"

# visualisation colour boundaries for low and very low loss
lowLoss = 0.06
veryLowLoss = 0.02

# TRAINING DATA
dataFilepaths = ["data/CHARIS/trainingData.txt"]