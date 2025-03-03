#from torch import relu 
from torch.nn.functional import leaky_relu

vocabSize = 2000
embedDimension = 32
numNeurons = 10000
epochs = 2
trainingWindow = 5

#leaky reLU avoids dead neurons by never forcing them to send a 0 when negative, better for tiny models)
leakyRelu = lambda x: leaky_relu(x, negative_slope=0.01)
#activationFunction = relu
activationFunction = leakyRelu

dataFilepaths = ["data/CHARIS/trainingData.txt"]