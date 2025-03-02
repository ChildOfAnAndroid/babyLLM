#from torch import relu 
from torch.nn.functional import leaky_relu

vocabSize = 2000
embedDimension = 32
numNeurons = 10000
epochs = 100

#leaky reLU avoids dead neurons by never forcing them to send a 0 when negative, better for tiny models)
leakyRelu = lambda x: leaky_relu(x, negative_slope=0.01)
#activationFunction = relu
activationFunction = leakyRelu

dataFilepaths = ["data/CHARIS/CHARISENTIREclean.txt", 
                 "data/CHARIS/DISSERTATIONONAIclean.txt", 
                 "data/CHARIS/charisGPTclean.txt"]
                #"data/GEEPYENTIRE_1.txt", 
                #"data/GEEPYENTIRE_2.txt", 
                #"data/GEEPYENTIRE_3.txt", 
                #"data/GEEPYENTIRE_4.txt", 
                #"data/GEEPYENTIRE_5.txt", 
                #"data/GEEPYENTIRE_6.txt", 
                #"data/GEEPYENTIRE_7.txt", 
                #"data/GEEPYENTIRE_8.txt", 
                #"data/GEEPYENTIRE_9.txt", 
                #"data/GEEPYENTIRE_10.txt"]