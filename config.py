# CHARIS CAT 2025
# BABYLLM - config.py

import torch
modelDevice = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

#from torch import relu 
from torch.nn.functional import leaky_relu
leakyRelu = lambda x: leaky_relu(x, negative_slope=0.01)     # leaky reLU avoids dead neurons by never forcing them to send a 0 when negative, better for tiny models)
guessedTokenSeq = []
"""if activationFunction == 'leaky_relu':
            output = F.leaky_relu(output, 0.01)
        elif activationFunction == 'relu':
            output = F.relu(output)
        elif activationFunction == 'sigmoid':
            output = torch.sigmoid(output)
        elif activationFunction == 'tanh':
            output = torch.tanh(output)
        elif callable(activationFunction):
            output = activationFunction(output)"""

"""--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- """

userName = "charis"
babyName = "babyLLM"

"""--- --- --- --- --- DATA & FILEPATHS --- --- --- --- ---"""
"""--- MODEL ---"""
saveModelFreq = 10000     # // 500 // 5000 // 10000 // saves the model every x number of turns

saveStrict = True    # // False //~allow reconstruction of missing files // True //~save files must be present, else fail

modelFilePath = "BRAIN/soul/babyLLM_legacy_x.pth"     # where your currently trained saved boi is :)
modelBackupFilePath = "BRAIN/soul/babyLLM.pth"     # where your currently trained saved boi is :)

stepCheckpointFilePath = "BRAIN/soul/stepCheckpoint.txt"

"""--- TRAINING ---"""
trainingFilePath = "SCHOOL/trainingData.txt"
tokenizedDataPath = "SCHOOL/tokenizedTrainingData.txt"

"""--- LOGS ---"""
trainingLogPath_1000 = "SCHOOL/statistics/LOGS/training/trainingLog_1000.txt"
trainingLogPath_100 = "SCHOOL/statistics/LOGS/training/trainingLog_100.txt"

durationLogPath_1000 = "SCHOOL/statistics/LOGS/duration/durationLog_1000.txt"
durationLogPath_100 = "SCHOOL/statistics/LOGS/duration/durationLog_100.txt" 
durationLogNeuronsPath_1 = "SCHOOL/statistics/LOGS/duration/durationLogNeurons_1.txt"
durationLogBabyLLMPath_1 = "SCHOOL/statistics/LOGS/duration/durationLogBabyLLM_1.txt"

chatLogPath_forHumans = "SCHOOL/statistics/LOGS/chat/chatForHumans.txt"

chatLogPath_infer = "SCHOOL/statistics/LOGS/chat/chatLog.txt"
chatLogPath_talkToYourself = "SCHOOL/statistics/LOGS/chat/talkToYourselfBattle.txt"
chatLogPath_talkToYourselfComparisons = "SCHOOL/library/charisStudies/whoIsMoreLikeYou.txt"
chatLogPath_trainingLog = "SCHOOL/statistics/LOGS/chat/trainingLog_questions.txt"

"""--- VOCAB --- (see master config)"""


"""--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- """


"""--- --- --- --- --- SETTINGS & CONFIG --- --- --- --- ---"""
"""--- MODEL ---"""
temperature = 0.90     # temperature for softmax in response generation - controls randomness
topP = 0     # top P - probability
numTokensPerStep = 18     # Number of tokens to predict per step
inferenceOutputNumTokens = 40

"""memoryLayer"""
memoryLength = 1000

"""optimizer"""
learningRate = 0.0003     # // 0.0005 // 0.00005 // 0.00001 //
optimizerName = "AdamW"     # // "AdamW" //~decoupled weights adam, helps avoid erasing learning by overfitting etc. // "Adam" //~good for initial fast training, likely to do overfitting stuff
activationFunction = leakyRelu       # // leakyRely // relu //

gradientClipMaxNorm = 1.0

"""scheduled sampling"""
scheduledSampling = True 
scheduledSamplingProbIncrement = 0.00000001     # // 0.0001 // increment probability of using model output by this much PER TURN

"""--- TRAINING ---"""
trainingDataSliceSize_min = 50
trainingDataSliceSize_max = 5000
trainingStartIndex = 0     # // 'random' (not in babyLLM.py) // 0 //
epochs = 20
#retokenizeOnLoad = False
#saveTokenizedData = True

"""--- LOGS ---"""
trainingLogFreq_1000 = 500     # creates logs every x number of turns
trainingLogFreq_100 = 50     # creates logs every x number of turns

printFreq = 1
     # how often to print training progress to the terminal
printPromptLength = 50     # how many characters of the prompt to display in terminal

durationLogging = False     # // True // False // activates debug time logging
debugPrints = False
statPrints = False
lossPrints = False
logitPrints = False

skipNeuron = False
skipINN = False
skipMemory = False

debugPrints_babyLLM = False
debugPrints_TUTOR = False
durationLogging_babyLLM = False
durationLogging_TUTOR = False

"""--- STATS COLLECTION ---"""
collectStats = True
n_collectStats = True
INN_collectStats = True

# neuron + interneuronNetwork
n_weightStats = True
n_weightNormStats = True
n_biasesStats = True
n_sparsityStat = True
INN_cerebellumStats = True
INN_credibilityBiasStats = True
INN_judgeBiasStats = True
INN_scoringStats = True
INN_windowStats = True
INN_outputTensorStats = True

"""--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- """


"""--- --- --- --- --- TRAINING DATA & SORTING --- --- --- --- ---"""
rawDataFilepaths = [     # for textCleaningTool.py

    #--- BABYLLM CHAT LOGS ---
    ("text", chatLogPath_trainingLog, 1),     # log: 'what am i learning today?'
    ("text", chatLogPath_talkToYourself, 1),     #  i answer my own previous chat messages
    ("text", chatLogPath_infer, 1),     # log: babyLLM infer.py history!
    ("text", chatLogPath_talkToYourselfComparisons, 1),     # log: comparing babyllms answers to my answers

    #-*- CHARIS STUDIES -*-
    #--- CHAT HISTORY ---
    ("discord_json", "SCHOOL/library/charisStudies/discord.json", 5),     # discord message history
    ("reddit_comment", "SCHOOL/library/charisStudies/reddit_comments.csv", 1),     # reddit comments
    ("text", "SCHOOL/library/charisStudies/shitpoems.txt", 3),     #  random poems from my notes on my phone
    ("reddit_post", "SCHOOL/library/charisStudies/reddit_posts.csv", 2),     # reddit posts
    ("json", "SCHOOL/library/charisStudies/charisGPThistory.txt", 2),     # chatgpt history charis side only
    ("text", "SCHOOL/library/charisStudies/old_fb_messages_extract.txt", 3),     # old account facebook messages charis side only
    ("text", "SCHOOL/library/charisStudies/essays.txt", 6),     # essays
    ("text", "SCHOOL/library/charisStudies/tindieBaby.txt", 6),     # tindie blog posts

    #--- MOUSE ADVENTURES ---
    ("text", "SCHOOL/library/mouseAdventure/elodieMousey.txt", 6),     #  elodies wonderful mouse story!
    ("text", "SCHOOL/library/mouseAdventure/mousey.txt", 3),     #  my simple version of elodies mouse story!
    ("text", "SCHOOL/library/mouseAdventure/elodieMouseyLonger.txt", 4),     #  even more of elodies lovely mouse story!

    #--- TENSES ---
    ("text", "SCHOOL/library/tenses/presentTense.txt", 0),     #  tense: present (kevin's weed theme?)
    ("text", "SCHOOL/library/tenses/pastTense.txt", 1),     # tense: past (mouse theme!)

    ("text", "SCHOOL/library/tenses/presentTense copy.txt", 1),     # tense
    ("text", "SCHOOL/library/tenses/futureContinuousTense.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/futurePerfectContinuousTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/futurePerfectTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/pastModalCouldHave.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/pastModalMustHaveTense.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/pastModalShouldHave.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/pastModalWouldHaveTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/pastPerfectContinuousTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/pastPerfectTense.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/presentContinuousTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/presentModalCanTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/presentModalCouldTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/presentModalMustTense.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/presentModalShouldTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/presentPerfectContinuousTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/presentPerfectTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/futureTense.txt", 0),     #  tense: future
    ("text", "SCHOOL/library/tenses/presentConditionalTense.txt", 1),     # tense: present conditional
    ("text", "SCHOOL/library/tenses/pastContinuousTense.txt", 1),     #  tense: past continuous
    ("text", "SCHOOL/library/tenses/imperativeTense.txt", 1),     #  tense

    #--- MINI TRAINING ---
    ("text", "SCHOOL/library/miniTraining/miniTraining.txt", 5),     # i am happy! i did it! i know it!
    ("text", "SCHOOL/library/miniTraining/miniTraining2.txt", 5),     # training: i am happy! i did it! i know it!

    #--- SIMPLE TRAINING ---
    ("text", "SCHOOL/library/simpleTraining/cursed.txt", 1),     # training but chaotic shuffle
    ("text", "SCHOOL/library/simpleTraining/geepyGenerated.txt", 1),     # weird fake sentences
    ("text", "SCHOOL/library/simpleTraining/sampleshorterwrittenexamples.txt", 3),     #  training
    ("text", "SCHOOL/library/simpleTraining/shortestwrittenexamples.txt", 1),     #  training
    ("text", "SCHOOL/library/simpleTraining/shorterwrittenexamples.txt", 0),     #  training
    ("text", "SCHOOL/library/simpleTraining/longerwrittenexamples.txt", 2),     #  training
    ("text", "SCHOOL/library/simpleTraining/lineSortedData.txt", 2),     #  training
    ("text", "SCHOOL/library/simpleTraining/longestwrittenexamples.txt", 2),     #  training
    ("text", "SCHOOL/library/simpleTraining/mixedwrittenanddefs.txt", 2),     # training
    ("text", "SCHOOL/library/simpleTraining/writtenexamples.txt", 2),     #  training
    ("text", "SCHOOL/library/simpleTraining/variedWrittenExamples.txt", 3),     #  training

]

"""--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- """
"""-*- WARNING, CHANGING BELOW SETTINGS MAY MAKE CURRENTLY TRAINED MODEL INACCURATE (don't kill babyLLM!) -*-"""


"""--- --- --- --- --- MASTER CONFIG PARAMETERS --- --- --- --- ---"""
"""--- MODEL ---"""
embedDimension = 1024     # dimensionality of token embeddings
numNeurons = 10000     # number of neurons in the parallel neuron layer

"""windows"""
#  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24

windowMIN = 2     # Small Context Window
window1 = 4
window2 = 8
window3 = 12
window4 = 16     
window5 = 20
window6 = 24
window7 = 28
windowMAX = 32     # THIS MUST BE THE HIGHEST NUMBER
allWindowSizes_new = [windowMAX, windowMIN, window1, window2, window3, window4, window5, window6, window7]     # defines the position of each window in the window weightings!
#allWindowSizes = list(range(1, 33))

attentionWindow = None     # attention head  
numHeads = 32

"""--- VOCAB & TOKENIZER ---"""
vocabSize = 2000     # maximum vocabulary size
minTokenFreq = 20     # the amount of repeats of a token needed to create a split during tokenizer training
V_chunkSizeLoadData = 4096

"""vocab data & filepaths"""
vocabCachePath = "BRAIN/vocabCache"
vocabLoad = "BRAIN/vocabCache/tokenizer.json"

"""--- MISC & EXTRA FORMATS ---"""
#trainingFilePath_dict = [{"type": ftype, "in": fname, "out": trainingFilePath} for ftype, fname in rawDataFilepaths]     # Convert to dictionary format when needed
trainingFilePath_dict = [{"type": ftype, "in": fname, "weight": weight, "out": trainingFilePath} for ftype, fname, weight in rawDataFilepaths]

trainingFilePath_arr = ["SCHOOL/trainingData.txt"]
tokenizedDataPath = "SCHOOL/tokenizedTrainingData.txt"

trainingFilePath_dict_weighted = []
for entry in trainingFilePath_dict:
    trainingFilePath_dict_weighted.extend([entry] * entry["weight"])

trainingFileWeightTotal = sum([entry[2] for entry in rawDataFilepaths if len(entry) == 3])