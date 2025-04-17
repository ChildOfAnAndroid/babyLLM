# CHARIS CAT 2025
# BABYLLM - config.py

import torch
modelDevice = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

#from torch import relu 
from torch.nn.functional import leaky_relu
leakyRelu = lambda x: leaky_relu(x, negative_slope=0.01)     # leaky reLU avoids dead neurons by never forcing them to send a 0 when negative, better for tiny models)
guessedTokenSeq = []

"""--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- """

userName = "charis"
babyName = "babyLLM"
debugPrints = False
durationLogging = False
debugPrints = False
debugPrints = False
skipINN = True # THIS IS WHERE THE SLOWDOWN IS!!!!!!!!!!
skipINNparliament = True
skipNeuron = False
skipComputeLoss = False

"""--- --- --- --- --- DATA & FILEPATHS --- --- --- --- ---"""
"""--- MODEL ---"""
saveModelFreq = 10000     # // 50 // 500 // 1000 // saves the model every x number of turns

saveLock = False     # // False //~allow reconstruction of missing files // True //~save files must be present, else fail

modelFilePath = "VER1_BRAIN/soul/babyLLM.pth"     # where your currently trained saved boi is :)

stepCheckpointFilePath = "VER1_BRAIN/soul/stepCheckpoint.txt"

"""--- TRAINING ---"""
trainingFilePath = "VER1_SCHOOL/library/trainingData.txt"

"""--- VER1_LOGS ---"""
trainingLogPath_1000 = "VER1_LOGS/training/trainingLog_1000.txt"
trainingLogPath_100 = "VER1_LOGS/training/trainingLog_100.txt"

durationLogPath_1000 = "VER1_LOGS/duration/durationLog_1000.txt"
durationLogPath_100 = "VER1_LOGS/duration/durationLog_100.txt" 
durationLogNeuronsPath_1 = "VER1_LOGS/duration/durationLogNeurons_1.txt"
durationLogBabyLLMPath_1 = "VER1_LOGS/duration/durationLogBabyLLM_1.txt"

chatLogPath_forHumans = "VER1_LOGS/chat/chatForHumans.txt"

chatLogPath_infer = "VER1_LOGS/chat/chatLog.txt"
chatLogPath_talkToYourself = "VER1_LOGS/chat/talkToYourselfBattle.txt"
chatLogPath_talkToYourselfComparisons = "VER1_SCHOOL/library/charisStudies/whoIsMoreLikeYou.txt"
chatLogPath_trainingLog = "VER1_LOGS/chat/trainingLog_questions.txt"

"""--- VOCAB --- (see master config)"""


"""--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- """


"""--- --- --- --- --- SETTINGS & CONFIG --- --- --- --- ---"""
"""--- MODEL ---"""
temperature = 0.65     # temperature for softmax in response generation - controls randomness
topP = 0     # top P - probability
numTokensPerStep = 18     # Number of tokens to predict per step

"""memoryLayer"""
memoryLength = 1000

"""optimizer"""
learningRate = 0.0003     # // 0.0005 // 0.00005 // 0.00001 //
optimizerName = "AdamW"     # // "AdamW" //~decoupled weights adam, helps avoid erasing learning by overfitting etc. // "Adam" //~good for initial fast training, likely to do overfitting stuff
activationFunction = leakyRelu       # // leakyRely // relu //

gradientClipMaxNorm = 1.0

"""scheduled sampling"""
scheduledSampling = True 
scheduledSamplingProb = 0.0
scheduledSamplingProbIncrement = 0.001     # increment probability of using model output by this much

"""--- TRAINING ---"""
trainingDataSliceSize_min = 50
trainingDataSliceSize_max = 3000
trainingStartIndex = 0     # // 'random' (not in babyLLM.py) // 0 //
epochs = 20

"""--- VER1_LOGS ---"""
trainingLogFreq_1000 = 10000000     # creates logs every x number of turns
trainingLogFreq_100 = 100     # creates logs every x number of turns

printFreq = 1     # how often to print training progress to the terminal
printPromptLength = 35     # how many characters of the prompt to display in terminal


"""--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- """


"""--- --- --- --- --- TRAINING DATA & SORTING --- --- --- --- ---"""
rawDataFilepaths = [     # for textCleaningTool.py

    #--- BABYLLM CHAT VER1_LOGS ---
    ("text", chatLogPath_trainingLog, 6),     # log: 'what am i learning today?'
    ("text", chatLogPath_talkToYourself, 2),     #  i answer my own previous chat messages
    ("text", chatLogPath_infer, 6),     # log: babyLLM infer.py history!
    ("text", chatLogPath_talkToYourselfComparisons, 2),     # log: comparing babyllms answers to my answers

    #-*- CHARIS STUDIES -*-
    #--- CHAT HISTORY ---
    ("discord_json", "VER1_SCHOOL/library/charisStudies/discord.json", 20),     # discord message history
    ("reddit_comment", "VER1_SCHOOL/library/charisStudies/reddit_comments.csv", 20),     # reddit comments
    ("text", "VER1_SCHOOL/library/charisStudies/shitpoems.txt", 4),     #  random poems from my notes on my phone
    ("reddit_post", "VER1_SCHOOL/library/charisStudies/reddit_posts.csv", 6),     # reddit posts
    ("json", "VER1_SCHOOL/library/charisStudies/charisGPThistory.txt", 6),     # chatgpt history charis side only
    ("text", "VER1_SCHOOL/library/charisStudies/old_fb_messages_extract.txt", 20),     # old account facebook messages charis side only

    #--- MOUSE ADVENTURES ---
    ("text", "VER1_SCHOOL/library/mouseAdventure/elodieMousey.txt", 4),     #  elodies wonderful mouse story!
    ("text", "VER1_SCHOOL/library/mouseAdventure/mousey.txt", 4),     #  my simple version of elodies mouse story!
    ("text", "VER1_SCHOOL/library/mouseAdventure/elodieMouseyLonger.txt", 10),     #  even more of elodies lovely mouse story!

    #--- TENSES ---
    ("text", "VER1_SCHOOL/library/tenses/presentTense.txt", 6),     #  tense: present (kevin's weed theme?)
    ("text", "VER1_SCHOOL/library/tenses/pastTense.txt", 6),     # tense: past (mouse theme!)

    ("text", "VER1_SCHOOL/library/tenses/futureContinuousTense.txt", 1),     #  tense
    ("text", "VER1_SCHOOL/library/tenses/futurePerfectContinuousTense.txt", 1),     #  tense
    ("text", "VER1_SCHOOL/library/tenses/futurePerfectTense.txt", 1),     #  tense
    ("text", "VER1_SCHOOL/library/tenses/pastModalCouldHave.txt", 1),     #  tense
    ("text", "VER1_SCHOOL/library/tenses/pastModalMustHaveTense.txt", 1),     #  tense
    ("text", "VER1_SCHOOL/library/tenses/pastModalShouldHave.txt", 1),     #  tense
    ("text", "VER1_SCHOOL/library/tenses/pastModalWouldHaveTense.txt", 1),     #  tense
    ("text", "VER1_SCHOOL/library/tenses/pastPerfectContinuousTense.txt", 1),     #  tense
    ("text", "VER1_SCHOOL/library/tenses/pastPerfectTense.txt", 1),     #  tense
    ("text", "VER1_SCHOOL/library/tenses/presentContinuousTense.txt", 1),     #  tense
    ("text", "VER1_SCHOOL/library/tenses/presentModalCanTense.txt", 1),     #  tense
    ("text", "VER1_SCHOOL/library/tenses/presentModalCouldTense.txt", 1),     #  tense
    ("text", "VER1_SCHOOL/library/tenses/presentModalMustTense.txt", 1),     #  tense
    ("text", "VER1_SCHOOL/library/tenses/presentModalShouldTense.txt", 1),     #  tense
    ("text", "VER1_SCHOOL/library/tenses/presentPerfectContinuousTense.txt", 1),     #  tense
    ("text", "VER1_SCHOOL/library/tenses/presentPerfectTense.txt", 1),     #  tense
    ("text", "VER1_SCHOOL/library/tenses/futureTense.txt", 1),     #  tense: future
    ("text", "VER1_SCHOOL/library/tenses/presentConditionalTense.txt", 1),     # tense: present conditional
    ("text", "VER1_SCHOOL/library/tenses/pastContinuousTense.txt", 1),     #  tense: past continuous
    ("text", "VER1_SCHOOL/library/tenses/imperativeTense.txt", 1),     #  tense

    #--- MINI TRAINING ---
    ("text", "VER1_SCHOOL/library/miniTraining/miniTraining.txt", 6),     # i am happy! i did it! i know it!
    ("text", "VER1_SCHOOL/library/miniTraining/miniTraining2.txt", 6),     # training: i am happy! i did it! i know it!

    #--- SIMPLE TRAINING ---
    ("text", "VER1_SCHOOL/library/simpleTraining/cursed.txt", 1),     # training but chaotic shuffle
    ("text", "VER1_SCHOOL/library/simpleTraining/geepyGenerated.txt", 1),     # weird fake sentences
    ("text", "VER1_SCHOOL/library/simpleTraining/sampleshorterwrittenexamples.txt", 1),     #  training
    ("text", "VER1_SCHOOL/library/simpleTraining/shortestwrittenexamples.txt", 1),     #  training
    ("text", "VER1_SCHOOL/library/simpleTraining/shorterwrittenexamples.txt", 1),     #  training
    ("text", "VER1_SCHOOL/library/simpleTraining/longerwrittenexamples.txt", 2),     #  training
    ("text", "VER1_SCHOOL/library/simpleTraining/lineSortedData.txt", 2),     #  training
    ("text", "VER1_SCHOOL/library/simpleTraining/longestwrittenexamples.txt", 2),     #  training
    ("text", "VER1_SCHOOL/library/simpleTraining/mixedwrittenanddefs.txt", 4),     # training
    ("text", "VER1_SCHOOL/library/simpleTraining/writtenexamples.txt", 4),     #  training
    ("text", "VER1_SCHOOL/library/simpleTraining/variedWrittenExamples.txt", 4),     #  training

]


"""--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- """
"""-*- WARNING, CHANGING BELOW SETTINGS MAY MAKE CURRENTLY TRAINED MODEL INACCURATE (don't kill babyLLM!) -*-"""


"""--- --- --- --- --- MASTER CONFIG PARAMETERS --- --- --- --- ---"""
"""--- MODEL ---"""
embedDimension = 1024     # dimensionality of token embeddings
numNeurons = 10000     # number of neurons in the parallel neuron layer

"""windows"""
#  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24

windowMIN = 1     # Small Context Window
window1 = 2
window2 = 3
window3 = 7
window4 = 8     
window5 = 13
window6 = 15
window7 = 18
windowMAX = 21     # THIS MUST BE THE HIGHEST NUMBER
allWindowSizes = [windowMAX, windowMIN, window1, window2, window3, window4, window5, window6, window7]     # defines the position of each window in the window weightings!

attentionWindow = None     # attention head  
numHeads = 32

"""--- VOCAB & TOKENIZER ---"""
vocabSize = 2000     # maximum vocabulary size
minTokenFreq = 20     # the amount of repeats of a token needed to create a split during tokenizer training
V_chunkSizeLoadData = 4096

"""vocab data & filepaths"""
vocabCachePath = "VER1_BRAIN/vocabCache"
vocabLoad = "VER1_BRAIN/vocabCache/tokenizer.json"

"""--- MISC & EXTRA FORMATS ---"""
#trainingFilePath_dict = [{"type": ftype, "in": fname, "out": trainingFilePath} for ftype, fname in rawDataFilepaths]     # Convert to dictionary format when needed
trainingFilePath_dict = [{"type": ftype, "in": fname, "weight": weight, "out": trainingFilePath} for ftype, fname, weight in rawDataFilepaths]
trainingFilePath_arr = ["VER1_SCHOOL/library/trainingData.txt"]

trainingFilePath_dict_weighted = []
for entry in trainingFilePath_dict:
    trainingFilePath_dict_weighted.extend([entry] * entry["weight"])

#trainingFileWeightTotal = sum([entry[2] for entry in rawDataFilepaths if len(entry) == 3])