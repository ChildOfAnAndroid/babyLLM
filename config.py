# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM CONFIG FILE // config.py

import torch
modelDevice = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#modelDevice = torch.device("cpu")

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
babyName = "babyllm"
scribeName = "scribe"
enemyName = "george"
extraNames = {"kevin", "froggy", "pete", "ace", "elodie"}

"""--- --- --- --- --- DATA & FILEPATHS --- --- --- --- ---"""
"""--- MODEL ---"""
saveModelFreq = 999     # // 500 // 5000 // 10000 // saves the model every x number of turns

saveStrict = True    # // False //~allow reconstruction of missing files // True //~save files must be present, else fail

modelFilePath = "BRAIN/soul/babyllm.pth"     # where your currently trained saved boi is :)
modelBackupFilePath = "BRAIN/soul/babyLLM.pth"     # where your currently trained saved boi is :)

stepCheckpointFilePath = "BRAIN/soul/stepCheckpoint.txt"

"""--- TRAINING ---"""
trainingFilePath = "SCHOOL/library/trainingData.txt"

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
temperature = 0.01     # 0.8 // temperature for softmax in response generation - controls randomness
temperatureIncrement = 0.1 # 0.00001 
minTemp = 0.01 # 0.4
maxTemp = 1.4

topP = 0     # top P - probability
numTokensPerStep = 32     # Number of tokens to predict per step
tokenIncrement = 0.0001
inferenceOutputNumTokens = 40

"""memoryLayer"""
memoryLength = 5
memoryLengthIncrement = 0.0001

"""optimizer"""
learningRate = 0.00035     # // 0.0005 // 0.00005 // 0.00001 //
optimizerName = "AdamW"     # // "AdamW" //~decoupled weights adam, helps avoid erasing learning by overfitting etc. // "Adam" //~good for initial fast training, likely to do overfitting stuff
activationFunction = leakyRelu       # // leakyRely // relu //

gradientClipMaxNorm = 1.0
gradClipIncrement = 0.00001
minGradClip = 0.6
maxGradClip = 1.4

"""scheduled sampling"""
scheduledSampling = True 
scheduledSamplingRate = 0
scheduledSamplingIncrement = 0.00001     # // 0.0001 //~increment probability of using model output by this much PER TURN
minSchedSamp = -0.01
maxSchedSamp = 1.0

"""repetition penalty"""
penaltyWindow = 16          # how many tokens to look back for repetition
repetitionPenalty = 1.3
repetitionPenaltyIncrement = -0.00001
minRepPen = 0.01
maxRepPen = 1.4
windowEntropyBonus = True

"""--- TRAINING ---"""
trainingDataSliceSize_min = 10
trainingDataSliceSize_max = 15000
trainingStartIndex = 0     # // 'random' (not in babyLLM.py)
epochs = 20
#retokenizeOnLoad = False
#saveTokenizedData = True

"""--- LOGS ---"""
trainingLogFreq_1000 = 10    # creates logs every x number of turns
trainingLogFreq_100 = 100     # creates logs every x number of turns

printFreq = 5     # how often to print training progress to the terminal
printPromptLength = 90     # how many characters of the prompt to display in terminal

durationLogging = False     # // True // False // activates debug time logging
debugPrints = False
anomalyDetect = False

skipNeuron = False
skipINN = True # THIS IS WHERE THE SLOWDOWN IS!!!!!
skipINNparliament = False
skipMemory = False
skipWobble = True

skipComputeLoss = False

debugPrints_babyLLM = False
debugPrints_TUTOR = False
durationLogging_babyLLM = False
durationLogging_TUTOR = False

"""--- STATS COLLECTION ---"""
profiler = False
mpsProfiler = False
forwardProfiler = False

collectStats = True
static_collectStats = True

embed_collectStats = True
token_collectStats = True 

logit_collectStats = True

n_collectStats = True
INN_collectStats = True

memory_collectStats = True

# neuron + interneuronNetwork
n_weightStats = True
n_weightNormStats = True
n_biasesStats = True
n_sparsityStat = True

INN_cerebellumStats = True
INN_credibilityBiasStats = False
INN_judgeBiasStats = False
INN_scoringStats = False
INN_windowStats = True
INN_outputTensorStats = True

"""--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- """


"""--- --- --- --- --- TRAINING DATA & SORTING --- --- --- --- ---"""
rawDataFilepaths = [     # for textCleaningTool.py

    #-*- CHARIS STUDIES -*-
    #--- CHAT HISTORY ---
    ("discord_json", "SCHOOL/library/charisStudies/discord.json", 10),     # discord message history
    ("reddit_comment", "SCHOOL/library/charisStudies/reddit_comments.csv", 5),     # reddit comments
    ("text", "SCHOOL/library/charisStudies/shitpoems.txt", 10),     #  random poems from my notes on my phone
    ("reddit_post", "SCHOOL/library/charisStudies/reddit_posts.csv", 5),     # reddit posts
    ("json", "SCHOOL/library/charisStudies/charisGPThistory.txt", 10),     # chatgpt history charis side only
    ("text", "SCHOOL/library/charisStudies/old_fb_messages_extract.txt", 10),     # old account facebook messages charis side only
    ("text", "SCHOOL/library/charisStudies/essays.txt", 5),     # essays
    ("text", "SCHOOL/library/charisStudies/tindieBaby.txt", 5),     # tindie blog posts

    #--- MOUSE ADVENTURES ---
    ("text", "SCHOOL/library/mouseAdventure/elodieMousey.txt", 5),     #  elodies wonderful mouse story!
    ("text", "SCHOOL/library/mouseAdventure/mousey.txt", 5),     #  my simple version of elodies mouse story!
    ("text", "SCHOOL/library/mouseAdventure/elodieMouseyLonger.txt", 10),     #  even more of elodies lovely mouse story!

    #--- MINI TRAINING ---
    ("text", "SCHOOL/library/miniTraining/miniTraining.txt", 10),     # i am happy! i did it! i know it!
    ("text", "SCHOOL/library/miniTraining/miniTraining2.txt", 10),     # training: i am happy! i did it! i know it!

    #--- BABYLLM CHAT LOGS ---
    ("text", chatLogPath_trainingLog, 1),     # log: 'what am i learning today?'
    ("text", chatLogPath_talkToYourself, 1),     #  i answer my own previous chat messages
    ("text", chatLogPath_infer, 1),     # log: babyLLM infer.py history!
    ("text", chatLogPath_talkToYourselfComparisons, 1),     # log: comparing babyllms answers to my answers

    #--- TENSES ---
    ("text", "SCHOOL/library/tenses/presentTense.txt", 1),     #  tense: present (kevin's weed theme?)
    ("text", "SCHOOL/library/tenses/pastTense.txt", 1),     # tense: past (mouse theme!)

    ("text", "SCHOOL/library/tenses/presentTense copy.txt", 1),     # tense
    ("text", "SCHOOL/library/tenses/futureContinuousTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/futurePerfectContinuousTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/futurePerfectTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/pastModalCouldHave.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/pastModalMustHaveTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/pastModalShouldHave.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/pastModalWouldHaveTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/pastPerfectContinuousTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/pastPerfectTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/presentContinuousTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/presentModalCanTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/presentModalCouldTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/presentModalMustTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/presentModalShouldTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/presentPerfectContinuousTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/presentPerfectTense.txt", 1),     #  tense
    ("text", "SCHOOL/library/tenses/futureTense.txt", 1),     #  tense: future
    ("text", "SCHOOL/library/tenses/presentConditionalTense.txt", 1),     # tense: present conditional
    ("text", "SCHOOL/library/tenses/pastContinuousTense.txt", 1),     #  tense: past continuous
    ("text", "SCHOOL/library/tenses/imperativeTense.txt", 1),     #  tense

    #--- SIMPLE TRAINING ---
    ("text", "SCHOOL/library/simpleTraining/cursed.txt", 1),     # training but chaotic shuffle
    ("text", "SCHOOL/library/simpleTraining/geepyGenerated.txt", 1),     # weird fake sentences
    ("text", "SCHOOL/library/simpleTraining/sampleshorterwrittenexamples.txt", 1),     #  training
    ("text", "SCHOOL/library/simpleTraining/shortestwrittenexamples.txt", 1),     #  training
    ("text", "SCHOOL/library/simpleTraining/shorterwrittenexamples.txt", 1),     #  training
    ("text", "SCHOOL/library/simpleTraining/longerwrittenexamples.txt", 1),     #  training
    ("text", "SCHOOL/library/simpleTraining/lineSortedData.txt", 1),     #  training
    ("text", "SCHOOL/library/simpleTraining/longestwrittenexamples.txt", 1),     #  training
    ("text", "SCHOOL/library/simpleTraining/mixedwrittenanddefs.txt", 1),     # training
    ("text", "SCHOOL/library/simpleTraining/writtenexamples.txt", 1),     #  training
    ("text", "SCHOOL/library/simpleTraining/variedWrittenExamples.txt", 1),     #  training
    ("text", "SCHOOL/library/charisStudies/thames.txt", 1),
    ("text", "SCHOOL/library/charisStudies/weirdMixedStuff.txt", 1),

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

trainingFilePath_arr = ["SCHOOL/library/trainingData.txt"]
tokenizedDataPath = "SCHOOL/tokenizedTrainingData.txt"

trainingFilePath_dict_weighted = []
for entry in trainingFilePath_dict:
    trainingFilePath_dict_weighted.extend([entry] * entry["weight"]) # each one times the number in its weight

trainingFileWeightTotal = sum([entry[2] for entry in rawDataFilepaths if len(entry) == 3])