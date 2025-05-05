# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM CONFIG FILE // config.py

import torch
modelDevice = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#modelDevice = torch.device("cpu")

#from torch import relu 
from torch.nn.functional import leaky_relu
leakyRelu = lambda x: leaky_relu(x, negative_slope = 0.01)  # leaky reLU avoids dead neurons by never forcing them to send a 0 when negative, better for tiny models)
import torch.nn as nn
relu6 = nn.ReLU6()
gelu = nn.GELU()

import inspect
def whocalled(func):
    if debugPrints:
        def inner(*args, **kwargs):
            caller_stack = []
            for stack in inspect.stack():
                caller_stack.append(stack[0].f_code.co_qualname)
            print(f"Calling {func.__qualname__} from: {', '.join(caller_stack)}")

            return func(*args, **kwargs)

        return inner
    return func

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
saveModelFreq = 1298    # // 500 // 5000 // 10000 // saves the model every x number of turns

modelFilePath = "BRAIN/soul/babyllm.pth"    # where your currently trained saved boi is :)
modelBackupFilePath = "BRAIN/soul/babyLLM.pth"  # where your currently trained saved boi is :)

stepCheckpointFilePath = "BRAIN/soul/stepCheckpoint.txt"

"""--- TRAINING ---"""
trainingFilePathCLEANED = "SCHOOL/library/trainingData.txt"
trainingFilePathTEST = "SCHOOL/library/trainingDataTEST.txt"

"""--- LOGS ---"""
printFreq = 1   # how often to print training progress to the terminal
printPromptLength = 1000    # how many characters of the prompt to display in terminal
gradientLength = 3000

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
numTokensPerStep = 64   # Number of tokens to predict per step
inferenceOutputNumTokens = 40

"""memoryLayer"""
memoryLengthGOAL = 1

"""optimizer"""
learningRate = 0.00035  # // 0.0005 // 0.00005 // 0.00001 //
optimizerName = "AdamW" # // "AdamW" //~decoupled weights adam, helps avoid erasing learning by overfitting etc. // "Adam" //~good for initial fast training, likely to do overfitting stuff
activationFunction = gelu   # // leakyRelu // relu // relu6 // gelu //

gradientClipMaxNorm = 1.0

"""scheduled sampling"""
scheduledSampling = True 

"""repetition penalty"""
repetitionWindowGOAL = 31   # how many tokens to look back for repetition
windowEntropyBonus = True

"""--- LOGS ---"""
detailedLogging = True

trainingLogFreq_A = 1000    # creates logs every x number of turns
trainingLogFreq_B = 10000   # creates logs every x number of turns

dontSaveEveryPrint = True
saveFreq_littleLog = 500

newLineBetweenStats = True

durationLogging = False # // True // False // activates debug time logging
debugPrints = False
anomalyDetect = False

skipNeuron = False
skipINN = True # THIS IS WHERE THE SLOWDOWN IS!!!!!
skipINNparliament = False
skipMemory = False

skipComputeLoss = False
skipMetaLoss = True

"""--- STATS COLLECTION ---"""
mostImportantStats  =   [
            # EMBED STATS
                "1E_0_embedVector_norm",                            # IMPORTANT LAYER TRACKER !! (INPUT)
            #       "1E_0_embedVector_scale",
            #       "1E_0_embedVector_norm_token",          
            #       "1E_0_embedVector_norm_neuron",                
            #       "1E_1_embedNormed_norm",
            #           "1E_1_embedNormed_scale",  
            #           "1E_1_embedNormed_norm_token",
            #           "1E_1_embedNormed_norm_neuron",       
                "1E_x_embedFinal_norm",                             # IMPORTANT LAYER TRACKER !! (EMBEDS)
            #       "1E_x_embedFinal_norm_token",
            #       "1E_x_embedFinal_norm_neuron",

            # NEURON STATS
            #                                                       "2N_0_rawInput_norm", # MATCHES 2B_0_inputEmbeds_norm & 1E_x_embedFinal_norm
            #           "2N_0_rawInput_norm_token",            # might be unneeded if this is already per token, check later
            #           "2N_0_rawInput_norm_neurons",
            #       "2N_1_rawOutput_norm",
            #           "2N_1_rawOutput_norm_token",            
            #           "2N_1_rawOutput_norm_neuron",
            #       "2N_2_activatedOutput_norm", 
            #           "2N_2_activatedOutput_norm_token",      
            #           "2N_2_activatedOutput_norm_neuron", 
                "2N_x_normedOutput_norm",                           # IMPORTANT LAYER TRACKER !! (NEURONS)
            #                "2N_x_normedOutput_norm_token",         
            #                "2N_x_normedOutput_norm_neuron",

            # INTERNEURON NETWORK STATS
            #                                                        "3INN_0_rawActivations_norm", # MATCHES 2N_x_normedOutput_norm
            #           "3INN_0_rawActivations_norm_token",         
            #           "3INN_0_rawActivations_norm_neuron",       
            #       "3INN_1_rawActivationsLayerNorm_norm",  
            #           "3INN_1_rawActivationsLayerNorm_norm_token",
            #           "3INN_1_rawActivationsLayerNorm_norm_neuron",
            #       "3INN_2_combinedActivations_norm",       
            #           "3INN_2_combinedActivations_scale", 
            #           "3INN_2_combinedActivations_norm_token",    
            #           "3INN_2_combinedActivations_norm_neuron",  
            #       "3INN_3_refinedActivations_norm",        
            #           "3INN_3_refinedActivations_scale",
            #           "3INN_3_refinedActivations_norm_token",     
            #           "3INN_3_refinedActivations_norm_neuron", 
            #       "3INN_4_combinedActivationsMeta_norm",
            #           "3INN_4_combinedActivationsMeta_norm_token", 
            #           "3INN_4_combinedActivationsMeta_norm_neuron",
                "3INN_x_FINALoutLayerNorm_norm",                # IMPORTANT LAYER TRACKER !! (INTERNEURON NETWORK)
            #       "3INN_x_FINALoutLayerNorm_norm_token",      
            #       "3INN_x_FINALoutLayerNorm_norm_neuron",
                "_INN_windowSizesMean",
                "INN_cerebellumMean",  

            # MEMORY STATS
            #                                                       "4M_0_rawActivations_norm", # MATCHES 3INN_x_FINALoutLayerNorm_norm
            #   "4M_1_shortTermMemory_norm",
            #   "4M_1_longTermMemory_norm",                
                "4M_x_FINALmemory_norm",                        # IMPORTANT LAYER TRACKER !! (MEMORY)
            #
            #   "4M_longDecay",
            #   "4M_shortDecay",
                "_4M_shortGateScale",
                "_4M_longGateScale",
                "_4M_activationsGateScale",  

            # BABYLLM STATS
            #                                                       "2B_0_inputEmbeds_norm", # MATCHES 2N_0_rawInput_norm & 1E_x_embedFinal_norm
            #                                                       "3B_1_INNOutput_norm", # MATCHES 3INN_x_FINALoutLayerNorm_norm
            #                                                       "5B_0_memoryOutput_norm", # MATCHES 4M_x_FINALmemory_norm
                "5B_x_finalNormLayer_norm",                     # IMPORTANT LAYER TRACKER !! (BABYLLM)
            #                                                       "7B_x_FINALlogits_norm", # MATCHES 6L_x_finalLogit_norm
                "_B_floatMemoryLength",
                "_B_repetitionWindow", 
                "_B_temperature",

            # LOGIT STATS
            #                                                       "6L_0_activationsTensor_norm", # MATCHES 5B_x_finalNormLayer_norm
            #                                                           "6L_0_activationsTensor_scale",
            #        "6L_1_normedActivationsTensor_norm",    
            #           "6L_1_normedActivationsTensor_scale",
            #        "6L_2_scaledActivations_norm",
            #        "6L_3_logitOutput_norm",
            #           "6L_3_logitOutput_scale",
            #        "6L_4_logitNormed_norm",
            #           "6L_4_logitNormed_scale", 
                "6L_x_finalLogit_norm",                         # IMPORTANT LAYER TRACKER !! (LOGIT)

            # MISC/UNSORTED STATS
                # base stats
                    "LR",   "learningRate", "lR",
                    "latestLossDelta",  "AvgLoss",  "loss",
                    #"temperature",
                    #"memoryLength",
                    #"gradNorm",
                    #"gradientClipMaxNorm",
                    #"scheduledSamplingRate",    "sampledTokens", 

                # learnable parameters
                    "repetitionPenalty",   
                        ]

allRecordedOtherStats = ["avgLoss",                         "stepLoss",                     "tokenCount",
                         "trainingStepCount",               "windowWeight",                 "INN_cerebellumStd",
                         "latestMemoryGates",               "embedNormMean",                "embedNormStd",
                         "embedNormMax",                    "embedDimensionMean",           "embedDimensionSparsity",
                         "embeddingDrift",                  "logitMin",                     "logitMax",                     
                         "logitSeq",                        "logitWeightNormMean",          "logitWeightNormStd",           
                         "logitWeightNormMax",              "logitWeightSparsity",          "logitWeightDrift",             
                         "logitBiasMean",                   "logitBiasStd",                 "logitBiasMax",                 
                         "n_weightMean",                    "n_weightStd",                  "n_weightMin",                  
                         "n_weightMax",                     "n_biasesMean",                 "n_biasesStd",                  
                         "n_biasesMin",                     "n_biasesMax",                  "n_sparsity"]

allRecordedOtherStats += [
                        "temperature",                      "memoryLength",                 "gradNorm",
                        "gradientClipMaxNorm",              "scheduledSamplingRate",        "sampledTokens", 
                        "6L_0_activationsTensor_norm", # MATCHES 5B_x_finalNormLayer_norm
                        "6L_0_activationsTensor_scale",
                        "6L_1_normedActivationsTensor_norm",    
                        "6L_1_normedActivationsTensor_scale",
                        "6L_2_scaledActivations_norm",
                        "6L_3_logitOutput_norm",
                        "6L_3_logitOutput_scale",
                        "6L_4_logitNormed_norm",
                        "6L_4_logitNormed_scale", 
                        "2B_0_inputEmbeds_norm", # MATCHES 2N_0_rawInput_norm & 1E_x_embedFinal_norm
                        "3B_1_INNOutput_norm", # MATCHES 3INN_x_FINALoutLayerNorm_norm
                        "5B_0_memoryOutput_norm", # MATCHES 4M_x_FINALmemory_norm
                        "7B_x_FINALlogits_norm", # MATCHES 6L_x_finalLogit_norm
                        "4M_longDecay",
                        "4M_shortDecay",
                        "4M_0_rawActivations_norm", # MATCHES 3INN_x_FINALoutLayerNorm_norm
                        "4M_1_shortTermMemory_norm",
                        "4M_1_longTermMemory_norm",     
                        "3INN_x_FINALoutLayerNorm_norm_token",      
                        "3INN_x_FINALoutLayerNorm_norm_neuron",
                        "1E_0_embedVector_scale",
                        "1E_0_embedVector_norm_token",          
                        "1E_0_embedVector_norm_neuron",                
                        "1E_1_embedNormed_norm",
                        "1E_1_embedNormed_scale",  
                        "1E_1_embedNormed_norm_token",
                        "1E_1_embedNormed_norm_neuron",       
                        "1E_x_embedFinal_norm_token",
                        "1E_x_embedFinal_norm_neuron",
                        "2N_0_rawInput_norm",
                        "2N_0_rawInput_norm_token",
                        "2N_0_rawInput_norm_neurons",
                        "2N_1_rawOutput_norm",
                        "2N_1_rawOutput_norm_token",            
                        "2N_1_rawOutput_norm_neuron",
                        "2N_2_activatedOutput_norm", 
                        "2N_2_activatedOutput_norm_token",      
                        "2N_2_activatedOutput_norm_neuron", 
                        "2N_x_normedOutput_norm_token",         
                        "2N_x_normedOutput_norm_neuron",
                        "3INN_0_rawActivations_norm", # MATCHES 2N_x_normedOutput_norm
                        "3INN_0_rawActivations_norm_token",         
                        "3INN_0_rawActivations_norm_neuron",       
                        "3INN_1_rawActivationsLayerNorm_norm",  
                        "3INN_1_rawActivationsLayerNorm_norm_token",
                        "3INN_1_rawActivationsLayerNorm_norm_neuron",
                        "3INN_2_combinedActivations_norm",       
                        "3INN_2_combinedActivations_scale", 
                        "3INN_2_combinedActivations_norm_token",    
                        "3INN_2_combinedActivations_norm_neuron",  
                        "3INN_3_refinedActivations_norm",        
                        "3INN_3_refinedActivations_scale",
                        "3INN_3_refinedActivations_norm_token",     
                        "3INN_3_refinedActivations_norm_neuron", 
                        "3INN_4_combinedActivationsMeta_norm",
                        "3INN_4_combinedActivationsMeta_norm_token", 
                        "3INN_4_combinedActivationsMeta_norm_neuron",
]

percentileBands = [100.0, 99.8, 95, 90, 85, 80, 65, 50, 35, 20, 10, 5, 0.2, 0.00]

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

profiler = False
mpsProfiler = False
forwardProfiler = False

"""--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- """


"""--- --- --- --- --- TRAINING DATA & SORTING --- --- --- --- ---"""

trainingFilePath = trainingFilePathCLEANED # //trainingFilePathCLEANED //trainingFilePathTEST
trainingDataSliceSize_min = 5000000000
trainingDataSliceSize_max = 3000000000000
reflectionFreq = 3456
# --- #
trainingDataPairNumber = 69420
trainingStartIndex = 0     # // 'random' (not in babyLLM.py)
epochs = 20

rawDataFilepaths = [     # for textCleaningTool.py

    #-*- CHARIS STUDIES -*-
    #--- CHAT HISTORY ---
    ("discord_json", "SCHOOL/library/charisStudies/discord.json", 0),     # discord message history
    ("reddit_comment", "SCHOOL/library/charisStudies/reddit_comments.csv", 0),     # reddit comments
    ("text", "SCHOOL/library/charisStudies/shitpoems.txt", 0),     #  random poems from my notes on my phone
    ("reddit_post", "SCHOOL/library/charisStudies/reddit_posts.csv", 0),     # reddit posts
    ("json", "SCHOOL/library/charisStudies/charisGPThistory.txt", 0),     # chatgpt history charis side only
    ("text", "SCHOOL/library/charisStudies/old_fb_messages_extract.txt", 1),     # old account facebook messages charis side only
    ("text", "SCHOOL/library/charisStudies/essays.txt", 0),     # essays
    ("text", "SCHOOL/library/charisStudies/tindieBaby.txt", 0),     # tindie blog posts

    #--- MOUSE ADVENTURES ---
    ("text", "SCHOOL/library/mouseAdventure/elodieMousey.txt", 0),     #  elodies wonderful mouse story!
    ("text", "SCHOOL/library/mouseAdventure/mousey.txt", 0),     #  my simple version of elodies mouse story!
    ("text", "SCHOOL/library/mouseAdventure/elodieMouseyLonger.txt", 0),     #  even more of elodies lovely mouse story!

    #--- MINI TRAINING ---
    ("text", "SCHOOL/library/miniTraining/miniTraining.txt", 0),     # i am happy! i did it! i know it!
    ("text", "SCHOOL/library/miniTraining/miniTraining2.txt", 0),     # training: i am happy! i did it! i know it!

    #--- BABYLLM CHAT LOGS ---
    ("text", chatLogPath_talkToYourself, 0),     #  i answer my own previous chat messages
    ("text", chatLogPath_trainingLog, 0),     # log: 'what am i learning today?'
    ("text", chatLogPath_infer, 0),     # log: babyLLM infer.py history!
    ("text", chatLogPath_talkToYourselfComparisons, 0),     # log: comparing babyllms answers to my answers

    #--- TENSES ---
    ("text", "SCHOOL/library/tenses/presentTense.txt", 0),     #  tense: present (kevin's weed theme?)
    ("text", "SCHOOL/library/tenses/pastTense.txt", 0),     # tense: past (mouse theme!)

    ("text", "SCHOOL/library/tenses/presentTense copy.txt", 0),     # tense
    ("text", "SCHOOL/library/tenses/futureContinuousTense.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/futurePerfectContinuousTense.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/futurePerfectTense.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/pastModalCouldHave.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/pastModalMustHaveTense.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/pastModalShouldHave.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/pastModalWouldHaveTense.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/pastPerfectContinuousTense.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/pastPerfectTense.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/presentContinuousTense.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/presentModalCanTense.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/presentModalCouldTense.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/presentModalMustTense.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/presentModalShouldTense.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/presentPerfectContinuousTense.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/presentPerfectTense.txt", 0),     #  tense
    ("text", "SCHOOL/library/tenses/futureTense.txt", 0),     #  tense: future
    ("text", "SCHOOL/library/tenses/presentConditionalTense.txt", 0),     # tense: present conditional
    ("text", "SCHOOL/library/tenses/pastContinuousTense.txt", 0),     #  tense: past continuous
    ("text", "SCHOOL/library/tenses/imperativeTense.txt", 0),     #  tense

    #--- SIMPLE TRAINING ---
    ("text", "SCHOOL/library/simpleTraining/cursed.txt", 0),     # training but chaotic shuffle
    ("text", "SCHOOL/library/simpleTraining/geepyGenerated.txt", 0),     # weird fake sentences
    ("text", "SCHOOL/library/simpleTraining/sampleshorterwrittenexamples.txt", 0),     #  training
    ("text", "SCHOOL/library/simpleTraining/shortestwrittenexamples.txt", 0),     #  training
    ("text", "SCHOOL/library/simpleTraining/shorterwrittenexamples.txt", 0),     #  training
    ("text", "SCHOOL/library/simpleTraining/longerwrittenexamples.txt", 0),     #  training
    ("text", "SCHOOL/library/simpleTraining/lineSortedData.txt", 0),     #  training
    ("text", "SCHOOL/library/simpleTraining/longestwrittenexamples.txt", 0),     #  training
    ("text", "SCHOOL/library/simpleTraining/mixedwrittenanddefs.txt", 0),     # training
    ("text", "SCHOOL/library/simpleTraining/writtenexamples.txt", 0),     #  training
    ("text", "SCHOOL/library/simpleTraining/variedWrittenExamples.txt", 0),     #  training
    ("text", "SCHOOL/library/charisStudies/thames.txt", 0),
    ("text", "SCHOOL/library/charisStudies/weirdMixedStuff.txt", 0),
    ("text", "SCHOOL/library/simpleTraining/computingKnowledge.txt", 0),

]

"""--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- """
"""-*- WARNING, CHANGING BELOW SETTINGS MAY MAKE CURRENTLY TRAINED MODEL INACCURATE (don't kill babyLLM!) -*-"""

"""--- --- --- --- --- MASTER CONFIG PARAMETERS --- --- --- --- ---"""
saveStrict = False   # // False //~allow reconstruction of missing files // True //~save files must be present, else fail

"""--- MODEL ---"""
embedDimension = 1024   # dimensionality of token embeddings
numNeurons = 10000  # number of neurons in the parallel neuron layer

"""windows"""
#  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24

windowMIN = 1   # Small Context Window

window0 = 32.01 #32
window1 = 28.01 #28
window2 = 24.01 #248
window3 = 20.01 #20
window4 = 16.01 #16    
window5 = 12.01 #12
window6 = 8.01 #8
window7 = 4.01 #4
window8 = 2.01 #2

windowMAX = 64  # THIS MUST BE THE HIGHEST NUMBER
allWindowSizes_new = [window8, window0, window1, window2, window3, window4, window5, window6, window7]     # defines the position of each window in the window weightings!
#allWindowSizes = list(range(1, 33))

attentionWindow = None  # attention head  
numHeads = 32

"""--- VOCAB & TOKENIZER ---"""
vocabSize = 2000    # maximum vocabulary size
minTokenFreq = 20   # the amount of repeats of a token needed to create a split during tokenizer training
V_chunkSizeLoadData = 4096

"""vocab data & filepaths"""
vocabCachePath = "BRAIN/vocabCache"
vocabLoad = "BRAIN/vocabCache/tokenizer.json"

"""--- MISC & EXTRA FORMATS ---"""
#trainingFilePath_dict = [{"type": ftype, "in": fname, "out": trainingFilePath} for ftype, fname in rawDataFilepaths]     # Convert to dictionary format when needed
trainingFilePath_dict = [{"type": ftype, "in": fname, "weight": weight, "out": trainingFilePath} for ftype, fname, weight in rawDataFilepaths]

trainingFilePath_arr = [trainingFilePath]
#tokenizedDataPath = "SCHOOL/tokenizedTrainingData.txt"

trainingFilePath_dict_weighted = []
for entry in trainingFilePath_dict:
    weight = entry["weight"]
    if weight == -1:
        # one clean copy
        entry["out"] = "trainingData.txt"
        trainingFilePath_dict_weighted.append(entry)
    elif weight > 0:
        trainingFilePath_dict_weighted.extend([entry] * weight)


trainingFileWeightTotal = sum([entry[2] for entry in rawDataFilepaths if len(entry) == 3])