# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM CONFIG FILE // config.py

import datetime as CONFIGDATE
date = CONFIGDATE.date.today()


import torch
modelDevice = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#modelDevice = torch.device("cpu")

#from torch import relu 
from torch.nn.functional import leaky_relu
leakyRelu = lambda x: leaky_relu(x, negative_slope = 0.01)  # leaky reLU avoids dead neurons by never forcing them to send a 0 when negative, better for tiny models)
import torch.nn as nn
relu6 = nn.ReLU6()
from torch.nn.functional import gelu

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
saveModelFreq = 500    # // 500 // 5000 // 10000 // saves the model every x number of turns

modelFilePath = "BRAIN/soul/babyllm_4200.pth"    # where your currently trained saved boi is :)
modelBackupFilePath = "BRAIN/soul/babyLLM.pth"  # where your currently trained saved boi is :)

stepCheckpointFilePath = "BRAIN/soul/stepCheckpoint.txt"
lossCheckpointFilePath = "BRAIN/soul/lossCheckpoint.txt"

"""--- TRAINING ---"""
trainingFilePathCLEANED = "SCHOOL/library/trainingData.txt"
trainingFilePathTEST = "SCHOOL/library/trainingDataTEST.txt"

"""--- LOGS ---"""
printFreq = 1   # how often to print training progress to the terminal
printPromptLength = 17500    # how many characters of the prompt to display in terminal
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
babyLogPathFull = f"SCHOOL/statistics/LOGS/chat/babyLogFull_{date}.txt"

"""--- VOCAB --- (see master config)"""


"""--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- """


"""--- --- --- --- --- SETTINGS & CONFIG --- --- --- --- ---"""
"""--- MODEL ---"""
numTokensPerStepSTART = 8 # Number of tokens to predict per step, // 1024 = crash, 512 is POSSIBLE but its the slowest thing in existence.
perfectionistPassRate = 20
perfectionistPassRateSTART = 80
perfectionistMaxRetries = 10
inferenceOutputNumTokens = 40

skipPixels = False

"""memoryLayer"""
memoryLengthGOAL = 3

"""optimizer"""
learningRate = 0.00035  # // 0.0005 // 0.00005 // 0.0s001 //
learningRateGOAL = 0.00035
temperatureGOAL = 0.85
optimizerName = "Adan" # //"AdamW" # // "AdamW" //~decoupled weights adam, helps avoid erasing learning by overfitting etc. // "Adam" //~good for initial fast training, likely to do overfitting stuff
activationFunction = gelu   # // leakyRelu // relu // relu6 // gelu //

gradientClipMaxNorm = 1.0

"""scheduled sampling"""
scheduledSampling = True 

"""repetition penalty"""
repetitionWindowGOAL = 16   # how many tokens to look back for repetition
repetitionPenaltyGOAL = 2.0
windowEntropyBonus = True

"""--- LOGS ---"""
detailedLogging = True

trainingLogFreq_A = 100    # creates logs every x number of turns
trainingLogFreq_B = 1000   # creates logs every x number of turns

dontSaveEveryPrint = True
saveFreq_littleLog = 500

newLineBetweenStats = True

durationLogging = False # // True // False // activates debug time logging
debugPrints = False
anomalyDetect = True

skipNeuron = False
skipINN = True # THIS IS WHERE THE SLOWDOWN IS!!!!!
skipINNparliament = False
skipMemory = False

skipComputeLoss = False
skipMetaLoss = True
skipAuxLoss = False

skipFINALlogitNorm = True
pixelStyling = True

"""--- STATS COLLECTION ---"""
refreshRollingTokenTotalsWhen = 10000
mostImportantStats  =   [
            # EMBED STATS
                "1E_0_vector_norm",                            # IMPORTANT LAYER TRACKER !! (INPUT)
                   "1E_0_vector_scale",
            #       "1E_0_embedVector_norm_token",          
            #       "1E_0_embedVector_norm_neuron",                
                   "1E_1_normed_norm",
                       "1E_1_normed_scale",  
            #           "1E_1_embedNormed_norm_token",
            #           "1E_1_embedNormed_norm_neuron",       
                "1E_x_final_norm",                             # IMPORTANT LAYER TRACKER !! (EMBEDS)
            #       "1E_x_embedFinal_norm_token",
            #       "1E_x_embedFinal_norm_neuron",
            "1E_1_posEmbeddings",

            # NEURON STATS
            #                                                       "2N_0_rawInput_norm", # MATCHES 2B_0_inputEmbeds_norm & 1E_x_embedFinal_norm
            #           "2N_0_rawInput_norm_token",            # might be unneeded if this is already per token, check later
            #           "2N_0_rawInput_norm_neurons",
                    "2N_1_normedInput_norm",
            #            "2N_1_normedInput_norm_token",
            #            "2N_1_normedInput_norm_neuron",
                   "2N_2_rawOutput_norm",
            #           "2N_2_rawOutput_norm_token",            
            #           "2N_2_rawOutput_norm_neuron",
                   "2N_x_actOut_norm",                     # IMPORTANT LAYER TRACKER !! (NEURONS)
                       "2N_x_actOut_norm_token",      
                       "2N_x_actOut_norm_neuron", 
            #    "2N_x_normedOutput_norm",                          # DISABLED   
            #                "2N_x_normedOutput_norm_token",         
            #                "2N_x_normedOutput_norm_neuron",

            # INTERNEURON NETWORK STATS
            #                                                        "3INN_0_rawActivations_norm", # MATCHES 2N_x_normedOutput_norm
            #           "3INN_0_rawActivations_norm_token",         
            #           "3INN_0_rawActivations_norm_neuron",       
                   "3INN_1_rawActivationsLayerNorm_norm",  
            #           "3INN_1_rawActivationsLayerNorm_norm_token",
            #           "3INN_1_rawActivationsLayerNorm_norm_neuron",
                   "3INN_2_combinedActivations_norm",       
            #           "3INN_2_combinedActivations_scale",         # disabled
            #           "3INN_2_combinedActivations_norm_token",    
            #           "3INN_2_combinedActivations_norm_neuron",  
                   "3INN_x_refinedActivations_norm",                # IMPORTANT LAYER TRACKER !! (INTERNEURON NETWORK)
            #           "3INN_3_refinedActivations_scale",          # disabled
            #           "3INN_3_refinedActivations_norm_token",     
            #           "3INN_3_refinedActivations_norm_neuron", 
            #       "3INN_x_combinedActivationsMeta_norm",           # DISABLED
            #           "3INN_x_combinedActivationsMeta_norm_token", 
            #           "3INN_x_combinedActivationsMeta_norm_neuron",
            #    "3INN_x_FINALoutLayerNorm_norm",                   # DISABLED
            #       "3INN_x_FINALoutLayerNorm_norm_token",      
            #       "3INN_x_FINALoutLayerNorm_norm_neuron",
                "3INN_windowSizesMean",
                "3INN_cerebellumMean",  
                "3INN_windowFractionalityMean",

            # MEMORY STATS
                    "4M_0_rawActivations_norm",
                    "4M_1_shortTermMemory_norm",
                    "4M_2_longTermMemory_norm",
                    "4M_4_gateLayer",
                    "4M_1_shortGateScale",
                    "4M_2_longGateScale",
                    "4M_0_activationsGateScale",
                    "4M_7_memoryGateScale",
                    "4M_5_projectedMemory_norm",
                    "4M_7_memoryGate_norm",
                    "4M_6_mixedEmbed_norm",
                    "4M_x_FINALmemory_norm",
                    "4M_1_shortDecay",
                    "4M_1_longDecay",

            # BABYLLM STATS
            #                                                       "2B_0_inputEmbeds_norm", # MATCHES 2N_0_rawInput_norm & 1E_x_embedFinal_norm
            #                                                       "3B_1_INNOutput_norm", # MATCHES 3INN_x_FINALoutLayerNorm_norm
            #                                                       "5B_0_memoryOutput_norm", # MATCHES 4M_x_FINALmemory_norm
                    "7B_1_penalisedOutput_norm",
                #"5B_x_finalNormLayer_norm",                     # IMPORTANT LAYER TRACKER !! (BABYLLM)
                #                                                   "7B_x_FINALlogits_norm", # MATCHES 6L_x_finalLogit_norm
                #"B_floatMemoryLength",
                "L_CEloss",
                "L_PIXELloss",
                "L_PIXELloss_scaled",
                "L_AUXlossCos",
                "L_AUXlossKL",
                "L_LRclamp",
                "L_tempClamp",
                "L_repPenClamp",
                "L_repLoss",
                "L_triesLoss",
                "L_perfLoss",
                "L_entropyLoss",
                "L_pixelDistLoss",


            # LOGIT STATS
                                                                   "6L_0_activationsTensor_norm", # MATCHES 5B_x_finalNormLayer_norm
            #                                                           "6L_0_activationsTensor_scale",
                    "6L_1_normedActivationsTensor_norm",    
            #           "6L_1_normedActivationsTensor_scale",
                    "6L_2_scaledActivations_norm",
                    "6L_3_out_norm",
                        "6L_3_out_scale",
                        "6L_3_outSigmoid_scale",
                    "6L_4_outNorm_norm",
                        "6L_4_outNorm_scale", 
                        "6L_4_outNormSigmoid_scale",
                "6L_x_finalLogit_norm",                         # IMPORTANT LAYER TRACKER !! (LOGIT)
                "6L_logitMax", "6L_logitMin", "6L_logitMean", "6L_logitStd", "6L_logitEntropy", "6L_topLogits", "6L_topIndices", 
                "6L_0_activationsTensor_scale", "6L_1_normedActivationsTensor_scale", "6L_3_logitOutput_scale", "6L_4_logitNormed_scale",
                "B_blendPixel",
                "B_blendPos",
                "B_blendToken",

            # MISC/UNSORTED STATS
                # base stats
                    "LR",   "learningRate", "lR",
                    "latestLossDelta",  "AvgLoss",  "loss", "avgLoss", "totalAvgLoss", "lastRunLoss",
                    #"temperature",
                    #"memoryLength",
                    #"gradNorm",
                    #"gradientClipMaxNorm",
                    #"scheduledSamplingRate",    "sampledTokens", 

                # learnable parameters
                    "repetitionPenalty",   
                    "B_repetitionWindow", 
                    "B_expWindow",
                    "B_temperature",
                    "B_PIXELloss_scaled",
                    "B_PIXELloss",
                    "totalLossAbsDelta",
                    "totalAvgAbsDelta",
                    "learningRateGOAL",
                    "avgPixelDist",
                    "totalAvgPixelDist",
                        ]

mostImportantStats += [
    "2N_x_actOut_std_token",      # average stdev per token (across neurons)
    "2N_x_actOut_std_neuron",     # average stdev per neuron (across tokens)
    "2N_x_actOut_saturation",     # % of values near zero
    "2N_x_actOut_min",            # min activation value
    "2N_x_actOut_max",            # max activation value
]

allRecordedOtherStats = ["l"]
mostImportantStats   += ["stepLoss",                        "tokenCount",
                         "trainingStepCount",               "windowWeight",                 "3INN_cerebellumStd",
                         "latestMemoryGates",               "1E_weightNormMean",            "1E_weightNormStd",
                         "1E_weightNormMax",                "1E_dimMean",                   "1E_dimSparsity",
                         "1E_drift",                  "logitMin",                     "logitMax",                     
                         "logitSeq",                        "logitWeightNormMean",          "logitWeightNormStd",           
                         "logitWeightNormMax",              "logitWeightSparsity",          "logitWeightDrift",             
                         "logitBiasMean",                   "logitBiasStd",                 "logitBiasMax",                 
                         "2N_weightMean",                    "2N_weightStd",                "2N_weightMin",                  
                         "2N_weightMax",                     "2N_biasesMean",               "2N_biasesStd",                  
                         "2N_biasesMin",                     "2N_biasesMax",                "2N_sparsity"]

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
trainingDataSliceSize_min = 10000
trainingDataSliceSize_max = 100000
reflectionFreq = 10000
stableFallThreshold = 10 # min 2 cause loss delta is a turn behind lol
perfectionistRun = True
# --- #
trainingDataPairNumber = 100 #169420
trainingDataStride = 1
trainingStartIndex = 0     # // 'random' (not in babyLLM.py)
epochs = 1

rawDataFilepaths = [
    ("text", "SCHOOL/library/charisStudies/essays.txt", -1),     # discord message history
]

rawDataFilepaths = [     # for textCleaningTool.py
    #-*- CHARIS STUDIES -*-
    #--- CHAT HISTORY ---
    ("text", "SCHOOL/library/charisStudies/charisParisProductions.txt", -1),     # discord message history
    ("text", "SCHOOL/library/charisStudies/discordtxt.txt", 0.1),     # discord message history
    ("text", "SCHOOL/library/charisStudies/discordtxt2.txt", 0.1),     # discord message history part2
    ("text", "SCHOOL/library/charisStudies/discordtxt3.txt", 0.1),     # discord message history part3
    ("text", "SCHOOL/library/charisStudies/discordtxt4.txt", 0.1),     # discord message history part4
    ("text", "SCHOOL/library/charisStudies/discordtxt5.txt", 0.1),     # discord message history part5
    ("text", "SCHOOL/library/charisStudies/discordtxt6.txt", 0.1),     # discord message history part6
    ("text", "SCHOOL/library/charisStudies/discordtxt7.txt", 0.1),     # discord message history part7
    ("text", "SCHOOL/library/charisStudies/discordtxt8.txt", 0.1),     # discord message history part8
    ("text", "SCHOOL/library/charisStudies/discordtxt9.txt", 0.1),     # discord message history part8
    ("discord_json", "SCHOOL/library/charisStudies/discord.json", 1),     # discord message history JSON
    ("reddit_comment", "SCHOOL/library/charisStudies/reddit_comments.csv", 1),     # reddit comments
    ("text", "SCHOOL/library/charisStudies/shitpoems.txt", -1),     #  random poems from my notes on my phone
    ("reddit_post", "SCHOOL/library/charisStudies/reddit_posts.csv", 1),     # reddit posts
    ("json", "SCHOOL/library/charisStudies/charisGPThistory.txt", 1),     # chatgpt history charis side only
    ("text", "SCHOOL/library/charisStudies/old_fb_messages_extract.txt", 1),     # old account facebook messages charis side only
    ("text", "SCHOOL/library/charisStudies/essays.txt", -1),     # essays
    ("text", "SCHOOL/library/charisStudies/tindieBaby.txt", 1),     # tindie blog posts

    #--- MOUSE ADVENTURES ---
    ("text", "SCHOOL/library/mouseAdventure/elodieMousey.txt", -1),     #  elodies wonderful mouse story!
    ("text", "SCHOOL/library/mouseAdventure/mousey.txt", -1),     #  my simple version of elodies mouse story!
    ("text", "SCHOOL/library/mouseAdventure/elodieMouseyLonger.txt", -1),     #  even more of elodies lovely mouse story!

    #--- MINI TRAINING ---
    ("text", "SCHOOL/library/miniTraining/miniTraining.txt", -1),     # i am happy! i did it! i know it!
    ("text", "SCHOOL/library/miniTraining/miniTraining2.txt", -1),     # training: i am happy! i did it! i know it!

    #--- BABYLLM CHAT LOGS ---
    ("text", chatLogPath_talkToYourself, 0.01),     #  i answer my own previous chat messages
    ("text", chatLogPath_trainingLog, 0.01),     # log: 'what am i learning today?'
    ("text", chatLogPath_infer, 0.01),     # log: babyLLM infer.py history!
    ("text", chatLogPath_talkToYourselfComparisons, 0.01),     # log: comparing babyllms answers to my answers
    ("text", "scribeSays.txt", 0.01),

    #--- TENSES ---
    ("text", "SCHOOL/library/tenses/presentTense.txt", 0.001),     #  tense: present (kevin's weed theme?)
    ("text", "SCHOOL/library/tenses/pastTense.txt", 0.001),     # tense: past (mouse theme!)

    ("text", "SCHOOL/library/tenses/presentTense copy.txt", 0.001),     # tense
    ("text", "SCHOOL/library/tenses/futureContinuousTense.txt", 0.001),     #  tense
    ("text", "SCHOOL/library/tenses/futurePerfectContinuousTense.txt", 0.001),     #  tense
    ("text", "SCHOOL/library/tenses/futurePerfectTense.txt", 0.001),     #  tense
    ("text", "SCHOOL/library/tenses/pastModalCouldHave.txt", 0.001),     #  tense
    ("text", "SCHOOL/library/tenses/pastModalMustHaveTense.txt", 0.001),     #  tense
    ("text", "SCHOOL/library/tenses/pastModalShouldHave.txt", 0.001),     #  tense
    ("text", "SCHOOL/library/tenses/pastModalWouldHaveTense.txt", 0.001),     #  tense
    ("text", "SCHOOL/library/tenses/pastPerfectContinuousTense.txt", 0.001),     #  tense
    ("text", "SCHOOL/library/tenses/presentContinuousTense.txt", 0.001),    #  tense
    ("text", "SCHOOL/library/tenses/pastPerfectTense.txt", 0.001),     #  tense
    ("text", "SCHOOL/library/tenses/presentModalCanTense.txt", 0.001),     #  tense
    ("text", "SCHOOL/library/tenses/presentModalCouldTense.txt", 0.001),     #  tense
    ("text", "SCHOOL/library/tenses/presentModalMustTense.txt", 0.001),     #  tense
    ("text", "SCHOOL/library/tenses/presentModalShouldTense.txt", 0.001),     #  tense
    ("text", "SCHOOL/library/tenses/presentPerfectContinuousTense.txt", 0.001),     #  tense
    ("text", "SCHOOL/library/tenses/presentPerfectTense.txt", 0.001),     #  tense
    ("text", "SCHOOL/library/tenses/futureTense.txt", 0.001),    #  tense: future
    ("text", "SCHOOL/library/tenses/presentConditionalTense.txt", 0.001),     # tense: present conditional
    ("text", "SCHOOL/library/tenses/pastContinuousTense.txt", 0.001),     #  tense: past continuous
    ("text", "SCHOOL/library/tenses/imperativeTense.txt", 0.001),     #  tense
]

rawDataFilepaths += [
    #--- SIMPLE TRAINING ---
    ("text", "SCHOOL/library/simpleTraining/cursed.txt", 0.1),     # training but chaotic shuffle
    ("text", "SCHOOL/library/simpleTraining/geepyGenerated.txt", 0.1),     # weird fake sentences
    ("text", "SCHOOL/library/simpleTraining/sampleshorterwrittenexamples.txt", 0.1),     #  training
    ("text", "SCHOOL/library/simpleTraining/shortestwrittenexamples.txt", 0.1),     #  training
    ("text", "SCHOOL/library/simpleTraining/shorterwrittenexamples.txt", 0.1),     #  training
    ("text", "SCHOOL/library/simpleTraining/longerwrittenexamples.txt", 0.1),     #  training
    ("text", "SCHOOL/library/simpleTraining/lineSortedData.txt", 0.1),     #  training
    ("text", "SCHOOL/library/simpleTraining/longestwrittenexamples.txt", 0.1),     #  training
    ("text", "SCHOOL/library/simpleTraining/mixedwrittenanddefs.txt", 0.1),     # training
    ("text", "SCHOOL/library/simpleTraining/writtenexamples.txt", 0.1),     #  training
    ("text", "SCHOOL/library/simpleTraining/variedWrittenExamples.txt", 0.1),     #  training
    ("text", "SCHOOL/library/charisStudies/thames.txt", 0.1),
    ("text", "SCHOOL/library/charisStudies/weirdMixedStuff.txt", 0.1),
    ("text", "SCHOOL/library/simpleTraining/computingKnowledge.txt", 1),
    ("text", "SCHOOL/library/miniTraining/why.txt", 1),
    ("text", "SCHOOL/library/miniTraining/why2.txt", 1),
    ("text", "SCHOOL/library/miniTraining/why3.txt", 1),
    ("text", "SCHOOL/library/miniTraining/why4.txt", 1),]

rawDataFilepaths += [
    #--- MY OWN CODE?? ---
    ("text", "babyLLM.py", 0.1),
    ("text", "config.py", 0.1),
    ("text", "infer.py", 0.1),
    ("text", "talkToYourself.py", 0.01),
    ("text", "textCleaningTool.py", 0.1),
    ("text", "wakeup.py", 0.1),
    ("text", "SCHOOL/staffroom/calligraphist.py", 0.1),
    ("text", "SCHOOL/staffroom/counsellor.py", 0.1),
    ("text", "SCHOOL/staffroom/HE_IS_SCRIBE.py", 0.1),
    ("text", "SCHOOL/staffroom/librarian.py", 0.1),
    ("text", "SCHOOL/staffroom/tutor.py", 0.1),
    ("text", "BRAIN/vocabCache/tokenizer_4200.json", 0.1),
    ("text", "BRAIN/readmeactuallyprobablydont.txt", 0.1),
    ("text", "BRAIN/LAYERS/embed.py", 0.1),
    ("text", "BRAIN/LAYERS/interneuronNetwork.py", 0.1),
    ("text", "BRAIN/LAYERS/logits.py", 0.1),
    ("text", "BRAIN/LAYERS/memory.py", 0.1),
    ("text", "SCHOOL/notebook/notes.txt", 0.1),
    ("text", "SCHOOL/notebook/python notes etc", 0.1),
    ("text", "SCHOOL/notebook/test.py", 0.1),
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

windowMAXSTART = numTokensPerStepSTART  # THIS MUST BE THE HIGHEST NUMBER
allWindowSizes_new = [window8, window0, window1, window2, window3, window4, window5, window6, window7]     # defines the position of each window in the window weightings!
#allWindowSizes = list(range(1, 33))

attentionWindow = None  # attention head  
numHeads = 32

boostWindowContrast = False
boostWindowSizeContrast = False 
clampWindows = False

"""--- VOCAB & TOKENIZER ---"""
vocabSize = 4200    # maximum vocabulary size
minTokenFreq = 20   # the amount of repeats of a token needed to create a split during tokenizer training
V_chunkSizeLoadData = 4096

"""vocab data & filepaths"""
vocabCachePath = "BRAIN/vocabCache"
vocabLoad = f"BRAIN/vocabCache/tokenizer_{vocabSize}.json"

"""--- MISC & EXTRA FORMATS ---"""
#trainingFilePath_dict = [{"type": ftype, "in": fname, "out": trainingFilePath} for ftype, fname in rawDataFilepaths]     # Convert to dictionary format when needed
trainingFilePath_dict = [{"type": ftype, "in": fname, "weight": weight, "out": trainingFilePath} for ftype, fname, weight in rawDataFilepaths]

trainingFilePath_arr = [trainingFilePath]
#tokenizedDataPath = "SCHOOL/tokenizedTrainingData.txt"

trainingFilePath_dict_weighted = []
for entry in trainingFilePath_dict:
    weight = entry.get("weight", 1)
    if weight != 0:
        entry["out"] = "trainingData.txt"
        trainingFilePath_dict_weighted.append(entry)


trainingFileWeightTotal = sum([entry[2] for entry in rawDataFilepaths if len(entry) == 3])