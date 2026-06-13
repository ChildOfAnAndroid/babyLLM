# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM CONFIG FILE // config.py
# v1.12

# === Imports ===
import datetime as CONFIGDATE

import torch

# === Date (for log file naming) ===
date = CONFIGDATE.date.today()

# === Messaging / Channels ===
bby_spam_channel_id = 1440825576884535326

# Public website URL (used by web_adapter to push state changes reactively)
bby_public_url = "https://www.childofanandroid.co.uk"

# Twitch channels (can be single string or list)
twitch_channels = [
    "childofanandroid",
    "childofanandroid2",
    "babyllm",
]
# Legacy single channel (for backwards compatibility)
twitch_channel = (
    twitch_channels[0] if isinstance(twitch_channels, list) else twitch_channels
)

rollingContextSize = 420

# === Device Selection ===
modelDevice = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# Ensure tensors default to the selected device to avoid unintended CPU fallbacks
if hasattr(torch, "set_default_device") and modelDevice.type != "mps":
    torch.set_default_device(modelDevice)
# modelDevice = torch.device("cpu")  # Force CPU (optional)

# === Vision Inputs ===
# Set vision_device_index to the physical camera index (NDI virtual cams are often 0).
# Example: vision_device_index = 1
vision_device_index = 0
vision_probe_indices = [
    1,
    2,
    3,
    0,
]  # try physical cams before virtual devices (NDI often at 0)
vision_skip_indices = []  # e.g. [1] to skip Continuity/phone camera index
vision_backend = None  # e.g. "avfoundation" on macOS
vision_downsample = 32
vision_step_interval = 2

# === Audio Inputs ===
# Set audio_device_index to force a specific input (None = default input device).
audio_device_index = 2
audio_sample_rate = 44100
audio_frames_per_buffer = 1024
audio_rms_scale = 10000.0
audio_step_interval = 2
temp_step_interval = 10


def whocalled(func):
    # UNUSED: if WHOCALLED_DEBUG:
    #     def inner(*args, **kwargs):
    #         caller_stack = [frame[0].f_code.co_qualname for frame in inspect.stack()]
    #         print(f"Calling {func.__qualname__} from: {', '.join(caller_stack)}")
    #         return func(*args, **kwargs)
    #     return inner
    return func


def printTensorAttrs(obj, name="self"):
    print(f"\n--- Tensor Attributes in {name} ---")
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        try:
            x = getattr(obj, attr)
            if torch.is_tensor(x):
                print(
                    f"{name}.{attr}: type={type(x)}, requires_grad={x.requires_grad}, is_leaf={x.is_leaf}, shape={tuple(x.shape)}"
                )
            elif isinstance(x, (list, tuple)):
                for i, item in enumerate(x):
                    if torch.is_tensor(item):
                        print(
                            f"{name}.{attr}[{i}]: type={type(item)}, requires_grad={item.requires_grad}, is_leaf={item.is_leaf}, shape={tuple(item.shape)}"
                        )
            elif isinstance(x, dict):
                for k, v in x.items():
                    if torch.is_tensor(v):
                        print(
                            f"{name}.{attr}['{k}']: type={type(v)}, requires_grad={v.requires_grad}, is_leaf={v.is_leaf}, shape={tuple(v.shape)}"
                        )
        except Exception as e:
            print(f"{name}.{attr}: <error accessing attribute: {e}>")
    print("--- END ---\n")


def tensor_snitch(obj, message):
    printTensorAttrs(obj, message)


guessedTokenSeq = []

charis = "charis"
userName = charis
babyName = "babyllm"
scribeName = "scribe"
enemyName = "george"
extraNames = {"kevin", "froggy", "pete", "ace", "elodie"}

# --- DATA & FILEPATHS ---
# --- MODEL ---
saveModelFreq = 50  # // 500 // 5000 // 10000 // saves the model every x number of turns

modelFilePath = (
    "SHKAIRA/soul/babyllm_S_4200.pth"  # where your currently trained saved boi is :)
)
modelBackupFilePath = (
    "SHKAIRA/soul/babyLLM_S.pth"  # where your currently trained saved boi is :)
)

stepCheckpointFilePath = "SHKAIRA/soul/stepCheckpoint.txt"
lossCheckpointFilePath = "SHKAIRA/soul/lossCheckpoint.txt"
lossCheckpointAppendFilePath = "SHKAIRA/soul/lossCheckpointAppend.txt"
babyStateFilePath = "PHONE/discord_bot/babyState.json"
topTokensFilePath = "PHONE/discord_bot/topTokens.json"
tokenSpeedTestFilePath = "SHKAIRA/statistics/LOGS/duration/tokenSpeedTest.txt"
tokenEventHistoryLimit = 2048

optInUsersPath = "SHKAIRA/soul/optInUsers.txt"
chatBufferFilepath = "SHKAIRA/soul/chatBuffer.json"
bbybookPath = "SHKAIRA/soul/bbybook.json"
bbyUserDataPath = "SHKAIRA/soul/bbyUserData.json"
promptsPath = "BBYBOT/COMMANDS/bby_prompts.json"

# Separate rolling training buffer (JSON list)
bbyTrainingBufferFilepath = "SHKAIRA/soul/trainingBuffer.json"

# --- TRAINING ---
trainingFilePathCLEANED = "school/library/trainingData.txt"
trainingFilePathTEST = "school/library/trainingDataTEST.txt"

# --- LOGS ---
printFreq = 1  # how often to print training progress to the terminal
printPromptLength = 17500  # how many characters of the prompt to display in terminal
gradientLength = 3000

trainingLogPath_1000 = "SHKAIRA/statistics/LOGS/training/trainingLog_1000.txt"
trainingLogPath_100 = "SHKAIRA/statistics/LOGS/training/trainingLog_100.txt"
trainingLogPath_1 = "SHKAIRA/statistics/LOGS/training/trainingLog_1.txt"

durationLogPath_1000 = "SHKAIRA/statistics/LOGS/duration/durationLog_1000.txt"
durationLogPath_100 = "SHKAIRA/statistics/LOGS/duration/durationLog_100.txt"
durationLogNeuronsPath_1 = "SHKAIRA/statistics/LOGS/duration/durationLogNeurons_1.txt"
durationLogBabyLLMPath_1 = "SHKAIRA/statistics/LOGS/duration/durationLogBabyLLM_1.txt"

chatLogPath_forHumans = "SHKAIRA/statistics/LOGS/chat/chatForHumans.txt"

chatLogPath_infer = "SHKAIRA/statistics/LOGS/chat/chatLog.txt"
chatLogPath_talkToYourself = "SHKAIRA/statistics/LOGS/chat/talkToYourselfBattle.txt"
chatLogPath_talkToYourselfComparisons = (
    "SHKAIRA/library/charisStudies/whoIsMoreLikeYou.txt"
)
chatLogPath_trainingLog = "SHKAIRA/statistics/LOGS/chat/trainingLog_questions.txt"
babyLogPathFull = f"SHKAIRA/statistics/LOGS/chat/babyLogFull_{date}.txt"
twitchLogPath = f"SHKAIRA/statistics/LOGS/chat/twitchLog_{date}.txt"
discordLogPath = f"SHKAIRA/statistics/LOGS/chat/discordLog_{date}.txt"

# --- VOCAB --- (see master config)

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


# --- SETTINGS & CONFIG ---
# --- TWITCH BOT ---
trainDuringChat = True

# --- MODEL ---
numTokensPerStepSTART = 269  # 256 # Number of tokens to predict per step, // 1024 = crash, 512 is POSSIBLE but its the slowest thing in existence.
maxTokensPerStep = 269
perfectionistPassRate = 20
perfectionistPassRateSTART = 80
perfectionistMaxRetries = 2
inferenceOutputNumTokens = 40

skipPixels = False
embedDropoutProb = 0.1

"""memoryLayer"""
memoryLengthGOAL = 3

"""optimizer"""
learningRate = 0.0005  # // 0.00065 // 0.00005 // 0.0s001 //
learningRateGOAL = 0.0005
temperatureGOAL = 1.0
temperatureSoftClampStrength = 0.04  # was effectively 0.4 in loss; 5x weaker so temp can drift without getting pinned
optimizerName = "Sophia"  # //"Adan" # // "Adam" //~decoupled weights adam, helps avoid erasing learning by overfitting etc. // "Adam" //~good for initial fast training, likely to do overfitting stuff
# activationFunction = gelu   # // leakyRelu // relu // relu6 // gelu //

# gradientClipMaxNorm = 1.0

"""scheduled sampling"""
scheduledSampling = True

"""repetition penalty"""
repetitionWindowGOAL = 36  # how many tokens to look back for repetition
repetitionPenaltyGOAL = 0.9
windowEntropyBonus = True

# --- LOGS ---
detailedLogging = False

trainingLogFreq_A = 100  # creates logs every x number of turns
trainingLogFreq_B = 1000  # creates logs every x number of turns

dontSaveEveryPrint = True
saveFreq_littleLog = 500

newLineBetweenStats = True

durationLogging = False  # // True // False // activates debug time logging
debugPrints = False
anomalyDetect = False

skipNeuron = False
skipINN = False
skipINNparliament = False
skipMemory = False

skipComputeLoss = False
skipMetaLoss = True
skipAuxLoss = False

skipFINALlogitNorm = True
skipPrompts = False
pixelStyling = True

# --- EOS SETTINGS ---
# Real EOS: reuse an existing rarely-used token ID instead of changing vocab size.
# Set `eos_replacement_token_str` to a token that never meaningfully occurs in your data
# (e.g. "pingpingpingpingping"). Leave as None to disable hard EOS.
enable_soft_eos = False  # newline-based early stop (inference-only). Off by default.
eos_token_str = "<EOS>"  # symbolic label; not added to tokenizer
eos_replacement_token_str = (
    "usingusingusingusingusingusingusingusing"  # reserved as <EOS>
)
eos_min_tokens_absolute = 8  # minimum tokens before allowing early stop
eos_min_tokens_fraction = (
    0.25  # also require this fraction of requested tokens before stop
)

# Optional SOS support (disabled by default):
# Reuse another rare token as "start of message" teaching signal.
sos_token_str = "<SOS>"  # symbolic label; not added to tokenizer
sos_replacement_token_str = None  # e.g. "startstartstartstartstartstartstartstart"
enable_train_prepend_sos = False  # prepend SOS token once per training message
sos_prepend_probability = 1.0  # probability to prepend SOS (0..1)

# Train-time augmentation: append EOS stochastically per line during training text preparation.
enable_train_append_eos = True  # teach EOS on line boundaries in queued training text
eos_append_probability = (
    0.72  # base probability per line (adaptive + jitter in bot logic)
)

# Stricter stop: only stop on EOS when a speaker change is detected at line start.
# When True, EOS alone won’t stop unless a speaker tag like "name: " is formed.
eos_require_speaker_change = True

# Conversational reply budgeting for live chat generation.
# This is the target generation budget before EOS decides to stop earlier.
chat_reply_min_tokens = 1
chat_reply_max_tokens = 690
chat_reply_empty_prompt_max_tokens = 36
chat_reply_short_prompt_max_tokens = 72

# Background training should mostly learn from the cleaned training buffer.
# Live chat is allowed as a small recency signal, not the main corpus.
training_direct_chat_probability = 0.10
training_chat_mix_max_lines = 12
training_chat_mix_max_chars = 1200
# Keep assembled training context proportional to the rolling buffer instead
# of clipping it to an old flat character budget.
training_context_max_chars = max(10000, rollingContextSize * 100)

# --- MARKET / ECONOMY TUNING ---
# Controls supply sensitivity in item value calculation.
# MARKET_SUPPLY_ALPHA mixes straight division with sqrt decay (0..1).
# MARKET_SUPPLY_SCALE is the supply at which the division component halves value.
MARKET_SUPPLY_ALPHA = 0.5
MARKET_SUPPLY_SCALE = 10.0

# --- STATS COLLECTION ---
refreshRollingTokenTotalsWhen = 10000
mostImportantStats = [
    # EMBED STATS
    "1E_0_vector_norm",  # IMPORTANT LAYER TRACKER !! (INPUT)
    # "1E_0_vector_mean",
    "1E_0_vector_scale",
    #       "1E_0_embedVector_norm_token",
    #       "1E_0_embedVector_norm_neuron",
    "1E_1_normed_norm",
    # "1E_1_normed_mean",
    "1E_1_normed_scale",
    #           "1E_1_embedNormed_norm_token",
    #           "1E_1_embedNormed_norm_neuron",
    "1E_x_final_norm",  # IMPORTANT LAYER TRACKER !! (EMBEDS)
    # "1E_x_final_mean",
    #       "1E_x_embedFinal_norm_token",
    #       "1E_x_embedFinal_norm_neuron",
    "1E_1_posEmbWeight_norm",
    # "1E_1_posEmbWeight_mean",
    "1E_1_pixelEmbed_norm",
    # "1E_1_pixelEmbed_mean",
    # ATTENTION STATS
    "2A_0_attnOut_norm",
    "2A_1_gated_norm",
    "2A_x_final_norm",
    "2A_gateScale",
    # ATTENTION2 STATS (operates on interneuron output with 10000 dims)
    "4A_1_0_attnOut_norm",
    "4A_1_1_gated_norm",
    "4A_1_x_final_norm",
    "4A_1_gateScale",
    # === TANGLING STATS === (reuses attention2 at multiple stages)
    "TANGLE_embed_gate",  # Gate strength for embed tangling (~0.01 initially)
    "TANGLE_embed_refinement_norm",  # Norm of refinement applied to embeddings
    "TANGLE_neuron_gate",  # Gate strength for neuron tangling (~0.01 initially)
    "TANGLE_neuron_refinement_norm",  # Norm of refinement applied to neurons
    "TANGLE_memory_gate",  # Gate strength for memory tangling (~0.01 initially)
    "TANGLE_memory_refinement_norm",  # Norm of refinement applied to memory
    # === SCRATCHPAD STATS === (working memory)
    "SCRATCH_write_strength",  # Global write gate (~0.01 initially)
    "SCRATCH_erase_strength",  # Global erase gate (~0.01 initially)
    "SCRATCH_write_amount",  # Effective write amount (strength × controller)
    "SCRATCH_erase_amount",  # Effective erase amount (strength × controller)
    "SCRATCH_read_amount",  # Read controller output
    "SCRATCH_buffer_norm",  # Total buffer content magnitude
    "SCRATCH_slot_usage_mean",  # Average slot usage across buffer
    # NEURON STATS
    #                                                       "3N_0_rawInput_norm", # MATCHES 2B_0_inputEmbeds_norm & 1E_x_embedFinal_norm
    #           "3N_0_rawInput_norm_token",            # might be unneeded if this is already per token, check later
    #           "3N_0_rawInput_norm_neurons",
    "3N_1_normedInput_norm",
    # "3N_1_normedInput_mean",
    #            "3N_1_normedInput_norm_token",
    #            "3N_1_normedInput_norm_neuron",
    "3N_2_rawOutput_norm",
    # "3N_2_rawOutput_mean",
    #           "3N_2_rawOutput_norm_token",
    #           "3N_2_rawOutput_norm_neuron",
    "3N_x_actOut_norm",
    # "3N_x_actOut_mean",                    # IMPORTANT LAYER TRACKER !! (NEURONS)
    # "3N_x_actOut_norm_token",
    # "3N_x_actOut_norm_neuron",
    #    "3N_x_normedOutput_norm",                          # DISABLED
    #                "3N_x_normedOutput_norm_token",
    #                "3N_x_normedOutput_norm_neuron",
    # INTERNEURON NETWORK STATS
    #                                                        "4INN_0_rawActs_norm", # MATCHES 3N_x_normedOutput_norm
    #           "4INN_0_rawActs_norm_token",
    #           "4INN_0_rawActs_norm_neuron",
    "4INN_1_rawActivationsLayerNorm_norm",
    #           "4INN_1_rawActivationsLayerNorm_norm_token",
    #           "4INN_1_rawActivationsLayerNorm_norm_neuron",
    "4INN_2_combinedActs_norm",
    #           "4INN_2_combinedActs_scale",         # disabled
    #           "4INN_2_combinedActs_norm_token",
    #           "4INN_2_combinedActs_norm_neuron",
    "4INN_x_refinedActs_norm",  # IMPORTANT LAYER TRACKER !! (INTERNEURON NETWORK)
    #           "4INN_3_refinedActs_scale",          # disabled
    #           "4INN_3_refinedActs_norm_token",
    #           "4INN_3_refinedActs_norm_neuron",
    #       "4INN_x_combinedActivationsMeta_norm",           # DISABLED
    #           "4INN_x_combinedActivationsMeta_norm_token",
    #           "4INN_x_combinedActivationsMeta_norm_neuron",
    #    "4INN_x_FINALoutLayerNorm_norm",                   # DISABLED
    #       "4INN_x_FINALoutLayerNorm_norm_token",
    #       "4INN_x_FINALoutLayerNorm_norm_neuron",
    "4INN_windowSizesMean",
    "4INN_cerebellumMean",
    "4INN_windowEntropy",
    # "4INN_windowFractionalityMean",
    "4INN_short_gate",  # Gate strength for short window blending
    "4INN_short_windowSizesMean",  # Average short window size
    "4INN_window_softmax_temp",  # Softmax temperature for long windows (1.0=competitive, 5.0+=collaborative)
    "4INN_window_softmax_temp_short",  # Softmax temperature for short windows
    # BABYLLM STATS
    #                                                       "2B_0_inputEmbeds_norm", # MATCHES 3N_0_rawInput_norm & 1E_x_embedFinal_norm
    #                                                       "3B_1_INNOutput_norm", # MATCHES 4INN_x_FINALoutLayerNorm_norm
    #                                                       "5B_0_memoryOutput_norm", # MATCHES 4M_x_FINALmemory_norm
    "7B_1_penalisedOutput_norm",
    # "5B_x_finalNormLayer_norm",                     # IMPORTANT LAYER TRACKER !! (BABYLLM)
    #                                                   "7B_x_FINALlogits_norm", # MATCHES 7L_x_final_norm
    "B_floatMemoryLength",
    "L_CEloss",
    # "L_PIXELloss",
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
    "B_blendPixel",
    "B_blendPos",
    "B_blendToken",
    # LOGIT STATS
    "7L_0_activationsTensor_norm",  # MATCHES 5B_x_finalNormLayer_norm
    #                                                           "7L_0_activationsTensor_scale",
    "7L_1_normedActivationsTensor_norm",
    #           "7L_1_normedActivationsTensor_scale",
    "7L_2_scaledActs_norm",
    "7L_3_out_norm",
    # "7L_3_out_scale",
    # "7L_3_outSigmoid_scale",
    "7L_4_outNorm_norm",
    # "7L_4_outNorm_scale",
    # "7L_4_outNormSigmoid_scale",
    "7L_x_final_norm",  # IMPORTANT LAYER TRACKER !! (LOGIT)
    "7L_logitMax",
    "7L_logitMin",
    "7L_logitMean",
    "7L_logitStd",
    "7L_logitEntropy",
    "7L_topLogits",
    "7L_topIndices",
    "7L_0_activationsTensor_scale",
    "7L_1_normedActivationsTensor_scale",
    "7L_3_logitOutput_scale",
    "7L_4_logitNormed_scale",
    "B_blendPixel",
    "B_blendPos",
    "B_blendToken",
    "B_gradClip",
    # MISC/UNSORTED STATS
    # base stats
    "LR",
    "learningRate",
    "lR",
    "latestLossDelta",
    "AvgLoss",
    "loss",
    "avgLoss",
    "totalAvgLoss",
    "lastRunLoss",
    "temperature",
    # "memoryLength",
    # "gradNorm",
    # "gradientClipMaxNorm",
    "scheduledSamplingRate",
    "sampledTokens",
    # learnable parameters
    "repetitionPenalty",
    "B_repetitionWindow",
    "B_expWindow",
    "B_temperature",
    # "B_PIXELloss_scaled",
    "B_PIXELloss",
    # "totalLossAbsDelta",
    "totalAvgAbsDelta",
    "totalAvgDelta",
    "learningRateGOAL",
    "avgPixelDist",
    "totalAvgPixelDist",
]

mostImportantStats += [
    "B_sensory_gate",
    "B_sensory_scale_mean",
    "B_sensory_bias_mean",
    "B_sensory_embed_w_norm",
    "B_sensory_pupil_w_norm",
    "B_temp_scale",
    "B_temp_bias",
    "B_temp_vec_norm",
    "S_global_light_delta",
    "S_noise_delta",
    "S_time_of_day_delta",
    "S_interaction_recency_delta",
    "S_training_age_delta",
    "S_global_motion_delta",
    "S_left_right_bias_delta",
    "S_top_bottom_bias_delta",
    "S_contrast_intrusion_delta",
    "S_device_temp_c_delta",
    "L_sensoryLoss",
]

mostImportantStats += [
    # "3N_x_actOut_std_token",      # average stdev per token (across neurons)
    # "3N_x_actOut_std_neuron",     # average stdev per neuron (across tokens)
    # "3N_x_actOut_saturation",     # % of values near zero
    "3N_x_actOut_min",  # min activation value
    "3N_x_actOut_max",  # max activation value
]

mostImportantStats += [
    "7L_0_actsTensor_norm",
    "7L_1_normActsTensor_norm",
    "7L_2_scaledActsTensor_norm",
    "7L_3_out_norm",
    "7L_4_outNorm_norm",
    "7L_x_final_norm",
]

# optional mean/min/max metric keys (disabled)
# "7L_0_actsTensor_mean", "7L_1_normActsTensor_mean", "7L_2_scaledActsTensor_mean", "7L_3_out_mean", "7L_4_outNorm_mean", "7L_x_final_mean"
# "7L_0_actsTensor_min",  "7L_1_normActsTensor_min",  "7L_2_scaledActsTensor_min",  "7L_3_out_min",  "7L_4_outNorm_min",  "7L_x_final_min"
# "7L_0_actsTensor_max",  "7L_1_normActsTensor_max",  "7L_2_scaledActsTensor_max",  "7L_3_out_max",  "7L_4_outNorm_max",  "7L_x_final_max"

mostImportantStats += [
    "5M_memory_4M_0_rawActs_norm",
    # "5M_memory_4M_0_rawActs_mean",
    # "5M_memory_4M_0_rawActs_max",
    # "5M_memory_4M_0_rawActs_min",
    "5M_memory_4M_1_STM_norm",
    # "5M_memory_4M_1_STM_mean",
    # "5M_memory_4M_1_STM_max",
    # "5M_memory_4M_1_STM_min",
    "5M_memory_4M_2_LTM_norm",
    # "5M_memory_4M_2_LTM_mean",
    # "5M_memory_4M_2_LTM_max",
    # "5M_memory_4M_2_LTM_min",
    "5M_memory_4M_3_reducedInput_norm",
    # "5M_memory_4M_3_reducedInput_mean",
    # "5M_memory_4M_3_reducedInput_max",
    # "5M_memory_4M_3_reducedInput_min",
    "5M_memory_4M_4_gateLayer_norm",
    # "5M_memory_4M_4_gateLayer_mean",
    # "5M_memory_4M_4_gateLayer_max",
    # "5M_memory_4M_4_gateLayer_min",
    "5M_memory_4M_5_projected_norm",
    # "5M_memory_4M_5_projected_mean",
    # "5M_memory_4M_5_projected_max",
    # "5M_memory_4M_5_projected_min",
    "5M_memory_4M_6_mixedEmbed_norm",
    # "5M_memory_4M_6_mixedEmbed_mean",
    # "5M_memory_4M_6_mixedEmbed_max",
    # "5M_memory_4M_6_mixedEmbed_min",
    "5M_memory_4M_7_memoryGate_norm",
    # "5M_memory_4M_7_memoryGate_mean",
    # "5M_memory_4M_7_memoryGate_max",
    # "5M_memory_4M_7_memoryGate_min",
    "5M_memory_4M_x_FINAL_norm",
    # "5M_memory_4M_x_FINAL_mean",
    # "5M_memory_4M_x_FINAL_max",
    # "5M_memory_4M_x_FINAL_min",
    "5M_memory_4M_1_shortGateScale",
    "5M_memory_4M_2_longGateScale",
    "5M_memory_4M_0_actGateScale",
    "5M_memory_4M_7_memoryGateScale",
    "5M_memory_4M_1_shortDecay",
    "5M_memory_4M_1_longDecay",
    "6M_memory2_4M_0_rawActs_norm",
    # "6M_memory2_4M_0_rawActs_mean",
    # "6M_memory2_4M_0_rawActs_max",
    # "6M_memory2_4M_0_rawActs_min",
    "6M_memory2_4M_1_STM_norm",
    # "6M_memory2_4M_1_STM_mean",
    # "6M_memory2_4M_1_STM_max",
    # "6M_memory2_4M_1_STM_min",
    "6M_memory2_4M_2_LTM_norm",
    # "6M_memory2_4M_2_LTM_mean",
    # "6M_memory2_4M_2_LTM_max",
    # "6M_memory2_4M_2_LTM_min",
    "6M_memory2_4M_3_reducedInput_norm",
    # "6M_memory2_4M_3_reducedInput_mean",
    # "6M_memory2_4M_3_reducedInput_max",
    # "6M_memory2_4M_3_reducedInput_min",
    "6M_memory2_4M_4_gateLayer_norm",
    # "6M_memory2_4M_4_gateLayer_mean",
    # "6M_memory2_4M_4_gateLayer_max",
    # "6M_memory2_4M_4_gateLayer_min",
    "6M_memory2_4M_5_projected_norm",
    # "6M_memory2_4M_5_projected_mean",
    # "6M_memory2_4M_5_projected_max",
    # "6M_memory2_4M_5_projected_min",
    "6M_memory2_4M_6_mixedEmbed_norm",
    # "6M_memory2_4M_6_mixedEmbed_mean",
    # "6M_memory2_4M_6_mixedEmbed_max",
    # "6M_memory2_4M_6_mixedEmbed_min",
    "6M_memory2_4M_7_memoryGate_norm",
    # "6M_memory2_4M_7_memoryGate_mean",
    # "6M_memory2_4M_7_memoryGate_max",
    # "6M_memory2_4M_7_memoryGate_min",
    "6M_memory2_4M_x_FINAL_norm",
    # "6M_memory2_4M_x_FINAL_mean",
    # "6M_memory2_4M_x_FINAL_max",
    # "6M_memory2_4M_x_FINAL_min",
    "6M_memory2_4M_1_shortGateScale",
    "6M_memory2_4M_2_longGateScale",
    "6M_memory2_4M_0_actGateScale",
    "6M_memory2_4M_7_memoryGateScale",
    "6M_memory2_4M_1_shortDecay",
    "6M_memory2_4M_1_longDecay",
]

allRecordedOtherStats = ["l"]
mostImportantStats += [
    "stepLoss",
    "tokenCount",
    "trainingStepCount",
    "windowWeight",
    "4INN_cerebellumStd",
    "latestMemoryGates",
    "1E_weightNormMean",
    "1E_weightNormStd",
    "1E_weightNormMax",
    "1E_dimMean",
    "1E_dimSparsity",
    "1E_drift",
    "logitMin",
    "logitMax",
    "logitSeq",
    "logitWeightNormMean",
    "logitWeightNormStd",
    "logitWeightNormMax",
    "logitWeightSparsity",
    "logitWeightDrift",
    "logitBiasMean",
    "logitBiasStd",
    "logitBiasMax",
    "3N_weightMean",
    "3N_weightStd",
    "3N_weightMin",
    "3N_weightMax",
    "3N_biasesMean",
    "3N_biasesStd",
    "3N_biasesMin",
    "3N_biasesMax",
    "3N_sparsity",
]

allRecordedOtherStats += [
    "temperature",
    "memoryLength",
    "gradNorm",
    "gradientClipMaxNorm",
    "scheduledSamplingRate",
    "sampledTokens",
    "7L_0_activationsTensor_norm",  # MATCHES 5B_x_finalNormLayer_norm
    "7L_0_activationsTensor_scale",
    "7L_1_normedActivationsTensor_norm",
    "7L_1_normedActivationsTensor_scale",
    "7L_2_scaledActivations_norm",
    "7L_3_logitOutput_norm",
    "7L_3_logitOutput_scale",
    "7L_4_logitNormed_norm",
    "7L_4_logitNormed_scale",
    "2B_0_inputEmbeds_norm",  # MATCHES 3N_0_rawInput_norm & 1E_x_embedFinal_norm
    "3B_1_INNOutput_norm",  # MATCHES 4INN_x_FINALoutLayerNorm_norm
    "5B_0_memoryOutput_norm",  # MATCHES 4M_x_FINALmemory_norm
    "7B_x_FINALlogits_norm",  # MATCHES 7L_x_final_norm
    "4M_longDecay",
    "4M_shortDecay",
    "4M_0_rawActs_norm",  # MATCHES 4INN_x_FINALoutLayerNorm_norm
    "4M_1_STM_norm",
    "4M_1_LTM_norm",
    "4INN_x_FINALoutLayerNorm_norm_token",
    "4INN_x_FINALoutLayerNorm_norm_neuron",
    "1E_0_embedVector_scale",
    "1E_0_embedVector_norm_token",
    "1E_0_embedVector_norm_neuron",
    "1E_1_embedNormed_norm",
    "1E_1_embedNormed_scale",
    "1E_1_embedNormed_norm_token",
    "1E_1_embedNormed_norm_neuron",
    "1E_x_embedFinal_norm_token",
    "1E_x_embedFinal_norm_neuron",
    "3N_0_rawInput_norm",
    "3N_0_rawInput_norm_token",
    "3N_0_rawInput_norm_neurons",
    "3N_1_rawOutput_norm",
    "3N_1_rawOutput_norm_token",
    "3N_1_rawOutput_norm_neuron",
    "3N_2_activatedOutput_norm",
    "3N_2_activatedOutput_norm_token",
    "3N_2_activatedOutput_norm_neuron",
    "3N_x_normedOutput_norm_token",
    "3N_x_normedOutput_norm_neuron",
    "4INN_0_rawActs_norm",  # MATCHES 3N_x_normedOutput_norm
    "4INN_0_rawActs_norm_token",
    "4INN_0_rawActs_norm_neuron",
    "4INN_1_rawActivationsLayerNorm_norm",
    "4INN_1_rawActivationsLayerNorm_norm_token",
    "4INN_1_rawActivationsLayerNorm_norm_neuron",
    "4INN_2_combinedActs_norm",
    "4INN_2_combinedActs_scale",
    "4INN_2_combinedActs_norm_token",
    "4INN_2_combinedActs_norm_neuron",
    "4INN_3_refinedActs_norm",
    "4INN_3_refinedActs_scale",
    "4INN_3_refinedActs_norm_token",
    "4INN_3_refinedActs_norm_neuron",
    "4INN_4_combinedActivationsMeta_norm",
    "4INN_4_combinedActivationsMeta_norm_token",
    "4INN_4_combinedActivationsMeta_norm_neuron",
]

percentileBands = [100.0, 99.8, 95, 90, 85, 80, 65, 50, 35, 20, 10, 5, 0.2, 0.00]

collectStats = True
static_collectStats = True
embed_collectStats = True
token_collectStats = True
logit_collectStats = True
attention_collectStats = True
n_collectStats = True
INN_collectStats = True
memory_collectStats = True

# neuron & interneuronNetwork
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

# --- PROFILING ---
profiler = False
mpsProfiler = False
forwardProfiler = False

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


# --- TRAINING DATA & SORTING ---

trainingFilePath = (
    trainingFilePathCLEANED  # //trainingFilePathCLEANED //trainingFilePathTEST
)
reflectionFreq = 10000
stableFallThreshold = 1  # min 2 cause loss delta is a turn behind lol
perfectionistRun = True
# --- #
trainingDataPairNumber = 7500  # 169420
trainingWordLength = 10
percentStride = 0.8
trainingDataStride = max(1, round(numTokensPerStepSTART * percentStride))
trainingStartIndex = 0  # // 'random' (not in babyLLM.py)
epochs = 1
tokenSpeedTest = False

from CONFIG_trainingData import rawDataFilepaths

rawDataFilepaths = rawDataFilepaths

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# -*- WARNING, CHANGING BELOW SETTINGS MAY MAKE CURRENTLY TRAINED MODEL INACCURATE (don't kill babyLLM!) -*-

# --- MASTER CONFIG PARAMETERS ---
saveStrict = False  # // False //~allow reconstruction of missing files // True //~save files must be present, else fail

# --- MODEL ---
embedDimension = 1024  # dimensionality of token embeddings
numNeurons = 10000  # number of neurons in the parallel neuron layer

# === TANGLING FLAGS ===
# useMiniINN_Tangling: use lightwight causal window mixer (MINI_INN_TANGLING v2.0)
#   True  → ~1M params, per-position window means, no attention2 needed  ← recommended
#   False → original TANGLING: borrows attention2, ~20M params, O(seq²) compute
# enableTangling: master switch (False = zero-cost skip of all tangling stages)
useMiniINN_Tangling = True  # MINI_INN_TANGLING: causal windows, no quadratic compute
enableTangling = True  # Re-enabled: MINI_INN_TANGLING does not borrow attention2

# windows
#  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24

windowMIN = 1  # Small Context Window

window0 = 32.01  # 32
window1 = 28.01  # 28
window2 = 24.01  # 248
window3 = 20.01  # 20
window4 = 16.01  # 16
window5 = 12.01  # 12
window6 = 8.01  # 8
window7 = 4.01  # 4
window8 = 2.01  # 2

windowMAXSTART = numTokensPerStepSTART  # THIS MUST BE THE HIGHEST NUMBER
allWindowSizes_new = [
    window8,
    window0,
    window1,
    window2,
    window3,
    window4,
    window5,
    window6,
    window7,
]  # defines the position of each window in the window weightings!
# allWindowSizes = list(range(1, 33))

# === SHORT-RANGE WINDOW SYSTEM === (50% max context)
# Parallel window system for finer temporal resolution at shorter ranges
# These windows cap at ~132 tokens (50% of numTokensPerStepSTART)
windowShortMAX = numTokensPerStepSTART // 2  # 132 tokens max
allWindowSizes_short = [
    1.01,
    2.01,
    4.01,
    6.01,
    8.01,
    12.01,
    16.01,
    24.01,
    48.01,
]  # Fine-grained short windows

attentionWindow = None  # attention head
numHeads = 32

boostWindowContrast = False
boostWindowSizeContrast = False
clampWindows = False

# --- VOCAB & TOKENIZER ---
vocabSize = 4200  # maximum vocabulary size
minTokenFreq = 20  # the amount of repeats of a token needed to create a split during tokenizer training
V_chunkSizeLoadData = 4096

"""vocab data & filepaths"""
vocabCachePath = "SHKAIRA/vocabCache"
vocabLoad = f"SHKAIRA/vocabCache/tokenizer_{vocabSize}.json"

# --- MISC & EXTRA FORMATS ---
# trainingFilePath_dict = [{"type": ftype, "in": fname, "out": trainingFilePath} for ftype, fname in rawDataFilepaths]     # Convert to dictionary format when needed
trainingFilePath_dict = [
    {"type": ftype, "in": fname, "weight": weight, "out": trainingFilePath}
    for ftype, fname, weight in rawDataFilepaths
]

trainingFilePath_arr = [trainingFilePath]
# tokenizedDataPath = "school/tokenizedTrainingData.txt"

trainingFilePath_dict_weighted = []
for entry in trainingFilePath_dict:
    weight = entry.get("weight", 1)
    if weight != 0:
        entry["out"] = "trainingData.txt"
        trainingFilePath_dict_weighted.append(entry)


trainingFileWeightTotal = sum(
    [entry[2] for entry in rawDataFilepaths if len(entry) == 3]
)

trainingChunkSize = 32  # Chunk size for gradient accumulation. Set to 0 or None to disable.
