# CHARIS CAT 2025
from config import *
from datetime import datetime
import shutil
import sys

"""TERMINAL CODES"""
RESET = "\033[0m" # normal terminal
BOLD = "\033[1m"
DIM = "\033[2m" # reduces intensity of text colour
UNDERLINE = "\033[4m"
FLASH = "\033[5m"

"""COLOURS!!!"""
PURPLE_PALE = "\033[94m"
PURPLE = "\033[38;5;225m" #256 colour palette
MAGENTA = "\033[35m"
BLUE = "\033[34m"

ORANGE = "\033[38;5;52m" #256 colour palette
RED = "\033[38;5;124m" #256 colour palette
RED_BRIGHT = "\033[91m"

"""extra colours"""
GOLD = "\033[93m"
RED_ALT = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
PURPLE_ALT = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

"""TERMINAL OUTPUT STYLES - CATEGORY MAPPING"""
S_types = {
    "perfect":       [BOLD, PURPLE_PALE],   # 100%
    "almostPerfect": [PURPLE_PALE],         # 90%
    "great":         [BOLD, MAGENTA],              # 80%
    "good":          [MAGENTA],         # 70%
    "fine":          [PURPLE],           # 60%
    "almostFine":    [PURPLE, DIM],                 # 50%
    "meh":           [BLUE],              # 40%
    "bad":           [BLUE, DIM],                 # 30%
    "worse":         [PURPLE_PALE, DIM],          # 20%
    "emergency":     [DIM],   # 10%

    "reset":         [RESET],               # normal terminal
    "dim":           [RESET, DIM],          # dim style for background elements - arrows, colons, etc.
    "bold":          [BOLD]
}

DIM = [RESET, DIM]

statThresholds = {
    "loss": { #got these sorted :)
        "perfect":       0.005,
        "almostPerfect": 0.2,
        "great":         0.8,
        "good":          1.8,
        "fine":          3.5,
        "almostFine":    6.5,
        "meh":           15.0,
        "bad":           30.0,
        "worse":         60.0,
        "emergency":     float('inf')
    },
    "guessSimilarity": { #dont know enough to know how to set it! how is it different from loss???
        "perfect":       0.99,
        "almostPerfect": 0.95,
        "great":         0.90,
        "good":          0.80,
        "fine":          0.70,
        "almostFine":    0.60,
        "meh":           0.50,
        "bad":         0.40,
        "worse":          0.30,
        "emergency":     0.0
    },
    "logits": { #dont know enough to know how to set it!
        "perfect":       0.99,
        "almostPerfect": 0.95,
        "great":         0.90,
        "good":          0.80,
        "fine":          0.70,
        "almostFine":    0.60,
        "meh":           0.50,
        "bad":         0.40,
        "worse":          0.30,
        "emergency":     0.0
    },
    "windowWeights": { # same as mem gates they are percentage aligned hmm, if i pull the NORMALIZED weights.
        "perfect":       0.99, # somehow need this to be the highest rated window!
        "almostPerfect": 0.95, #2nd highest
        "great":         0.90, #3rd
        "good":          0.80, #u get me
        "fine":          0.70,
        "almostFine":    0.60,
        "meh":           0.50,
        "bad":         0.40,
        "worse":          0.30,
        "emergency":     0.0
    },
    "memGates": { # ima base this on percentage, so higher percentage (closer to 100% is 'stronger' and more dominant, but theres not really a 'meh' dominance so quickly?)
        "perfect":       0.99, #otherwise, maybe i should just to highest rated
        "almostPerfect": 0.95, #2nd
        "great":         0.90, #etc
        "good":          0.80,
        "fine":          0.70,
        "almostFine":    0.60,
        "meh":           0.50,
        "bad":         0.40,
        "worse":          0.30,
        "emergency":     0.0
    },
    "gradNorm": { #dont know enough about this to know how to set it!
        "perfect":       0.99,
        "almostPerfect": 0.95,
        "great":         0.90,
        "good":          0.80,
        "fine":          0.70,
        "almostFine":    0.60,
        "meh":           0.50,
        "bad":         0.40,
        "worse":          0.30,
        "emergency":     0.0
    },
}

statTypes = ["loss", "guessSimilarity", "logits", "windowWeights", "memGates", "gradNorm"]
infoTypes = ["learningRate", "optimizerName", "activationFunction", "gradientClipMaxNorm"]

def S_getStat(statType, statVal):
    if statType not in statThresholds:
        return "reset"

    thresholds = statThresholds[statType]

    if statVal <= thresholds["perfect"]:
        return "perfect"
    elif statVal <= thresholds["almostPerfect"]:
        return "almostPerfect"
    elif statVal <= thresholds["great"]:
        return "great"
    elif statVal <= thresholds["good"]:
        return "good"
    elif statVal <= thresholds["fine"]:
        return "fine"
    elif statVal <= thresholds["almostFine"]:
        return "almostFine"
    elif statVal <= thresholds["meh"]:
        return "meh"
    elif statVal <= thresholds["bad"]:
        return "bad"
    elif statVal <= thresholds["worse"]:
        return "worse"
    else:
        return "emergency"

def S_apply(S_type, text):
    if S_type in S_types:
        S_codes = S_types[S_type]
        return "".join(S_codes) + text + RESET
    else:
        return text

def colourPrintTraining(step, inputSentence, guessedSeqStr, targetSeqStr, loss, isCorrect, isPerfect):

    if isPerfect:
        S_type = "perfect"
    else:
        S_type = S_getStat("loss", loss)

    S_bold = "".join(S_types["bold"])

    guessedTokens = guessedSeqStr.split()
    targetTokens = targetSeqStr.split()
    S_guessedTokens = []
    S_targetTokens = []

    """    every word is in a list, and theres another list that matches
    each word has a list index thing
    so save them all separate instead of into the string
    guessedtoken1 = targettoken1? fancy colour whatever
    guessedtoken2 != targettoken2? plain"""

    for i, word in enumerate(guessedTokens):
        if i < len(targetTokens) and word == targetTokens[i]:
            S_guessedTokens.append(f"{S_bold}{word}{RESET}") # Bold if match, then reset
        else:
            S_guessedTokens.append(f"{S_apply(S_type, word)}") # Apply S_type style if no match

    # Style target words
    for i, word in enumerate(targetTokens):
        if i < len(guessedTokens) and word == guessedTokens[i]:
            S_targetTokens.append(f"{S_bold}{word}{RESET}") # Bold if match, then reset
        else:
            S_targetTokens.append(f"{S_apply(S_type, word)}") # Apply S_type style if no match

    S_guessedSeq_str = " ".join(S_guessedTokens) # Rejoin styled guessed words
    S_targetSeq_str = " ".join(S_targetTokens) # Rejoin styled target words

    fullStringCorrect = (guessedSeqStr.strip() == targetSeqStr.strip())

    #croppedInputSentence = inputSentence.strip()
    #if len(croppedInputSentence) > inputSentenceVisualLength:
    #    croppedInputSentence = "..." + croppedInputSentence[-(inputSentenceVisualLength - 3):]

    formattedWords = (
        f"{S_apply('dim', f'{step}|')}" 
        f"{S_apply('dim', inputSentence)}{S_apply('dim', ' → ')}" #{DIM} → {RESET}
        f"{S_guessedSeq_str}"
        f"{S_apply(S_type, ' [!] ') if fullStringCorrect else S_apply('dim', ' [?] ')}"
        f"{S_targetSeq_str}"
        f"{S_apply('dim', ' | ')}" #{DIM} | {RESET}"
        f"{S_apply('dim', 'Loss: ')}{S_apply(S_type, f'{loss:.3f}')}"
    )

    print(formattedWords)

#colourPrintTraining(1, "Translate this:", "Hola mundo", "Hello world", 0.005, True, True) # perfect
#colourPrintTraining(2, "Translate this:", "Hola mundo", "Hello world", 0.02, True, False) # almostPerfect
#colourPrintTraining(3, "Translate this:", "Hola mundo", "Hello world", 0.08, True, False) # good
#colourPrintTraining(4, "Translate this:", "Hola mundo", "Hello world", 0.4, True, False)  # almostGood
#colourPrintTraining(5, "Translate this:", "Hola mundo", "Hello world", 0.8, True, False)  # fine
#colourPrintTraining(6, "Translate this:", "Translate plz", "Hello world", 0.6, False, False) # almostFine
#colourPrintTraining(7, "Translate this:", "Translate plz", "Hello world", 1.2, False, False) # meh
#colourPrintTraining(8, "Translate this:", "Translate plz", "Hello world", 1.8, False, False) # almostmeh
#colourPrintTraining(9, "Translate this:", "Translate plz", "Hello world", 2.5, False, False) # worse
#colourPrintTraining(10, "Translate this:", "Translate plz", "Hello world", 5.0, False, False) # emergency

"""def HUD_fixScroll(self):
    height = shutil.get_terminal_size().lines
    #reserved_hud_lines = 5
    #training_lines_height = height - reserved_hud_lines

    #sys.stdout.write("\033[?25l\033[H\033[2J")  # Hide cursor, clear, move to top
    #sys.stdout.flush()

    # You should print training lines here *before* calling this if you want control

    # Move to bottom section and draw HUD
    training_lines_height = training_lines_height
    sys.stdout.write(f"\033[{training_lines_height + 1};0H")  # Move to HUD zone
    sys.stdout.flush()

    self.printHUD(
        windowWeights=(self.parallelNeuronLayer.windowWeighting + 0.1).detach().cpu().numpy(),
        guessHUD=self.guessHUD
    )

    sys.stdout.write(f"\033[{height};0H")  # Move cursor just above HUD for next cycle
    sys.stdout.flush()"""


def logTraining(logFilePath, step, avgLoss, learningRate, logitRange_str="", windowWeights_str="", gradientNorm_str="", scheduledSamplingProb_str="", epoch_str="", prompt="", guess="", truth="", memoryGates_str="", topTokens_str="", durationLog_str="", otherInfo=""):

    #height = shutil.get_terminal_size().lines
    #reserved_hud_lines = 5
    #training_lines_height = height - reserved_hud_lines

    #sys.stdout.write("\033[?25l\033[H\033[2J")  # Hide cursor, clear, move to top
    #sys.stdout.flush()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    S_base = "".join(S_types["dim"]) # Base style for background elements
    S_loss = S_getStat("loss", avgLoss)

    logOutput = f"{S_base}{timestamp} | Step {S_apply('reset', f'{step:f}'):<6} | LR: {S_apply('reset', f'{learningRate:.5f}')} | Scheduled Sampling: {S_apply('reset', f'{scheduledSamplingProb_str}')}{RESET}" # Reset style for step and LR labels
    logOutput += f"{S_base} | Avg Loss: {S_apply(S_loss, f'{avgLoss:.4f}')}{RESET}"

    if logitRange_str:
        S_logit = S_getStat("logits", float(logitRange_str.split(',')[0].strip()) if logitRange_str else "reset")
        logOutput += f"{S_base} | Logits: {S_apply(S_logit, logitRange_str)}{RESET}"
    #if windowWeights_str:
    #    S_window = S_getStat("windowWeights", float(windowWeights_str.split(',')[0].strip()) if windowWeights_str else "reset")
    #    logOutput += f"{S_base} | Window Weights: {S_apply(S_window, windowWeights_str)}{RESET}"
    if windowWeights_str:
        # No more S_getStat or float conversion for windowWeights_str
        logOutput += f"{S_base} | Window Weights: {windowWeights_str}{RESET}"
    if gradientNorm_str:
        S_gradNorm = S_getStat("gradNorm", float(gradientNorm_str) if gradientNorm_str else 0.0)
        logOutput += f"{S_base} | Grad Norm: {S_apply(S_gradNorm, gradientNorm_str)}{RESET}"
    #if memoryGates_str:
    #    S_memGates = S_getStat("memGates", float(memoryGates_str.split(',')[0].strip()) if memoryGates_str else "reset")
    #    logOutput += f"{S_base} | Memory Gates: {S_apply(S_memGates, memoryGates_str)}{RESET}"
    if memoryGates_str:
        # No more S_getStat or float conversion for windowWeights_str
        logOutput += f"{S_base} | Memory Gates: {memoryGates_str}{RESET}"
    if topTokens_str:
        logOutput += f"{S_base} | Top Tokens: {topTokens_str}{RESET}"
    if durationLog_str:
        logOutput = logOutput + f"\n{durationLog_str}"

    """displays some extra data when training from prompts"""
    if prompt:
        logOutput = logOutput + f"{S_base} | Prompt: {S_apply('reset', prompt)} | Guess: {S_apply('reset', guess)} | Truth: {S_apply('reset', truth)}" # Keep prompt/guess/truth reset style
    else:
        logOutput = logOutput
    if otherInfo:
        logOutput += f"{S_base} | {S_apply('reset', otherInfo)}"

    logOutput += f"{RESET}" # Ensure reset at the very end of the log line

    print(logOutput)
    with open(logFilePath, "a") as logFile:
        logFile.write(logOutput + "\n")

    #HUD_fixScroll(self)

# Example logTraining (adjust loss and other params as needed)
#logTraining("training.log", 100, 0.03, 0.001)
#logTraining("training.log", 200, 0.1, 0.0005, epoch_str="Epoch 1")
#logTraining("training.log", 300, 0.8, 0.0001, prompt="Translate cat", guess="Meow gato", truth="Cat meow")
#logTraining("training.log", 400, 2.0, 0.00005, otherInfo="Memory issues?")