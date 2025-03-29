# CHARIS CAT 2025
from config import *
from datetime import datetime

"""TERMINAL CODES"""
RESET = "\033[0m" # normal terminal
BOLD = "\033[1m"
DIM = "\033[2m" # reduces intensity of text colour
UNDERLINE = "\033[4m"
FLASH = "\033[5m"

"""COLOURS!!!"""
PURPLE_PALE = "\033[94m"
PURPLE = "\033[38;5;225m" #256 colour palette
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
typeStyles = {
    "perfect":       [PURPLE_PALE, FLASH],  # 100%
    "almostPerfect": [PURPLE_PALE],         # 90%
    "great":         [PURPLE],              # 80%
    "good":          [PURPLE, DIM],         # 70%
    "fine":          [BLUE, DIM],           # 60%
    "almostFine":    [DIM],                 # 50%
    "bad":           [ORANGE],              # 40%
    "worse":         [RED],                 # 30%
    "shit":          [RED_BRIGHT],          # 20%
    "emergency":     [RED_BRIGHT, FLASH],   # 10%

    "reset":         [RESET],               # normal terminal
    "dim":           [RESET, DIM]           # dim style for background elements - arrows, colons, etc.
}

bgStyle = [RESET, DIM]

statThresholds = {
    "loss": { #got these sorted :)
        "perfect":       0.00001,
        "almostPerfect": 0.1,
        "great":         0.5,
        "good":          1.0,
        "fine":          2.5,
        "almostFine":    5.0,
        "bad":           10.0,
        "worse":         15.0,
        "shit":          25.0,
        "emergency":     float('inf')
    },
    "guessSimilarity": { #dont know enough to know how to set it! how is it different from loss???
        "perfect":       0.99,
        "almostPerfect": 0.95,
        "great":         0.90,
        "good":          0.80,
        "fine":          0.70,
        "almostFine":    0.60,
        "bad":           0.50,
        "worse":         0.40,
        "shit":          0.30,
        "emergency":     0.0
    },
        "logits": { #dont know enough to know how to set it!
        "perfect":       0.99,
        "almostPerfect": 0.95,
        "great":         0.90,
        "good":          0.80,
        "fine":          0.70,
        "almostFine":    0.60,
        "bad":           0.50,
        "worse":         0.40,
        "shit":          0.30,
        "emergency":     0.0
    },
        "windowWeights": { # same as mem gates they are percentage aligned hmm, if i pull the NORMALIZED weights.
        "perfect":       0.99, # somehow need this to be the highest rated window!
        "almostPerfect": 0.95, #2nd highest
        "great":         0.90, #3rd
        "good":          0.80, #u get me
        "fine":          0.70,
        "almostFine":    0.60,
        "bad":           0.50,
        "worse":         0.40,
        "shit":          0.30,
        "emergency":     0.0
    },
        "memGates": { # ima base this on percentage, so higher percentage (closer to 100% is 'stronger' and more dominant, but theres not really a 'bad' dominance so quickly?)
        "perfect":       0.99, #otherwise, maybe i should just to highest rated
        "almostPerfect": 0.95, #2nd
        "great":         0.90, #etc
        "good":          0.80,
        "fine":          0.70,
        "almostFine":    0.60,
        "bad":           0.50,
        "worse":         0.40,
        "shit":          0.30,
        "emergency":     0.0
    },
        "gradNorm": { #dont know enough about this to know how to set it!
        "perfect":       0.99,
        "almostPerfect": 0.95,
        "great":         0.90,
        "good":          0.80,
        "fine":          0.70,
        "almostFine":    0.60,
        "bad":           0.50,
        "worse":         0.40,
        "shit":          0.30,
        "emergency":     0.0
    },
}

statTypes = ["loss", "guessSimilarity", "logits", "windowWeights", "memGates", "gradNorm"]
infoTypes = ["learningRate", "optimizerName", "activationFunction", "gradientClipMaxNorm"]

def getStatStyle(statType, statVal):
    if statType not in statThresholds:
        return "reset"  # Default style if stat_type is not defined

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
    elif statVal <= thresholds["bad"]:
        return "bad"
    elif statVal <= thresholds["worse"]:
        return "worse"
    elif statVal <= thresholds["shit"]:
        return "shit"
    else:  # statVal > thresholds["shit"]
        return "emergency"

def applyStyle(style_name, text):
    """Applies the terminal style codes to the given text."""
    if style_name in typeStyles:
        style_codes = typeStyles[style_name]
        return "".join(style_codes) + text + RESET
    else:
        return text

def colourPrintTraining(step, inputSentence, guessedSeqStr, targetSeqStr, loss, isCorrect, isPerfect):
    """Refined print function using styles and thresholds."""

    if isPerfect:
        typeStyle = "perfect"  # Still handle "perfect" case directly if needed
    else:
        typeStyle = getStatStyle("loss", loss) # Get style based on loss

    style = typeStyles.get(typeStyle, typeStyles["reset"]) # Default to "reset" if category not found
    bgStyle = "".join(typeStyles["dim"]) # Use "dim" style for base elements

    formattedWords = (
        f"{bgStyle}Step {applyStyle(typeStyle, str(step))}: "
        f"{applyStyle('reset', inputSentence)}{bgStyle} → {RESET}" # Input sentence can have reset style
        f"{applyStyle(typeStyle, guessedSeqStr)}{bgStyle}"
        f"{'[!]' if isCorrect else '[?]'} {RESET}"
        f"{applyStyle(typeStyle, targetSeqStr)}{bgStyle} | {RESET}"
        f"{bgStyle}Loss: {RESET}{applyStyle(typeStyle, f'{loss:.3f}')} {RESET}"
    )

    print(formattedWords)

#colourPrintTraining(1, "Translate this:", "Hola mundo", "Hello world", 0.005, True, True) # perfect
#colourPrintTraining(2, "Translate this:", "Hola mundo", "Hello world", 0.02, True, False) # almostPerfect
#colourPrintTraining(3, "Translate this:", "Hola mundo", "Hello world", 0.08, True, False) # good
#colourPrintTraining(4, "Translate this:", "Hola mundo", "Hello world", 0.4, True, False)  # almostGood
#colourPrintTraining(5, "Translate this:", "Hola mundo", "Hello world", 0.8, True, False)  # fine
#colourPrintTraining(6, "Translate this:", "Translate plz", "Hello world", 0.6, False, False) # almostFine
#colourPrintTraining(7, "Translate this:", "Translate plz", "Hello world", 1.2, False, False) # bad
#colourPrintTraining(8, "Translate this:", "Translate plz", "Hello world", 1.8, False, False) # almostBad
#colourPrintTraining(9, "Translate this:", "Translate plz", "Hello world", 2.5, False, False) # shit
#colourPrintTraining(10, "Translate this:", "Translate plz", "Hello world", 5.0, False, False) # emergency


def logTraining(logFilePath, step, avgLoss, learningRate, logitRange_str="", windowWeights_str="", gradientNorm_str="", scheduledSamplingProb_str="", epoch_str="", prompt="", guess="", truth="", memoryGates_str="", topTokens_str="", durationLog_str="", otherInfo=""):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logOutput_bg = "".join(typeStyles["dim"]) # Base style for background elements

    logOutput = f"{logOutput_bg}{timestamp} | Step {applyStyle('reset', f'{step:f}'):<6} | LR: {applyStyle('reset', f'{learningRate:.5f}')} | Scheduled Sampling: {applyStyle('reset', f'{scheduledSamplingProb_str}')}" # Reset style for step and LR labels

    lossStyle = getStatStyle("loss", avgLoss)
    logOutput += f"| Avg Loss: {applyStyle(lossStyle, f'{avgLoss:.4f}')}"

    if logitRange_str:
        logitStyle = getStatStyle("logits", float(logitRange_str.split(',')[0].strip()) if logitRange_str else "reset")
        logOutput += f"{logOutput_bg} | Logits: {applyStyle(logitStyle, logitRange_str)}"
    if windowWeights_str:
        windowStyle = getStatStyle("windowWeights", float(windowWeights_str.split(',')[0].strip()) if windowWeights_str else "reset")
        logOutput += f"{logOutput_bg} | Window Weights: {applyStyle(windowStyle, windowWeights_str)}"
    if gradientNorm_str:
        gradNormStyle = getStatStyle("gradNorm", float(gradientNorm_str) if gradientNorm_str else 0.0)
        logOutput += f"{logOutput_bg} | Grad Norm: {applyStyle(gradNormStyle, gradientNorm_str)}"
    if memoryGates_str:
        memGatesStyle = getStatStyle("memGates", float(memoryGates_str.split(',')[0].strip()) if memoryGates_str else "reset")
        logOutput += f"{logOutput_bg} | Memory Gates: {applyStyle(memGatesStyle, memoryGates_str)}"
    if topTokens_str:
        logOutput += f" | Top Tokens: {topTokens_str}"
    if durationLog_str:
        logOutput = logOutput + f"\n{durationLog_str}"

    """displays some extra data when training from prompts"""
    if prompt:
        logOutput = logOutput + f"{logOutput_bg} | Prompt: {applyStyle('reset', prompt)} | Guess: {applyStyle('reset', guess)} | Truth: {applyStyle('reset', truth)}" # Keep prompt/guess/truth reset style
    else:
        logOutput = logOutput
    if otherInfo:
        logOutput += f"{logOutput_bg} | {applyStyle('reset', otherInfo)}"

    logOutput += f"{RESET}" # Ensure reset at the very end of the log line

    print(logOutput)
    with open(logFilePath, "a") as logFile:
        logFile.write(logOutput + "\n")

# Example logTraining (adjust loss and other params as needed)
#logTraining("training.log", 100, 0.03, 0.001)
#logTraining("training.log", 200, 0.1, 0.0005, epoch_str="Epoch 1")
#logTraining("training.log", 300, 0.8, 0.0001, prompt="Translate cat", guess="Meow gato", truth="Cat meow")
#logTraining("training.log", 400, 2.0, 0.00005, otherInfo="Memory issues?")