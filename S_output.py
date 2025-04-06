# CHARIS CAT 2025
from config import *
from datetime import datetime
import shutil
import re

"""TERMINAL CODES"""
RESET = "\033[0m" # normal terminal
BOLD = "\033[1m"
DIM = "\033[2m" # reduces intensity of text colour
UNDERLINE = "\033[4m"
FLASH = "\033[5m"

"""COLOURS!!!"""
PURPLE = "\033[94m"
PURPLE_PALE = "\033[38;5;225m" #256 colour palette
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
    "perfect":       [BOLD, PURPLE_PALE],   #[BOLD, PURPLE],   # 100%
    "almostPerfect": [PURPLE_PALE],         #[PURPLE],         # 90%
    "great":         [BOLD, PURPLE],        #[BOLD, MAGENTA],              # 80%
    "good":          [PURPLE],              #[MAGENTA],         # 70%
    "fine":          [BOLD, MAGENTA],       #[PURPLE_PALE],           # 60%
    "almostFine":    [MAGENTA],             #[PURPLE_PALE, DIM],                 # 50%
    "meh":           [BOLD, BLUE],          #[BLUE],              # 40%
    "bad":           [BLUE],                #[BLUE, DIM],                 # 30%
    "worse":         [DIM, CYAN],           #[PURPLE_PALE, DIM],          # 20%
    "emergency":     [CYAN],                #[DIM],   # 10%

    "reset":         [RESET],               # normal terminal
    "dim":           [RESET, DIM],          # dim style for background elements - arrows, colons, etc.
    "bold":          [BOLD]
}

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

statDisplayMap = {
    "loss": {
        "name": "Loss",
        "type": "avg",
    },
    "gradNorm": {
        "name": "Grad Norm",
        "type": "avg",
    },
    "logitMin": {
        "name": "Logit Min",
        "type": "avg",
    },
    "logitMax": {
        "name": "Logit Max",
        "type": "avg",
    },
    "memoryGate": {
        "name": "Memory Gate",
        "type": "avg",
    },
    "scheduledSampling": {
        "name": "Scheduled Sampling",
        "type": "avg",
    },
    "tokenCount": {
        "name": "Token Count",
        "type": "count",
    },
    "perfectTokenGuess": {
        "name": "Perfect Guess",
        "type": "percentageCount"
    }
}

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
        return "".join(S_codes) + str(text) + RESET
    else:
        return text
    
def S_stripForLogging(text):
    ansi = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi.sub('', text)

def colourPrintTraining(step, inputSentence, guessedSeqStr, targetSeqStr, loss):
    S_bold = "".join(S_types["bold"])
    S_type = S_getStat("loss", loss)
    S_guessedTokens = []
    S_targetTokens = []

    """    every token is in a list, and theres another list that matches
    each token has a list index thing
    so save them all separate instead of into the string
    guessedtoken1 = targettoken1? fancy colour whatever
    guessedtoken2 != targettoken2? plain"""

    for i, token in enumerate(guessedSeqStr):
        if i < len(targetSeqStr) and token == targetSeqStr[i]:
            S_guessedTokens.append(f"{S_bold}{token}{RESET}") # Bold if match, then reset
        else:
            S_guessedTokens.append(f"{S_apply(S_type, token)}") # Apply S_type style if no match

    # Style target tokens
    for i, token in enumerate(targetSeqStr):
        if i < len(guessedSeqStr) and token == guessedSeqStr[i]:
            S_targetTokens.append(f"{S_bold}{DIM}{token}{RESET}") # Bold if match, then reset
        else:
            S_targetTokens.append(f"{DIM}{S_apply(S_type, token)}") # Apply S_type style if no match

    S_guessedSeq_str = "".join(S_guessedTokens).replace("Ġ", " ") # Rejoin styled guessed tokens
    S_targetSeq_str = "".join(S_targetTokens).replace("Ġ", " ") # Rejoin styled target tokens

    fullStringCorrect = (S_guessedSeq_str.strip() == S_targetSeq_str.strip())
    if fullStringCorrect:
        S_type = "perfect"


    S_inputSentence = ''.join(inputSentence).replace("Ġ", " ").strip()
    if len(S_inputSentence) > printPromptLength:
        S_inputSentence = S_inputSentence[-(printPromptLength):]

    formattedWords = (
        f"{S_apply('dim', f'{step}|')}" 
        f"{S_apply('dim', S_inputSentence)}{S_apply('dim', ' → ')}" #{DIM} → {RESET}
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


def logTraining(logFilePath, trainingStepCounter, stats, freq, windowWeights_str="", memoryGates_str="", topTokens_str="", prompt="", guess="", truth="", otherInfo_str=""):

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    S_base = "".join(S_types["dim"]) # Base style for background elements
    delimiter = S_apply(S_base, ' | ')
    logOutput = S_apply(S_base, str(timestamp)) + (delimiter)
    logOutput += S_apply(S_base, f"{trainingStepCounter:.0f}") + delimiter
    logOutput += S_apply(S_base, f'LR{learningRate}') + (delimiter)

    # Compute all averages
    avgStats = {k: ((raw/freq) if freq != 0 else 0) if k not in ["tokenCount"] else raw for (k, raw) in stats.items()}

    logOutput += delimiter.join([
        S_apply('dim', key + ":") + S_apply(S_getStat(key, value), f"{value:.4f}")
        for key, value in 
        (avgStats).items()
        if (value != "" and value is not None)
    ])

    if windowWeights_str:
        logOutput += delimiter
        logOutput += f"windowWeights{S_apply('reset', windowWeights_str)}"
    if memoryGates_str:
        logOutput += delimiter
        logOutput += f"memoryGates{S_apply('reset', memoryGates_str)}"
    if topTokens_str:
        logOutput += delimiter
        logOutput += f"topTokens{S_apply('reset', topTokens_str)}"

    """displays some extra data when training from prompts"""
    if prompt:
        logOutput = logOutput + f"{S_base} | Prompt: {S_apply('reset', prompt)} | Guess: {S_apply('reset', guess)} | Truth: {S_apply('reset', truth)}" # Keep prompt/guess/truth reset style
    if otherInfo_str:
        logOutput += f"{S_base} | {S_apply(S_base, otherInfo_str)}"

    logOutput += f"{RESET}"
    print(logOutput)
    with open(logFilePath, "a") as logFile:
        logFile.write(S_stripForLogging(logOutput) + "\n")

    #HUD_fixScroll(self)

if __name__ == "__main__":
    print(S_apply('perfect', "ELODIE IS PERFECT"))
    print(S_apply('almostPerfect', "BABYLLM IS ALMOST PERFECT"))
    print(S_apply('great', "BABYLLM IS GREAT"))
    print(S_apply('good', "BABYLLM IS GOOD"))
    print(S_apply('fine', "BABYLLM IS FINE"))
    print(S_apply('almostFine', "CHARIS IS ALMOST FINE"))
    print(S_apply('meh', "BABYLLM IS MEH"))
    print(S_apply('bad', "BABYLLM IS BAD"))
    print(S_apply('worse', "GEORGE IS WORSE"))
    print(S_apply('emergency', "BABYLLM IS EMERGENCY"))
