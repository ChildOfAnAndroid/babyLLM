# CHARIS CAT 2025
# BABYLLM - S_output.py

from config import *
from datetime import datetime
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

S_statThresholds = {
    "loss":     {"perfect": 0.21875, "almostPerfect": 0.4375, "great": 0.875, "good": 1.75, "fine": 3.5, "almostFine": 3.75, "meh": 7.5, "bad": 15.0, "worse": 30.0, "emergency": float('inf')},
    "logits":   { "perfect": 0.99, "almostPerfect": 0.95, "great": 0.90, "good": 0.80, "fine": 0.70, "almostFine": 0.60, "meh": 0.50, "bad": 0.40, "worse": 0.30, "emergency": 0.0},
    "gradNorm": {"perfect": 0.99, "almostPerfect": 0.95, "great": 0.90, "good": 0.80, "fine": 0.70, "almostFine": 0.60, "meh": 0.50, "bad": 0.40, "worse": 0.30, "emergency": 0.0}, 
}

def S_getStat(statType, statVal):
    thresholds = S_statThresholds.get(statType)
    if not thresholds: return "reset"
    for label, limit in thresholds.items():
        if statVal <= limit: return label
    return "emergency"

def S_apply(S_type, text):
    return "".join(S_types.get(S_type, [])) + str(text) + RESET
    
def S_stripForLogging(text):
    return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)

def S_colourPrintTraining(step, inputSeq, guessedSeq_str, targetSeq_str, loss, recentLoss=None, totalLoss=None, totalTokenCount=None):
    S_type = S_getStat("loss", loss)
    S_bold = "".join(S_types["bold"])

    guess = [f"{S_bold}{t}{RESET}" if i < len(targetSeq_str) and t == targetSeq_str[i] else S_apply(S_type, t) for i, t in enumerate(guessedSeq_str)]
    truth  = [f"{S_bold}{DIM}{t}{RESET}" if i < len(guessedSeq_str) and t == guessedSeq_str[i] else f"{DIM}{S_apply(S_type, t)}" for i, t in enumerate(targetSeq_str)]

    guess_str = "".join(guess).replace("Ġ", " ")
    truth_str = "".join(truth).replace("Ġ", " ")
    match = guess_str.strip() == truth_str.strip()
    if match: S_type = "perfect"

    prompt_str = ''.join(inputSeq).replace("Ġ", " ").strip()[-printPromptLength:]
    delta_str = ""

    # Calculate delta
    if recentLoss is not None:
        delta = recentLoss - loss
        delta_str = f"{S_apply('dim', 'Δ')}{S_apply(S_type, f'{delta:+.3f}')}{'↑' if delta < 0 else '↓'}"

    print(f"{S_apply('dim', f'{step}')}|{S_apply('dim', prompt_str)}|{S_apply('dim', 'loss: ')}{S_apply(S_type, f'{loss:.3f}')}{S_apply('dim', '/1 ')}"
          + (f"{S_apply(S_type, f'{recentLoss:.3f}')}{S_apply('dim', f'/{printFreq} ')}" if recentLoss else "")
          + delta_str + "|\n"
          + f"{S_apply('dim', 'guess → ')}{guess_str}{S_apply(S_type, ' [!] ') if match else S_apply('dim', ' [?] ')}\n"
          + f"{S_apply('dim', 'truth → ')}{truth_str}{S_apply('dim', ' | ')}")

def S_logTraining(trainingLogPath, trainingStepCounter, stats, freq, windowWeights_str="", memoryGates_str="", topTokens_str="", prompt="", guess="", truth="", otherInfo_str=""):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    delimiter = S_apply("dim", " | ")

    doNotAverage = ["tokenCount", "topWindowWeight", "windowEntropy", "effectiveWindowCount", "windowStd", "memoryGateMean", "memoryGateStd"]
    avgStats = {k: raw if k in doNotAverage else (raw / freq if freq else 0) for k, raw in stats.items()}

    logOutput = delimiter.join([S_apply("dim", timestamp), S_apply("dim", f"{trainingStepCounter:.0f}"), S_apply("dim", f"LR{learningRate}")])

    logOutput += delimiter + delimiter.join([S_apply("dim", f"{k}:")
               + S_apply(S_getStat(k, v), f"{v:.4f}") for k, v in avgStats.items() if v not in (None, "")])

    if windowWeights_str: logOutput += delimiter + f"windowWeights{S_apply('reset', windowWeights_str)}"
    if memoryGates_str: logOutput += delimiter + f"memoryGates{S_apply('reset', memoryGates_str)}"
    if topTokens_str: logOutput += delimiter + f"topTokens{S_apply('reset', topTokens_str)}"

    if prompt: logOutput += f"{delimiter}prompt → {S_apply('reset', prompt)} | guess → {S_apply('reset', guess)} | truth → {S_apply('reset', truth)}"
    if otherInfo_str: logOutput += f"{delimiter}{S_apply('reset', otherInfo_str)}"

    print(logOutput + RESET)
    with open(trainingLogPath, "a") as f: f.write(S_stripForLogging(logOutput) + "\n")

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
