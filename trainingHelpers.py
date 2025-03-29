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
GOLD = "\033[93m"
RED = "\033[38;5;124m" #256 colour palette
RED_BRIGHT = "\033[91m"
ORANGE = "\033[38;5;52m" #256 colour palette

"""extra colours"""
RED_ALT = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
PURPLE_ALT = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

"""TERMINAL OUTPUT STYLES - CATEGORY MAPPING"""
CATEGORY_STYLES = {
    "PERFECT":      {"step": GOLD,        "arrow": GOLD,         "guess": GOLD,         "truth": GOLD,         "loss": GOLD,         "attribute": BOLD},
    "ALMOST_PERFECT":{"step": DIM,        "arrow": PURPLE_PALE,  "guess": PURPLE_PALE,  "truth": PURPLE_PALE,  "loss": PURPLE_PALE,  "attribute": RESET},
    "VERY_GOOD":    {"step": DIM,         "arrow": PURPLE,       "guess": PURPLE,       "truth": PURPLE,       "loss": PURPLE,       "attribute": RESET},
    "OK":           {"step": DIM,         "arrow": RESET,        "guess": WHITE,        "truth": WHITE,        "loss": RESET,        "attribute": RESET},
    "BAD":          {"step": DIM,         "arrow": ORANGE,       "guess": WHITE,        "truth": WHITE,        "loss": ORANGE,       "attribute": RESET},
    "WORSE":        {"step": DIM,         "arrow": RED,          "guess": WHITE,        "truth": WHITE,        "loss": RED,          "attribute": RESET},
    "SHIT":         {"step": DIM,         "arrow": YELLOW,       "guess": YELLOW,       "truth": YELLOW,       "loss": YELLOW,       "attribute": BOLD},
    "EMERGENCY":    {"step": RED_BRIGHT,  "arrow": RED_BRIGHT,   "guess": RED_BRIGHT,   "truth": RED_BRIGHT,   "loss": RED_BRIGHT,   "attribute": BOLD + FLASH},
    "DEFAULT":      {"step": DIM,         "arrow": RESET,        "guess": WHITE,        "truth": WHITE,        "loss": RESET,        "attribute": RESET}
}

def colourPrintTraining(step, inputSentence, guessedSeqStr, targetSeqStr, loss, isCorrect, isPerfect):
    if isPerfect:
        formattedWords = f"{GOLD} Step {step}: {inputSentence}{RESET}{DIM} → {RESET}{GOLD}{guessedSeqStr}{RESET}{DIM}[!] {RESET}{GOLD}{targetSeqStr}{RESET}{DIM} | {RESET}{GOLD}Loss: {loss:.3f} {RESET}"
    elif isCorrect and loss < veryLowLoss:  # correct, very low loss
        formattedWords = f"{DIM}Step {step}: {RESET}{PURPLE}{inputSentence}{RESET}{DIM} → {RESET}{PURPLE}{guessedSeqStr}{RESET}{DIM}[!] {RESET}{PURPLE}{targetSeqStr}{RESET}{DIM} | {RESET}{PURPLE}Loss: {loss:.3f}{RESET}"
    elif isCorrect and loss < lowLoss:  # correct, low loss
        formattedWords = f"{DIM}Step {step}: {RESET}{PURPLE_PALE}{inputSentence}{RESET}{DIM} → {RESET}{PURPLE}{guessedSeqStr}{RESET}{DIM}[!] {RESET}{PURPLE}{targetSeqStr}{RESET}{DIM} | {RESET}{PURPLE_PALE}Loss: {loss:.3f}{RESET}"
    elif loss > superHighLoss:  # super high loss
        formattedWords = f"{DIM}Step {step}: {inputSentence} → {guessedSeqStr}[?] {targetSeqStr} | {RESET}{RED_BRIGHT}Loss: {loss:.3f}{RESET}"
    elif loss > highLoss:  # high loss
        formattedWords = f"{DIM}Step {step}: {inputSentence} → {guessedSeqStr}[?] {targetSeqStr} | {RESET}{RED}Loss: {loss:.3f}{RESET}"
    elif loss > prettyHighLoss:  # pretty high loss
        formattedWords = f"{DIM}Step {step}: {inputSentence} → {guessedSeqStr}[?] {targetSeqStr} | {RESET}{ORANGE}Loss: {loss:.3f}{RESET}"
    elif loss < veryLowLoss:  # incorrect, very low loss
        formattedWords = f"{DIM}Step {step}: {inputSentence} → {guessedSeqStr}[?] {targetSeqStr} | {RESET}{PURPLE}Loss: {loss:.3f}{RESET}"
    elif loss < lowLoss:  # incorrect, low loss
        formattedWords = f"{DIM}Step {step}: {inputSentence} → {guessedSeqStr}[?] {targetSeqStr} | {RESET}{PURPLE}Loss: {loss:.3f}{RESET}"
    elif isCorrect:  # correct, normal loss
        formattedWords = f"{DIM}Step {step}: {RESET}{PURPLE_PALE}{inputSentence}{RESET}{DIM} → {RESET}{PURPLE}{guessedSeqStr}{RESET}{DIM}[!]  {RESET}{PURPLE}{targetSeqStr}{RESET} {DIM}| Loss: {loss:.3f}{RESET}"
    else:  # default
        formattedWords = f"{DIM}Step {step}: {inputSentence} → {guessedSeqStr}[?] {targetSeqStr} | Loss: {loss:.3f}{RESET}"

    print(formattedWords)

def logTraining(logFilePath, step, avgLoss, learningRate, logitRange_str="", windowWeights_str="", gradientNorm_str="", scheduledSamplingProb_str="", epoch_str="", prompt="", guess="", truth="", memoryGates_str="", otherInfo=""):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logOutput = f"{timestamp} | Step {step:<6} | LR: {learningRate:.5f} | Avg Loss: {avgLoss:.4f}"
    if epoch_str:
        logOutput += f" | Epoch: {epoch_str}"
    if logitRange_str:
        logOutput += f" | Logits: {logitRange_str}"
    if windowWeights_str:
        logOutput += f" | Weights: {windowWeights_str}"
    if gradientNorm_str:
        logOutput += f" | Grad Norm: {gradientNorm_str}"
    if scheduledSamplingProb_str:
        logOutput += f" | Scheduled Sampling: {scheduledSamplingProb_str}"
    if memoryGates_str:
        logOutput += f" | Memory Gates: {memoryGates_str}"
    """displays some extra data when training from prompts"""
    if prompt:
        logOutputDetail = logOutput + f" | Prompt: {prompt} | Guess: {guess} | Truth: {truth}"
    else:
        logOutputDetail = logOutput
    if otherInfo:
        logOutputDetail += f" | {otherInfo}"

    print(logOutputDetail)
    with open(logFilePath, "a") as logFile:
        logFile.write(logOutputDetail + "\n")