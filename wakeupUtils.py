# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM WAKEUP UTILS // wakeupUtils.py
# v1.1

import os
import random
import traceback
from datetime import datetime
from config import *


def handle_exception(exc_type, exc_value, exc_traceback):
    if not issubclass(exc_type, KeyboardInterrupt):
        print("[RIP ʕっₓᴥₓʔっ] Uncaught Exception:")
        traceback.print_exception(exc_type, exc_value, exc_traceback)

def append_to_files(text, *paths, encoding="utf-8"):
    for path in paths:
        with open(path, "a", encoding=encoding) as logFile: logFile.write(text)

def setStartIndex():
    if os.path.exists(stepCheckpointFilePath):
        with open(stepCheckpointFilePath, "r") as f:
            try:
                savedStep = int(f.read().strip())
            except ValueError:
                babyNote_loadCheckpoint = (
                    f"{babyName} 'oh. i couldn't load step checkpoint file from {stepCheckpointFilePath}, resetting to 0...' "
                )
                print(babyNote_loadCheckpoint)
                savedStep = 0
    else:
        babyNote_loadCheckpoint = (
            f"{babyName} 'ah, the step checkpoint file {stepCheckpointFilePath} doesn't exist, resetting to 0...' "
        )
        print(babyNote_loadCheckpoint)
        savedStep = 0

    return savedStep + trainingStartIndex


def checkLossCheckpoint():
    if os.path.exists(lossCheckpointFilePath):
        with open(lossCheckpointFilePath, "r") as f:
            try:
                lastTurnLoss = float(f.read().strip())
            except ValueError:
                babyNote_loadLossCheckpoint = (f"{babyName} 'noooo! i couldn't load loss checkpoint file from {lossCheckpointFilePath}, resetting to 0...' ")
                print(babyNote_loadLossCheckpoint)
                lastTurnLoss = 0
    else:
        babyNote_loadLossCheckpoint = (f"{babyName} 'right, well, the loss checkpoint file {lossCheckpointFilePath} doesn't actually exist... so i'll reset it to 0.' ")
        print(babyNote_loadLossCheckpoint)
        lastTurnLoss = 0

    return lastTurnLoss


def openingQuestions(_counsellor, _librarian, _windowMAX, _first):
    counsellor = _counsellor
    with counsellor.infodump("openingQuestions") as ʕっʘ‿ʘʔっ:
        librarian = _librarian
        if debugPrints:
            ʕっʘ‿ʘʔっ("setStartIndex")
        newStartIndex = setStartIndex()
        lastRunLoss = checkLossCheckpoint()
        mode = "train"

        babyNote_loadCheckpointCheck = (f"[{babyName}]: right, last time i got to step {newStartIndex} and my average loss was {lastRunLoss}... want to restart from there?")
        if debugPrints:
            ʕっʘ‿ʘʔっ("choice = input♥")
        if _first:
            choice = input(babyNote_loadCheckpointCheck + f"\n[{userName}]: ").lower()
        else:
            choice = "yes"

        userNote_loadCheckpoint = f"[{userName}]: {choice}"

        if choice == "" or choice.startswith("y"):
            if debugPrints:
                ʕっʘ‿ʘʔっ("♥choice = y")
            startIndex = newStartIndex
            babyNote_loadCheckpoint = (
                f"[{babyName}]: ok! let's go to step {newStartIndex}!"
            )
            print(babyNote_loadCheckpoint, end="")

        elif choice.startswith("r") or choice in [
            "random",
            "i dont care",
            "i don't care",
            "idc",
        ]:
            if debugPrints:
                ʕっʘ‿ʘʔっ("♥choice = r")
            newStartIndex = random.randint(0, len(librarian.tokens) - _windowMAX - 1)
            startIndex = newStartIndex
            babyNote_loadCheckpoint = (f"[{babyName}]: oh, cool! i'll pick a random spot to start from... umm... let's go to step {newStartIndex}!")
            print(babyNote_loadCheckpoint, end="")

        elif choice.startswith("n") or choice in ["start again", "restart"]:
            if debugPrints:
                ʕっʘ‿ʘʔっ("♥choice = n")
            startIndex = newStartIndex
            babyNote_loadCheckpoint = (f"[{babyName}]: alright, step {newStartIndex}, let's go back to the beginning :)")
            print(babyNote_loadCheckpoint, end="")

        elif choice.isdigit():
            if debugPrints:
                ʕっʘ‿ʘʔっ("♥choice = digit")
            newStartIndex = int(choice)
            startIndex = newStartIndex
            babyNote_loadCheckpoint = (f"[{babyName}] damn that's specific! heading to step {newStartIndex}...")
            print(babyNote_loadCheckpoint, end="")

        else:
            if debugPrints:
                ʕっʘ‿ʘʔっ("♥choice = None")
            startIndex = newStartIndex
            babyNote_loadCheckpoint = (f"[{babyName}] umm... i don't think i heard you properly, i'll just start from step {newStartIndex} :) but,")
            print(babyNote_loadCheckpoint, end="")

        if debugPrints:
            ʕっʘ‿ʘʔっ("runStart")
        printStartLogs(
            babyNote_loadCheckpointCheck,
            userNote_loadCheckpoint,
            babyNote_loadCheckpoint,
            _first=_first,
            _windowMAX=_windowMAX,
        )

    return startIndex

def printStartLogs(
    _babyNote_loadCheckpointCheck,
    _userNote_loadCheckpoint,
    _babyNote_loadCheckpoint,
    _first,
    _windowMAX,
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    babyNote_runStart = (
        " what am i learning today?"
    )
    if _first:
        userInput = input(babyNote_runStart + f"\n[{userName}] ").strip().lower()
    else:
        userInput = f"numTokens = {_windowMAX}"

    userNote_runStart = f"[{userName}] " + userInput + ""
    notesString = (
        f"--- {timestamp} --- \n{_babyNote_loadCheckpointCheck}\n{_userNote_loadCheckpoint}\n{_babyNote_loadCheckpoint}{babyNote_runStart}\n{userNote_runStart}"
    )
    print(notesString)
    append_to_files(notesString, chatLogPath_forHumans, trainingLogPath_100, trainingLogPath_1000, chatLogPath_trainingLog,)
