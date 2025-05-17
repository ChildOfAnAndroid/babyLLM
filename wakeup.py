# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 

from rich.traceback import install
#from torch.profiler import profile, record_function, ProfilerActivity
import sys, traceback, warnings, torch, os, random
from datetime import datetime

from babyLLM import BABYLLM
from SCHOOL.staffroom.counsellor import COUNSELLOR
from SCHOOL.staffroom.calligraphist import S_OUTPUT
from SCHOOL.staffroom.librarian import LIBRARIAN
from SCHOOL.staffroom.HE_IS_SCRIBE import SCRIBE
from SCHOOL.staffroom.tutor import TUTOR
# from BRAIN.LAYERS.sensoryWobble import WOBBLE
# from SCHOOL.staffroom.newsletter import STATS
from config import *

def handle_exception(exc_type, exc_value, exc_traceback):
    if not issubclass(exc_type, KeyboardInterrupt):
        print("[RIP ʕっₓᴥₓʔっ] Uncaught Exception:")
        traceback.print_exception(exc_type, exc_value, exc_traceback)

sys.excepthook = handle_exception
warnings.simplefilter("default") # show all warnings (PyTorch hides some by default)
install(show_locals = True)
torch.autograd.set_detect_anomaly(mode = anomalyDetect, check_nan = debugPrints)

def wakeup(windowMAX, dataStride, first = True):
    try:
        # WAKE UP THE SCHOOL :)
        counsellor              = COUNSELLOR("babyLLM", _debug = debugPrints, _durations = durationLogging)
        with counsellor.infodump("wakeup") as ʕっʘ‿ʘʔっ:

            # OPEN THE LIBRARY :)
            ʕっʘ‿ʘʔっ("waking the librarian...")
            librarian           = LIBRARIAN (_counsellor = counsellor, _baseTokenizerPath = None, _forceRetrain = False) #_baseTokenizerPath = "BRAIN/vocabCache/2000_20/tokenizer_2000.json", _forceRetrain = True)

            if False: exit(0)
            ʕっʘ‿ʘʔっ("opening questions...")
            newStartIndex       = openingQuestions(_counsellor = counsellor, _librarian = librarian, _windowMAX = windowMAX, _first = first)

            ʕっʘ‿ʘʔっ("generating training data pairs...")
            trainingDataPairs   =           librarian.genTrainingData(_windowMAX = windowMAX, _startIndex = newStartIndex, _stride = dataStride)
            if debugPrints:                 print(f"Total trainingDataPairs: {len(trainingDataPairs)}")

            ʕっʘ‿ʘʔっ("loading chaos agents...")
            calligraphist       = S_OUTPUT  (_counsellor                = counsellor)

            scribe              = SCRIBE    (_counsellor                = counsellor, 
                                                _calligraphist          = calligraphist, 
                                                _librarian              = librarian,
                                                _numTokensPerStep       = windowMAX,
                                                )
            
            # WAKE UP THE BABY :)
            ʕっʘ‿ʘʔっ("loading babyLLM...")
            babyLLM             = BABYLLM   (_counsellor                = counsellor,
                                                _calligraphist          = calligraphist, 
                                                _scribe                 = scribe,
                                                _librarian              = librarian, 
                                                _device                 = modelDevice,
                                                _numTokensPerStep       = windowMAX,
                                                _first                  = first)

            tutor               = TUTOR     (_counsellor                = counsellor,
                                                _calligraphist          = calligraphist, 
                                                _scribe                 = scribe,
                                                _librarian              = librarian, 
                                                _model                  = babyLLM,
                                                _device                 = modelDevice,
                                                _numTokensPerStep       = windowMAX,
                                                _first                  = first)
            
            babyLLM.loadModel()
            babyLLM.to(modelDevice)

            # START THE LESSONS :)
            ʕっʘ‿ʘʔっ("starting lessons!")
            tutor.trainModel                (_trainingDataPairs = trainingDataPairs, _epochs = epochs, _startIndex = newStartIndex)
            return tutor.totalAvgLoss

    except Exception as e:
        print(f"[RIP ʕっₓᴥₓʔっ]")
        raise
    except KeyboardInterrupt: #as k
        for name, p in babyLLM.named_parameters():
            if p.grad is None:
                print(f"keyboard interrupt = {babyLLM.calligraphist.S_apply("emergency", f"NO GRAD: {name}")}")
            else: 
                grad = p.grad
                shape = tuple(grad.shape)
                norm = grad.norm().item()
                nonzero = grad.count_nonzero().item()
                total = grad.numel()
                sparsity = 1 - (nonzero / total)
                mean = grad.mean().item()
                std = grad.std().item()
                print(f"keyboard interrupt = {babyLLM.calligraphist.S_apply("almostPerfect", f"yes grad: {name} | shape: {shape} | norm: {norm:.4f} | sparsity: {sparsity:.2%} | mean: {mean:.4f} | std: {std:.4f}")}")
        ʕっʘ‿ʘʔっ("♥keyboardInterrupt")
        if tutor.trainingStepCounter:
            step = tutor.trainingStepCounter
            totalAvgLoss = tutor.totalAvgLoss
        else:
            step = 1
        choice = input("save, cancel (do not save before exit), restart or interact?" + f"\n{userName}: ").lower()
        if choice in ("save", "") or choice.startswith("s"): 
            ʕっʘ‿ʘʔっ("♥choice = s")
            babyLLM.saveModel(_newStartIndex = newStartIndex, _trainingStepCounter = step, _totalAvgLoss = totalAvgLoss, _first = first)
            print("\nit's rude to interrupt people.. but, bye bye! :)")
        elif choice == "cancel" or choice.startswith("c"): 
            ʕっʘ‿ʘʔっ("♥choice = c")
            print("\nhey! i wanted to remember that! :(")
        elif choice == "interact" or choice.startswith("i"):
            ʕっʘ‿ʘʔっ("♥choice = i")
            babyLLM.saveModel(_newStartIndex = newStartIndex, _trainingStepCounter = step, _totalAvgLoss = totalAvgLoss, _first = first)
            import code
            print("try:\nbabyLLM.stats\nbabyLLM.scheduledSampling\nbabyLLM.memory.memory\nbabyLLM.interneuronNetwork.cerebellum\nbabyLLM.logits.forward(...)\nUse `exit()` to return to terminal.\n")
            code.interact(local = locals())
        elif choice == "restart" or choice.startswith("r"):
            ʕっʘ‿ʘʔっ("♥choice = r")
            babyLLM.saveModel(_newStartIndex = newStartIndex, _trainingStepCounter = step, _totalAvgLoss = totalAvgLoss, _first = first)
            print("you spin me right round, babyllm, right round...")
            return totalAvgLoss
        else: 
            ʕっʘ‿ʘʔっ("♥choice = None")
            babyLLM.saveModel(_newStartIndex = newStartIndex, _trainingStepCounter = step, _totalAvgLoss = totalAvgLoss, _first = first)
            print("\nuhh... i'm confused, but i saved anyway!")
        if modelDevice.type == 'mps':
            torch.mps.empty_cache()
            print(f"cache emptied")
        exit(8)

def setStartIndex():
    if os.path.exists(stepCheckpointFilePath):
        with open(stepCheckpointFilePath, "r") as f:
            try: savedStep = int(f.read().strip())
            except ValueError:
                babyNote_loadCheckpoint = f"{babyName} 'oh. i couldn't load step checkpoint file from {stepCheckpointFilePath}, resetting to 0...' "
                print(babyNote_loadCheckpoint)
                savedStep = 0
    else:
        babyNote_loadCheckpoint = f"{babyName} 'ah, the step checkpoint file {stepCheckpointFilePath} doesn't exist, resetting to 0...' "
        print(babyNote_loadCheckpoint)
        savedStep = 0

    savedStartIndex = savedStep + trainingStartIndex
    
    return savedStartIndex

def checkLossCheckpoint():
    if os.path.exists(lossCheckpointFilePath):
        with open(lossCheckpointFilePath, "r") as f:
            try: lastTurnLoss = int(f.read().strip())
            except ValueError:
                babyNote_loadLossCheckpoint = f"{babyName} 'noooo! i couldn't load loss checkpoint file from {lossCheckpointFilePath}, resetting to 0...' "
                print(babyNote_loadLossCheckpoint)
                lastTurnLoss = 0
    else:
        babyNote_loadLossCheckpoint = f"{babyName} 'right, well, the loss checkpoint file {lossCheckpointFilePath} doesn't actually exist... so i'll reset it to 0.' "
        print(babyNote_loadLossCheckpoint)
        lastTurnLoss = 0
    
    return lastTurnLoss

def openingQuestions(_counsellor, _librarian, _windowMAX, _first):
    counsellor = _counsellor
    with counsellor.infodump("babyLLM") as ʕっʘ‿ʘʔっ:
        librarian = _librarian
        #babyLLM.to(modelDevice)
        ʕっʘ‿ʘʔっ("setStartIndex")
        newStartIndex = setStartIndex()
        lastRunLoss   = checkLossCheckpoint()

        babyNote_loadCheckpointCheck = f"[{babyName}] right, last time i got to step {newStartIndex} and my average loss was {lastRunLoss}... want to restart from there?"
        ʕっʘ‿ʘʔっ("choice = input♥")
        if _first:
            choice = input(babyNote_loadCheckpointCheck + f"\n[{userName}] ").lower()
        else:
            choice = "yes"
        userNote_loadCheckpoint = f"[{userName}] {choice}"

        if choice == "" or choice.startswith("y"):
            ʕっʘ‿ʘʔっ("♥choice = y")
            startIndex = newStartIndex
            babyNote_loadCheckpoint = f"[{babyName}] ok! let's go to step {newStartIndex}!"
            print(babyNote_loadCheckpoint, end="")

        elif choice.startswith("r") or choice in ["random", "i dont care", "i don't care", "idc"]:
            ʕっʘ‿ʘʔっ("♥choice = r")
            newStartIndex = random.randint(0, len(librarian.tokens) - _windowMAX - 1)
            startIndex = newStartIndex
            babyNote_loadCheckpoint = f"[{babyName}] oh, cool! i'll pick a random spot to start from... umm... let's go to step {newStartIndex}!"
            print(babyNote_loadCheckpoint, end="")

        elif choice.startswith("n") or choice in ["start again", "restart"]:
            ʕっʘ‿ʘʔっ("♥choice = n")
            startIndex = newStartIndex
            babyNote_loadCheckpoint = f"[{babyName}] alright, step {newStartIndex}, let's go back to the beginning :)"
            print(babyNote_loadCheckpoint, end="")
            
        elif choice.isdigit():
            ʕっʘ‿ʘʔっ("♥choice = digit")
            newStartIndex = int(choice)
            startIndex = newStartIndex
            babyNote_loadCheckpoint = f"[{babyName}] damn that's specific! heading to step {newStartIndex}..."
            print(babyNote_loadCheckpoint, end="")

        else:
            ʕっʘ‿ʘʔっ("♥choice = None")
            startIndex = newStartIndex
            babyNote_loadCheckpoint = f"[{babyName}] umm... i don't think i heard you properly, i'll just start from step {newStartIndex} :) but,"
            print(babyNote_loadCheckpoint, end="")

        ʕっʘ‿ʘʔっ("runStart")
        printStartLogs(babyNote_loadCheckpointCheck, userNote_loadCheckpoint, babyNote_loadCheckpoint, _first = _first, _windowMAX = _windowMAX)

    return startIndex

def printStartLogs(_babyNote_loadCheckpointCheck, _userNote_loadCheckpoint, _babyNote_loadCheckpoint, _first, _windowMAX):
    #ʕっʘ‿ʘʔっ("♥bootPrints") # BOOT PRINTS TO TXT AND TERMINAL
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    babyNote_runStart = f" what am i learning today?" # no tag of 'babyllm:' because it merges with the end of above message in logs
    if _first:
        userInput = input(babyNote_runStart + f"\n[{userName}] ").strip().lower()
    else:
        userInput = f"numTokens = {_windowMAX}"

    userNote_runStart = f"[{userName}] " + userInput + ""
    notesString = f"--- {timestamp} --- \n{_babyNote_loadCheckpointCheck}\n{_userNote_loadCheckpoint}\n{_babyNote_loadCheckpoint}{babyNote_runStart}\n{userNote_runStart}"
    print(notesString)
    #ʕっʘ‿ʘʔっ("♥printStartLogs")
    with open(chatLogPath_forHumans, "a") as logFile: logFile.write(notesString)
    with open(trainingLogPath_100, "a") as logFile: logFile.write(notesString)
    with open(trainingLogPath_1000, "a") as logFile: logFile.write(notesString)
    with open(chatLogPath_trainingLog, "a") as logFile: logFile.write(notesString)

def main():
    windowMAX           = numTokensPerStepSTART
    dataStride          = trainingDataStride
    maxTokensPerStep    = 512 
    lastRunLoss         = checkLossCheckpoint()
    while windowMAX <= maxTokensPerStep:
        print(f"\n--- STARTING NEW TRAINING LOOP ---")
        print(f"numTokensPerStep = {windowMAX}")

        thisRunLoss = wakeup(windowMAX, dataStride, first = (windowMAX == numTokensPerStepSTART))

        print(f"BEFORE UPDATE: thisRunLoss = {thisRunLoss}, lastRunLoss = {lastRunLoss}, windowMAX = {windowMAX}, dataStride = {dataStride}")
        if thisRunLoss > lastRunLoss:
            if random.choice([True, False]):
                windowMAX += 1
            else:
                dataStride += 1
        elif windowMAX > 1:
            windowMAX -= 1
            dataStride -= 1
        
        dataStride = max(1, max(dataStride, windowMAX - 1))

        lastRunLoss = thisRunLoss
        print(f"AFTER UPDATE: thisRunLoss = {thisRunLoss}, lastRunLoss = {lastRunLoss}, windowMAX = {windowMAX}, dataStride = {dataStride}")

if __name__ == "__main__":
    main()