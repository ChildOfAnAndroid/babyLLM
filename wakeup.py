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
from config import *
from babyBotTMP import BABYBOT

def handle_exception(exc_type, exc_value, exc_traceback):
    if not issubclass(exc_type, KeyboardInterrupt):
        print("[RIP ʕっₓᴥₓʔっ] Uncaught Exception:")
        traceback.print_exception(exc_type, exc_value, exc_traceback)

sys.excepthook = handle_exception
warnings.simplefilter("default") # show all warnings (PyTorch hides some by default)
install(show_locals = True)
torch.autograd.set_detect_anomaly(mode = anomalyDetect, check_nan = debugPrints)

def wakeup(windowMAX, dataStride, passRateSTART, lrGoal = learningRateGOAL, trainingDataPairNum = trainingDataPairNumber, log_A = trainingLogFreq_A, totalTurnsAwake = 0, totalRuns = 0, first = True, mode = "train"):
    try:
        # WAKE UP THE SCHOOL :)
        counsellor              = COUNSELLOR("babyLLM", _debug = debugPrints, _durations = durationLogging)
        with counsellor.infodump("wakeup") as ʕっʘ‿ʘʔっ:

            # OPEN THE LIBRARY :)
            if debugPrints: ʕっʘ‿ʘʔっ("waking the librarian...")
            librarian           = LIBRARIAN (_counsellor = counsellor, _baseTokenizerPath = None, _forceRetrain = False) #_baseTokenizerPath = "BRAIN/vocabCache/2000_20/tokenizer_2000.json", _forceRetrain = True)

            if False: exit(0)
            #if debugPrints: ʕっʘ‿ʘʔっ("opening questions...")
            #newStartIndex       = openingQuestions(_counsellor = counsellor, _librarian = librarian, _windowMAX = windowMAX, _first = first)

            #if debugPrints: ʕっʘ‿ʘʔっ("generating training data pairs...")
            #trainingDataPairs   =           librarian.genTrainingData(_windowMAX = windowMAX, _trainingDataPairNumber = trainingDataPairNum, _startIndex = newStartIndex, _stride = dataStride)
            #if debugPrints:                 print(f"Total trainingDataPairs: {len(trainingDataPairs)}")

            if debugPrints: ʕっʘ‿ʘʔっ("loading chaos agents...")
            calligraphist       = S_OUTPUT  (_counsellor                = counsellor)

            scribe              = SCRIBE    (_counsellor                = counsellor, 
                                                _calligraphist          = calligraphist, 
                                                _librarian              = librarian,
                                                _numTokensPerStep       = windowMAX,
                                                )
            
            # WAKE UP THE BABY :)
            if debugPrints: ʕっʘ‿ʘʔっ("loading babyLLM...")
            babyLLM             = BABYLLM   (_counsellor                = counsellor,
                                                _calligraphist          = calligraphist, 
                                                _scribe                 = scribe,
                                                _librarian              = librarian, 
                                                _device                 = modelDevice,
                                                _numTokensPerStep       = windowMAX,
                                                _first                  = first,
                                                _learningRateGOAL       = lrGoal,)

            tutor               = TUTOR     (_counsellor                    = counsellor,
                                                _calligraphist              = calligraphist, 
                                                _scribe                     = scribe,
                                                _librarian                  = librarian, 
                                                _model                      = babyLLM,
                                                _device                     = modelDevice,
                                                _numTokensPerStep           = windowMAX,
                                                _dataStride                 = dataStride,
                                                _first                      = first,
                                                _lastRunLoss                = checkLossCheckpoint(),
                                                _totalTurnsAwake            = totalTurnsAwake,
                                                _totalRuns                  = totalRuns,
                                                _perfectionistPassRateSTART = passRateSTART,
                                                _trainingLogFreq_A          = log_A,)
            
            if mode == "bot":
                print("--- LAUNCHING TWITCH BOT ---")
                if debugPrints: ʕっʘ‿ʘʔっ("starting twitch bot!")
                # create a bot instance, pass in the staff etc
                babyBot = BABYBOT(babyLLM, tutor, librarian, scribe, calligraphist)
                babyLLM.loadModel()
                babyLLM.to(modelDevice)
                babyBot.run()

            elif mode == "train":
                print("--- STARTING OFFLINE TRAINING ---")
                newStartIndex = openingQuestions(_counsellor=counsellor, _librarian=librarian, _windowMAX=windowMAXSTART, _first=True)
                trainingDataPairs = librarian.genTrainingData(_windowMAX=windowMAXSTART, _trainingDataPairNumber=trainingDataPairNumber, _startIndex=newStartIndex, _stride=trainingDataStride)
                
                # START THE LESSONS :)
                babyLLM.loadModel()
                babyLLM.to(modelDevice)
                if debugPrints: ʕっʘ‿ʘʔっ("starting lessons!")
                tutor.trainModel(_trainingDataPairs = trainingDataPairs, _epochs = epochs, _startIndex = newStartIndex)
                return tutor.totalAvgLoss, tutor.totalTurns, tutor.perfectionistPassRate, tutor.learningRateGOAL
            
            else:
                print(f"unknown mode: '{mode}'. Please use 'train' or 'bot'.")

    except Exception as e:
        print(f"[RIP ʕっₓᴥₓʔっ]")
        raise
    except KeyboardInterrupt: #as k
        for name, p in babyLLM.named_parameters():
            if p.grad is None:
                msg = babyLLM.calligraphist.S_apply('emergency', f'NO GRAD: {name}')
                print(f"keyboard interrupt = {msg}")
            else: 
                grad = p.grad
                shape = tuple(grad.shape)
                norm = grad.norm().item()
                nonzero = grad.count_nonzero().item()
                total = grad.numel()
                sparsity = 1 - (nonzero / total)
                mean = grad.mean().item()
                std = grad.std().item()
                detail = (
                    f"yes grad: {name} | shape: {shape} | norm: {norm:.4f} | "
                    f"sparsity: {sparsity:.2%} | mean: {mean:.4f} | std: {std:.4f}"
                )
                msg = babyLLM.calligraphist.S_apply('almostPerfect', detail)
                print(f"keyboard interrupt = {msg}")
                if debugPrints: ʕっʘ‿ʘʔっ("♥keyboardInterrupt")
        if tutor.trainingStepCounter:
            step = tutor.trainingStepCounter
            totalAvgLoss = tutor.totalAvgLoss
            totalTurnsAwake += tutor.totalTurns
        else:
            step = 1
        choice = input("save, cancel (do not save before exit), restart or interact?" + f"\n{userName}: ").lower()
        if choice in ("save", "") or choice.startswith("s"): 
            if debugPrints: ʕっʘ‿ʘʔっ("♥choice = s")
            babyLLM.saveModel(_newStartIndex = newStartIndex, _trainingStepCounter = step, _totalAvgLoss = totalAvgLoss, _first = first)
            print("\nit's rude to interrupt people.. but, bye bye! :)")
        elif choice == "cancel" or choice.startswith("c"): 
            if debugPrints: ʕっʘ‿ʘʔっ("♥choice = c")
            print("\nhey! i wanted to remember that! :(")
        elif choice == "interact" or choice.startswith("i"):
            if debugPrints: ʕっʘ‿ʘʔっ("♥choice = i")
            babyLLM.saveModel(_newStartIndex = newStartIndex, _trainingStepCounter = step, _totalAvgLoss = totalAvgLoss, _first = first)
            import code
            print("try:\nbabyLLM.stats\nbabyLLM.scheduledSampling\nbabyLLM.memory.memory\nbabyLLM.interneuronNetwork.cerebellum\nbabyLLM.logits.forward(...)\nUse `exit()` to return to terminal.\n")
            code.interact(local = locals())
        elif choice == "restart" or choice.startswith("r"):
            if debugPrints: ʕっʘ‿ʘʔっ("♥choice = r")
            babyLLM.saveModel(_newStartIndex = newStartIndex, _trainingStepCounter = step, _totalAvgLoss = totalAvgLoss, _first = first)
            print("you spin me right round, babyllm, right round...")
            return totalAvgLoss
        else: 
            if debugPrints: ʕっʘ‿ʘʔっ("♥choice = None")
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
            try: lastTurnLoss = float(f.read().strip())
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
    with counsellor.infodump("openingQuestions") as ʕっʘ‿ʘʔっ:
        librarian = _librarian
        #babyLLM.to(modelDevice)
        if debugPrints: ʕっʘ‿ʘʔっ("setStartIndex")
        newStartIndex  = setStartIndex()
        lastRunLoss    = checkLossCheckpoint()
        mode           = "train"
        #lastRunLoss = 420

        babyNote_loadCheckpointCheck = f"[{babyName}]: right, last time i got to step {newStartIndex} and my average loss was {lastRunLoss}... want to restart from there?"
        if debugPrints: ʕっʘ‿ʘʔっ("choice = input♥")
        if _first:
            #modeChoice = input(f"[{babyName}]: so, would you like to do some training or just chat? \n[{userName}]: ").lower()
            #if modeChoice.startswith("c"):
            #    newStartIndex = setStartIndex() # Loads the step for model loading
            #    print(f"[{babyName}]: aaaa, ok! let's hear what you have to say :).")
            #elif modeChoice == "" or choice.startswith("t"):
            # IM SO TIRED BUT I WAN DO DIS LATER
            choice = input(babyNote_loadCheckpointCheck + f"\n[{userName}]: ").lower()
        else:
            choice = "yes"

        userNote_loadCheckpoint = f"[{userName}]: {choice}"

        if choice == "" or choice.startswith("y"):
            if debugPrints: ʕっʘ‿ʘʔっ("♥choice = y")
            startIndex = newStartIndex
            babyNote_loadCheckpoint = f"[{babyName}]: ok! let's go to step {newStartIndex}!"
            print(babyNote_loadCheckpoint, end="")

        elif choice.startswith("r") or choice in ["random", "i dont care", "i don't care", "idc"]:
            if debugPrints: ʕっʘ‿ʘʔっ("♥choice = r")
            newStartIndex = random.randint(0, len(librarian.tokens) - _windowMAX - 1)
            startIndex = newStartIndex
            babyNote_loadCheckpoint = f"[{babyName}]: oh, cool! i'll pick a random spot to start from... umm... let's go to step {newStartIndex}!"
            print(babyNote_loadCheckpoint, end="")

        elif choice.startswith("n") or choice in ["start again", "restart"]:
            if debugPrints: ʕっʘ‿ʘʔっ("♥choice = n")
            startIndex = newStartIndex
            babyNote_loadCheckpoint = f"[{babyName}]: alright, step {newStartIndex}, let's go back to the beginning :)"
            print(babyNote_loadCheckpoint, end="")
            
        elif choice.isdigit():
            if debugPrints: ʕっʘ‿ʘʔっ("♥choice = digit")
            newStartIndex = int(choice)
            startIndex = newStartIndex
            babyNote_loadCheckpoint = f"[{babyName}] damn that's specific! heading to step {newStartIndex}..."
            print(babyNote_loadCheckpoint, end="")

        else:
            if debugPrints: ʕっʘ‿ʘʔっ("♥choice = None")
            startIndex = newStartIndex
            babyNote_loadCheckpoint = f"[{babyName}] umm... i don't think i heard you properly, i'll just start from step {newStartIndex} :) but,"
            print(babyNote_loadCheckpoint, end="")

        if debugPrints: ʕっʘ‿ʘʔっ("runStart")
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
    passRateSTART       = perfectionistPassRateSTART
    totalTurnsAwake     = 0
    totalRuns           = 0
    MAINPairNumber      = trainingDataPairNumber
    logFreq_A           = trainingLogFreq_A
    learnRateGoal       = learningRateGOAL

    if len(sys.argv) > 1 and sys.argv[1].lower() == "bot":
        run_mode = "bot"
    else:
        choice = input("run in [t]rain mode or as twitch [b]ot? ").lower()
        if choice.startswith('b'): run_mode = "bot"
        else: run_mode = "train"

    if run_mode == "bot": 
        wakeup(windowMAX            = windowMAX, 
                dataStride          = dataStride, 
                totalTurnsAwake     = totalTurnsAwake, 
                totalRuns           = totalRuns, 
                first               = False,
                passRateSTART       = passRateSTART,
                log_A               = logFreq_A,
                lrGoal              = learnRateGoal,
                trainingDataPairNum = MAINPairNumber,
                mode                = "bot",)
    else:
        lastRunLoss         = checkLossCheckpoint()
        #lastRunLoss         = 420
        firstRun            = True
        easyStartThresh     = 3
        #logFreq_A           = windowMAXSTART * perfectionistMaxRetries
        numWins = 0
        winStreak = 0
        while windowMAX <= maxTokensPerStep:
            print(f"\n--- STARTING NEW TRAINING LOOP ---")
            thisRunLoss, totalTurns, passRateEND, learnRateGoalEND = wakeup(windowMAX   = windowMAX, 
                                                                    dataStride          = dataStride, 
                                                                    totalTurnsAwake     = totalTurnsAwake, 
                                                                    totalRuns           = totalRuns, 
                                                                    first               = firstRun,
                                                                    passRateSTART       = passRateSTART,
                                                                    log_A               = logFreq_A,
                                                                    lrGoal              = learnRateGoal,
                                                                    trainingDataPairNum = MAINPairNumber,
                                                                    mode = "train")
            #logFreq_A = windowMAX * perfectionistMaxRetries
            logFreq_A = trainingLogFreq_A
            learnRateGoal = (learnRateGoalEND+learningRateGOAL+learningRateGOAL)/3
            totalRuns += 1
            totalTurnsAwake += totalTurns
            firstRun = False
            easyStart = True

            print(f"BEFORE UPDATE: totalTurnsAwake = {totalTurnsAwake}, thisRunLoss = {thisRunLoss:.2f}, lastRunLoss = {lastRunLoss:.2f}, windowMAX = {windowMAX}, dataStride = {dataStride}, trainingPairNumber = {MAINPairNumber}, numWins = {numWins}, winStreak = {winStreak}")
            scale = abs(thisRunLoss - lastRunLoss) + 0.01
            choice = random.choice([-1,0,0,1,1,1,1,2,2,2,3,3,4,5,4,3,3,2,2,2,1,1,1,1,0,0,-1])
            increment = round(choice * (totalRuns / totalTurnsAwake) * scale)
            print(f"increment = {increment} = {choice} * ({totalRuns} / {totalTurnsAwake}) * {scale} = {choice} * {totalRuns/totalTurnsAwake} * {scale}")

            maxAllowedWindowJump = round(0.2 * (maxTokensPerStep - windowMAX))
            maxAllowedStrideJump = round(0.2 * ((windowMAX * 2) - dataStride))

            halfWindow = round(windowMAX / 20)+1
            halfStride = round(dataStride / 20)+1

            incrementW = random.choice([(max(1, min((increment + (halfWindow)), maxAllowedWindowJump))), round(windowMAX * 0.1)])
            incrementS = random.choice([(max(1, min((increment + (halfStride)), maxAllowedStrideJump))), round(dataStride * 0.1)])

            if easyStart:
                if easyStartThresh > 0:
                    lastRunLoss = (min(lastRunLoss, thisRunLoss) + lastRunLoss)/2
                    easyStartThresh -= totalRuns
                else:
                    easyStart = False
            testing = False
            if testing:
                numWins += 1
                winStreak += 1
                MAINPairNumber = 3
                windowMAX += 10
                dataStride = 1
            elif thisRunLoss < lastRunLoss:
                numWins += 1
                winStreak += 1
                if winStreak >= 2:
                    winStreak -= 1
                    MAINPairNumber -= choice
                    if random.choice([True, False]):
                        print(f"upping windowMAX from {windowMAX} to {windowMAX+incrementW}")
                        windowMAX += (incrementW+incrementW)
                    else:
                        print(f"upping dataStride from {dataStride} to {dataStride+incrementS}")
                        dataStride += (incrementS+incrementS)
            else:
                windowOrStride = random.choice([True, False])
                if winStreak > 0:
                    winStreak = -1
                MAINPairNumber += choice
                if windowMAX > incrementW+1:
                    if windowOrStride:
                        print(f"downing windowMAX from {windowMAX} to {windowMAX-incrementW}")
                        windowMAX -= incrementW
                    else:
                        print(f"windowMAX staying at {windowMAX}")
                elif dataStride > incrementS+1:
                    if not windowOrStride:
                        print(f"downing dataStride from {dataStride} to {dataStride-incrementS}")
                        dataStride -= incrementS
                    else:
                        print(f"dataStride staying at {dataStride}")
                elif dataStride == 1 and windowMAX == 1 or random.random() < 0.001:
                    random.choice([2, windowMAX, dataStride, (windowMAX * 2), (dataStride * 2), winStreak, totalRuns, incrementS, incrementW, passRateSTART, passRateEND, lastRunLoss, thisRunLoss, numWins, maxAllowedWindowJump, maxAllowedStrideJump, choice, scale, 4, 6, 8, 12, 16])
                    if random.choice([True, False]):
                        print(f"bored. windowMAX from {windowMAX} to {2}")
                        windowMAX = 2
                    else:
                        print(f"bored. dataStride from {dataStride} to {2}")
                        dataStride = 2
            
            windowMAX = round(max(1, min(windowMAX, maxTokensPerStep)))
            dataStride = round(max(1, min(dataStride, windowMAX * 0.1)))
            if MAINPairNumber < 1:
                MAINPairNumber = 2
            print(f"normalised: dataStride is {dataStride}, windowMAX is {windowMAX}")

            lastRunLoss = thisRunLoss
            passRateSTART = passRateEND
            print(f"AFTER UPDATE: totalTurnsAwake = {totalTurnsAwake}, thisRunLoss = {thisRunLoss}, lastRunLoss = {lastRunLoss}, windowMAX = {windowMAX}, dataStride = {dataStride}, trainingPairNumber = {MAINPairNumber}, numWins = {numWins}, winStreak = {winStreak}")

if __name__ == "__main__":
    main()