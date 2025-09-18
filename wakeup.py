# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# v4.14

from rich.traceback import install
#from torch.profiler import profile, record_function, ProfilerActivity
import sys, warnings, torch, random
from datetime import datetime
import time

from babyLLM import BABYLLM
from school.staffroom.counsellor import COUNSELLOR
from school.staffroom.calligraphist import S_OUTPUT
from school.staffroom.librarian import LIBRARIAN
from school.staffroom.HE_IS_SCRIBE import SCRIBE
from school.staffroom.tutor import TUTOR
from config import *
from secret import *
from phone.babyBot import BABYBOT_TWITCH
from phone.babyBot_discord import *
from wakeupUtils import handle_exception, setStartIndex, checkLossCheckpoint, openingQuestions, append_to_files
from helpers import get_grad_stats, empty_mps_cache

sys.excepthook = handle_exception
warnings.simplefilter("default") # show all warnings (PyTorch hides some by default)
install(show_locals = True)
torch.autograd.set_detect_anomaly(mode = anomalyDetect, check_nan = debugPrints)

def wakeup(windowMAX, dataStride, passRateSTART, lrGoal = learningRateGOAL, trainingDataPairNum = trainingDataPairNumber, log_A = trainingLogFreq_A, totalTurnsAwake = 0, totalRuns = 0, first = True, mode = "train"):
    try:
        # WAKE UP THE school :)
        counsellor              = COUNSELLOR("babyLLM", _debug = debugPrints, _durations = durationLogging)
        with counsellor.infodump("wakeup") as ʕっʘ‿ʘʔっ:

            # OPEN THE LIBRARY :)
            if debugPrints: ʕっʘ‿ʘʔっ("waking the librarian...")
            librarian           = LIBRARIAN (_counsellor = counsellor, _baseTokenizerPath = None, _forceRetrain = False) #_baseTokenizerPath = "brain/vocabCache/2000_20/tokenizer_2000.json", _forceRetrain = True)

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
            
            if mode == "twitch":
                print("--- LAUNCHING TWITCH BOT ---")
                if debugPrints: ʕっʘ‿ʘʔっ("starting twitch bot!")
                # create a bot instance, pass in the staff etc
                babyBot_twitch = BABYBOT_TWITCH(babyLLM, tutor, librarian, scribe, calligraphist)
                babyLLM.loadModel()
                babyLLM.to(modelDevice)
                babyBot_twitch.run()

            elif mode == "discord":
                print("--- LAUNCHING DISCORD BOT ---")
                if debugPrints: ʕっʘ‿ʘʔっ("starting discord bot!")
                # create a bot instance, pass in the staff etc
                run_discord_bot(babyLLM, tutor, librarian, scribe, calligraphist, SECRETdiscordTokenSECRET)

            elif mode == "train":
                print("--- STARTING OFFLINE TRAINING ---")
                if first == True: newStartIndex = openingQuestions(_counsellor=counsellor, _librarian=librarian, _windowMAX=windowMAX, _first=True)
                else: newStartIndex = setStartIndex()
                if tokenSpeedTest == True: tokenSpeedTestStart = time.time()
                trainingDataPairs = librarian.genTrainingData(_windowMAX=windowMAX, _trainingDataPairNumber=trainingDataPairNumber, _startIndex=newStartIndex, _stride=trainingDataStride)
                
                # START THE LESSONS :)
                babyLLM.loadModel()
                babyLLM.to(modelDevice)
                if debugPrints: ʕっʘ‿ʘʔっ("starting lessons!")
                tutor.trainModel(_trainingDataPairs = trainingDataPairs, _epochs = epochs, _startIndex = newStartIndex)

                if tokenSpeedTest == True:
                    tokenSpeedTestEnd = time.time()
                    thisTestSpeed = (tokenSpeedTestEnd - tokenSpeedTestStart)
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    tokenSpeedTest_str = f"\n{timestamp}| numTokens: {windowMAX}, speed: {thisTestSpeed:.2f} (tokenAvg: {(thisTestSpeed/windowMAX):.2f}), avgLoss: {tutor.totalAvgLoss:.2f} (delta: {tutor.totalAvgDelta:.2f})"
                    append_to_files(tokenSpeedTest_str, tokenSpeedTestFilePath)

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
                stats = get_grad_stats(p.grad)
                shape = stats["shape"]
                norm = stats["norm"]
                sparsity = stats["sparsity"]
                mean = stats["mean"]
                std = stats["std"]
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
            empty_mps_cache()
            print(f"cache emptied")
        exit(8)

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
        run_mode = "twitch"
    else:
        choice = input("run in train mode, [d]iscord, or as [t]witch bot? ").lower()
        
        if choice.startswith('t'): run_mode = "twitch"
        elif choice.startswith('d'): run_mode = "discord"
        else: run_mode = "train"

        print(f"Choice {choice} run_mode => {run_mode}")

    if run_mode in ["twitch", "discord"]: 
        wakeup(windowMAX            = windowMAX, 
                dataStride          = dataStride, 
                totalTurnsAwake     = totalTurnsAwake, 
                totalRuns           = totalRuns, 
                first               = False,
                passRateSTART       = passRateSTART,
                log_A               = logFreq_A,
                lrGoal              = learnRateGoal,
                trainingDataPairNum = MAINPairNumber,
                mode                = run_mode,)
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
            testing = True
            if testing:
                numWins += 1
                winStreak += 1
                MAINPairNumber = trainingDataPairNumber
                #if winStreak % 2 == 0:
                if windowMAX >= 1:
                    windowInc = random.choice([2, 0.5])
                    windowMAX *= windowInc
                    dataStride = max(1,(windowMAX * 0.1))
                else:
                    windowMAX = 1
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
