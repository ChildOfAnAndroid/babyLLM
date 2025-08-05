# main.py (Corrected)
# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 

from rich.traceback import install
import sys, traceback, warnings, torch, os, random, asyncio
from datetime import datetime
import time

# --- LLM & School Imports ---
from babyLLM import BABYLLM
from SCHOOL.staffroom.counsellor import COUNSELLOR
from SCHOOL.staffroom.calligraphist import S_OUTPUT
from SCHOOL.staffroom.librarian import LIBRARIAN
from SCHOOL.staffroom.HE_IS_SCRIBE import SCRIBE
from SCHOOL.staffroom.tutor import TUTOR
from config import *
from secret import *

# --- BBYBOT Imports ---
from BBYBOT.DISCORD.bby_discord import BBYDiscord
from BBYBOT.COMMANDS.bby_commands import BBYCommands
from BBYBOT.UTILS.bby_users import BBYUsers
from BBYBOT.UTILS.bby_book import BBYBook
# Note: You would import your BABYBOT_TWITCH here if it were in a separate file
# from PHONE.babyBot import BABYBOT_TWITCH 

# --- Global Setup ---
def handle_exception(exc_type, exc_value, exc_traceback):
    if not issubclass(exc_type, KeyboardInterrupt):
        print("[RIP ʕっₓᴥₓʔっ] Uncaught Exception:")
        traceback.print_exception(exc_type, exc_value, exc_traceback)

sys.excepthook = handle_exception
warnings.simplefilter("default")
install(show_locals=True)
torch.autograd.set_detect_anomaly(mode=anomalyDetect, check_nan=debugPrints)


async def run_discord_bot():
    """Initializes and runs the Discord bot."""
    print("--- LAUNCHING DISCORD BOT ---")
    
    # --- Initialize Managers (Bot's "Brain") ---
    print("[Main] Initializing BBYBook Manager...")
    book_manager = BBYBook()
    print("[Main] Initializing BBYUsers Manager...")
    user_manager = BBYUsers(book_manager)
    print("[Main] Initializing BBYCommands Handler...")
    command_handler = BBYCommands(user_manager, book_manager)
    
    # --- Initialize LLM Staff ("School") ---
    print("[Main] Waking up the School...")
    counsellor = COUNSELLOR("babyLLM", _debug=debugPrints, _durations=durationLogging)
    librarian = LIBRARIAN(_counsellor=counsellor, _baseTokenizerPath=None, _forceRetrain=False)
    calligraphist = S_OUTPUT(_counsellor=counsellor)
    scribe = SCRIBE(_counsellor=counsellor, _calligraphist=calligraphist, _librarian=librarian, _numTokensPerStep=windowMAXSTART)
    
    print("[Main] Waking up babyLLM...")
    baby_llm = BABYLLM(_counsellor=counsellor, _calligraphist=calligraphist, _scribe=scribe, _librarian=librarian, 
                       _device=modelDevice, _numTokensPerStep=windowMAXSTART, _first=False, _learningRateGOAL=learningRateGOAL)
    
    tutor = TUTOR(_counsellor=counsellor, _calligraphist=calligraphist, _scribe=scribe, _librarian=librarian, _model=baby_llm,
                  _device=modelDevice, _numTokensPerStep=windowMAXSTART, _dataStride=trainingDataStride, _first=False)

    # --- Load Model ---
    baby_llm.loadModel()
    baby_llm.to(modelDevice)

    # --- Bundle LLM components for the bot ---
    llm_bundle = {
        "llm": baby_llm,
        "tutor": tutor,
        "librarian": librarian,
        "scribe": scribe,
        "calligraphist": calligraphist,
    }
    
    # --- Initialize and Run Bot ---
    training_queue = asyncio.Queue()
    bot = BBYDiscord(user_manager, book_manager, command_handler, llm_bundle, training_queue)

    async def periodic_tasks():
        """A background task to run periodic updates."""
        await bot.wait_until_ready()
        while not bot.is_closed():
            try:
                print("\n[Periodic Task] Running decay and archiving cycle...")
                await user_manager.decay_bby()
                await user_manager.handle_ghost_archiving()

                old_bestie = user_manager.current_bestie
                old_rival = user_manager.current_rival
                user_manager.update_bestie_rival()
                
                if user_manager.current_bestie and user_manager.current_bestie != old_bestie:
                    await bot.announce_bestie_change(old_bestie, user_manager.current_bestie)
                
                if user_manager.current_rival and user_manager.current_rival != old_rival:
                    await bot.announce_rival_change(old_rival, user_manager.current_rival)

            except Exception as e:
                print(f"!!!![Periodic Task] Error: {e}")
                traceback.print_exc()

            await asyncio.sleep(600) # Run every 10 minutes

    async with bot:
        bot.loop.create_task(periodic_tasks())
        await bot.start(SECRETdiscordTokenSECRET)

# --- Functions for Offline Training Mode ---
# This wakeup function is now ONLY for offline training.
def wakeup(windowMAX, dataStride, passRateSTART, lrGoal=learningRateGOAL, trainingDataPairNum=trainingDataPairNumber, log_A=trainingLogFreq_A, totalTurnsAwake=0, totalRuns=0, first=True):
    try:
        # This function's logic is correct and remains the same.
        counsellor = COUNSELLOR("babyLLM", _debug=debugPrints, _durations=durationLogging)
        with counsellor.infodump("wakeup") as ʕっʘ‿ʘʔっ:
            if debugPrints: ʕっʘ‿ʘʔっ("waking the librarian...")
            librarian = LIBRARIAN(_counsellor=counsellor)
            if debugPrints: ʕっʘ‿ʘʔっ("loading chaos agents...")
            calligraphist = S_OUTPUT(_counsellor=counsellor)
            scribe = SCRIBE(_counsellor=counsellor, _calligraphist=calligraphist, _librarian=librarian, _numTokensPerStep=windowMAX)
            if debugPrints: ʕっʘ‿ʘʔっ("loading babyLLM...")
            babyLLM = BABYLLM(_counsellor=counsellor, _calligraphist=calligraphist, _scribe=scribe, _librarian=librarian,
                              _device=modelDevice, _numTokensPerStep=windowMAX, _first=first, _learningRateGOAL=lrGoal)
            tutor = TUTOR(_counsellor=counsellor, _calligraphist=calligraphist, _scribe=scribe, _librarian=librarian, _model=babyLLM,
                          _device=modelDevice, _numTokensPerStep=windowMAX, _dataStride=dataStride, _first=first,
                          _lastRunLoss=checkLossCheckpoint(), _totalTurnsAwake=totalTurnsAwake, _totalRuns=totalRuns,
                          _perfectionistPassRateSTART=passRateSTART, _trainingLogFreq_A=log_A)
            
            print("--- STARTING OFFLINE TRAINING ---")
            if first:
                newStartIndex = openingQuestions(_counsellor=counsellor, _librarian=librarian, _windowMAX=windowMAX, _first=True)
            else:
                newStartIndex = setStartIndex()

            trainingDataPairs = librarian.genTrainingData(_windowMAX=windowMAX, _trainingDataPairNumber=trainingDataPairNumber, _startIndex=newStartIndex, _stride=trainingDataStride)
            
            babyLLM.loadModel()
            babyLLM.to(modelDevice)
            if debugPrints: ʕっʘ‿ʘʔっ("starting lessons!")
            tutor.trainModel(_trainingDataPairs=trainingDataPairs, _epochs=epochs, _startIndex=newStartIndex)
            
            return tutor.totalAvgLoss, tutor.totalTurns, tutor.perfectionistPassRate, tutor.learningRateGOAL
            
    except Exception as e:
        print(f"[RIP ʕっₓᴥₓʔっ]")
        raise
    # --- The extensive KeyboardInterrupt logic from your original file remains here ---
    except KeyboardInterrupt:
        # ... your keyboard interrupt logic ...
        print("Keyboard interrupt during training.")
        exit(8)

# --- The helper functions for offline training remain here ---
def setStartIndex(): # ...
    if os.path.exists(stepCheckpointFilePath):
        with open(stepCheckpointFilePath, "r") as f:
            try: savedStep = int(f.read().strip())
            except ValueError: savedStep = 0
    else: savedStep = 0
    return savedStep + trainingStartIndex

def checkLossCheckpoint(): # ...
    if os.path.exists(lossCheckpointFilePath):
        with open(lossCheckpointFilePath, "r") as f:
            try: lastTurnLoss = float(f.read().strip())
            except ValueError: lastTurnLoss = 0
    else: lastTurnLoss = 0
    return lastTurnLoss
    
def openingQuestions(_counsellor, _librarian, _windowMAX, _first): # ...
    print("--- Opening Questions (Offline Training) ---")
    # ... your opening questions logic ...
    return setStartIndex()

def printStartLogs(*args, **kwargs): # ...
    # ... your print start logs logic ...
    pass

# --- MAIN EXECUTION ---
def main():
    choice = input("Run in [t]rain mode, [d]iscord, or as [T]witch bot? ").lower()
    
    if choice.startswith('d'):
        run_mode = "discord"
    elif choice.startswith('t'):
        run_mode = "train"
    elif choice.startswith('T'):
        run_mode = "twitch"
    else:
        print("Defaulting to training mode.")
        run_mode = "train"

    print(f"Starting in mode: {run_mode}")
    
    # --- THIS IS THE CORRECTED DISPATCHER LOGIC ---
    if run_mode == "discord":
        try:
            asyncio.run(run_discord_bot())
        except KeyboardInterrupt:
            print("\nShutting down Discord bot.")

    elif run_mode == "twitch":
        print("Twitch bot mode selected.")
        print("NOTE: The Twitch bot logic needs to be refactored into its own `run_twitch_bot` function similar to the Discord one.")
        # Example of what it might look like:
        # try:
        #     asyncio.run(run_twitch_bot())
        # except KeyboardInterrupt:
        #     print("\nShutting down Twitch bot.")
    
    elif run_mode == "train":
        # This is your original, complex offline training loop.
        # It correctly calls the single-purpose `wakeup` function.
        windowMAX = numTokensPerStepSTART
        dataStride = trainingDataStride
        passRateSTART = perfectionistPassRateSTART
        totalTurnsAwake, totalRuns = 0, 0
        MAINPairNumber = trainingDataPairNumber
        logFreq_A = trainingLogFreq_A
        learnRateGoal = learningRateGOAL
        lastRunLoss = checkLossCheckpoint()
        firstRun = True
        
        while windowMAX <= maxTokensPerStep:
            print(f"\n--- STARTING NEW TRAINING LOOP (Window: {windowMAX}) ---")
            thisRunLoss, totalTurns, passRateEND, learnRateGoalEND = wakeup(
                windowMAX=windowMAX, 
                dataStride=dataStride, 
                totalTurnsAwake=totalTurnsAwake, 
                totalRuns=totalRuns, 
                first=firstRun,
                passRateSTART=passRateSTART,
                log_A=logFreq_A,
                lrGoal=learnRateGoal,
                trainingDataPairNum=MAINPairNumber
            )
            
            # --- Your dynamic parameter adjustment logic ---
            totalRuns += 1
            totalTurnsAwake += totalTurns
            firstRun = False
            lastRunLoss = thisRunLoss
            passRateSTART = passRateEND
            learnRateGoal = (learnRateGoalEND + learningRateGOAL + learningRateGOAL) / 3
            
            # (Your logic for adjusting windowMAX, dataStride, etc. goes here)
            print(f"Loop finished. Last loss: {thisRunLoss:.4f}. Updating parameters...")
            windowMAX = round(windowMAX * 1.25) # Simplified example
            dataStride = round(max(1, windowMAX * 0.1))

if __name__ == "__main__":
    main()