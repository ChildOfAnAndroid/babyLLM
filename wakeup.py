# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ ---
# v1.12

import random

# from torch.profiler import profile, record_function, ProfilerActivity
import sys
import threading
import time
import warnings
from datetime import datetime

import torch
from rich.traceback import install

from babyLLM import BABYLLM
from config import *
from phone.babyBot import BABYBOT_TWITCH
from phone.discord_bot import run_discord_bot
from school.staffroom.calligraphist import S_OUTPUT
from school.staffroom.counsellor import COUNSELLOR
from school.staffroom.HE_IS_SCRIBE import SCRIBE
from school.staffroom.librarian import LIBRARIAN
from school.staffroom.tutor import TUTOR
from secret import *
from utils.helpers import empty_mps_cache, get_grad_stats
from utils.wakeupUtils import (
    append_to_files,
    checkLossCheckpoint,
    handle_exception,
    openingQuestions,
    setStartIndex,
)
from utils.mps_trace import mps_trace, cleanup_mps_temp_dirs

sys.excepthook = handle_exception
warnings.simplefilter("default")  # show all warnings (PyTorch hides some by default)
install(show_locals=True)
torch.autograd.set_detect_anomaly(mode=anomalyDetect, check_nan=debugPrints)


class AdoptedWebTunnel:
    """Handle one exact pre-existing SSH process, checking identity before signals."""

    def __init__(self, pid, identity):
        self.pid = pid
        self.identity = identity

    def poll(self):
        import subprocess
        result = subprocess.run(
            ["ps", "-p", str(self.pid), "-o", "lstart=", "-o", "command="],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError("Cannot verify adopted BabyLLM tunnel identity")
        return None if result.returncode == 0 and result.stdout.strip() == self.identity else 0

    def _signal(self, value):
        import os
        if self.poll() is None:
            try:
                os.kill(self.pid, value)
            except ProcessLookupError:
                pass

    def terminate(self):
        import signal
        self._signal(signal.SIGTERM)

    def kill(self):
        import signal
        self._signal(signal.SIGKILL)

    def wait(self, timeout):
        import subprocess
        import time
        deadline = time.monotonic() + timeout
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired("adopted BabyLLM tunnel", timeout)
            time.sleep(0.1)
        return 0


def adopt_web_tunnel():
    """Adopt only a standalone known BabyLLM tunnel, never a combined session."""
    import logging
    import shlex
    import subprocess

    result = subprocess.run(
        ["ps", "-axo", "pid=,command="], capture_output=True, text=True,
        check=True, timeout=3,
    )
    matches = []
    allowed_options = {
        "BatchMode=yes", "ExitOnForwardFailure=yes", "ServerAliveInterval=30",
        "ServerAliveCountMax=3", "ConnectTimeout=5",
    }
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        try:
            args = shlex.split(fields[1])
        except ValueError:
            continue
        if not args or args[0] not in ("ssh", "/usr/bin/ssh"):
            continue
        if args[-1] not in ("midgard.loveangel1337.ovh", "charis@midgard.loveangel1337.ovh"):
            continue
        tokens = args[1:-1]
        forward = False
        no_command = False
        valid = True
        while tokens:
            flag = tokens.pop(0)
            if flag == "-N" and not no_command:
                no_command = True
            elif flag == "-f":
                continue
            elif flag == "-o" and tokens and tokens[0] in allowed_options:
                tokens.pop(0)
            elif flag == "-R" and tokens and tokens.pop(0) == "1420:127.0.0.1:4420" and not forward:
                forward = True
            else:
                valid = False
                break
        if valid and forward and no_command:
            pid = int(fields[0])
            identity = subprocess.run(
                ["ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
                capture_output=True, text=True, timeout=3, check=True,
            ).stdout.strip()
            # Recheck the command after enumeration so a replaced PID is rejected.
            identity_fields = identity.split(None, 5)
            if len(identity_fields) == 6 and identity_fields[5] == fields[1]:
                matches.append(AdoptedWebTunnel(pid, identity))
    if len(matches) > 1:
        raise RuntimeError("Multiple standalone BabyLLM tunnels found; refusing ambiguous adoption")
    if matches:
        logging.getLogger(__name__).warning("Adopted existing BabyLLM tunnel: pid=%s remote=1420 local=4420", matches[0].pid)
        return matches[0]
    return None


async def supervise_web_tunnel(stop):
    """Keep the existing BabyLLM reverse forwarding usable for unified mode."""
    import asyncio
    import logging
    import subprocess

    logger = logging.getLogger(__name__)
    failures = 0
    owned = None
    try:
        while not stop.is_set():
            try:
                healthy, owned = await asyncio.to_thread(check_web_tunnel, owned)
                if healthy:
                    if failures:
                        logger.warning("BabyLLM tunnel recovered: remote=1420 local=4420 failures=%s", failures)
                    failures = 0
                else:
                    failures += 1
                    if failures == 1 or failures % 5 == 0:
                        logger.error("BabyLLM forwarding unavailable: remote=1420 local=4420 failures=%s", failures)
            except (OSError, RuntimeError, subprocess.SubprocessError):
                failures += 1
                logger.exception("BabyLLM tunnel check failed: remote=1420 local=4420 failures=%s", failures)
            delay = 30 if not failures else min(60, 2 ** min(failures, 6))
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                continue
    finally:
        if owned is not None and owned.poll() is None:
            owned.terminate()
            try:
                await asyncio.to_thread(owned.wait, timeout=5)
            except subprocess.TimeoutExpired:
                owned.kill()
                await asyncio.to_thread(owned.wait, timeout=5)


def check_web_tunnel(owned=None):
    """Probe HTTP through forwarding; never replace another owner's tunnel."""
    import json
    import subprocess
    import urllib.request

    # A stopped brain is not a tunnel failure. Wait without creating SSH loops.
    try:
        with urllib.request.urlopen("http://127.0.0.1:4420/api/state", timeout=3) as response:
            if response.status != 200 or not isinstance(json.load(response), dict):
                return False, owned
    except (OSError, ValueError):
        return False, owned

    if owned is None or owned.poll() is not None:
        owned = adopt_web_tunnel()
    host = "midgard.loveangel1337.ovh"
    probe = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
         "curl --fail --silent --max-time 5 http://127.0.0.1:1420/api/state"],
        capture_output=True, text=True, timeout=12,
    )
    if probe.returncode == 0:
        try:
            if isinstance(json.loads(probe.stdout), dict):
                return True, owned
        except ValueError:
            pass
    if owned is not None and owned.poll() is None:
        # A live SSH process with broken forwarding is not health. Replace only
        # a created or exactly adopted standalone tunnel, never a shared session.
        owned.terminate()
        try:
            owned.wait(timeout=5)
        except subprocess.TimeoutExpired:
            owned.kill()
            owned.wait(timeout=5)
    process = subprocess.Popen(
        ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
         "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
         "-o", "ServerAliveCountMax=3", "-R", "1420:127.0.0.1:4420", host],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
    )
    # The next probe, not the PID, acknowledges successful recovery.
    return False, process


def wakeup(
    windowMAX,
    dataStride,
    passRateSTART,
    lrGoal=learningRateGOAL,
    trainingDataPairNum=trainingDataPairNumber,
    log_A=trainingLogFreq_A,
    totalTurnsAwake=0,
    totalRuns=0,
    first=True,
    mode="train",
):
    try:
        cleanup_mps_temp_dirs()
    except Exception as e:
        print(f"[WARN] MPSGraph temp directory cleanup failed: {e}")

    mps_trace("WAKEUP_START", f"mode={mode} windowMAX={windowMAX} dataStride={dataStride}")
    try:
        import os
        # WAKE UP THE school :)
        counsellor = COUNSELLOR(
            "babyLLM", _debug=debugPrints, _durations=durationLogging
        )
        with counsellor.infodump("wakeup") as ʕっʘ‿ʘʔっ:
            # OPEN THE LIBRARY :)
            if debugPrints:
                ʕっʘ‿ʘʔっ("waking the librarian...")
            librarian = LIBRARIAN(
                _counsellor=counsellor, _baseTokenizerPath=None, _forceRetrain=False
            )  # _baseTokenizerPath = "brain/vocabCache/2000_20/tokenizer_2000.json", _forceRetrain = True)

            try:
                librarian.loadTrainingData(trainingFilePath_arr)
            except Exception as e:
                print(f"[WARN] failed to initialise streaming training data: {e}")

            if False:
                exit(0)
            # if debugPrints: ʕっʘ‿ʘʔっ("opening questions...")
            # newStartIndex       = openingQuestions(_counsellor = counsellor, _librarian = librarian, _windowMAX = windowMAX, _first = first)

            # if debugPrints: ʕっʘ‿ʘʔっ("generating training data pairs...")
            # trainingDataPairs   =           librarian.genTrainingData(_windowMAX = windowMAX, _trainingDataPairNumber = trainingDataPairNum, _startIndex = newStartIndex, _stride = dataStride)
            # if debugPrints:                 print(f"Total trainingDataPairs: {len(trainingDataPairs)}")

            if debugPrints:
                ʕっʘ‿ʘʔっ("loading chaos agents...")
            calligraphist = S_OUTPUT(_counsellor=counsellor)

            scribe = SCRIBE(
                _counsellor=counsellor,
                _calligraphist=calligraphist,
                _librarian=librarian,
                _numTokensPerStep=windowMAX,
            )

            # WAKE UP THE BABY :)
            if debugPrints:
                ʕっʘ‿ʘʔっ("loading babyLLM...")
            babyLLM = BABYLLM(
                _counsellor=counsellor,
                _calligraphist=calligraphist,
                _scribe=scribe,
                _librarian=librarian,
                _device=modelDevice,
                _numTokensPerStep=windowMAX,
                _first=first,
                _learningRateGOAL=lrGoal,
            )

            babyLLM.model_thread_lock = threading.Lock()
            tutor = TUTOR(
                _counsellor=counsellor,
                _calligraphist=calligraphist,
                _scribe=scribe,
                _librarian=librarian,
                _model=babyLLM,
                _model_thread_lock=babyLLM.model_thread_lock,
                _device=modelDevice,
                _numTokensPerStep=windowMAX,
                _dataStride=dataStride,
                _first=first,
                _lastRunLoss=checkLossCheckpoint(),
                _totalTurnsAwake=totalTurnsAwake,
                _totalRuns=totalRuns,
                _perfectionistPassRateSTART=passRateSTART,
                _trainingLogFreq_A=log_A,
            )

            if mode == "mps_stress":
                # Local integration entrypoint: exercise the exact wakeup
                # construction/load path without connecting any chat service.
                mps_trace("LOAD_MODEL_BEFORE", "mode=mps_stress")
                babyLLM.loadModel()
                babyLLM.to(modelDevice)
                babyLLM.activate_mps_owner()
                mps_trace("LOAD_MODEL_AFTER", "mode=mps_stress")
                return babyLLM, tutor, librarian, scribe, calligraphist

            if mode == "twitch":
                print("--- LAUNCHING TWITCH BOT ---")
                if debugPrints:
                    ʕっʘ‿ʘʔっ("starting twitch bot!")
                # create a bot instance, pass in the staff etc
                babyBot_twitch = BABYBOT_TWITCH(
                    babyLLM, tutor, librarian, scribe, calligraphist
                )
                mps_trace("LOAD_MODEL_BEFORE", "mode=twitch")
                babyLLM.loadModel()
                babyLLM.to(modelDevice)
                babyLLM.activate_mps_owner()
                mps_trace("LOAD_MODEL_AFTER", "mode=twitch")
                babyBot_twitch.run()

            elif mode == "discord":
                print("--- LAUNCHING DISCORD BOT ---")
                if debugPrints:
                    ʕっʘ‿ʘʔっ("starting discord bot!")
                # create a bot instance, pass in the staff etc
                run_discord_bot(
                    babyLLM,
                    tutor,
                    librarian,
                    scribe,
                    calligraphist,
                    SECRETdiscordTokenSECRET,
                )

            elif mode == "unified":
                print("=" * 60)
                print("LAUNCHING UNIFIED MULTI-PLATFORM BOT (Discord + Twitch + Web)")
                print("=" * 60)
                if debugPrints:
                    ʕっʘ‿ʘʔっ("starting unified multi-platform bot!")

                import os
                import subprocess

                # Ensure BBY_PUBLIC_URL is set so the bot reactively pushes state updates to the website
                if "BBY_PUBLIC_URL" not in os.environ:
                    os.environ["BBY_PUBLIC_URL"] = "https://childofanandroid.co.uk"
                    print(f"[UNIFIED] Auto-configured BBY_PUBLIC_URL: {os.environ['BBY_PUBLIC_URL']}")

                # ICHARIS2 owns the Capture Review process. BabyLLM may reuse an
                # already-running canonical listener, but must never create a
                # second hidden startup route for it.
                icharis_root = os.environ.get(
                    "ICHARIS2_ROOT",
                    "/Users/charis/00_Icharis/02_LAB/icharis2",
                )
                try:
                    health = subprocess.run(
                        [
                            "curl",
                            "--fail",
                            "--silent",
                            "--max-time",
                            "2",
                            "http://127.0.0.1:8765/_capture_review_health",
                        ],
                        check=False,
                    )
                    icharis_running = health.returncode == 0
                except OSError:
                    icharis_running = False
                if not icharis_running and os.path.isdir(icharis_root):
                    print(
                        "[UNIFIED] icharis2 Capture Review is not running; "
                        "start it through its canonical ICHARIS2 owner when needed."
                    )

                # BabyLLM owns only its API transport. ICHARIS2 launchd owns
                # Capture Review and the website's authenticated /art tunnel.
                tunnel_specs = (("1420:127.0.0.1:4420", "BabyLLM API"),)
                missing_specs = []
                for reverse_spec, label in tunnel_specs:
                    try:
                        result = subprocess.run(
                            ["pgrep", "-f", f"ssh.*{reverse_spec}"],
                            capture_output=True,
                            text=True,
                        )
                        tunnel_exists = bool(result.stdout.strip())
                    except OSError:
                        tunnel_exists = False
                    if tunnel_exists:
                        print(f"[UNIFIED] ssh tunnel for {label} already established.")
                    else:
                        missing_specs.append(reverse_spec)

                if missing_specs:
                    print("[UNIFIED] establishing missing ssh reverse tunnels...")
                    command = [
                        "ssh",
                        "-f",
                        "-N",
                        "-o",
                        "ExitOnForwardFailure=yes",
                        "-o",
                        "ServerAliveInterval=30",
                        "-o",
                        "ServerAliveCountMax=3",
                    ]
                    for reverse_spec in missing_specs:
                        command.extend(["-R", reverse_spec])
                    command.append("midgard.loveangel1337.ovh")
                    try:
                        # ssh -f forks into the background, so subprocess.run returns immediately
                        subprocess.run(command, check=True)
                        print("[UNIFIED] missing ssh reverse tunnels established successfully.")
                    except subprocess.CalledProcessError as e:
                        print(f"[ERROR] Failed to establish ssh reverse tunnels: {e}")

                import asyncio

                from phone.discord_bot.bot import BABYBOT_DISCORD

                # Create bot instance
                bot = BABYBOT_DISCORD(
                    babyLLM=babyLLM,
                    tutor=tutor,
                    librarian=librarian,
                    scribe=scribe,
                    calligraphist=calligraphist,
                    discordToken=SECRETdiscordTokenSECRET,
                    discordChannel=bby_spam_channel_id,
                )

                async def run_unified():
                    # Set up bot
                    await bot.setup_bot()

                    # Enable Twitch
                    print("[UNIFIED] Enabling Twitch integration...")
                    try:
                        await bot.enable_twitch()  # Reads from config.twitch_channels
                        print(
                            f"[UNIFIED] Twitch enabled for channels: {twitch_channels}"
                        )
                    except Exception as e:
                        print(f"[ERROR] Failed to enable Twitch: {e}")
                        import traceback

                        traceback.print_exc()
                        print("[UNIFIED] Continuing without Twitch...")

                    tunnel_stop = asyncio.Event()
                    tunnel_task = None

                    # Enable Web API
                    print("[UNIFIED] Enabling Web API integration...")
                    try:
                        await bot.enable_web(port=4420)
                        tunnel_task = asyncio.create_task(supervise_web_tunnel(tunnel_stop))
                        print("[UNIFIED] Web API enabled on http://127.0.0.1:4420")
                    except Exception as e:
                        print(f"[ERROR] Failed to enable Web API: {e}")
                        import traceback

                        traceback.print_exc()
                        print("[UNIFIED] Continuing without Web API...")

                    # Start Discord bot
                    print("\n[UNIFIED] Starting Discord bot...")
                    print("=" * 60)
                    print("UNIFIED MULTI-PLATFORM BOT IS NOW RUNNING")
                    print("Platforms: Discord + Twitch + Web API")
                    print("Press Ctrl+C to stop")
                    print("=" * 60)

                    # Unified mode is long-lived inference. Keep the ~5 GB
                    # optimizer checkpoint off-device until Tutor.trainModel
                    # reaches its existing wait_for_optimizer_ready() barrier.
                    mps_trace("LOAD_MODEL_BEFORE", "mode=unified")
                    babyLLM.loadModel(defer_optimizer=True)
                    babyLLM.to(modelDevice)
                    babyLLM.activate_mps_owner()
                    mps_trace("LOAD_MODEL_AFTER", "mode=unified")

                    try:
                        await bot.start(SECRETdiscordTokenSECRET)
                    except KeyboardInterrupt:
                        print("\n[SHUTDOWN] Stopping unified bot...")
                    finally:
                        tunnel_stop.set()
                        if tunnel_task is not None:
                            await tunnel_task
                        if hasattr(bot, "disable_twitch"):
                            await bot.disable_twitch()
                        if hasattr(bot, "disable_web"):
                            await bot.disable_web()
                        await bot.close()
                        print("[SHUTDOWN] Unified bot stopped")

                # Run the unified bot
                asyncio.run(run_unified())

            elif mode == "train":
                print("--- STARTING OFFLINE TRAINING ---")
                if first == True and os.getenv("NONINTERACTIVE") != "1":
                    newStartIndex = openingQuestions(
                        _counsellor=counsellor,
                        _librarian=librarian,
                        _windowMAX=windowMAX,
                        _first=True,
                    )
                else:
                    newStartIndex = setStartIndex()
                if tokenSpeedTest == True:
                    tokenSpeedTestStart = time.time()
                trainingDataPairs = librarian.genTrainingData(
                    _windowMAX=windowMAX,
                    _trainingDataPairNumber=trainingDataPairNumber,
                    _startIndex=newStartIndex,
                    _stride=trainingDataStride,
                )

                # START THE LESSONS :)
                mps_trace("LOAD_MODEL_BEFORE", "mode=train")
                babyLLM.loadModel()
                babyLLM.to(modelDevice)
                mps_trace("LOAD_MODEL_AFTER", "mode=train")
                if debugPrints:
                    ʕっʘ‿ʘʔっ("starting lessons!")
                mps_trace("OFFLINE_TRAINING_BEFORE", f"epochs={epochs} pairs={len(trainingDataPairs)}")
                tutor.trainModel(
                    _trainingDataPairs=trainingDataPairs,
                    _epochs=epochs,
                    _startIndex=newStartIndex,
                )
                mps_trace("OFFLINE_TRAINING_AFTER")

                if tokenSpeedTest == True:
                    tokenSpeedTestEnd = time.time()
                    thisTestSpeed = tokenSpeedTestEnd - tokenSpeedTestStart
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    tokenSpeedTest_str = f"\n{timestamp}| numTokens: {windowMAX}, speed: {thisTestSpeed:.2f} (tokenAvg: {(thisTestSpeed / windowMAX):.2f}), avgLoss: {tutor.totalAvgLoss:.2f} (delta: {tutor.totalAvgDelta:.2f})"
                    append_to_files(tokenSpeedTest_str, tokenSpeedTestFilePath)

                return (
                    tutor.totalAvgLoss,
                    tutor.totalTurns,
                    tutor.perfectionistPassRate,
                    tutor.learningRateGOAL,
                )

            else:
                print(f"unknown mode: '{mode}'. Please use 'train' or 'bot'.")

    except Exception:
        print("[RIP ʕっₓᴥₓʔっ]")
        raise
    except KeyboardInterrupt:  # as k
        for name, p in babyLLM.named_parameters():
            if p.grad is None:
                msg = babyLLM.calligraphist.S_apply("emergency", f"NO GRAD: {name}")
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
                msg = babyLLM.calligraphist.S_apply("almostPerfect", detail)
                print(f"keyboard interrupt = {msg}")
                if debugPrints:
                    ʕっʘ‿ʘʔっ("♥keyboardInterrupt")
        if tutor.trainingStepCounter:
            step = tutor.trainingStepCounter
            totalAvgLoss = tutor.totalAvgLoss
            totalTurnsAwake += tutor.totalTurns
        else:
            step = 1
        choice = input(
            "save, cancel (do not save before exit), restart or interact?"
            + f"\n{userName}: "
        ).lower()
        if choice in ("save", "") or choice.startswith("s"):
            if debugPrints:
                ʕっʘ‿ʘʔっ("♥choice = s")
            babyLLM.saveModel(
                _newStartIndex=newStartIndex,
                _trainingStepCounter=step,
                _totalAvgLoss=totalAvgLoss,
                _first=first,
            )
            print("\nit's rude to interrupt people.. but, bye bye! :)")
        elif choice == "cancel" or choice.startswith("c"):
            if debugPrints:
                ʕっʘ‿ʘʔっ("♥choice = c")
            print("\nhey! i wanted to remember that! :(")
        elif choice == "interact" or choice.startswith("i"):
            if debugPrints:
                ʕっʘ‿ʘʔっ("♥choice = i")
            babyLLM.saveModel(
                _newStartIndex=newStartIndex,
                _trainingStepCounter=step,
                _totalAvgLoss=totalAvgLoss,
                _first=first,
            )
            import code

            print(
                "try:\nbabyLLM.stats\nbabyLLM.scheduledSampling\nbabyLLM.memory.memory\nbabyLLM.interneuronNetwork.cerebellum\nbabyLLM.logits.forward(...)\nUse `exit()` to return to terminal.\n"
            )
            code.interact(local=locals())
        elif choice == "restart" or choice.startswith("r"):
            if debugPrints:
                ʕっʘ‿ʘʔっ("♥choice = r")
            babyLLM.saveModel(
                _newStartIndex=newStartIndex,
                _trainingStepCounter=step,
                _totalAvgLoss=totalAvgLoss,
                _first=first,
            )
            print("you spin me right round, babyllm, right round...")
            return totalAvgLoss
        else:
            if debugPrints:
                ʕっʘ‿ʘʔっ("♥choice = None")
            babyLLM.saveModel(
                _newStartIndex=newStartIndex,
                _trainingStepCounter=step,
                _totalAvgLoss=totalAvgLoss,
                _first=first,
            )
            print("\nuhh... i'm confused, but i saved anyway!")
        if modelDevice.type == "mps":
            empty_mps_cache()
            print("cache emptied")
        exit(8)


def main():
    windowMAX = numTokensPerStepSTART
    dataStride = trainingDataStride
    passRateSTART = perfectionistPassRateSTART
    totalTurnsAwake = 0
    totalRuns = 0
    MAINPairNumber = trainingDataPairNumber
    logFreq_A = trainingLogFreq_A
    learnRateGoal = learningRateGOAL

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "bot":
            run_mode = "twitch"
        elif arg in ("unified", "u", "all"):
            run_mode = "unified"
        elif arg in ("discord", "d"):
            run_mode = "discord"
        elif arg in ("twitch", "t"):
            run_mode = "twitch"
        else:
            run_mode = "train"
    else:
        choice = input(
            "run in train mode, [d]iscord, [t]witch, or [u]nified (discord+twitch)? "
        ).lower()

        if choice.startswith("t"):
            run_mode = "twitch"
        elif choice.startswith("d"):
            run_mode = "discord"
        elif choice.startswith("u"):
            run_mode = "unified"
        else:
            run_mode = "train"

        print(f"Choice {choice} run_mode => {run_mode}")

    if run_mode in ["twitch", "discord", "unified"]:
        # Bot startup should use the real target pass-rate baseline, not the fresh-training bootstrap value. 
        # Otherwise daily eval output can look fake-good immediately after startup.
        botPassRateStart = perfectionistPassRate
        wakeup(windowMAX            = windowMAX,
                dataStride          = dataStride,
                totalTurnsAwake     = totalTurnsAwake,
                totalRuns           = totalRuns,
                first               = False,
                passRateSTART       = botPassRateStart,
                log_A               = logFreq_A,
                lrGoal              = learnRateGoal,
                trainingDataPairNum = MAINPairNumber,
                mode                = run_mode,)
    else:
        lastRunLoss = checkLossCheckpoint()
        # lastRunLoss         = 420
        firstRun = True
        easyStartThresh = 3
        # logFreq_A           = windowMAXSTART * perfectionistMaxRetries
        numWins = 0
        winStreak = 0
        while windowMAX <= maxTokensPerStep:
            print("\n--- STARTING NEW TRAINING LOOP ---")
            thisRunLoss, totalTurns, passRateEND, learnRateGoalEND = wakeup(
                windowMAX=windowMAX,
                dataStride=dataStride,
                totalTurnsAwake=totalTurnsAwake,
                totalRuns=totalRuns,
                first=firstRun,
                passRateSTART=passRateSTART,
                log_A=logFreq_A,
                lrGoal=learnRateGoal,
                trainingDataPairNum=MAINPairNumber,
                mode="train",
            )
            # logFreq_A = windowMAX * perfectionistMaxRetries
            logFreq_A = trainingLogFreq_A
            learnRateGoal = (learnRateGoalEND + learningRateGOAL + learningRateGOAL) / 3
            totalRuns += 1
            totalTurnsAwake += totalTurns
            firstRun = False
            easyStart = True

            print(
                f"BEFORE UPDATE: totalTurnsAwake = {totalTurnsAwake}, thisRunLoss = {thisRunLoss:.2f}, lastRunLoss = {lastRunLoss:.2f}, windowMAX = {windowMAX}, dataStride = {dataStride}, trainingPairNumber = {MAINPairNumber}, numWins = {numWins}, winStreak = {winStreak}"
            )
            scale = abs(thisRunLoss - lastRunLoss) + 0.01
            choice = random.choice(
                [
                    -1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    3,
                    3,
                    4,
                    5,
                    4,
                    3,
                    3,
                    2,
                    2,
                    2,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    -1,
                ]
            )
            increment = round(choice * (totalRuns / totalTurnsAwake) * scale)
            print(
                f"increment = {increment} = {choice} * ({totalRuns} / {totalTurnsAwake}) * {scale} = {choice} * {totalRuns / totalTurnsAwake} * {scale}"
            )

            maxAllowedWindowJump = round(0.2 * (maxTokensPerStep - windowMAX))
            maxAllowedStrideJump = round(0.2 * ((windowMAX * 2) - dataStride))

            halfWindow = round(windowMAX / 20) + 1
            halfStride = round(dataStride / 20) + 1

            incrementW = random.choice(
                [
                    (max(1, min((increment + (halfWindow)), maxAllowedWindowJump))),
                    round(windowMAX * 0.1),
                ]
            )
            incrementS = random.choice(
                [
                    (max(1, min((increment + (halfStride)), maxAllowedStrideJump))),
                    round(dataStride * 0.1),
                ]
            )

            if easyStart:
                if easyStartThresh > 0:
                    lastRunLoss = (min(lastRunLoss, thisRunLoss) + lastRunLoss) / 2
                    easyStartThresh -= totalRuns
                else:
                    easyStart = False
            testing = True
            if testing:
                numWins += 1
                winStreak += 1
                MAINPairNumber = trainingDataPairNumber
                # if winStreak % 2 == 0:
                if windowMAX >= 1:
                    windowInc = random.choice([2, 0.5])
                    windowMAX *= windowInc
                    dataStride = max(1, (windowMAX * 0.1))
                else:
                    windowMAX = 1
            elif thisRunLoss < lastRunLoss:
                numWins += 1
                winStreak += 1
                if winStreak >= 2:
                    winStreak -= 1
                    MAINPairNumber -= choice
                    if random.choice([True, False]):
                        print(
                            f"upping windowMAX from {windowMAX} to {windowMAX + incrementW}"
                        )
                        windowMAX += incrementW + incrementW
                    else:
                        print(
                            f"upping dataStride from {dataStride} to {dataStride + incrementS}"
                        )
                        dataStride += incrementS + incrementS
            else:
                windowOrStride = random.choice([True, False])
                if winStreak > 0:
                    winStreak = -1
                MAINPairNumber += choice
                if windowMAX > incrementW + 1:
                    if windowOrStride:
                        print(
                            f"downing windowMAX from {windowMAX} to {windowMAX - incrementW}"
                        )
                        windowMAX -= incrementW
                    else:
                        print(f"windowMAX staying at {windowMAX}")
                elif dataStride > incrementS + 1:
                    if not windowOrStride:
                        print(
                            f"downing dataStride from {dataStride} to {dataStride - incrementS}"
                        )
                        dataStride -= incrementS
                    else:
                        print(f"dataStride staying at {dataStride}")
                elif dataStride == 1 and windowMAX == 1 or random.random() < 0.001:
                    random.choice(
                        [
                            2,
                            windowMAX,
                            dataStride,
                            (windowMAX * 2),
                            (dataStride * 2),
                            winStreak,
                            totalRuns,
                            incrementS,
                            incrementW,
                            passRateSTART,
                            passRateEND,
                            lastRunLoss,
                            thisRunLoss,
                            numWins,
                            maxAllowedWindowJump,
                            maxAllowedStrideJump,
                            choice,
                            scale,
                            4,
                            6,
                            8,
                            12,
                            16,
                        ]
                    )
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
            print(
                f"AFTER UPDATE: totalTurnsAwake = {totalTurnsAwake}, thisRunLoss = {thisRunLoss}, lastRunLoss = {lastRunLoss}, windowMAX = {windowMAX}, dataStride = {dataStride}, trainingPairNumber = {MAINPairNumber}, numWins = {numWins}, winStreak = {winStreak}"
            )


if __name__ == "__main__":
    main()
