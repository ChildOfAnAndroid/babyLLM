#!/usr/bin/env python3
"""Real BabyLLM/PyTorch-MPS live-training overlap regression.

The parent process treats a Metal hard abort as a test failure.  The child loads
Baby's real model and optimizer checkpoint, runs real backward/optimizer steps,
and requests real autoregressive text while that lesson remains active.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _parent(timeout: float, overlaps: int) -> int:
    interpreter = "/opt/homebrew/bin/python3.14"
    command = [
        interpreter,
        str(Path(__file__).resolve()),
        "--child",
        "--overlaps",
        str(overlaps),
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env={
                **os.environ,
                "BABYLLM_MPS_STRESS_CHILD": "1",
                "BBY_WINDOW_SIZE": "269",
                "BBY_STRIDE_PERCENT": "0.66666",
                "BBY_STEP_CHECKPOINT": "dummy.txt",
            },
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"FAIL: real MPS overlap timed out after {timeout:.0f}s", flush=True)
        return 124
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        print(
            "FAIL: real MPS overlap child exited "
            f"{completed.returncode} after {elapsed:.1f}s",
            flush=True,
        )
        return completed.returncode or 1
    print(f"PASS: real MPS overlap child completed in {elapsed:.1f}s", flush=True)
    return 0


def _child(overlaps: int) -> int:
    import torch

    if not torch.backends.mps.is_available():
        print("FAIL: Apple MPS is not available", flush=True)
        return 2
    if not str(torch.__version__).startswith("2.14"):
        print(f"FAIL: expected PyTorch 2.14, found {torch.__version__}", flush=True)
        return 2

    import school.staffroom.tutor as tutor_module
    from config import (
        learningRateGOAL,
        modelDevice,
        modelFilePath,
        perfectionistPassRateSTART,
        trainingDataPairNumber,
        trainingDataStride,
    )
    from phone.discord_bot.cog import babyBot_DISCORD_COG
    from wakeup import wakeup

    if torch.device(modelDevice).type != "mps":
        print(f"FAIL: Baby selected {modelDevice}, not MPS", flush=True)
        return 2
    if not Path(modelFilePath).is_file() or not Path(modelFilePath + ".optim").is_file():
        print("FAIL: real Baby model/optimizer checkpoint is missing", flush=True)
        return 2

    # Keep the stress run observational: no checkpoint or training-log writes.
    tutor_module.printFreq = 10**9
    tutor_module.trainingLogFreq_B = 10**9
    tutor_module.saveModelFreq = 10**9

    print(
        f"[STRESS] entering wakeup path with torch={torch.__version__} "
        f"window={tutor_module.numTokensPerStepSTART} "
        f"stride={trainingDataStride} step_checkpoint={os.environ.get('BBY_STEP_CHECKPOINT')}",
        flush=True,
    )
    model, tutor, librarian, _scribe, _calligraphist = wakeup(
        windowMAX=tutor_module.numTokensPerStepSTART,
        dataStride=trainingDataStride,
        passRateSTART=perfectionistPassRateSTART,
        lrGoal=learningRateGOAL,
        trainingDataPairNum=trainingDataPairNumber,
        first=False,
        mode="mps_stress",
    )

    # The normal lesson always saves at its end.  Suppress that side effect in
    # this regression only; forward/backward/optimizer semantics remain real.
    tutor.saveFreqActions = lambda: None

    window = tutor_module.numTokensPerStepSTART
    corpus = (
        "charis: hey baby, tell me what you are thinking about today.\n"
        "babyllm: i am learning while we talk and keeping the thread warm.\n"
    ) * (window * 3)
    token_ids = librarian.tokenizer.encode(corpus)
    if len(token_ids) < (window * 2 + 1):
        print("FAIL: stress corpus did not tokenize to two windows", flush=True)
        return 3
    pair = (token_ids[:window], token_ids[1 : window + 1])

    stop_pairs = threading.Event()

    class ActivePairs:
        def __len__(self):
            return 1000

        def __iter__(self):
            for _ in range(1000):
                if stop_pairs.is_set():
                    return
                yield pair

    training_started = threading.Event()
    training_done = threading.Event()
    optimizer_steps = 0
    original_train_step = tutor.trainStep

    class StressComplete(Exception):
        pass

    def observed_train_step(*args, **kwargs):
        nonlocal optimizer_steps
        if stop_pairs.is_set() and optimizer_steps >= 1:
            raise StressComplete
        training_started.set()
        result = original_train_step(*args, **kwargs)
        optimizer_steps += 1
        return result

    tutor.trainStep = observed_train_step

    training_error: list[BaseException] = []

    def train_lesson():
        try:
            model.run_on_mps_owner(
                tutor.trainModel,
                _trainingDataPairs=ActivePairs(),
                _epochs=1,
                _startIndex=1,
            )
        except StressComplete:
            pass
        except BaseException as error:
            training_error.append(error)
        finally:
            training_done.set()

    training_thread = threading.Thread(target=train_lesson, name="stress-training")
    training_thread.start()
    if not training_started.wait(timeout=120):
        print("FAIL: live training did not begin", flush=True)
        return 4

    class BotStub:
        pass

    bot = BotStub()
    bot.babyLLM = model
    bot.tutor = tutor
    bot.librarian = librarian
    bot.MAXwindow = window
    bot.chatWindowMAX = window
    bot.numTokensPerStep = window
    bot.buffer = []
    bot.babyName = "babyllm"
    bot.getNickname = lambda name: str(name)

    cog = object.__new__(babyBot_DISCORD_COG)
    cog.bot = bot
    cog._active_generations = 0
    cog._generation_lock = asyncio.Lock()

    generation_started = threading.Event()
    generation_started_at = 0.0
    original_generation = cog._generate_response_blocking

    def observed_generation(*args, **kwargs):
        nonlocal generation_started_at
        generation_started_at = time.monotonic()
        generation_started.set()
        if training_done.is_set():
            raise AssertionError("generation began only after the lesson finished")
        return original_generation(*args, **kwargs)

    cog._generate_response_blocking = observed_generation

    replies: list[str] = []

    async def exercise_overlap():
        for attempt in range(1, overlaps + 1):
            if training_done.is_set():
                raise AssertionError("lesson ended before overlap request")
            generation_started.clear()
            requested_at = time.monotonic()
            text, error = await asyncio.wait_for(
                cog._generate_response_async(
                    "charis: can you reply while you are still learning?\nbabyllm:",
                    8,
                ),
                timeout=180,
            )
            if not generation_started.is_set():
                raise AssertionError("generation never began")
            if training_done.is_set():
                raise AssertionError("reply returned after the lesson finished")
            if error:
                raise AssertionError(f"generation returned an error: {error}")
            if not text or not text.strip():
                raise AssertionError("generation returned no text")
            replies.append(text)
            completed_at = time.monotonic()
            print(
                f"[STRESS] overlap={attempt}/{overlaps} "
                f"queue_wait={generation_started_at - requested_at:.3f}s "
                f"generation={completed_at - generation_started_at:.3f}s "
                f"request_to_reply={completed_at - requested_at:.3f}s "
                f"optimizer_steps={optimizer_steps} reply={text!r}",
                flush=True,
            )

    try:
        asyncio.run(exercise_overlap())
    finally:
        stop_pairs.set()

    training_thread.join(timeout=240)
    if training_thread.is_alive():
        print("FAIL: training deadlocked after overlap", flush=True)
        return 5
    if training_error:
        raise training_error[0]
    if optimizer_steps < 1:
        print(
            f"FAIL: no optimizer step completed for {overlaps} overlaps",
            flush=True,
        )
        return 6
    print(
        f"PASS: {len(replies)} real replies completed during active training; "
        f"optimizer_steps={optimizer_steps}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--overlaps", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.overlaps < 2:
        parser.error("--overlaps must be at least 2")
    if args.child:
        return _child(args.overlaps)
    return _parent(args.timeout, args.overlaps)


if __name__ == "__main__":
    raise SystemExit(main())
