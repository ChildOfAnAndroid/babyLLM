# infer.py
# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ ---
# Inference script for BABYLLM with interactive learning

import torch
import random
import os
import time
import math
import sys
import traceback
import warnings
from rich.traceback import install as rich_traceback_install
from datetime import datetime # For conversation log timestamp

# Assuming config.py is in the same directory or Python path
from config import *

# waking up the school :)
from SCHOOL.staffroom.counsellor import COUNSELLOR
counsellor = COUNSELLOR("infer", _debug = debugPrints, _durations = durationLogging)

from SCHOOL.staffroom.librarian import LIBRARIAN
librarian = LIBRARIAN(_counsellor = counsellor, _baseTokenizerPath = None, _forceRetrain = False)

from SCHOOL.staffroom.calligraphist import S_OUTPUT
calligraphist = S_OUTPUT(_counsellor = counsellor)

from SCHOOL.staffroom.HE_IS_SCRIBE import SCRIBE
scribe = SCRIBE(_counsellor = counsellor, _calligraphist = calligraphist, _librarian = librarian, _numTokensPerStep = windowMAXSTART,)

from babyLLM import BABYLLM
babyLLM = BABYLLM(_counsellor = counsellor, _calligraphist = calligraphist, _scribe = scribe, _librarian = librarian, 
                            _device = modelDevice, _numTokensPerStep = windowMAXSTART, _first = False, _learningRateGOAL = learningRateGOAL,)

from SCHOOL.staffroom.tutor import TUTOR
tutor = TUTOR(_counsellor = counsellor, _calligraphist = calligraphist, _scribe = scribe, _librarian = librarian, _model = babyLLM,
                _device = modelDevice, _numTokensPerStep = windowMAXSTART, _dataStride = trainingDataStride, _first = False, _lastRunLoss = 0, 
                _totalTurnsAwake = 0, _totalRuns = 0, _perfectionistPassRateSTART = perfectionistPassRateSTART, _trainingLogFreq_A = trainingLogFreq_A,)

babyLLM.loadModel()
babyLLM.to(modelDevice)
userPrompt = ""

# Setup exception handling and warnings
rich_traceback_install(show_locals = True)
warnings.simplefilter("default")
torch.autograd.set_detect_anomaly(mode = anomalyDetect, check_nan = debugPrints)

def handle_exception_infer(exc_type, exc_value, exc_traceback):
    if not issubclass(exc_type, KeyboardInterrupt):
        print("[RIP ʕっₓᴥₓʔっ INFER] Uncaught Exception:")
        traceback.print_exception(exc_type, exc_value, exc_traceback)
sys.excepthook = handle_exception_infer


# Helper function from wakeup.py (or ensure it's importable)
def checkLossCheckpoint():
    if os.path.exists(lossCheckpointFilePath):
        with open(lossCheckpointFilePath, "r") as f:
            try:
                lastTurnLoss = float(f.read().strip())
            except ValueError:
                thingy = f"[{babyName}]: 'noooo! i couldn't load loss checkpoint file from {lossCheckpointFilePath}, resetting to 0...' "
                print(thingy)
                userPrompt += thingy
                lastTurnLoss = 0.0 # Default if file is corrupt
    else:
        notThingy = f"{babyName} 'right, well, the loss checkpoint file {lossCheckpointFilePath} doesn't actually exist... so i'll reset it to 0.' "
        print(notThingy)
        userPrompt += notThingy
        lastTurnLoss = 0.0
    return lastTurnLoss

def log_conversation(message, log_file_path="SCHOOL/statistics/LOGS/chat/inference_chat_log.txt"):
    """Appends a message to the conversation log file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} | {message}\n")
    except Exception as e:
        print(f"[ERROR] could not write to conversation log: {e}")


def generate_response(baby_model, librarian, counsellor, prompt_tokens_ids, max_len, temperature_override = None):
    """
    Generates a response from the baby_model autoregressively.
    """
    with counsellor.infodump("generate_response") as ʕっʘ‿ʘʔっ: #婴儿
        generated_sequence_ids = list(prompt_tokens_ids)
        
        current_visual_input_for_model = None
        #if not skipPixels:
        #    if tutor.pixelNext is None:
        #        tutor.pixelNext = tutor.getPixelForStep(0)

        response_display_buffer = []

        for i in range(max_len):
            ʕっʘ‿ʘʔっ(f"generation_step_{i}") #婴儿
            #tutor.pixelNow = tutor.pixelNext.clone()
            #tutor.pixelNext = tutor.getPixelForStep(j)
            #if debugPrints: print(f"now: {pixelNow}, next: {tutor.pixelNext}", end="")
            #tutor.model.nextPixelTarget = tutor.pixelNext
            current_visual_input_for_model = tutor.pixelNow

            current_input_segment_ids = generated_sequence_ids[-baby_model.numTokensPerStep:]
            input_ids_tensor = torch.tensor(current_input_segment_ids, dtype = torch.long, device = modelDevice)
            
            logits = baby_model.forward(input_ids_tensor, _pixel = current_visual_input_for_model)
            
            #original_logtemp = None
            #if temperature_override is not None:
            #    original_logtemp = baby_model.logTemp.item()
            #    baby_model.logTemp.data.fill_(math.log(temperature_override))

            next_token_id_tensor = baby_model.getResponseFromLogits(logits, _training = True)
            next_token_id = next_token_id_tensor.item()

            #if original_logtemp is not None:
            #    baby_model.logTemp.data.fill_(original_logtemp)

            generated_sequence_ids.append(next_token_id)
            
            #if not skipPixels and hasattr(baby_model, 'predPixel'):
            #    current_visual_input_for_baby_model = baby_model.predPixel.clone()

            next_token_str = librarian.indexToToken.get(next_token_id, "<UNK>")
            response_display_buffer.append(next_token_str.replace("Ġ", " "))
            print(response_display_buffer[-1], end="", flush = True)
        
        print() 
        response_ids_only = generated_sequence_ids[len(prompt_tokens_ids):] # IDs of the content
        response_text_content = "".join(response_display_buffer)
        response_text_plus_tag = f"[{babyName}]: {response_text_content}" # Add prefix HERE
        return response_text_plus_tag, response_ids_only


def chat_with_baby(baby_model, librarian, scribe, calligraphist, counsellor, tutor):
    userPrompt = ""

    initial_greeting = f"hey! nice to see you, what did you want to tell me? also, my current numTokensPerStep is {baby_model.numTokensPerStep}."
    scribe.scribeSay(initial_greeting, _vibe="happy", _scribeName = babyName)
    userPrompt += f"[{babyName}]: {initial_greeting}\n"
    log_conversation(f"[{babyName}]: {initial_greeting}")

    conversation_token_ids_for_learning = [] # This buffer stores [user]: msg [baby]: resp
    show_learning_stats = False
    interaction_count_for_saving = 0
    save_every_n_interactions = 20

    learning_buffer_offset = 0

    while True:
        try:
            raw_user_input = input(f"\n{calligraphist.S_apply('boldWhite', f'[{userName}]: ')}")
            if not raw_user_input.strip():
                continue

            current_turn_user_prompt_log = f"[{userName}]: {raw_user_input}"
            log_conversation(current_turn_user_prompt_log)
            userPrompt += f"{current_turn_user_prompt_log}\n"

            # --- commands ---
            if raw_user_input.lower() == "!togglestats":
                raw_user_input += f"[{userName}]: {raw_user_input.lower()}\n"
                show_learning_stats = not show_learning_stats
                feedback_msg = f"learning stats display is now {'ON' if show_learning_stats else 'OFF'}."
                scribe.scribeSay(feedback_msg, _vibe="neutral", _scribeName="scribe")
                log_conversation(f"[scribe]: {feedback_msg}")
                raw_user_input += f"[scribe]: {feedback_msg}"
                continue
            #if raw_user_input.lower().startswith("!username"):
            #    raw_user_input += f"[{userName}]: {raw_user_input.lower()}\n"
            #    show_learning_stats = not show_learning_stats
            #    feedback_msg = f"changed username to {userName}."
            #    scribe.scribeSay(feedback_msg, _vibe="neutral", _scribeName="scribe")
            #    log_conversation(f"[scribe]: {feedback_msg}")
            #    raw_user_input += f"[scribe]: {feedback_msg}"
            #    continue
            if raw_user_input.lower() in ["exit", "quit", "bye", "/exit"]:
                exit_msg = "bye! it was nice chatting with you."
                scribe.scribeSay(exit_msg, _vibe="sleepy", _scribeName = babyName)
                log_conversation(f"[{babyName}]: {exit_msg}")
                break
            if raw_user_input.lower() == "!savemodel":
                raw_user_input += f"[{userName}]: {raw_user_input.lower()}"
                scribe.scribeSay("oop, you wanna save this? uhhh.. ok!", _vibe="writes", _scribeName = babyName)
                raw_user_input += f"[babyllm]: oop, you wanna save this? uhhh.. ok!"
                tutor.saveFreqActions()
                scribe.scribeSay("saved, sealed, sorted!", _vibe="happy", _scribeName = babyName)
                raw_user_input += f"[babyllm]: saved, sealed, sorted!"
                continue

            # --- User Input Processing for Learning Buffer ---
            user_input_processed = raw_user_input.lower()
            user_input_for_learning_buffer = f"[{userName}]: {user_input_processed}"
            prompt_tokens_ids_for_learning_buffer = [librarian.tokenToIndex.get(t, librarian.tokenToIndex.get("<UNK>",0)) for t in librarian.tokenizeText(user_input_for_learning_buffer)]
            conversation_token_ids_for_learning.extend(prompt_tokens_ids_for_learning_buffer)

            # --- Generation ---
            # The prompt for generation should be the full accumulated userPrompt
            generation_prompt_token_strings = librarian.tokenizeText(userPrompt)
            generation_prompt_ids = [librarian.tokenToIndex.get(t, librarian.tokenToIndex.get("<UNK>",0)) for t in generation_prompt_token_strings]

            # --- Response Length Control (based on the actual prompt being used for generation) ---
            user_prompt_len_for_gen = len(generation_prompt_ids)
            desired_max_response_len = min(inferenceOutputNumTokens, max(5, user_prompt_len_for_gen))
            if debugPrints: print(f"DEBUG INFER: User prompt (gen) len: {user_prompt_len_for_gen}, Desired max response: {desired_max_response_len}")

            # Debug prints to confirm what's being passed to generate_response
            print(f"DEBUG INFER: Type of generation_prompt_ids for generate_response: {type(generation_prompt_ids)}")
            if generation_prompt_ids:
                print(f"DEBUG INFER: Type of first element in generation_prompt_ids for gen: {type(generation_prompt_ids[0])}")
                print(f"DEBUG INFER: First 10 elements of generation_prompt_ids for gen: {generation_prompt_ids[:10]}")
            else:
                print("DEBUG INFER: generation_prompt_ids for gen is empty!")

            print(calligraphist.S_apply('italic', f"[{babyName}]: "), end="")
            raw_response_with_prefix, raw_response_ids_only = generate_response(
                baby_model, librarian, counsellor,
                generation_prompt_ids, # <--- PASS THE CORRECT LIST OF IDs
                max_len = desired_max_response_len,
                temperature_override = None
            )

            # --- Editing Step ---
            print(calligraphist.S_apply('dim', "\n--- Edit BabyLLM's response? (Press Enter to accept, or type correction) ---"))
            raw_response_content_only = raw_response_with_prefix.replace(f"[{babyName}]: ", "", 1)

            edited_response_content_input = input(f"{calligraphist.S_apply('dim', 'Original: ')}{raw_response_content_only}\n{calligraphist.S_apply('boldWhite', 'Correction: ')}")

            final_response_with_prefix = ""
            was_edited = False
            if not edited_response_content_input.strip():
                final_response_with_prefix = raw_response_with_prefix
            else:
                final_response_content_only = edited_response_content_input.strip()
                final_response_with_prefix = f"[{babyName}]: {final_response_content_only}"
                was_edited = True
                print(calligraphist.S_apply('dim', "--- Correction applied. ---"))

            # --- Logging ---
            # Log the content part for clarity of what was original vs corrected
            log_conversation(f"Original Content [{babyName}]: {raw_response_content_only}")
            if was_edited:
                log_conversation(f"CORRECTED Content [{babyName}]: {final_response_content_only}")
                # Optional: log_correction(raw_response_content_only, final_response_content_only)

            # --- Update userPrompt (for next generation's context) ---
            # userPrompt gets the FINAL version with prefix
            userPrompt += f"{final_response_with_prefix}\n"

            # --- Update conversation_token_ids_for_learning (for training) ---
            final_response_ids_for_learning_buffer = [librarian.tokenToIndex.get(t, librarian.tokenToIndex["<UNK>"]) for t in librarian.tokenizeText(final_response_with_prefix)]
            conversation_token_ids_for_learning.extend(final_response_ids_for_learning_buffer)

            learning_triggered_this_turn = False
            tokens_needed_for_one_pair = windowMAXSTART * 2
            if len(conversation_token_ids_for_learning) - learning_buffer_offset >= tokens_needed_for_one_pair:
                scribe_message_printed = False
                while learning_buffer_offset + tokens_needed_for_one_pair <= len(conversation_token_ids_for_learning):
                    tutor.totalTurns += 1
                    if not scribe_message_printed:
                        learning_msg = "thinking about what we just said... (processing conversation)"
                        scribe.scribeSay(learning_msg, _vibe="writes", _scribeName="tutor")
                        scribe_message_printed = True
                    
                    learning_triggered_this_turn = True

                    current_segment_ids = conversation_token_ids_for_learning[learning_buffer_offset : learning_buffer_offset + tokens_needed_for_one_pair]
                    current_input_ids  = current_segment_ids[:baby_model.numTokensPerStep]
                    current_target_ids = current_segment_ids[baby_model.numTokensPerStep:]

                    current_input_text  = [librarian.indexToToken.get(id, "<UNK>") for id in current_input_ids]
                    current_target_text = [librarian.indexToToken.get(id, "<UNK>") for id in current_target_ids]
                    
                    if debugPrints: print(f"DEBUG INFER: Training on slice: offset {learning_buffer_offset}, input len {len(current_input_ids)}, target len {len(current_target_ids)}")

                    total_possible_pairs_in_buffer = max(0, len(conversation_token_ids_for_learning) - (tokens_needed_for_one_pair -1))
                    current_pair_index_in_buffer = learning_buffer_offset

                    learning_successful = tutor.interactiveLearning(
                        current_input_ids, current_target_ids,
                        current_input_text, current_target_text,
                        calligraphist,
                        show_detailed_stats = show_learning_stats,
                        current_dataset_total_pairs = total_possible_pairs_in_buffer,
                        current_dataset_step_index = current_pair_index_in_buffer
                    )

                    if learning_successful:
                        interaction_count_for_saving += 1
                        if interaction_count_for_saving % save_every_n_interactions == 0:
                            tutor.saveFreqActions()
                    else:
                        not_learned_msg = "hmm... well, i tried to get my homework done on time, but i'm not really sure where you left it..."
                        scribe.scribeSay(not_learned_msg, _vibe="worried", _scribeName = babyName)
                        log_conversation(f"[{babyName}]: {not_learned_msg} (Slice offset: {learning_buffer_offset})")
                    
                    learning_buffer_offset += 25 # KEY: Slide the window by 1 token DATASTRIDE!!!! bsically

                if scribe_message_printed:
                    learned_msg_confirm = "nice :)"
                    scribe.scribeSay(learned_msg_confirm, _vibe="happy", _scribeName = babyName)
                    log_conversation(f"[{babyName}]: {learned_msg_confirm}")
            else:
                if debugPrints: print(f"DEBUG INFER: Not enough new tokens for a learning step. Available new: {len(conversation_token_ids_for_learning) - learning_buffer_offset}, Needed: {tokens_needed_for_one_pair}")

            # this buffer is for creating training pairs.
            max_learning_buffer_len = baby_model.numTokensPerStep * 5
            if len(conversation_token_ids_for_learning) > max_learning_buffer_len:
                amount_to_trim = len(conversation_token_ids_for_learning) - max_learning_buffer_len
                conversation_token_ids_for_learning = conversation_token_ids_for_learning[amount_to_trim:]
                learning_buffer_offset = max(0, learning_buffer_offset - amount_to_trim)

                trim_msg = f"learning buffer trimmed, removed {amount_to_trim} tokens. new offset: {learning_buffer_offset}."
                if show_learning_stats:
                    scribe.scribeSay(trim_msg, _vibe="neutral", _scribeName = babyName)
                    log_conversation(f"[babyllm]: {trim_msg}")
            
            max_generation_prompt_tokens = baby_model.numTokensPerStep * 3
            user_prompt_tokens_temp = librarian.tokenizeText(userPrompt)
            if len(user_prompt_tokens_temp) > max_generation_prompt_tokens:
                trimmed_prompt_tokens = user_prompt_tokens_temp[-max_generation_prompt_tokens:]
                userPrompt = "".join(trimmed_prompt_tokens).replace("Ġ", " ").strip() + "\n"
                if debugPrints: print(f"DEBUG INFER: Generation prompt string was trimmed.")
        
        except EOFError:
            eof_msg = "see you later :)"
            scribe.scribeSay(eof_msg, _vibe="sleepy", _scribeName = babyName)
            log_conversation(f"[{babyName}]: {eof_msg}")
            break
        except Exception as e:
            error_msg = f"\n{calligraphist.S_apply('emergency', 'An error occurred in the chat loop:')}"
            print(error_msg)
            traceback.print_exc()
            log_conversation(f"[SYSTEM_ERROR]: {str(e)}\n{traceback.format_exc()}")
            scribe.scribeSay("uhh... i... what? can i try that one again?", _vibe="worried", _scribeName = babyName)


if __name__ == "__main__":
    # Initialize calligraphist bands for tutor (using the global tutor instance)
    if hasattr(tutor, 'ʕっෆ‿ෆʔっ'): # tutor is now global
        calligraphist.refreshStatBands(_rollingAverages = tutor.ʕっෆ‿ෆʔっ)
        if not tutor.ʕっෆ‿ෆʔっ.get("loss"):
            tutor.ʕっෆ‿ෆʔっ['loss'] = tutor.makeStatRecord()
            tutor.ʕっෆ‿ෆʔっ['loss'][str(printFreq)] = [0.1] * printFreq
            tutor.updateRollingStats(tutor.ʕっෆ‿ෆʔっ['loss'], tutor.ʕっෆ‿ෆʔっ['loss'][str(printFreq)], printFreq, str(printFreq), percentileBands)
            calligraphist.refreshStatBands(_rollingAverages = tutor.ʕっෆ‿ෆʔっ)

    chat_log_dir = os.path.join("SCHOOL", "statistics", "LOGS", "chat")
    os.makedirs(chat_log_dir, exist_ok = True)

    try:
        # Pass the globally initialized components
        chat_with_baby(babyLLM, librarian, scribe, calligraphist, counsellor, tutor)
    except KeyboardInterrupt:
        kb_interrupt_msg = "\nBabyLLM is going to sleep. Consider /saveexit next time to save progress!"
        print(kb_interrupt_msg)
        log_conversation(f"[system]: {kb_interrupt_msg}")
    except Exception as e:
        main_error_msg = f"\n[RIP ʕっₓᴥₓʔっ MAIN INFER] Uncaught Exception:"
        print(main_error_msg)
        traceback.print_exc()
        log_conversation(f"[SYSTEM_ERROR_MAIN]: {str(e)}\n{traceback.format_exc()}")
    finally:
        if modelDevice.type == 'mps':
            torch.mps.empty_cache()
        final_cleanup_msg = "Cleaning up and shutting down..."
        scribe.scribeSay(final_cleanup_msg, _vibe="sleepy", _scribeName="system")
        log_conversation(f"[system]: {final_cleanup_msg}")