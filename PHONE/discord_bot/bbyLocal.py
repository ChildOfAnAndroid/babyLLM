# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM // phone/discord_bot/bbyLocal.py
# v1.2

# bby_brain_server.py
# RUN THIS ON YOUR LOCAL MACBOOK.
# This version has enhanced logging to help us see if chat requests are arriving.

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import json
import time
import uuid
import threading
import random

import sys
# Ensure project root is on sys.path when running from phone/discord_bot
# bbyLocal.py is in .../phone/discord_bot/. We need to go up three levels
# to reach the repository root where helpers.py lives.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from helpers import save_json_if_changed, load_json_if_exists
except ModuleNotFoundError:
    # Fallback: explicitly load helpers.py from project root
    try:
        import importlib.util
        HELPERS_PATH = os.path.join(PROJECT_ROOT, "helpers.py")
        spec = importlib.util.spec_from_file_location("helpers", HELPERS_PATH)
        if spec and spec.loader:
            helpers = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helpers)
            save_json_if_changed = getattr(helpers, "save_json_if_changed")
            load_json_if_exists = getattr(helpers, "load_json_if_exists")
        else:
            raise ModuleNotFoundError("helpers module spec not found")
    except Exception as e:
        raise ModuleNotFoundError(f"Could not import 'helpers' via sys.path or file path: {e}")

app = Flask(__name__)
CORS(app)

# ====== PATHS (Check that these are correct for your machine) ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REQUEST_FILE_PATH = os.environ.get("REQUEST_FILE_PATH", os.path.join(BASE_DIR, "bby_request.json"))
RESPONSE_DIR      = os.environ.get("RESPONSE_DIR",      os.path.join(BASE_DIR, "bby_responses"))
STATE_FILE_PATH   = os.environ.get("STATE_FILE_PATH",   os.path.join(BASE_DIR, "babyState.json"))
BBYBOOK_PATH      = os.environ.get("BBYBOOK_PATH",      os.path.expanduser("~/Dropbox/00_Icharis/02_LAB/01_babyLLM/SHKAIRA/soul/bbybook.json"))
os.makedirs(RESPONSE_DIR, exist_ok=True)

TIMEOUT_SECONDS = int(os.environ.get("BRAIN_TIMEOUT", "180"))

# --- THE BABY SOUL (STATE & BEHAVIOR) ---
state_lock = threading.Lock()
babyState = {
    "eyes": 5, "mouth": 1, "cheeks_on": False, "tears_on": False, "jumping": False,
    "stretch_left": False, "stretch_right": False, "stretch_up": False, "stretch_down": False,
    "squish_left": False, "squish_right": False, "squish_up": False, "squish_down": False,
    "isSpeaking": False, "speechText": "",
    "R": 133, "G": 239, "B": 238,
    "cerebralLoad": 0.0, "dreamIntensity": 0.0, "memoryFlux": 0.0, "learningStability": 0.0,
    "metabolicRate": 0.0,
}
baseColour = {"R": 133, "G": 239, "B": 238}
targetColour = {"R": 133, "G": 239, "B": 238}
lastTargetColour = {"R": 133, "G": 239, "B": 238}

if os.path.exists(STATE_FILE_PATH):
    try:
        with open(STATE_FILE_PATH, 'r') as f:
            babyState.update(json.load(f))
            print("Loaded initial state from babyState.json")
    except Exception as e:
        print(f"[ERROR] Could not load initial state: {e}")

# --- BACKGROUND LOOPS (The "Living" Part) ---

def state_persister_loop():
    print("[STATE_PERSISTER_LOOP] active.")
    while True:
        time.sleep(5)
        try:
            with state_lock:
                save_json_if_changed(STATE_FILE_PATH, babyState)
        except Exception as e: print(f"[ERROR] state_persister_loop: {e}")

def state_reader_loop():
    print("[STATE_READER_LOOP] active.")
    last_read_time = 0
    while True:
        try:
            if os.path.exists(STATE_FILE_PATH):
                mod_time = os.path.getmtime(STATE_FILE_PATH)
                if mod_time > last_read_time:
                    with open(STATE_FILE_PATH, 'r') as f: content = f.read()
                    if content.strip():
                        updates = json.loads(content)
                        with state_lock:
                            if babyState.get("isSpeaking", False): updates.pop("mouth", None)
                            babyState.update(updates)
                            babyState["lastUpdated"] = time.time()  # Track when we last got fresh data
                            if "R" in updates: targetColour["R"] = updates["R"]
                            if "G" in updates: targetColour["G"] = updates["G"]
                            if "B" in updates: targetColour["B"] = updates["B"]
                            dreamIntensity = babyState.get("dreamIntensity", 0.0)
                            metabolicRate = round(dreamIntensity) * (60 / (62 * 16))
                            babyState["metabolicRate"] = metabolicRate
                        last_read_time = mod_time
        except Exception: pass
        time.sleep(0.1)

def blink_loop():
    print("[BLINK_LOOP] active.")
    while True:
        try:
            with state_lock:
                is_speaking = babyState.get("isSpeaking")
                metabolicRate = babyState.get("metabolicRate", 0.5) * 0.5
                dreamIntensity = babyState.get("dreamIntensity", 10.0)
                original_eyes = babyState["eyes"]
            if is_speaking: time.sleep(0.2); continue
            wakefulness = max(1, round(dreamIntensity))
            time.sleep(2 + (random.random() * wakefulness))
            blinkDirection = random.choice([0, 1])
            with state_lock: babyState["eyes"] = blinkDirection
            time.sleep(metabolicRate)
            with state_lock: babyState["eyes"] = original_eyes
        except Exception as e: print(f"[ERROR] blink_loop: {e}")

def speak_loop():
    print("[SPEAK_LOOP] active.")
    restingMouth = 1; lastState = False
    while True:
        try:
            with state_lock:
                is_speaking = babyState.get("isSpeaking", False)
                metabolicRate = babyState.get("metabolicRate", 0.1)
            if is_speaking:
                if not lastState:
                    with state_lock: restingMouth = babyState["mouth"]
                    lastState = True
                with state_lock: babyState["mouth"] = random.randint(55, 65)
                time.sleep(max(0.05, metabolicRate))
                if random.random() < 0.25:
                    with state_lock: babyState["mouth"] = restingMouth
                    time.sleep(max(0.05, metabolicRate))
            else:
                if lastState:
                    with state_lock: babyState["mouth"] = restingMouth
                    lastState = False
                time.sleep(0.1)
        except Exception as e: print(f"[ERROR] speak_loop: {e}")

def pulse_loop():
    print("[PULSE_LOOP] active.")
    while True:
        try:
            with state_lock:
                metabolicRate = babyState.get("metabolicRate", 0.1)
                if metabolicRate <= 0: metabolicRate = 0.1
                stimChoice = random.choice(["random", "tense", "dreamy", "flux", "blushy"])
                if stimChoice == "tense":
                    is_tense = babyState.get("cerebralLoad", 0.0) > random.uniform(0.1, 1.5)
                    babyState[random.choice(["stretch_up", "squish_left", "squish_right"])] = is_tense
                    babyState[random.choice(["stretch_down", "stretch_left", "stretch_right"])] = not is_tense
                elif stimChoice == "dreamy":
                    is_dreamy = babyState.get("learningStability", 0.0) > random.uniform(0.4, 0.6)
                    babyState["stretch_up"] = is_dreamy
                    babyState[random.choice(["stretch_down", "squish_up"])] = not is_dreamy
                elif stimChoice == "flux":
                    babyState["squish_down"] = babyState.get("memoryFlux", 0.0) > random.uniform(0.4, 0.6)
                elif stimChoice == "random":
                    key = random.choice(list(babyState.keys()))
                    if key.startswith("stretch_") or key.startswith("squish_"): babyState[key] = random.choice([True, False])
                elif stimChoice == "blushy":
                    if babyState["cheeks_on"]: babyState["cheeks_on"] = random.choice([True, False])
            time.sleep(metabolicRate)
        except Exception as e: print(f"[ERROR] pulse_loop: {e}")

def smart_jump_loop():
    print("[SMART_JUMP_LOOP] active.")
    while True:
        try:
            with state_lock:
                is_jumping = babyState.get("jumping")
                metabolicRate = babyState.get("metabolicRate", 0.1) * 0.5
                if metabolicRate <= 0: metabolicRate = 0.05
            if is_jumping:
                if random.random() < 0.05:
                    with state_lock: babyState["cheeks_on"] = True
                time.sleep(metabolicRate)
                with state_lock:
                    babyState["jumping"] = False
                    babyState["cheeks_on"] = False
            else: time.sleep(0.1)
        except Exception as e: print(f"[ERROR] smart_jump_loop: {e}")

def living_colour_loop():
    global lastTargetColour
    print("[LIVING_COLOUR_LOOP] active (Final Corrected Logic).")
    while True:
        try:
            with state_lock:
                metabolicRate = babyState.get("metabolicRate", 0.1)
                if metabolicRate <= 0: metabolicRate = 0.1

                for channel in ["R", "G", "B"]:
                    blend_speed = 0.25 * (random.choice([0.25, 0.5, 1, 2]) * metabolicRate)
                    return_speed = 0.25 * (random.choice([0.25, 0.5]) * metabolicRate)
                    if lastTargetColour != targetColour:
                        delta = targetColour[channel] - babyState[channel]
                        babyState[channel] += int(delta * (blend_speed * 2))
                    else:
                        distance_to_target = abs(babyState['R'] - targetColour['R']) + \
                                             abs(babyState['G'] - targetColour['G']) + \
                                             abs(babyState['B'] - targetColour['B'])
                        arrival_threshold = 10 
                        if distance_to_target > arrival_threshold:
                            delta_target = targetColour[channel] - babyState[channel]
                            babyState[channel] += int(delta_target * (blend_speed * 0.25))
                        else:
                            delta_base = baseColour[channel] - babyState[channel]
                            babyState[channel] += int(delta_base * return_speed)

                    babyState[channel] = int(max(0, min(255, babyState[channel])))

            with state_lock: lastTargetColour = targetColour.copy()

        except Exception as e: print(f"[ERROR] living_colour_loop: {e}")
        time.sleep(0.05)

def speech_controller_loop():
    print("[SPEECH_CONTROLLER] active.")
    speak_start_time = None
    while True:
        try:
            with state_lock: is_speaking = babyState.get("isSpeaking", False)
            if is_speaking and speak_start_time is None: speak_start_time = time.time()
            if is_speaking and speak_start_time is not None:
                if time.time() - speak_start_time > 10:
                    with state_lock:
                        babyState["isSpeaking"] = False
                        babyState["speechText"] = ""
                    speak_start_time = None
            if not is_speaking: speak_start_time = None
        except Exception as e: print(f"[ERROR] speech_controller_loop: {e}")
        time.sleep(0.5)

# --- Start all background threads ---
for fn in (state_persister_loop, state_reader_loop, blink_loop, pulse_loop, smart_jump_loop, living_colour_loop, speech_controller_loop, speak_loop):
    threading.Thread(target=fn, daemon=True).start()

# --- API ENDPOINTS ---

@app.get("/api/ping")
def ping():
    print("[PING] Received ping request, sending pong.")
    return jsonify(ok=True, msg="hello from local brain")

# --- Set state endpoint (inserted) ---
@app.post("/api/state")
def set_state():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify(error="bad json"), 400
    allowed = {
        "eyes","mouth","cheeks_on","tears_on","jumping",
        "stretch_left","stretch_right","stretch_up","stretch_down",
        "squish_left","squish_right","squish_up","squish_down",
        "isSpeaking","speechText","R","G","B",
        "cerebralLoad","dreamIntensity","memoryFlux","learningStability"
    }
    with state_lock:
        for k,v in data.items():
            if k in allowed:
                babyState[k] = v
        try:
            save_json_if_changed(STATE_FILE_PATH, babyState)
        except Exception as e:
            return jsonify(error=f"persist failed: {e}"), 500
    return jsonify(ok=True, state=babyState)

@app.get("/api/state")
def get_state():
    with state_lock:
        return jsonify(babyState)

@app.get("/api/babystate")
def get_babystate_cors():
    """CORS-enabled babyState endpoint for website integration"""
    with state_lock:
        # Add staleness info
        current_time = time.time()
        last_updated = babyState.get("lastUpdated", 0)
        timestamp = babyState.get("timestamp", 0)
        
        response_data = {
            **babyState,
            "meta": {
                "dataAge": current_time - timestamp if timestamp else None,
                "lastSeenAge": current_time - last_updated if last_updated else None,
                "isStale": (current_time - timestamp) > 30 if timestamp else True,
                "serverTime": current_time
            }
        }
        
        response = jsonify(response_data)
        response.headers.add('Access-Control-Allow-Origin', '*')  # Allow website access
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response

@app.get("/api/bbybook")
def get_bbybook():
    data = load_json_if_exists(BBYBOOK_PATH, {})
    return jsonify(data)

@app.post("/api/say")
def say():
    data = request.json or {}
    text = (data.get("text") or "").strip()
    author = (data.get("author") or "anon").strip()
    if not text:
        print("[/api/say][ERROR] Received request with no text.")
        return jsonify(status="error", reply="no text :("), 400

    req_id = str(uuid.uuid4())
    print(f"[/api/say][INFO] Received job {req_id} from '{author}': '{text[:50]}...'")

    req_payload = {"id": req_id, "text": text, "author": author}
    try:
        with open(REQUEST_FILE_PATH, "w", encoding="utf-8") as f: json.dump(req_payload, f)
        print(f"[/api/say][INFO] Wrote request to {REQUEST_FILE_PATH}")
    except Exception as e:
        print(f"[/api/say][FATAL] Could not write request file: {e}")
        return jsonify(status="error", reply=f"write request failed: {e}"), 500

    resp_path = os.path.join(RESPONSE_DIR, f"{req_id}.json")
    start = time.time()
    reply = f"... (timeout after {TIMEOUT_SECONDS}s)"
    print(f"[/api/say][INFO] Waiting for response file: {resp_path}")

    while time.time() - start < TIMEOUT_SECONDS:
        if os.path.exists(resp_path):
            print(f"[/api/say][INFO] Response file found for {req_id}!")
            try:
                time.sleep(0.05)
                with open(resp_path, "r", encoding="utf-8") as f:
                    reply = (json.load(f) or {}).get("reply", "... (read empty reply)")
                os.remove(resp_path)
            except Exception as e:
                print(f"[/api/say][ERROR] Could not read or delete response file: {e}")
                reply = f"... (read error: {e})"
            break
        time.sleep(0.1)
    else:
        print(f"[/api/say][WARN] Timed out waiting for response file for job {req_id}.")

    with state_lock:
        babyState["speechText"] = reply
        babyState["isSpeaking"] = True
    
    print(f"[/api/say][INFO] Responding for job {req_id} with: '{reply[:50]}...'")
    return jsonify(status="ok", reply=reply)

if __name__ == "__main__":
    print("=== bby_brain_server (Local) on http://127.0.0.1:4420 ===")
    app.run(host="127.0.0.1", port=4420)
