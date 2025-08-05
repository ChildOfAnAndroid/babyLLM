from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import json
import os
import random
import uuid

# --- setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE_PATH = os.path.join(SCRIPT_DIR, "babyState.json")
app = Flask(__name__)
CORS(app)

# --- ADD THESE FILE PATHS ---
REQUEST_FILE_PATH = os.path.join(SCRIPT_DIR, "bby_request.json")
RESPONSE_DIR = os.path.join(SCRIPT_DIR, "bby_responses")
os.makedirs(RESPONSE_DIR, exist_ok=True) # Ensure the response directory exists
BBYBOOK_FILE_PATH = "/Users/charis/Dropbox/00_Icharis/02_LAB/01_babyLLM/SHKAIRA/soul/bbybook.json"

# --- THE BABY SOUL ---
babyState = {
    "eyes": 5, "mouth": 1, "cheeks_on": False, "tears_on": False, "jumping": False,
    "stretch_left": False, "stretch_right": False, "stretch_up": False, "stretch_down": False,
    "squish_left": False, "squish_right": False, "squish_up": False, "squish_down": False,
    "isSpeaking": False, "speechText": "",
    "R": 133, "G": 239, "B": 238,
    "cerebralLoad": 0.0, "dreamIntensity": 0.0, "memoryFlux": 0.0,
    "learningStability": 0.0, "metabolicRate": 0.0,
}

baseColour = {"R": 133, "G": 239, "B": 238}
targetColour = {"R": 133, "G": 239, "B": 238}
lastTargetColour = {"R": 133, "G": 239, "B": 238}


# --- THE BABY STATE ---
def state_reader_loop():
    """Reads babyState.json and performs the original, quirky logic."""
    print("[STATE_READER_LOOP] state_reader_loop active.")
    while True:
        if os.path.exists(STATE_FILE_PATH):
            try:
                with open(STATE_FILE_PATH, 'r') as f:
                    content = f.read()
                    if not content.strip(): continue
                    updates = json.loads(content)

                    babyState.update(updates)
                    if "R" in updates: targetColour["R"] = updates["R"]
                    if "G" in updates: targetColour["G"] = updates["G"]
                    if "B" in updates: targetColour["B"] = updates["B"]
                    if babyState.get("correct") == True:
                        babyState["R"] = min(255, babyState["R"] * 1.05)
                        if babyState["mouth"] >= 0:
                            smile = random.choice([0,0,0,1])
                            babyState["mouth"] += smile
                        else:
                            babyState["mouth"] = 5
                    
                    dreamIntensity = babyState.get("dreamIntensity", 0.0)
                    bpm = 62
                    bpm32th = 60 / (bpm * 16)
                    metabolicRate = round(dreamIntensity) * bpm32th
                    babyState["metabolicRate"] = metabolicRate
                    
            except Exception as e:
                print(f"[ERROR] state_reader_loop: {e}")
        time.sleep(0.1)

# --- THE BABY BLINK ---
def blink_loop():
    print("[BLINK_LOOP] active! ")
    while True:
        if babyState.get("isSpeaking"):
            time.sleep(0.2)
            continue
            
        metabolicRate = babyState.get("metabolicRate", 0.5) * 0.5
        dreamIntensity = babyState.get("dreamIntensity", 10.0)
        wakefulness = max(1, round(dreamIntensity))
        time.sleep(2 + (time.time() % wakefulness))
        
        original_eyes = babyState["eyes"]
        blinkDirection = random.choice([0, 1])
        
        babyState["eyes"] = blinkDirection
        time.sleep(metabolicRate)
        babyState["eyes"] = original_eyes
        
        if random.random() < 0.05:
            babyState["mouth"] += 1
            time.sleep(metabolicRate)
            babyState["eyes"] = blinkDirection
            time.sleep(metabolicRate)
            babyState["eyes"] = original_eyes
            if random.random() < 0.05:
                babyState["mouth"] += 1
                time.sleep(metabolicRate)
                babyState["eyes"] = blinkDirection
                time.sleep(metabolicRate)
                babyState["eyes"] = original_eyes


# --- THE BABY WIGGLE ---
def pulse_loop():
    print("[PULSE_LOOP] pulse_loop active.")
    while True:
        metabolicRate = babyState.get("metabolicRate", 0.1)
        if metabolicRate <= 0: metabolicRate = 0.1 # Prevent sleeping forever
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
            key = random.choice(["stretch_left", "stretch_right", "stretch_up", "stretch_down", "squish_left", "squish_right", "squish_up", "squish_down"])
            babyState[key] = random.choice([True, False])
        elif stimChoice == "blushy":
            if babyState["cheeks_on"]:
                babyState["cheeks_on"] = random.choice([True, False])
        
        time.sleep(metabolicRate)


# --- THE BABY JUMP ---
def smart_jump_loop():
    print("[SMART_JUMP_LOOP] smart_jump_loop active.")
    while True:
        if babyState.get("jumping"):
            metabolicRate = babyState.get("metabolicRate", 0.1) * 0.5
            if metabolicRate <= 0: metabolicRate = 0.05

            if random.random() < 0.05: babyState["cheeks_on"] = True
            time.sleep(metabolicRate)
            babyState["jumping"] = False
            babyState["cheeks_on"] = False
            if random.random() < 0.2:
                time.sleep(metabolicRate)
                babyState["jumping"] = True
                time.sleep(metabolicRate)
                babyState["jumping"] = False
                if random.random() < 0.15:
                    time.sleep(metabolicRate)
                    babyState["jumping"] = True
                    babyState["cheeks_on"] = True
                    time.sleep(metabolicRate)
                    babyState["jumping"] = False
                    if random.random() < 0.1:
                        time.sleep(metabolicRate)
                        babyState["jumping"] = True
                        time.sleep(metabolicRate)
                        babyState["jumping"] = False
                        if random.random() < 0.05:
                            time.sleep(metabolicRate)
                            babyState["jumping"] = True
                            time.sleep(metabolicRate)
                            babyState["jumping"] = False
            babyState["jumping"] = False
        time.sleep(0.1)


# --- THE BABY BLEND ---
def living_colour_loop():
    global lastTargetColour
    print("[LIVING_COLOUR_LOOP] living_colour_loop active.")
    while True:
        metabolicRate = babyState.get("metabolicRate", 0.1)
        if metabolicRate <= 0: metabolicRate = 0.1

        for channel in ["R", "G", "B"]:
            blend_speed = 0.25 * (random.choice([0.25, 0.5, 1, 2]) * metabolicRate)
            return_speed = 0.25 * (random.choice([0.25, 0.5]) * metabolicRate)

            if lastTargetColour != targetColour:
                delta = targetColour[channel] - babyState[channel]
                babyState[channel] += delta * (blend_speed * 2)
            else:
                delta_target = targetColour[channel] - babyState[channel]
                babyState[channel] += delta_target * (blend_speed * 0.25)
                delta_base = baseColour[channel] - babyState[channel]
                babyState[channel] += delta_base * return_speed

            babyState[channel] = int(max(0, min(255, babyState[channel])))
        
        lastTargetColour = targetColour.copy()
        time.sleep(0.05)

def speech_controller_loop():
    """Manages the isSpeaking state."""
    print("[SPEECH_CONTROLLER] Speech controller active.")
    speak_start_time = None
    while True:
        is_speaking = babyState.get("isSpeaking", False)

        if is_speaking and speak_start_time is None: speak_start_time = time.time()
        if is_speaking and speak_start_time is not None:
            # If speaking for more than 10 seconds, reset automatically
            if time.time() - speak_start_time > 10:
                print("[SPEECH_CONTROLLER] Speech timed out, resetting.")
                babyState["isSpeaking"] = False
                babyState["speechText"] = ""
                speak_start_time = None

        if not is_speaking: speak_start_time = None
            
        time.sleep(0.5)

# --- THE BABY MOVES! ---
threading.Thread(target=state_reader_loop, daemon=True).start()
threading.Thread(target=blink_loop, daemon=True).start()
threading.Thread(target=pulse_loop, daemon=True).start()
threading.Thread(target=smart_jump_loop, daemon=True).start()
threading.Thread(target=living_colour_loop, daemon=True).start()
threading.Thread(target=speech_controller_loop, daemon=True).start() # <-- ADDED THIS LINE


# --- app routes ---
@app.route("/api/state")
def get_state(): return jsonify(babyState)

@app.route("/api/say", methods=["POST"])
def user_say():
    """
    Receives text, passes it to the Discord bot via a file,
    waits for a response file, and returns the reply.
    """
    data = request.json
    text = data.get("text", "")
    
    if not text:
        return jsonify({"status": "error", "reply": "No text provided."}), 400

    request_id = str(uuid.uuid4())
    request_data = {"id": request_id, "text": text}
    response_file_path = os.path.join(RESPONSE_DIR, f"{request_id}.json")
    
    print(f"[API_SAY] Creating request {request_id} for text: '{text}'")
    
    try:
        with open(REQUEST_FILE_PATH, 'w') as f:
            json.dump(request_data, f)
    except Exception as e:
        print(f"[ERROR] Could not write request file: {e}")
        return jsonify({"status": "error", "reply": "..."}), 500

    start_time = time.time()
    timeout = 180
    reply_text = "... timeout :("

    try:
        while time.time() - start_time < timeout:
            if os.path.exists(response_file_path):
                # Response found!
                print(f"[API_SAY] Found response file for {request_id}")
                with open(response_file_path, 'r') as f:
                    response_data = json.load(f)
                reply_text = response_data.get("reply", "...")
                
                # Set the baby's state to speaking with the new text
                babyState["speechText"] = reply_text
                babyState["isSpeaking"] = True
                
                break # Exit the waiting loop
            time.sleep(0.1) # Check every 100ms to not fry the CPU
        else:
            # This 'else' belongs to the 'while' loop, it runs if the loop finishes without a 'break'
            print(f"[API_SAY] Timed out waiting for response for request {request_id}")

    finally:
        # 4. Clean up the files
        if os.path.exists(REQUEST_FILE_PATH):
            # We clear the request file instead of deleting to avoid race conditions
            # The bot will see it's an old request and ignore it.
            open(REQUEST_FILE_PATH, 'w').close()
        if os.path.exists(response_file_path):
            os.remove(response_file_path)

    return jsonify({"status": "ok", "reply": reply_text})

@app.route("/api/set", methods=["POST"])
def set_state():
    """Client can request changes, which the artistic loops will use."""
    updates = request.json
    
    if "R" in updates: targetColour["R"] = updates["R"]
    if "G" in updates: targetColour["G"] = updates["G"]
    if "B" in updates: targetColour["B"] = updates["B"]
    
    if "speechText" in updates: del updates["speechText"]
    if "isSpeaking" in updates: del updates["isSpeaking"] 

    babyState.update(updates)
    return jsonify({"status": "ok", "updated": updates})

@app.route("/api/bbybook")
def get_bbybook():
    """Reads and returns the contents of the bbybook.json file."""
    if os.path.exists(BBYBOOK_FILE_PATH):
        try:
            with open(BBYBOOK_FILE_PATH, 'r', encoding='utf-8') as f:
                facts = json.load(f)
            return jsonify(facts)
        except Exception as e:
            print(f"[ERROR] Could not read bbybook.json: {e}")
            return jsonify({"error": "could not read bbybook file"}), 500
    else:
        return jsonify({})

if __name__ == "__main__":
    print("--- bby on http://localhost:420 ---")
    app.run(host="0.0.0.0", port=420)