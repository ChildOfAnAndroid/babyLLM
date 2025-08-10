# bbyServer.py

from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import json
import os
import random
import uuid
import queue
import base64
from collections import deque
import array

# --- setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE_PATH = os.path.join(SCRIPT_DIR, "babyState.json")
PAINT_STATE_FILE_PATH = os.path.join(SCRIPT_DIR, "paintState.raw")
PAINT_TIMESTAMP_FILE_PATH = os.path.join(SCRIPT_DIR, "paintTimestamps.raw")
REQUEST_FILE_PATH = os.path.join(SCRIPT_DIR, "bby_request.json")
RESPONSE_DIR = os.path.join(SCRIPT_DIR, "bby_responses")
os.makedirs(RESPONSE_DIR, exist_ok=True)
BBYBOOK_FILE_PATH = os.path.expanduser("~/Dropbox/00_Icharis/02_LAB/01_babyLLM/SHKAIRA/soul/bbybook.json")
PAINT_LIFESPAN_FILE_PATH = os.path.join(SCRIPT_DIR, "paintLifespans.raw") 
CHAT_HISTORY_FILE_PATH = os.path.join(SCRIPT_DIR, "chatHistory.json")

app = Flask(__name__)
CORS(app)

# --- shared state ---
chat_history = []
chat_lock = threading.Lock()
paint_lock = threading.Lock()
paint_event_log = deque(maxlen=500)
paint_event_lock = threading.Lock()

# --- shared paint data ---
PAINT_DATA_SIZE = 64 * 64 * 4
PAINT_PIXEL_COUNT = 64 * 64
paint_rgba_data = bytearray(PAINT_DATA_SIZE)
paint_timestamp_data = array.array('L', [0] * PAINT_PIXEL_COUNT)
paint_lifespan_data = array.array('f', [0.0] * (PAINT_PIXEL_COUNT * 2))

# --- load data ---
try:
    if os.path.exists(CHAT_HISTORY_FILE_PATH):
        with chat_lock:
            with open(CHAT_HISTORY_FILE_PATH, 'r', encoding='utf-8') as f:
                chat_history = json.load(f)
                print(f"Loaded {len(chat_history)} messages from chatHistory.json")
except Exception as e:
    print(f"[ERROR] Could not load chat history: {e}")

try:
    if os.path.exists(PAINT_STATE_FILE_PATH):
        with paint_lock:
            with open(PAINT_STATE_FILE_PATH, 'rb') as f:
                data = f.read()
                if len(data) == PAINT_DATA_SIZE:
                    paint_rgba_data = bytearray(data)
                    print("Loaded existing RGBA state.")
    if os.path.exists(PAINT_TIMESTAMP_FILE_PATH):
        with paint_lock:
            with open(PAINT_TIMESTAMP_FILE_PATH, 'rb') as f:
                paint_timestamp_data.fromfile(f, PAINT_PIXEL_COUNT)
                # --- DIAGNOSTIC 1: Check data immediately after loading ---
                non_zero_timestamps = sum(1 for t in paint_timestamp_data if t > 0)
                print(f"[DATA_LOAD] Loaded existing timestamp state. Found {non_zero_timestamps} active pixels on disk.")

    if os.path.exists(PAINT_LIFESPAN_FILE_PATH):
        with paint_lock:
            with open(PAINT_LIFESPAN_FILE_PATH, 'rb') as f:
                paint_lifespan_data.fromfile(f, PAINT_PIXEL_COUNT * 2)
                print("Loaded existing lifespan state.")
except Exception as e:
    print(f"[ERROR] Could not load paint state: {e}")

def pixel_aging_loop():
    print("[PIXEL_AGING_LOOP] active.")
    while True:
        time.sleep(60) # Move sleep to the top to prevent a race condition on the first run
        
        pixels_faded = 0
        pixels_erased = 0
        active_pixels = 0
        try:
            with paint_lock:
                # --- DIAGNOSTIC 3: Log what the thread sees ---
                non_zero_in_thread = sum(1 for t in paint_timestamp_data if t > 0)
                print(f"[AGING_THREAD_CHECK] The aging loop sees {non_zero_in_thread} active pixels in memory.")

                now = int(time.time())
                pixels_changed_on_disk = False
                for i in range(PAINT_PIXEL_COUNT):
                    timestamp = paint_timestamp_data[i]
                    if timestamp == 0: continue 

                    active_pixels += 1 # Correctly increment counter
                    
                    age = now - timestamp
                    alpha_index = i * 4 + 3
                    current_alpha = paint_rgba_data[alpha_index]
                    fade_start_seconds = paint_lifespan_data[i * 2]
                    fade_end_seconds = paint_lifespan_data[i * 2 + 1]
                    fade_duration = fade_end_seconds - fade_start_seconds

                    if fade_duration <= 0: continue

                    new_alpha = current_alpha
                    if age > fade_end_seconds:
                        if current_alpha > 0:
                            new_alpha = 0
                            paint_timestamp_data[i] = 0
                            pixels_erased += 1 # Correctly increment counter
                    elif age > fade_start_seconds:
                        fade_progress = (age - fade_start_seconds) / fade_duration
                        new_alpha = int(255 * (1.0 - fade_progress))
                        if new_alpha < current_alpha:
                            pixels_faded += 1 # Correctly increment counter
                    
                    if new_alpha != current_alpha:
                        paint_rgba_data[alpha_index] = min(255, max(0, new_alpha))
                        pixels_changed_on_disk = True
                
                if pixels_changed_on_disk: 
                    with open(PAINT_STATE_FILE_PATH, 'wb') as f: f.write(paint_rgba_data)
                    
            print(f"[PIXEL_AGING_REPORT | {time.strftime('%H:%M:%S')}] Active Pixels: {active_pixels}, Fading: {pixels_faded}, Erased This Cycle: {pixels_erased}")
        except Exception as e: print(f"[ERROR] pixel_aging_loop: {e}")


# --- THE BABY SOUL ---
babyState = { "eyes": 5, "mouth": 1, "cheeks_on": False, "tears_on": False, "jumping": False, "stretch_left": False, "stretch_right": False, "stretch_up": False, "stretch_down": False, "squish_left": False, "squish_right": False, "squish_up": False, "squish_down": False, "isSpeaking": False, "speechText": "", "R": 133, "G": 239, "B": 238, "cerebralLoad": 0.0, "dreamIntensity": 0.0, "memoryFlux": 0.0, "learningStability": 0.0, "metabolicRate": 0.0, }
baseColour = {"R": 133, "G": 239, "B": 238}
targetColour = {"R": 133, "G": 239, "B": 238}
lastTargetColour = {"R": 133, "G": 239, "B": 238}

# --- background loops ---
def state_reader_loop():
    print("[STATE_READER_LOOP] active.")
    while True:
        if os.path.exists(STATE_FILE_PATH):
            try:
                with open(STATE_FILE_PATH, 'r') as f:
                    content = f.read()
                    if not content.strip():
                        time.sleep(0.1); continue
                    updates = json.loads(content)
                    if babyState.get("isSpeaking", False): updates.pop("mouth", None)
                    babyState.update(updates)
                    if "R" in updates: targetColour["R"] = updates["R"]
                    if "G" in updates: targetColour["G"] = updates["G"]
                    if "B" in updates: targetColour["B"] = updates["B"]
                    
                    if babyState.get("correct") == True:
                        babyState["R"] = min(255, int(babyState["R"] * 1.05))
                        if babyState["mouth"] >= 0:
                            smile = random.choice([0,0,0,1])
                            babyState["mouth"] += smile
                        else: babyState["mouth"] = 5
                    
                    dreamIntensity = babyState.get("dreamIntensity", 0.0)
                    bpm = 62
                    bpm32th = 60 / (bpm * 16)
                    metabolicRate = round(dreamIntensity) * bpm32th
                    babyState["metabolicRate"] = metabolicRate
                    
            except Exception as e:
                print(f"[ERROR] state_reader_loop: {e}")
        time.sleep(0.1)

def blink_loop():
    print("[BLINK_LOOP] active.")
    while True:
        try:
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
        except Exception as e:
            print(f"[ERROR] blink_loop: {e}")

def speak_loop():
    print("[SPEAK_LOOP] active.")
    restingMouth = 1
    lastState = False
    
    while True:
        try:
            if babyState.get("isSpeaking", False):
                if not lastState:
                    restingMouth = babyState["mouth"]
                    lastState = True

                babyState["mouth"] = random.randint(55, 65)
                time.sleep(max(0.05, babyState.get("metabolicRate", 0.1)))

                if random.random() < 0.25:
                    babyState["mouth"] = restingMouth
                    time.sleep(max(0.05, babyState.get("metabolicRate", 0.1)))

            else:
                if lastState:
                    babyState["mouth"] = restingMouth
                    lastState = False
                time.sleep(0.1)
        except Exception as e: print(f"[ERROR] speak_loop: {e}")

def pulse_loop():
    print("[PULSE_LOOP] active.")
    while True:
        try:
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
                key = random.choice(["stretch_left", "stretch_right", "stretch_up", "stretch_down", "squish_left", "squish_right", "squish_up", "squish_down"])
                babyState[key] = random.choice([True, False])
            elif stimChoice == "blushy":
                if babyState["cheeks_on"]:
                    babyState["cheeks_on"] = random.choice([True, False])
            time.sleep(babyState.get("metabolicRate", 0.1))
        except Exception as e:
            print(f"[ERROR] pulse_loop: {e}")

def smart_jump_loop():
    print("[SMART_JUMP_LOOP] active.")
    while True:
        try:
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
        except Exception as e:
            print(f"[ERROR] smart_jump_loop: {e}")

def living_colour_loop():
    global lastTargetColour
    print("[LIVING_COLOUR_LOOP] active.")
    while True:
        try:
            metabolicRate = babyState.get("metabolicRate", 0.1)
            if metabolicRate <= 0: metabolicRate = 0.1

            for channel in ["R", "G", "B"]:
                blend_speed = 0.25 * (random.choice([0.25, 0.5, 1, 2]) * metabolicRate)
                return_speed = 0.25 * (random.choice([0.25, 0.5]) * metabolicRate)

                if lastTargetColour != targetColour:
                    delta = targetColour[channel] - babyState[channel]
                    babyState[channel] += int(delta * (blend_speed * 2))
                else:
                    delta_target = targetColour[channel] - babyState[channel]
                    babyState[channel] += int(delta_target * (blend_speed * 0.25))
                    delta_base = baseColour[channel] - babyState[channel]
                    babyState[channel] += int(delta_base * return_speed)

                babyState[channel] = int(max(0, min(255, babyState[channel])))
            
            lastTargetColour = targetColour.copy()
        except Exception as e:
            print(f"[ERROR] living_colour_loop: {e}")
        time.sleep(0.05)

def speech_controller_loop():
    print("[SPEECH_CONTROLLER] active.")
    speak_start_time = None
    while True:
        try:
            is_speaking = babyState.get("isSpeaking", False)

            if is_speaking and speak_start_time is None:
                speak_start_time = time.time()
            if is_speaking and speak_start_time is not None:
                if time.time() - speak_start_time > 10:
                    babyState["isSpeaking"] = False
                    babyState["speechText"] = ""
                    speak_start_time = None

            if not is_speaking:
                speak_start_time = None
        except Exception as e:
            print(f"[ERROR] speech_controller_loop: {e}")
        time.sleep(0.5)

# start background state loops
for fn in (state_reader_loop, blink_loop, pulse_loop, smart_jump_loop, living_colour_loop, speech_controller_loop, pixel_aging_loop, speak_loop):
    threading.Thread(target=fn, daemon=True).start()

# --- queue ---
job_queue = queue.Queue()
pending = {}  # request_id -> {"event": threading.Event(), "reply": str}

def worker_loop():
    print("[WORKER] active. sequentially processing /api/say jobs")
    while True:
        job = job_queue.get()
        if job is None:
            break
        request_id = job["id"]
        text = job["text"]
        author = job["author"]
        response_file_path = os.path.join(RESPONSE_DIR, f"{request_id}.json")

        try:
            with open(REQUEST_FILE_PATH, "w", encoding="utf-8") as f: json.dump({"id": request_id, "text": text, "author": author}, f)
        except Exception as e: print("[WORKER][ERROR] writing request:", e)

        start = time.time(); timeout = 180.0; reply_text = "... timeout :("
        while time.time() - start < timeout:
            if os.path.exists(response_file_path):
                try:
                    with open(response_file_path, "r", encoding="utf-8") as f: data = json.load(f)
                    reply_text = data.get("reply", "...")
                except Exception as e: reply_text = f"... error reading reply: {e}"
                break
            time.sleep(0.1)

        try:
            if os.path.exists(REQUEST_FILE_PATH): open(REQUEST_FILE_PATH, "w").close()
            if os.path.exists(response_file_path): os.remove(response_file_path)
        except Exception: pass

        babyState["speechText"] = reply_text
        babyState["isSpeaking"] = True
        item = pending.get(request_id)
        if item:
            item["reply"] = reply_text
            item["event"].set()

threading.Thread(target=worker_loop, daemon=True).start()

# --- routes ---

@app.route("/api/state")
def get_state(): return jsonify(babyState)

@app.route("/api/chat_history")
def get_chat_history():
    with chat_lock: return jsonify(chat_history)

@app.route("/api/get_paint_canvas")
def get_paint_canvas():
    with paint_lock: paint_b64 = base64.b64encode(paint_rgba_data).decode('utf-8')
    return jsonify({"paintOverlayData_b64": paint_b64})

@app.route("/api/paint_pixel", methods=["POST"])
def paint_pixel():
    data = request.json or {}
    pixels = data.get('pixels')
    if not pixels or not isinstance(pixels, list):
        return jsonify({"status": "error", "message": "Invalid payload"}), 400

    try:
        now = int(time.time())
        with paint_lock:
            for p in pixels:
                x, y, r, g, b, a = p['x'], p['y'], p['r'], p['g'], p['b'], p['a']
                if 0 <= x < 64 and 0 <= y < 64:
                    pixel_index = y * 64 + x
                    rgba_index = pixel_index * 4
                    lifespan_index = pixel_index * 2
                    paint_rgba_data[rgba_index:rgba_index+4] = [r, g, b, a]
                    if a > 0:
                        paint_timestamp_data[pixel_index] = now
                        # --- DIAGNOSTIC 2: Confirm the timestamp was set ---
                        print(f"[PAINT_PIXEL] Set timestamp for pixel ({p['x']},{p['y']}) to {now}")
                        start_fade = (random.random() * 1) * 60 * 60
                        end_fade = start_fade + ((random.random() * 47) + 1) * 60 * 60
                        paint_lifespan_data[lifespan_index] = start_fade
                        paint_lifespan_data[lifespan_index + 1] = end_fade
                    else:
                        paint_timestamp_data[pixel_index] = 0
                        paint_lifespan_data[lifespan_index] = 0.0
                        paint_lifespan_data[lifespan_index + 1] = 0.0
            
            with open(PAINT_STATE_FILE_PATH, 'wb') as f: f.write(paint_rgba_data)
            with open(PAINT_TIMESTAMP_FILE_PATH, 'wb') as f: paint_timestamp_data.tofile(f)
            with open(PAINT_LIFESPAN_FILE_PATH, 'wb') as f: paint_lifespan_data.tofile(f)
            
        event = {'id': str(uuid.uuid4()), 'ts': time.time(), 'pixels': pixels}
        with paint_event_lock:
            paint_event_log.append(event)
        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"[ERROR] /api/paint_pixel: {e}")
        return jsonify({"status": "error", "message": "Failed to process pixel data"}), 500
    
@app.route("/api/paint_events")
def get_paint_events():
    since_id = request.args.get('since')
    with paint_event_lock:
        if since_id:
            try:
                events_to_send = []
                for event in reversed(paint_event_log):
                    if event['id'] == since_id:
                        break
                    events_to_send.append(event)
                return jsonify(list(reversed(events_to_send)))
            except ValueError: return jsonify(list(paint_event_log))
        else: return jsonify([])

@app.route("/api/set", methods=["POST"])
def set_state():
    updates = request.json or {}
    if "R" in updates: targetColour["R"] = int(updates["R"])
    if "G" in updates: targetColour["G"] = int(updates["G"])
    if "B" in updates: targetColour["B"] = int(updates["B"])
    updates.pop("R", None); updates.pop("G", None); updates.pop("B", None)
    updates.pop("speechText", None); updates.pop("isSpeaking", None)
    babyState.update(updates)
    return jsonify({"status": "ok", "updated": updates})

@app.route("/api/bbybook")
def get_bbybook():
    if os.path.exists(BBYBOOK_FILE_PATH):
        try:
            with open(BBYBOOK_FILE_PATH, 'r', encoding='utf-8') as f:
                facts = json.load(f)
            return jsonify(facts)
        except Exception as e:
            print(f"[ERROR] could not read bbybook.json: {e}")
            return jsonify({"error": "could not read bbybook file"}), 500
    else:
        return jsonify({})

@app.route("/api/say", methods=["POST"])
def user_say():
    data = request.json or {}
    text = data.get("text", "")
    author = data.get("author", "kevinonline420")
    colour = data.get("colour", {"r": 133, "g": 239, "b": 238})

    if not text: return jsonify({"status": "error", "reply": "no text :("}), 400

    user_message = { "id": str(uuid.uuid4()), "author": author, "text": text, "timestamp": time.time(), "colour": colour }
    with chat_lock:
        chat_history.append(user_message)
        if len(chat_history) > 100: chat_history.pop(0)

    request_id = str(uuid.uuid4())
    done = threading.Event()
    pending[request_id] = {"event": done, "reply": "..."}
    job_queue.put({"id": request_id, "text": text, "author": author})

    timeout = 180.0
    finished = done.wait(timeout)
    reply_text = pending[request_id]["reply"]
    pending.pop(request_id, None)

    bot_bubble_colour = { "r": babyState.get("R", 133), "g": babyState.get("G", 239), "b": babyState.get("B", 238) }
    bot_message = { "id": str(uuid.uuid4()), "author": "babyLLM", "text": reply_text, "timestamp": time.time(), "colour": bot_bubble_colour }
    
    with chat_lock:
        chat_history.append(bot_message)
        if len(chat_history) > 100: chat_history.pop(0)

        try:
            with open(CHAT_HISTORY_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(chat_history, f)
        except Exception as e:
            print(f"[ERROR] cant save chat history: {e}")

    return jsonify({"status": "ok" if finished else "timeout", "reply": reply_text, "author": author})

if __name__ == "__main__":
    print("--- bby queued on http://localhost:420 ---")
    app.run(host="0.0.0.0", port=420, threaded=True)