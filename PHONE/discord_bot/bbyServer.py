# bbyServer.py

from flask import Flask, jsonify, request, send_from_directory
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

# --- snapshots (server keeps data; client can attach face PNG) ---
SNAP_DIR = os.path.join(SCRIPT_DIR, "snapshots")
SNAP_META = os.path.join(SNAP_DIR, "index.json")
os.makedirs(SNAP_DIR, exist_ok=True)
try:
    with open(SNAP_META, "r", encoding="utf-8") as f:
        snapshot_index = json.load(f)
except Exception:
    snapshot_index = []  # [{id, ts, label, has_png}]

def _write_snap_index():
    try:
        with open(SNAP_META, "w", encoding="utf-8") as f:
            json.dump(snapshot_index, f)
    except Exception as e:
        print("[ERROR] writing snapshot index:", e)

def _save_snapshot(label=""):
    """Save raw paint + current baby state. Returns meta dict."""
    snap_id = str(uuid.uuid4())
    ts = int(time.time())
    raw_path   = os.path.join(SNAP_DIR, f"{snap_id}.raw")
    state_path = os.path.join(SNAP_DIR, f"{snap_id}.state.json")
    with paint_lock:
        buf = bytes(paint_rgba_data)
        state = dict(babyState)
    with open(raw_path, "wb") as f:
        f.write(buf)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f)
    meta = {"id": snap_id, "ts": ts, "label": label, "has_png": False}
    snapshot_index.append(meta)
    _write_snap_index()
    return meta

def _attach_png(snap_id: str, png_bytes: bytes):
    """Attach a composite PNG (full face) to an existing snapshot."""
    png_path = os.path.join(SNAP_DIR, f"{snap_id}.png")
    with open(png_path, "wb") as f:
        f.write(png_bytes)
    for m in snapshot_index:
        if m["id"] == snap_id:
            m["has_png"] = True
            break
    _write_snap_index()

app = Flask(__name__)
CORS(app)

# ---- fade config ----
P_SHORT  = 0.001   # ~1%: very short
P_MEDIUM = 0.899   # ~89%: 4–72 hours 
P_LONG   = 0.10   # ~10%: 6–21 days
REPAINT_REFRESHES_LIFE = False
STROKE_COHERENCE = True
COHERENCE_JITTER = 0.12
REPAINT_POLICY = "diff_color_refresh"  # "always" | "never" | "diff_color_refresh"

def _sample_total_seconds():
    u = random.random()
    if u < P_SHORT:
        return random.uniform(2 * 60, 60 * 60)                 # 2–60 min
    elif u < P_SHORT + P_MEDIUM:
        return random.uniform(4 * 3600, 72 * 3600)            # 4–72 h
    else:
        return random.uniform(6 * 24 * 3600, 21 * 24 * 3600)   # 6–21 days

def _split_linger_fade(total_seconds: float):
    linger_frac = random.uniform(0.0, 0.25)
    start = total_seconds * linger_frac
    end = max(start + 1.0, total_seconds)  # ensure at least 1s of fade
    return float(start), float(end)

# --- shared state ---
chat_history = []
chat_lock = threading.Lock()
paint_lock = threading.Lock()
paint_event_log = deque(maxlen=2000)
paint_event_lock = threading.Lock()

# --- shared paint data ---
PAINT_W = 64
PAINT_H = 64
PAINT_PIXEL_COUNT = PAINT_W * PAINT_H
PAINT_DATA_SIZE = PAINT_PIXEL_COUNT * 4

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
                paint_timestamp_data = array.array('L')
                paint_timestamp_data.fromfile(f, PAINT_PIXEL_COUNT)
                print("Loaded existing timestamp state.")
    if os.path.exists(PAINT_LIFESPAN_FILE_PATH):
        with paint_lock:
            with open(PAINT_LIFESPAN_FILE_PATH, 'rb') as f:
                paint_lifespan_data = array.array('f')
                paint_lifespan_data.fromfile(f, PAINT_PIXEL_COUNT * 2)
                print("Loaded existing lifespan state.")
except Exception as e:
    print(f"[ERROR] Could not load paint state: {e}")

def pixel_aging_loop():
    print("[PIXEL_AGING_LOOP] active.")
    while True:
        try:
            pixels_faded = 0
            pixels_erased = 0
            active_pixels = 0
            changed_pixels = []

            with paint_lock:
                now = int(time.time())
                for i in range(PAINT_PIXEL_COUNT):
                    ts = paint_timestamp_data[i]
                    if ts == 0:
                        continue
                    active_pixels += 1

                    age = now - ts
                    a_idx = i * 4 + 3
                    cur_a = paint_rgba_data[a_idx]
                    start_fade = paint_lifespan_data[i * 2]
                    end_fade   = paint_lifespan_data[i * 2 + 1]
                    dur = end_fade - start_fade
                    if dur <= 0:
                        continue

                    new_a = cur_a
                    if age > end_fade:
                        if cur_a > 0:
                            new_a = 0
                            paint_timestamp_data[i] = 0
                            paint_lifespan_data[i * 2] = 0.0
                            paint_lifespan_data[i * 2 + 1] = 0.0
                            pixels_erased += 1
                    elif age > start_fade:
                        fade_progress = (age - start_fade) / dur
                        fade_progress = max(0.0, min(1.0, fade_progress))
                        new_a = int(255 * (1.0 - fade_progress))  # absolute fade

                    if new_a != cur_a:
                        paint_rgba_data[a_idx] = max(0, min(255, new_a))
                        x = i % PAINT_W
                        y = i // PAINT_W
                        rgba = i * 4
                        changed_pixels.append({
                            "x": x, "y": y,
                            "r": paint_rgba_data[rgba],
                            "g": paint_rgba_data[rgba + 1],
                            "b": paint_rgba_data[rgba + 2],
                            "a": paint_rgba_data[a_idx],
                        })
                        if start_fade < age <= end_fade:
                            pixels_faded += 1

                if changed_pixels:
                    with open(PAINT_STATE_FILE_PATH, 'wb') as f: f.write(paint_rgba_data)
                    with open(PAINT_TIMESTAMP_FILE_PATH, 'wb') as f: paint_timestamp_data.tofile(f)
                    with open(PAINT_LIFESPAN_FILE_PATH, 'wb') as f: paint_lifespan_data.tofile(f)

            print(f"[PIXEL_AGING_REPORT | {time.strftime('%H:%M:%S')}] Active Pixels: {active_pixels}, Fading: {pixels_faded}, Erased This Cycle: {pixels_erased}")

            if changed_pixels:
                event = {"id": str(uuid.uuid4()), "ts": time.time(), "pixels": changed_pixels}
                with paint_event_lock:
                    paint_event_log.append(event)

        except Exception as e:
            print(f"[ERROR] pixel_aging_loop: {e}")

        time.sleep(60)

# --- activity tracker for auto-snapshots ---
activity_lock = threading.Lock()
BURST_WINDOW = 30            # look back N seconds
BURST_THRESHOLD_PX = 200     # “lots of activity” in that window
IDLE_SNAPSHOT_AFTER = 60     # after burst ends, wait this long with no paint
recent_paints = deque()      # (ts, num_pixels)
last_paint_ts = 0.0
burst_active = False
burst_start_ts = 0.0
last_autosnap_ts = 0.0
last_autosnap_id = None

def _register_paint(n):
    """Call this once per /api/paint_pixel to record activity."""
    global last_paint_ts, burst_active, burst_start_ts
    now = time.time()
    with activity_lock:
        last_paint_ts = now
        recent_paints.append((now, n))
        cutoff = now - BURST_WINDOW
        while recent_paints and recent_paints[0][0] < cutoff:
            recent_paints.popleft()
        total = sum(k for _, k in recent_paints)
        if not burst_active and total >= BURST_THRESHOLD_PX:
            burst_active = True
            burst_start_ts = now
        return total

def autosnap_loop():
    """Create a snapshot after a burst cools off for > IDLE_SNAPSHOT_AFTER."""
    global burst_active, last_autosnap_ts, last_autosnap_id
    print("[AUTOSNAP_LOOP] active.")
    while True:
        time.sleep(5)
        now = time.time()
        with activity_lock:
            should = burst_active and (now - last_paint_ts) >= IDLE_SNAPSHOT_AFTER
        if should:
            meta = _save_snapshot(label="auto-burst")
            with activity_lock:
                burst_active = False
                last_autosnap_ts = now
                last_autosnap_id = meta["id"]
            print(f"[AUTOSNAP] {meta['id']}")

# --- THE BABY SOUL ---
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
        except Exception as e:
            print(f"[ERROR] speak_loop: {e}")

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
for fn in (
    state_reader_loop,
    blink_loop,
    pulse_loop,
    smart_jump_loop,
    living_colour_loop,
    speech_controller_loop,
    pixel_aging_loop,
    speak_loop,
    autosnap_loop,  # <-- new
):
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
            with open(REQUEST_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump({"id": request_id, "text": text, "author": author}, f)
        except Exception as e:
            print("[WORKER][ERROR] writing request:", e)

        start = time.time(); timeout = 180.0; reply_text = "... timeout :("
        while time.time() - start < timeout:
            if os.path.exists(response_file_path):
                try:
                    with open(response_file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    reply_text = data.get("reply", "...")
                except Exception as e:
                    reply_text = f"... error reading reply: {e}"
                break
            time.sleep(0.1)

        try:
            if os.path.exists(REQUEST_FILE_PATH): open(REQUEST_FILE_PATH, "w").close()
            if os.path.exists(response_file_path): os.remove(response_file_path)
        except Exception:
            pass

        babyState["speechText"] = reply_text
        babyState["isSpeaking"] = True
        item = pending.get(request_id)
        if item:
            item["reply"] = reply_text
            item["event"].set()

threading.Thread(target=worker_loop, daemon=True).start()

# --- routes ---

@app.route("/api/state")
def get_state():
    return jsonify(babyState)

@app.route("/api/chat_history")
def get_chat_history():
    with chat_lock:
        return jsonify(chat_history)

@app.route("/api/get_paint_canvas")
def get_paint_canvas():
    with paint_lock:
        paint_b64 = base64.b64encode(paint_rgba_data).decode('utf-8')
    return jsonify({"paintOverlayData_b64": paint_b64})

@app.route("/api/paint_pixel", methods=["POST"])
def paint_pixel():
    data = request.json or {}
    pixels = data.get('pixels')
    if not pixels or not isinstance(pixels, list):
        return jsonify({"status": "error", "message": "Invalid payload"}), 400

    try:
        now = int(time.time())
        stroke_base_total = _sample_total_seconds() if STROKE_COHERENCE else None

        with paint_lock:
            for p in pixels:
                x, y, r, g, b, a = p['x'], p['y'], p['r'], p['g'], p['b'], p['a']
                if 0 <= x < PAINT_W and 0 <= y < PAINT_H:
                    pixel_index = y * PAINT_W + x
                    rgba_index = pixel_index * 4
                    lifespan_index = pixel_index * 2

                    old_r = paint_rgba_data[rgba_index]
                    old_g = paint_rgba_data[rgba_index + 1]
                    old_b = paint_rgba_data[rgba_index + 2]
                    same_colour_rgb = (old_r == r and old_g == g and old_b == b)

                    # decide refresh policy
                    if REPAINT_POLICY == "always":
                        refresh = True
                    elif REPAINT_POLICY == "never":
                        refresh = (paint_timestamp_data[pixel_index] == 0)
                    elif REPAINT_POLICY == "diff_color_refresh":
                        refresh = (paint_timestamp_data[pixel_index] == 0) or (not same_colour_rgb)
                    else:
                        refresh = (paint_timestamp_data[pixel_index] == 0)

                    # write RGBA
                    paint_rgba_data[rgba_index:rgba_index + 4] = [r, g, b, a]

                    if a > 0:
                        if refresh:
                            paint_timestamp_data[pixel_index] = now
                            if STROKE_COHERENCE and stroke_base_total is not None:
                                total = stroke_base_total * random.uniform(1 - COHERENCE_JITTER, 1 + COHERENCE_JITTER)
                            else:
                                total = _sample_total_seconds()
                            start_fade, end_fade = _split_linger_fade(total)
                            paint_lifespan_data[lifespan_index] = start_fade
                            paint_lifespan_data[lifespan_index + 1] = end_fade
                    else:
                        paint_timestamp_data[pixel_index] = 0
                        paint_lifespan_data[lifespan_index] = 0.0
                        paint_lifespan_data[lifespan_index + 1] = 0.0

            # persist (simple and fine at this scale)
            with open(PAINT_STATE_FILE_PATH, 'wb') as f: f.write(paint_rgba_data)
            with open(PAINT_TIMESTAMP_FILE_PATH, 'wb') as f: paint_timestamp_data.tofile(f)
            with open(PAINT_LIFESPAN_FILE_PATH, 'wb') as f: paint_lifespan_data.tofile(f)

        # record activity for auto-snapshot logic
        _register_paint(len(pixels))

        # echo batch as event
        event = {'id': str(uuid.uuid4()), 'ts': time.time(), 'pixels': pixels}
        with paint_event_lock: paint_event_log.append(event)
        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"[ERROR] /api/paint_pixel: {e}")
        return jsonify({"status": "error", "message": "Failed to process pixel data"}), 500

@app.route("/api/paint_events")
def get_paint_events():
    since_id = request.args.get('since')
    with paint_event_lock:
        events = list(paint_event_log)
        if not events:
            return jsonify([])

        if not since_id:
            # bootstrap: give the latest id so clients can latch
            last = events[-1]
            return jsonify([{"id": last["id"], "pixels": []}])

        out = []
        for ev in reversed(events):
            if ev['id'] == since_id:
                break
            out.append(ev)
        return jsonify(list(reversed(out)))

# --- snapshot + activity routes ---

@app.route("/api/snapshot", methods=["POST"])
def create_snapshot():
    data = request.json or {}
    label = data.get("label", "")
    meta = _save_snapshot(label)

    png_url = None
    b64 = data.get("composite_png_b64")
    if b64:
        _attach_png(meta["id"], base64.b64decode(b64.split(",")[-1]))
        meta["has_png"] = True
        png_url = request.host_url.rstrip("/") + f"/snapshots/{meta['id']}.png"

    return jsonify({"status": "ok", "snapshot": meta, "png_url": png_url})

@app.route("/api/snapshot_attach_png/<snap_id>", methods=["POST"])
def attach_snapshot_png(snap_id):
    b64 = (request.json or {}).get("composite_png_b64")
    if not b64:
        return jsonify({"error":"missing composite_png_b64"}), 400
    _attach_png(snap_id, base64.b64decode(b64.split(",")[-1]))
    return jsonify({"status":"ok"})

@app.route("/api/snapshots")
def list_snapshots():
    out = []
    base = request.host_url.rstrip("/")
    for m in snapshot_index:
        item = dict(m)
        if m.get("has_png"):
            item["png_url"] = f"{base}/snapshots/{m['id']}.png"
        out.append(item)
    return jsonify(out)


@app.route("/api/snapshot/<snap_id>")
def get_snapshot(snap_id):
    path = os.path.join(SNAP_DIR, f"{snap_id}.raw")
    if not os.path.exists(path):
        return jsonify({"error":"not found"}), 404
    with open(path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    return jsonify({"w": 64, "h": 64, "rgba_b64": b64})

@app.route("/snapshots/<snap_id>.png")
def serve_snapshot_png(snap_id):
    path = os.path.join(SNAP_DIR, f"{snap_id}.png")
    if not os.path.exists(path): return jsonify({"error": "not found"}), 404
    return send_from_directory(SNAP_DIR, f"{snap_id}.png", mimetype="image/png")

@app.route("/api/activity")
def activity():
    now = time.time()
    with activity_lock:
        total = sum(k for t, k in recent_paints if t >= now - BURST_WINDOW)
        idle = (now - last_paint_ts) if last_paint_ts else 1e9
        return jsonify({
            "pixels_last_window": total,
            "window_seconds": BURST_WINDOW,
            "idle_seconds": idle,
            "burst_active": burst_active,
            "burst_started": burst_start_ts if burst_active else None,
            "last_autosnap_id": last_autosnap_id,
            "last_autosnap_ts": last_autosnap_ts,
        })

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

# --- chat ---

job_queue = job_queue  # (already defined above)
pending = pending

@app.route("/api/say", methods=["POST"])
def user_say():
    data = request.json or {}
    text = data.get("text", "")
    author = data.get("author", "kevinonline420")
    colour = data.get("colour", {"r": 133, "g": 239, "b": 238})

    if not text:
        return jsonify({"status": "error", "reply": "no text :("}), 400

    user_message = {"id": str(uuid.uuid4()), "author": author, "text": text, "timestamp": time.time(), "colour": colour}
    with chat_lock:
        chat_history.append(user_message)
        if len(chat_history) > 100:
            chat_history.pop(0)

    request_id = str(uuid.uuid4())
    done = threading.Event()
    pending[request_id] = {"event": done, "reply": "..."}
    job_queue.put({"id": request_id, "text": text, "author": author})

    timeout = 180.0
    finished = done.wait(timeout)
    reply_text = pending[request_id]["reply"]
    pending.pop(request_id, None)

    bot_bubble_colour = {"r": babyState.get("R", 133), "g": babyState.get("G", 239), "b": babyState.get("B", 238)}
    bot_message = {"id": str(uuid.uuid4()), "author": "babyLLM", "text": reply_text, "timestamp": time.time(), "colour": bot_bubble_colour}

    with chat_lock:
        chat_history.append(bot_message)
        if len(chat_history) > 100:
            chat_history.pop(0)

        try:
            with open(CHAT_HISTORY_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(chat_history, f)
        except Exception as e:
            print(f"[ERROR] cant save chat history: {e}")

    return jsonify({"status": "ok" if finished else "timeout", "reply": reply_text, "author": author})

if __name__ == "__main__":
    print("--- bby queued on http://localhost:420 ---")
    app.run(host="0.0.0.0", port=420, threaded=True)
