
from flask import Flask, send_file, request
from PIL import Image
import io
import threading
import time
import math
import random

app = Flask(__name__)

bbyBODY = Image.open("bbyBODY.png").convert("RGBA")
bbyCHEEKS = Image.open("bbyCHEEKS.png").convert("RGBA")
bbyEYES = Image.open("bbyEYES.png").convert("RGBA")
bbyMOUTH = Image.open("bbyMOUTH.png").convert("RGBA")

spriteSize = (64, 64)
bbyBODY_numLayers = 5

canvasSize = (80, 80)
offsetCalc = round((80-64) / 2)

# normally be updated live by BabyLLM's training process
babyState = {
    "eyes": 1,
    "mouth": 5,
    "cheeks_on": False,
    "tears_on": False,
    "jumping": False,
    "stretch_left": False,
    "stretch_right": False,
    "stretch_up": False,
    "stretch_down": False,
    "squish_left": False,
    "squish_right": False,
    "squish_up": False,
    "squish_down": False,
    "speech": "",
}

def blink_loop():
    while True:
        time.sleep(1 + (time.time() % 6))

        sleepTime = random.choice([0.119, 0.119, 0.119, 0.119, 0.119, 0.119, 0.119, 0.119, 0.238, 0.238, 0.472])
        original_eyes = babyState["eyes"]  # save current emotion
        babyState["eyes"] = 0  # blink column
        print(f"*blimk*")
        time.sleep(sleepTime)
        
        babyState["eyes"] = original_eyes  # restore emotion
        
        if random.random() < 0.05:
            time.sleep(sleepTime)
            babyState["eyes"] = 0  # blink column
            print(f"**blimk**")
            time.sleep(sleepTime)
            
            babyState["eyes"] = original_eyes  # restore emotion
            
            if random.random() < 0.05:
                time.sleep(sleepTime)
                babyState["eyes"] = 0  # blink column
                print(f"***blimk***")
                time.sleep(sleepTime)
                
                babyState["eyes"] = original_eyes  # restore emotion
                time.sleep(sleepTime)


threading.Thread(target=blink_loop, daemon=True).start()

def pulse_loop():
    while True:
        # This could be replaced later with a real layer-based signal
        #babyState["pulse"] = 0.5 + 0.5 * math.sin(time.time() * 4)  # smooth breathing
        keyChoice = random.choice(["stretch_left", "stretch_right", "stretch_up", "stretch_down", "squish_left", "squish_right", "squish_up", "squish_down",])
        babyState[keyChoice] = random.choice([True, False])
        time.sleep(0.472)

threading.Thread(target=pulse_loop, daemon=True).start()

def jump_reset():
    jumpTime = random.choice([0.119, 0.0595])
    while True:
        if babyState.get("jumping"):
            time.sleep(jumpTime)
            babyState["jumping"] = False
            if random.random() < 0.2:
                time.sleep(jumpTime)
                babyState["jumping"] = True
                time.sleep(jumpTime)
                babyState["jumping"] = False
                if random.random() < 0.15:
                    time.sleep(jumpTime)
                    babyState["jumping"] = True
                    time.sleep(jumpTime)
                    babyState["jumping"] = False
                    if random.random() < 0.1:
                        time.sleep(jumpTime)
                        babyState["jumping"] = True
                        time.sleep(jumpTime)
                        babyState["jumping"] = False
                        if random.random() < 0.05:
                            time.sleep(jumpTime)
                            babyState["jumping"] = True
                            time.sleep(jumpTime)
                            babyState["jumping"] = False
        time.sleep(0.1)

threading.Thread(target=jump_reset, daemon=True).start()

def buildBabySprite():
    # Directional stretch (max 1px per side)
    stretch_left  = int(bool(babyState.get("stretch_left", False)))
    stretch_right = int(bool(babyState.get("stretch_right", False)))
    stretch_up    = int(bool(babyState.get("stretch_up", False)))
    stretch_down  = int(bool(babyState.get("stretch_down", False)))

    squish_left  = int(bool(babyState.get("squish_left", False)))
    squish_right = int(bool(babyState.get("squish_right", False)))
    squish_up    = int(bool(babyState.get("squish_up", False)))
    squish_down  = int(bool(babyState.get("squish_down", False)))

    # Final target dimensions
    stretch_x = spriteSize[0] + stretch_left + stretch_right - squish_left - squish_right # 62-66
    stretch_y = spriteSize[1] + stretch_up + stretch_down - squish_up - squish_down   # 62-66

    # Create base body layer
    bbyHeight = bbyBODY.height // bbyBODY_numLayers
    bbyBODY_full = Image.new("RGBA", spriteSize, (0, 0, 0, 0))
    for i in range(bbyBODY_numLayers):
        frame = bbyBODY.crop((0, i * bbyHeight, bbyBODY.width, (i + 1) * bbyHeight))
        bbyBODY_full.paste(frame, (0, 0), frame)

    # Stretch body
    body_stretched = bbyBODY_full.resize((stretch_x, stretch_y), resample=Image.NEAREST)

    # Final canvas
    final = Image.new("RGBA", canvasSize, (0, 0, 0, 0))

    # Centering offset, accounting for left/up stretch
    offset_x = (canvasSize[0] - stretch_x) // 2 - stretch_left
    offset_y = (canvasSize[1] - stretch_y) // 2 - stretch_up

    # Add jump
    jumpset = 0
    if babyState.get("jumping"):
        jumpset = -4
        offset_y += jumpset

    offsetCalc_y = offsetCalc + jumpset

    # Paste stretched body
    final.paste(body_stretched, (offset_x, offset_y), body_stretched)

    # -- FACE LAYERS --

    # Eyes
    eye_rows = [0, 1, 2]
    if babyState["tears_on"]:
        eye_rows.append(3)

    bbyEYES_full = Image.new("RGBA", spriteSize, (0, 0, 0, 0))
    for row in eye_rows:
        frame = bbyEYES.crop((
            babyState["eyes"] * spriteSize[0], row * spriteSize[1],
            (babyState["eyes"] + 1) * spriteSize[0], (row + 1) * spriteSize[1]
        ))
        bbyEYES_full.paste(frame, (0, 0), frame)

    bbyCHEEKS_full = bbyCHEEKS.crop((0, 0, spriteSize[0], spriteSize[1]))
    bbyMOUTH_full = bbyMOUTH.crop((
        babyState["mouth"] * spriteSize[0], 0,
        (babyState["mouth"] + 1) * spriteSize[0], spriteSize[1]
    ))

    # Paste facial layers (same offset as body)
    if babyState["cheeks_on"]:
        final.paste(bbyCHEEKS_full, (offsetCalc, offsetCalc_y), bbyCHEEKS_full)
    final.paste(bbyEYES_full, (offsetCalc, offsetCalc_y), bbyEYES_full)
    final.paste(bbyMOUTH_full, (offsetCalc, offsetCalc_y), bbyMOUTH_full)

    return final

@app.route("/baby.png")
def serve_baby():
    img = buildBabySprite()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.route("/set", methods=["POST"])
def set_state():
    updates = request.json
    for key in updates:
        if key in babyState:
            babyState[key] = updates[key]
    return {"status": "ok", "updated": updates}

@app.route("/say", methods=["POST"])
def baby_say():
    babyState["speech"] = request.json.get("text", "")
    return {"status": "ok"}

@app.route("/speech.txt")
def speech_text():
    return babyState.get("speech", ""), 200, {"Content-Type": "text/plain"}

@app.route("/baby.html")
def baby_html():
    return f"""
    <html>
    <head>
        <style>
            html, body {{margin: 0; padding: 0; background: transparent;}}
            img {{width: 400px; height: 400px; image-rendering: pixelated;}}
        </style>
        <script>
            function refresh() {{
                const img = document.getElementById("baby");
                img.src = "/baby.png?t=" + new Date().getTime();
            }}
            setInterval(refresh, 150);  // refresh every 150ms
        </script>
    </head>
    <body>
        <img id="baby" src="/baby.png" />
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=420)
