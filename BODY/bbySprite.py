
from flask import Flask, send_file, request
from PIL import Image
import io
import threading
import time
import math
import random
import json
import os

app = Flask(__name__)

bbyBODY = Image.open("bbyBODY.png").convert("RGBA")
bbyCHEEKS = Image.open("bbyCHEEKS.png").convert("RGBA")
bbyEYES = Image.open("bbyEYES.png").convert("RGBA")
bbyMOUTH = Image.open("bbyMOUTH.png").convert("RGBA")

spriteSize = (64, 64)
bbyBODY_numLayers = 5

canvasSize = (80, 80)
offsetCalc = round((80-64) / 2)

babyStateFilePath = "babyState.json"

# normally be updated live by BabyLLM's training process
babyState = {
    "eyes": 2,
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
    "cerebralLoad": 0,
    "dreamIntensity": 0,
    "memoryFlux": 0, 
    "learningStability": 0,
    "metabolicRate": 0,
}

baseColour = {"R": 133, "G": 239, "B": 238}
currentColour = {"R": 133, "G": 239, "B": 238}
targetColour = {"R": 133, "G": 239, "B": 238}

def state_reader_loop():
    while True:
        if os.path.exists(babyStateFilePath):
            try:
                with open(babyStateFilePath, 'r') as f:
                    babyState.update(json.load(f)) 

                    for ch in ["R", "G", "B"]:
                        if ch in babyState:
                            targetColour[ch] = babyState[ch]

                    if "tintStrength" not in babyState:
                        babyState["tintStrength"] = 1.0 

                    dreamIntensity = babyState.get("dreamIntensity", 0.0)
                    bpm = 126
                    bpm32th = 60 / (bpm * 8)
                    metabolicRate = round(dreamIntensity) * bpm32th
                    babyState["metabolicRate"] = metabolicRate

            except (json.JSONDecodeError, IOError):
                pass
        time.sleep(0.1)

threading.Thread(target=state_reader_loop, daemon=True).start()

def blink_loop():
    while True:
        time.sleep(1 + (time.time() % 6))
        metabolicRate = babyState.get("metabolicRate", 0.0)

        metabolicRate = random.choice([0.119, 0.119, 0.119, 0.119, 0.119, 0.119, 0.119, 0.119, 0.238, 0.238, 0.472])
        original_eyes = babyState["eyes"]  # save current emotion
        babyState["eyes"] = 0  # blink column
        print(f"*blimk*")
        time.sleep(metabolicRate)
        
        babyState["eyes"] = original_eyes  # restore emotion
        
        if random.random() < 0.05:
            time.sleep(metabolicRate)
            babyState["eyes"] = 0  # blink column
            print(f"**blimk**")
            time.sleep(metabolicRate)
            
            babyState["eyes"] = original_eyes  # restore emotion
            
            if random.random() < 0.05:
                time.sleep(metabolicRate)
                babyState["eyes"] = 0  # blink column
                print(f"***blimk***")
                time.sleep(metabolicRate)
                
                babyState["eyes"] = original_eyes  # restore emotion
                time.sleep(metabolicRate)

threading.Thread(target=blink_loop, daemon=True).start()

def pulse_loop_random():
    while True:
        # This could be replaced later with a real layer-based signal
        #babyState["pulse"] = 0.5 + 0.5 * math.sin(time.time() * 4)  # smooth breathing
        keyChoice = random.choice(["stretch_left", "stretch_right", "stretch_up", "stretch_down", "squish_left", "squish_right", "squish_up", "squish_down", "cheeks_on", "tears_on"])
        babyState[keyChoice] = random.choice([True, False])
        #time.sleep(0.472)
        time.sleep(0.236)

def pulse_loop():
    while True:
        cerebralLoad = babyState.get("cerebralLoad", 0.0)
        learningStability = babyState.get("learningStability", 0.0)
        memoryFlux = babyState.get("memoryFlux", 0.0)
        metabolicRate = babyState.get("metabolicRate", 0.0)
        
        # --- body language ---
        stimChoice = random.choice(["random", "tense", "dreamy"])
        
        if stimChoice == "tense":
            # cerebral load -> tense (tall/thin) vs. relaxed (short/wide)
            tense_threshold = random.uniform(0.1, 1.5)
            is_tense = cerebralLoad > tense_threshold
            tenseKey = random.choice(["stretch_up", "squish_left", "squish_right"])
            babyState[tenseKey] = is_tense
            relaxedKey = random.choice(["stretch_down", "stretch_left", "stretch_right"])
            babyState[relaxedKey] = not is_tense

        elif stimChoice == "dreamy":
            # dream intensity -> dreamy (tall) vs. grounded (short)
            dreamy_threshold = random.uniform(0.4, 0.6)
            is_dreamy = learningStability > dreamy_threshold
            babyState["stretch_up"] = is_dreamy
            groundedKey = random.choice(["stretch_down", "squish_up"])
            babyState[groundedKey] = not is_dreamy

        elif stimChoice == "flux":
            flux_threshold = random.uniform(0.4, 0.6)
            is_flux = memoryFlux > flux_threshold
            babyState["squish_down"] = is_flux

        elif stimChoice == "random":
            keyChoice = random.choice(["stretch_left", "stretch_right", "stretch_up", "stretch_down", "squish_left", "squish_right", "squish_up", "squish_down", "cheeks_on", "tears_on"])
            babyState[keyChoice] = random.choice([True, False])
            
        time.sleep(metabolicRate)

threading.Thread(target=pulse_loop, daemon=True).start()

def jump_reset():
    metabolism = babyState.get("metabolicRate", 0.0)
    metabolicRate = metabolism * 5
    while True:
        if babyState.get("jumping"):
            time.sleep(metabolicRate)
            babyState["jumping"] = False
            if random.random() < 0.2:
                time.sleep(metabolicRate)
                babyState["jumping"] = True
                time.sleep(metabolicRate)
                babyState["jumping"] = False
                if random.random() < 0.15:
                    time.sleep(metabolicRate)
                    babyState["jumping"] = True
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

    blend_speed = 0.0002  # smaller = slower
    return_speed = 0.00002

    for channel in ["R", "G", "B"]:
        delta = targetColour[channel] - currentColour[channel]
        currentColour[channel] += delta * blend_speed
        delta = baseColour[channel] - currentColour[channel]
        currentColour[channel] += delta * return_speed

    tintStrength = babyState.get("tintStrength", 1.0)

    # Greyscale body first, then tint toward currentColor
    pixels = bbyBODY_full.load()
    for x in range(bbyBODY_full.width):
        for y in range(bbyBODY_full.height):
            pr, pg, pb, pa = pixels[x, y]
            gray = int((pr + pg + pb) / 3)

            # What the tint *would be* at full strength
            rTint = int(gray * currentColour["R"] / 255)
            gTint = int(gray * currentColour["G"] / 255)
            bTint = int(gray * currentColour["B"] / 255)

            # Blend original color toward the tint, based on tintStrength
            outR = int((1 - tintStrength) * pr + tintStrength * rTint)
            outG = int((1 - tintStrength) * pg + tintStrength * gTint)
            outB = int((1 - tintStrength) * pb + tintStrength * bTint)

            pixels[x, y] = (outR, outG, outB, pa)

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
    babyState["speech"] = request.json.get("speech", "")
    return {"status": "ok"}

@app.route("/speech.txt")
def speech_text():
    return babyState.get("speech", ""), 200, {"Content-Type": "text/plain"}

@app.route("/baby.html")
def baby_html():
    return f"""
    <html>
    <head>
        <link href="https://fonts.cdnfonts.com/css/silkscreen" rel="stylesheet">
        <style>
            html, body {{
                margin: 0px;
                padding: 0px;
                background: transparent;
                overflow: hidden;
                font-family: 'Silkscreen', monospace;
            }}
            #wrap {{
                position: relative;
                width: 550px;
                height: 400px;
            }}
            #baby {{
                position: absolute;
                left: 450px;
                bottom: 0px;
                width: 400px;
                height: 400px;
                image-rendering: pixelated;
            }}
            #speechBubble {{
                position: absolute;
                left: 10px;
                bottom: 230px;
                width: 540px;
                height: 150px;
                background: rgba(133, 239, 238, 0.2);
                border: 2px solid rgba(0, 85, 170, 0.8);
                border-radius: 10px;
                box-shadow: 3px 3px black;
                font-size: 36px;
                color: white;
                font-family: 'Silkscreen', monospace;
                padding: 8px 10px;
                overflow: hidden;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
                opacity: 0;
                transition: opacity 1s ease;
                pointer-events: none;
                box-sizing: border-box;
            }}
            #speechBubble.visible {{
                opacity: 1;
            }}
            #speechContent {{
                overflow: hidden;
                flex-grow: 1;
                max-height: 100%;
                width: 100%;
                white-space: pre-wrap;
                line-height: 1.2em;
                display: flex;
                flex-direction: column-reverse;
            }}
        </style>
        <script>
            let lastSpeech = "";
            let currentIndex = 0;
            let displayTimer = null;
            let fadeTimer = null;

            function typeOut(text, targetEl) {{
                clearTimeout(displayTimer);
                clearTimeout(fadeTimer);
                targetEl.innerHTML = "";
                currentIndex = 0;
                targetEl.parentElement.classList.add("visible");

                function typeChar() {{
                    if (currentIndex < text.length) {{
                        targetEl.innerHTML += text[currentIndex];
                        currentIndex++;
                        displayTimer = setTimeout(typeChar, 75);
                    }}
                }}
                typeChar();

                fadeTimer = setTimeout(() => {{
                    targetEl.parentElement.classList.remove("visible");
                }}, 20000);
            }}

            function refresh() {{
                const img = document.getElementById("baby");
                img.src = "/baby.png?t=" + new Date().getTime();

                fetch("/speech.txt")
                    .then(res => res.text())
                    .then(text => {{
                        const contentDiv = document.getElementById("speechContent");
                        if (text.trim() !== lastSpeech.trim()) {{
                            lastSpeech = text.trim();
                            typeOut(lastSpeech, contentDiv);
                        }}
                    }});
            }}

            setInterval(refresh, 150);
        </script>
    </head>
    <body>
        <div id="wrap">
            <img id="baby" src="/baby.png" />
            <div id="speechBubble">
                <div id="speechContent"></div>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=420)