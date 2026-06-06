# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM // SHKAIRA/notebook/tools/genBoi.py
# v1.1

import random


def makeSafeBoi():
    # ʕ"❀"ෆ.ෆʔっ❀ or ⊂ʕʘ‿ʘ"❀"ʔ
    hairThingys = ["❀", "♥", "𖡼", "♡"]

    # << -- "ς"ʕʘ‿ʘςʔ
    leftArmsOut = ["૮", "ς", "⊂"]

    # |-- "ʕ"っʘ‿ʘʔっ
    leftSides = ["ʕ"]

    # >> -- ʕ"っ"☯‿☯ʔっ
    leftArmsIn = [
        "ฅ",
        "⊃",
    ]  # "ゝ"
    ### leftArmsIn += hairThingys # what the hell, she said!

    # ʕっ"ʘ"‿"ʘ"ʔっ
    eyes = [
        "⚈",
        "◉",
        "•",
        "o",
        "ʘ",
        "ₓ",
        "ʘ",
        "•̀",
        "•́",
        "⋆",
        "✰",
        "♡",
        "‿",
        "ෆ",
        "ᵔ",
        "-",
        "☯",
        "⊗",
        "☉",
    ]

    # ʕっʘ"‿"ʘʔっ
    mouthes = [
        "‿",
        "ᴥ",
        ".",
        "o",
    ]

    # -- << ςʕʘ‿ʘ"ς"ʔ
    rightArmsIn = [
        "ฅ",
        "૮",
        "⊂",
    ]

    # --| ʕっʘ‿ʘ"ʔ"っ
    rightSides = ["ʔ"]

    # -- >> ʕっ☯‿☯ʔ"っ"
    rightArmsOut = [
        "⊃",
    ]  # "ゝ"

    safeBoi = random.choice(leftSides)  # 'ʕ'

    pointLeft = random.choice([True, False])
    if pointLeft:
        safeBoi = random.choice(leftArmsOut) + safeBoi  # '⊂'

    leftHair = random.choice([True, False])
    if leftHair is True:
        safeBoi += random.choice(hairThingys)  # 'ʕ❀'/'⊂ʕ❀'/' ⊂ʕ❀'/'❀⊂ʕ❀'/'❀ ⊂ʕ❀'
    if pointLeft is False and leftHair is False:
        hugLeft = random.choice([True, False])
        if hugLeft:
            safeBoi += random.choice(leftArmsIn)  # 'ʕ⊃'/'❀ʕ⊃'/'❀ ʕ⊃'

    # safeBoi = partA, partB, partC
    if len(safeBoi) < 3:
        safeBoi = " " * (3 - len(safeBoi)) + safeBoi

    mouth = random.choice(mouthes)
    eye = random.choice(eyes)
    unmatchedEyes = random.choice([True, False])
    safeBoi += eye + mouth
    if unmatchedEyes:
        eye = random.choice(eyes)
    safeBoi += eye

    rightBoi = ""
    hugRight = random.choice([True, False])
    if hugRight:
        rightBoi += random.choice(rightArmsIn)
    rightHair = random.choice([True, False])
    if rightHair and hugRight is False and leftHair is False:
        rightBoi += random.choice(hairThingys)
    rightBoi += random.choice(rightSides)

    if hugRight is False:
        pointRight = random.choice([True, False])
        if pointRight:
            rightBoi += random.choice(rightArmsOut)
        else:
            rightBoi += " "

    if len(rightBoi) < 3:
        rightBoi += " " * (3 - len(rightBoi))

    safeBoi += rightBoi

    return safeBoi



if __name__ == "__main__":
    for i in range(20):
        face = makeSafeBoi()
        print(f"{face} (len={len(face)})")
