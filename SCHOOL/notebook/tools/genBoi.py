import random

def makeSafeBoi():
    # ʕ"❀"ෆ.ෆʔっ❀ or ⊂ʕʘ‿ʘ"❀"ʔ
    hairThingys = ["❀", "♥", "𖡼", "♡"]

    # << -- "ς"ʕʘ‿ʘςʔ
    leftArmsOut = ["૮", "ς", "⊂"]

    # |-- "ʕ"っʘ‿ʘʔっ
    leftSides = ["ʕ"]

    # >> -- ʕ"っ"☯‿☯ʔっ
    leftArmsIn = ["ฅ", "⊃",] #"ゝ"
    ### leftArmsIn += hairThingys # what the hell, she said!

    # ʕっ"ʘ"‿"ʘ"ʔっ
    eyes = ["⚈", "◉", "•", "o", "ʘ", "ₓ", "ʘ", "•̀", "•́", 
            "⋆", "✰", "♡", "‿", "ෆ", "ᵔ", "-", "☯", "⊗", "☉",]

    # ʕっʘ"‿"ʘʔっ
    mouthes = ["‿", "ᴥ", ".", "o",]

    # -- << ςʕʘ‿ʘ"ς"ʔ
    rightArmsIn = ["ฅ", "૮", "⊂",]

    # --| ʕっʘ‿ʘ"ʔ"っ
    rightSides = ["ʔ"]

    # -- >> ʕっ☯‿☯ʔ"っ"
    rightArmsOut = ["⊃",] #"ゝ"

    safeBoi = random.choice(leftSides) # 'ʕ'

    pointLeft = random.choice([True, False])
    if pointLeft: safeBoi = random.choice(leftArmsOut) + safeBoi # '⊂'

    leftHair = random.choice([True, False])
    if leftHair is True: safeBoi += random.choice(hairThingys) # 'ʕ❀'/'⊂ʕ❀'/' ⊂ʕ❀'/'❀⊂ʕ❀'/'❀ ⊂ʕ❀'
    if pointLeft is False and leftHair is False:
        hugLeft = random.choice([True, False])
        if hugLeft: safeBoi += random.choice(leftArmsIn) # 'ʕ⊃'/'❀ʕ⊃'/'❀ ʕ⊃'

    #safeBoi = partA, partB, partC
    if len(safeBoi) < 3:
        safeBoi = " " * (3 - len(safeBoi)) + safeBoi

    mouth = random.choice(mouthes)
    eye = random.choice(eyes)
    unmatchedEyes = random.choice([True, False])
    safeBoi += eye + mouth
    if unmatchedEyes: eye = random.choice(eyes)
    safeBoi += eye

    rightBoi = ""
    hugRight = random.choice([True, False])
    if hugRight: rightBoi += random.choice(rightArmsIn)
    rightHair = random.choice([True, False])
    if rightHair and hugRight is False and leftHair is False: rightBoi += random.choice(hairThingys)
    rightBoi += random.choice(rightSides)

    if hugRight is False:
        pointRight = random.choice([True, False])
        if pointRight: rightBoi += random.choice(rightArmsOut)
        else: rightBoi += " "
    
    if len(rightBoi) < 3:
        rightBoi += (" " * (3 - len(rightBoi)))

    safeBoi += rightBoi

    return safeBoi

def makeDatBoi():
    # ʕ"❀"ෆ.ෆʔっ❀ or ⊂ʕʘ‿ʘ"❀"ʔ
    hairThingys = ["❀", "♥", "𖡼", "❄︎", "♡"]

    # ʕっʘ.ʘʔっ"❄"
    thingys = hairThingys
    thingys += ["︵", "✎", "♥",] #"𓂸", "𓆟", "𓆞", "✰✰⋆⋆", "𖤣", ]

    # << -- "ς"ʕʘ‿ʘςʔ
    leftArmsOut = ["૮", "ς", "⊂"]

    # |-- "ʕ"っʘ‿ʘʔっ
    leftSides = ["ʕ"]

    # >> -- ʕ"っ"☯‿☯ʔっ
    leftArmsIn = ["ฅ", "っ", "ノ", "⊃",] #"ゝ"
    ### leftArmsIn += hairThingys # what the hell, she said!

    # ʕっ"ʘ"‿"ʘ"ʔっ
    eyes = ["⚈", "◉", "•", "o", "ʘ", "ₓ", "ʘ", "•̀", "•́", "꩜ ", 
            "⋆", "✰", "♡", "‿", "ෆ", "ᵔ", "-", "☯", "⊗", "˙",
            "☉",]

    # ʕっʘ"‿"ʘʔっ
    mouthes = ["‿", "︿", "ω", "ᴥ", ".", "o", "ᗜ"]

    # -- << ςʕʘ‿ʘ"ς"ʔ
    rightArmsIn = ["ฅ", "૮", "ς", "⊂"]

    # --| ʕっʘ‿ʘ"ʔ"っ
    rightSides = ["ʔ"]

    # -- >> ʕっ☯‿☯ʔ"っ"
    rightArmsOut = ["っ", "ノ", "⊃",] #"ゝ"

    datBoi = random.choice(leftSides) # 'ʕ'

    pointLeft = random.choice([True, False])
    if pointLeft: datBoi = random.choice(leftArmsOut) + datBoi # '⊂'
    #throwLeft = random.choice([True, False])
    #if throwLeft: datBoi = " " + datBoi #' ⊂'
    thingyLeft = random.choice([True, False])
    if thingyLeft: datBoi = random.choice(thingys) + datBoi # '❀⊂' / '❀ ⊂'

    # 'ʕ'/'⊂ʕ'/' ⊂ʕ'/'❀⊂ʕ'/'❀ ⊂ʕ'/'❀ʕ'/'❀ ʕ'

    leftHair = random.choice([True, False])
    if leftHair and thingyLeft is False: datBoi += random.choice(hairThingys) # 'ʕ❀'/'⊂ʕ❀'/' ⊂ʕ❀'/'❀⊂ʕ❀'/'❀ ⊂ʕ❀'
    if pointLeft is False and leftHair is False:
        hugLeft = random.choice([True, False])
        if hugLeft: datBoi += random.choice(leftArmsIn) # 'ʕ⊃'/'❀ʕ⊃'/'❀ ʕ⊃'

    mouth = random.choice(mouthes)
    eye = random.choice(eyes)
    unmatchedEyes = random.choice([True, False])
    datBoi += eye + mouth
    if unmatchedEyes: eye = random.choice(eyes)
    datBoi += eye

    hugRight = random.choice([True, False])
    if hugRight: datBoi += random.choice(rightArmsIn)
    rightHair = random.choice([True, False])
    if rightHair and hugRight is False and leftHair is False and thingyLeft is False: datBoi += random.choice(hairThingys)
    datBoi += random.choice(rightSides)

    if hugRight is False:
        pointRight = random.choice([True, False])
        if pointRight: datBoi += random.choice(rightArmsOut)

    #throwRight = random.choice([True, False])
    #if throwRight: datBoi += " "
    thingyRight = random.choice([True, False])
    if thingyRight and rightHair is False and leftHair is False and thingyLeft is False: datBoi += random.choice(thingys)

    return datBoi

if __name__ == "__main__":
    for i in range(20):
        face = makeSafeBoi()
        print(f"{face} (len={len(face)})")
