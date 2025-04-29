import random

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
    leftArmsIn = ["ฅ", "っ", "ノ", "⊃", "ゝ"]
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
    rightArmsOut = ["っ", "ノ", "⊃", "ゝ"]

    datBoi = random.choice(leftSides) # 'ʕ'

    pointLeft = random.choice([True, False])
    if pointLeft: datBoi = random.choice(leftArmsOut) + datBoi # '⊂'
    throwLeft = random.choice([True, False])
    if throwLeft: datBoi = " " + datBoi #' ⊂'
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

    throwRight = random.choice([True, False])
    if throwRight: datBoi += " "
    thingyRight = random.choice([True, False])
    if thingyRight and rightHair is False and leftHair is False and thingyLeft is False: datBoi += random.choice(thingys)

    return datBoi

if __name__ == "__main__":
    face = makeDatBoi()
    print(face)

    face = makeDatBoi()
    print(face)

    face = makeDatBoi()
    print(face)

    face = makeDatBoi()
    print(face)

    face = makeDatBoi()
    print(face)

    face = makeDatBoi()
    print(face)

    face = makeDatBoi()
    print(face)

    face = makeDatBoi()
    print(face)

    face = makeDatBoi()
    print(face)

    face = makeDatBoi()
    print(face)

    face = makeDatBoi()
    print(face)

    face = makeDatBoi()
    print(face)

    face = makeDatBoi()
    print(face)

    face = makeDatBoi()
    print(face)

    face = makeDatBoi()
    print(face)

    face = makeDatBoi()
    print(face)

    face = makeDatBoi()
    print(face)

    face = makeDatBoi()
    print(face)
