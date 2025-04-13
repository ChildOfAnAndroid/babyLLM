# CHARIS CAT 2025 --- SCHOOL/staffroom/languageAndLiterature.py

# --- Language and Literature --- 
#from BRAIN.LAYERS.vocab import VOCAB

def genTrainingDataFromTokens(tokens, windowMAX, numTarget=1, startIndex=0):
    dataPairs = []
    endIndex = len(tokens) - windowMAX
    for i in range(startIndex, endIndex):
        inputSeq = tokens[i:i+windowMAX]
        targetSeq = tokens[i+windowMAX : i+windowMAX+numTarget]
        if len(targetSeq) < numTarget: continue
        dataPairs.append((inputSeq, targetSeq))
    return dataPairs