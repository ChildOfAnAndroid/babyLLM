from collections import deque
import sys
from S_output import *
from config import *
import shutil  

def drawWindowEQ(values, maxHeight=3, labels=None, styleFn=None, spacing=1):
    bars = []
    for i, v in enumerate(values):
        h = int(round(v * maxHeight))
        column = []
        for row in range(maxHeight):
            char = '█' if maxHeight - row <= h else ' '
            if styleFn:
                char = styleFn(v, i, char)
            column.append(char)
        bars.append(column)

    lines = []
    for row in range(maxHeight):
        line = ""
        for col in bars:
            line += " " * spacing + col[row]
        lines.append(line)

    baseLine = " " + "–" * ((spacing + 1) * len(values))
    labelLine = " " + " ".join(f"W{i+1}" for i in range(len(values)))
    return lines + [baseLine, labelLine]

def windowBarStyleFn(value, index, char):
    if char == ' ':
        return ' '
    S_type = S_getStat("windowWeights", value)
    return S_apply(S_type, char)

class rainbowHUD:
    def __init__(self, maxArms=20):
        self.maxArms = maxArms
        self.history = deque(maxlen=maxArms)

    def addArm(self, arm):
        if len(arm) != 3:
            raise ValueError("Each guess arm must have exactly 3 tokens.")
        self.history.append(arm)

    def drawRainbowHistory(self, spacing=1):
        rows = ["", "", ""]
        for arm in self.history:
            for i in range(3):
                rows[i] += " " * spacing + arm[i]
        return rows

def printHUD(windowWeights, guessHUD):
    eq_lines = drawWindowEQ(windowWeights, maxHeight=3, styleFn=windowBarStyleFn)
    rainbow_lines = guessHUD.drawRainbowHistory(spacing=1)

    fused_lines = []
    for i in range(3):
        rainbow = rainbow_lines[i] if i < len(rainbow_lines) else ""
        eq = eq_lines[i] if i < len(eq_lines) else ""
        fused_lines.append(rainbow + "   " + eq)

    fused_lines.append(eq_lines[3])  # baseline
    fused_lines.append(eq_lines[4])  # labels

    print("\n".join(fused_lines))
