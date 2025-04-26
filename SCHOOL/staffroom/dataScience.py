import re
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors

trainingLogPath_A = "/Users/charis/Documents/GitHub/Shkaira/SCHOOL/statistics/LOGS/training/trainingLog_100_withTotalSteps.txt"

# Load log
with open(trainingLogPath_A, "r") as f:
    log_text = f.read()

# Pattern to extract total steps + all Wxx weights
pattern = re.compile(
    r"(Total Steps:\s*(?P<total_steps>\d+)).*?"
    r"([lL]oss:? ?(?P<loss>[0-9\.]+)).*?"
    r"([lL][Rr]:? ?(?P<lr>[0-9\.]+)).*?"
    r"(scheduledSamplingRate:? ?(?P<scheduledSamplingRate>[0-9\.]+)).*?"
    r"(repetitionPenalty:? ?(?P<repetitionPenalty>[0-9\.]+)).*?"
    r"(temperature:? ?(?P<temperature>[0-9\.]+)).*?"
    r"(sampledTokens:? ?(?P<sampledTokens>[0-9\.]+)).*?"
    r"(W\d+[:\s]-?\d+\.\d+(?: \([0-9\.]+\))?(?:,?\s*W\d+[:\s]-?\d+\.\d+(?: \([0-9\.]+\))?)*)",
    re.DOTALL)

# Extract data
entries = []
for match in re.finditer(pattern, log_text):
    loss = float(match.group('loss'))*0.1
    lr = float(match.group('lr'))*25000
    scheduledSamplingRate = float(match.group('scheduledSamplingRate'))*5
    repetitionPenalty = float(match.group('repetitionPenalty'))*5
    temperature = float(match.group('temperature'))*5
    sampledTokens = float(match.group('sampledTokens'))*0.001
    #total_steps = match.group('total_steps')
    step_match = re.search(r"Total Steps:\s*(\d+)", match.group(0))
    if not step_match:
        continue
    total_steps = int(step_match.group(1))
    weights = dict(re.findall(r"(W\d+)[\s:]+(-?\d+\.\d+)", match.group(0)))
    row = {"total_steps": total_steps, "loss": loss, "lr": lr, "scheduledSamplingRate": scheduledSamplingRate, "repetitionPenalty": repetitionPenalty, "temperature": temperature, "sampledTokens": sampledTokens,}
    print(row)
    row.update({k: float(v) for k, v in weights.items()})
    entries.append(row)

# Create DataFrame indexed by total_steps
df = pd.DataFrame(entries).set_index("total_steps").sort_index()

# Plot
ax = df.plot(figsize=(14, 6))
plt.title("Window Weights Over Total Training Steps")
plt.ylabel("Weight")
plt.xlabel("Total Steps")
plt.legend(title="Window", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

cursor = mplcursors.cursor(ax.lines, hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))

plt.show()


# add the loss to the graph

# fix window data from old logs to say the new thing for windows