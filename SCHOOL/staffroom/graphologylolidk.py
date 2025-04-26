import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mplcursors
import numpy as np

trainingLogPath_A = "/Users/charis/Documents/GitHub/Shkaira/SCHOOL/statistics/LOGS/training/trainingLog_100_withTotalSteps.txt"

# Load and parse
with open(trainingLogPath_A, "r") as f:
    log_text = f.read()

log_blocks = log_text.split("\n")

entries = []

for block in log_blocks:
    row = {}
    
    step_match = re.search(r"Total Steps:\s*(\d+)", block)
    if step_match:
        row['total_steps'] = int(step_match.group(1))
    else:
        number_match = re.search(r"^\d+\s+\|\s+\d+", block)
        if number_match:
            number = re.findall(r"\d+", block)
            if len(number) >= 2:
                row['total_steps'] = int(number[1])

    fields = {
        'loss': r"loss:([0-9\.]+)",
        'lr': r"LR:? ?([0-9\.]+)",
        'sampledTokens': r"sampledTokens:([0-9\.]+)",
        'scheduledSamplingRate': r"scheduledSamplingRate:([0-9\.]+)",
        'repetitionPenalty': r"repetitionPenalty:([0-9\.]+)",
        'temperature': r"temperature:([0-9\.]+)",
        'memoryLength': r"memoryLength:([0-9\.]+)",
        'repetitionWindow': r"repetitionWindow:([0-9\.]+)",
    }
    for key, regex in fields.items():
        match = re.search(regex, block)
        if match:
            row[key] = float(match.group(1))
    
    weight_matches = re.findall(r"(W\d+):(-?\d+\.\d+)", block)
    for w_name, w_val in weight_matches:
        row[w_name] = float(w_val)

    if 'total_steps' in row:
        entries.append(row)

df = pd.DataFrame(entries).set_index("total_steps").sort_index()

# Remove window weights with too few unique values (flat or useless)
df = df.drop(columns=[col for col in df.columns if col.startswith('W') and df[col].nunique() <= 3])

# Clip outliers at 1st/99th percentile
lower = df.quantile(0.01)
upper = df.quantile(0.99)
df_clipped = df.clip(lower = lower, upper = upper, axis = 1)

# Fancy scaling: 0 is neutral, pos → [0, 1], neg → [0, -1]
df_scaled = pd.DataFrame(index = df_clipped.index)

for col in df_clipped.columns:
    positive = df_clipped[col].where(df_clipped[col] > 0)
    negative = df_clipped[col].where(df_clipped[col] < 0)
    scaled_col = pd.Series(index = df_clipped.index, dtype = float)

    if positive.notna().any():
        scaled_col.update(positive / positive.max())
    if negative.notna().any():
        scaled_col.update(negative / (-negative.min()))

    df_scaled[col] = scaled_col

# Split groups
metric_cols = [col for col in df_scaled.columns if not col.startswith('W')]
weight_cols = [col for col in df_scaled.columns if col.startswith('W')]

# Colors
metric_colors = cm.get_cmap('viridis', len(metric_cols))
weight_colors = cm.get_cmap('autumn', len(weight_cols))

# Plot
fig, ax1 = plt.subplots(figsize=(22, 12))
lines = []

# Plot metrics (solid, coloured)
for i, col in enumerate(metric_cols):
    color = metric_colors(i)
    line, = ax1.plot(df_scaled.index, df_scaled[col], label = f"[METRIC] {col}", color = color, linewidth = 2)
    lines.append(line)

ax1.set_ylabel('Scaled Metrics', fontsize = 14)
ax1.set_xlabel('Total Steps', fontsize = 14)
ax1.grid(True)

# Weights (dashed, warm)
ax2 = ax1.twinx()
for i, col in enumerate(weight_cols):
    color = weight_colors(i)
    line, = ax2.plot(df_scaled.index, df_scaled[col], label = f"[WEIGHT] {col}", color = color, linestyle='--', linewidth = 2)
    lines.append(line)

ax2.set_ylabel('Scaled Window Weights', fontsize = 14)
plt.title("Training Metrics + Window Weights (Cleaned & Beautiful)", fontsize = 20)

# Smart legend (grouped + readable)
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, title="Legend", bbox_to_anchor=(1.20, 1), loc="upper left", fontsize = 10)

plt.tight_layout()

# Hover still works
cursor = mplcursors.cursor(lines, hover = True)
cursor.connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))

plt.show()
