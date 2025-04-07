import re
import pandas as pd
import matplotlib.pyplot as plt

trainingLogPath_100 = "/Users/charis/Documents/GitHub/Shkaira/LOGS/training/trainingLog_100.txt"

# Load your log
with open(trainingLogPath_100, "r") as f:
    log_text = f.read()

# Extract timestamp and all window weights (W1, W2, ..., W21)
pattern = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"(W\d+[:\s]-?\d+\.\d+(?:,\s*W\d+[:\s]-?\d+\.\d+)*)",
    re.DOTALL
)

# Parse and store data
entries = []
for match in re.finditer(pattern, log_text):
    timestamp = pd.to_datetime(match.group("timestamp"))
    weights = dict(re.findall(r"(W\d+)[\s:]+(-?\d+\.\d+)", match.group(0)))
    row = {"timestamp": timestamp}
    row.update({k: float(v) for k, v in weights.items()})
    entries.append(row)

# Create DataFrame
df = pd.DataFrame(entries).set_index("timestamp").sort_index().fillna(0)

# Plot as line graph
df.plot(figsize=(14, 6))
plt.title("Window Weights Over Time")
plt.ylabel("Weight")
plt.xlabel("Timestamp")
plt.xticks(rotation=45)
plt.ylim(-0.1, 0.3)  # Adjust if needed
plt.legend(title="Window", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
