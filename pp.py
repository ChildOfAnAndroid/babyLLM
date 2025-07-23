import re
import json
from datasets import load_dataset

# Define gentle-ish words
KEYWORDS = [
r"\bweed", r"\bmusic"
]
regex = re.compile("|".join(KEYWORDS), re.IGNORECASE)

def extract_strings(obj):
    """Recursively extracts all strings from a nested structure."""
    if isinstance(obj, str):
        return [obj]
    elif isinstance(obj, dict):
        return sum([extract_strings(v) for v in obj.values()], [])
    elif isinstance(obj, list):
        return sum([extract_strings(item) for item in obj], [])
    return []

def is_gentle_reasoning(example):
    text_blob = " ".join(extract_strings(example)).lower()
    return bool(regex.search(text_blob))

# Load a sample of the dataset
dataset = load_dataset("open-thoughts/OpenThoughts-114k", split="train[:10%]")

# Filter using fuzzy regex
filtered = [ex for ex in dataset if is_gentle_reasoning(ex)]

print(f"Filtered {len(filtered)} examples out of {len(dataset)}")

# Optional: Save to disk
with open("gentle_examples.jsonl", "w") as f:
    for ex in filtered:
        f.write(json.dumps(ex) + "\n")
