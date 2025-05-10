import json
import os

# === CONFIG ===
old_tokenToIndex_path = "BRAIN/vocabCache/2000_20/vocab2000_20_to_index.json"
new_tokenToIndex_path = "BRAIN/vocabCache/vocab4000_20_to_index.json"
new_indexToToken_path = "BRAIN/vocabCache/vocab4000_20_to_token.json"
new_vocabList_path   = "BRAIN/vocabCache/vocab4000_20_list.json"

out_tokenToIndex = "BRAIN/vocabCache/vocabFIXED_to_index.json"
out_indexToToken = "BRAIN/vocabCache/vocabFIXED_to_token.json"
out_vocabList    = "BRAIN/vocabCache/vocabFIXED_list.json"

# === LOAD ===
with open(old_tokenToIndex_path, "r", encoding="utf-8") as f:
    old_tokenToIndex = json.load(f)

with open(new_tokenToIndex_path, "r", encoding="utf-8") as f:
    new_tokenToIndex = json.load(f)

with open(new_indexToToken_path, "r", encoding="utf-8") as f:
    new_indexToToken = {int(k): v for k, v in json.load(f).items()}

with open(new_vocabList_path, "r", encoding="utf-8") as f:
    new_vocabList = json.load(f)

# BUILD REMAPPED STRUCTURES
fixed_tokenToIndex = {}
fixed_indexToToken = {}
fixed_vocabList = []

used_indices = set()

# Copy old tokens into their original positions
for token, old_idx in old_tokenToIndex.items():
    old_idx = int(old_idx)
    fixed_tokenToIndex[token] = old_idx
    fixed_indexToToken[old_idx] = token
    used_indices.add(old_idx)

# Fill fixed_vocabList to correct length
max_vocab_len = max(len(new_vocabList), max(used_indices) + 1)
fixed_vocabList = ["<UNK>"] * max_vocab_len
for idx, token in fixed_indexToToken.items():
    fixed_vocabList[idx] = token

# Append new tokens until cap is hit
next_free_index = max(used_indices) + 1
max_vocab_size = 4200

for token in new_tokenToIndex:
    if token in fixed_tokenToIndex:
        continue  # already present

    if next_free_index >= max_vocab_size:
        print(f"skipping token '{token}' — vocab size cap reached")
        continue

    # Assign token to next free slot
    while next_free_index in used_indices:
        next_free_index += 1
        if next_free_index >= max_vocab_size:
            break

    if next_free_index >= max_vocab_size:
        break

    fixed_tokenToIndex[token] = next_free_index
    fixed_indexToToken[next_free_index] = token
    if next_free_index >= len(fixed_vocabList):
        fixed_vocabList.append(token)
    else:
        fixed_vocabList[next_free_index] = token
    used_indices.add(next_free_index)
    next_free_index += 1

# === SAVE FIXED FILES ===
os.makedirs("BRAIN/vocabCache", exist_ok=True)

with open(out_tokenToIndex, "w", encoding="utf-8") as f:
    json.dump(fixed_tokenToIndex, f, indent=2)

with open(out_indexToToken, "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in fixed_indexToToken.items()}, f, indent=2)

with open(out_vocabList, "w", encoding="utf-8") as f:
    json.dump(fixed_vocabList, f, indent=2)

print("vocab mapping preserved and new tokens appended!")
