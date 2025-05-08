import json

# Load tokenizer
with open("tokenizer_4200.json", "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

# Replace vocab
with open("vocabFIXED_to_index.json", "r", encoding="utf-8") as f:
    new_vocab = json.load(f)  # This should be a {token: index} dict

tokenizer_data["model"]["vocab"] = new_vocab

# (Optional) You could patch merges too if needed like this:
# with open("fixed_merges.txt", "r", encoding="utf-8") as f:
#     merges = [line.strip() for line in f if line.strip()]
# tokenizer_data["model"]["merges"] = merges

# Save tokenizer
with open("tokenizer_4200_FIXED.json", "w", encoding="utf-8") as f:
    json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)

print("Tokenizer patched and saved as tokenizer_4200_FIXED.json.")
