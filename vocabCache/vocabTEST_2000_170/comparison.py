import json

with open('tokenizer.json', 'r') as f1, open('tokenizerNEW.json', 'r') as f2:
    tokenizer_old = json.load(f1)
    tokenizer_new = json.load(f2)

vocab_old = tokenizer_old['model']['vocab']
vocab_new = tokenizer_new['model']['vocab']

overlap_map = {}
for token_str, token_id_old in vocab_old.items():
    if token_str in vocab_new:
        token_id_new = vocab_new[token_str]
        overlap_map[token_str] = {'old_id': token_id_old, 'new_id': token_id_new}

print(f"Overlapping tokens: {len(overlap_map)}")

# Load the tokenizerNEW.json again
with open('tokenizerNEW.json', 'r') as f_new:
    tokenizer_new_data = json.load(f_new)

# Create a copy of the vocab to modify
vocab_new_modified = tokenizer_new_data['model']['vocab'].copy()

# Iterate through the overlap map and re-index in the new vocab
for token_str, ids in overlap_map.items():
    old_id = ids['old_id']
    new_id = ids['new_id']

    # Assign the OLD tokenizer's ID to the token in the NEW tokenizer's vocab
    vocab_new_modified[token_str] = old_id

# Update the vocab in the tokenizer_new_data copy
tokenizer_new_data['model']['vocab'] = vocab_new_modified

# Save the modified tokenizer to a *NEW* file (e.g., tokenizerNEW_REINDEXED.json)
output_file = 'tokenizerNEW_REINDEXED.json'
with open(output_file, 'w') as f_output:
    json.dump(tokenizer_new_data, f_output, indent=2)

print(f"Modified tokenizer saved to: {output_file}")
print("Important: Verify the tokenizer before using it for training!")