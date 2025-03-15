import json
from transformers import AutoTokenizer

# Load tokenizer.json and tokenizerNEW.json
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

# Load tokenizerNEW.json data
with open('tokenizerNEW.json', 'r') as f_new:
    tokenizer_new_data = json.load(f_new)

vocab_new_reindexed = {}

print("\n--- Re-indexing Overlapping Tokens ---")
for token_str, ids in overlap_map.items():
    old_id = ids['old_id']
    new_id = ids['new_id']
    vocab_new_reindexed[token_str] = old_id
    print(f"Remapping: Token='{token_str}', old_id={old_id}, new_id={ids['new_id']}, assigned_id={old_id}")

print("\n--- Adding New Tokens ---")
for token_str, new_id in vocab_new.items():
    if token_str not in overlap_map:
        vocab_new_reindexed[token_str] = new_id
        print(f"Adding new token: Token='{token_str}', new_id={new_id}")

vocab_new_reindexed_sorted = dict(sorted(vocab_new_reindexed.items()))
tokenizer_new_data['model']['vocab'] = vocab_new_reindexed_sorted

output_file = 'tokenizerNEW_REINDEXED.json'
with open(output_file, 'w') as f_output:
    json.dump(tokenizer_new_data, f_output, indent=2)

print(f"\nModified tokenizer saved to: {output_file}")

# --- Verification Immediately After Saving ---
print("\n--- Verification: Loading and Checking tokenizerNEW_REINDEXED.json from disk ---")
with open('tokenizerNEW_REINDEXED.json', 'r') as f_verify:
    tokenizer_reloaded_data = json.load(f_verify)
vocab_reloaded = tokenizer_reloaded_data['model']['vocab']

# Check a few overlapping tokens in the reloaded vocab
verify_tokens = ["!", "\"", "#", "Ġis", "Ġthe", "hello", "world", "music", "Ġmusic"]
print("Verifying IDs in reloaded tokenizer:")
for token_str in verify_tokens:
    if token_str in overlap_map and token_str in vocab_reloaded:
        old_id = overlap_map[token_str]['old_id']
        reloaded_id = vocab_reloaded[token_str]
        if old_id == reloaded_id:
            print(f"  ✅ Token='{token_str}', Old ID={old_id}, Reloaded ID={reloaded_id} - MATCH")
        else:
            print(f"  ❌ Token='{token_str}', Old ID={old_id}, Reloaded ID={reloaded_id} - MISMATCH (SHOULD MATCH)")
    else:
        print(f"  Token='{token_str}' - NOT in overlap or reloaded vocab")


print("Important: Verify the tokenizer before using it for training!")

tokenizer_original = AutoTokenizer.from_pretrained("vocab_2000", trust_remote_code=True) # CORRECTED PATH
tokenizer_reindexed = AutoTokenizer.from_pretrained("vocabTEST_2000_170", trust_remote_code=True, force_download=True) # CORRECTED PATH

sample_text = "This is a test sentence with some overlapping words."

tokens_original = tokenizer_original.tokenize(sample_text)
tokens_reindexed = tokenizer_reindexed.tokenize(sample_text)

ids_original = tokenizer_original.encode(sample_text, add_special_tokens=False)
ids_reindexed = tokenizer_reindexed.encode(sample_text, add_special_tokens=False)

print("Original Tokenizer Tokens:", tokens_original)
print("Re-indexed Tokenizer Tokens:", tokens_reindexed) # SHOULD BE IDENTICAL TO Original Tokens
print("Original Tokenizer IDs:", ids_original)
print("Re-indexed Tokenizer IDs:", ids_reindexed)

# Check if token sequences are identical
if tokens_original == tokens_reindexed:
    print("✅ Token sequences are IDENTICAL (Good!)")
else:
    print("❌ Token sequences are DIFFERENT (PROBLEM!)")

test_token = "Ġis" # Choose a token that *should* be in overlap_map

if test_token in overlap_map:
    old_id = overlap_map[test_token]['old_id']
    reindexed_id = tokenizer_reindexed.encode(test_token, add_special_tokens=False)[0]
    print(f"Verification for token: '{test_token}'")
    print(f"  Old ID (tokenizer.json): {old_id}")
    print(f"  Re-indexed ID (tokenizerNEW_REINDEXED.json): {reindexed_id}")
    if old_id == reindexed_id:
        print(f"  ✅ ID MATCH")
    else:
        print(f"  ❌ ID MISMATCH")
else:
    print(f"Token '{test_token}' NOT in overlap_map (Unexpected for 'Ġis'!)")


# --- Detailed Verification Output ---
print("\n--- Detailed Token ID Remapping Verification ---")
print("Token String\tOld ID (tokenizer.json)\tNew ID (tokenizerNEW_REINDEXED.json)")
print("-" * 80)

sorted_overlap_tokens = sorted(overlap_map.keys()) # Sort tokens alphabetically for easier reading

for token_str in sorted_overlap_tokens:
    old_id = overlap_map[token_str]['old_id']
    reindexed_id = tokenizer_reindexed.encode(token_str, add_special_tokens=False)[0]
    if old_id == reindexed_id:
        status = "✅ Match"
    else:
        status = "❌ MISMATCH"
    print(f"{token_str}\t\t{old_id}\t\t{reindexed_id}\t\t{status}")

print("\n--- Verification Complete ---")