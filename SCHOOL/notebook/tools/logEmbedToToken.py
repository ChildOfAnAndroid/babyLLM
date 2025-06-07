import torch
import numpy as np
import json
from SCHOOL.staffroom.librarian import VOCAB
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Load VOCAB class
vocab = VOCAB()

# Load vocab list
vocabList = vocab.vocabList  # The list of all tokens

# Load embeddings from embedLayer.weights file
embedFile = "model_parameters/embedLayer_weights.txt"
similarTokensFile = "model_parameters/similarTokens.txt"

# Read embeddings
with open(embedFile, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Extract only the numbers (skip header lines)
embeddings = []
for line in lines:
    if not line.startswith("Parameter") and not line.startswith("Shape"):
        embeddings.append(np.array([float(x) for x in line.strip().split(",")]))

# Convert to NumPy array
embeddings = np.vstack(embeddings)  # Shape: (2000, 32)

# ✅ **Create a dictionary mapping tokens → embeddings**
token_embeddings = {vocabList[i]: embeddings[i].tolist() for i in range(len(vocabList))}

# Save token-embedding pairs as JSON
with open("token_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(token_embeddings, f, indent = 2)

print("✅ Token embeddings saved to 'token_embeddings.json'")

# Load token embeddings
with open("token_embeddings.json", "r", encoding="utf-8") as f:
    token_embeddings = json.load(f)

# Convert JSON data into NumPy arrays
tokens = list(token_embeddings.keys())
embeddings = np.array([token_embeddings[token] for token in tokens])

# Compute cosine similarity between all tokens
similarity_matrix = cosine_similarity(embeddings)

average_similarities = {
    token: np.mean(similarity_matrix[i]) for i, token in enumerate(tokens)
}

# **🔥 Sort Tokens by Their Overall Similarity Strength 🔥**
sorted_tokens = sorted(average_similarities.items(), key = lambda x: x[1], reverse = True)

def debug_token(tokenizer, token):
    """Prints a token's raw byte encoding and its true decoded text from the tokenizer."""
    
    # Decode using the tokenizer itself (BEST way to fix weird tokens)
    decoded_text = tokenizer.decode([token]) if isinstance(token, int) else token
    
    # Convert token to raw UTF-8 bytes
    utf8_bytes = token.encode("utf-8", errors="ignore") if isinstance(token, str) else bytes([token])
    
    hex_values = " ".join([f"0x{b:02x}" for b in utf8_bytes])  # Convert bytes to hex

    print(f"🔍 Token: {repr(token)}")
    print(f"   🔹 UTF-8 Byte Values: {list(utf8_bytes)}")
    print(f"   🔹 Hex Representation: {hex_values}")
    print(f"   🔹 Decoded from Tokenizer: '{decoded_text}'")  # Fixes tokenizer artifacts
    print("-" * 50)


# Function to find most similar words
def find_similar(token, top_n = 5):
    if token not in tokens:
        print(f"❌ Token '{token}' not in vocabulary.")
        return
    
    token_idx = tokens.index(token)
    similarities = similarity_matrix[token_idx]
    
    # Get top similar words (excluding itself)
    similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
    #similar_tokens = [(tokens[i], similarities[i]) for i in similar_indices]
    similar_tokens = [f"{tokens[i].replace('Ġ', ' ').strip()} ({similarities[i]:.4f})" for i in similar_indices]
    
    return ', '.join(similar_tokens)

with open(similarTokensFile, "w", encoding="utf-8") as f:
    for token, avg_sim in sorted_tokens:
        similar = find_similar(token, top_n = 5)
        response = ''.join(token).replace('Ġ', ' ').strip() # replace Ġ with space
        response = ' '.join(response.split())
        debug_token(vocab.tokenizer, token)
        f.write(f'Most similar tokens to "{response}" (Avg Sim: {avg_sim:.4f}): {similar}\n')

# 🔥 Example Usage:
print("Most similar tokens to 'charis':", find_similar("charis"))
print("Most similar tokens to 'elodie':", find_similar("elodie"))

 __pycache__ BRAIN BRAIN/LAYERS BRAIN/LAYERS/__pycache__ BRAIN/LAYERS/embedLayer.py BRAIN/LAYERS/memoryLayer.py BRAIN/LAYERS/outputLayer.py BRAIN/LAYERS/parallelNeuronLayer.py BRAIN/LAYERS/S_output.py BRAIN/LAYERS/vocab.py BRAIN/shapeofwords BRAIN/soul BRAIN/soul/archive BRAIN/soul/babyLLM_batched.pth BRAIN/soul/babyLLM_legacy.pth BRAIN/soul/babyLLM.pth BRAIN/soul/babyLLM.pth copy BRAIN/soul/stepCheckpoint.txt BRAIN/vocabCache BRAIN/readmeactuallyprobablydont.txt LOGS LOGS/chat LOGS/dataArchive LOGS/duration LOGS/training SCHOOL SCHOOL/library SCHOOL/library/archive SCHOOL/library/charisStudies SCHOOL/library/miniTraining SCHOOL/library/mouseAdventure SCHOOL/library/simpleTraining SCHOOL/library/tenses SCHOOL/notebook SCHOOL/notebook/archive SCHOOL/notebook/tools SCHOOL/notebook/tools/archive SCHOOL/notebook/tools/addTotalSteps.py SCHOOL/notebook/tools/brainAnalysis.py SCHOOL/notebook/tools/geepymatplot.py SCHOOL/notebook/tools/logEmbedToToken.py SCHOOL/notebook/tools/logModel.py SCHOOL/notebook/inferenceQuestionIdeas SCHOOL/notebook/test.pth SCHOOL/notebook/test.py SCHOOL/__trainingData.txt SCHOOL/library/trainingData.txt .gitattributes .gitignore babyLLM.py chatLogPath_infer config.py infer.py injectHeroinBro.py talkToYourself.py textCleaningTool.py