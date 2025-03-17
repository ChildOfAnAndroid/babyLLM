import torch
from babyLLM import BABYLLM
from vocab import VOCAB
from config import *

# Load the model
model_path = "babyLLM.pth"  
print(f"🔄 Loading model from {model_path}...")
try:
    vocab = VOCAB(vocabSize = vocabSize)
    babyLLM = BABYLLM(vocab = vocab, embedDimension = embedDimension, numNeurons = numNeurons, activationFunction = activationFunction)
    print(f"DEBUG: tokenToIndex first 50 keys: {list(babyLLM.vocab.tokenToIndex.keys())[:50]}")
    print(f"DEBUG: indexToToken first 50 values: {list(babyLLM.vocab.indexToToken.values())[:50]}")
    babyLLM.load_state_dict(torch.load(model_path), strict=False)
    babyLLM.eval()
    print(f"✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Model file not found!")
    exit()

# Test if he still "knows" anything
test_phrases = [
    ["hello", "," ,"how", "are", "you", "?"],
    ["i", "love", "music"],
    ["tell", "me", "a", "joke"],
]

for test_input in test_phrases:
    input_indices = [vocab.tokenToIndex.get(token, vocab.tokenToIndex["<UNK>"]) for token in test_input]
    logits = babyLLM.forward(input_indices)
    predicted_index = babyLLM.getResponseFromLogits(logits)
    predicted_token = vocab.indexToToken.get(str(predicted_index), "<UNK>")

    print(f"\n🧠 **Input:** {' '.join(test_input)}")
    print(f"🤖 **Model Response:** \"{predicted_token}\"")
