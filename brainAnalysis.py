import torch
import json
import numpy as np
import pandas as pd

def load_model(path):
    """Loads the BabyLLM model safely"""
    print("Loading BabyLLM's brain...")
    return torch.load(path, map_location=torch.device('cpu'))

def load_vocab(vocab_path):
    """Loads the vocabulary from a JSON file and ensures correct key-value mapping"""
    print(f"Loading vocabulary from {vocab_path}...")
    
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
    except Exception as e:
        print(f"❌ ERROR: Failed to load vocabulary file: {e}")
        return {}
    
    # Check if vocab is inside 'model'
    if "model" in vocab_data and "vocab" in vocab_data["model"]:
        vocab_data = vocab_data["model"]["vocab"]
    else:
        print("❌ ERROR: Could not find 'model->vocab' in tokenizer.json!")
        return {}

    vocab_fixed = {}
    for k, v in vocab_data.items():
        if v is None or isinstance(v, (list, dict)):
            print(f"⚠️ Skipping key {k} due to invalid value: {v}")
            continue
        try:
            vocab_fixed[int(v)] = k  # Convert safely
        except (ValueError, TypeError) as e:
            print(f"❌ Error converting {v} to int for key {k}: {e}")
    
    return vocab_fixed

def extract_word_embeddings(state_dict, vocab, num_words=2000):
    """Extracts word embeddings and returns a word-embedding dictionary"""
    print("Extracting word embeddings...")
    
    embeddings = None
    for key in state_dict.keys():
        if "embed" in key:
            embeddings = state_dict[key]
            break
    
    if embeddings is None:
        print("❌ No embedding layer found!")
        return None, None
    
    vocab_indices = sorted(vocab.keys())[:num_words]
    words = [vocab[idx] for idx in vocab_indices if idx in vocab]
    extracted_embeddings = embeddings[vocab_indices].detach().cpu().numpy()
    
    return words, extracted_embeddings

def compute_cosine_similarity(embeddings):
    """Computes cosine similarity between word embeddings"""
    embeddings = np.array(embeddings)  # Ensure NumPy array
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize vectors
    return np.dot(embeddings, embeddings.T) / (norms * norms.T)  # Cosine similarity

def save_similarity_table(words, similarity_matrix, output_file="similarity_matrix.csv"):
    """Saves similarity scores to a CSV file"""
    df = pd.DataFrame(similarity_matrix, index=words, columns=words)
    df.to_csv(output_file)
    print(f"\n🔍 Word Similarity Matrix saved to {output_file}")

if __name__ == "__main__":
    model_path = "babyLLM.pth"
    vocab_path = "vocabCache/tokenizer.json"

    state_dict = load_model(model_path)
    vocab = load_vocab(vocab_path)

    if not vocab:
        print("❌ Vocabulary loading failed!")
        exit()

    words, embeddings = extract_word_embeddings(state_dict, vocab, num_words=2000)

    if words and embeddings is not None:
        similarity_matrix = compute_cosine_similarity(embeddings)
        save_similarity_table(words, similarity_matrix)
    else:
        print("❌ Could not extract embeddings. Check model and vocab files!")
