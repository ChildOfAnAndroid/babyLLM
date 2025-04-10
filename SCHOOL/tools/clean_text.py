import os
import re
import json
from collections import Counter
filepath = "SCHOOL/trainingData.txt"

def get_char_frequencies(filepath):
    """Reads text and counts character frequencies."""
    char_counter = Counter()
    
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().lower()
        char_counter.update(text)  # ✅ Count characters

    return char_counter

def preprocess_text(filepath, char_frequencies, min_occurrences=100):
    """Reads and lowercases text, removing rare characters."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().lower()

    # ✅ Keep only characters that appear at least `min_occurrences` times
    cleaned_text = "".join(char if char_frequencies[char] >= min_occurrences else " " for char in text)

    # ✅ Remove double spaces from removed characters
    cleaned_text = " ".join(cleaned_text.split())

    return cleaned_text

def clean_file(input_file, output_file, min_occurrences=100):
    """Cleans a text file by removing rare characters and saves the result."""
    
    print(f"📂 Processing file: {input_file}")
    
    # ✅ Step 1: Count character frequencies
    char_frequencies = get_char_frequencies(input_file)

    # ✅ Step 2: Process text with filtering
    cleaned_text = preprocess_text(input_file, char_frequencies, min_occurrences)

    # ✅ Step 3: Save cleaned text
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"✅ Cleaned text saved to: {output_file}")

# --------------- RUN THE SCRIPT ---------------
if __name__ == "__main__":
    # Set input and output file paths
    input_file = "SCHOOL/trainingData.txt"  # Change this to your actual file
    output_file = "data/CHARIS/trainingData_lessCharacters.txt"

    # Clean the text
    clean_file(input_file, output_file, min_occurrences=100)

    print("🚀 Cleaning complete! You can now use cleaned_corpus.txt for training.")
