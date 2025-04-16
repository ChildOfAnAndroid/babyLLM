import re
from collections import Counter
from VER1_config import *

# Path to your dataset
file_path = trainingFilePath

# Process file efficiently
words = []
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        # Tokenize and clean words
        line_words = [re.sub(r"[^a-zA-Z]", "", word).lower() for word in line.split()]
        words.extend([word for word in line_words if 3 < len(word) <= 35])  # Ignore single-letter words

# Count word frequency
word_counts = Counter(words)

# Set a threshold (e.g., only words that appear at least 100 times)
common_words = [word for word, count in word_counts.items() if count > 500]

# Generate structured training examples
training_examples = []
for word in sorted(set(common_words)):
    training_examples.append(f"Is {word} a word? Yes, {word} is a word")
    training_examples.append(f"What is {word}? {word.capitalize()} is something important")
    training_examples.append(f"Please say {word}. {word.capitalize()}!")
    training_examples.append(f"Please say {word} twice. {word.capitalize()} {word.capitalize()}!")

# Save the cleaned training data
output_file = "babyllm_literal_training_large.txt"
with open(output_file, "w", encoding="utf-8") as file:
    file.write("\n".join(training_examples))

print(f"✅ Done! Training data saved as {output_file}")

