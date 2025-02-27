import json
import sys
import time
import threading
from transformers import pipeline

# Load Mistral 7B model for text generation
model = pipeline("text-generation", model="mistralai/Mixtral-8x7B-Instruct-v0.1", device="auto")  # Use CPU (auto/-1) for now

# Load memory from a file
def load_memory():
    try:
        with open('mistral_memory.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"history": []}  # If the file doesn't exist, start with an empty history

# Save memory to the file
def save_memory(memory):
    with open('mistral_memory.json', 'w') as f:
        json.dump(memory, f, indent=4)

# Simple loading animation
def loading_animation():
    """Creates a 'thinking' animation while the AI is generating a response."""
    symbols = ['.  ', '.. ', '...']
    i = 0
    while not stop_loading:
        sys.stdout.write(f"\rMistral is thinking {symbols[i % len(symbols)]}")
        sys.stdout.flush()
        i += 1
        time.sleep(0.5)
    sys.stdout.write("\r" + " " * 30 + "\r")  # Clears animation after completion

# Generate a response using Mistral
def generate_response(user_input, memory):
    global stop_loading
    stop_loading = False  # Reset loading flag

    # Start the loading animation in a separate thread
    loading_thread = threading.Thread(target=loading_animation)
    loading_thread.start()

    # Limit context to the last 5 messages for better focus
    context = "\n".join(memory["history"][-5:])  # Keep the last 5 interactions
    prompt = f"Context: {context}\nUser: {user_input}\nMistral:"

    response = model(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"]
    
    # Stop loading animation
    stop_loading = True
    loading_thread.join()

    return response.strip()

# Main chat function
def chat_with_mistral():
    print("Start chatting with Mistral! Type 'exit' to quit.")
    memory = load_memory()  # Load memory

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Ending conversation...")
            break

        # Generate and print the response from Mistral
        response = generate_response(user_input, memory)
        print(f"Mistral says: {response}")

        # Update memory with the new conversation
        memory["history"].append(f"User: {user_input}")
        memory["history"].append(f"Mistral: {response}")
        save_memory(memory)  # Save the updated memory

# Start chatting
chat_with_mistral()
