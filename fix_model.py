import torch
import torch.nn as nn
from config import *  # Import your config.py file
from parallelNeuronLayer import PARALLELNEURONLAYER # Import the layer definition

model_path = "babyLLM.pth"  # Path to your saved model

try:
    # 1. Load the state dictionary from the saved model file
    state_dict = torch.load(model_path)
    print(f"Loaded state dictionary from {model_path}")

    # 2. Create a dummy PARALLELNEURONLAYER instance to access current config (allWindowSizes, numNeurons)
    #    We don't need to use this full layer, just access its config for correct sizing.
    dummy_parallel_layer = PARALLELNEURONLAYER(numNeurons=numNeurons, embedDimension=embedDimension, activationFunction=activationFunction)

    # 3. Create a NEW combinationLayer with the CORRECT size based on current config
    new_combinationLayer = nn.Linear(numNeurons * len(dummy_parallel_layer.allWindowSizes), numNeurons)

    # 4. Get the state dict of this NEW, correctly sized combinationLayer (randomly initialized)
    new_combinationLayer_state = new_combinationLayer.state_dict()

    # 5. Replace the combinationLayer weights and biases in the LOADED state_dict
    state_dict['parallelNeuronLayer.combinationLayer.weight'] = new_combinationLayer_state['weight']
    state_dict['parallelNeuronLayer.combinationLayer.bias'] = new_combinationLayer_state['bias']

    # 6. Save the MODIFIED state dictionary back to the SAME file, overwriting it.
    torch.save(state_dict, model_path)
    print(f"✅ Successfully replaced combinationLayer in {model_path} with correctly sized and re-initialized version.")
    print("   Your babyLLM.pth file is now permanently fixed.")

except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
except Exception as e:
    print(f"An error occurred: {e}")
    print("The model file may not have been fixed. Check the error message.")