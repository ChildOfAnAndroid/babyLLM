import torch

model_path = "babyLLM.pth"
try:
    state_dict = torch.load(model_path)
    shape = state_dict['parallelNeuronLayer.combinationLayer.weight'].shape
    print(f"Shape of combinationLayer.weight in {model_path}: {shape}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
except KeyError:
    print(f"Error: combinationLayer.weight not found in {model_path} state_dict.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")