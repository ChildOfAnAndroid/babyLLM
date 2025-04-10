import torch
import os
from config import *

# Load the checkpoint
model_path = modelFilePath  # Change to actual path
checkpoint = torch.load(model_path, map_location="cpu")

# Extract state_dict (or raw tensors if state_dict doesn’t exist)
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint  # Assume raw weights

# Create directory to store parameters
output_dir = "model_parameters"
os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

for key, value in state_dict.items():
    # Clean key name for safe filenames
    safe_key = key.replace(".", "_").replace("/", "_")

    # File path for this parameter
    file_path = os.path.join(output_dir, f"{safe_key}.txt")

    # Convert tensor to NumPy
    if isinstance(value, torch.Tensor):
        value = value.cpu().numpy()

    # **🚀 Stream data row-by-row instead of holding it all in RAM**
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f'Parameter: {key}\n')
        f.write(f'Shape: {tuple(value.shape) if hasattr(value, "shape") else len(value)}\n')

        # 🚀 Write tensor row-by-row (prevents RAM overload)
        # **Fix 0-D Scalars (Single Numbers)**
        if value.ndim == 0:
            f.write(str(value.item()) + "\n")

        # **Fix 1D Arrays (Write in One Line)**
        elif value.ndim == 1:
            f.write(",".join(map(str, value)) + "\n")

        # **Fix 2D+ Arrays (Write Row-by-Row)**
        else:
            for row in value:
                f.write(",".join(map(str, row)) + "\n")

print(f"✅ Model parameters saved in '{output_dir}/' (one file per parameter, no RAM explosions).")
