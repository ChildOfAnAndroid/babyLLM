import torch
import torch.nn.functional as F
import numpy as np

# Load the model
model_path = "babyLLM.pth"
fixed_model_path = "babyLLM.pth"

print(f"🔄 Loading state dictionary from {model_path}...")
try:
    state_dict = torch.load(model_path)
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {model_path}")
    exit(1)

changes = []

# Fix extreme weights & repetitive neurons
for key, tensor in state_dict.items():
    if isinstance(tensor, torch.Tensor):
        tensor_np = tensor.cpu().numpy()

        # Detect & fix extreme weights
        abs_mean = np.mean(np.abs(tensor_np))
        abs_max = np.max(np.abs(tensor_np))

        if abs_max > abs_mean * 100:  # Extreme outliers
            tensor_fixed = np.clip(tensor_np, -abs_mean * 10, abs_mean * 10)
            state_dict[key] = torch.tensor(tensor_fixed)
            changes.append(f"🔧 Fixed extreme weights in {key}: Clipped to ±{abs_mean * 10:.4f}")

        # Detect & fix repetitive neurons (small variance)
        if len(tensor_np.shape) == 2 and tensor_np.shape[0] > 1:
            row_variances = np.var(tensor_np, axis=1)
            too_similar = np.sum(row_variances < 1e-6)

            if too_similar > 0:
                jitter = np.random.normal(scale=1e-4, size=tensor_np.shape)
                tensor_fixed = tensor_np + jitter
                state_dict[key] = torch.tensor(tensor_fixed)
                changes.append(f"🔧 Added small noise to {too_similar} similar neurons in {key}")

# Save fixed model
torch.save(state_dict, fixed_model_path)
print(f"✅ Fixed model saved as {fixed_model_path}")

# Print all changes
if changes:
    print("\n🛠️ Changes made:")
    for change in changes:
        print(change)
else:
    print("✅ No major issues detected. Model is fine!")
