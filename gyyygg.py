import torch
import torch.nn as nn
import numpy as np

# Load the saved model state
model_path = "babyLLM.pth"
fixed_model_path = "babyLLM.pth"

print("🔄 Loading state dictionary from babyLLM.pth...")

try:
    state_dict = torch.load(model_path)

    # Track changes
    changes = []

    def reinitialize_tensor(tensor):
        """Reinitializes a tensor with random values (same shape, normal distribution)"""
        return torch.randn_like(tensor) * 0.02  # Small initialization to keep stability

    # Identify problematic layers based on previous logs
    problematic_layers = [
        "parallelNeuronLayer.windowAttn.attention.in_proj_weight",
        "parallelNeuronLayer.windowAttn.attention.out_proj.weight",
        "parallelNeuronLayer.windowAttn.ffn.0.weight",
        "parallelNeuronLayer.windowAttn.ffn.2.weight"
    ]

    # Track reinitialized neurons
    neurons_reset = 0

    for layer in problematic_layers:
        if layer in state_dict:
            tensor = state_dict[layer]

            # Find neurons that have barely changed despite multiple noise injections
            std_dev = torch.std(tensor).item()
            mean_abs = torch.mean(torch.abs(tensor)).item()
            min_val, max_val = torch.min(tensor).item(), torch.max(tensor).item()

            if std_dev < 0.0005 or mean_abs < 0.0005:  # If too "frozen," reinitialize
                state_dict[layer] = reinitialize_tensor(tensor)
                neurons_reset += tensor.numel()
                changes.append(f"🔄 Fully reset {layer} ({tensor.shape})")

            elif min_val == max_val:  # If all weights are identical (bad sign), reinitialize
                state_dict[layer] = reinitialize_tensor(tensor)
                neurons_reset += tensor.numel()
                changes.append(f"⚠️ Identical weights in {layer}, reinitialized!")

    # Save the updated model
    torch.save(state_dict, fixed_model_path)
    print(f"✅ Fixed model saved as {fixed_model_path}")

    # Print out what changed
    if changes:
        print("\n🛠️ **Changes made:**")
        for change in changes:
            print(change)
        print(f"🔧 Total neurons reset: {neurons_reset}")
    else:
        print("✅ No extreme issues detected, model was already in good shape!")

except Exception as e:
    print(f"❌ An error occurred: {e}")
    print("The model file may not have been fixed. Check the error message.")
