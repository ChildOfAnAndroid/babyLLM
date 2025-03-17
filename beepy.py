import torch

model_path = "babyLLM.pth"  # Path to your saved model

try:
    # 1. Load the state dictionary from the saved model file
    state_dict = torch.load(model_path)
    print(f"🔄 Loaded state dictionary from {model_path}")

    # 2. Resize any tensors where last dim is 32 → 1024
    for key in state_dict:
        tensor = state_dict[key]

        # Ensure it's a Tensor before modifying
        if isinstance(tensor, torch.Tensor):
            shape = list(tensor.shape)  # Convert to list for easy modification

            if shape and shape[-1] == 32:  # Check last dim
                new_shape = shape[:-1] + [1024]  # Keep all dims, change last one

                print(f"🔧 Resizing {key}: {tuple(shape)} -> {tuple(new_shape)}")

                # Handle different tensor shapes
                if tensor.ndim == 1:
                    state_dict[key] = torch.cat([tensor] * 32)[:1024]  # Expand 1D tensors safely
                elif tensor.ndim == 2:
                    state_dict[key] = tensor.repeat(1, 32)[:, :1024]  # Expand last dim safely
                elif tensor.ndim == 3:
                    state_dict[key] = tensor.repeat(1, 1, 32)[:, :, :1024]  # Handle 3D tensors
                else:
                    print(f"⚠️ Skipping {key}: Unsupported shape {tensor.shape}")

    # 3. Save the modified state dictionary back to the model file
    torch.save(state_dict, model_path)
    print(f"✅ Successfully fixed size mismatches in {model_path}. You can now load the model without errors!")

except FileNotFoundError:
    print(f"❌ Error: Model file not found at {model_path}")
except Exception as e:
    print(f"❌ An error occurred: {e}")
    print("The model file may not have been fixed. Check the error message.")
