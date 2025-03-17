import torch

model_path = "babyLLM.pth"  # Path to your saved model

try:
    # 1. Load the state dictionary from the saved model file
    state_dict = torch.load(model_path)
    print(f"🔄 Loaded state dictionary from {model_path}")

    # 2. Fix tensor size mismatches
    new_state_dict = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            expected_shape = None

            # Detect mismatches and fix them
            if "windowAttn.attention.in_proj_weight" in key and tensor.shape != (3072, 1024):
                expected_shape = (3072, 1024)
            elif "windowAttn.attention.in_proj_bias" in key and tensor.shape != (3072,):
                expected_shape = (3072,)
            elif "windowAttn.attention.out_proj.weight" in key and tensor.shape != (1024, 1024):
                expected_shape = (1024, 1024)
            elif "windowAttn.ffn.0.weight" in key and tensor.shape != (2048, 1024):
                expected_shape = (2048, 1024)
            elif "windowAttn.ffn.0.bias" in key and tensor.shape != (2048,):
                expected_shape = (2048,)
            elif "windowAttn.ffn.2.weight" in key and tensor.shape != (1024, 2048):
                expected_shape = (1024, 2048)

            # Resize if needed
            if expected_shape:
                print(f"🔧 Resizing {key}: {tensor.shape} -> {expected_shape}")

                # If 1D tensor (bias), resize differently
                if len(expected_shape) == 1:
                    new_tensor = torch.zeros(expected_shape[0])  # 1D tensor
                    min_size = min(tensor.shape[0], expected_shape[0])
                    new_tensor[:min_size] = tensor[:min_size]
                else:
                    new_tensor = torch.zeros(expected_shape)  # 2D tensor
                    min_size = tuple(min(s1, s2) for s1, s2 in zip(tensor.shape, expected_shape))
                    new_tensor[:min_size[0], :min_size[1]] = tensor[:min_size[0], :min_size[1]]

                new_state_dict[key] = new_tensor
            else:
                new_state_dict[key] = tensor  # Keep unchanged

    # 3. Save the fixed state dictionary
    torch.save(new_state_dict, model_path)
    print(f"✅ Successfully fixed tensor mismatches in {model_path}")

except FileNotFoundError:
    print(f"❌ Error: Model file not found at {model_path}")
except Exception as e:
    print(f"❌ An error occurred: {e}")
    print("The model file may not have been fixed. Check the error message.")
