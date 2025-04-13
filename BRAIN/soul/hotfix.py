import torch

path_in = "babyLLM_legacy.pth"
path_out = "babyLLM_fixed.pth"

state = torch.load(path_in, map_location="cpu")

# 🔥 Purge old memory tensors if they exist
for key in ["memory.shortTermMemory", "memory.longTermMemory"]:
    if key in state:
        print(f"🧹 Deleting old memory key: {key}")
        del state[key]

# ✅ Save cleaned state dict
torch.save(state, path_out)
print(f"✅ Saved cleaned model to: {path_out}")