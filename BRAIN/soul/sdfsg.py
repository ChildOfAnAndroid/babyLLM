import torch

# Load the saved state dict (DO NOT load with the model!)
bad_state = torch.load("babyLLM_legacy_x.pth", map_location="cpu")

# Remove the unwanted key
if "embed.lastSavedEmbeds" in bad_state:
    print("Removing embed.lastSavedEmbeds...")
    del bad_state["embed.lastSavedEmbeds"]
else:
    print("Key not found? Already clean maybe.")

# Save the cleaned state dict
torch.save(bad_state, "babyllm.pth")
print("Cleaned and saved as your_model_clean.pt ✅")