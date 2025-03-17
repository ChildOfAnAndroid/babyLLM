import torch

# Load the saved model state dict
model_path = "babyLLM.pth"  # Change this if needed
try:
    state_dict = torch.load(model_path)
    print(f"🔄 Loaded model state from {model_path}")
except FileNotFoundError:
    print("❌ Model file not found!")
    exit()

# Check for size mismatches
model_layers = state_dict.keys()
mismatch_count = 0
dead_neurons = {}

for layer, weights in state_dict.items():
    if "weight" in layer or "bias" in layer:
        # Check for extreme values (exploding or dying)
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            print(f"❌ {layer} has NaN or Inf values!")
            dead_neurons[layer] = "NaN/Inf detected"
        elif (weights.abs() < 1e-10).sum() > 0.99 * weights.numel():
            print(f"⚠️ {layer} has 99%+ near-zero values (possible dead neurons)")
            dead_neurons[layer] = "Mostly dead"
        elif (weights.abs() > 1e3).sum() > 0.1 * weights.numel():
            print(f"⚠️ {layer} has extreme values (may be unstable)")
            dead_neurons[layer] = "Possible explosion"

if not dead_neurons:
    print("✅ No NaN/Inf detected. Model is at least numerically stable!")

# Print summary
print(f"\n🛠️ Total broken layers: {len(dead_neurons)}")
for layer, issue in dead_neurons.items():
    print(f"   🔧 {layer} → {issue}")
