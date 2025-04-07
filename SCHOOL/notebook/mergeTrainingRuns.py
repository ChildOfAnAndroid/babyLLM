import torch
from config import *

#i'd have to set each to save to each
checkpoint1 = torch.load('babyLLM_MAC.pth')
checkpoint2 = torch.load('babyLLM_WINDOWS.pth')

merged_checkpoint = {}

# weighted merge (simple averaging)
w1, w2 = 0.5, 0.5

for key in checkpoint1.keys():
    if checkpoint1[key].shape == checkpoint2[key].shape:
        merged_checkpoint[key] = w1 * checkpoint1[key] + w2 * checkpoint2[key]
    else:
        print(f"skipped {key} due to shape mismatch")

torch.save(merged_checkpoint, modelFilePath)
print("merge completed :)")
