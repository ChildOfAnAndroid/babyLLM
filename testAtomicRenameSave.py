import torch.nn as nn
import torch
import os
import time
class ModelTest(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, pth="test.pth"):
        tmp_path = pth + ".tmp"
        torch.save(self.state_dict(), tmp_path)
        os.replace(tmp_path, pth)
    def load(self, pth="test.pth"):
        self.load_state_dict(torch.load(pth), strict = False)



t = ModelTest()

t.load()
t.bigTensor = torch.randn([10000,10000,10000])
print("ok model loaded now to wait 10 sec")
time.sleep(10)
print("gonna save now")
t.save()
print("too late save done")