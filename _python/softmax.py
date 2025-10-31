import torch

tensorOrig = torch.arange(700)
tensorOrig.softmax(dim=0, dtype=torch.float16)
print(tensorOrig)