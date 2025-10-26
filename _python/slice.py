import torch

tensorOrig = torch.arange(70)
tensor = tensorOrig.view(2, 35)
print(tensor)

tensor = tensor[:, :28]
print(tensor)

tensor = tensorOrig[:7]
print(tensor)