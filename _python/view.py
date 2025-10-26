import torch

tensorOrig = torch.arange(700)
tensor = tensorOrig.view(2, 35, 10)
print(tensor)

tensor = tensor.view(-1, 10)
print(tensor)
print(tensor.shape)