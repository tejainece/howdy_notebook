import torch

tensor = torch.arange(70)
print(tensor.shape)
print(tensor)
tensor = tensor.expand((1, -1))
print(tensor.shape)
print(tensor)