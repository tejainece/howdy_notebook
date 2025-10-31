import torch

tensor = torch.ones([4, 4])
print(tensor)
out = torch.nn.functional.pad(tensor, [0, 0, 0, 0], "edge")
print(out)