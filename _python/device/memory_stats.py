import torch
from pprint import pprint

tensor = torch.randn(1, 3, 64, 64).to(device='cuda:0')

if not torch.cuda.is_available():
    print("CUDA is not available.")
    exit()

pprint(torch.cuda.memory_stats(device='cuda:0'))
print(torch.cuda.get_device_properties(device='cuda:0').total_memory)
