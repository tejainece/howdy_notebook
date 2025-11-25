import torch

torch.manual_seed(0)

device = 'cpu'
conv = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, device=device)

#print(conv)
#print(conv.weight)
#print(conv.bias)
input = torch.ones([1, 32, 28, 28])
output = conv.forward(input)
#print(out.shape)
#print(out)

tensors = {
    "test1.input": input,
    "test1.output": output,
}

import os
os.makedirs("test_data/conv2d", exist_ok=True)

from safetensors.torch import save_file
save_file(tensors, "test_data/conv2d/conv2d_tests.safetensors")
