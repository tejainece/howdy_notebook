import torch
from diffusers.models.resnet import ResnetBlock2D

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
sample = torch.randn(1, 32, 64, 64).to(torch_device)
temb = torch.randn(1, 128).to(torch_device)
resnet = ResnetBlock2D(in_channels=32, temb_channels=128).to(torch_device)
output_tensor = resnet.forward(sample, temb)

tensors = {
    "test1.input": sample,
    "test1.temb": temb,
    "test1.output": output_tensor,
    **{f"test1.{k}": v for k, v in resnet.state_dict().items()},
}
#print(tensors.keys())

import os
os.makedirs("test_data/resnet", exist_ok=True)

from safetensors.torch import save_file
save_file(tensors, "test_data/resnet/resnet_tests.safetensors")

print("All tests passed!")