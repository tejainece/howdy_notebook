"""
Script to invoke one of the ResNet blocks from the Stable Diffusion 1.5 VAE.

This script loads the SD 1.5 VAE model and demonstrates how to access and
invoke one of its ResnetBlock2D components from the decoder.
"""

import torch
from diffusers import StableDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_path = "models/diffusion/v1-5-pruned-emaonly.safetensors"
pipe = StableDiffusionPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
resnet = pipe.to(device).vae.decoder.mid_block.resnets[0]

batch_size = 1
in_channels = resnet.in_channels
height, width = 8, 8

torch.manual_seed(42)
input = torch.randn(
    batch_size, in_channels, height, width,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device=device
)
output = resnet(input, temb=None)
print(output)

print("\n✓ Successfully wrote resnet testcases")