import torch
from diffusers import StableDiffusionPipeline
from diffusers.models.resnet import ResnetBlock2D

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_path = "models/diffusion/v1-5-pruned-emaonly.safetensors"
pipe = StableDiffusionPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
resnet_block = pipe.to(device).unet.down_blocks[0].resnets[0]

batch_size = 1
in_channels = resnet_block.in_channels
height, width = 64, 64

torch.manual_seed(42)
sample = torch.randn(
    batch_size, in_channels, height, width,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device=device
)
time_emb_dim = 1280
temb = torch.randn(
    batch_size, time_emb_dim,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device=device
)

output = resnet_block(sample, temb)
print(output)

