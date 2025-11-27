import torch
from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D
import os
from safetensors.torch import save_file

torch.manual_seed(0)
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Using standard parameters similar to what might be used in a VAE or UNet
in_channels = 32
out_channels = 64
down_block = DownEncoderBlock2D(
    in_channels=in_channels,
    out_channels=out_channels,
    num_layers=2,
    resnet_eps=1e-6,
    resnet_act_fn="swish",
    resnet_groups=32,
    resnet_pre_norm=True,
    output_scale_factor=1.0,
    add_downsample=True,
    downsample_padding=1
).to(torch_device)

sample = torch.randn(1, in_channels, 64, 64).to(torch_device)

output = down_block(sample)

name = "simple1"
tensors = {
    f"{name}.input": sample,
    f"{name}.output": output,
    **{f"{name}.block{k}": v for k, v in down_block.state_dict().items()}
}

os.makedirs("test_data/vae/down_encoder", exist_ok=True)

save_path = "test_data/vae/down_encoder/down_encoder_simple.safetensors"
save_file(tensors, save_path)


print("\n✓ Successfully generated DownEncoderBlock2D testcases")
