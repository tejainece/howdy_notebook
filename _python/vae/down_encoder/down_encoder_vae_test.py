from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "models/diffusion/v1-5-pruned-emaonly.safetensors"
pipe = StableDiffusionPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
downEncoder: DownEncoderBlock2D = pipe.to(device).vae.encoder.down_blocks

# TODO

output = downEncoder.forward(input)
#print(output)

name = "vae"
tensors = {
    f"{name}.input": input,
    f"{name}.output": output,
    **{f"{name}.block{k}": v for k, v in downEncoder.state_dict().items()},
}
#print(tensors.keys())

import os
os.makedirs("test_data/vae/down_encoder", exist_ok=True)

from safetensors.torch import save_file
save_file(tensors, "test_data/vae/down_encoder/down_encoder_vae.safetensors")

print("\n✓ Successfully generated DownEncoderBlock2D vae testcases")