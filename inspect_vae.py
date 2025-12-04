from diffusers import StableDiffusionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "models/diffusion/v1-5-pruned-emaonly.safetensors"

try:
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe.to(device)
    
    print("VAE Down Blocks:")
    for i, block in enumerate(pipe.vae.encoder.down_blocks):
        print(f"Block {i}: {type(block)}")
        print(f"  In Channels: {block.resnets[0].in_channels}")
        print(f"  Out Channels: {block.resnets[0].out_channels}")
        # Check if it has downsamplers
        if hasattr(block, "downsamplers") and block.downsamplers is not None:
             print(f"  Downsamplers: Yes")
        else:
             print(f"  Downsamplers: No")

except Exception as e:
    print(f"Error: {e}")
