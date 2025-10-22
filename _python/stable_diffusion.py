import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_single_file("/home/tejag/comfy/ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors", torch_dtype=torch.float16, use_safetensors=True, local_files_only=True).to("cuda")