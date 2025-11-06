import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


device = "cuda"
dtype = torch.half

pipe = StableDiffusionPipeline.from_single_file("./models/diffusion/v1-5-pruned-emaonly.safetensors", torch_dtype=torch.float16, use_safetensors=True, local_files_only=True).to(device)
#print(pipe.vae.decoder)

img = Image.open("./images/swordsman1_512.png")
imgTensor = pil_to_tensor(img)[:3, :, :].unsqueeze(0) / 255.0
print(imgTensor.shape)
#print(imgTensor)
imgTensor =imgTensor.to(device=device, dtype=dtype)

latent: DiagonalGaussianDistribution = pipe.vae.encode(imgTensor, return_dict=False)
print(latent)

print(pipe.vae.quant_conv)

print('Finished!')