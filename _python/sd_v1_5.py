import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_single_file("./models/diffusion/v1-5-pruned-emaonly.safetensors", torch_dtype=torch.float16, use_safetensors=True, local_files_only=True).to("cuda")
promptEncoded = pipe.encode_prompt('a photograph of an astronaut riding a horse', device="cuda",
                   num_images_per_prompt=1, do_classifier_free_guidance=False)
#print(promptEncoded)
#print(promptEncoded[0].shape)
#print(pipe.text_encoder)
#print(pipe.unet)
print(pipe.vae.decoder)


print('Finished!')