from safetensors.torch import safe_open, save_file, load_file

tensors = safe_open("/Users/tejag/projects/dart/libtorchdart/data/safetensors/SophraxiaChroma-000012.safetensors", framework="pt", device=0)
tensor = tensors.get_tensor('lora_unet_double_blocks_0_img_attn_proj.alpha')
print(tensor)
print(tensor.shape)
print(tensor.dim())
print(tensor.item())
tensor = tensors.get_tensor('lora_unet_single_blocks_31_linear2.lora_down.weight')
print(tensor[0].shape)
print(tensor[0][0].shape)
print(tensor[0][15359].item())