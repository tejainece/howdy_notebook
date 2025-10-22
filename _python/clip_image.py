from torchvision.io import read_image
from transformers import CLIPImageProcessor

images = read_image("images/swordsman1_512.png")
clipImage = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
clipImage(images=images, device="cuda")
print('Fnished!')