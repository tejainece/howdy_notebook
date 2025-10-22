from transformers import CLIPTextModel

clipText = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
# clipText()
print('Finished!')