import torch

tensor = torch.ones([1, 1, 14, 14])
conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding='same', padding_mode='reflect')
print(conv)
out = conv.forward(tensor)
print(out)
