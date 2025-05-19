import torch
import torchvision

model = torch.load("vgg16_method_1.pth")
print(model)

vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load("vgg16_method_2.pth"))
print(vgg16)