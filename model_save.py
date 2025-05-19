import torch
import torch.nn as nn
import torchvision

vgg16 = torchvision.models.vgg16(weights=None)
torch.save(vgg16, "vgg16_method_1.pth")

torch.save(vgg16.state_dict(), "vgg16_method_2.pth")