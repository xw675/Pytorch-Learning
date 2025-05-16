import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

pretrained_model_1 = torchvision.models.vgg16(pretrained=True)
pretrained_model_2 = torchvision.models.vgg16(pretrained=False)

dataset = torchvision.datasets.CIFAR10(root='./Dataset', train=False, transform=torchvision.transforms.ToTensor(), download=False)
data_loader = DataLoader(dataset, batch_size=64)

# adding a new layer to the pretrained model
pretrained_model_1.classifier.add_module("MyVGG16", nn.Linear(1000, 10))
print(pretrained_model_1)

# editing the pretrained model
pretrained_model_2.classifier[6] = nn.Linear(4096, 10)
print(pretrained_model_2)