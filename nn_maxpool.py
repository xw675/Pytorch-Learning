import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./Dataset', train=False, transform=torchvision.transforms.ToTensor(), download=False)
data_loader = DataLoader(dataset, batch_size=64)

class MyMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(3, ceil_mode=True)

    def forward(self, x):
        x = self.maxpool(x)
        return x

model = MyMaxPool2d()

writer = SummaryWriter("logs")

for i, (image, label) in enumerate(data_loader):
    images, labels = image, label
    output = model(images)
    writer.add_images('maxpool_outputs', output, i)
    print(output.shape)

writer.close()