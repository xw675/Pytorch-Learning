import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./Dataset', train=False, transform=torchvision.transforms.ToTensor(), download=False)
data_loader = DataLoader(dataset, batch_size=64)

class MyConv2dd(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 3, 1, 0)

    def forward(self, x):
        x = self.conv(x)
        return x

model = MyConv2dd()
# print(model)

writer = SummaryWriter("logs")

for i, (image, label) in enumerate(data_loader):
    images, labels = image, label
    output = model(images)
    output_reshaped = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images('conv_outputs', output_reshaped, i)
    print(output_reshaped.shape)

writer.close()