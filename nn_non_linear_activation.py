import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='./Dataset', train=False, transform=torchvision.transforms.ToTensor(), download=False)

data_loader = DataLoader(dataset, batch_size=64)

class MyNonLinearActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(x)
        x = self.sigmoid(x)
        return x

model = MyNonLinearActivation()
writer = SummaryWriter("logs")

for i, (image, label) in enumerate(data_loader):
    images, labels = image, label
    output = model(images)
    writer.add_images('non_linear_activation_outputs', output, i)
    print(output.shape)

writer.close()