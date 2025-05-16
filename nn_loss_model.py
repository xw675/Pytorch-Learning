import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./Dataset', train=False, transform=torchvision.transforms.ToTensor(), download=False)
data_loader = DataLoader(dataset, batch_size=1)

class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 10)
        )

    def forward(self, x):
        return self.model(x)

model = MyLoss()
for i, (image, label) in enumerate(data_loader):
    images, labels = image, label
    output = model(images)
    loss = nn.CrossEntropyLoss()(output, labels)
    loss.backward()
    print(f"Loss: {loss.item()}")
