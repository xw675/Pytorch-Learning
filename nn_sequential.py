import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./Dataset', train=False, transform=torchvision.transforms.ToTensor(), download=False)
data_loader = DataLoader(dataset, batch_size=64)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, 5, 1, 2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model_1 = MyModel()

print(model_1)

class MySequentialModel(nn.Module):
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

model_2 = MySequentialModel()
print(model_2)

writer = SummaryWriter("logs")
writer.add_graph(model_1, torch.randn(64, 3, 32, 32))
writer.add_graph(model_2, torch.randn(64, 3, 32, 32))
writer.close()

# conv1 = nn.Conv2d(3, 32, 5, 1, 2)
# conv2 = nn.Conv2d(32, 32, 5, 1, 2)
# conv3 = nn.Conv2d(32, 64, 5, 1, 2)
# maxpool1 = nn.MaxPool2d(2)
# maxpool2 = nn.MaxPool2d(2)
# maxpool3 = nn.MaxPool2d(2)
# flatten = nn.Flatten()
# fc1 = nn.Linear(64 * 4 * 4, 10)
# x = torch.randn(64, 3, 32, 32)
# x = conv1(x)
# print(f"conv1: {x.shape}")
# x = maxpool1(x)
# print(f"maxpool1: {x.shape}")
# x = conv2(x)
# print(f"conv2: {x.shape}")
# x = maxpool2(x)
# print(f"maxpool2: {x.shape}")
# x = conv3(x)
# print(f"conv3: {x.shape}")
# x = maxpool3(x)
# print(f"maxpool3: {x.shape}")
# x = flatten(x)
# print(f"flatten: {x.shape}")
# x = fc1(x)
# print(f"fc1: {x.shape}")