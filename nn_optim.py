import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = torchvision.datasets.CIFAR10(root='./Dataset', train=False, transform=torchvision.transforms.ToTensor(), download=False)
data_loader = DataLoader(dataset, batch_size=64)

class MyModel(nn.Module):
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

model = MyModel().to(device)
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    print(f"Epoch {epoch + 1}")
    running_loss = 0.0
    for i, (image, label) in enumerate(data_loader):
        image, label = image.to(device), label.to(device)
        images, labels = image, label
        output = model(images)
        result_loss = loss(output, labels)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss += result_loss.item()
        if running_loss < 0.01:
            break
    print(f"Loss: {running_loss / len(data_loader)}")