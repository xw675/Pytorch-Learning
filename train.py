import torch
import torch.nn as nn
import torchvision
from IPython.core.pylabtools import figsize
from torch.utils.data import DataLoader
from CIFAR10_model import MyCIFAR10
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

mean = (0.4914, 0.4822, 0.4465)
std = (0.247, 0.243, 0.261)
train_transform = torchvision.transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomRotation(15),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)])
test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean, std)])

train_dataset = torchvision.datasets.CIFAR10(root='./Dataset', train=True, transform=train_transform, download=False)
test_dataset = torchvision.datasets.CIFAR10(root='./Dataset', train=False, transform=test_transform, download=False)
train_data_loader = DataLoader(train_dataset, batch_size=64)
test_data_loader = DataLoader(test_dataset, batch_size=64)

model = MyCIFAR10().to(device)
model.load_state_dict(torch.load("CIFAR10_model.pth"))
loss = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')


writer = SummaryWriter("logs")

def train():
    model.train()
    for epoch in range(30):
        accuracy = 0
        print(f"Epoch {epoch + 1}")
        train_loss = 0.0
        for i , (image, label) in enumerate(train_data_loader):
            image, label = image.to(device), label.to(device)
            images, labels = image, label
            output = model(images)
            result_loss = loss(output, labels)

            optimizer.zero_grad()
            result_loss.backward()
            optimizer.step()
            train_loss += result_loss.item()
            writer.add_scalar("train_loss", result_loss.item(), epoch * len(train_data_loader) + i)
            pred = torch.argmax(output, dim=1)
            accuracy += (pred == labels).sum().item()

        avg_loss = train_loss / len(train_data_loader)
        scheduler.step(avg_loss)

        print(f"Average Loss: {avg_loss}")
        print(f"Accuracy: {accuracy / len(train_dataset)}")

    writer.close()
    torch.save(model.state_dict(), "CIFAR10_model.pth")

def test():
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for i, (image, label) in enumerate(test_data_loader):
            image, label = image.to(device), label.to(device)
            images, labels = image, label
            output = model(images)

            result_loss = loss(output, labels)
            writer.add_scalar("test_loss", result_loss.item(), i * len(test_data_loader) + i)

            pred = torch.argmax(output, dim=1)
            accuracy += (pred == labels).sum().item()

        writer.add_scalar("test_accuracy", accuracy / len(test_dataset), i * len(test_data_loader) + i)
        print(f"Test Accuracy: {accuracy / len(test_dataset)}")

def show():
    class_name = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    fig, ax = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            image, label = test_dataset[random.randint(0, len(test_dataset) - 1)]
            image = image.to(device)
            output = model(image.unsqueeze(0))
            pred = torch.argmax(output, dim=1)

            image = image.cpu().numpy().transpose(1, 2, 0)
            image = (image * std + mean) * 255
            image = np.clip(image, 0, 255).astype(np.uint8)

            ax[i][j].imshow(image)
            ax[i][j].axis('off')
            ax[i][j].set_title(f"Pred: {class_name[pred]} \n Label: {class_name[label]}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train()
    test()
    show()