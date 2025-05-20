import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from CIFAR10_model import MyCIFAR10
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                                            torchvision.transforms.RandomCrop(32, padding=4),
                                            torchvision.transforms.RandomHorizontalFlip(),
                                            ])
train_dataset = torchvision.datasets.CIFAR10(root='./Dataset', train=True, transform=transform, download=False)
test_dataset = torchvision.datasets.CIFAR10(root='./Dataset', train=False, transform=transform, download=False)
train_data_loader = DataLoader(train_dataset, batch_size=64)
test_data_loader = DataLoader(test_dataset, batch_size=64)

model = MyCIFAR10().to(device)
# model.load_state_dict(torch.load("CIFAR10_model.pth"))
loss = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter("logs")

def train():
    model.train()
    for epoch in range(10):
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
            writer.add_scalar("final_train_loss", result_loss.item(), epoch * len(train_data_loader) + i)

        print(f"Average Loss: {train_loss / len(train_data_loader)}")

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
            writer.add_scalar("final_test_loss", result_loss.item(), i * len(test_data_loader) + i)

            pred = torch.argmax(output, dim=1)
            accuracy += (pred == labels).sum().item()

        writer.add_scalar("final_test_accuracy", accuracy / len(test_dataset), i * len(test_data_loader) + i)
        print(f"Accuracy: {accuracy / len(test_dataset)}")

if __name__ == "__main__":
    train()
    test()