import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(root='./Dataset', train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# image, label = test_set[0]
# print(f"Image: {image}, Label: {label}")
# print(image.shape)

writer = SummaryWriter("logs")
for epoch in range(2):
    for i, (image, label) in enumerate(test_loader):
        images, labels = image, label
        print(f"Batch {i}:")
        print(f"Images: {images.shape}, Labels: {labels}")

        writer.add_images('Epoch: {}'.format(epoch), images, i)

writer.close()
