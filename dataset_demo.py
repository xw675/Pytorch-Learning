import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root='./Dataset', train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./Dataset', train=False, transform=dataset_transform, download=True)

# print(f"Test set: {len(test_set)}")
# print(test_set[0])
# print(test_set.classes)
# image, label = test_set[0]
# print(f"Image: {image}, Label: {label}")
# image.show()

writer = SummaryWriter("logs")
for i in range(10):
    image, label = test_set[i]
    writer.add_image('test_set', image, i)
writer.close()