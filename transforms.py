from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

writer = SummaryWriter("logs")
image_path = 'Dataset/hymenoptera/train/bees_image/16838648_415acd9e3f.jpg'
image = Image.open(image_path)
# print(image)

# ToTensor
image_tensor = transforms.ToTensor()
# print(image_tensor(image), "\n", image_tensor(image).shape)  # torch.Size([3, 224, 224])

# Normalize
image_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(image_tensor(image))
# print(image_norm(image_tensor(image)), "\n", image_norm(image_tensor(image)).shape)  # torch.Size([3, 224, 224])

# Resize
resize_transform = transforms.Resize((256, 256))
image_resize = resize_transform(image)
image_resize_tensor = transforms.ToTensor()
# print(image_resize_tensor(image_resize), "\n", image_resize_tensor(image_resize).shape)  # torch.Size([3, 256, 256])

# Compose
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),                                    # Resize the image to 256x256
    transforms.ToTensor(),                                            # Convert the image to a PyTorch tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize the tensor
])
image_transform_tensor = image_transform(image)

# RandomCrop
trans_random = transforms.RandomCrop(96)
image_random = transforms.Compose([
    trans_random,
    transforms.ToTensor(),
])

for i in range(10):
    writer.add_image('Image Random', image_random(image), i)
writer.close()