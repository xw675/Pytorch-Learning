from PIL import Image
import torchvision
from CIFAR10_model import MyCIFAR10
import torch

image_path = "C:/Users/xw112/Pictures/Screenshots/Screenshot 2025-05-20 232151.png"
image = Image.open(image_path).convert("RGB")
# image.show()

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                             torchvision.transforms.ToTensor(),])
image_tensor = transforms(image)
# print(image_tensor.shape)

model = MyCIFAR10()
model.load_state_dict(torch.load("CIFAR10_model.pth"))
model.eval()
image_tensor = image_tensor.reshape(1, 3, 32, 32)
with torch.no_grad():
    output = model(image_tensor)
    pred = torch.argmax(output, dim=1)
    print(image_tensor)
    print(output)
    print(pred.item())