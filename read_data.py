from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self, root_dir: str, label_dir: str):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.image_path = os.listdir(self.path)

    def __getitem__(self, index):
        image_name = self.image_path[index]
        image_item_path = os.path.join(self.path, image_name)
        image = Image.open(image_item_path)
        label = self.label_dir
        return image, label

    def __len__(self):
        return len(self.image_path)

ants_dataset = MyData('D:/Projects/Python/Pytorch Learning/Dataset/hymenoptera/train', 'ants_image')
bees_dataset = MyData('D:/Projects/Python/Pytorch Learning/Dataset/hymenoptera/train', 'bees_image')
train_dataset = ants_dataset + bees_dataset

for i in range(len(train_dataset)):
    image, label = train_dataset[i]
    print(f"Image: {image}, Label: {label.split('_')[0]}")

