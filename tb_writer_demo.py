from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import os

writer = SummaryWriter('logs')
folder_path = 'D:/Projects/Python/Pytorch Learning/Dataset/hymenoptera/train/ants_image'

for step in enumerate(os.listdir(folder_path)):
    if step[1].endswith('.jpg'):
        image_path = os.path.join(folder_path, step[1])
        image_array = np.array(Image.open(image_path))
        writer.add_image('ants', image_array, step[0] + 1, dataformats='HWC')

# for i in range(100):
#     writer.add_scalar('y=x^2', i**2, i)
#     writer.add_scalar('y=x^3', i**3, i)
#
writer.close()