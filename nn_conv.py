import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([[1, 2, 0, 3, 1],
                  [0, 1, 2, 3, 1],
                  [1, 2, 1, 0, 0],
                  [5, 2, 3, 1, 1],
                  [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

x_reshaped = torch.reshape(x, (1, 1, 5, 5))
kernel_reshaped = torch.reshape(kernel, (1, 1, 3, 3))

# print(x_reshaped.shape)
# print(kernel_reshaped.shape)

output = F.conv2d(x_reshaped, kernel_reshaped, stride=1)
print(output)

output_2 = F.conv2d(x_reshaped, kernel_reshaped, stride=2)
print(output_2)

output_3 = F.conv2d(x_reshaped, kernel_reshaped, stride=1, padding=1)
print(output_3)

output_4 = F.conv2d(x_reshaped, kernel_reshaped, stride=2, padding=1)
print(output_4)