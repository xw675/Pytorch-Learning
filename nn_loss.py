import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

input = torch.randn([1, 3, 32, 32])
output =torch.randn([1, 3, 32, 32])

loss_mae = nn.L1Loss(reduction='mean')
print(loss_mae(input, output))

loss_mse = nn.MSELoss(reduction='mean')
print(loss_mse(input, output))

x = torch.tensor([0.1, 0.5, 0.4])
y = torch.tensor([0., 1., 0.])

loss_cross_entropy = nn.CrossEntropyLoss(reduction='mean')
print(loss_cross_entropy(x, y))