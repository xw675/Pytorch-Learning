import torch
from torch import nn

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    def forward(self, x):
        x = x + 1
        return x

if __name__ == "__main__":
    model = MyModule()
    data = torch.tensor([2, 3])
    print(model(data))