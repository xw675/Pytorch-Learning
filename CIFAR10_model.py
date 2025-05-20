import torch
import torch.nn as nn

class MyCIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),

            nn.Conv2d(32, 32, 3, 1, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, 1, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding="same"),
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),

            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":
    model = MyCIFAR10()
    print(model)
    input = torch.randn(64, 3, 32, 32)
    output = model(input)
    print(output.shape)