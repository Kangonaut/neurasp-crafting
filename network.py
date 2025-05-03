import torch
from torch import Tensor, nn


# img input size: 3x224x224
class ItemClassifier(nn.Module):
    def __init__(self, num_classes: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cnn = nn.Sequential(
            # (3,32,32)
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
            ),  # (16,30,30)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
            ),  # (32,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (32,14,14)
        )
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=32 * 14 * 14,
                out_features=num_classes,
            ),
            nn.Softmax(1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
