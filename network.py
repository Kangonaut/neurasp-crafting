import torch
import torchvision
from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock, _resnet


# img input size: 3x224x224
class ItemClassifier(nn.Module):
    def __init__(self, num_classes: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cnn = nn.Sequential(
            # (3,224,224)
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
            ),  # (16,222,222)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
            ),  # (32,220,220)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (32,110,110)
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
            ),  # (32,108,108)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32,54,54)
        )
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=32 * 54 * 54,
                out_features=512,
            ),
            nn.Linear(
                in_features=512,
                out_features=num_classes,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
