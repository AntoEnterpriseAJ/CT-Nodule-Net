from torch import Tensor, nn


class NoduleBlock(nn.Module):
    def __init__(self, in_channels: int, conv_channels: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_channels, out_channels=conv_channels, kernel_size=3, padding=1
        )
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(
            in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding=1
        )
        self.relu2 = nn.ReLU()

        self.pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x: Tensor) -> Tensor:
        """x = (nBatch, nChannels, height, width, depth)."""
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)

        return x
