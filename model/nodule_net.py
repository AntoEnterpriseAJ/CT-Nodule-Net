from torch import Tensor, nn

from model.nodule_block import NoduleBlock


class NoduleNet(nn.Module):
    def __init__(self, in_channels: int = 1, conv_channels: int = 8) -> None:
        super().__init__()

        # This acts as a normalization layer, TODO: norm after each conv ?
        self.tail = nn.BatchNorm3d(num_features=in_channels)

        self.backbone = nn.Sequential(
            NoduleBlock(in_channels=in_channels, conv_channels=conv_channels),
            NoduleBlock(in_channels=conv_channels, conv_channels=conv_channels * 2),
            NoduleBlock(in_channels=conv_channels * 2, conv_channels=conv_channels * 4),
            NoduleBlock(in_channels=conv_channels * 4, conv_channels=conv_channels * 8),
        )

        self.head = nn.Linear(in_features=1152, out_features=2)

    def forward(self, x: Tensor) -> Tensor:
        """Tensor dim: (nBatch, nChannels, height, width, depth)."""
        x = self.tail(x)
        x = self.backbone(x)

        x_flatten = x.view(x.shape[0], -1)

        x = self.head(x_flatten)

        return x
