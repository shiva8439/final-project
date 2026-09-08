import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet


class VirtualContrastModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = AttentionUnet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )

    def forward(self, x):
        return self.model(x)