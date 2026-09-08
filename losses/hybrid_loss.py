import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import SSIMLoss


class EdgeLoss(nn.Module):

    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred, target):
        # Explicitly perform convolution in float32 to avoid HalfTensor issues with F.conv2d
        # Inputs 'pred' and 'target' are likely HalfTensor due to autocast.
        # The sobel kernels are already float32. Cast inputs to float32 for F.conv2d.

        sobel_x_float32 = self.sobel_x  # Already float32
        sobel_y_float32 = self.sobel_y  # Already float32

        pred_float32 = pred.to(torch.float32)

        pred_x = F.conv2d(
            pred_float32,
            sobel_x_float32,
            padding=1
        )

        pred_y = F.conv2d(
            pred_float32,
            sobel_y_float32,
            padding=1
        )

        target_float32 = target.to(torch.float32)

        target_x = F.conv2d(
            target_float32,
            sobel_x_float32,
            padding=1
        )

        target_y = F.conv2d(
            target_float32,
            sobel_y_float32,
            padding=1
        )

        # The outputs of F.conv2d (pred_x, pred_y, target_x, target_y) will be float32.
        # F.l1_loss can operate with float32 inputs, so no further casting needed here.
        edge_loss_x = F.l1_loss(
            pred_x,
            target_x
        )

        edge_loss_y = F.l1_loss(
            pred_y,
            target_y
        )

        return edge_loss_x + edge_loss_y


class HybridLoss(nn.Module):

    def __init__(
        self,
        lambda_ssim=0.5,
        lambda_edge=0.2
    ):
        super().__init__()

        self.l1 = nn.L1Loss()

        self.ssim = SSIMLoss(
            spatial_dims=2,
            win_size=7
        )

        self.edge = EdgeLoss()

        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge

    def forward(self, pred, target):

        l1 = self.l1(
            pred,
            target
        )

        ssim = self.ssim(
            pred,
            target
        )

        edge = self.edge(
            pred,
            target
        )

        total = (
            l1
            + self.lambda_ssim * ssim
            + self.lambda_edge * edge
        )

        return total, l1, ssim, edge
