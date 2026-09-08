import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def calculate_mae(pred, target):
    """Mean Absolute Error"""
    return torch.mean(torch.abs(pred - target)).item()


def calculate_mse(pred, target):
    """Mean Squared Error"""
    return torch.mean((pred - target) ** 2).item()


def calculate_psnr(pred, target):
    """Peak Signal-to-Noise Ratio"""
    pred = pred.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()

    return peak_signal_noise_ratio(
        target,
        pred,
        data_range=target.max() - target.min()
    )


def calculate_ssim(pred, target):
    """Structural Similarity Index"""
    pred = pred.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()

    return structural_similarity(
        target,
        pred,
        data_range=target.max() - target.min()
    )


def evaluate_prediction(pred, target):
    """Return all metrics"""

    metrics = {
        "MAE": calculate_mae(pred, target),
        "MSE": calculate_mse(pred, target),
        "PSNR": calculate_psnr(pred, target),
        "SSIM": calculate_ssim(pred, target),
    }

    return metrics