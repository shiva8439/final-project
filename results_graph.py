"""
Results Graph Generator
Generates training curves and evaluation metrics visualizations
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import json
from typing import Optional, Dict, List


def plot_training_curves(train_losses: List[float], 
                        val_losses: List[float],
                        save_path: str = "results/training_curves.png"):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_path}")


def plot_metrics(metrics_history: Dict[str, List[float]],
                save_path: str = "results/metrics_history.png"):
    """
    Plot evaluation metrics over epochs.
    
    Args:
        metrics_history: Dictionary with metric names as keys and lists of values
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    epochs = range(1, len(list(metrics_history.values())[0]) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metric_names = ['PSNR', 'SSIM', 'MAE', 'MSE']
    colors = ['green', 'blue', 'orange', 'red']
    
    for idx, (metric, color) in enumerate(zip(metric_names, colors)):
        if metric in metrics_history:
            axes[idx].plot(epochs, metrics_history[metric], 
                          color=color, linewidth=2, marker='o', markersize=4)
            axes[idx].set_xlabel('Epoch', fontsize=11)
            axes[idx].set_ylabel(metric, fontsize=11)
            axes[idx].set_title(f'{metric} over Epochs', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics history saved to {save_path}")


def plot_comparison_metrics(metrics: Dict[str, float],
                          save_path: str = "results/metrics_comparison.png"):
    """
    Plot bar chart of final metrics.
    
    Args:
        metrics: Dictionary with metric names and final values
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Filter metrics for comparison
    comparison_metrics = {
        'PSNR (dB)': metrics.get('PSNR', 0),
        'SSIM': metrics.get('SSIM', 0),
        'MAE': metrics.get('MAE', 0),
        'MSE': metrics.get('MSE', 0)
    }
    
    names = list(comparison_metrics.keys())
    values = list(comparison_metrics.values())
    
    colors = ['green', 'blue', 'orange', 'red']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}' if value < 10 else f'{value:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Final Evaluation Metrics', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics comparison saved to {save_path}")


def plot_confidence_distribution(confidence_map: np.ndarray,
                                 save_path: str = "results/confidence_distribution.png"):
    """
    Plot distribution of confidence values.
    
    Args:
        confidence_map: 2D or 3D confidence/uncertainty map
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(confidence_map.flatten(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Uncertainty Value', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Uncertainty Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(confidence_map.flatten(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1].set_ylabel('Uncertainty Value', fontsize=11)
    axes[1].set_title('Uncertainty Box Plot', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confidence distribution saved to {save_path}")


def plot_prediction_comparison(prediction: np.ndarray,
                             ground_truth: np.ndarray,
                             confidence_map: Optional[np.ndarray] = None,
                             save_path: str = "results/prediction_comparison.png"):
    """
    Plot prediction vs ground truth with optional confidence map.
    
    Args:
        prediction: Predicted volume or slice
        ground_truth: Ground truth volume or slice
        confidence_map: Optional confidence/uncertainty map
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Get middle slice for 3D volumes
    if prediction.ndim == 3:
        mid_slice = prediction.shape[2] // 2
        prediction = prediction[:, :, mid_slice]
        ground_truth = ground_truth[:, :, mid_slice]
        if confidence_map is not None:
            confidence_map = confidence_map[:, :, mid_slice]
    
    if confidence_map is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(ground_truth, cmap='gray')
        axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(prediction, cmap='gray')
        axes[1].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        im = axes[2].imshow(confidence_map, cmap='viridis')
        axes[2].set_title('Uncertainty Map', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(ground_truth, cmap='gray')
        axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(prediction, cmap='gray')
        axes[1].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Prediction comparison saved to {save_path}")


def generate_all_results(train_losses: List[float],
                         val_losses: List[float],
                         final_metrics: Dict[str, float],
                         prediction: Optional[np.ndarray] = None,
                         ground_truth: Optional[np.ndarray] = None,
                         confidence_map: Optional[np.ndarray] = None,
                         output_dir: str = "results"):
    """
    Generate all result graphs.
    
    Args:
        train_losses: Training loss history
        val_losses: Validation loss history
        final_metrics: Final evaluation metrics
        prediction: Optional prediction volume/slice
        ground_truth: Optional ground truth volume/slice
        confidence_map: Optional confidence/uncertainty map
        output_dir: Directory to save all graphs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating results graphs...")
    
    # Training curves
    plot_training_curves(train_losses, val_losses, 
                       os.path.join(output_dir, "training_curves.png"))
    
    # Metrics comparison
    plot_comparison_metrics(final_metrics,
                           os.path.join(output_dir, "metrics_comparison.png"))
    
    # Prediction comparison
    if prediction is not None and ground_truth is not None:
        plot_prediction_comparison(prediction, ground_truth, confidence_map,
                                  os.path.join(output_dir, "prediction_comparison.png"))
    
    # Confidence distribution
    if confidence_map is not None:
        plot_confidence_distribution(confidence_map,
                                     os.path.join(output_dir, "confidence_distribution.png"))
    
    print(f"\n✅ All results graphs saved to {output_dir}/")


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Generating example results graphs with synthetic data...")
    
    # Synthetic training data
    epochs = 50
    train_losses = np.exp(-np.linspace(0, 3, epochs)) + np.random.normal(0, 0.01, epochs)
    val_losses = np.exp(-np.linspace(0, 3, epochs)) + np.random.normal(0, 0.02, epochs)
    
    # Synthetic metrics
    final_metrics = {
        'PSNR': 28.5,
        'SSIM': 0.85,
        'MAE': 0.045,
        'MSE': 0.0032
    }
    
    # Synthetic prediction data
    prediction = np.random.rand(240, 240)
    ground_truth = prediction + np.random.normal(0, 0.05, prediction.shape)
    confidence_map = np.random.rand(240, 240) * 0.1
    
    # Generate all graphs
    generate_all_results(
        train_losses.tolist(),
        val_losses.tolist(),
        final_metrics,
        prediction,
        ground_truth,
        confidence_map,
        output_dir="results"
    )
    
    print("\nExample graphs generated. Replace with actual training data for real results.")
