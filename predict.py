import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from models.attention_unet import VirtualContrastModel
from dataset import get_dataset
from transforms import train_transforms


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    model = VirtualContrastModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle checkpoint dictionary
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def predict_single_slice(model, t1, t2, flair, device, use_mixed_precision=True):
    """
    Predict T1CE from T1, T2, FLAIR slices.
    
    Args:
        model: Trained VirtualContrastModel
        t1: T1 slice (numpy array or tensor)
        t2: T2 slice (numpy array or tensor)
        flair: FLAIR slice (numpy array or tensor)
        device: torch device
        use_mixed_precision: Whether to use mixed precision
        
    Returns:
        predicted_t1ce: Predicted T1CE slice
    """
    # Convert to tensor if numpy
    if isinstance(t1, np.ndarray):
        t1 = torch.from_numpy(t1).float()
    if isinstance(t2, np.ndarray):
        t2 = torch.from_numpy(t2).float()
    if isinstance(flair, np.ndarray):
        flair = torch.from_numpy(flair).float()
    
    # Stack channels: [3, H, W]
    input_tensor = torch.stack([t1, t2, flair], dim=0).unsqueeze(0).to(device)
    
    # Normalize input
    input_tensor = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min() + 1e-8)
    
    # Predict with mixed precision
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == "cuda" and use_mixed_precision):
            output = model(input_tensor)
    
    # Remove batch dimension and channel: [H, W]
    predicted_t1ce = output.squeeze().cpu().numpy()
    
    return predicted_t1ce


def predict_patient(model, patient_data, device, use_batch=True, use_mixed_precision=True, save_path=None):
    """
    Predict T1CE for entire patient volume.
    
    Args:
        model: Trained VirtualContrastModel
        patient_data: Dictionary with patient paths
        device: torch device
        use_batch: Whether to use batch prediction (10x faster)
        use_mixed_precision: Whether to use mixed precision
        save_path: Optional path to save prediction
        
    Returns:
        predicted_volume: 3D numpy array of predicted T1CE
    """
    # Load and transform patient data
    data = train_transforms(patient_data)
    
    # Extract volumes
    t1_vol = data["t1"][0]  # [H, W, D]
    t2_vol = data["t2"][0]
    flair_vol = data["flair"][0]
    
    depth = t1_vol.shape[-1]
    
    if use_batch:
        # Batch prediction: process all slices at once (10x faster)
        # Stack all slices: [D, 3, H, W]
        batch_input = torch.stack([
            t1_vol.permute(2, 0, 1),  # [D, H, W]
            t2_vol.permute(2, 0, 1),
            flair_vol.permute(2, 0, 1)
        ], dim=1).to(device)  # [D, 3, H, W]
        
        # Normalize
        batch_input = (batch_input - batch_input.min()) / (batch_input.max() - batch_input.min() + 1e-8)
        
        # Predict with mixed precision
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=device.type == "cuda" and use_mixed_precision):
                output = model(batch_input.permute(1, 0, 2, 3))  # [3, D, H, W] -> [1, 3, D, H, W]
        
        # Reshape back to [H, W, D]
        predicted_volume = output.squeeze().permute(1, 2, 0).cpu().numpy()
    else:
        # Slice-by-slice prediction (fallback)
        predicted_volume = np.zeros((t1_vol.shape[0], t1_vol.shape[1], depth))
        
        for i in range(depth):
            predicted_volume[:, :, i] = predict_single_slice(
                model, 
                t1_vol[:, :, i].numpy(), 
                t2_vol[:, :, i].numpy(), 
                flair_vol[:, :, i].numpy(), 
                device,
                use_mixed_precision
            )
    
    # Auto-save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, predicted_volume)
        print(f"Prediction saved to {save_path}")
    
    return predicted_volume


def predict_with_confidence(model, patient_data, device, n_samples=10, use_batch=True, use_mixed_precision=True):
    """
    Predict T1CE with confidence map using Monte Carlo Dropout.
    
    Args:
        model: Trained VirtualContrastModel
        patient_data: Dictionary with patient paths
        device: torch device
        n_samples: Number of forward passes for uncertainty estimation
        use_batch: Whether to use batch prediction
        use_mixed_precision: Whether to use mixed precision
        
    Returns:
        predicted_volume: Mean prediction
        confidence_map: Uncertainty map (standard deviation)
    """
    # Load and transform patient data
    data = train_transforms(patient_data)
    
    # Extract volumes
    t1_vol = data["t1"][0]
    t2_vol = data["t2"][0]
    flair_vol = data["flair"][0]
    
    depth = t1_vol.shape[-1]
    predictions = []
    
    # Enable dropout for MC Dropout
    model.train()
    
    # Multiple forward passes for uncertainty estimation
    for _ in range(n_samples):
        if use_batch:
            batch_input = torch.stack([
                t1_vol.permute(2, 0, 1),
                t2_vol.permute(2, 0, 1),
                flair_vol.permute(2, 0, 1)
            ], dim=1).to(device)
            
            batch_input = (batch_input - batch_input.min()) / (batch_input.max() - batch_input.min() + 1e-8)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=device.type == "cuda" and use_mixed_precision):
                    output = model(batch_input.permute(1, 0, 2, 3))
            
            pred = output.squeeze().permute(1, 2, 0).cpu().numpy()
        else:
            pred = predict_patient(model, patient_data, device, use_batch=False, use_mixed_precision=use_mixed_precision)
        
        predictions.append(pred)
    
    # Calculate mean and variance
    predictions = np.array(predictions)  # [n_samples, H, W, D]
    predicted_volume = np.mean(predictions, axis=0)
    confidence_map = np.std(predictions, axis=0)
    
    # Set model back to eval mode
    model.eval()
    
    return predicted_volume, confidence_map
def batch_predict(model, dataset, device, max_patients=None, use_batch=True, use_mixed_precision=True, save_dir=None):
    """
    Predict T1CE for multiple patients.
    
    Args:
        model: Trained VirtualContrastModel
        dataset: List of patient dictionaries
        device: torch device
        max_patients: Maximum number of patients to process (None for all)
        use_batch: Whether to use batch prediction
        use_mixed_precision: Whether to use mixed precision
        save_dir: Directory to save predictions (optional)
        
    Returns:
        predictions: Dictionary mapping patient_id to predicted volume
    """
    predictions = {}
    
    if max_patients:
        dataset = dataset[:max_patients]
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    for patient in dataset:
        patient_id = patient["id"]
        print(f"Predicting for {patient_id}...")
        
        save_path = os.path.join(save_dir, f"{patient_id}_prediction.npy") if save_dir else None
        predicted_volume = predict_patient(model, patient, device, use_batch, use_mixed_precision, save_path)
        predictions[patient_id] = predicted_volume
    
    return predictions


def visualize_prediction(predicted_volume, ground_truth=None, save_path=None):
    """
    Visualize prediction with middle slice.
    
    Args:
        predicted_volume: Predicted 3D volume
        ground_truth: Optional ground truth volume
        save_path: Optional path to save visualization
    """
    middle_slice = predicted_volume.shape[2] // 2
    
    fig, axes = plt.subplots(1, 2 if ground_truth is not None else 1, figsize=(12, 6))
    
    if ground_truth is not None:
        axes = axes.flatten()
        axes[0].imshow(ground_truth[:, :, middle_slice], cmap='gray')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        axes[1].imshow(predicted_volume[:, :, middle_slice], cmap='gray')
        axes[1].set_title('Prediction')
        axes[1].axis('off')
    else:
        axes.imshow(predicted_volume[:, :, middle_slice], cmap='gray')
        axes.set_title('Prediction')
        axes.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model("best_model.pth", device)
    
    # Load dataset
    dataset = get_dataset("dataset/PKG - BraTS-Africa/MICCAI_BraTS2020_TrainingData")
    
    # Predict for first patient with batch prediction (10x faster)
    patient_data = dataset[0]
    print(f"Predicting for {patient_data['id']}...")
    
    predicted_volume = predict_patient(
        model, 
        patient_data, 
        device, 
        use_batch=True, 
        use_mixed_precision=True,
        save_path="predictions/prediction.npy"
    )
    
    print(f"Predicted volume shape: {predicted_volume.shape}")
    
    # Visualize
    visualize_prediction(predicted_volume, save_path="predictions/visualization.png")
    
    # Predict with confidence (research gap)
    print("\nGenerating confidence map...")
    pred_with_conf, confidence_map = predict_with_confidence(
        model, 
        patient_data, 
        device, 
        n_samples=10,
        use_batch=True,
        use_mixed_precision=True
    )
    
    print(f"Confidence map shape: {confidence_map.shape}")
    np.save("predictions/confidence_map.npy", confidence_map)
    
    # Visualize confidence
    middle_slice = confidence_map.shape[2] // 2
    plt.figure(figsize=(8, 8))
    plt.imshow(confidence_map[:, :, middle_slice], cmap='viridis')
    plt.title('Confidence Map (Uncertainty)')
    plt.colorbar()
    plt.axis('off')
    plt.savefig("predictions/confidence_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Prediction complete with confidence map!")
