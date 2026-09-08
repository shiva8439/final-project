import nibabel as nib
import numpy as np
import torch
from typing import Dict

def train_transforms(patient_data: Dict) -> Dict:
    """
    Load and transform patient MRI data for training/inference.
    
    Args:
        patient_data: Dictionary with patient file paths
            - t1: path to T1 MRI
            - t2: path to T2 MRI
            - flair: path to FLAIR MRI
            - t1ce: path to T1CE MRI (optional)
    
    Returns:
        Dictionary with loaded volumes as tensors
    """
    def load_volume(path):
        """Load NIfTI volume and return as tensor."""
        img = nib.load(path)
        data = img.get_fdata()
        
        # Normalize to [0, 1]
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        # Convert to tensor and add channel dimension: [1, H, W, D]
        tensor = torch.from_numpy(data).float().unsqueeze(0)
        
        return tensor
    
    result = {
        "t1": load_volume(patient_data["t1"]),
        "t2": load_volume(patient_data["t2"]),
        "flair": load_volume(patient_data["flair"])
    }
    
    # Load T1CE if available
    if "t1ce" in patient_data and patient_data["t1ce"]:
        result["t1ce"] = load_volume(patient_data["t1ce"])
    
    return result
