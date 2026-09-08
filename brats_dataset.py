import torch
import sys
import os
from torch.utils.data import Dataset
from dataset import get_dataset

# Add parent directory to path for transforms import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from transforms import train_transforms


class BraTSSliceDataset(Dataset):

    def __init__(self, root_dir):
        self.patients = get_dataset(root_dir)
        self.transforms = train_transforms
        
        # Calculate total slices without loading data
        self.slice_indices = []
        for patient_idx, patient in enumerate(self.patients):
            # Assume standard BraTS depth of 155
            for slice_idx in range(40, 120):
                self.slice_indices.append((patient_idx, slice_idx))
        
        print("Total patients:", len(self.patients))
        print("Total slices:", len(self.slice_indices))

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx):
        patient_idx, slice_idx = self.slice_indices[idx]
        patient = self.patients[patient_idx]
        
        # Load and transform only the requested patient
        data = self.transforms(patient)
        
        # Extract the specific slice
        image = torch.stack([
            data["t1"][0, :, :, slice_idx],
            data["t2"][0, :, :, slice_idx],
            data["flair"][0, :, :, slice_idx]
        ], dim=0)
        
        target = data["t1ce"][0, :, :, slice_idx].unsqueeze(0)
        
        return image.float(), target.float()