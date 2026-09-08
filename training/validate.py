import torch
from torch.utils.data import DataLoader

from models.attention_unet import VirtualContrastModel
from brats_dataset import BraTSSliceDataset
from evaluation.metrics import evaluate_prediction

# --------------------------
# Settings
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "checkpoints/best_model.pth"
DATASET_PATH = "dataset/PKG - BraTS-Africa/MICCAI_BraTS2020_TrainingData"

BATCH_SIZE = 8

# --------------------------
# Dataset
# --------------------------

dataset = BraTSSliceDataset(DATASET_PATH)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Validation Samples:", len(dataset))

# --------------------------
# Model
# --------------------------

model = VirtualContrastModel().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

print("Model Loaded Successfully")

# --------------------------
# Validation
# --------------------------

total_mae = 0
total_mse = 0
total_psnr = 0
total_ssim = 0

count = 0

with torch.no_grad():

    for images, targets in loader:

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(images)

        result = evaluate_prediction(outputs, targets)

        total_mae += result["MAE"]
        total_mse += result["MSE"]
        total_psnr += result["PSNR"]
        total_ssim += result["SSIM"]

        count += 1

print("\n========== FINAL RESULTS ==========")

print(f"MAE  : {total_mae/count:.4f}")
print(f"MSE  : {total_mse/count:.4f}")
print(f"PSNR : {total_psnr/count:.2f} dB")
print(f"SSIM : {total_ssim/count:.4f}")