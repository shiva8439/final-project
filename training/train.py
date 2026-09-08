import sys
import os
import time

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from brats_dataset import BraTSSliceDataset
from models.attention_unet import VirtualContrastModel
from losses.hybrid_loss import HybridLoss


def main():

    DATASET_PATH = "dataset/PKG - BraTS-Africa/MICCAI_BraTS2020_TrainingData"

    # -----------------------------
    # Dataset
    # -----------------------------
    dataset = BraTSSliceDataset(DATASET_PATH)

    # Use smaller subset for faster training
    subset_size = min(2000, len(dataset))
    dataset = torch.utils.data.Subset(dataset, range(subset_size))
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Larger batch size for faster training
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print("Train Samples:", len(train_dataset))
    print("Validation Samples:", len(val_dataset))

    # -----------------------------
    # Device
    # -----------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # -----------------------------
    # Model
    # -----------------------------
    model = VirtualContrastModel().to(device)

    # -----------------------------
    # Loss
    # -----------------------------
    criterion = HybridLoss().to(device)

    # -----------------------------
    # Optimizer
    # -----------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )

    # Mixed Precision
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    print("Model Loaded Successfully")
    print("Mixed Precision:", device.type == "cuda")

    # -----------------------------
    # Training
    # -----------------------------
    EPOCHS = 5  # Reduced for faster training
    best_loss = float("inf")

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):

        print("\n" + "=" * 60)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("=" * 60)

        model.train()

        train_loss = 0.0

        progress = tqdm(
            train_loader,
            desc="Training",
            unit="batch"
        )

        start_time = time.time()

        for images, targets in progress:

            images = images.to(
                device,
                non_blocking=True
            )

            targets = targets.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):

                outputs = model(images)

                loss, l1, ssim, edge = criterion(
                    outputs,
                    targets
                )

            # Backpropagation
            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            train_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss /= len(train_loader)

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            progress_val = tqdm(
                val_loader,
                desc="Validation",
                unit="batch"
            )

            for images, targets in progress_val:

                images = images.to(
                    device,
                    non_blocking=True
                )

                targets = targets.to(
                    device,
                    non_blocking=True
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=(device.type == "cuda")
                ):

                    outputs = model(images)

                    loss, _, _, _ = criterion(
                        outputs,
                        targets
                    )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        elapsed = time.time() - start_time

        print("\nEpoch Result")
        print("-----------------------------")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Time       : {elapsed / 60:.2f} minutes")

        # -----------------------------
        # Save Best Model
        # -----------------------------
        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                "checkpoints/best_model.pth"
            )

            print("✅ Best Model Saved")

        # Save latest checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            "checkpoints/latest_model.pth"
        )

    print("\n🎉 TRAINING COMPLETE!")
    print("Best Validation Loss:", best_loss)


if __name__ == "__main__":
    main()