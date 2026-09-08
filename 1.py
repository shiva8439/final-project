import os
import sys

# Add project root path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)


import torch
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

from brats_dataset import BraTSSliceDataset
from models.attention_unet import VirtualContrastModel
from losses.hybrid_loss import HybridLoss



# =========================
# CONFIGURATION
# =========================

DATASET_PATH = "dataset/PKG - BraTS-Africa/MICCAI_BraTS2020_TrainingData"

BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4

MODEL_PATH = "checkpoints/best_model.pth"



# =========================
# TRAIN FUNCTION
# =========================

def train_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
        device,
        scaler
):

    model.train()

    running_loss = 0


    for images, targets in loader:


        images = images.to(device)
        targets = targets.to(device)


        optimizer.zero_grad()



        # Mixed precision
        with torch.cuda.amp.autocast():

            outputs = model(images)

            loss, l1, ssim, edge = criterion(
                outputs,
                targets
            )



        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()



        running_loss += loss.item()



    return running_loss / len(loader)




# =========================
# VALIDATION FUNCTION
# =========================

@torch.no_grad()
def validate(
        model,
        loader,
        criterion,
        device
):

    model.eval()

    total_loss = 0



    for images, targets in loader:


        images = images.to(device)
        targets = targets.to(device)


        outputs = model(images)


        loss, _, _, _ = criterion(
            outputs,
            targets
        )


        total_loss += loss.item()



    return total_loss / len(loader)




# =========================
# MAIN
# =========================


def main():



    print("Loading Dataset...")


    dataset = BraTSSliceDataset(
        DATASET_PATH
    )



    # Train Validation Split

    train_size = int(
        0.8 * len(dataset)
    )

    val_size = len(dataset)-train_size



    train_dataset, val_dataset = random_split(

        dataset,

        [
            train_size,
            val_size
        ],

        generator=torch.Generator().manual_seed(42)

    )



    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0

    )



    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0

    )



    print(
        "Training Samples:",
        len(train_dataset)
    )

    print(
        "Validation Samples:",
        len(val_dataset)
    )



    # Device

    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"

    )


    print(
        "Using Device:",
        device
    )



    # Model

    model = VirtualContrastModel()

    model = model.to(device)



    # Loss

    criterion = HybridLoss()



    # Optimizer

    optimizer = optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=1e-5

    )



    # Scheduler

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        patience=5,

        factor=0.5

    )



    # AMP

    scaler = torch.cuda.amp.GradScaler()



    # Create checkpoint folder

    os.makedirs(

        "checkpoints",

        exist_ok=True

    )



    best_loss = float("inf")



    print("\nTraining Started...\n")



    for epoch in range(EPOCHS):


        train_loss = train_one_epoch(

            model,

            train_loader,

            optimizer,

            criterion,

            device,

            scaler

        )



        val_loss = validate(

            model,

            val_loader,

            criterion,

            device

        )



        scheduler.step(val_loss)



        print(
            f"""
Epoch [{epoch+1}/{EPOCHS}]

Train Loss : {train_loss:.5f}

Val Loss   : {val_loss:.5f}

Learning Rate : {optimizer.param_groups[0]['lr']}

"""
        )



        # Save best model

        if val_loss < best_loss:


            best_loss = val_loss


            torch.save(

                {

                    "epoch": epoch+1,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "loss":
                        best_loss

                },

                MODEL_PATH

            )


            print(
                "✅ Best Model Saved"
            )



    print("\nTraining Completed")




if __name__ == "__main__":

    main()