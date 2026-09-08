from brats_dataset import BraTSSliceDataset

dataset = BraTSSliceDataset(
    "dataset/PKG - BraTS-Africa/MICCAI_BraTS2020_TrainingData"
)

print("\nDataset Length:", len(dataset))

image, target = dataset[0]

print("\nInput Shape :", image.shape)
print("Target Shape:", target.shape)

print("\nInput dtype :", image.dtype)
print("Target dtype:", target.dtype)