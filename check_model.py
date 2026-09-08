import torch

checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')

print('Model Checkpoint Info:')
print(f'Epoch: {checkpoint["epoch"]}')
print(f'Validation Loss: {checkpoint["val_loss"]:.4f}')

total_params = sum(p.numel() for p in checkpoint["model_state_dict"].values())
print(f'Total Parameters: {total_params:,}')

print('\nModel Layers:')
for key in list(checkpoint["model_state_dict"].keys())[:10]:
    print(f'  {key}')
