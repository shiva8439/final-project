import torch
import numpy as np


def enable_dropout(model):
    """
    Enable dropout layers during inference.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout) or isinstance(m, torch.nn.Dropout2d):
            m.train()


def generate_confidence_map(
    model,
    input_tensor,
    device,
    n_samples=10
):
    """
    Monte Carlo Dropout
    """

    model.eval()
    enable_dropout(model)

    predictions = []

    with torch.no_grad():

        for _ in range(n_samples):

            output = model(input_tensor.to(device))

            predictions.append(
                output.cpu().numpy()
            )

    predictions = np.array(predictions)

    mean_prediction = predictions.mean(axis=0)

    uncertainty = predictions.std(axis=0)

    return mean_prediction, uncertainty