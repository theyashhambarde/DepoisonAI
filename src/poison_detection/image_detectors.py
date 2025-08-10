import torch
import numpy as np
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
from utils.helpers import LOGGER


def get_activations(model, dataloader, device):
    """Extracts penultimate layer activations from a model for a given dataset."""
    model.eval()
    activations = []

    # Hook to capture activations
    def hook(module, input, output):
        activations.append(output.detach().cpu().numpy())

    # Find the penultimate layer (usually before the final classifier)
    handle = None
    if hasattr(model, 'fc'):  # ResNet, etc.
        handle = model.fc.register_forward_pre_hook(hook)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.Sequential):  # VGG
        handle = model.classifier[-1].register_forward_pre_hook(hook)
    else:
        raise TypeError("Could not find a suitable layer to hook for activations.")

    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Extracting Activations"):
            inputs = inputs.to(device)
            _ = model(inputs)

    handle.remove()  # Clean up hook
    return np.concatenate(activations, axis=0).reshape(len(dataloader.dataset), -1)


def detect_backdoor_by_activation_clustering(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                                             device: str, contamination: float = 0.05) -> np.ndarray:
    """
    Detects backdoor samples by finding outliers in the activation space of a model.
    Backdoored samples often cluster together in the latent space.

    Args:
        model (torch.nn.Module): A trained (ideally on clean data) neural network.
        dataloader (DataLoader): DataLoader for the dataset to be scanned.
        device (str): 'cuda' or 'cpu'.
        contamination (float): Estimated proportion of poisoned data.

    Returns:
        np.ndarray: Indices of suspected poisoned samples.
    """
    LOGGER.info("Detecting backdoors via activation space clustering...")
    model.to(device)

    # 1. Extract activations
    activations = get_activations(model, dataloader, device)
    LOGGER.info(f"Extracted activations of shape: {activations.shape}")
                                                 

    # 2. Use Isolation Forest to find outliers in the activation space
    iforest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    predictions = iforest.fit_predict(activations)

    outlier_indices = np.where(predictions == -1)[0]
    LOGGER.info(f"Identified {len(outlier_indices)} potential backdoored samples via activation clustering.")
    return outlier_indices
