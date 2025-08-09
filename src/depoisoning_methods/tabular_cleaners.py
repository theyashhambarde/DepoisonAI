import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from ..utils.helpers import LOGGER  # <-- This line is now fixed


def filter_samples(X: pd.DataFrame, y: pd.Series, outlier_indices: np.ndarray) -> (pd.DataFrame, pd.Series):
    """
    Removes identified outlier samples from the dataset.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Label Series.
        outlier_indices (np.ndarray): Indices of samples to remove.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Cleaned features and labels.
    """
    LOGGER.info(f"Filtering {len(outlier_indices)} identified malicious samples.")
    X_cleaned = X.drop(index=outlier_indices).reset_index(drop=True)
    y_cleaned = y.drop(index=outlier_indices).reset_index(drop=True)
    LOGGER.info(f"Dataset size after filtering: {len(X_cleaned)} samples.")
    return X_cleaned, y_cleaned


def relabel_with_knn(X: np.ndarray, y: np.ndarray, suspicious_indices: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Corrects labels of suspicious samples using k-Nearest Neighbors.
    It trains a k-NN on the 'clean' data and predicts new labels for the suspicious points.

    Args:
        X (np.ndarray): Full feature data.
        y (np.ndarray): Full (potentially poisoned) label data.
        suspicious_indices (np.ndarray): Indices of samples to relabel.
        k (int): Number of neighbors to use for voting.

    Returns:
        np.ndarray: A new label array with corrected labels.
    """
    LOGGER.info(f"Attempting to relabel {len(suspicious_indices)} suspicious samples using k-NN (k={k}).")
    y_corrected = np.copy(y)

    # Identify clean data for training the k-NN
    clean_mask = np.ones(len(y), dtype=bool)
    clean_mask[suspicious_indices] = False

    X_clean = X[clean_mask]
    y_clean = y[clean_mask]

    X_suspicious = X[suspicious_indices]

    if len(X_clean) == 0:
        LOGGER.warning("No clean data available to train k-NN. Skipping relabeling.")
        return y_corrected

    # Train k-NN on trusted data
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_clean, y_clean)

    # Predict new labels for suspicious data
    new_labels = knn.predict(X_suspicious)

    # Update the labels
    original_labels = y_corrected[suspicious_indices]
    y_corrected[suspicious_indices] = new_labels

    num_changed = np.sum(original_labels != new_labels)
    LOGGER.info(f"Relabeling complete. {num_changed} out of {len(suspicious_indices)} labels were changed.")

    return y_corrected