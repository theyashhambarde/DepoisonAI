import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from ..utils.helpers import LOGGER

def detect_feature_outliers_iforest(X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
    """
    Detects feature-space outliers using Isolation Forest.
    """
    LOGGER.info(f"Running Isolation Forest outlier detection with contamination={contamination}...")
    # FIX: Added with_mean=False to handle sparse matrices proactively.
    X_scaled = StandardScaler(with_mean=False).fit_transform(X)
    
    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(X_scaled)
    
    outlier_indices = np.where(predictions == -1)[0]
    LOGGER.info(f"Identified {len(outlier_indices)} potential feature outliers.")
    return outlier_indices

def detect_label_outliers_lof(X_processed: np.ndarray, y: np.ndarray, n_neighbors: int = 20) -> np.ndarray:
    """
    Detects label-flipped outliers using Local Outlier Factor (LOF).
    Assumes the input data X_processed has already been scaled/encoded.
    """
    LOGGER.info(f"Running LOF-based label outlier detection...")
    
    # FIX: The redundant StandardScaler step has been removed.
    
    all_outlier_indices = []
    
    for class_label in np.unique(y):
        class_indices = np.where(y == class_label)[0]
        # Use .toarray() in case the slice is a sparse matrix
        X_class = X_processed[class_indices]
        if hasattr(X_class, "toarray"):
            X_class = X_class.toarray()
            
        if len(X_class) < n_neighbors:
            LOGGER.warning(f"Skipping LOF for class {class_label} due to insufficient samples ({len(X_class)}).")
            continue
            
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False, contamination='auto')
        predictions = lof.fit_predict(X_class)
        
        class_outlier_indices = class_indices[predictions == -1]
        all_outlier_indices.extend(class_outlier_indices)

    LOGGER.info(f"Identified {len(all_outlier_indices)} potential label-flipped outliers.")
    return np.array(all_outlier_indices)
