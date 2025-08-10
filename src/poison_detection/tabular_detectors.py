import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from ..utils.helpers import LOGGER

def detect_feature_outliers_iforest(
    X: np.ndarray,
    contamination: float = 0.1
) -> np.ndarray:
    """
    Detect unusual samples in feature space using Isolation Forest.
    """
    LOGGER.info(f"🔍 Running Isolation Forest with contamination={contamination}...")
    # Using with_mean=False for compatibility with sparse matrices
    X_scaled = StandardScaler(with_mean=False).fit_transform(X)
    
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    predictions = model.fit_predict(X_scaled)
    
    outlier_indices = np.where(predictions == -1)[0]
    LOGGER.info(f"✅ Found {len(outlier_indices)} feature outliers.")
    return outlier_indices


def detect_label_outliers_lof(
    X_processed: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 20
) -> np.ndarray:
    """
    Detect label-flipped outliers using Local Outlier Factor (LOF).
    The input X_processed should already be scaled/encoded.
    """
    LOGGER.info(f"🔍 Running LOF-based label outlier detection (n_neighbors={n_neighbors})...")
    
    all_outlier_indices = []
    
    for class_label in np.unique(y):
        class_indices = np.where(y == class_label)[0]
        
        X_class = X_processed[class_indices]
        if hasattr(X_class, "toarray"):  # convert sparse to dense if needed
            X_class = X_class.toarray()
            
        if len(X_class) < n_neighbors:
            LOGGER.warning(
                f"⚠️ Skipping LOF for class '{class_label}' — "
                f"only {len(X_class)} samples (need ≥ {n_neighbors})."
            )
            continue
            
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=False,
            contamination='auto',
            n_jobs=-1
        )
        predictions = lof.fit_predict(X_class)
        
        class_outlier_indices = class_indices[predictions == -1]
        all_outlier_indices.extend(class_outlier_indices)

    LOGGER.info(f"✅ Found {len(all_outlier_indices)} label-based outliers.")
    return np.array(all_outlier_indices)
