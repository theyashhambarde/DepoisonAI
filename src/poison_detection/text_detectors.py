
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.helpers import LOGGER


def detect_text_outliers_tfidf(X_text: np.ndarray, contamination: float = 0.1) -> np.ndarray:
    """
    Detects outliers in text data using TF-IDF followed by Isolation Forest.

    Args:
        X_text (np.ndarray): Array of text documents.
        contamination (float): The expected proportion of outliers.

    Returns:
        np.ndarray: Indices of suspected poisoned samples.
    """
    LOGGER.info("Detecting text outliers using TF-IDF and Isolation Forest...")

    # 1. Vectorize text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X_text)
    LOGGER.info(f"Vectorized text data to shape: {X_tfidf.shape}")

    # 2. Use Isolation Forest to find outliers
    iforest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    predictions = iforest.fit_predict(X_tfidf.toarray())

    outlier_indices = np.where(predictions == -1)[0]
    LOGGER.info(f"Identified {len(outlier_indices)} potential text outliers.")
    return outlier_indices
