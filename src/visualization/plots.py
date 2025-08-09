import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


def plot_tsne_with_outliers(X: np.ndarray, outlier_indices: np.ndarray,
                            title: str = 't-SNE Visualization of Data with Outliers'):
    """
    Generates a t-SNE plot and highlights identified outliers.

    Args:
        X (np.ndarray): The feature data.
        outlier_indices (np.ndarray): Indices of the outliers.
        title (str): The plot title.
    """
    print("Generating t-SNE plot... This may take a moment.")

    # FIX: Replaced the removed 'n_iter' argument with 'max_iter'.
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1), max_iter=300)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(12, 8))

    # Create a label array for coloring: 0 for inliers, 1 for outliers
    labels = np.zeros(X.shape[0])
    if len(outlier_indices) > 0:
        labels[outlier_indices] = 1

    scatter = sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=labels,
        palette={0: 'skyblue', 1: 'red'},
        style=labels,
        markers={0: 'o', 1: 'X'},
        s=50,
        alpha=0.7
    )

    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    handles, _ = scatter.get_legend_handles_labels()
    scatter.legend(handles, ['Clean Samples', 'Detected Poison'], title='Sample Type')

    plt.show()