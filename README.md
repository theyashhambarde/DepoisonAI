# DepoisonAI: Automated Detection and Reversal of Data Poisoning Attacks

[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

DepoisonAI is a full-stack, production-ready AI system built from scratch to automatically detect and reverse data poisoning attacks in machine learning datasets. It leverages classical ML, deep learning, and statistical reasoning using only local, open-source tools.



---

## 🎯 System Goals

- **Detect**: Identify if a dataset has been poisoned via label flipping, feature perturbation, or backdoors.
- **Identify**: Pinpoint the specific samples or features that have been compromised.
- **Reverse**: Clean the dataset by filtering malicious samples, relabeling incorrect data, or reconstructing perturbed features.
- **Evaluate**: Quantify the improvement in model performance after the cleaning process.
- **Visualize**: Provide clear visualizations of data distributions and identified poisoning patterns.

---

## ⚙️ Core Features & Technical Stack

- **Backend**: Python 3.9+
- **ML/DL**: `scikit-learn`, `PyTorch`, `transformers`, `sentence-transformers`
- **Data Handling**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`, `t-SNE`
- **UI**: `Streamlit` (Optional Interface)
- **Local-First**: **No reliance on any commercial APIs or closed-source tools.** All models and computations run locally.

### Supported Attack Types
- **Tabular Data**: Label Flipping, Feature Perturbation.
- **Image Data**: Backdoor Attacks (e.g., trigger-based pixel patterns).
- **Text Data**: Label Flipping, Backdoor Attacks (e.g., trigger phrases).

### Detection & Cleaning Methods
- **Detection**: Local Outlier Factor (LOF), Isolation Forest, PCA, Clustering, Saliency Maps, Embedding Analysis, TF-IDF Outlier Detection.
- **Cleaning**: Malicious Sample Filtering, k-NN Relabeling, Autoencoder-based Denoising, Masked Language Model Reconstruction.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- `pip` for package management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/DepoisonAI.git
    cd DepoisonAI
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Demonstrations

Explore the `notebooks/` directory to see live demonstrations of the system in action:

-   `adult_income_demo.ipynb`: Demonstrates detection and cleaning of label-flipped data in the UCI Adult Income dataset.
-   `cifar10_demo.ipynb`: Shows how to detect and mitigate simple backdoor attacks in the CIFAR-10 image dataset.
-   `imdb_text_demo.ipynb`: A walkthrough of detecting and filtering label-flipped samples in the IMDB movie review dataset.