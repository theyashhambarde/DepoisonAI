import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
from ..utils.helpers import LOGGER  # <-- This line is now fixed


def train_model(model, dataloader, criterion, optimizer, device, epochs=5):
    """Generic model training loop."""
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        LOGGER.info(f"Epoch {epoch + 1} Training Loss: {running_loss / len(dataloader):.4f}")
    return model


def evaluate_model(model, dataloader, device, class_names):
    """Generic model evaluation function."""
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)

    LOGGER.info(f"Evaluation Accuracy: {accuracy:.4f}")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    return {"accuracy": accuracy, "report": report, "preds": all_preds, "labels": all_labels}


def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    """Plots a confusion matrix using seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()


def compare_evaluations(eval_before, eval_after, metric='accuracy'):
    """Compares evaluation metrics before and after cleaning."""
    before_metric = eval_before[metric]
    after_metric = eval_after[metric]
    improvement = after_metric - before_metric

    LOGGER.info("--- Performance Comparison ---")
    LOGGER.info(f"Metric: {metric.capitalize()}")
    LOGGER.info(f"Before Cleaning: {before_metric:.4f}")
    LOGGER.info(f"After Cleaning:  {after_metric:.4f}")
    LOGGER.info(f"Improvement:     {improvement:+.4f} ({improvement / before_metric:+.2%})")

    # Plotting the comparison
    fig, ax = plt.subplots()
    bars = ax.bar(['Before Cleaning', 'After Cleaning'], [before_metric, after_metric], color=['#ff9999', '#66b3ff'])
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'Model Performance Comparison ({metric.capitalize()})')
    ax.set_ylim(0, max(1.0, max(before_metric, after_metric) * 1.1))
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.4f}', va='bottom', ha='center')
    plt.show()