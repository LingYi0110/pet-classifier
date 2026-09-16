"""Classification metrics and publication-friendly experiment plots."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm


def classification_metrics(targets, predictions, top5, num_classes):
    return {
        "top1": float(np.mean(np.array(targets) == np.array(predictions))),
        "top5": float(np.mean(top5)),
        "macro_f1": float(f1_score(targets, predictions, labels=list(range(num_classes)),
                                    average="macro", zero_division=0)),
    }


@torch.no_grad()
def evaluate_model(model, loader, device, desc="Evaluate"):
    model.eval()
    targets, predictions, top5 = [], [], []
    loss_sum = 0.0
    progress = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True)
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        batch_loss_sum = torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        loss_sum += batch_loss_sum
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
        top5.extend(logits.topk(min(5, logits.shape[1]), dim=1).indices.eq(labels[:, None])
                    .any(1).cpu().tolist())
        progress.set_postfix(loss=f"{batch_loss_sum / len(labels):.4f}",
                             avg_loss=f"{loss_sum / len(targets):.4f}", refresh=False)
    if not targets:
        raise ValueError("Cannot evaluate an empty dataset")
    metrics = classification_metrics(targets, predictions, top5, logits.shape[1])
    metrics["loss"] = loss_sum / len(targets)
    return metrics, targets, predictions


def plot_confusion(targets, predictions, classes, path):
    matrix = confusion_matrix(targets, predictions, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes, xlabel="Predicted", ylabel="True")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    pairs = [(int(matrix[i, j] + matrix[j, i]), classes[i], classes[j])
             for i in range(len(classes)) for j in range(i + 1, len(classes))]
    return [{"count": n, "classes": [a, b]} for n, a, b in sorted(pairs, reverse=True)[:3]
            if n > 0]


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row["epoch"] for row in history]
    for key in ("train_loss", "val_loss"):
        axes[0].plot(epochs, [r[key] for r in history], label=key)
    for key in ("train_accuracy", "val_top1"):
        axes[1].plot(epochs, [r[key] for r in history], label=key)
    for ax, label in zip(axes, ("Loss", "Accuracy")):
        ax.set(xlabel="Epoch", ylabel=label)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
