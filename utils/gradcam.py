"""Grad-CAM for the predicted class, including frozen backbones."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F


def save_gradcam(model, image, target, classes, cfg, device, path):
    model.eval()
    activations = []
    handle = model.backbone.layer4[-1].register_forward_hook(
        lambda module, inputs, output: activations.append(output))
    try:
        with torch.enable_grad():
            inputs = image.unsqueeze(0).to(device).detach().requires_grad_(True)
            logits = model(inputs)
            predicted = logits.argmax(1).item()
            features = activations[0]
            gradients = torch.autograd.grad(logits[0, predicted], features)[0]
            cam = (gradients.mean((2, 3), keepdim=True) * features).sum(1, keepdim=True).relu()
            cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().cpu()
            cam = (cam - cam.min()) / (cam.max() - cam.min()).clamp_min(1e-8)
    finally:
        handle.remove()
    mean = torch.tensor(cfg.dataset.normalization_mean)[:, None, None]
    std = torch.tensor(cfg.dataset.normalization_std)[:, None, None]
    rgb = (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax in axes:
        ax.imshow(rgb)
        ax.axis("off")
    axes[1].imshow(cam.numpy(), cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[0].set_title("True: " + classes[target])
    axes[1].set_title("Predicted: " + classes[predicted])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
