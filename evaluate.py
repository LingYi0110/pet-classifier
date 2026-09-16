"""Evaluate a saved best checkpoint on its held-out split and export figures."""

import argparse
import csv
import json

import torch
from torch.utils.data import DataLoader

from data.datasets import build_datasets
from models.model import create_model
from utils.config import ExperimentConfig, project_path
from utils.gradcam import save_gradcam
from utils.metrics import evaluate_model, plot_confusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    checkpoint = torch.load(project_path(args.checkpoint), map_location="cpu", weights_only=True)
    cfg = ExperimentConfig.model_validate(checkpoint["config"])
    cfg.model.pretrained = False
    if args.data_root:
        cfg.dataset.root = args.data_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    datasets, _ = build_datasets(cfg, split_payload=checkpoint["split"])
    dataset = datasets[args.split]
    if dataset.classes != checkpoint["classes"]:
        raise ValueError("Checkpoint and dataset class ordering differ")
    model = create_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.dataset.num_workers, pin_memory=cfg.dataset.pin_memory)
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    metrics, targets, predictions = evaluate_model(model, loader, device,
                                                   desc=f"Evaluate {args.split}")
    metrics["split"] = args.split
    metrics["epoch"] = checkpoint["epoch"]
    metrics["confused_pairs"] = plot_confusion(targets, predictions, dataset.classes,
                                               output / "confusion_matrix.png")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        table = csv.writer(stream)
        table.writerow(["global_index", "true", "predicted", "true_name", "predicted_name"])
        for index, target, pred in zip(dataset.indices, targets, predictions):
            table.writerow([index, target, pred, dataset.classes[target], dataset.classes[pred]])
    metrics["gradcam"] = {}
    for kind, match in (("correct", True), ("incorrect", False)):
        index = next((i for i, (a, b) in enumerate(zip(targets, predictions)) if (a == b) == match), None)
        if index is None:
            metrics["gradcam"][kind] = "No matching example in this split"
            continue
        image, target = dataset[index]
        save_gradcam(model, image, target, dataset.classes, cfg, device, output / f"gradcam_{kind}.png")
        metrics["gradcam"][kind] = {"global_index": dataset.indices[index]}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
