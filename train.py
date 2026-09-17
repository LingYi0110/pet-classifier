"""Train a reproducible baseline or single-variable ablation."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.datasets import PetData, create_dataloaders, create_split
from models.model import create_model
from utils.config import ExperimentConfig, load_config, project_path, save_config
from utils.metrics import evaluate_model, plot_history


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def prepare_data(cfg: ExperimentConfig) -> PetData:
    """Reuse a saved split, or generate it before constructing training loaders."""
    path = project_path(cfg.dataset.split_file)
    existing = path.exists()
    if existing:
        split = json.loads(path.read_text(encoding="utf-8"))
    else:
        split = create_split(cfg)
    # Validate the split through dataset construction before persisting it.
    data = create_dataloaders(cfg, split_payload=split)
    if not existing:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(split, stream, indent=2)
    print("Reusing split:" if existing else "Created split:", path)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/baseline.yml",
                        help="Absolute path or path relative to this training script")
    parser.add_argument("--override", nargs="+", action="append", default=[])
    args = parser.parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    cfg = load_config(config_path.resolve(), args.override)
    if cfg.train.resume_from:
        raise ValueError("Resume is not supported yet; use a new experiment name for a fresh run")
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    torch.cuda.manual_seed_all(cfg.train.seed)
    torch.backends.cudnn.benchmark = cfg.runtime.benchmark and not cfg.runtime.deterministic
    torch.backends.cudnn.deterministic = cfg.runtime.deterministic
    torch.use_deterministic_algorithms(cfg.runtime.deterministic, warn_only=True)
    device = resolve_device(cfg.runtime.device)
    output = project_path(cfg.paths.experiment_dir)
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = project_path(cfg.paths.checkpoint_dir)
    checkpoints.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output / "config.yaml")
    data = prepare_data(cfg)
    if data.num_classes != cfg.model.num_classes:
        raise ValueError("Dataset class count does not match model")
    (output / "split.json").write_text(json.dumps(data.split, indent=2), encoding="utf-8")
    model = create_model(cfg).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    options = dict(lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    if cfg.train.optimizer == "sgd":
        optimizer = torch.optim.SGD(parameters, momentum=cfg.train.momentum, **options)
    else:
        cls = torch.optim.AdamW if cfg.train.optimizer == "adamw" else torch.optim.Adam
        optimizer = cls(parameters, betas=tuple(cfg.train.betas), **options)
    scheduler = None
    if cfg.train.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.train.scheduler_step_size,
                                                    gamma=cfg.train.scheduler_gamma)
    elif cfg.train.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.train.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, best, stale, step = [], -1.0, 0, 0
    started = time.perf_counter()
    with SummaryWriter(str(project_path(cfg.paths.log_dir))) as writer:
        for epoch in range(1, cfg.train.epochs + 1):
            model.train()
            loss_sum, correct, count = 0.0, 0.0, 0
            progress = tqdm(data.train, desc=f"Train {epoch}/{cfg.train.epochs}",
                            unit="batch", dynamic_ncols=True)
            for images, labels in progress:
                if epoch == 1 and count == 0:
                    print(f"First training batch: images={list(images.shape)}, labels={list(labels.shape)}")
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                lam, other = 1.0, labels
                if cfg.augmentation.mixup:
                    lam = float(np.random.beta(cfg.augmentation.mixup_alpha, cfg.augmentation.mixup_alpha))
                    order = torch.randperm(len(labels), device=device)
                    images = lam * images + (1 - lam) * images[order]
                    other = labels[order]
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = model(images)
                    loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, other)
                scaler.scale(loss).backward()
                if cfg.train.gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(parameters, cfg.train.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                n = len(labels)
                batch_loss = loss.item()
                loss_sum += batch_loss * n
                pred = logits.argmax(1)
                correct += lam * pred.eq(labels).sum().item() + (1 - lam) * pred.eq(other).sum().item()
                count += n
                progress.set_postfix(loss=f"{batch_loss:.4f}",
                                     avg_loss=f"{loss_sum / count:.4f}", refresh=False)
                step += 1
                if step % cfg.train.log_every_n_steps == 0:
                    writer.add_scalar("step/loss", batch_loss, step)
            row = dict(epoch=epoch, train_loss=loss_sum / count, train_accuracy=correct / count,
                       lr=optimizer.param_groups[0]["lr"])
            validate = epoch % cfg.train.validate_every_n_epochs == 0 or epoch == cfg.train.epochs
            improved = False
            if validate:
                metrics, _, _ = evaluate_model(model, data.val, device,
                                               desc=f"Val {epoch}/{cfg.train.epochs}")
                row.update({"val_" + key: value for key, value in metrics.items()})
                improved = metrics["top1"] > best
                if improved:
                    best, stale = metrics["top1"], 0
                else:
                    stale += 1
            else:
                row.update({"val_" + key: float("nan") for key in ("loss", "top1", "top5", "macro_f1")})
            if scheduler:
                scheduler.step()
            history.append(row)
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
            payload = dict(model=model.state_dict(), config=cfg.model_dump(), classes=data.classes,
                           split=data.split, epoch=epoch, best_val_top1=best)
            torch.save(payload, checkpoints / "last_model.pth")
            if improved:
                torch.save(payload, checkpoints / "best_model.pth")
            with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
                table = csv.DictWriter(stream, fieldnames=list(row))
                table.writeheader()
                table.writerows(history)
            print(row)
            writer.flush()
            if validate and cfg.train.early_stopping_patience and stale >= cfg.train.early_stopping_patience:
                break
    plot_history(history, output / "curves.png")
    summary = dict(best_val_top1=best, epochs=len(history), seconds=time.perf_counter() - started,
                   device=str(device), torch_version=str(torch.__version__),
                   hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
    (output / "training.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved experiment:", output)


if __name__ == "__main__":
    main()
