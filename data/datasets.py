import random

import numpy as np
import torch
from PIL import ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import OxfordIIITPet

from utils.config import project_path

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PetSubset(Dataset):
    """Map a global index onto the official trainval then test torchvision splits."""

    def __init__(self, sources, indices, transform):
        self.sources = sources
        self.indices = list(indices)
        self.transform = transform
        self.classes = sources[0].classes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        source = self.sources[0]
        if index >= len(source):
            index -= len(source)
            source = self.sources[1]
        image, target = source[index]
        return self.transform(image), target


class PetData:
    """Train/val/test loaders plus the 37 breed names."""

    def __init__(self, train, val, test, classes, split=None):
        self.train = train
        self.val = val
        self.test = test
        self.classes = list(classes)
        self.split = split

    @property
    def num_classes(self):
        return len(self.classes)


def build_transforms(cfg, training):
    """Build the PIL → tensor pipeline for one split.

    Training keeps stochastic crop/flip (and optional RandAugment) online so
    every epoch sees a different view. Val/test stay deterministic.
    MixUp is applied later in the training step, not here.
    """

    dataset_cfg = cfg.dataset
    aug = cfg.augmentation
    steps = [T.Resize(dataset_cfg.resize_size)]
    if training:
        if aug.random_crop:
            steps.append(T.RandomCrop(dataset_cfg.image_size))
        else:
            steps.append(T.CenterCrop(dataset_cfg.image_size))
        if aug.random_horizontal_flip:
            steps.append(T.RandomHorizontalFlip())
        if aug.rand_augment:
            steps.append(
                T.RandAugment(
                    num_ops=aug.rand_augment_num_ops,
                    magnitude=aug.rand_augment_magnitude,
                )
            )
    else:
        steps.append(T.CenterCrop(dataset_cfg.image_size))
    return T.Compose(
        steps
        + [
            T.ToTensor(),
            T.Normalize(
                mean=list(dataset_cfg.normalization_mean),
                std=list(dataset_cfg.normalization_std),
            ),
        ]
    )


def _read_split_records(root):
    records = []
    boundaries = []
    annotations = root / "oxford-iiit-pet" / "annotations"
    for split in ("trainval", "test"):
        lines = (annotations / "{}.txt".format(split)).read_text().splitlines()
        records.extend(
            (parts[0], int(parts[1]) - 1)
            for line in lines
            if (parts := line.split())
        )
        boundaries.append(len(records))
    return records, boundaries


def create_split(cfg):
    """Prepare source data and return a new split; the caller owns persistence."""
    _load_sources(cfg)
    records, _ = _read_split_records(project_path(cfg.dataset.root))
    labels = np.array([target for _, target in records])
    indices = np.arange(len(labels))
    train, rest = train_test_split(
        indices,
        train_size=cfg.dataset.train_ratio,
        stratify=labels,
        random_state=cfg.dataset.split_seed,
    )
    val, test = train_test_split(
        rest,
        train_size=cfg.dataset.val_ratio / (cfg.dataset.val_ratio + cfg.dataset.test_ratio),
        stratify=labels[rest],
        random_state=cfg.dataset.split_seed,
    )
    return {
        "seed": cfg.dataset.split_seed,
        "train_ratio": cfg.dataset.train_ratio,
        "val_ratio": cfg.dataset.val_ratio,
        "test_ratio": cfg.dataset.test_ratio,
        "train": train.tolist(),
        "val": val.tolist(),
        "test": test.tolist(),
    }


def _validate_split(split_payload, n_records, cfg):
    if split_payload["seed"] != cfg.dataset.split_seed:
        raise ValueError("Split seed mismatch; use a different dataset.split_file")
    for key in ("train_ratio", "val_ratio", "test_ratio"):
        if key in split_payload and split_payload[key] != getattr(cfg.dataset, key):
            raise ValueError("Split {} mismatch; use a different dataset.split_file".format(key))
    all_indices = sum((split_payload[key] for key in ("train", "val", "test")), [])
    if sorted(all_indices) != list(range(n_records)):
        raise ValueError("Splits must be disjoint and cover the dataset exactly")


def _load_sources(cfg):
    """Load or download the two official source datasets."""
    root = project_path(cfg.dataset.root)
    sources = [
        OxfordIIITPet(
            root,
            split=split,
            target_types="category",
            download=cfg.dataset.download,
        )
        for split in ("trainval", "test")
    ]
    if list(sources[0].classes) != list(sources[1].classes):
        raise RuntimeError("Oxford-IIIT Pet trainval/test class lists do not match")
    return sources


def build_datasets(cfg, split_payload):
    """Build subsets using an explicit split, without creating or saving one."""

    root = project_path(cfg.dataset.root)
    sources = _load_sources(cfg)

    records, boundaries = _read_split_records(root)
    # PetSubset maps a global record index onto sources[0] then sources[1] by
    # subtracting len(sources[0]). Verify that assumption so any ordering or
    # filtering drift raises instead of silently pairing images with the wrong labels.
    if boundaries[0] != len(sources[0]) or boundaries[1] - boundaries[0] != len(sources[1]):
        raise ValueError(
            "Annotation lists and torchvision splits disagree; subset indices would shift"
        )

    _validate_split(split_payload, len(records), cfg)

    datasets = {
        key: PetSubset(sources, split_payload[key], build_transforms(cfg, key == "train"))
        for key in ("train", "val", "test")
    }
    return datasets, split_payload


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _make_loader(dataset, cfg, shuffle):
    kwargs = {
        "dataset": dataset,
        "batch_size": cfg.train.batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.dataset.num_workers,
        "pin_memory": cfg.dataset.pin_memory,
        "drop_last": False,
        "worker_init_fn": seed_worker if cfg.dataset.num_workers > 0 else None,
    }
    if cfg.dataset.num_workers > 0:
        kwargs["persistent_workers"] = cfg.dataset.persistent_workers
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(cfg.train.seed)
        kwargs["generator"] = generator
    return DataLoader(**kwargs)


def create_dataloaders(cfg, split_payload):
    """Build shuffled train and deterministic val/test loaders from ``cfg``."""

    datasets, split_payload = build_datasets(cfg, split_payload=split_payload)
    return PetData(
        train=_make_loader(datasets["train"], cfg, shuffle=True),
        val=_make_loader(datasets["val"], cfg, shuffle=False),
        test=_make_loader(datasets["test"], cfg, shuffle=False),
        classes=datasets["train"].classes,
        split=split_payload,
    )
