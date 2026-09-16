import math
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def project_path(value="."):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


class ConfigSection(BaseModel):
    """Shared pydantic settings for every config section."""

    model_config = ConfigDict(extra="forbid")


class DatasetConfig(ConfigSection):
    """Dataset, split and input preprocessing defaults."""

    name: str = Field(default="oxford_iiit_pet")
    root: str = Field(default="datasets")
    download: bool = Field(default=True)
    num_classes: int = Field(default=37, gt=0)
    train_ratio: float = Field(default=0.70, gt=0, lt=1)
    val_ratio: float = Field(default=0.15, gt=0, lt=1)
    test_ratio: float = Field(default=0.15, gt=0, lt=1)
    split_seed: int = Field(default=42)
    split_file: str = Field(default="datasets/oxford_iiit_pet_split.json")
    resize_size: int = Field(default=256, gt=0)
    image_size: int = Field(default=224, gt=0)
    normalization_mean: list = Field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalization_std: list = Field(default_factory=lambda: [0.229, 0.224, 0.225])
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = Field(default=True)
    persistent_workers: bool = Field(default=True)

    @field_validator("normalization_mean", "normalization_std")
    @classmethod
    def _check_normalization(cls, value, info):
        name = "dataset." + info.field_name
        if len(value) != 3:
            raise ValueError(name + " must contain 3 values")
        if any(isinstance(item, bool) or not isinstance(item, (int, float))
               or not math.isfinite(item) for item in value):
            raise ValueError(name + " must contain only finite numbers")
        if info.field_name == "normalization_std" and any(item <= 0 for item in value):
            raise ValueError("dataset.normalization_std values must be positive")
        return [float(item) for item in value]

    @model_validator(mode="after")
    def _check_dataset(self):
        if self.resize_size < self.image_size:
            raise ValueError("dataset.resize_size must be >= dataset.image_size")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError(
                "dataset.persistent_workers cannot be true when dataset.num_workers is 0"
            )
        return self


class AugmentationConfig(ConfigSection):
    """Training-only augmentation and MixUp defaults."""

    random_crop: bool = Field(default=True)
    random_horizontal_flip: bool = Field(default=True)
    rand_augment: bool = Field(default=False)
    rand_augment_num_ops: int = Field(default=2, gt=0)
    rand_augment_magnitude: int = Field(default=9, ge=0, le=30)
    mixup: bool = Field(default=False)
    mixup_alpha: float = Field(default=0.2)

    @model_validator(mode="after")
    def _check_mixup(self):
        if self.mixup and self.mixup_alpha <= 0:
            raise ValueError("augmentation.mixup_alpha must be positive when MixUp is enabled")
        return self


class ModelConfig(ConfigSection):
    """Backbone and classifier-head defaults."""

    name: str = Field(default="resnet18")
    pretrained: bool = Field(default=True)
    num_classes: int = Field(default=37, gt=0)
    dropout: float = Field(default=0.0, ge=0, lt=1)
    freeze_backbone: bool = Field(default=False)


class TrainConfig(ConfigSection):
    """Optimization and training-loop defaults."""

    seed: int = Field(default=42)
    epochs: int = Field(default=15, gt=0)
    batch_size: int = Field(default=32, gt=0)
    optimizer: str = Field(default="adamw")
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    momentum: float = Field(default=0.9, ge=0)
    betas: list = Field(default_factory=lambda: [0.9, 0.999])
    scheduler: str = Field(default="none")
    scheduler_step_size: int = Field(default=5, gt=0)
    scheduler_gamma: float = Field(default=0.1, gt=0)
    label_smoothing: float = Field(default=0.0, ge=0, lt=1)
    gradient_clip_norm: float | None = Field(default=None, gt=0)
    amp: bool = Field(default=True)
    early_stopping_patience: int | None = Field(default=None, gt=0)
    log_every_n_steps: int = Field(default=50, gt=0)
    validate_every_n_epochs: int = Field(default=1, gt=0)
    resume_from: str | None = Field(default=None)

    @field_validator("optimizer")
    @classmethod
    def _check_optimizer(cls, value):
        value = str(value).lower()
        allowed = {"adamw", "adam", "sgd"}
        if value not in allowed:
            raise ValueError(
                "train.optimizer must be one of {}, got {!r}".format(sorted(allowed), value)
            )
        return value

    @field_validator("scheduler")
    @classmethod
    def _check_scheduler(cls, value):
        value = str(value).lower()
        allowed = {"none", "step", "cosine"}
        if value not in allowed:
            raise ValueError(
                "train.scheduler must be one of {}, got {!r}".format(sorted(allowed), value)
            )
        return value

    @field_validator("betas")
    @classmethod
    def _check_betas(cls, value):
        values = list(value)
        if len(values) != 2 or any(not 0 < item < 1 for item in values):
            raise ValueError("train.betas must contain two values in (0, 1)")
        return values


class RuntimeConfig(ConfigSection):
    """Hardware and reproducibility defaults."""

    device: str = Field(default="auto")
    deterministic: bool = Field(default=True)
    benchmark: bool = Field(default=False)

    @field_validator("device")
    @classmethod
    def _check_device(cls, value):
        if not str(value).strip():
            raise ValueError("runtime.device cannot be empty")
        return value


class PathsConfig(ConfigSection):
    """Output locations shared by training and evaluation."""

    output_dir: str = Field(default="experiments")
    experiment_dir: str = Field(default="${paths.output_dir}/${experiment.name}")
    checkpoint_dir: str = Field(default="${paths.experiment_dir}/checkpoints")
    log_dir: str = Field(default="${paths.experiment_dir}/logs")
    metrics_dir: str = Field(default="${paths.experiment_dir}/metrics")
    visualization_dir: str = Field(default="${paths.experiment_dir}/visualizations")


class ExperimentMetaConfig(ConfigSection):
    """Identity of one experiment: its name and the files it inherits."""

    name: str = Field(default="baseline")
    bases: list = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _check_name(cls, value):
        if not str(value).strip():
            raise ValueError("experiment.name cannot be empty")
        return value

    @field_validator("bases")
    @classmethod
    def _check_bases(cls, value):
        names = [str(item) for item in value]
        if any(not item.strip() for item in names):
            raise ValueError("experiment.bases entries cannot be empty")
        return names


class ExperimentConfig(ConfigSection):
    """Root schema for one reproducible experiment."""

    experiment: ExperimentMetaConfig = Field(default_factory=ExperimentMetaConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @model_validator(mode="after")
    def _check_experiment(self):
        ratios = [
            self.dataset.train_ratio,
            self.dataset.val_ratio,
            self.dataset.test_ratio,
        ]
        if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "Dataset split ratios must sum to 1.0, got {:.6f}".format(sum(ratios))
            )
        if self.model.num_classes != self.dataset.num_classes:
            raise ValueError("model.num_classes must equal dataset.num_classes")
        return self


def get_default_config():
    """Return the default experiment configuration."""

    return ExperimentConfig()


def _resolve_config_file(config_path):
    """Validate an absolute file path without guessing directories or suffixes."""
    path = Path(config_path)
    if not path.is_absolute():
        raise ValueError("Configuration path must be absolute: {}".format(path))
    if not path.is_file():
        raise FileNotFoundError("Configuration file does not exist: {}".format(path))
    return path.resolve()


def _normalise_base_list(value):
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(item) for item in value]
    raise TypeError("experiment.bases must be a file name or a list of file names")


def _experiment_base_names(loaded):
    """Read the direct ``experiment.bases`` entries of a raw YAML document."""

    experiment = loaded.get("experiment")
    if isinstance(experiment, DictConfig):
        return _normalise_base_list(experiment.get("bases"))
    return []


def _load_yaml_with_bases(config_path, active_chain=()):
    """Load an absolute YAML path; resolve bases relative to their declaring file."""

    path = _resolve_config_file(config_path)
    if path in active_chain:
        chain = " -> ".join(str(item) for item in (*active_chain, path))
        raise ValueError("Circular configuration inheritance detected: {}".format(chain))

    loaded = OmegaConf.load(str(path))
    if loaded is None:
        loaded = OmegaConf.create({})
    if not isinstance(loaded, DictConfig):
        raise TypeError("Configuration root must be a mapping: {}".format(path))

    base_names = _experiment_base_names(loaded)

    merged = OmegaConf.create({})
    next_chain = (*active_chain, path)
    for base in base_names:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = OmegaConf.merge(
            merged, _load_yaml_with_bases(base_path, active_chain=next_chain)
        )

    child = OmegaConf.to_container(loaded, resolve=False)
    resolved = OmegaConf.merge(merged, child)
    OmegaConf.update(resolved, "experiment.bases", list(base_names), force_add=True)
    return resolved


def _split_override_string(value):
    """Split comma-separated overrides without breaking list values."""

    result = []
    current = []
    depth = 0
    quote = None
    escaped = False

    for char in value:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\" and quote is not None:
            current.append(char)
            escaped = True
            continue
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
        elif char in "[{(":
            depth += 1
            current.append(char)
        elif char in "]})":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                result.append(item)
            current = []
        else:
            current.append(char)

    item = "".join(current).strip()
    if item:
        result.append(item)
    return result


def _normalise_overrides(overrides):
    """Accept a list of ``key=value`` strings, or a single comma-separated string."""

    if overrides is None:
        return []
    if isinstance(overrides, str):
        values = [overrides]
    else:
        values = overrides

    result = []
    for value in values:
        if isinstance(value, (list, tuple)):
            result.extend(_normalise_overrides(value))
        else:
            result.extend(_split_override_string(str(value)))

    malformed = [item for item in result if "=" not in item]
    if malformed:
        raise ValueError(
            "Every configuration override must use key=value syntax; invalid value(s): {}".format(
                ", ".join(malformed)
            )
        )
    return result


def apply_overrides(config, overrides):
    """Merge dotlist overrides into an OmegaConf config."""

    override_list = _normalise_overrides(overrides)
    if not override_list:
        return config
    return OmegaConf.merge(config, OmegaConf.from_dotlist(override_list))


def _to_omega(config):
    if isinstance(config, BaseModel):
        return OmegaConf.create(config.model_dump())
    if isinstance(config, DictConfig):
        return config
    return OmegaConf.create(config)


def load_config(config_path, overrides=None):
    """Load an absolute YAML path, merge overrides, then validate.

    Parameters
    ----------
    config_path:
        Absolute path to an existing YAML file, including its extension.
    overrides:
        List of dotlist strings such as ``[\"train.epochs=20\"]``.

    Priority, from low to high, is:

    1. pydantic schema defaults;
    2. recursively loaded YAML base files;
    3. the selected YAML file;
    4. ``overrides``.

    A YAML file can inherit files through ``experiment.bases``. Relative base
    paths are resolved against the declaring YAML's directory; absolute paths
    are used directly. The files are merged in order, so a later
    entry overrides an earlier one and this file wins over all of them::

        experiment:
          name: mixup
          bases:
            - default.yml
    """

    config = _to_omega(ExperimentConfig())
    config = OmegaConf.merge(config, _load_yaml_with_bases(config_path))
    config = apply_overrides(config, overrides)
    OmegaConf.resolve(config)
    data = OmegaConf.to_container(config, resolve=True)
    if not isinstance(data, dict):
        raise TypeError("Expected a mapping configuration")
    return ExperimentConfig.model_validate(data)


def config_to_dict(config, resolve=True):
    """Convert a config into a plain Python dictionary."""

    if isinstance(config, BaseModel):
        return config.model_dump()
    value = OmegaConf.to_container(_to_omega(config), resolve=resolve)
    if not isinstance(value, dict):
        raise TypeError("Expected a mapping configuration")
    return value


def config_to_yaml(config, resolve=True):
    """Return a readable YAML representation of a config."""

    if isinstance(config, BaseModel):
        return OmegaConf.to_yaml(OmegaConf.create(config.model_dump()), resolve=resolve)
    return OmegaConf.to_yaml(_to_omega(config), resolve=resolve)


def save_config(config, path, resolve=True):
    """Save a config to the requested path and return that path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config_to_dict(config, resolve=resolve)), str(target))
    return target


__all__ = [
    "AugmentationConfig",
    "DatasetConfig",
    "ExperimentConfig",
    "ExperimentMetaConfig",
    "ModelConfig",
    "PathsConfig",
    "RuntimeConfig",
    "TrainConfig",
    "apply_overrides",
    "project_path",
    "config_to_dict",
    "config_to_yaml",
    "get_default_config",
    "load_config",
    "save_config",
]
