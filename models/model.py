from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class PetClassifier(nn.Module):

    def __init__(self, num_classes=37, pretrained=True, dropout=0.0,
                 freeze_backbone=False):
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        self.set_freeze_backbone(freeze_backbone)

    def set_freeze_backbone(self, freeze=True):
        """Switch between head-only and full fine-tuning.

        Set this before creating the optimizer, or rebuild its parameter groups
        when unfreezing if it was constructed from trainable parameters only.
        """
        self.freeze_backbone = freeze
        self.backbone.requires_grad_(not freeze)
        self.backbone.train(self.training and not freeze)
        return self

    def train(self, mode=True):
        super().train(mode)
        # Freezing parameters alone does not freeze BatchNorm running statistics.
        # Keep the feature extractor in eval mode during head-only training.
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images):
        # Do not wrap this in no_grad: Grad-CAM may need input gradients even
        # when backbone parameters are frozen.
        features = self.backbone(images)
        return self.classifier(features)


def create_model(cfg):
    """Build the model from the project's full experiment configuration."""
    model_cfg = cfg.model
    if model_cfg.name != "resnet18":
        raise ValueError("Unsupported model: {!r}; expected 'resnet18'".format(model_cfg.name))
    if model_cfg.num_classes != cfg.dataset.num_classes:
        raise ValueError("model.num_classes must equal dataset.num_classes")
    return PetClassifier(
        num_classes=model_cfg.num_classes,
        pretrained=model_cfg.pretrained,
        dropout=model_cfg.dropout,
        freeze_backbone=model_cfg.freeze_backbone,
    )
