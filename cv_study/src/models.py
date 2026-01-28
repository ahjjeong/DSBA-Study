import timm
import torch.nn as nn

def build_model(model_cfg, num_classes: int) -> nn.Module:
    return timm.create_model(
        model_cfg.name,
        pretrained=bool(model_cfg.pretrained),
        num_classes=num_classes,
    )