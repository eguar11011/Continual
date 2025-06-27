#!/usr/bin/env python3
"""models.py – backbones y utilidades para Continual Learning

Incluye:
* `get_backbone(name, pretrained, **kw)` – ResNet‑18/50, ViT‑tiny/ small (si tienes `timm`).
* `Classifier(backbone, num_classes)` – head lineal opcionalmente añadible.
* Soporte multi‑GPU con `nn.DataParallel` o `torch.nn.parallel.DistributedDataParallel`.
* Pequeño *smoke test*: hace forward de un batch sintético y
  muestra dimensiones y memoria de la GPU.
"""
from __future__ import annotations

import argparse
from typing import Tuple

import torch
from torch import nn
from torch.cuda import memory_allocated

try:
    import timm  # para ViT, ConvNeXt, etc.
    _HAS_TIMM = True
except ModuleNotFoundError:
    _HAS_TIMM = False

import torchvision.models as tvm

__all__ = ["get_backbone", "Classifier"]

# ----------------------------------------------------------------------------
# 1.  Backbones
# ----------------------------------------------------------------------------

def _resnet18(pretrained: bool, **kw):
    return tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None, **kw)

def _resnet50(pretrained: bool, **kw):
    return tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None, **kw)

def _timm_model(model_name: str, pretrained: bool, **kw):
    assert _HAS_TIMM, "Instala timm: pip install timm"
    return timm.create_model(model_name, pretrained=pretrained, **kw)


_BACKBONE_REGISTRY = {
    "resnet18": _resnet18,
    "resnet50": _resnet50,
    "vit_tiny": lambda pretrained, **kw: _timm_model("vit_tiny_patch16_224", pretrained, **kw),
    "vit_small": lambda pretrained, **kw: _timm_model("vit_small_patch16_224", pretrained, **kw),
}


def get_backbone(name: str = "resnet18", *, pretrained: bool = False, **kw) -> nn.Module:
    """Devuelve un modelo *feature extractor* sin la capa final.

    Para ResNet devuelve ``model`` con ``model.fc`` expuesto para conocer `out_features`.
    Para ViT devuelve ``model.head`` removido y expone attr ``embed_dim``.
    """
    if name not in _BACKBONE_REGISTRY:
        raise ValueError(f"Backbone no soportado: {name}")
    model = _BACKBONE_REGISTRY[name](pretrained, **kw)

    # Quitar capas de clasificación finales para extraer solo features
    if name.startswith("resnet"):
        # Mantener .fc para que fuera del módulo obtengan características
        pass  # dejamos la línea fc; la cabeza final la ponemos en Classifier
    elif name.startswith("vit"):
        model.reset_classifier(0)
    return model

# ----------------------------------------------------------------------------
# 2.  Clasificador simple
# ----------------------------------------------------------------------------
class Classifier(nn.Module):
    """Backbone + head lineal configurable."""

    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        # Obtener dimensiones de salida
        if hasattr(backbone, "fc") and isinstance(backbone.fc, nn.Linear):
            in_feat = backbone.fc.in_features
            backbone.fc = nn.Identity()  # eliminamos la fc original
        elif hasattr(backbone, "embed_dim"):
            in_feat = backbone.embed_dim  # para ViT (timm)
        else:
            raise AttributeError("No se pudo inferir dimensión del backbone")
        self.head = nn.Linear(in_feat, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, tuple):  # algunos modelos devuelven (features, cls_token)
            feats = feats[0]
        return self.head(feats)

# models.py
class ExpandableClassifier(nn.Module):
    """Backbone + head lineal que admite crecimiento dinámico de clases."""

    def __init__(self, backbone: nn.Module, num_classes: int = 0):
        super().__init__()
        self.backbone = backbone
        # 1.- Inferir dimensión de salida del backbone
        if hasattr(backbone, "fc") and isinstance(backbone.fc, nn.Linear):
            in_feat = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif hasattr(backbone, "embed_dim"):           # ViT-timm
            in_feat = backbone.embed_dim
        else:
            raise AttributeError("No se pudo inferir dimensión del backbone")

        # 2.- Head inicial (puede empezar en 0-clases)
        self.head = nn.Linear(in_feat, num_classes, bias=True)

    # ─────────────── método clave ────────────────────
    def add_classes(self, n_new: int):
        """Añade *n_new* salidas; conserva pesos previos."""
        if n_new <= 0:
            return
        in_feat = self.head.in_features
        old_out = self.head.out_features

        new_head = nn.Linear(in_feat, old_out + n_new, bias=True)
        # Copiar pesos/bias antiguos
        with torch.no_grad():
            new_head.weight[:old_out].copy_(self.head.weight)
            new_head.bias  [:old_out].copy_(self.head.bias)
        self.head = new_head.to(self.head.weight.device)

    # ────────────────────────────────────────────────
    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, tuple):      # p.ej. ViT
            feats = feats[0]
        return self.head(feats)

# ----------------------------------------------------------------------------
# 3.  Multi‑GPU wrapper helper
# ----------------------------------------------------------------------------

def wrap_dataparallel(model: nn.Module, devices: str | None = None):
    """Envuelve con DataParallel si hay ≥2 GPUs disponibles."""
    if devices is None:
        devices = "cuda" if torch.cuda.is_available() else "cpu"
    if devices == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model.to(devices)

# ----------------------------------------------------------------------------
# 4.  Smoke test
# ----------------------------------------------------------------------------

def _smoke_test(args):
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = get_backbone(args.backbone, pretrained=False)
    model = Classifier(backbone, args.num_classes)
    model = wrap_dataparallel(model, "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.backbone} | Params: {sum(p.numel() for p in model.parameters())/1e6:.1f} M")

    x = torch.randn(args.batch, 3, args.img_size, args.img_size, device=dev)
    with torch.no_grad():
        logits = model(x)
    print(f"Input: {tuple(x.shape)} → Logits: {tuple(logits.shape)} | GPU mem: {memory_allocated(dev)/1e6:.1f} MB")


def _parse_args():
    p = argparse.ArgumentParser(description="Smoke test de modelos")
    p.add_argument("--backbone", choices=list(_BACKBONE_REGISTRY.keys()), default="resnet18")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    _smoke_test(_parse_args())
