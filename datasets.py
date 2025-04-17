#!/usr/bin/env python3
"""datasets.py – utilidades de carga y división de datasets para Continual Learning

(1)  Carga datasets comunes mediante ``torchvision``.
(2)  Genera divisiones por tareas *task‑incremental* (p.ej. Split CIFAR‑10 → 5 tasks * 2 clases).
(3)  Expone un pequeño *smoke‑test* en el bloque *main* para verificar que todo corre.

Uso rápido (smoke test)
-----------------------
$ python datasets.py --dataset split_cifar10

Mostrará algo como:
```
[split_cifar10] task=0  classes=[0, 1]  train=10000  test=2000
[split_cifar10] task=1  classes=[2, 3]  train=10000  test=2000
...
```
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as T

__all__ = [
    "load_cifar10",
    "load_cifar100",
    "make_class_splits",
    "build_split_datasets",
]

# ---------------------------------------------------------------------------
# 1.  Carga de datasets base
# ---------------------------------------------------------------------------

def _default_transform(img_size: int | None = None):
    tx = [T.ToTensor()]
    if img_size is not None:
        tx.insert(0, T.Resize(img_size))
    return T.Compose(tx)


def load_cifar10(root: str | Path = "~/.torchvision", *, img_size: int | None = None):
    """Devuelve (trainset, testset) de CIFAR‑10 con transform básica."""
    root = Path(root).expanduser()
    tfm = _default_transform(img_size)
    trainset = CIFAR10(root, train=True, download=True, transform=tfm)
    testset = CIFAR10(root, train=False, download=True, transform=tfm)
    return trainset, testset


def load_cifar100(root: str | Path = "~/.torchvision", *, img_size: int | None = None):
    root = Path(root).expanduser()
    tfm = _default_transform(img_size)
    trainset = CIFAR100(root, train=True, download=True, transform=tfm)
    testset = CIFAR100(root, train=False, download=True, transform=tfm)
    return trainset, testset

# ---------------------------------------------------------------------------
# 2.  Construcción de splits por clases
# ---------------------------------------------------------------------------

def make_class_splits(
    trainset: Dataset,
    testset: Dataset,
    classes_per_task: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Agrupa índices por tareas con *classes_per_task* etiquetas cada una.

    Devuelve:
        train_idx_tasks, test_idx_tasks – listas de lista de índices.
    """
    num_classes = len(set(trainset.targets))  # CIFAR = 10/100
    assert num_classes % classes_per_task == 0, "El nº de clases debe dividirse exacto en las tareas"

    # Mapear etiqueta → índices
    idx_by_class_train: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for i, lbl in enumerate(trainset.targets):
        idx_by_class_train[lbl].append(i)

    idx_by_class_test: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for i, lbl in enumerate(testset.targets):
        idx_by_class_test[lbl].append(i)

    train_tasks, test_tasks = [], []
    for start in range(0, num_classes, classes_per_task):
        cls_slice = list(range(start, start + classes_per_task))
        train_tasks.append(sum((idx_by_class_train[c] for c in cls_slice), []))
        test_tasks.append(sum((idx_by_class_test[c] for c in cls_slice), []))

    return train_tasks, test_tasks


# ---------------------------------------------------------------------------
# 3.  Helper de alto nivel: build_split_datasets
# ---------------------------------------------------------------------------

def build_split_datasets(
    dataset: str,
    classes_per_task: int,
    root: str | Path = "~/.torchvision",
    img_size: int | None = None,
):
    """Devuelve listas de *Subset* PyTorch por task.

    Ejemplo:
    >>> train_tasks, test_tasks = build_split_datasets("cifar10", 2)
    """
    if dataset == "cifar10":
        trainset, testset = load_cifar10(root, img_size=img_size)
    elif dataset == "cifar100":
        trainset, testset = load_cifar100(root, img_size=img_size)
    else:
        raise ValueError(f"Dataset no soportado: {dataset}")

    train_idx_tasks, test_idx_tasks = make_class_splits(trainset, testset, classes_per_task)

    train_subsets = [Subset(trainset, idxs) for idxs in train_idx_tasks]
    test_subsets = [Subset(testset, idxs) for idxs in test_idx_tasks]
    return train_subsets, test_subsets


# ---------------------------------------------------------------------------
# 4.  Smoke‑test CLI
# ---------------------------------------------------------------------------

def _smoke_test(args):
    train_ts, test_ts = build_split_datasets(args.dataset, args.cpt, img_size=args.img)
    for t, (tr, te) in enumerate(zip(train_ts, test_ts)):
        cls_start = t * args.cpt
        cls_end = cls_start + args.cpt - 1
        print(
            f"[{args.dataset}] task={t}  classes=[{cls_start}, {cls_end}]  "
            f"train={len(tr)}  test={len(te)}"
        )


def _parse_args():
    p = argparse.ArgumentParser(description="Smoke test de datasets para CL")
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "split_cifar10"], default="split_cifar10")
    p.add_argument("--cpt", type=int, default=2, help="Clases por tarea")
    p.add_argument("--img", type=int, default=None, help="Resize opcional de imagen (px)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # Alias para conveniencia – split_cifar10 ≡ cifar10 con 2 clases/tarea
    if args.dataset == "split_cifar10":
        args.dataset = "cifar10"
        args.cpt = 5
    _smoke_test(args)
