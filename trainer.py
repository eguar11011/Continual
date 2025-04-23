#!/usr/bin/env python3
"""trainer.py – Bucle de entrenamiento por tareas para Continual Learning

Combina:
  • datasets.py  → construcción de *splits* de datos
  • models.py    → backbone + head
  • learner.py   → estrategia CL (finetune, replay, ewc…)

Ejemplo mínimo (3 tasks × 2 clases, 1 época cada una):

```bash
python trainer.py \
    --dataset cifar10 --classes-per-task 2 \
    --backbone resnet18 --strategy finetune --epochs 1
```
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import build_split_datasets
from models import get_backbone, Classifier
from learner import build_learner

# ----------------------------------------------------------------------------
# 1. Helper: label mapping per task
# ----------------------------------------------------------------------------

def map_labels(batch: Tuple[torch.Tensor, torch.Tensor], class_map: Dict[int, int]):
    x, y = batch
    y_mapped = torch.tensor([class_map[int(lbl)] for lbl in y], dtype=torch.long)
    return x, y_mapped


class MappedLoader:
    """Wrapper que remapea las etiquetas *on-the-fly* usando class_map."""

    def __init__(self, loader: DataLoader, class_map: Dict[int, int]):
        self.loader = loader
        self.class_map = class_map

    def __iter__(self):
        for batch in self.loader:
            yield map_labels(batch, self.class_map)

    def __len__(self):
        return len(self.loader)


# ----------------------------------------------------------------------------
# 2. Trainer
# ----------------------------------------------------------------------------
class Trainer:
    def __init__(
        self,
        learner,
        train_tasks: List[torch.utils.data.Dataset],
        test_tasks: List[torch.utils.data.Dataset],
        device: torch.device,
        classes_per_task: int,
        epochs: int = 1,
        batch_size: int = 64,
        output_dir: Path | None = None,
    ) -> None:
        self.learner = learner
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        self.device = device
        self.classes_per_task = classes_per_task
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_dir = output_dir
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: List[float] = []

    # ---------------------- internal helpers ----------------------------
    def _loader(self, subset, shuffle: bool):
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True,
        )

    # --------------------------- main loop ------------------------------
    def run(self):
        for t, (train_subset, test_subset) in enumerate(zip(self.train_tasks, self.test_tasks)):
            task_classes = list(range(t * self.classes_per_task, (t + 1) * self.classes_per_task))
            class_map = {orig: idx for idx, orig in enumerate(task_classes)}

            print(f"\n===== Task {t} | classes {task_classes} =====")
            train_loader = MappedLoader(self._loader(train_subset, True), class_map)
            test_loader = MappedLoader(self._loader(test_subset, False), class_map)

            # ---- epochs loop ----
            for epoch in range(self.epochs):
                pbar = tqdm(train_loader, desc=f"Task {t} | Epoch {epoch+1}/{self.epochs}")
                for batch in pbar:
                    loss = self.learner.observe(batch)
                    pbar.set_postfix(loss=f"{loss:.4f}")

            # ---- end-of-task hook ----
            self.learner.end_task(train_loader)

            # ---- evaluation ----
            acc = self.learner.evaluate(test_loader)
            self.metrics.append(acc)
            print(f"Accuracy after task {t}: {acc*100:.2f}%")

            # ---- checkpoint ----
            if self.output_dir is not None:
                torch.save(
                    {
                        "state_dict": self.learner.state_dict(),
                        "task": t,
                        "acc": acc,
                    },
                    self.output_dir / f"model_task{t}.pt",
                )

        # ---------------- summary ----------------
        print("\nFinished! Accuracies per task:")
        for t, a in enumerate(self.metrics):
            print(f"  Task {t}: {a*100:.2f}%")


# ----------------------------------------------------------------------------
# 3. CLI / smoke-test
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Continual Learning Trainer")
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--classes-per-task", "--cpt", type=int, default=2)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--strategy", choices=["finetune", "replay", "ewc"], default="finetune")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--buffer", type=int, default=1000)
    p.add_argument("--ewc", type=float, default=10.0)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--img-size", type=int, default=224)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- datasets ----
    train_tasks, test_tasks = build_split_datasets(
        args.dataset,
        classes_per_task=args.classes_per_task,
        img_size=args.img_size,
    )

    # ---- model + learner ----
    backbone = get_backbone(args.backbone)
    model = Classifier(backbone, num_classes=args.classes_per_task)  # head per task

    learner = build_learner(
        args.strategy,
        model,
        buffer_size=args.buffer,
        ewc_lambda=args.ewc,
        lr=1e-3,
    )

    trainer = Trainer(
        learner,
        train_tasks,
        test_tasks,
        device,
        classes_per_task=args.classes_per_task,
        epochs=args.epochs,
        batch_size=args.batch,
        output_dir=args.output,
    )
    trainer.run()


if __name__ == "__main__":
    main()
