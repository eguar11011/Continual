"""
1. **TaskConfig** – metadata for each task (dataset paths, transforms, epochs…)
2. **ContinualLearner** – wrapper around the backbone model plus any CL-specific
   strategy (e.g. rehearsal buffer, regularisation, adapters).
3. **Trainer** – orchestration layer that loops over tasks, handles evaluation,
   logging, checkpointing and metric aggregation.

The goal is to give you a solid starting point that you can customise while
keeping everything in a single, self‑contained file that is easy to debug on
machines with limited GPU memory (e.g. your GTX 950Mx    with 4 GB vRAM).

Dependencies
------------
* Python ≥3.10 (3.11.9 recommended)
* PyTorch ≥2.2
* tqdm (progress bars)
* torchvision (for common CV datasets) – optional but handy

Usage (example)
---------------
$ python cl_experiment.py --config configs/split_cifar10.json --output runs/split_cifar10

The JSON config declares each task and general hyper‑parameters.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# 1.  Task configuration
# ---------------------------------------------------------------------------
@dataclass
class TaskConfig:
    name: str
    train_indices: List[int] | None = None  # indices inside the base dataset
    test_indices: List[int] | None = None
    epochs: int = 1
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TaskConfig":
        return TaskConfig(**d)


# ---------------------------------------------------------------------------
# 2.  Simple continual dataset wrapper (task‑incremental setting)
# ---------------------------------------------------------------------------
class ContinualDataset(Dataset):
    """Wraps a *base* dataset and serves only the indices belonging to one task."""

    def __init__(self, base: Dataset, indices: List[int]):
        self.base = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


# ---------------------------------------------------------------------------
# 3.  Continual learner model
# ---------------------------------------------------------------------------
class ContinualLearner(nn.Module):
    """Backbone + CL strategy.  Extend hooks to implement EWC, replay, etc."""

    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.fc.out_features, num_classes)
        # Example replay buffer (feature + label)
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.buffer_size = 500  # configurable

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

    # ---- Replay buffer helpers -------------------------------------------------
    def add_to_buffer(self, x: torch.Tensor, y: torch.Tensor):
        """Store a random subset of (x, y) examples."""
        for xi, yi in zip(x, y):
            if len(self.buffer) < self.buffer_size:
                self.buffer.append((xi.cpu(), yi.cpu()))
            else:
                # Reservoir sampling
                j = random.randint(0, len(self.buffer) - 1)
                if j < self.buffer_size:
                    self.buffer[j] = (xi.cpu(), yi.cpu())

    def sample_buffer(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.buffer) == 0:
            raise ValueError("Replay buffer is empty!")
        idx = random.sample(range(len(self.buffer)), min(n, len(self.buffer)))
        xb, yb = zip(*[self.buffer[i] for i in idx])
        return torch.stack(xb).to(xb[0].device), torch.stack(yb).to(yb[0].device)


# ---------------------------------------------------------------------------
# 4.  Trainer
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(
        self,
        model: ContinualLearner,
        tasks: List[TaskConfig],
        base_dataset: Dataset,
        device: torch.device,
        output_dir: Path,
        eval_every: int = 1,
    ) -> None:
        self.model = model.to(device)
        self.tasks = tasks
        self.base_dataset = base_dataset
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_every = eval_every

        self.criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------------------------
    def _get_loader(self, indices: List[int], task_cfg: TaskConfig, train: bool):
        ds = ContinualDataset(self.base_dataset, indices)
        return DataLoader(
            ds,
            batch_size=task_cfg.batch_size,
            shuffle=train,
            num_workers=2,
            pin_memory=True,
        )

    # ---------------------------------------------------------------------
    def train(self):
        for t, task_cfg in enumerate(self.tasks):
            print(f"\n=== Task {t} / {task_cfg.name} ===")
            train_loader = self._get_loader(task_cfg.train_indices, task_cfg, train=True)
            test_loader = self._get_loader(task_cfg.test_indices, task_cfg, train=False)

            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=task_cfg.lr,
                weight_decay=task_cfg.weight_decay,
            )

            # ------------------- Train loop ----------------------
            for epoch in range(task_cfg.epochs):
                self.model.train()
                pbar = tqdm(train_loader, desc=f"Task {t} | Epoch {epoch+1}/{task_cfg.epochs}")
                for x, y in pbar:
                    x, y = x.to(self.device), y.to(self.device)

                    # Mix with replay buffer (simple rehearsal)
                    if len(self.model.buffer) > 0:
                        xb, yb = self.model.sample_buffer(len(x) // 2)
                        x = torch.cat([x, xb])
                        y = torch.cat([y, yb])

                    logits = self.model(x)
                    loss = self.criterion(logits, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Add current batch to buffer at the end
                    self.model.add_to_buffer(x.detach(), y.detach())

                    pbar.set_postfix(loss=f"{loss.item():.4f}")

            # ------------------- Evaluation ----------------------
            if (t + 1) % self.eval_every == 0:
                acc = self.evaluate(test_loader)
                print(f"Accuracy after task {t}: {acc*100:.2f}%")
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "task": t,
                        "acc": acc,
                    },
                    self.output_dir / f"checkpoint_task{t}.pt",
                )

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        self.model.eval()
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# 5.  Utilities
# ---------------------------------------------------------------------------

def load_split_cifar10(root: str = "~/.torchvision"):
    """Example helper that creates a base CIFAR‑10 dataset and returns class indices."""
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as T

    tfm = T.Compose([T.ToTensor()])
    trainset = CIFAR10(root, train=True, download=True, transform=tfm)
    testset = CIFAR10(root, train=False, download=True, transform=tfm)

    # Build per‑class index map
    class_to_indices_train: Dict[int, List[int]] = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(trainset):
        class_to_indices_train[label].append(idx)

    class_to_indices_test: Dict[int, List[int]] = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(testset):
        class_to_indices_test[label].append(idx)

    return trainset, testset, class_to_indices_train, class_to_indices_test


# ---------------------------------------------------------------------------
# 6.  Entry‑point helpers
# ---------------------------------------------------------------------------

def build_experiment(cfg_path: Path, output_dir: Path):
    with cfg_path.open() as f:
        cfg = json.load(f)

    # ---- Example: CIFAR‑10 split into tasks of two classes each ----------
    if cfg["dataset"] == "split_cifar10":
        trainset, testset, train_map, test_map = load_split_cifar10()
        tasks: List[TaskConfig] = []
        for i in range(0, 10, 2):
            tasks.append(
                TaskConfig(
                    name=f"classes_{i}_{i+1}",
                    train_indices=train_map[i] + train_map[i + 1],
                    test_indices=test_map[i] + test_map[i + 1],
                    **cfg["task_defaults"],
                )
            )
        num_classes = 2  # per task head; change if using single head
    else:
        raise ValueError("Unknown dataset in config")

    # ---- Backbone --------------------------------------------------------
    from torchvision.models import resnet18

    backbone = resnet18(num_classes=cfg.get("feature_dim", 512))

    learner = ContinualLearner(backbone, num_classes=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        learner,
        tasks,
        trainset,  # using trainset for simplicity; merge with testset in practice
        device,
        output_dir,
        eval_every=1,
    )
    trainer.train()


# ---------------------------------------------------------------------------
# 7.  CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Continual Learning Experiment")
    p.add_argument("--config", type=Path, required=True, help="Path to JSON config file")
    p.add_argument("--output", type=Path, default=Path("runs"), help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    build_experiment(args.config, args.output)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# 8.  Minimal JSON config example (save as configs/split_cifar10.json)
# ---------------------------------------------------------------------------
# {
#     "dataset": "split_cifar10",
#     "feature_dim": 512,
#     "task_defaults": {
#         "epochs": 3,
#         "batch_size": 64,
#         "lr": 0.0005,
#         "weight_decay": 0.0001
#     }
# }
# ---------------------------------------------------------------------------
# TODO
# ----
# * Implement weight consolidation (EWC, MAS…) hooks inside ContinualLearner
# * Add support for class‑incremental evaluation (single growing head)
# * Plug‑in WandB or TensorBoard logging
# * Implement early stopping and LR schedulers per task
# * Replace replay buffer with sophisticated reservoir / herding strategies
# * Allow mixed‑precision (torch.cuda.amp) for memory‑limited GPUs
# * Provide more dataset helpers (e.g. Split MiniImagenet, CORe50)
# ---------------------------------------------------------------------------
