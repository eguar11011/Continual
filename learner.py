#!/usr/bin/env python3
"""learner.py – Estrategias de Continual Learning y Fine‑Tune básico

Incluye tres modos listos para usar:

* **FinetuneLearner** – entrenamiento secuencial *naïve* (baseline).  No buffer,
  sin regularización: simplemente optimiza los parámetros con cada nueva tarea.

* **ReplayLearner** – mezcla un búfer de ejemplos anteriores (reservoir sampling)
  con el lote actual.

* **EwcLearner** – añade la regularización de *Elastic Weight Consolidation*
  (Fisher diagonal) para anclar los parámetros importantes de tareas pasadas.

Cada clase implementa:
    * `observe(batch)` – paso de entrenamiento sobre un lote.
    * `end_task(dataloader)` – hook opcional que se llama al terminar una tarea.

Las tres comparten una interfaz uniforme para que `trainer.py` pueda cambiar de
estrategia con una única línea.

Bloque *smoke‑test* disponible:

```bash
# Fine‑tune naïve
python learner.py --strategy finetune
# Replay de 200 ejemplos
python learner.py --strategy replay --buffer 200
# EWC con λ=5
python learner.py --strategy ewc --ewc 5
```
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# 0.  Utils comunes ----------------------------------------------------------
# ---------------------------------------------------------------------------

Criterion = nn.CrossEntropyLoss()


def _reservoir_add(buffer: List[Tuple[torch.Tensor, torch.Tensor]], sample: Tuple[torch.Tensor, torch.Tensor], buf_size: int):
    """Añade (x,y) al búfer con *reservoir sampling*."""
    if len(buffer) < buf_size:
        buffer.append(sample)
    else:
        j = random.randint(0, len(buffer) - 1)
        if j < buf_size:
            buffer[j] = sample


def _sample_buffer(buffer: List[Tuple[torch.Tensor, torch.Tensor]], k: int, device: torch.device):
    idx = random.sample(range(len(buffer)), min(k, len(buffer)))
    xb, yb = zip(*[buffer[i] for i in idx])
    return torch.stack(xb).to(device), torch.stack(yb).to(device)


# ---------------------------------------------------------------------------
# 1.  Fine‑Tune naïve --------------------------------------------------------
# ---------------------------------------------------------------------------
class FinetuneLearner(nn.Module):
    """Entrenamiento secuencial sin mitigación de olvido (baseline)."""

    def __init__(self, model: nn.Module, lr: float = 1e-3, device: str | torch.device = "cuda", **_):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.optim = optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = Criterion

    def observe(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = (b.to(self.device) for b in batch)
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    def end_task(self, *_, **__):
        return  # nada que hacer

    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        self.model.eval()
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        return correct / total if total else 0.0


# ---------------------------------------------------------------------------
# 2.  ReplayLearner ----------------------------------------------------------
# ---------------------------------------------------------------------------
class ReplayLearner(FinetuneLearner):
    """Añade un buffer de *rehearsal* que se mezcla con el input corriente."""

    def __init__(self, model: nn.Module, buffer_size: int = 1000, **kw):
        super().__init__(model, **kw)
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.buffer_size = buffer_size

    def observe(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        if self.buffer:
            xb, yb = _sample_buffer(self.buffer, len(x) // 2, self.device)
            x = torch.cat([x, xb])
            y = torch.cat([y, yb])
        loss = super().observe((x, y))
        for xi, yi in zip(x.detach().cpu(), y.detach().cpu()):
            _reservoir_add(self.buffer, (xi, yi), self.buffer_size)
        return loss


# ---------------------------------------------------------------------------
# 3.  EwcLearner -------------------------------------------------------------
# ---------------------------------------------------------------------------
class EwcLearner(ReplayLearner):
    """Añade regularización EWC tras cada tarea (usa buffer opcional)."""

    def __init__(self, model: nn.Module, ewc_lambda: float = 10.0, **kw):
        super().__init__(model, **kw)
        self.ewc_lambda = ewc_lambda
        self.prev_params: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}

    def observe(self, batch):
        loss = super().observe(batch)  # forward + replay + grad
        if self.ewc_lambda > 0 and self.fisher:
            ewc_penalty = 0.0
            for n, p in self.model.named_parameters():
                if n in self.fisher:
                    ewc_penalty += (self.fisher[n] * (p - self.prev_params[n]).pow(2)).sum()
            add = self.ewc_lambda * ewc_penalty
            self.optim.zero_grad()
            add.backward()
            self.optim.step()
            loss += add.item()
        return loss

    @torch.no_grad()
    def end_task(self, dataloader: DataLoader, fisher_samples: int = 1024):
        self.prev_params = {n: p.clone().detach() for n, p in self.model.named_parameters()}
        self.fisher = defaultdict(torch.zeros_like)
        self.model.eval()
        cnt = 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            loss = Criterion(self.model(x), y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.pow(2)
            cnt += 1
            if cnt * x.size(0) >= fisher_samples:
                break
        for n in self.fisher:
            self.fisher[n] /= cnt


# ---------------------------------------------------------------------------
# 4.  Factory helper ---------------------------------------------------------
# ---------------------------------------------------------------------------

def build_learner(strategy: str, model: nn.Module, **kw):
    strategy = strategy.lower()
    if strategy == "finetune":
        return FinetuneLearner(model, **kw)
    elif strategy == "replay":
        return ReplayLearner(model, **kw)
    elif strategy == "ewc":
        return EwcLearner(model, **kw)
    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")


# ---------------------------------------------------------------------------
# 5.  Smoke‑test -------------------------------------------------------------
# ---------------------------------------------------------------------------

def _smoke_test(args):
    torch.manual_seed(0)
    from models import get_backbone, Classifier

    backbone = get_backbone(args.backbone)
    model = Classifier(backbone, num_classes=args.nc)
    learner = build_learner(args.strategy, model, buffer_size=args.buffer, ewc_lambda=args.ewc, lr=1e-3)

    def make_loader(cls_offset: int):
        x = torch.randn(256, 3, args.img, args.img)
        y = torch.randint(0, 2, (256,)) + cls_offset
        return DataLoader(list(zip(x, y)), batch_size=32, shuffle=True)

    loaders = [make_loader(0), make_loader(2)]
    for t, loader in enumerate(loaders):
        print(f"-- Task {t} ({args.strategy}) --")
        for epoch in range(2):
            losses = [learner.observe(batch) for batch in loader]
            print(f"  epoch={epoch}  loss={sum(losses)/len(losses):.4f}")
        learner.end_task(loader)
    print("Smoke‑test OK ✅")


def _parse_args():
    p = argparse.ArgumentParser(description="Smoke test de learner")
    p.add_argument("--strategy", choices=["finetune", "replay", "ewc"], default="finetune")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--img", type=int, default=64)
    p.add_argument("--nc", type=int, default=4, help="num classes")
    p.add_argument("--buffer", type=int, default=1000)
    p.add_argument("--ewc", type=float, default=10.0)
    return p.parse_args()


if __name__ == "__main__":
    _smoke_test(_parse_args())
