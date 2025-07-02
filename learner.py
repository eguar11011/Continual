#!/usr/bin/env python3
"""
learner.py – Estrategias de Continual Learning

Modos incluidos
───────────────
* FinetuneLearner – entrenamiento secuencial naïve (sin mitigación).
* ReplayLearner  – búfer balanceado por clase que crece por tarea.
* EwcLearner     – Elastic Weight Consolidation + replay balanceado.

Cada clase implementa:
    • observe(batch)     – paso de entrenamiento.
    • end_task(dataloader) – hook al terminar cada tarea.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# ────────────────────────────────
# 0. Utilidades comunes
# ────────────────────────────────
Criterion = nn.CrossEntropyLoss()

# ────────────────────────────────
# 1. Finetune naïve
# ────────────────────────────────
class FinetuneLearner(nn.Module):
    """Entrenamiento secuencial sin mitigación del olvido (baseline)."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: str | torch.device = "cuda",
        **_,
    ):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.optim = optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = Criterion

    # --------------------- entrenamiento por lote ---------------------
    def observe(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = (b.to(self.device) for b in batch)
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    # --------------------- gestión de nuevas clases -------------------
    def add_classes(self, n_new: int):
        """
        Expande la cabeza SIN reiniciar el optimizador.

        • Detecta qué parámetros son “nuevos” tras llamar a
          `model.add_classes(n_new)` y los añade como otro param-group
          al AdamW que ya existe, conservando los estados de momento
          del resto de pesos.
        """
        if n_new <= 0:
            return

        # 1) referencia a los parámetros actuales *antes* de crecer
        old_param_ids = {id(p) for p in self.model.parameters()}

        # 2) hace crecer la cabeza
        self.model.add_classes(n_new)

        # 3) localiza los parámetros añadidos
        new_params = [
            p for p in self.model.parameters() if id(p) not in old_param_ids
        ]

        # 4) los incorpora al optimizador manteniendo su estado anterior
        if new_params:  # evita error si n_new == 0
            self.optim.add_param_group({"params": new_params})

    # --------------------- hook fin de tarea --------------------------
    def end_task(self, *_):
        pass  # no hace nada

    # --------------------- evaluación rápida --------------------------
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


# ────────────────────────────────
# 2. Replay balanceado
# ────────────────────────────────
class ReplayLearner(FinetuneLearner):
    """
    Búfer balanceado por clase.

    buffer: dict  cls_id → list[(x_cpu, y_cpu)]
    Cada clase mantiene ≤ slots_per_class elementos.
    """

    def __init__(self, model: nn.Module, buffer_size: int = 1000, **kw):
        super().__init__(model, **kw)
        self.buffer: DefaultDict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = (
            defaultdict(list)
        )
        self.buffer_size = buffer_size
        self.slots_per_class = 0  # se calcula tras cada tarea

    # --------------------- entrenamiento por lote ---------------------
    def observe(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # mezcla con replay
        if any(self.buffer.values()):
            xb, yb = self._sample_replay(len(x) // 2)
            x = torch.cat([x, xb])
            y = torch.cat([y, yb])

        return super().observe((x, y))

    # --------------------- muestreo balanceado ------------------------
    def _sample_replay(self, n: int):
        """Devuelve ≤ n ejemplos balanceados entre clases."""
        classes = [c for c, samples in self.buffer.items() if samples]
        per_cls = max(1, n // max(1, len(classes)))

        chosen: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for c in classes:
            k = min(per_cls, len(self.buffer[c]))
            chosen.extend(random.sample(self.buffer[c], k))

        # completa si faltan
        if len(chosen) < n:
            flat = [s for lst in self.buffer.values() for s in lst]
            chosen.extend(random.sample(flat, n - len(chosen)))

        xb, yb = zip(*chosen)
        return torch.stack(xb).to(self.device), torch.stack(yb).to(self.device)

    # --------------------- hook fin de tarea --------------------------
    @torch.no_grad()
    def end_task(self, dataloader: DataLoader, *_, **__):
        """Al cerrar una tarea, actualiza el búfer balanceado."""
        total_classes = self.model.head.out_features
        self.slots_per_class = max(1, self.buffer_size // total_classes)

        # recorta clases antiguas si sobran
        for c in list(self.buffer.keys()):
            if len(self.buffer[c]) > self.slots_per_class:
                self.buffer[c] = random.sample(self.buffer[c], self.slots_per_class)

        # recolecta nuevos ejemplos
        candidates: DefaultDict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = (
            defaultdict(list)
        )
        for x, y in dataloader:
            for xi, yi in zip(x, y):
                c = yi.item()
                if len(candidates[c]) < self.slots_per_class:
                    candidates[c].append((xi.cpu(), yi.cpu()))
        # fusiona
        for c, lst in candidates.items():
            self.buffer[c].extend(lst)
            if len(self.buffer[c]) > self.slots_per_class:
                self.buffer[c] = random.sample(self.buffer[c], self.slots_per_class)


# ────────────────────────────────
# 3. EWC + Replay
# ────────────────────────────────
class EwcLearner(ReplayLearner):
    """Elastic Weight Consolidation + búfer replay balanceado."""

    def __init__(self, model: nn.Module, ewc_lambda: float = 10.0, **kw):
        super().__init__(model, **kw)
        self.ewc_lambda = ewc_lambda
        self.prev_params: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}

    # --------------------- entrenamiento por lote ---------------------
    def observe(self, batch):
        loss = super().observe(batch)  # incluye replay
        if self.ewc_lambda > 0 and self.fisher:
            penalty = 0.0
            for n, p in self.model.named_parameters():
                if n in self.fisher:
                    penalty += (self.fisher[n] * (p - self.prev_params[n]).pow(2)).sum()
            add = self.ewc_lambda * penalty
            self.optim.zero_grad()
            add.backward()
            self.optim.step()
            loss += add.item()
        return loss

    # --------------------- hook fin de tarea --------------------------
    @torch.no_grad()
    def end_task(self, dataloader: DataLoader, fisher_samples: int = 1024):
        # guarda parámetros
        self.prev_params = {
            n: p.clone().detach() for n, p in self.model.named_parameters()
        }
        self.fisher = {}
        self.model.eval()

        # estima Fisher diagonal
        cnt = 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad(set_to_none=True)
            Criterion(self.model(x), y).backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher.setdefault(n, torch.zeros_like(p.grad))
                    self.fisher[n] += p.grad.pow(2)
            cnt += 1
            if cnt * x.size(0) >= fisher_samples:
                break
        for n in self.fisher:
            self.fisher[n] /= cnt

        # actualiza buffer balanceado
        super().end_task(dataloader)


# ────────────────────────────────
# 4. Factory helper
# ────────────────────────────────
def build_learner(strategy: str, model: nn.Module, **kw):
    strategy = strategy.lower()
    if strategy == "finetune":
        return FinetuneLearner(model, **kw)
    if strategy == "replay":
        return ReplayLearner(model, **kw)
    if strategy == "ewc":
        return EwcLearner(model, **kw)
    raise ValueError(f"Estrategia desconocida: {strategy}")


# ────────────────────────────────
# 5. Smoke-test
# ────────────────────────────────
def _smoke_test(args):
    torch.manual_seed(0)
    from models import get_backbone, Classifier

    backbone = get_backbone(args.backbone)
    model = Classifier(backbone, num_classes=args.nc)
    learner = build_learner(
        args.strategy,
        model,
        buffer_size=args.buffer,
        ewc_lambda=args.ewc,
        lr=1e-3,
    )

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
    print("Smoke-test OK ✅")


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
