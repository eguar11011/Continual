#!/usr/bin/env python3
"""infer_cross_task.py – Evalúa checkpoints de aprendizaje continuo en tareas cruzadas.

Mejoras v3
----------
* **Barra de progreso** (`tqdm`) para saber cuántos experimentos faltan.
* **Registro JSON** con la precisión de cada evaluación (`--json-out`).
* Mantiene soporte para CSV y PNG.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Iterable, Tuple

import yaml
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm no instalado → definición mínima
    def tqdm(x, *args, **kwargs):
        return x  # type: ignore

from datasets import build_split_datasets
from models import Classifier, get_backbone

# ---------------------------------------------------------------------------
# 1. Utilidades
# ---------------------------------------------------------------------------

def _map_labels(y: torch.Tensor, mapping: Dict[int, int]) -> torch.Tensor:
    return torch.tensor([mapping[int(lbl)] for lbl in y], dtype=torch.long)


def data_loader(subset, batch: int, mapping: Dict[int, int]):
    loader = DataLoader(subset, batch_size=batch, shuffle=False,
                        num_workers=2, pin_memory=True)
    for x, y in loader:
        yield x, _map_labels(y, mapping)


def confusion_matrix(k: int, y_true: List[int], y_pred: List[int]) -> torch.Tensor:
    cm = torch.zeros(k, k, dtype=torch.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

# ---------------------------------------------------------------------------
# 2. Checkpoint utils
# ---------------------------------------------------------------------------

def clean_state_dict(ckpt: dict) -> dict:
    sd = ckpt.get("state_dict", ckpt)
    if any(k.startswith("model.") for k in sd):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items() if k.startswith("model.")}
    return sd

# ---------------------------------------------------------------------------
# 3. Evaluación cruzada
# ---------------------------------------------------------------------------
@torch.no_grad()
def cross_eval(method_dir: Path, model_task: int, eval_task: int, device: torch.device,
               *, save_plot: bool = False) -> Tuple[float, Path]:
    """Devuelve (accuracy, csv_path)."""

    cfg_path = method_dir / "config_used.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    k = cfg["classes_per_task"]

    # dataset
    _, test_tasks = build_split_datasets(cfg["dataset"], k, img_size=cfg.get("img_size"))
    test_subset = test_tasks[eval_task]
    mapping = {orig: idx for idx, orig in enumerate(range(eval_task * k, (eval_task + 1) * k))}
    loader = data_loader(test_subset, batch=cfg["batch"], mapping=mapping)

    # model
    ckpt = torch.load(method_dir / f"ckpt_t{model_task}.pt", map_location=device)
    model = Classifier(get_backbone(cfg["backbone"]), num_classes=k).to(device)
    model.load_state_dict(clean_state_dict(ckpt), strict=False)
    model.eval()

    # inference
    y_true, y_pred = [], []
    for x, y in loader:
        y_true.extend(y.tolist())
        y_pred.extend(model(x.to(device)).argmax(1).cpu().tolist())

    cm = confusion_matrix(k, y_true, y_pred)
    acc = cm.diag().sum().item() / cm.sum().item()

    # save CSV
    out_csv = method_dir / f"confmat_model{model_task}_on_task{eval_task}.csv"
    pd.DataFrame(cm.numpy(), index=range(k), columns=range(k)).to_csv(out_csv, index=True)

    # optional PNG
    if save_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm.numpy(), cmap="Blues")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Etiqueta real")
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_title(f"T{model_task}→T{eval_task} | acc={acc:.2%}")
        for (i, j), v in np.ndenumerate(cm.numpy()):
            ax.text(j, i, int(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        plt.savefig(out_csv.with_suffix(".png"), dpi=150)
        plt.close(fig)

    print(f"[{method_dir.name}] T{model_task}→T{eval_task} | acc={acc:.2%} | {out_csv.name}")
    return acc, out_csv

# ---------------------------------------------------------------------------
# 4. Generador de experimentos
# ---------------------------------------------------------------------------

def parse_plan(path: Path) -> List[Tuple[Path, int, int]]:
    plan = yaml.safe_load(path.read_text())
    experiments = []
    for exp in plan.get("experiments", []):
        mdir = Path(exp["method"])
        for mt, et in exp["pairs"]:
            experiments.append((mdir, int(mt), int(et)))
    return experiments

# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def build_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross‑task inference con progreso y JSON")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", type=Path, help="Plan YAML de experimentos")
    g.add_argument("--auto", action="store_true", help="Recorre runs/* y cruza T0↔T1")
    g.add_argument("--method", type=Path, help="Método individual")

    p.add_argument("--model-task", type=int, default=0)
    p.add_argument("--eval-task", type=int, default=1)
    p.add_argument("--plot", action="store_true", help="Guardar PNG de la matriz")
    p.add_argument("--json-out", type=Path, default=Path("results.json"),
                   help="Archivo JSON para guardar precisión")
    return p.parse_args()

# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_cli()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []  # acumula dicts para JSON

    # ----- construir lista de experimentos -----
    if args.plan:
        exps = parse_plan(args.plan)
    elif args.auto:
        exps = [(mdir, mt, et)
                for mdir in Path("runs").iterdir() if mdir.is_dir()
                for mt, et in [(0, 1), (1, 0)]]
    else:
        exps = [(args.method, args.model_task, args.eval_task)]

    # ----- bucle con barra de progreso -----
    for mdir, mt, et in tqdm(exps, desc="Experimentos"):
        acc, csv_path = cross_eval(mdir, mt, et, device, save_plot=args.plot)
        results.append({
            "method": str(mdir),
            "model_task": mt,
            "eval_task": et,
            "accuracy": round(acc, 4),
            "confusion_csv": str(csv_path)
        })

    # ----- guardar JSON -----
    args.json_out.write_text(json.dumps({"experiments": results}, indent=2))
    print(f"\nResultados guardados en {args.json_out.resolve()}")


if __name__ == "__main__":
    main()
