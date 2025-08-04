#!/usr/bin/env python3
"""
infer_cross_task.py — Evalúa checkpoints de aprendizaje continuo en tareas cruzadas.

Versión 4  ·  Agosto 2025
────────────────────────────────────────────────────────────────────────────
✓ Barra de progreso (tqdm)
✓ Registro JSON con los resultados de cada evaluación
✓ Soporte para CSV y PNG de la matriz de confusión
✓ NUEVO: puede tomar la *cabeza* (clasificador) de un modelo distinto
         especificándolo en el plan YAML mediante tripletas
         [model_task, eval_task, head_task]
         (head_task = None ⇒ usa su propia cabeza)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import yaml
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:                                 # tqdm no instalado → fallback
    def tqdm(x, *args, **kwargs):
        return x   # type: ignore

from datasets import build_split_datasets
from models   import Classifier, get_backbone

# ────────────────────────────────────────────────────────────────────────────
# 1. Utilidades
# ────────────────────────────────────────────────────────────────────────────
def _map_labels(y: torch.Tensor, mapping: Dict[int, int]) -> torch.Tensor:
    """Remapea las etiquetas usando un dict {original → nuevo_idx}."""
    return torch.tensor([mapping[int(lbl)] for lbl in y], dtype=torch.long)


def data_loader(subset, batch: int, mapping: Dict[int, int]) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    loader = DataLoader(subset, batch_size=batch, shuffle=False,
                        num_workers=2, pin_memory=True)
    for x, y in loader:
        yield x, _map_labels(y, mapping)


def confusion_matrix(n_classes: int,
                     y_true:   List[int],
                     y_pred:   List[int]) -> torch.Tensor:
    cm = torch.zeros(n_classes, n_classes, dtype=torch.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# ────────────────────────────────────────────────────────────────────────────
# 2. Checkpoint utils
# ────────────────────────────────────────────────────────────────────────────
def clean_state_dict(ckpt: dict) -> dict:
    """Elimina el prefijo 'model.' que añade Lightning si es necesario."""
    sd = ckpt.get("state_dict", ckpt)
    if any(k.startswith("model.") for k in sd):
        sd = {k.replace("model.", "", 1): v
              for k, v in sd.items() if k.startswith("model.")}
    return sd


# ────────────────────────────────────────────────────────────────────────────
# 3. Evaluación cruzada
# ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
@torch.no_grad()
def cross_eval(method_dir: Path,
               model_task: int,
               eval_task:  int,
               head_task:  int | None,
               device: torch.device,
               *,
               save_plot: bool = False) -> Tuple[float, Path]:

    # ───── 1. Config y dataset ──────────────────────────────────────────────
    cfg = yaml.safe_load((method_dir / "config_train_used.yaml").read_text())
    k   = cfg["classes_per_task"]

    _, test_tasks = build_split_datasets(cfg["dataset"], k,
                                         img_size=cfg.get("img_size"))
    test_subset = test_tasks[eval_task]
    mapping     = {orig: orig for orig in range(10)}
    loader      = data_loader(test_subset, batch=cfg["batch"], mapping=mapping)

    # ───── 2. Cargar checkpoints ────────────────────────────────────────────
    ckpt_base = torch.load(method_dir / f"ckpt_t{model_task}.pt", map_location=device)
    sd_base   = clean_state_dict(ckpt_base)

    if head_task is None:
        sd_head          = sd_base
        num_out_classes  = k
    else:
        ckpt_head        = torch.load(method_dir / f"ckpt_t{head_task}.pt",
                                      map_location=device)
        sd_head          = clean_state_dict(ckpt_head)
        num_out_classes  = sd_head["head.weight"].shape[0]   # 10

    # ───── 3. Construir modelo con salidas = num_out_classes ────────────────
    model = Classifier(get_backbone(cfg["backbone"]),
                       num_classes=num_out_classes).to(device)

    # → quitar head.* del state_dict base si no coincide con el tamaño nuevo
    if ("head.weight" in sd_base and
        sd_base["head.weight"].shape[0] != num_out_classes):
        sd_base = {k: v for k, v in sd_base.items()
                   if not k.startswith("head.")}

    model.load_state_dict(sd_base, strict=False)        # ahora sí

    # copiar PESOS y SESGO de la cabeza seleccionada
    with torch.no_grad():
        model.head.weight.copy_(sd_head["head.weight"])
        model.head.bias.copy_(sd_head["head.bias"])

    model.eval()

    # ───── 4. Inferencia ────────────────────────────────────────────────────
    y_true, y_pred = [], []
    for x, y in loader:
        y_true.extend(y.tolist())
        y_pred.extend(model(x.to(device)).argmax(1).cpu().tolist())

    cm  = confusion_matrix(num_out_classes, y_true, y_pred)
    acc = cm.diag().sum().item() / cm.sum().item()

    # ───── 5. Guardar resultados y (opcional) figura ───────────────────────
    suffix = f"_model{model_task}_on_task{eval_task}"
    if head_task is not None:
        suffix += f"_head{head_task}"
    out_csv = method_dir / f"confmat{suffix}.csv"
    pd.DataFrame(cm.numpy(), index=range(num_out_classes),
                 columns=range(num_out_classes)).to_csv(out_csv)

    if save_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm.numpy(), cmap="Blues")
        ax.set_xticks(range(num_out_classes))
        ax.set_yticks(range(num_out_classes))
        ax.set_xlabel("Predicción"); ax.set_ylabel("Etiqueta real")
        ax.set_title(f"T{model_task}→T{eval_task} (head={head_task})\nacc={acc:.2%}")
        for (i, j), v in np.ndenumerate(cm.numpy()):
            ax.text(j, i, int(v), ha="center", va="center",
                    color="white" if v > cm.max()/2 else "black", fontsize=8)
        fig.tight_layout()
        plt.savefig(out_csv.with_suffix(".png"), dpi=150)
        plt.close(fig)

    print(f"[{method_dir.name}] T{model_task}→T{eval_task}"
          f"{'' if head_task is None else f' (head={head_task})'} | acc={acc:.2%}")
    return acc, out_csv


# ────────────────────────────────────────────────────────────────────────────
# 4. Generador de experimentos
# ────────────────────────────────────────────────────────────────────────────
def parse_plan(path: Path) -> List[Tuple[Path, int, int, int | None]]:
    """
    Formatos admitidos en el YAML:

      experiments:
        - method: runs/mi_experimento
          pairs:   [[model_task, eval_task], …]          # cabeza propia
        - method: runs/otro
          triples: [[model_task, eval_task, head_task]]  # cabeza externa
    """
    plan = yaml.safe_load(path.read_text())
    experiments: list[tuple[Path, int, int, int | None]] = []

    for exp in plan.get("experiments", []):
        mdir = Path(exp["method"])
        if "triples" in exp:
            for mt, et, ht in exp["triples"]:
                experiments.append((mdir, int(mt), int(et), int(ht)))
        else:
            for mt, et in exp["pairs"]:
                experiments.append((mdir, int(mt), int(et), None))
    return experiments


# ────────────────────────────────────────────────────────────────────────────
# 5. CLI
# ────────────────────────────────────────────────────────────────────────────
def build_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-task inference con progreso y JSON")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan",   type=Path, help="Plan YAML de experimentos")
    g.add_argument("--auto",   action="store_true", help="Recorre runs/* y cruza T0↔T1")
    g.add_argument("--method", type=Path, help="Método individual (sin YAML)")

    p.add_argument("--model-task", type=int, default=0, help="Solo con --method")
    p.add_argument("--eval-task",  type=int, default=1, help="Solo con --method")
    p.add_argument("--head-task",  type=int, help="Copia la cabeza de esta tarea (solo con --method)")

    p.add_argument("--plot",      action="store_true", help="Guardar PNG de la matriz")
    p.add_argument("--json-out",  type=Path, default=Path("results.json"),
                   help="Archivo JSON para guardar precisión")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# 6. Main
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args   = build_cli()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Construir lista de experimentos ─────────────────────────────────────
    if args.plan:
        exps = parse_plan(args.plan)                                    # 4-tuplas
    elif args.auto:
        exps = [(mdir, mt, et, None)
                for mdir in Path("runs").iterdir() if mdir.is_dir()
                for mt, et in [(0, 1), (1, 0)]]
    else:  # --method
        exps = [(args.method, args.model_task, args.eval_task, args.head_task)]

    # ── Ejecutar ────────────────────────────────────────────────────────────
    results = []
    for mdir, mt, et, ht in tqdm(exps, desc="Experimentos"):
        acc, csv_path = cross_eval(mdir, mt, et, ht, device,
                                   save_plot=args.plot)
        results.append({
            "method":        str(mdir),
            "model_task":    mt,
            "eval_task":     et,
            "head_task":     ht,
            "accuracy":      round(acc, 4),
            "confusion_csv": str(csv_path)
        })

    # ── Guardar JSON ───────────────────────────────────────────────────────
    args.json_out.write_text(json.dumps({"experiments": results}, indent=2))
    print(f"\nResultados guardados en {args.json_out.resolve()}")


if __name__ == "__main__":
    main()
