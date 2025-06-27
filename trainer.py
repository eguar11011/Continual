#!/usr/bin/env python3
"""trainer.py – Orquestador CL con clasificador expandible y métricas completas

Características
───────────────
* Clasificador que CRECE por tarea (`ExpandableClassifier`)
* Guarda **checkpoint** por tarea (`ckpt_t{t}.pt`)
* Exporta **preds+labels** por tarea (`preds_task{t}.pt`)
* Matriz de confusión CSV por tarea (`confmat_task{t}.csv`)
* Evaluación GLOBAL sobre la unión de tareas (`confmat_global.csv`)
* Registro de métricas en `metrics.json`
* Guarda la configuración efectiva en `config_train_used.yaml`
* ⏱️  Añade duración total (seg y HH:MM:SS) al YAML
"""
from __future__ import annotations

import argparse, json, csv, time, datetime
from pathlib import Path
from typing import List, Dict

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import build_split_datasets
from learner import build_learner
from models import get_backbone, ExpandableClassifier

CONFIG_DIR = Path(__file__).parent / "Configs_Trainer"

# ------------------------------------------------------------------ helpers
def _load_config(path: Path) -> Dict:
    ext = path.suffix.lower()
    if ext in {".yml", ".yaml"}:
        return yaml.safe_load(path.read_text()) or {}
    if ext == ".json":
        return json.loads(path.read_text()) or {}
    raise ValueError("Config debe ser .json, .yml o .yaml")

def _merge(cli: argparse.Namespace, cfg: Dict):
    """Fusiona parámetros CLI con el archivo de configuración."""
    for k, v in cfg.items():
        k_attr = k.replace("-", "_")
        if hasattr(cli, k_attr):
            if cli.__dict__[k_attr] == getattr(cli, k_attr):
                setattr(cli, k_attr, v)
    return cli

# ------------------------------------------------------------ confusión
def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, ncls: int):
    cm = torch.zeros((ncls, ncls), dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

# ----------------------------------------------------------------- Trainer
class Trainer:
    def __init__(self, learner, train_tasks, test_tasks,
                 device, k, epochs, batch, out: Path | None):
        self.learner = learner
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        self.device = device
        self.k = k
        self.epochs = epochs
        self.batch = batch
        self.out = out
        if self.out:
            self.out.mkdir(parents=True, exist_ok=True)
        self.metrics: List[Dict] = []

    # --------------------------- internal utils --------------------
    def _loader(self, subset, shuffle: bool):
        return DataLoader(
            subset,
            batch_size=self.batch,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True,
        )

    # --------------------------- run --------------------------------
    def run(self):
        # -------- bucle por tarea --------
        for t, (ds_tr, ds_te) in enumerate(zip(self.train_tasks, self.test_tasks)):
            print(f"\n===== Task {t} | clases {list(range(t*self.k, (t+1)*self.k))} =====")

            # 1) expandir la cabeza antes de ver la nueva tarea
            self.learner.add_classes(self.k)

            te_loader   = self._loader(ds_te,  False)
            full_loader = self._loader(ds_tr,  False)

            # 2) entrenamiento
            for epoch in range(self.epochs):
                tr_loader = self._loader(ds_tr, True)
                pbar = tqdm(tr_loader, desc=f"T{t} E{epoch+1}/{self.epochs}")
                for batch in pbar:
                    loss = self.learner.observe(batch)
                    pbar.set_postfix(loss=f"{loss:.4f}")

            # 3) finalizar la tarea (p.ej. cálculo de Fisher en EWC)
            self.learner.end_task(full_loader)

            # 4) evaluación por tarea
            y_true, y_pred = [], []
            self.learner.model.eval()
            with torch.no_grad():
                for x, y in te_loader:
                    x = x.to(self.device)
                    preds = self.learner.model(x).argmax(1).cpu()
                    y_true.append(y)
                    y_pred.append(preds)
            y_true = torch.cat(y_true)
            y_pred = torch.cat(y_pred)
            acc = (y_true == y_pred).float().mean().item()

            ncls = int(max(y_true.max(), y_pred.max())) + 1
            cm   = confusion_matrix(y_true, y_pred, ncls)

            # 5) guardado por tarea
            if self.out:
                torch.save({"state_dict": self.learner.state_dict()},
                           self.out / f"ckpt_t{t}.pt")
                torch.save({"y_true": y_true, "y_pred": y_pred},
                           self.out / f"preds_task{t}.pt")
                with open(self.out / f"confmat_task{t}.csv", "w", newline="") as f:
                    wr = csv.writer(f)
                    wr.writerow([""] + list(range(ncls)))
                    for i, row in enumerate(cm.tolist()):
                        wr.writerow([i] + row)

            print(f"Accuracy Task {t}: {acc*100:.2f}%")
            self.metrics.append({"task": t, "accuracy": acc})

        # -------- evaluación GLOBAL (todas las tareas) --------
        y_true_all, y_pred_all = [], []
        self.learner.model.eval()
        with torch.no_grad():
            for ds_te in self.test_tasks:
                for x, y in self._loader(ds_te, False):
                    x = x.to(self.device)
                    preds = self.learner.model(x).argmax(1).cpu()
                    y_true_all.append(y)
                    y_pred_all.append(preds)
        y_true_all = torch.cat(y_true_all)
        y_pred_all = torch.cat(y_pred_all)
        acc_global = (y_true_all == y_pred_all).float().mean().item()
        n_total    = int(max(y_true_all.max(), y_pred_all.max())) + 1
        cm_global  = confusion_matrix(y_true_all, y_pred_all, n_total)

        print(f"\n🟢  Accuracy GLOBAL: {acc_global*100:.2f}%")
        self.metrics.append({"task": "global", "accuracy": acc_global})

        if self.out:
            with open(self.out / "confmat_global.csv", "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow([""] + list(range(n_total)))
                for i, row in enumerate(cm_global.tolist()):
                    wr.writerow([i] + row)

            # guardar métricas resumen
            (self.out / "metrics.json").write_text(
                json.dumps(self.metrics, indent=2)
            )

        # -------- impresión resumen --------
        print("\nResumen final:")
        for m in self.metrics:
            print(f"  Task {m['task']}: {m['accuracy']*100:.2f}%")

# ------------------------------------------------------------ CLI
def parse_args():
    p = argparse.ArgumentParser(description="CL Trainer con guardado de métricas")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--classes-per-task", type=int, default=2)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--strategy", choices=["finetune", "replay", "ewc"], default="finetune")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--buffer", type=int, default=1000)
    p.add_argument("--ewc", type=float, default=10.0)
    p.add_argument("--output", type=Path, default=None,
                   help="Ruta base de salida; si se omite se genera automáticamente")
    p.add_argument("--img-size", type=int, default=224)
    return p.parse_args()

# ------------------------------------------------------------ main
def main():
    args = parse_args()

    # ---------- localizar archivo de configuración ----------
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            cfg_path = CONFIG_DIR / cfg_path
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"No se encontró '{args.config}' ni dentro de {CONFIG_DIR}/"
            )
        cfg = _load_config(cfg_path)
        args = _merge(args, cfg)

    # ---------- ruta de salida ----------
    if args.output is None:
        args.output = Path(
            f"runs/{args.strategy}_clases-{args.classes_per_task}_"
            f"{args.dataset}_epochs-{args.epochs}"
        )
    else:
        args.output = Path(args.output)
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- diccionario de configuración ----------
    cfg_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    # ---------- inicio cronómetro ----------
    start = time.time()

    print("\n======= Configuración efectiva =======")
    print(yaml.safe_dump(cfg_dict, sort_keys=False).strip())

    # ---------- preparación de datos ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tasks, test_tasks = build_split_datasets(
        args.dataset, args.classes_per_task, img_size=args.img_size
    )

    # ---------- modelo + learner ----------
    model = ExpandableClassifier(get_backbone(args.backbone), num_classes=0)
    learner = build_learner(
        args.strategy,
        model,
        buffer_size=args.buffer,
        ewc_lambda=args.ewc,
        lr=1e-3,
    )

    Trainer(
        learner,
        train_tasks,
        test_tasks,
        device,
        args.classes_per_task,
        args.epochs,
        args.batch,
        out_dir,
    ).run()

    # ---------- detener cronómetro y guardar duración ----------
    elapsed = time.time() - start
    cfg_dict["duracion_segundos"] = round(elapsed, 2)
    cfg_dict["duracion_hms"] = str(datetime.timedelta(seconds=int(elapsed)))

    (out_dir / "config_train_used.yaml").write_text(
        yaml.safe_dump(cfg_dict, sort_keys=False)
    )

    print(
        f"\n⏱️  Entrenamiento completado en {cfg_dict['duracion_hms']} "
        f"({elapsed:.2f} s)"
    )

if __name__ == "__main__":
    main()
