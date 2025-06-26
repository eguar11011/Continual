#!/usr/bin/env python3
"""trainer.py – Orquestador con guardado por-tarea y métricas completas

Novedades:
* Guarda **checkpoint** por tarea (`ckpt_t{t}.pt`)
* Exporta **predicciones y etiquetas** por tarea (`preds_task{t}.pt`)
* Calcula y guarda la **matriz de confusión** en CSV (`confmat_task{t}.csv`)
* Registra todas las métricas en un `metrics.json` al finalizar
* Imprime la configuración efectiva y la guarda como `config_train_used.yaml`
* ⏱️  Añade la duración total del run (segundos y HH:MM:SS) al YAML
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
from models import get_backbone, Classifier
from learner import build_learner

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
    def __init__(self, learner, train_tasks, test_tasks, device, k, epochs, batch, out: Path | None):
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
        return DataLoader(subset, batch_size=self.batch, shuffle=shuffle, num_workers=2, pin_memory=True)

    def _mapy(self, y, m):
        return torch.tensor([m[int(lbl)] for lbl in y], dtype=torch.long)

    def _mapped(self, loader, m):
        for x, y in loader:
            yield x, self._mapy(y, m)

    # --------------------------- run --------------------------------
    def run(self):
        for t, (ds_tr, ds_te) in enumerate(zip(self.train_tasks, self.test_tasks)):
            task_classes = list(range(t * self.k, (t + 1) * self.k))
            m = {orig: idx for idx, orig in enumerate(task_classes)}
            print(f"\n===== Task {t} | classes {task_classes} =====")

            te_loader = list(self._mapped(self._loader(ds_te, False), m))
            full_loader = self._mapped(self._loader(ds_tr, False), m)

            for epoch in range(self.epochs):
                tr_loader = self._mapped(self._loader(ds_tr, True), m)
                pbar = tqdm(tr_loader, desc=f"T{t} E{epoch+1}/{self.epochs}")
                for batch in pbar:
                    l = self.learner.observe(batch)
                    pbar.set_postfix(loss=f"{l:.4f}")

            self.learner.end_task(full_loader)

            # ------- evaluación / confusión -------
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
            cm = confusion_matrix(y_true, y_pred, self.k)

            # ------- guardado -------
            if self.out:
                torch.save({"state_dict": self.learner.state_dict()}, self.out / f"ckpt_t{t}.pt")
                torch.save({"y_true": y_true, "y_pred": y_pred}, self.out / f"preds_task{t}.pt")
                with open(self.out / f"confmat_task{t}.csv", "w", newline="") as f:
                    wr = csv.writer(f)
                    wr.writerow([""] + list(range(self.k)))
                    for i, row in enumerate(cm.tolist()):
                        wr.writerow([i] + row)

            print(f"Accuracy Task {t}: {acc*100:.2f}%")
            self.metrics.append({"task": t, "accuracy": acc})

        # métricas resumen
        if self.out:
            (self.out / "metrics.json").write_text(json.dumps(self.metrics, indent=2))
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
            raise FileNotFoundError(f"No se encontró '{args.config}' ni dentro de {CONFIG_DIR}/")
        cfg = _load_config(cfg_path)
        args = _merge(args, cfg)

    # ---------- ruta de salida ----------
    if args.output is None:
        args.output = Path(
            f"runs/{args.strategy}_clases-{args.classes_per_task}_{args.dataset}_epochs--{args.epochs}"
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

    # ---------- entrenamiento ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tasks, test_tasks = build_split_datasets(
        args.dataset, args.classes_per_task, img_size=args.img_size
    )
    model = Classifier(get_backbone(args.backbone), num_classes=args.classes_per_task)
    learner = build_learner(
        args.strategy, model, buffer_size=args.buffer, ewc_lambda=args.ewc, lr=1e-3
    )
    Trainer(
        learner, train_tasks, test_tasks, device,
        args.classes_per_task, args.epochs, args.batch, out_dir
    ).run()

    # ---------- detener cronómetro y guardar duración ----------
    elapsed = time.time() - start
    cfg_dict["duracion_segundos"] = round(elapsed, 2)
    cfg_dict["duracion_hms"] = str(datetime.timedelta(seconds=int(elapsed)))

    # ---------- guardar YAML final ----------
    (out_dir / "config_train_used.yaml").write_text(
        yaml.safe_dump(cfg_dict, sort_keys=False)
    )

    print(f"\n⏱️  Entrenamiento completado en {cfg_dict['duracion_hms']} ({elapsed:.2f} s)")

if __name__ == "__main__":
    main()
