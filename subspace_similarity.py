#!/usr/bin/env python3
"""
subspace_similarity.py – Layer-wise SubspaceSimₖ(X,Y)

Tres modos de comparación
-------------------------
1. intra   → mismo checkpoint, tareas distintas      (--ckpt-task  t, --task-a A --task-b B)
2. inter   → checkpoints distintos, misma tarea      (--ckpt-a  a --ckpt-b  b --task  T)
3. cross   → checkpoints y tareas diferentes         (--ckpt-a  a --task-a A  --ckpt-b  b --task-b B)

También puede leer un plan YAML:  --config <alias>  (ver ejemplo más abajo).
"""

from __future__ import annotations
import argparse, functools, json, sys
from pathlib import Path
from typing import Dict, List, Optional

import torch, yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import build_split_datasets
from models   import Classifier, get_backbone


# ───────────────────────────── Collate (CPU, picklable) ──────────────────
def _collate(batch, mapping: Optional[dict]):
    xs, ys = zip(*batch)
    xs = torch.stack(xs)                                       # CPU tensor
    ys = torch.tensor([mapping.get(int(y), int(y)) for y in ys])
    return xs, ys


# ───────────────────────────── DataLoader helper ─────────────────────────
def build_loader(cfg: Dict, task_id: int, batch: int):
    _, test_tasks = build_split_datasets(
        cfg["dataset"], cfg["classes_per_task"],
        img_size=cfg.get("img_size")
    )
    subset = test_tasks[task_id]

    base = task_id * cfg["classes_per_task"]
    mapping = {base + i: i for i in range(cfg["classes_per_task"])}

    return DataLoader(
        subset,
        batch_size=batch,
        shuffle=False,
        num_workers=0,                           # 0 → sin multiprocessing, 100 % seguro
        collate_fn=functools.partial(_collate, mapping=mapping),
        drop_last=False,
    )


# ───────────────────────────── Hooks y PCA ───────────────────────────────
def _register_hooks(model, layers):
    store: Dict[str, List[torch.Tensor]] = {n: [] for n in layers}

    def mk(name):
        def hook(_, __, out):
            store[name].append(out.flatten(1) if out.dim() > 2 else out)
        return hook

    hds = [m.register_forward_hook(mk(n))
           for n, m in model.named_modules() if n in layers]
    return store, hds


def _topk_cpu(X: torch.Tensor, k: int):
    X = X.cpu()
    if hasattr(torch.linalg, "svd_lowrank"):
        _, _, V = torch.linalg.svd_lowrank(X, q=k)
    else:
        _, _, V = torch.linalg.svd(X, full_matrices=False)
    return V[:k].T                                 # p × k


def _subspace_sim(V, U, k):
    return (1 / k) * torch.norm(V.T @ U, p='fro').pow(2).item()


# ───────────────────────────── Activaciones y similitud ──────────────────
def collect_acts(model, loader, layers, n_samples, device):
    model.eval()
    store, hds = _register_hooks(model, layers)
    seen = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            _ = model(x)
            seen += x.size(0)
            if seen >= n_samples:
                break
    for h in hds:
        h.remove()

    acts = {}
    for n, lst in store.items():
        X = torch.cat(lst, 0)[:n_samples]
        X -= X.mean(0, keepdim=True)
        acts[n] = X
    return acts


def layerwise_sim(A, B, k):
    return {n: _subspace_sim(_topk_cpu(A[n], k), _topk_cpu(B[n], k), k) for n in A}


# ───────────────────────────── Un experimento ────────────────────────────
def run_single(args, cfg, device, layers, cache: Dict[int, torch.nn.Module]):
    def load(ck_idx: int):
        if ck_idx in cache:
            return cache[ck_idx]
        ck_path = args.method / f"ckpt_t{ck_idx}.pt"
        sd_all = torch.load(ck_path, map_location=device)
        sd = sd_all.get("state_dict", sd_all)
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()
              if k.startswith("model.")}
        model = Classifier(get_backbone(cfg["backbone"]),
                           cfg["classes_per_task"]).to(device)
        model.load_state_dict(sd, strict=False)
        model.eval()
        cache[ck_idx] = model
        return model

    # seleccionar par (activaciones A, activaciones B)
    if args.ckpt_task is not None:                       # --- intra ---
        model = load(args.ckpt_task)
        A = collect_acts(model, build_loader(cfg, args.task_a, args.batch),
                         layers, args.samples, device)
        B = collect_acts(model, build_loader(cfg, args.task_b, args.batch),
                         layers, args.samples, device)
    elif args.task is not None:                          # --- inter ---
        model_a = load(args.ckpt_a); model_b = load(args.ckpt_b)
        loader = build_loader(cfg, args.task, args.batch)
        A = collect_acts(model_a, loader, layers, args.samples, device)
        B = collect_acts(model_b, loader, layers, args.samples, device)
    else:                                                # --- cross ---
        model_a = load(args.ckpt_a); model_b = load(args.ckpt_b)
        A = collect_acts(model_a, build_loader(cfg, args.task_a, args.batch),
                         layers, args.samples, device)
        B = collect_acts(model_b, build_loader(cfg, args.task_b, args.batch),
                         layers, args.samples, device)

    sims = layerwise_sim(A, B, args.k)
    args.json_out.write_text(json.dumps({"k": args.k, "layers": sims}, indent=2))
    print(args.json_out.name, " ".join(f"{n}:{v:.3f}" for n, v in sims.items()))


# ───────────────────────────── CLI parsing ───────────────────────────────
def _get_args():
    pa = argparse.ArgumentParser()
    pa.add_argument("--config")               # YAML alias

    pa.add_argument("--method", type=Path)
    g = pa.add_mutually_exclusive_group()
    g.add_argument("--ckpt-task", type=int)
    g.add_argument("--ckpt-a", type=int)
    pa.add_argument("--ckpt-b", type=int)

    h = pa.add_mutually_exclusive_group()
    h.add_argument("--task", type=int)
    h.add_argument("--task-a", type=int)
    pa.add_argument("--task-b", type=int)

    pa.add_argument("--k", type=int, default=10)
    pa.add_argument("--samples", type=int, default=1000)
    pa.add_argument("--batch", type=int, default=128)
    pa.add_argument("--json-out", type=Path, default=Path("subspace.json"))
    return pa.parse_args()


# ───────────────────────────── Entrypoint ────────────────────────────────
def main():
    args = _get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- plan YAML ----------
    if args.config:
        plan = yaml.safe_load((Path("Configs_similarity") / f"{args.config}.yml").read_text())
        args.method = Path(plan["method"])
        cfg = yaml.safe_load((args.method / "config_train_used.yaml").read_text())

        layers = (["backbone.conv1", "backbone.layer1", "backbone.layer2",
                   "backbone.layer3", "backbone.layer4"])

        cache: Dict[int, torch.nn.Module] = {}
        for m in tqdm(plan["modes"], desc="Experimentos"):
            sub = argparse.Namespace(**vars(args))   # copia base
            sub.k       = m.get("k", 10)
            sub.samples = m.get("samples", 1000)
            sub.json_out = args.method / m["json_out"]

            sub.ckpt_task = sub.task = None
            sub.ckpt_a = sub.ckpt_b = sub.task_a = sub.task_b = None

            if m["type"] == "intra":
                sub.ckpt_task = m["ckpt_task"]
                sub.task_a, sub.task_b = m["task_a"], m["task_b"]
            elif m["type"] == "inter":
                sub.ckpt_a, sub.ckpt_b = m["ckpt_a"], m["ckpt_b"]
                sub.task = m["task"]
            elif m["type"] == "cross":
                sub.ckpt_a, sub.task_a = m["ckpt_a"], m["task_a"]
                sub.ckpt_b, sub.task_b = m["ckpt_b"], m["task_b"]
            else:
                sys.exit(f"Tipo desconocido {m['type']}")

            run_single(sub, cfg, device, layers, cache)
        return

    # ---------- modo manual ----------
    if args.ckpt_task:                    # intra
        if args.task_a is None or args.task_b is None:
            sys.exit("falta --task-a/--task-b con --ckpt-task")
    elif args.ckpt_a:                     # inter o cross
        if args.ckpt_b is None:
            sys.exit("falta --ckpt-b")
        if args.task is None and (args.task_a is None or args.task_b is None):
            sys.exit("da --task   (inter)  o  --task-a/--task-b (cross)")
    else:
        sys.exit("usa --ckpt-task  o  (--ckpt-a --ckpt-b)")

    cfg = yaml.safe_load((args.method / "config_used.yaml").read_text())
    layers = (["backbone.conv1", "backbone.layer1", "backbone.layer2",
               "backbone.layer3", "backbone.layer4"])
    run_single(args, cfg, device, layers, {})


if __name__ == "__main__":
    main()
