#!/usr/bin/env python3
"""subspace_similarity.py  – rápido y sin OOM."""
from __future__ import annotations
import argparse, functools, json, sys
from pathlib import Path
from typing import Dict, List, Optional

import torch, yaml
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import build_split_datasets
from models import Classifier, get_backbone

mp.set_start_method("spawn", force=True)

# ---------- collate ----------
def collate_map(batch, mapping: Optional[dict]):
    xs, ys = zip(*batch)
    xs = torch.stack(xs)          # CPU
    ys = torch.tensor([mapping.get(int(y), int(y)) for y in ys])
    return xs, ys                 # ambos CPU

# ---------- loader ----------
def build_loader(cfg, task_id, batch, device):
    _, tests = build_split_datasets(cfg["dataset"], cfg["classes_per_task"],
                                    img_size=cfg.get("img_size"))
    subset = tests[task_id]
    base = task_id * cfg["classes_per_task"]
    mapping = {base+i: i for i in range(cfg["classes_per_task"])}

    return DataLoader(subset,
                      batch_size=batch,
                      shuffle=False,
                      num_workers=0,
                      pin_memory=(device.type=="cuda"),
                      collate_fn=functools.partial(collate_map, mapping=mapping),
                      drop_last=False)

# ---------- hooks ----------
def reg_hooks(model, layers):
    acts: Dict[str, List[torch.Tensor]] = {n: [] for n in layers}
    def mk(n):
        def h(_, __, o): acts[n].append(o.flatten(1) if o.dim()>2 else o)
        return h
    hds=[m.register_forward_hook(mk(n)) for n,m in model.named_modules() if n in layers]
    return acts, hds

# ---------- PCA basis ----------
def topk_cpu(X: torch.Tensor, k: int):
    X = X.cpu()
    if hasattr(torch.linalg, "svd_lowrank"):
        # streaming, memoria O(k·p)
        _, _, V = torch.linalg.svd_lowrank(X, q=k)
    else:
        _, _, V = torch.linalg.svd(X, full_matrices=False)
    return V[:k].T        # p×k

def subspace(V,U,k): return (1/k)*torch.norm(V.T@U,p='fro').pow(2).item()

# ---------- activaciones ----------
def collect(model, loader, layers, sam, device):
    model.eval()
    acts, hds = reg_hooks(model, layers)
    n=0
    with torch.no_grad():
        for x,_ in loader:
            x=x.to(device,non_blocking=True)
            _=model(x); n+=x.size(0)
            if n>=sam: break
    for h in hds: h.remove()
    out={}
    for n,lst in acts.items():
        X=torch.cat(lst,0)[:sam]; X-=X.mean(0,keepdim=True)
        out[n]=X
    return out

# ---------- experimento único ----------
def one(args, cfg, dev, layers, cache):
    def load(idx):
        if idx in cache: return cache[idx]
        sd_all=torch.load(args.method/f"ckpt_t{idx}.pt",map_location=dev)
        sd=sd_all.get("state_dict",sd_all)
        sd={k.replace("model.","",1):v for k,v in sd.items() if k.startswith("model.")}
        m=Classifier(get_backbone(cfg["backbone"]),cfg["classes_per_task"]).to(dev)
        m.load_state_dict(sd,strict=False); m.eval(); cache[idx]=m; return m

    if args.ckpt_task is not None:
        m=load(args.ckpt_task)
        A=collect(m, build_loader(cfg,args.task_a,args.batch,dev), layers,args.samples,dev)
        B=collect(m, build_loader(cfg,args.task_b,args.batch,dev), layers,args.samples,dev)
    else:
        A=collect(load(args.ckpt_a), build_loader(cfg,args.task,args.batch,dev), layers,args.samples,dev)
        B=collect(load(args.ckpt_b), build_loader(cfg,args.task,args.batch,dev), layers,args.samples,dev)

    sim={n:subspace(topk_cpu(A[n],args.k), topk_cpu(B[n],args.k), args.k) for n in A}
    args.json_out.write_text(json.dumps({"k":args.k,"layers":sim},indent=2))
    print(args.json_out.name," ".join(f"{n}:{v:.3f}" for n,v in sim.items()))

# ---------- CLI ----------
p=parse=lambda: argparse.ArgumentParser(). \
    add_argument; args=parse
def parse():
    pa=argparse.ArgumentParser()
    pa.add_argument("--config"); pa.add_argument("--method",type=Path)
    g=pa.add_mutually_exclusive_group()
    g.add_argument("--ckpt-task",type=int); g.add_argument("--ckpt-a",type=int)
    pa.add_argument("--ckpt-b",type=int); h=pa.add_mutually_exclusive_group()
    h.add_argument("--task",type=int); h.add_argument("--task-a",type=int)
    pa.add_argument("--task-b",type=int)
    pa.add_argument("--k",type=int,default=10); pa.add_argument("--samples",type=int,default=1000)
    pa.add_argument("--batch",type=int,default=128)
    pa.add_argument("--json-out",type=Path,default=Path("subspace.json"))
    return pa.parse_args()
args=parse(); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- YAML plan o manual ----------
if args.config:
    plan=yaml.safe_load((Path("configs")/f"{args.config}.yml").read_text())
    args.method=Path(plan["method"])
    cfg=yaml.safe_load((args.method/"config_used.yaml").read_text())
    layers=["backbone.conv1","backbone.layer1","backbone.layer2","backbone.layer3","backbone.layer4"]
    cache={}
    for m in tqdm(plan["modes"],desc="Experimentos"):
        sub=argparse.Namespace(**vars(args))
        sub.k=m.get("k",10); sub.samples=m.get("samples",1000)
        sub.json_out=args.method/m["json_out"]
        sub.batch=args.batch
        if m["type"]=="intra":
            sub.ckpt_task,sub.task_a,sub.task_b=m["ckpt_task"],m["task_a"],m["task_b"]
            sub.ckpt_a=sub.ckpt_b=sub.task=None
        else:
            sub.ckpt_a,sub.ckpt_b,sub.task=m["ckpt_a"],m["ckpt_b"],m["task"]
            sub.ckpt_task=sub.task_a=sub.task_b=None
        one(sub,cfg,dev,layers,cache)
else:
    if args.ckpt_task and (args.task_a is None or args.task_b is None):
        sys.exit("falta --task-a/b")
    if args.ckpt_a and (args.ckpt_b is None or args.task is None):
        sys.exit("falta --ckpt-b/--task")
    cfg=yaml.safe_load((args.method/"config_used.yaml").read_text())
    layers=["backbone.conv1","backbone.layer1","backbone.layer2","backbone.layer3","backbone.layer4"]
    one(args,cfg,dev,layers,{})

