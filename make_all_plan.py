#!/usr/bin/env python3
"""Crea un plan YAML: todas las parejas de checkpoints (0‑9) **solo para la tarea final (9)**."""

import itertools, pathlib, yaml

METHOD_DIR = "runs/finetune_clases-2_cifar10_epochs--30"
OUT_FILE   = pathlib.Path("Configs_similarity/finetune_cifar10_2cl_last_model.yml")

plan = {"method": METHOD_DIR, "modes": []}
#task = 4                                           # tarea final
model = 4
for ck_a, ck_b in itertools.combinations(range(5), 2):   # 45 pares 0‑1, 0‑2, …, 8‑9
    plan["modes"].append({
        "type":     "cross",
        "ckpt_a":   model,
        "task_a":   ck_a,
        "ckpt_b":   model,
        "task_b":   ck_b,
        "k":        20,
        "samples":  1000,
        "json_out": f"sim_ck{model}_t{ck_a}_vs_ck{model}_t{ck_b}.json"
    })

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUT_FILE.write_text(yaml.dump(plan, sort_keys=False))
print(f"Plan guardado en {OUT_FILE}")
