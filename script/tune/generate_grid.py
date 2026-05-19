"""
Generate a cartesian hyperparameter gird list from a dictionary.
"""

import json
from pathlib import Path

from algoperf import halton


OUT_PATH = 'script/tune/search_spaces'
OUT_NAME = 'sweep_05.json'
OVERRIDE = False

sweep_dict = {
    "learning_rate": [
        1e-5, 1e-4, 1e-3, 1e-2,
        # 1e-2,
        # 1e-5, 5e-5, 2.5e-4, 1.25e-3, 6.25e-3, #3.125e-2, # log5
        # 1e-5, 4e-5, 1.6e-4, 6.4e-4, 2.56e-3, 1.024e-2, # log4
    ],
    "weight_decay": 0.0,

    # Muon
    "muon_beta": 0.9,
    "muon_adjust_lr": ['spectral_norm', 'match_adam'],
    "muon_nesterov": True,
    "muon_ns_steps": 5,
    "muon_ns_eps": 1e-7,

    # AdamW
    "adamw_beta1": 0.9,
    "adamw_beta2": 0.999,
    "adamw_eps": 1e-8,

    "dropout_rate": 0.0,
    "label_smoothing": 0.0,
    "warmup_factor": 0.05,

    # Step reduce
    "step_reduce": [1.0, 0.8],
}

sweeps = []
for k, v in sweep_dict.items():
    if isinstance(v, list):
        sweeps.append(halton.sweep(k, halton.discrete(v)))
    else:
        sweeps.append(halton.sweep(k, halton.discrete([v])))

grid = halton.product(sweeps)
print(f"Cartesian product results in {len(grid)} HP points.")

path = Path(OUT_PATH) / OUT_NAME

if path.exists() and not OVERRIDE:
    raise FileExistsError('Found esisting path.')
with path.open("w") as f:
    json.dump(grid, f, indent=2)
