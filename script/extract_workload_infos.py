# script/extract_workload_infos.py
import json
from pathlib import Path
from algoperf.workloads.workloads import WORKLOADS, BASE_WORKLOADS, BASE_WORKLOADS_DIR, import_workload


FIELDS = {
    "target_metric_name",
    "validation_target_value",
    "max_allowed_runtime_sec",
    "step_hint",
}

res = {}

for workload_name in BASE_WORKLOADS:
    workload_path = WORKLOADS[workload_name]['workload_path']
    workload_class_name = WORKLOADS[workload_name]['workload_class_name']

    workload_path = Path(BASE_WORKLOADS_DIR) / f'{workload_path}_pytorch' / 'workload.py'

    wl = import_workload(
        workload_path=str(workload_path),
        workload_class_name=workload_class_name,
        workload_init_kwargs={},
    )

    res[workload_name] = {
        field: getattr(wl, field) for field in FIELDS if hasattr(wl, field)
    }

with open("workload_infos.json", "w") as f:
    json.dump(res, f, indent=2)

print("Saved workload_infos.json")
