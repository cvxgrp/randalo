import numpy as np
import hashlib
import json
import os

with open("base_config.json") as f:
    config = json.load(f)

os.makedirs("configs", exist_ok=True)

for seed in range(2):
    config["seed"] = seed

    for lamda in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0]:
        config["method_kwargs"]["lamda"] = lamda

        for gamma in [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5]:
            config["method_kwargs"]["kernel_fun_kwargs"]["gamma"] = gamma

            id = hashlib.sha256(json.dumps(config).encode()).hexdigest()[:8]
            config["id"] = id

            with open(os.path.join("configs", f"run_{id}.json"), "w") as f:
                json.dump(config, f, indent=4)
