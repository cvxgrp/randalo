import numpy as np
import hashlib
import json
import os

with open("base_config.json") as f:
    config = json.load(f)

os.makedirs("configs", exist_ok=True)

for seed in range(1):
    config["seed"] = seed

    for lamda in [1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]:
        config["method"]["lamda"] = lamda

        for gamma in [1e-4, 3e-5, 1e-5, 3e-6, 1e-6]:
            config["method"]["kernel_fun_kwargs"]["gamma"] = gamma

            id = hashlib.sha256(json.dumps(config).encode()).hexdigest()[:8]
            config["id"] = id

            with open(os.path.join("configs", f"run_{id}.json"), "w") as f:
                json.dump(config, f, indent=4)
