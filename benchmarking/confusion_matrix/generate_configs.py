import numpy as np
import hashlib
import json
import os

with open("base_config.json") as f:
    config = json.load(f)

os.makedirs("configs", exist_ok=True)

for lamda0 in [10, 15]:

    config["method_kwargs"]["lamda0"] = lamda0

    for seed in range(1):
        config["seed"] = seed

        id = hashlib.sha256(json.dumps(config).encode()).hexdigest()[:8]
        config["id"] = id

        with open(os.path.join("configs", f"run_{id}.json"), "w") as f:
            json.dump(config, f, indent=4)
