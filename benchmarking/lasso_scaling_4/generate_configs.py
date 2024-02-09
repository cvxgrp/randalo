import hashlib
import json
import os

with open("base_config.json") as f:
    config = json.load(f)

n0 = 10
p0 = 10
s0 = 1

os.makedirs("configs", exist_ok=True)

for scale in [10, 20, 50, 100, 200, 500, 1000, 2000]:
    config["data"]["n_train"] = n0 * scale
    config["data"]["p"] = p = p0 * scale
    config["data"]["s"] = s0 * scale
    
    for lamda0 in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        
        config["method_kwargs"]["lamda0"] = lamda0

        for seed in range(10):
            config["seed"] = seed

            id = hashlib.sha256(json.dumps(config).encode()).hexdigest()[:8]
            config["id"] = id

            with open(os.path.join("configs", f"run_{id}.json"), "w") as f:
                json.dump(config, f, indent=4)
