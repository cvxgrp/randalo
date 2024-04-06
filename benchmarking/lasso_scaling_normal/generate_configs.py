import hashlib
import json
import os

with open("base_config.json") as f:
    config = json.load(f)

n0 = 10
p0 = 10
s0 = 1

os.makedirs("configs", exist_ok=True)

for scale in [100, 500, 2000]:
    config["data"]["n_train"] = n0 * scale
    config["data"]["p"] = p = p0 * scale
    config["data"]["s"] = s0 * scale

    for seed in range(100):
        config["seed"] = seed

        id = hashlib.sha256(json.dumps(config).encode()).hexdigest()[:8]
        config["id"] = id

        with open(os.path.join("configs", f"run_{id}.json"), "w") as f:
            json.dump(config, f, indent=4)
