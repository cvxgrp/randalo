import hashlib
import json
import os

with open("base_config.json") as f:
    config = json.load(f)

n0 = config["data"]["n_train"]
p0 = config["data"]["p"]
s0 = config["data"]["s"]

os.makedirs("configs", exist_ok=True)

# for scale in [10, 20, 50, 100, 200, 500, 1000, 2000]:
for scale in [10, 20, 50, 100]:
    config["data"]["n_train"] = n0 * scale
    config["data"]["p"] = p = p0 * scale
    config["data"]["s"] = s0 * scale

    for seed in range(5):
        config["seed"] = seed

        id = hashlib.sha256(json.dumps(config).encode()).hexdigest()[:8]
        config["id"] = id

        with open(os.path.join("configs", f"run_{id}.json"), "w") as f:
            json.dump(config, f, indent=4)
