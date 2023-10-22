from pathlib import Path

import re
import collections

import matplotlib.pyplot as plt
import numpy as np

pattern = re.compile(
    "n([0-9]*)-p([0-9]*)-(ridge|lasso)-lamda([0-9]*.[0-9]*)-(exacts|m([0-9]*)).npz"
)


class ALORun:
    def __init__(self, shape, problem, lamda, m, file):
        self.exact = m is None
        self.shape = shape
        self.problem = problem
        self.lamda = lamda
        self.data = np.load(file)


def fetch_data(predicate, groupby):
    data = collections.defaultdict(list)
    for file in Path("data").iterdir():
        match = pattern.match(file.name)
        n, p, problem, lamda, exacts, m = match.groups()
        n = int(n)
        p = int(p)
        lamda = float(lamda)
        m = int(m) if m is not None else None
        if not predicate(n, p, problem, lamda, exacts == "exacts", m):
            continue
        key = ()
        if "shape" in groupby:
            key += ((n, p),)
        if "problem" in groupby:
            key += (problem,)
        if "lamda" in groupby:
            key += (lamda,)
        if "m" in groupby:
            key += (m,) if m is not None else ("exacts",)

        data[key].append(ALORun((n, p), problem, lamda, m, file))
    return data


data = fetch_data(
    lambda n, p, problem, lamda, exacts, m: n == 1000
    and p == 2000
    and problem == "lasso"
    and not exacts
    and lamda > 0.1
    and lamda < 0.12,
    ["lamda", "m"],
)

for (lamda, m), [obj] in data.items():
    plt.errorbar(
        1 / obj.data["ms"].mean(axis=0),
        obj.data["risks"].mean(axis=0),
        yerr=obj.data["risks"].std(axis=0),
        label=f"{m=},{lamda=}",
    )

plt.title("n=1000, p=2000, LASSO")
plt.legend()
plt.xlabel("1/m")
plt.ylabel("risk with m samples")

plt.show()
