from pathlib import Path
import alogcv
import alogcv.utils

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
        self.m = m
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

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for i, (
    (lamda, m),
    [obj],
) in enumerate(data.items()):
    xs = 1 / obj.data["ms"][0]
    ys = obj.data["risks"][0]
    params, _ = alogcv.utils.robust_poly_fit(xs, ys, 1)
    hat_ys = np.vander(xs, 2, True) @ params

    plt.plot(xs, ys, label=f"{m=} data", color=color_cycle[i])

    plt.plot(xs, hat_ys, label=f"{m=} fit", color=color_cycle[i], linestyle="dashed")

    for j in range(1, 10):
        break
        plt.plot(
            1 / obj.data["ms"][j],
            obj.data["risks"][j],
            # color=color_cycle[i]
        )


plt.title("n=1000, p=2000, LASSO")
plt.legend()
plt.xlabel("1/m")
plt.ylabel("risk with m samples")

plt.show()

data = fetch_data(
    lambda n, p, problem, lamda, exacts, m: n == 1000
    and p == 2000
    and problem == "lasso"
    and lamda > 0.1
    and lamda < 0.12,
    [
        "shape",
        "problem",
        "lamda",
    ],
)


color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
i = 0
for ((n, p), problem, lamda), [*alos] in data.items():
    for alo in alos:
        if alo.exact:
            exact = alo

    sorted_alos = list(sorted([(alo.m, alo) for alo in alos if alo is not exact]))
    ms = np.array([m for m, _ in sorted_alos])
    ys = np.array(
        [
            (alo.data["values"] - exact.data["exact"]) / exact.data["exact"]
            for _, alo in sorted_alos
        ]
    )
    rs = np.array(
        [
            np.linalg.norm(alo.data["res_m_to_risk_fit"], axis=1)
            for _, alo in sorted_alos
        ]
    )
    std_rs = np.array(
        [
            np.linalg.norm(alo.data["res_m_to_risk_fit"], axis=1)
            for _, alo in sorted_alos
        ]
    )
    std_ys = np.array([alo.data["values"] for _, alo in sorted_alos])

    for j, m in enumerate(ms):
        plt.scatter(
            rs[j, :],
            ys[j, :],
            label=f"{m=}",
        )

plt.title(f"n=1000, p=2000, LASSO, {lamda=}")
plt.legend()
plt.xlabel("Residual Magnitude")
plt.ylabel("Relate Error(BKSRisk, ExactRisk)")

plt.show()
