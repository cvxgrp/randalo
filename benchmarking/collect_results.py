import glob
import json
import os
import sys

import numpy as np
from matplotlib import pyplot as plt


def lasso_scaling_1():

    results_dir = os.path.join("lasso_scaling_1", "results")
    results_files = glob.glob("run_*.json", root_dir=results_dir)
    results = []
    for results_file in results_files:
        with open(os.path.join(results_dir, results_file), "r") as f:
            results.append(json.load(f))

    ns = sorted(
        set(sorted([results["config"]["data"]["n_train"] for results in results]))
    )
    seeds = sorted(set(sorted([results["config"]["seed"] for results in results])))
    cv_k = results[0]["config"]["cv_k"]
    alo_m = results[0]["config"]["alo_m"]

    gen_risks = np.zeros((len(ns), len(seeds)))
    test_risks = np.zeros((len(ns), len(seeds)))
    cv_risks = np.zeros((len(ns), len(seeds), len(cv_k)))
    cv_times = np.zeros((len(ns), len(seeds), len(cv_k)))
    alo_exact_risks = np.zeros((len(ns), len(seeds)))
    alo_exact_times = np.zeros((len(ns), len(seeds)))
    alo_bks_risks = np.zeros((len(ns), len(seeds), len(alo_m)))
    alo_bks_times = np.zeros((len(ns), len(seeds), len(alo_m)))
    alo_poly_risks = np.zeros((len(ns), len(seeds), len(alo_m)))
    alo_poly_times = np.zeros((len(ns), len(seeds), len(alo_m)))

    for result in results:
        i = ns.index(result["config"]["data"]["n_train"])
        j = seeds.index(result["config"]["seed"])

        gen_risks[i, j] = result["gen_risk"]
        test_risks[i, j] = result["test_risk"]

        for l, k in enumerate(cv_k):
            cv_risks[i, j, l] = result[f"cv_{k}_risk"]
            cv_times[i, j, l] = result[f"cv_{k}_risk_time"]

        alo_exact_risks[i, j] = result["alo_exact_risk"]
        alo_exact_times[i, j] = (
            result["full_train_time"] + result["jac_time"] + result["alo_exact_time"]
        )

        for l, m in enumerate(alo_m):
            alo_bks_risks[i, j, l] = result[f"alo_{m}_bks_risk"]
            alo_bks_times[i, j, l] = (
                result["full_train_time"]
                + result["jac_time"]
                + result[f"alo_{m}_matvec_time"]
                + result[f"alo_{m}_bks_risk_time"]
            )
            alo_poly_risks[i, j, l] = result[f"alo_{m}_poly_risk"]
            alo_poly_times[i, j, l] = (
                result["full_train_time"]
                + result["jac_time"]
                + result[f"alo_{m}_matvec_time"]
                + result[f"alo_{m}_poly_risk_time"]
            )

    for i, k in enumerate(cv_k):
        if k in [2, 10]:
            continue
        plt.scatter(
            cv_times[:, :, i].mean(1),
            (np.abs(cv_risks[:, :, i] - gen_risks) / gen_risks).mean(1),
            c=np.log10(ns),
            marker="o",
            label=f"cv_{k}",
        )
    plt.scatter(
        alo_exact_times.mean(1),
        (np.abs(alo_exact_risks - gen_risks) / gen_risks).mean(1),
        c=np.log10(ns),
        marker="s",
        label="alo_exact",
    )
    for i, m in enumerate(alo_m):
        if m != 100:
            continue
        plt.scatter(
            alo_bks_times[:, :, i].mean(1),
            (np.abs(alo_bks_risks[:, :, i] - gen_risks) / gen_risks).mean(1),
            c=np.log10(ns),
            marker="x",
            label=f"alo_{m}_bks",
        )
        plt.scatter(
            alo_poly_times[:, :, i].mean(1),
            (np.abs(alo_poly_risks[:, :, i] - gen_risks) / gen_risks).mean(1),
            c=np.log10(ns),
            marker="v",
            label=f"alo_{m}_poly",
        )
        plt.colorbar(label="log10(n)")

    plt.title("Lasso Scaling")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("Relative Estimation Error")
    plt.legend()
    plt.show()


collect_mapping = {
    "lasso_scaling_1": lasso_scaling_1,
}


if __name__ == "__main__":

    benchmarks = sys.argv[1:]
    if len(benchmarks) == 0:
        benchmarks = collect_mapping.keys()

    for benchmark in benchmarks:
        if benchmark not in collect_mapping:
            raise ValueError(f"Unknown benchmark {benchmark}")
        collect_mapping[benchmark]()
