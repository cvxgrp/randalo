import glob
import json
import os
import sys

import numpy as np
from matplotlib import pyplot as plt


def load_results(results_dir):

    aggregate_fn = os.path.join(results_dir, "aggregate.json")

    if os.path.isfile(aggregate_fn):
        with open(aggregate_fn, "r") as f:
            return json.load(f)

    results_files = glob.glob("run_*.json", root_dir=results_dir)
    results = []
    for results_file in results_files:
        with open(os.path.join(results_dir, results_file), "r") as f:
            results.append(json.load(f))

    with open(aggregate_fn, "w") as f:
        json.dump(results, f)

    return results


def deep_get(d, keys):
    for key in keys:
        d = d[key]
    return d


def group_results(results, axes_spec):
    """Group results into an array.

    Parameters
    ----------
    results : list of dict
        List of results dictionaries.
    axes_spec : list of tuple of lists
        List of axes specifications. Each specification is a tuple (keys, values)
        where keys is a list of strings that specify the keys to access the
        results dictionary and values is a list of values that the key should take
        to be indexed into the produced array.

    Returns
    -------
    result : list of list of ... of dict
        Array of results with shape given by the axes specification.
    """

    shape = tuple(len(axis[1]) for axis in axes_spec)
    results_array = np.empty(shape, dtype=object)

    for result in results:
        indices = tuple(axis[1].index(deep_get(result, axis[0])) for axis in axes_spec)
        results_array[indices] = result

    return results_array


def extract_results(grouped_results, keys, depth=0):
    """Extract results from a grouped array.

    Recursively extract results from a grouped array.

    Parameters
    ----------
    grouped_results : list of list of ... of dict
        Array of results.
    keys : list of strings
        List of keys to access the results dictionary.

    Returns
    -------
    results : np.ndarray
        Array of results extracted from the dictionaries.
    """

    if isinstance(grouped_results, dict):
        return deep_get(grouped_results, keys)
    else:
        results = [
            extract_results(result, keys, depth=depth + 1) for result in grouped_results
        ]
        if depth == 0:
            results = np.asarray(results)
        return results


def lasso_scaling_1():

    results = load_results(os.path.join("lasso_scaling_1", "results"))

    n_keys = ["config", "data", "n_train"]
    seed_keys = ["config", "seed"]

    ns = sorted(set(deep_get(result, n_keys) for result in results))
    seeds = sorted(set(deep_get(result, seed_keys) for result in results))
    cv_k = deep_get(results[0], ["config", "cv_k"])
    alo_m = deep_get(results[0], ["config", "alo_m"])

    results = group_results(results, [(n_keys, ns), (seed_keys, seeds)])

    gen_risks = extract_results(results, ["gen_risk"])
    test_risks = extract_results(results, ["test_risk"])

    cv_risks = np.stack(
        [extract_results(results, [f"cv_{k}_risk"]) for k in cv_k],
        axis=-1,
    )
    cv_times = np.stack(
        [extract_results(results, [f"cv_{k}_risk_time"]) for k in cv_k],
        axis=-1,
    )

    alo_exact_risks = extract_results(results, ["alo_exact_risk"])
    full_train_times = extract_results(results, ["full_train_time"])
    jac_times = extract_results(results, ["jac_time"])
    alo_exact_times = (
        full_train_times + jac_times + extract_results(results, ["alo_exact_time"])
    )

    alo_bks_risks = np.stack(
        [extract_results(results, [f"alo_{m}_bks_risk"]) for m in alo_m],
        axis=-1,
    )
    alo_matvec_times = np.stack(
        [extract_results(results, [f"alo_{m}_matvec_time"]) for m in alo_m],
        axis=-1,
    )
    alo_bks_times = (
        full_train_times[..., None]
        + jac_times[..., None]
        + alo_matvec_times
        + np.stack(
            [extract_results(results, [f"alo_{m}_bks_risk_time"]) for m in alo_m],
            axis=-1,
        )
    )
    alo_poly_risks = np.stack(
        [extract_results(results, [f"alo_{m}_poly_risk"]) for m in alo_m],
        axis=-1,
    )
    alo_poly_times = (
        full_train_times[..., None]
        + jac_times[..., None]
        + alo_matvec_times
        + np.stack(
            [extract_results(results, [f"alo_{m}_poly_risk_time"]) for m in alo_m],
            axis=-1,
        )
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
