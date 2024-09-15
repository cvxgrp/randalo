from dataclasses import dataclass
import glob
import json
import os
import sys

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt, gridspec

try:
    import cmasher as cmr
except ImportError:
    cmr = None


def setup_matplotlib():
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}\usepackage{bm}")
    plt.rcParams["figure.figsize"] = [6.5, 3]
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = "medium"
    plt.rcParams["axes.titlepad"] = 3
    plt.rcParams["xtick.labelsize"] = "x-small"
    plt.rcParams["ytick.labelsize"] = plt.rcParams["xtick.labelsize"]
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["legend.handlelength"] = 1.5
    plt.rcParams["lines.markersize"] = 3
    plt.rcParams["lines.markeredgewidth"] = 0.5
    plt.rcParams["lines.linewidth"] = 0.7
    plt.rcParams["patch.linewidth"] = 0.7
    plt.rcParams["hatch.linewidth"] = 0.7
    plt.rcParams["axes.linewidth"] = 0.6
    plt.rcParams["grid.linewidth"] = 0.6
    plt.rcParams["boxplot.whiskerprops.linewidth"] = 0.7
    plt.rcParams["boxplot.boxprops.linewidth"] = 0.7
    plt.rcParams["boxplot.flierprops.markersize"] = 5
    plt.rcParams["boxplot.flierprops.linewidth"] = 0.2
    plt.rcParams["boxplot.flierprops.marker"] = "o"
    plt.rcParams["boxplot.flierprops.markeredgewidth"] = 0.7
    plt.rcParams["boxplot.capprops.linewidth"] = 0.7
    plt.rcParams["boxplot.medianprops.linewidth"] = 0.7
    plt.rcParams["xtick.major.width"] = 0.6
    plt.rcParams["xtick.minor.width"] = 0.4
    plt.rcParams["ytick.major.width"] = plt.rcParams["xtick.major.width"]
    plt.rcParams["ytick.minor.width"] = plt.rcParams["xtick.minor.width"]


def load_results(results_dir):
    aggregate_fn = os.path.join(results_dir, "aggregate.json")

    if os.path.isfile(aggregate_fn):
        with open(aggregate_fn, "r") as f:
            return json.load(f)

    results_files = glob.glob(os.path.join(results_dir, "run_*.json"))
    results = []
    for results_file in results_files:
        with open(results_file, "r") as f:
            results.append(json.load(f))

    with open(aggregate_fn, "w") as f:
        json.dump(results, f)

    return results


def deep_get(d, keys):
    for key in keys:
        if key not in d:
            return None
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
    elif grouped_results is None:
        return float("nan")
    else:
        results = [
            extract_results(result, keys, depth=depth + 1) for result in grouped_results
        ]
        if depth == 0:
            results = np.asarray(results, dtype=float)
        return results


@dataclass
class ResultsCollection(object):
    axes: list
    gen_risks: np.ndarray
    test_risks: np.ndarray
    cv_k: list
    cv_risks: np.ndarray
    cv_times: np.ndarray
    full_train_times: np.ndarray
    jac_times: np.ndarray
    alo_exact_risks: np.ndarray
    alo_exact_times: np.ndarray
    alo_m: list
    alo_matvec_times: np.ndarray
    alo_bks_risks: np.ndarray
    alo_bks_times: np.ndarray
    alo_poly_risks: np.ndarray
    alo_poly_times: np.ndarray


def extract_all_results(results, axes_keys):
    cv_k = deep_get(results[0], ["config", "cv_k"])
    alo_m = deep_get(results[0], ["config", "alo_m"])

    axes = [
        sorted(set(deep_get(result, keys) for result in results)) for keys in axes_keys
    ]
    axes_spec = list(zip(axes_keys, axes))
    results = group_results(results, axes_spec)

    gen_risks = extract_results(results, ["gen_risk"])
    test_risks = extract_results(results, ["test_risk"])

    if len(cv_k) > 0:
        cv_risks = np.stack(
            [extract_results(results, [f"cv_{k}_risk"]) for k in cv_k],
            axis=-1,
        )
        cv_times = np.stack(
            [extract_results(results, [f"cv_{k}_risk_time"]) for k in cv_k],
            axis=-1,
        )
    else:
        cv_risks = None
        cv_times = None

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

    return ResultsCollection(
        axes=axes,
        gen_risks=gen_risks,
        test_risks=test_risks,
        cv_k=cv_k,
        cv_risks=cv_risks,
        cv_times=cv_times,
        full_train_times=full_train_times,
        jac_times=jac_times,
        alo_exact_risks=alo_exact_risks,
        alo_exact_times=alo_exact_times,
        alo_m=alo_m,
        alo_matvec_times=alo_matvec_times,
        alo_bks_risks=alo_bks_risks,
        alo_bks_times=alo_bks_times,
        alo_poly_risks=alo_poly_risks,
        alo_poly_times=alo_poly_times,
    )


def relative_error(a, b, abs=True):
    if abs:
        return np.abs(a - b) / b
    else:
        return (a - b) / b


def grouped_boxplot(
    data,
    x_labels,
    group_labels,
    ax=None,
    legend=True,
    color_idx=None,
    hatch_idx=None,
    width_scale=1.0,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    n_groups = len(group_labels)
    n_boxes = len(x_labels)

    width = 0.8 / n_groups
    x = np.arange(n_boxes)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if color_idx is not None:
        color_cycle = [color_cycle[i] for i in color_idx]
    hatches = [
        "",
        "////",
        "xxxx",
        "oooo",
        "++++",
        "\\\\\\\\",
        "////",
        "xxxx",
        "oooo",
        "++++",
        "\\\\\\\\",
    ]
    if hatch_idx is not None:
        hatches = [hatches[i] for i in hatch_idx]

    for i, group_label in enumerate(group_labels):
        box_data = data[i]
        ax.boxplot(
            box_data.T,
            positions=x + (i - n_groups / 2 + 0.5) * width,
            widths=width * width_scale,
            patch_artist=True,
            boxprops=dict(facecolor="none", hatch=hatches[i], edgecolor=color_cycle[i]),
            medianprops=dict(color="black"),  # Set median line color to black
            **kwargs,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(-0.5, n_boxes - 0.5)

    artists = []
    for i, group_label in enumerate(group_labels):
        artist = plt.Rectangle(
            (0, 0), 1, 1, facecolor="none", edgecolor=color_cycle[i], hatch=hatches[i]
        )
        artists.append(artist)

    if legend:
        ax.legend(artists, group_labels)

    return artists


def scaling_subplots(results):
    markers = ["o", "s", "D", "v", "^", ">", "<", "p", "h", "H", "d", "P", "X"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ns, lamda0s, seeds = results.axes
    i_n = ns.index(5000)
    i_k = results.cv_k.index(5)
    i_m = results.alo_m.index(100)

    cv_risks_rel = relative_error(results.cv_risks[..., i_k], results.gen_risks)

    to_plot = [
        (results.cv_times[i_n, :, :, i_k], cv_risks_rel[i_n, :, :], "CV"),
        (
            results.alo_exact_times[i_n, :, :],
            relative_error(results.alo_exact_risks[...], results.gen_risks)[i_n, ...],
            "ALO Exact",
        ),
        (
            results.alo_bks_times[i_n, :, :, i_m],
            relative_error(results.alo_bks_risks[..., i_m], results.gen_risks)[
                i_n, ...
            ],
            "ALO BKS",
        ),
    ]

    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    ax = axes
    for i, (times, risks, label) in enumerate(to_plot):
        ax.plot(
            np.median(times, axis=1),
            np.median(risks, axis=1),
            marker=markers[i],
            color=colors[i],
            label=label,
        )
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend()

    plt.xlabel("Time (s)")
    plt.ylabel("Relative Estimation Error")
    plt.show()


def lasso_scaling_normal(results):
    axes_keys = [
        ["config", "data", "n_train"],
        ["config", "seed"],
        ["config", "method_kwargs", "direct"],
    ]

    results = extract_all_results(results, axes_keys)
    ns, seeds, direct = results.axes
    cv_k = results.cv_k
    alo_m = results.alo_m
    k = 5
    m1 = 30
    m2 = 100
    m3 = 50
    i_k = cv_k.index(k)
    i_m1 = alo_m.index(m1)
    i_m2 = alo_m.index(m2)
    i_m3 = alo_m.index(m3)

    direct_vals = [True, False, False]
    i_direct = [direct.index(tf) for tf in direct_vals]

    results.gen_risks = np.stack(
        [results.gen_risks[i, :, i_direct[i]] for i in range(len(ns))], axis=0
    )
    results.cv_risks = np.stack(
        [results.cv_risks[i, :, i_direct[i], :] for i in range(len(ns))], axis=0
    )
    results.alo_bks_risks = np.stack(
        [results.alo_bks_risks[i, :, i_direct[i], :] for i in range(len(ns))], axis=0
    )
    results.alo_poly_risks = np.stack(
        [results.alo_poly_risks[i, :, i_direct[i], :] for i in range(len(ns))], axis=0
    )

    results.full_train_times = np.stack(
        [results.full_train_times[i, :, i_direct[i]] for i in range(len(ns))], axis=0
    )
    results.cv_times = np.stack(
        [results.cv_times[i, :, i_direct[i], :] for i in range(len(ns))], axis=0
    )
    results.alo_bks_times = np.stack(
        [results.alo_bks_times[i, :, i_direct[i], :] for i in range(len(ns))], axis=0
    )
    results.alo_poly_times = np.stack(
        [results.alo_poly_times[i, :, i_direct[i], :] for i in range(len(ns))], axis=0
    )

    # First, plot only BKS

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), dpi=300)
    grouped_boxplot(
        [
            results.gen_risks,
            results.cv_risks[..., i_k],
            results.alo_bks_risks[..., i_m1],
            results.alo_bks_risks[..., i_m2],
        ],
        [f"n={n}" for n in ns],
        [
            "Conditional risk",
            f"CV($K={k}$)",
            f"BKS-ALO($m={m1}$)",
            f"BKS-ALO($m={m2}$)",
        ],
        ax=axes[0],
    )

    # axes[0].set_title("Risk vs. sample size")
    axes[0].set_ylabel("Squared error")
    axes[0].set_xlabel("Sample size")

    # axes[1].axhline(1, color="black", linestyle="--", label="Training")
    grouped_boxplot(
        [
            # results.full_train_times / results.full_train_times,
            results.cv_times[..., i_k] / results.full_train_times,
            results.alo_bks_times[..., i_m1] / results.full_train_times,
            results.alo_bks_times[..., i_m2] / results.full_train_times,
        ],
        [f"n={n}" for n in ns],
        [
            # "Training",
            f"CV($K={k}$)",
            f"BKS-ALO($m={m1}$)",
            f"BKS-ALO($m={m2}$)",
        ],
        ax=axes[1],
        color_idx=[1, 2, 3],
        hatch_idx=[1, 2, 3],
        legend=False,
    )

    # axes[1].set_title("Time vs. sample size")
    axes[1].set_ylabel("Time relative to model training")
    axes[1].set_xlabel("Sample size")
    # axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(
        os.path.join("figures", "lasso_scaling_normal_bks.pdf"), bbox_inches="tight"
    )

    # Next, plot both BKS and Poly

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), dpi=300)
    grouped_boxplot(
        [
            # results.gen_risks,
            relative_error(results.cv_risks[..., i_k], results.gen_risks, abs=False),
            relative_error(
                results.alo_bks_risks[..., i_m3], results.gen_risks, abs=False
            ),
            relative_error(
                results.alo_poly_risks[..., i_m3], results.gen_risks, abs=False
            ),
        ],
        [f"n={n}" for n in ns],
        [
            # "Conditional risk",
            f"CV($K={k}$)",
            f"BKS-ALO($m={m3}$)",
            f"RandALO($m={m3}$)",
        ],
        color_idx=[1, 2, 3],
        hatch_idx=[1, 2, 3],
        ax=axes[0],
    )

    # axes[0].set_title("Risk vs. sample size")
    axes[0].set_ylabel("Relative risk estimation error")
    axes[0].set_xlabel("Sample size")

    # axes[1].axhline(1, color="black", linestyle="--", label="Training")
    grouped_boxplot(
        [
            # results.full_train_times / results.full_train_times,
            results.cv_times[..., i_k] / results.full_train_times,
            results.alo_bks_times[..., i_m3] / results.full_train_times,
            results.alo_poly_times[..., i_m3] / results.full_train_times,
        ],
        [f"n={n}" for n in ns],
        [
            # "Training",
            f"CV($K={k}$)",
            f"BKS-ALO($m={m3}$)",
            f"RandALO($m={m3}$)",
        ],
        ax=axes[1],
        color_idx=[1, 2, 3],
        hatch_idx=[1, 2, 3],
        legend=False,
    )

    # axes[1].set_title("Time vs. sample size")
    axes[1].set_ylabel("Time relative to model training")
    axes[1].set_xlabel("Sample size")
    # axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(
        os.path.join("figures", "lasso_scaling_normal_both.pdf"), bbox_inches="tight"
    )


def lasso_bks_convergence(results):
    axes_keys = [
        ["config", "seed"],
    ]

    results = extract_all_results(results, axes_keys)
    (seeds,) = results.axes

    with open(os.path.join("lasso_bks_convergence", "one_example.json"), "r") as f:
        data_one_example = json.load(f)

    # remove half from squared error
    data_one_example["exact"] *= 2
    data_one_example["bks"]["risk"] = np.array(data_one_example["bks"]["risk"]) * 2
    for m0 in [30, 100]:
        data_one_example["alo"][str(m0)]["w"] = (
            np.array(data_one_example["alo"][str(m0)]["w"]) * 2
        )
        data_one_example["alo"][str(m0)]["risk"] = (
            np.array(data_one_example["alo"][str(m0)]["risk"]) * 2
        )

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), dpi=300)

    exact = data_one_example["exact"]
    axes[0].axhline(exact, color="black", linestyle=":", label="Exact ALO")
    xlim = np.asarray([0, 0.08])
    ylim = np.asarray([0.65, 0.76]) * 2
    aspect_ratio = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])

    # for angle in np.linspace(0, np.pi / 2, 9)[1:-1]:
    #     axes[0].plot(
    #         xlim,
    #         exact + np.tan(angle) * (xlim - xlim[0]) / aspect_ratio,
    #         color="black",
    #         linestyle=":",
    #         alpha=0.2,
    #     )

    bks = data_one_example["bks"]
    axes[0].plot(
        1 / np.asarray(bks["m"]),
        bks["risk"],
        linestyle="--",
        label="BKS-ALO",
    )

    marker_cycle = ["^", "*"]
    sizes = [22, 30]
    for i, m0 in enumerate([30, 100]):
        alo = data_one_example["alo"][str(m0)]
        y0, y1 = np.vander(np.array([0, 0.1]), 2, True) @ alo["w"]
        axes[0].plot(
            [0, 0.1],
            [y0, y1],
            color=color_cycle[i + 1],
        )
        axes[0].scatter(
            alo["inv_m"],
            alo["risk"],
            color=color_cycle[i + 1],
        )
        axes[0].scatter(
            [0],
            [y0],
            sizes[i],
            color=color_cycle[i + 1],
            marker=marker_cycle[i],
            zorder=10,
            clip_on=False,
            label=f"Subsampled ALO (m={m0})",
        )

    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    axes[0].set_title("Linear fit estimation of $R_0$")
    axes[0].set_xlabel("Reciprocal number of Jacobian--vector products $1/m$")
    axes[0].set_ylabel("Risk estimate")
    axes[0].legend(loc="upper left")

    ########################################

    ms = np.asarray(results.alo_m)
    results.alo_exact_risks *= 2
    results.alo_bks_risks *= 2
    results.alo_poly_risks *= 2
    # ms_recip = ms
    # ms_recip = np.concatenate([1 / ms[:-1], [0.0]])

    axes[1].axhline(
        np.mean(results.alo_exact_risks),
        color="black",
        linestyle=":",
        label="Exact ALO",
    )
    # fill between interquartile range of test risk
    # xlim = np.asarray([0, np.max(ms_recip)])
    # plt.xscale("log")
    xlim = np.asarray([np.min(ms), np.max(ms)])
    xlim = np.asarray([np.min(ms), 300])
    axes[1].fill_between(
        xlim,
        np.mean(results.alo_exact_risks) - np.std(results.alo_exact_risks),
        np.mean(results.alo_exact_risks) + np.std(results.alo_exact_risks),
        facecolor="black",
        alpha=0.2,
    )

    axes[1].plot(
        ms,
        np.mean(results.alo_bks_risks, axis=0).T,
        "--",
        color=color_cycle[0],
        label="BKS-ALO",
    )
    axes[1].plot(
        ms,
        np.mean(results.alo_poly_risks, axis=0).T,
        color=color_cycle[1],
        label="RandALO",
    )

    axes[1].fill_between(
        ms,
        np.mean(results.alo_bks_risks, axis=0) - np.std(results.alo_bks_risks, axis=0),
        np.mean(results.alo_bks_risks, axis=0) + np.std(results.alo_bks_risks, axis=0),
        color=color_cycle[0],
        alpha=0.2,
    )
    axes[1].fill_between(
        ms,
        np.mean(results.alo_poly_risks, axis=0)
        - np.std(results.alo_poly_risks, axis=0),
        np.mean(results.alo_poly_risks, axis=0)
        + np.std(results.alo_poly_risks, axis=0),
        color=color_cycle[1],
        alpha=0.2,
    )

    ylim = axes[1].get_ylim()

    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)

    axes[1].set_title("Convergence to ALO")
    axes[1].set_xlabel("Number of Jacobian--vector products $m$")
    axes[1].set_ylabel("Risk estimate")
    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join("figures", "lasso_bks_convergence.pdf"), bbox_inches="tight"
    )


def lasso_sweep(results):
    axes_keys = [
        ["config", "method_kwargs", "lamda0"],
        ["config", "seed"],
    ]

    results = extract_all_results(results, axes_keys)
    lamda0s, seeds = results.axes
    k = 5
    i_k = results.cv_k.index(k)
    m = 50
    i_m = results.alo_m.index(m)

    # remove scaling by 1/2
    results.cv_risks *= 2
    results.gen_risks *= 2
    results.test_risks *= 2
    results.alo_bks_risks *= 2
    results.alo_poly_risks *= 2
    results.alo_exact_risks *= 2

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["D", "o", "s", "v", "^", ">", "<", "p", "h", "H", "d", "P", "X"]
    linestyle = [":", "-", "--", "-."]

    data = [
        results.gen_risks,
        results.cv_risks[..., i_k],
        results.alo_bks_risks[..., i_m],
        results.alo_poly_risks[..., i_m],
    ]
    labels = [
        "Conditional risk",
        f"CV($K={k}$)",
        f"BKS-ALO($m={m}$)",
        f"RandALO($m={m}$)",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), dpi=300)

    for i, (label, color) in enumerate(zip(labels, color_cycle)):
        # shift markevery by 1 to avoid overlapping markers
        markevery = (i, len(data))
        axes[0].errorbar(
            lamda0s,
            np.mean(data[i], axis=1),
            yerr=np.std(data[i], axis=1),
            label=label,
            color=color,
            linestyle=linestyle[i],
            marker=markers[i],
            markevery=markevery,
            errorevery=markevery,
            capsize=2,
        )

    axes[0].set_xscale("log")
    # axes[0].set_title("Risk")
    axes[0].set_ylabel("Squared error")
    axes[0].set_xlabel("Regularization parameter $\\lambda_0$")
    axes[0].legend()

    data = [
        # results.full_train_times,
        results.cv_times[..., i_k] / results.full_train_times,
        results.alo_bks_times[..., i_m] / results.full_train_times,
        results.alo_poly_times[..., i_m] / results.full_train_times,
    ]
    labels = [
        # "Training",
        f"CV(K={k})",
        f"BKS-ALO(m={m})",
        f"RandALO(m={m})",
    ]

    for i, (label, color) in enumerate(zip(labels, color_cycle[1:])):
        # shift markevery by 1 to avoid overlapping markers
        markevery = (i, len(data))
        axes[1].errorbar(
            lamda0s,
            np.mean(data[i], axis=1),
            yerr=np.std(data[i], axis=1),
            label=label,
            color=color,
            linestyle=linestyle[i + 1],
            marker=markers[i + 1],
            markevery=markevery,
            errorevery=markevery,
            capsize=2,
        )

    axes[1].set_xscale("log")
    # axes[1].set_yscale("log")
    # axes[1].set_title("Time")
    axes[1].set_ylabel("Time relative to model training")
    axes[1].set_xlabel("Regularization parameter $\\lambda_0$")
    # axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join("figures", "lasso_sweep.pdf"), bbox_inches="tight")


def confusion_matrix(results):
    axes_keys = [
        ["config", "seed"],
        ["config", "method_kwargs", "lamda0"],
    ]

    results = extract_all_results(results, axes_keys)
    seeds, lamda0s = results.axes
    cv_k = results.cv_k
    alo_m = results.alo_m
    good_lamda0 = lamda0s.index(10)

    print("Correct selections")
    print("=====================================")

    amin = np.argmin(results.gen_risks, axis=1)
    print(f"Generalization error: {np.sum(amin == good_lamda0)}")

    for i, k in enumerate(cv_k):
        amin = np.argmin(results.cv_risks[..., i], axis=1)
        print(f"CV({k}) error: {np.sum(amin == good_lamda0)}")

    for i, m in enumerate(alo_m):
        amin = np.argmin(results.alo_bks_risks[..., i], axis=1)
        print(f"BKS-ALO({m}) error: {np.sum(amin == good_lamda0)}")

    for i, m in enumerate(alo_m):
        amin = np.argmin(results.alo_poly_risks[..., i], axis=1)
        print(f"RandALO({m}) error: {np.sum(amin == good_lamda0)}")


def lasso_cv_tradeoff(results):
    axes_keys = [
        ["config", "seed"],
    ]

    results = extract_all_results(results, axes_keys)
    (seeds,) = results.axes
    ks = [2, 3, 5, 10, 20]
    i_k = [results.cv_k.index(k) for k in ks]
    ms = [10, 30, 100, 300, 1000, 3000]
    i_m = [results.alo_m.index(m) for m in ms]

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    marker_cycle = ["o", "s", "v", "D", "^", "*", "<", "p", "h", "H", "d", "P", "X"]

    fig = plt.figure(figsize=(6.5, 3), dpi=300)

    gen_risk = np.mean(results.gen_risks)

    def plot_tradeoff(times, risks, marker, color, label, linestyle):
        risks = risks / gen_risk - 1
        risks_mean = np.mean(risks, axis=0)
        risks_sign = np.sign(risks_mean)
        risks_abs = np.abs(risks_mean)
        plt.plot(
            np.mean(times, axis=0),
            risks_abs,
            marker=marker,
            color=color,
            linestyle=linestyle,
            label=label,
        )
        return
        # quartiles
        # x_lower = np.quantile(times, 0.25, axis=0)
        # x_upper = np.quantile(times, 0.75, axis=0)
        # y_lower0 = np.quantile(risks, 0.25, axis=0)
        # y_upper0 = np.quantile(risks, 0.75, axis=0)

        # standard deviations
        x_lower = np.mean(times, axis=0) - np.std(times, axis=0)
        x_upper = np.mean(times, axis=0) + np.std(times, axis=0)
        y_lower0 = np.mean(risks, axis=0) - np.std(risks, axis=0)
        y_upper0 = np.mean(risks, axis=0) + np.std(risks, axis=0)

        # standard errors
        # x_lower = np.mean(times, axis=0) - np.std(times, axis=0) / np.sqrt(len(seeds))
        # x_upper = np.mean(times, axis=0) + np.std(times, axis=0) / np.sqrt(len(seeds))
        # y_lower0 = np.mean(risks, axis=0) - np.std(risks, axis=0) / np.sqrt(len(seeds))
        # y_upper0 = np.mean(risks, axis=0) + np.std(risks, axis=0) / np.sqrt(len(seeds))

        y_lower = y_lower0.copy()
        y_lower[risks_sign == -1] = -y_upper0[risks_sign == -1]
        y_upper = y_upper0.copy()
        y_upper[risks_sign == -1] = -y_lower0[risks_sign == -1]
        plt.fill(
            np.concatenate([x_upper, x_lower[::-1]]),
            np.concatenate([y_upper, y_lower[::-1]]),
            color,
            alpha=0.2,
        )
        for i in range(times.shape[1]):
            plt.scatter(
                np.mean(times[:, i]),
                risks_abs[i],
                s=15 if i == 2 else None,
                marker=marker_cycle[i],
                color=color,
                zorder=10,
            )

    plot_tradeoff(
        results.cv_times[..., i_k] / results.full_train_times[:, None],
        results.cv_risks[..., i_k],
        marker_cycle[0],
        color_cycle[0],
        "$K$-fold CV",
        "-",
    )
    plot_tradeoff(
        results.alo_bks_times[..., i_m] / results.full_train_times[:, None],
        results.alo_bks_risks[..., i_m],
        marker_cycle[1],
        color_cycle[1],
        "BKS-ALO",
        "--",
    )
    plot_tradeoff(
        results.alo_poly_times[..., i_m] / results.full_train_times[:, None],
        results.alo_poly_risks[..., i_m],
        marker_cycle[2],
        color_cycle[2],
        "RandALO",
        "-.",
    )

    plt.scatter(
        [np.mean(results.alo_exact_times / results.full_train_times)],
        [np.abs(np.mean(results.alo_exact_risks) / gen_risk - 1)],
        marker="D",
        color="red",
        label="Exact ALO",
    )

    stderr = np.std(results.gen_risks) / gen_risk / np.sqrt(len(seeds))
    plt.axhline(
        stderr,
        color="black",
        linestyle=":",
        label="True risk standard error",
    )
    plt.xscale("log")
    plt.yscale("symlog", linthresh=stderr, linscale=0.15)
    plt.yticks(
        [0, stderr, 0.003, 0.01, 0.03, 0.1, 0.3],
        [
            "$0$",
            f"${stderr * 100:.3f}\\%$",
            "$0.3\\%$",
            "$1\\%$",
            "$3\\%$",
            "$10\\%$",
            "$30\\%$",
        ],
    )
    plt.minorticks_off()
    plt.xticks(
        [2, 5, 10, 20], ["$2\\times$", "$5\\times$", "$10\\times$", "$20\\times$"]
    )
    plt.ylim(0, 0.5)

    plt.xlabel("Time relative to full model training")
    plt.ylabel("Relative risk estimation bias")
    plt.title("Efficiency of randomized ALO")

    plt.tight_layout()

    # plt.legend(loc=(1.02, 0.35), frameon=False)
    plt.legend(loc="upper right")

    # # add custom legend
    # ax = fig.add_axes([0.98, 0.17, 0.2, 0.5])
    # ax.axis("off")
    # ax.set_aspect("equal")

    # # Table data preparation
    # table_data = []
    # for marker, k, m in zip(marker_cycle, ks + [""], ms):
    #     table_data.append([marker, str(k), str(m)])

    # # Define the cell text and cell colors
    # cell_text = []
    # for row in table_data:
    #     cell_text.append([f"$\\{row[0]}$", row[1], row[2]])

    # # Create the table
    # table = plt.table(
    #     cellText=cell_text,
    #     colLabels=["", "$K$(CV)", "$m$(ALO)"],
    #     cellLoc="center",
    #     loc="center",
    #     edges="open",
    # )

    # # Adjust table properties
    # table.auto_set_font_size(False)
    # table.set_fontsize(8)
    # # table.scale(1, 2)  # Scale the table

    # # Replace the first column text with markers
    # ax.scatter([0], [0.01], alpha=0)
    # ax.scatter([1], [0.92], alpha=0)
    # for i, marker in enumerate(marker_cycle[:6], start=1):
    #     table[(i, 0)]._text.set_text("XX")
    #     table[(i, 0)]._text.set_color("white")
    #     ax.scatter(
    #         0.1,
    #         0.8 - table[(i, 0)].get_height() * i,
    #         marker=marker,
    #         color="black",
    #         linestyle="",
    #     )

    # table.auto_set_column_width(col=list(range(len(table_data[0]))))

    plt.savefig(os.path.join("figures", "lasso_cv_tradeoff.pdf"), bbox_inches="tight")


def comp_all():
    comps = [
        "logistic_comp",
        "multivariate_t_comp",
        "categorical_comp",
    ]
    titles = [
        "Logistic, Gaussian",
        "Lasso, multivariate $t$",
        "Lasso, categorical",
    ]

    # fig, axes = plt.subplots(1, 2 * len(comps), figsize=(6.5, 3), dpi=300)
    fig = plt.figure(figsize=(5.5, 3.5), dpi=300)
    outer = gridspec.GridSpec(1, len(comps), wspace=0.1)

    axes_keys = [
        ["config", "seed"],
    ]
    k = 5
    m = 100

    for i, comp in enumerate(comps):
        results = load_results(os.path.join(comp, "results"))
        results = extract_all_results(results, axes_keys)
        seeds = results.axes[0]
        i_k = results.cv_k.index(k)
        i_m = results.alo_m.index(m)

        inner = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=outer[i],
            hspace=0.08,  # width_ratios=[5, 4]
        )
        ax1, ax2 = plt.subplot(inner[0]), plt.subplot(inner[1])

        artists = grouped_boxplot(
            [
                # results.gen_risks,
                relative_error(
                    results.cv_risks[..., i_k], results.gen_risks, abs=False
                ),
                relative_error(
                    results.alo_bks_risks[..., i_m], results.gen_risks, abs=False
                ),
                relative_error(
                    results.alo_poly_risks[..., i_m], results.gen_risks, abs=False
                ),
            ],
            [""],
            [
                # "Test error",
                f"CV($K={k}$)",
                f"BKS-ALO($m={m}$)",
                f"RandALO($m={m}$)",
            ],
            ax=ax1,
            legend=(i == 2),
            color_idx=[1, 2, 3],
            hatch_idx=[1, 2, 3],
            width_scale=0.8,
        )
        ax1.set_title(titles[i])
        if i == 0:
            ax1.set_ylabel("Relative risk error")
        ax1.set_xticks([])
        # ylim = ax1.get_ylim()
        ylim = (-0.12, 0.27)
        # yticks = ax1.get_yticks()
        yticks = [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        ax1.set_yticks(yticks)
        ax1.set_yticklabels(f"${y:.2f}$" for y in yticks)
        ax1.set_ylim(ylim)

        if i == 2:
            ax1.legend(
                artists,
                [
                    # "Conditional risk",
                    f"CV($K={k}$)",
                    f"BKS-ALO($m={m}$)",
                    f"RandALO($m={m}$)",
                ],
                loc=(1.05, -0.25),
                frameon=False,
            )

        grouped_boxplot(
            [
                results.cv_times[..., i_k] / results.full_train_times,
                results.alo_bks_times[..., i_m] / results.full_train_times,
                results.alo_poly_times[..., i_m] / results.full_train_times,
            ],
            [""],
            [
                f"CV($K={k}$)",
                f"BKS-ALO($m={m}$)",
                f"RandALO($m={m}$)",
            ],
            ax=ax2,
            legend=False,
            color_idx=[1, 2, 3],
            hatch_idx=[1, 2, 3],
            width_scale=0.8,
        )
        if i == 0:
            ax2.set_ylabel("Relative time")
        ax2.set_xticks([])
        # ylim = ax2.get_ylim()
        ylim = (1.0, 6.0)
        # yticks = ax2.get_yticks()
        yticks = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(f"{y:.1f}" for y in yticks)
        ax2.set_ylim(ylim)

        if i > 0:
            ax1.set_yticklabels([])
            ax2.set_yticklabels([])

        # ax2.set_yticks(axes[0].get_yticks())
        # ax2.yaxis.tick_right()

        # ax3 = plt.subplot(outer[len(comps) + i])
        # ax3.set_title(titles[i])
        # ax3.axis("off")

    # plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join("figures", "comp_all.pdf"), bbox_inches="tight")


def categorical_comp(results):
    general_comp(results, "categorical_comp", k=5, m=100)


def logistic_comp(results):
    general_comp(results, "logistic_comp", k=5, m=100)


def multivariate_t_comp(results):
    general_comp(results, "multivariate_t_comp", k=5, m=100)


def random_forest_comp(results):
    general_comp(results, "random_forest_comp", k=5, m=100)


def general_comp(results, out_name, k=5, m=100):
    axes_keys = [
        ["config", "seed"],
    ]

    results = extract_all_results(results, axes_keys)
    seeds = results.axes[0]
    i_k = results.cv_k.index(k)
    i_m = results.alo_m.index(m)

    fig, axes = plt.subplots(1, 2, figsize=(3.25, 3), dpi=300)

    grouped_boxplot(
        [
            results.test_risks,
            results.cv_risks[..., i_k],
            results.alo_bks_risks[..., i_m],
            results.alo_poly_risks[..., i_m],
        ],
        [""],
        ["Test error", f"CV(K={k})", f"BKS-ALO(m={m})", f"RandALO(m={m})"],
        ax=axes[0],
    )
    axes[0].set_title("Risk")
    axes[0].set_xticks([])

    grouped_boxplot(
        [
            # results.full_train_times,
            results.cv_times[..., i_k] / results.full_train_times,
            results.alo_bks_times[..., i_m] / results.full_train_times,
            results.alo_poly_times[..., i_m] / results.full_train_times,
        ],
        [""],
        [
            # "Training",
            f"CV(K={k})",
            f"BKS-ALO(m={m})",
            f"RandALO(m={m})",
        ],
        ax=axes[1],
        color_idx=[1, 2, 3],
        hatch_idx=[1, 2, 3],
    )
    axes[1].set_title("Time relative to training")
    axes[1].set_xticks([])

    plt.tight_layout()
    plt.savefig(os.path.join("figures", f"{out_name}.pdf"), bbox_inches="tight")


def first_diff_scaling_1(results):
    axes_keys = [
        ["config", "data", "n_train"],
        ["config", "seed"],
        ["config", "method_kwargs", "lamda0"],
    ]

    results = extract_all_results(results, axes_keys)
    ns, seeds, lamda0s = results.axes
    k = 5
    lamda0 = 0.01
    m = 100
    i_k = results.cv_k.index(k)
    i_m = results.alo_m.index(m)
    i_lamda0 = lamda0s.index(lamda0)

    ns_filter = [500, 1000, 2000, 5000]
    i_ns = [ns.index(n) for n in ns_filter]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), dpi=300)
    grouped_boxplot(
        [
            # results.gen_risks[i_ns, ..., i_lamda0],
            relative_error(
                results.cv_risks[i_ns, ..., i_lamda0, i_k],
                results.gen_risks[i_ns, ..., i_lamda0],
                abs=False,
            ),
            relative_error(
                results.alo_bks_risks[i_ns, ..., i_lamda0, i_m],
                results.gen_risks[i_ns, ..., i_lamda0],
                abs=False,
            ),
            relative_error(
                results.alo_poly_risks[i_ns, ..., i_lamda0, i_m],
                results.gen_risks[i_ns, ..., i_lamda0],
                abs=False,
            ),
        ],
        [f"n={n}" for n in ns_filter],
        [
            # "Conditional risk",
            f"CV($K={k}$)",
            f"BKS-ALO($m={m}$)",
            f"RandALO($m={m}$)",
        ],
        ax=axes[0],
        color_idx=[1, 2, 3],
        hatch_idx=[1, 2, 3],
    )

    # axes[0].set_title("Risk vs. sample size")
    axes[0].set_ylabel("Relative risk estimation error")
    axes[0].set_xlabel("Sample size")

    # axes[1].axhline(1, color="black", linestyle="--", label="Training")
    grouped_boxplot(
        [
            # results.full_train_times[i_ns, ..., i_lamda0],
            results.cv_times[i_ns, ..., i_lamda0, i_k]
            / results.full_train_times[i_ns, ..., i_lamda0],
            results.alo_bks_times[i_ns, ..., i_lamda0, i_m]
            / results.full_train_times[i_ns, ..., i_lamda0],
            results.alo_poly_times[i_ns, ..., i_lamda0, i_m]
            / results.full_train_times[i_ns, ..., i_lamda0],
        ],
        [f"n={n}" for n in ns_filter],
        [
            # "Training",
            f"CV($K={k}$)",
            f"BKS-ALO($m={m}$)",
            f"RandALO($m={m}$)",
        ],
        ax=axes[1],
        color_idx=[1, 2, 3],
        hatch_idx=[1, 2, 3],
        legend=False,
    )

    # axes[1].set_title("Time vs. sample size")
    axes[1].set_ylabel("Time relative to model training")
    axes[1].set_xlabel("Sample size")
    axes[1].set_yscale(
        "function",
        functions=(
            lambda x: np.log(x - 1),
            lambda x: np.exp(x) + 1,
        ),
    )
    axes[1].set_yticks([1.01, 1.1, 2, 3, 4])
    axes[1].set_ylim(1.001, 5)

    plt.tight_layout()
    plt.savefig(
        os.path.join("figures", "first_diff_scaling_1.pdf"), bbox_inches="tight"
    )


def first_diff_scaling():
    # TODO: refactor!!!

    results = load_results(os.path.join("first_diff_scaling_1", "results"))

    axes_keys = [
        ["config", "data", "n_train"],
        ["config", "method_kwargs", "lamda0"],
        ["config", "seed"],
    ]

    (
        axes,
        gen_risks,
        test_risks,
        cv_k,
        cv_risks,
        cv_times,
        full_train_times,
        alo_exact_risks,
        alo_exact_times,
        alo_m,
        alo_bks_risks,
        alo_bks_times,
        alo_poly_risks,
        alo_poly_times,
    ) = extract_all_results(results, axes_keys)
    ns, lamda0s, seeds = axes

    k = 5
    m = 300
    lamda0 = 10.0
    ik = cv_k.index(k)
    im = alo_m.index(m)
    ilamda0 = lamda0s.index(lamda0)

    test_rel = relative_error(test_risks, gen_risks)[:, ilamda0, :]
    cv_rel = relative_error(cv_risks[..., ik], gen_risks)[:, ilamda0, :]
    alo_exact_rel = relative_error(alo_exact_risks, gen_risks)[:, ilamda0, :]
    alo_bks_rel = relative_error(alo_bks_risks[..., im], gen_risks)[:, ilamda0, :]
    alo_poly_rel = relative_error(alo_poly_risks[..., im], gen_risks)[:, ilamda0, :]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    normalize_factor = 1 / np.median(cv_rel, axis=1)[:, None]
    grouped_boxplot(
        [
            cv_rel * normalize_factor,
            alo_exact_rel * normalize_factor,
            alo_bks_rel * normalize_factor,
            alo_poly_rel * normalize_factor,
        ],
        [f"n={n}" for n in ns],
        [f"cv_{k}", "alo_exact", f"alo_{m}_bks", f"alo_{m}_poly"],
        ax=axes[0],
    )
    axes[0].axhline(1, color="black", linestyle="--")
    axes[0].set_title(f"First Difference Error Scaling for $\\lambda_0={lamda0}$")
    axes[0].set_ylabel("Relative Estimation Error (normalized by CV median)")

    # normalize_factor = 1 / np.median(cv_times[:, ilamda0, :, ik], axis=1)[:, None]
    normalize_factor = 1 / np.median(full_train_times[:, ilamda0, :], axis=1)[:, None]
    i = np.argmax(alo_bks_times[-1, ilamda0, :, im])
    print(alo_bks_times[-1, ilamda0, i, im] * normalize_factor[-1, 0])
    print(full_train_times[-1, ilamda0, i] * normalize_factor[-1, 0])

    grouped_boxplot(
        [
            cv_times[:, ilamda0, :, ik] * normalize_factor,
            alo_exact_times[:, ilamda0, :] * normalize_factor,
            alo_bks_times[:, ilamda0, :, im] * normalize_factor,
            alo_poly_times[:, ilamda0, :, im] * normalize_factor,
        ],
        [f"n={n}" for n in ns],
        [f"cv_{k}", "alo_exact", f"alo_{m}_bks", f"alo_{m}_poly"],
        ax=axes[1],
    )
    axes[1].axhline(1, color="black", linestyle="--")
    axes[1].axhline((k - 1) ** 2 / k, color="black", linestyle=":")
    # solve_time = np.median(full_train_times[-1, ilamda0, :] * normalize_factor[-1, :])
    # axes[1].axhline(solve_time, color="black", linestyle=":")
    axes[1].set_title(f"First Difference Time Scaling for $\\lambda_0={lamda0}$")
    axes[1].set_ylabel("Time (normalized by median model training)")
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.show()


def truncated_normal_viz():
    n, p, m = 100, 100, 50

    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, p)) * rng.uniform(0.5, 1, size=(n, 1))
    W = rng.integers(0, 2, size=(p, m)) * 2 - 1
    lamda = 0.1

    J = X @ np.linalg.solve(X.T @ X + lamda * np.eye(p), X.T)
    Jdiag = np.diag(J)
    D = (J @ W) * W
    mu = np.mean(D, axis=1)
    sigma = np.std(D, axis=1)
    a = (0 - mu) / sigma * np.sqrt(m)
    b = (1 - mu) / sigma * np.sqrt(m)
    d = stats.truncnorm.mean(a, b, loc=mu, scale=sigma / np.sqrt(m))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))

    sigma0 = np.mean(sigma) / np.sqrt(m) * 1.5
    Jtr = 0.97
    Jhat = 1.007
    a0 = (0 - Jhat) / sigma0
    b0 = (1 - Jhat) / sigma0

    Js = np.linspace(Jhat - sigma0 * 2.5, Jhat + sigma0 * 2.5, 500)
    f0 = stats.norm.pdf(Js, loc=Jhat, scale=sigma0)
    f1 = stats.truncnorm.pdf(Js, a0, b0, loc=Jhat, scale=sigma0)
    axes[0].axvline(Jtr, color="black", linestyle="--", label="True $\\tilde{J}_{ii}$")
    axes[0].plot(Js, f0, label="Gaussian likelihood")
    axes[0].axvline(
        Jhat, color=color_cycle[0], linestyle=":", label="Gaussian MLE $\\mu_i$"
    )
    axes[0].plot(Js, f1, label="Truncated normal posterior")
    axes[0].axvline(
        stats.truncnorm.mean(a0, b0, loc=Jhat, scale=sigma0),
        color=color_cycle[1],
        linestyle=":",
        label="Truncated normal MMSE",
    )
    ylim = axes[0].get_ylim()

    axes[0].set_title("Diagonal estimation")
    axes[0].set_ylim(ylim[0], 55)
    axes[0].legend(loc="upper left")
    axes[0].set_xlabel("$\\tilde{J}_{ii}$")

    J0 = np.linspace(np.min(Jdiag), 0.999, 100)
    axes[1].plot(
        J0, J0 / (1 - J0), "--k", label="$\\tilde{J}_{ii} / (1 - \\tilde{J}_{ii})$"
    )

    eps = 0
    bad_mask = mu > 1 - eps
    eps = 1e-3
    mu[bad_mask] = np.nan
    axes[1].scatter(Jdiag, mu / (1 - mu), label="Gaussian MLE $\\mu_i$")
    axes[1].scatter(
        Jdiag[bad_mask],
        [(1 - eps) / eps] * np.sum(bad_mask),
        marker="D",
        color="red",
        label="Gaussian MLE $\\mu_i$ (bad)",
    )
    axes[1].scatter(Jdiag, d / (1 - d), marker="x", label="Truncated normal MMSE")

    axes[1].set_title("Inversion sensitivity")
    axes[1].set_xlabel("$\\tilde{J}_{ii}$")
    axes[1].set_yscale("log")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join("figures", "truncated_normal_viz.pdf"), bbox_inches="tight"
    )


def array_argmin(a, axis=None):
    return np.unravel_index(np.argmin(a, axis=axis), a.shape)


def fashion_mnist(results):
    axes_keys = [
        ["config", "seed"],
        ["config", "method_kwargs", "lamda"],
        ["config", "method_kwargs", "kernel_fun_kwargs", "gamma"],
    ]

    results = extract_all_results(results, axes_keys)
    seeds, lamdas, gammas = results.axes
    ks = results.cv_k
    ms = results.alo_m

    seed = 0
    ik = 1
    im = 0

    vmin = np.min(results.test_risks[seed, ...])
    vmax = max(
        [
            np.max(x)
            for x in [
                results.test_risks[seed, ...],
                results.cv_risks[seed, ..., ik],
                results.alo_poly_risks[seed, ..., im],
            ]
        ]
    )
    if cmr is not None:
        cmap = cmr.get_sub_cmap(cmr.rainforest, 0.15, 1.0)
    else:
        cmap = "viridis"

    train_seconds = np.sum(results.full_train_times[seed, ...])
    print(f"Training time: {train_seconds / 60:.1f} min")
    cv_seconds = np.sum(results.cv_times[seed, ..., ik])
    alo_poly_seconds = np.sum(results.alo_poly_times[seed, ..., im])

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5), dpi=150)
    axes[0].imshow(
        results.test_risks[seed, ...], aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap
    )
    axes[0].set_xticks(range(0, len(gammas), 2))
    axes[0].set_xticklabels([f"$10^{{{np.log10(g):.1g}}}$" for g in gammas[0::2]])
    axes[0].set_ylabel("Regularization parameter $\\lambda$")
    axes[0].set_yticks(range(0, len(lamdas), 2))
    axes[0].set_yticklabels([f"$10^{{{np.log10(l):.1g}}}$" for l in lamdas[0::2]])
    axes[0].set_title("Test error")

    idx = array_argmin(results.test_risks[seed, ...])
    axes[0].scatter(idx[1], idx[0], color="red", marker="*")
    print(f"Test risk: {np.min(results.test_risks[seed, ...])}")

    axes[1].imshow(
        results.cv_risks[seed, ..., ik], aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap
    )
    axes[1].set_xlabel("Kernel parameter $\\gamma$")
    axes[1].set_xticks(range(0, len(gammas), 2))
    axes[1].set_xticklabels([f"$10^{{{np.log10(g):.1g}}}$" for g in gammas[0::2]])
    axes[1].set_yticks([])
    axes[1].set_title(f"CV($K={ks[ik]}$), {cv_seconds / 60:.1f} min")

    idx = array_argmin(results.cv_risks[seed, ..., ik])
    axes[1].scatter(idx[1], idx[0], color="red", marker="*")
    print(f"CV risk: {np.min(results.cv_risks[seed, ..., ik])}")

    axes[2].imshow(
        results.alo_poly_risks[seed, ..., im],
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    axes[2].set_xticks(range(0, len(gammas), 2))
    axes[2].set_xticklabels([f"$10^{{{np.log10(g):.1g}}}$" for g in gammas[0::2]])
    axes[2].set_yticks([])
    axes[2].set_title(f"RandALO($m={ms[im]}$), {alo_poly_seconds / 60:.1f} min")

    idx = array_argmin(results.alo_poly_risks[seed, ..., im])
    axes[2].scatter(idx[1], idx[0], color="red", marker="*")
    print(f"RandALO risk: {np.min(results.alo_poly_risks[seed, ..., im])}")

    plt.tight_layout()
    fig.subplots_adjust(right=0.875)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.025, 0.7])
    fig.colorbar(axes[0].images[0], cax=cbar_ax)

    plt.savefig(os.path.join("figures", "fashion_mnist.pdf"), bbox_inches="tight")


def bks_viz():
    n = 200
    p = 150
    m = 10
    n_trials = 1000
    lamda = n

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    rng = np.random.default_rng(0)
    Z = rng.integers(0, 2, size=(n, p)) * 2 - 1
    sample_scales = np.ones(n)
    # sample_scales += np.arange(1, n + 1) / n
    sample_scales[n // 4 :] = 2
    sample_scales[n // 2 :] = 3
    sample_scales[n // 4 * 3 :] = 4
    feature_scales = np.ones(p)
    feature_scales[p // 2 :] = 2
    X = Z * sample_scales[:, None] * feature_scales[None, :]

    XTX_G_inv = np.linalg.inv(X.T @ X + lamda * np.eye(p))
    eta = np.trace(XTX_G_inv * feature_scales[None, :] ** 2)
    nu = np.trace(XTX_G_inv @ X.T @ X @ XTX_G_inv * feature_scales[None, :] ** 2)

    J = X @ np.linalg.solve(X.T @ X + lamda * np.eye(p), X.T)
    bks_means = np.diag(J)
    bks_stds = sample_scales * np.sqrt(nu) / (1 + sample_scales**2 * eta)

    bks_W = rng.integers(0, 2, size=(n, m, n_trials)) * 2 - 1
    D = np.einsum("ij,jkl->ikl", J, bks_W) * bks_W
    mu = np.mean(D, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.7), dpi=150)

    ax = axes[0]

    xs = np.linspace(np.min(mu), np.max(mu), 100)
    bins = np.linspace(np.min(mu), np.max(mu), 50)

    ax.hist(
        bks_means,
        bins=bins,
        density=True,
        alpha=0.5,
        label="$\\tilde{J}_{ii}$",
        color=color_cycle[2],
    )
    ax.hist(
        mu.ravel(),
        bins=bins,
        density=True,
        alpha=0.5,
        label="$\\mu_i$",
        color=color_cycle[0],
    )
    ax.plot(
        xs,
        sum(
            (
                stats.norm.pdf(xs, loc=mean, scale=std) / n
                for mean, std in zip(bks_means, bks_stds / np.sqrt(m))
            ),
            start=0,
        ),
        label="Theorem 1",
        color=color_cycle[1],
    )
    ax.legend()
    ax.set_title("Empirical distribution of $\\mu_i$'s")

    ax = axes[1]

    z_scores = (D[:, :1, :] - bks_means[:, None, None]) / bks_stds[:, None, None]
    z_scores_marginal = z_scores.ravel()

    zs = np.linspace(-3.5, 3.5, 100)
    bins = np.linspace(-3.5, 3.5, 50)
    ax.hist(z_scores_marginal, bins=bins, density=True, alpha=0.5, label="$z$-scores")
    ax.plot(zs, stats.norm.pdf(zs), label="$\\mathcal{N}(0, 1)$")
    ax.legend()
    ax.set_title("Gaussianity of $z$-scores")
    ax.set_ylim(0, 0.5)

    ax = axes[2]

    z_0 = z_scores[:-1, 0, 0].ravel()
    z_1 = z_scores[1:, 0, 0].ravel()
    corr = np.corrcoef(z_0, z_1)[0, 1]

    ax.scatter(z_0, z_1, alpha=0.5)
    ax.set_title(f"Pairwise correlation: {corr:.2g}")
    ax.set_xlabel("$z_i$")
    ax.set_ylabel("$z_{i+1}$")
    ax.axis("equal")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    plt.tight_layout(w_pad=0.1)
    axes[1].set_position(
        [
            axes[1].get_position().x0 - 0.01,
            axes[1].get_position().y0,
            axes[1].get_position().width,
            axes[1].get_position().height,
        ]
    )
    plt.savefig(os.path.join("figures", "bks_viz.pdf"), bbox_inches="tight")


collect_mapping = {
    "categorical_comp": categorical_comp,
    "lasso_bks_convergence": lasso_bks_convergence,
    "lasso_cv_tradeoff": lasso_cv_tradeoff,
    "lasso_scaling_normal": lasso_scaling_normal,
    "lasso_sweep": lasso_sweep,
    "confusion_matrix": confusion_matrix,
    "adelie_lasso_sweep": lasso_sweep,
    "logistic_comp": logistic_comp,
    "multivariate_t_comp": multivariate_t_comp,
    "random_forest_comp": random_forest_comp,
    "comp_all": comp_all,
    "truncated_normal_viz": truncated_normal_viz,
    "first_diff_scaling_1": first_diff_scaling_1,
    "fashion_mnist": fashion_mnist,
    "bks_viz": bks_viz,
}


if __name__ == "__main__":
    setup_matplotlib()

    benchmarks = sys.argv[1:]
    if len(benchmarks) == 0:
        benchmarks = collect_mapping.keys()

    for benchmark in benchmarks:
        if benchmark not in collect_mapping:
            raise ValueError(f"Unknown benchmark {benchmark}")

        try:
            results = load_results(os.path.join(benchmark, "results"))
            collect_mapping[benchmark](results)
        except FileNotFoundError:
            collect_mapping[benchmark]()
