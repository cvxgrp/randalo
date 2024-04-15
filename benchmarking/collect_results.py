from dataclasses import dataclass
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


def relative_error(a, b):
    return np.abs(a - b) / b


def grouped_boxplot(data, x_labels, group_labels, ax=None, legend=True, **kwargs):

    if ax is None:
        ax = plt.gca()

    n_groups = len(group_labels)
    n_boxes = len(x_labels)

    width = 0.8 / n_groups
    x = np.arange(n_boxes)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    hatches = [
        "",
        "///",
        "xxx",
        "ooo",
        "+++",
        "\\\\\\",
        "///",
        "xxx",
        "ooo",
        "+++",
        "\\\\\\",
    ]

    for i, group_label in enumerate(group_labels):
        box_data = data[i]
        ax.boxplot(
            box_data.T,
            positions=x + (i - n_groups / 2 + 0.5) * width,
            widths=width,
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
    ]

    results = extract_all_results(results, axes_keys)
    ns, seeds = results.axes
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

    # First, plot only BKS

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    grouped_boxplot(
        [
            results.test_risks,
            results.cv_risks[..., i_k],
            results.alo_bks_risks[..., i_m1],
            results.alo_bks_risks[..., i_m2],
        ],
        [f"n={n}" for n in ns],
        [
            "Test error",
            f"CV($K={k}$)",
            f"BKS-ALO($m={m1}$)",
            f"BKS-ALO($m={m2}$)",
        ],
        ax=axes[0],
    )

    axes[0].set_title("Risk vs. Sample Size")
    axes[0].set_ylabel("Squared Error")
    axes[0].set_xlabel("Sample Size")

    grouped_boxplot(
        [
            results.full_train_times,
            results.cv_times[..., i_k],
            results.alo_bks_times[..., i_m1],
            results.alo_bks_times[..., i_m2],
        ],
        [f"n={n}" for n in ns],
        ["Training", f"CV($K={k}$)", f"BKS-ALO($m={m1}$)", f"BKS-ALO($m={m2}$)"],
        ax=axes[1],
    )

    axes[1].set_title("Time vs. Sample Size")
    axes[1].set_ylabel("Time (s)")
    axes[1].set_xlabel("Sample Size")
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(
        os.path.join("figures", "lasso_scaling_normal_bks.pdf"), bbox_inches="tight"
    )

    # Next, plot both BKS and Poly

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    grouped_boxplot(
        [
            results.test_risks,
            results.cv_risks[..., i_k],
            results.alo_bks_risks[..., i_m3],
            results.alo_poly_risks[..., i_m3],
        ],
        [f"n={n}" for n in ns],
        [
            "Test error",
            f"CV($K={k}$)",
            f"BKS-ALO($m={m3}$)",
            f"RandALO($m={m3}$)",
        ],
        ax=axes[0],
    )

    axes[0].set_title("Risk vs. Sample Size")
    axes[0].set_ylabel("Squared Error")
    axes[0].set_xlabel("Sample Size")

    grouped_boxplot(
        [
            results.full_train_times,
            results.cv_times[..., i_k],
            results.alo_bks_times[..., i_m3],
            results.alo_poly_times[..., i_m3],
        ],
        [f"n={n}" for n in ns],
        [
            "Training",
            f"CV($K={k}$)",
            f"BKS-ALO($m={m3}$)",
            f"RandALO($m={m3}$)",
        ],
        ax=axes[1],
    )

    axes[1].set_title("Time vs. Sample Size")
    axes[1].set_ylabel("Time (s)")
    axes[1].set_xlabel("Sample Size")
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(
        os.path.join("figures", "lasso_scaling_normal_both.pdf"), bbox_inches="tight"
    )


def lasso_poly_scatter(results):

    axes_keys = [
        ["config", "seed"],
    ]

    results = extract_all_results(results, axes_keys)
    (seeds,) = results.axes

    ms = np.asarray(results.alo_m)
    ms_recip = 1 / ms
    ms_recip = np.concatenate([1 / ms[:-1], [0.0]])

    plt.plot(1 / ms, np.median(results.alo_bks_risks, axis=0).T, label="BKS-ALO")
    # plt.plot(1 / ms, np.median(results.alo_poly_risks, axis=0).T, "-.", label="RandALO")

    plt.fill_between(
        ms_recip,
        *np.percentile(results.alo_bks_risks, [25, 75], axis=0),
        alpha=0.2,
    )
    # plt.fill_between(
    #   ms_recip,
    #   *np.percentile(results.alo_poly_risks, [25, 75], axis=0),
    #   alpha=0.2,
    # )

    plt.axhline(np.median(results.test_risks), color="black", linestyle="--")
    # fill between interquartile range of test risk
    xlim = np.asarray([0, np.max(ms_recip)])
    plt.fill_between(
        xlim,
        *np.percentile(results.test_risks, [25, 75])[:, None],
        facecolor="black",
        alpha=0.2,
    )
    ylim = plt.ylim()
    aspect_ratio = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])

    for angle in np.linspace(0, np.pi / 2, 9)[1:-1]:
        plt.plot(
            xlim,
            np.median(results.alo_bks_risks[:, -1])
            + np.tan(angle) * (xlim - xlim[0]) / aspect_ratio,
            color="black",
            linestyle=":",
            alpha=0.2,
        )

    # ms_filter = np.asarray([m for m in ms if m >= 25 and m <= 50])
    # i_ms = [results.alo_m.index(m) for m in ms_filter]

    inv_ms = np.array(
        [
            0.04,
            0.04,
            0.03846154,
            0.03846154,
            0.03703704,
            0.03703704,
            0.03571429,
            0.03571429,
            0.03448276,
            0.03448276,
            0.03333333,
            0.03333333,
            0.03225806,
            0.03225806,
            0.03125,
            0.03125,
            0.03030303,
            0.03030303,
            0.02941176,
            0.02941176,
            0.02857143,
            0.02857143,
            0.02777778,
            0.02777778,
            0.02702703,
            0.02702703,
            0.02631579,
            0.02631579,
            0.02564103,
            0.02564103,
            0.025,
            0.025,
            0.02439024,
            0.02439024,
            0.02380952,
            0.02380952,
            0.02325581,
            0.02325581,
            0.02272727,
            0.02272727,
            0.02222222,
            0.02222222,
            0.02173913,
            0.02173913,
            0.0212766,
            0.0212766,
            0.02083333,
            0.02083333,
            0.02040816,
            0.02,
        ]
    )

    risks = (
        np.array(
            [
                3498.54348982,
                3523.03338677,
                3499.892891,
                3512.83093777,
                3466.28244952,
                3477.67107397,
                3471.00952082,
                3482.95405323,
                3477.99106963,
                3470.18876317,
                3487.27051889,
                3507.85561273,
                3462.8255875,
                3483.77628774,
                3447.1151502,
                3466.61032851,
                3461.12945087,
                3439.49563918,
                3444.53416898,
                3423.68149226,
                3443.14867226,
                3454.15553333,
                3441.09913286,
                3421.56535423,
                3430.39672024,
                3431.31171564,
                3439.73497225,
                3438.51568312,
                3438.80234594,
                3427.48612042,
                3415.04565812,
                3442.95227054,
                3412.95298509,
                3419.79949664,
                3416.64411817,
                3425.05715343,
                3419.45085793,
                3407.72558733,
                3410.82475155,
                3414.77051308,
                3423.45816518,
                3421.52255356,
                3392.004082,
                3408.92798226,
                3415.76117281,
                3405.41355706,
                3405.37847544,
                3407.77875578,
                3405.28154779,
                3400.74429551,
            ]
        )
        / 5000
    )
    w = np.array([3292.98379179, 5329.12757204]) / 5000

    plt.scatter(inv_ms, risks, label="subsampled-sketch risks")
    plt.plot(
        np.linspace(0, 0.1),
        np.vander(np.linspace(0, 0.1), 2, True) @ w,
        label="$\hat{R}_0 + \hat{R}_1 / m$",
    )

    """
    risks2 = np.array(
                 [3397.34771693, 3404.29966769, 3398.08577415, 3382.13835186,
                  3411.48750507, 3403.72133267, 3377.29187079, 3374.96243406,
                  3401.34065039, 3387.74458755, 3397.01563313, 3355.18600619,
                  3385.97613629, 3396.00171822, 3392.19923611, 3369.49146275,
                  3387.12345259, 3352.85233383, 3368.07286225, 3356.16275025,
                  3330.18840125, 3353.53887847, 3334.69174004, 3330.95607431,
                  3339.93836132, 3329.02531471, 3340.34686949, 3367.89827597,
                  3334.46692016, 3357.2325125 , 3335.31413061, 3337.9627052 ,
                  3335.074799  , 3313.70698876, 3323.48381804, 3333.17909516,
                  3315.68539049, 3343.88620567, 3316.46477452, 3320.06683743,
                  3326.21291554, 3322.79253465, 3332.70922716, 3311.90864988,
                  3317.9416406 , 3327.93128793, 3320.6523137 , 3317.04343804,
                  3320.72311414, 3312.36866818]) / 5000
    w2 = np.array([3216.47397074, 4820.6113051 ]) / 5000

    plt.scatter(
        inv_ms,
        risks2,
        label='subsampled-sketch risks'
    )
    plt.plot(
        np.linspace(0, 0.1),
        np.vander(np.linspace(0, 0.1), 2, True) @ w2,
        label='$\hat{R}_0 + \hat{R}_1 / m$'
    )
    """

    plt.xlim(np.asarray([0, 0.06]))
    plt.ylim(np.asarray([0.638, 0.72]))

    # plt.title("Convergence of Risk Estimate in $m$")
    plt.xlabel("$1/m$")
    plt.ylabel("Risk estimate")
    plt.legend()

    plt.show()


def lasso_bks_convergence(results):

    axes_keys = [
        ["config", "seed"],
    ]

    results = extract_all_results(results, axes_keys)
    (seeds,) = results.axes

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ms = np.asarray(results.alo_m)
    # ms_recip = ms
    # ms_recip = np.concatenate([1 / ms[:-1], [0.0]])

    plt.plot(
        ms,
        np.median(results.alo_bks_risks, axis=0).T,
        "--",
        color=color_cycle[0],
        label="BKS-ALO",
    )
    plt.plot(
        ms,
        np.median(results.alo_poly_risks, axis=0).T,
        color=color_cycle[1],
        label="RandALO",
    )

    plt.fill_between(
        ms,
        *np.percentile(results.alo_bks_risks, [25, 75], axis=0),
        color=color_cycle[0],
        alpha=0.2,
    )
    plt.fill_between(
        ms,
        *np.percentile(results.alo_poly_risks, [25, 75], axis=0),
        color=color_cycle[1],
        alpha=0.2,
    )

    plt.axhline(
        np.median(results.test_risks), color="black", linestyle=":", label="Test error"
    )
    # fill between interquartile range of test risk
    # xlim = np.asarray([0, np.max(ms_recip)])
    # plt.xscale("log")
    xlim = np.asarray([np.min(ms), np.max(ms)])
    xlim = np.asarray([np.min(ms), 300])
    plt.fill_between(
        xlim,
        *np.percentile(results.test_risks, [25, 75])[:, None],
        facecolor="black",
        alpha=0.2,
    )
    ylim = plt.ylim()

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.title("Convergence to ALO")
    plt.xlabel("Number of matvecs $m$")
    plt.ylabel("Risk estimate")
    plt.legend()

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

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "D", "v", "^", ">", "<", "p", "h", "H", "d", "P", "X"]

    data = [
        results.test_risks,
        results.cv_risks[..., i_k],
        results.alo_bks_risks[..., i_m],
        results.alo_poly_risks[..., i_m],
    ]
    labels = [
        "Test error",
        f"CV(K={k})",
        f"BKS-ALO(m={m})",
        f"RandALO(m={m})",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    for i, (label, color) in enumerate(zip(labels, color_cycle)):
        # shift markevery by 1 to avoid overlapping markers
        markevery = (i, len(data))
        axes[0].errorbar(
            lamda0s,
            np.mean(data[i], axis=1),
            yerr=np.std(data[i], axis=1),
            label=label,
            color=color,
            marker=markers[i],
            markevery=markevery,
            errorevery=markevery,
            capsize=2,
        )

    axes[0].set_xscale("log")
    axes[0].set_title("Risk")
    axes[0].set_ylabel("Squared Error")
    axes[0].set_xlabel("Regularization parameter $\\lambda$")
    axes[0].legend()

    data = [
        results.full_train_times,
        results.cv_times[..., i_k],
        results.alo_bks_times[..., i_m],
        results.alo_poly_times[..., i_m],
    ]
    labels = [
        "Training",
        f"CV(K={k})",
        f"BKS-ALO(m={m})",
        f"RandALO(m={m})",
    ]

    for i, (label, color) in enumerate(zip(labels, color_cycle)):
        # shift markevery by 1 to avoid overlapping markers
        markevery = (i, len(data))
        axes[1].errorbar(
            lamda0s,
            np.mean(data[i], axis=1),
            yerr=np.std(data[i], axis=1),
            label=label,
            color=color,
            marker=markers[i],
            markevery=markevery,
            errorevery=markevery,
            capsize=2,
        )

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Time (s)")
    axes[1].set_xlabel("Regularization parameter $\\lambda$")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join("figures", "lasso_sweep.pdf"), bbox_inches="tight")


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

    fig, axes = plt.subplots(1, 2, figsize=(5, 4), dpi=300)

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
            results.full_train_times,
            results.cv_times[..., i_k],
            results.alo_bks_times[..., i_m],
            results.alo_poly_times[..., i_m],
        ],
        [""],
        ["Training", f"CV(K={k})", f"BKS-ALO(m={m})", f"RandALO(m={m})"],
        ax=axes[1],
    )
    axes[1].set_title("Time (s)")
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    grouped_boxplot(
        [
            results.test_risks[i_ns, ..., i_lamda0],
            results.cv_risks[i_ns, ..., i_lamda0, i_k],
            results.alo_bks_risks[i_ns, ..., i_lamda0, i_m],
            results.alo_poly_risks[i_ns, ..., i_lamda0, i_m],
        ],
        [f"n={n}" for n in ns_filter],
        [
            "Test error",
            f"CV($K={k}$)",
            f"BKS-ALO($m={m}$)",
            f"RandALO($m={m}$)",
        ],
        ax=axes[0],
    )

    axes[0].set_title("Risk vs. Sample Size")
    axes[0].set_ylabel("Squared Error")
    axes[0].set_xlabel("Sample Size")

    grouped_boxplot(
        [
            results.full_train_times[i_ns, ..., i_lamda0],
            results.cv_times[i_ns, ..., i_lamda0, i_k],
            results.alo_bks_times[i_ns, ..., i_lamda0, i_m],
            results.alo_poly_times[i_ns, ..., i_lamda0, i_m],
        ],
        [f"n={n}" for n in ns_filter],
        ["Training", f"CV($K={k}$)", f"BKS-ALO($m={m}$)", f"RandALO($m={m}$)"],
        ax=axes[1],
    )

    axes[1].set_title("Time vs. Sample Size")
    axes[1].set_ylabel("Time (s)")
    axes[1].set_xlabel("Sample Size")
    axes[1].set_yscale("log")

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


collect_mapping = {
    "categorical_comp": categorical_comp,
    "lasso_bks_convergence": lasso_bks_convergence,
    "lasso_poly_scatter": lasso_poly_scatter,
    "lasso_scaling_normal": lasso_scaling_normal,
    "lasso_sweep": lasso_sweep,
    "logistic_comp": logistic_comp,
    "multivariate_t_comp": multivariate_t_comp,
    "random_forest_comp": random_forest_comp,
    "first_diff_scaling_1": first_diff_scaling_1,
}


if __name__ == "__main__":

    benchmarks = sys.argv[1:]
    if len(benchmarks) == 0:
        benchmarks = collect_mapping.keys()

    for benchmark in benchmarks:
        if benchmark not in collect_mapping:
            raise ValueError(f"Unknown benchmark {benchmark}")

        results = load_results(os.path.join(benchmark, "results"))
        collect_mapping[benchmark](results)
