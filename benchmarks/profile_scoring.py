"""Interleaved CPU benchmark against a Git revision of the public scoring API.

Run from any directory, for example:
    python benchmarks/profile_scoring.py --baseline-ref HEAD --output timings.json

The baseline source is loaded without changing the checkout. Full API timings
use fresh probe caches; cached timings and truncated-normal timings isolate the
scoring work. No GPU, model fitting, or external datasets are required.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import linops as lo
import torch
from randalo import RandALO, truncnorm
from randalo import modeling_layer as ml


def baseline_module(filename, revision):
    module = types.ModuleType("randalo._baseline_" + filename)
    source = subprocess.check_output(
        ["git", "show", f"{revision}:randalo/{filename}.py"], cwd=ROOT, text=True
    )
    exec(compile(source, f"<baseline>/{filename}.py", "exec"), module.__dict__)  # noqa: S102 -- explicitly selected local Git source
    return module


def paired_times(functions, repetitions):
    for function in functions.values():
        function()
    times = {name: [] for name in functions}
    for i in range(repetitions):
        order = list(functions) if i % 2 else list(reversed(functions))
        for name in order:
            start = time.perf_counter()
            functions[name]()
            times[name].append(time.perf_counter() - start)
    return {
        "median_seconds": {name: statistics.median(t) for name, t in times.items()},
        "times": times,
    }


def run_case(args, original, threads, dtype):
    rng = torch.Generator().manual_seed(123)
    n, p, m = args.rows, args.features, 50
    x = torch.randn(n, p, generator=rng) / (2 * n) ** 0.5
    jac = lo.MatrixOperator(x) @ lo.MatrixOperator(x.T)
    y = torch.randn(n, generator=rng)
    fitted = torch.randn(n, generator=rng)
    loss = ml.MSELoss()
    risk = lambda y, z: loss(y, z).item()
    classes = {"before": original.RandALO, "after": RandALO}

    def estimator(label):
        return classes[label](
            loss=loss,
            jac=jac,
            y=y,
            y_hat=fitted,
            dtype=dtype,
            rng=torch.Generator().manual_seed(456),
        )

    subset_rng = torch.Generator().manual_seed(789)
    subsets = [
        torch.randperm(m, generator=subset_rng)[:k]
        for k in torch.linspace(m // 2, m, 50, dtype=int)
    ]
    cached = {label: estimator(label) for label in classes}
    for ra in cached.values():
        ra.evaluate_bks(risk, n_matvecs=m)
    scores = {
        label: ra.evaluate(risk, n_matvecs=m, subsets=subsets)
        for label, ra in cached.items()
    }
    torch.testing.assert_close(
        torch.tensor(scores["before"]),
        torch.tensor(scores["after"]),
        rtol=1e-6 if dtype == torch.float32 else 1e-11,
        atol=1e-7 if dtype == torch.float32 else 1e-12,
    )
    row = {
        "threads": threads,
        "dtype": str(dtype),
        "n": n,
        "p": p,
        "probes": m,
        "subsets": 50,
        "scores": scores,
    }
    row["cached_randalo"] = paired_times(
        {
            label: lambda ra=ra: ra.evaluate(risk, n_matvecs=m, subsets=subsets)
            for label, ra in cached.items()
        },
        args.repetitions,
    )
    row["cached_bks"] = paired_times(
        {
            label: lambda ra=ra: ra.evaluate_bks(risk, n_matvecs=m)
            for label, ra in cached.items()
        },
        args.repetitions,
    )

    h = cached["before"]._uniform_map_estimates(
        cached["before"]._normalized_diag_jac_estims[:, :m]
        @ torch.full((m, 50), 1.0 / m),
        torch.full((1, 50), m),
    )

    def old_correction():
        return [
            cached["before"]._y_tilde_from_normalized_jac(h[:, j]) for j in range(50)
        ]

    def new_correction():
        diagonal = h.T.contiguous().clamp(0, 1)
        ra = cached["after"]
        return ra._y_hat + ra._a_c * diagonal / (1 - diagonal)

    torch.testing.assert_close(
        torch.stack(old_correction()), new_correction(), rtol=0, atol=0
    )
    row["prediction_correction"] = paired_times(
        {"before": old_correction, "after": new_correction}, args.repetitions
    )

    def full(label):
        ra = estimator(label)
        ra.evaluate_bks(risk, n_matvecs=m)
        return ra.evaluate(risk, n_matvecs=m, subsets=subsets)

    row["full_api"] = paired_times(
        {label: lambda label=label: full(label) for label in classes},
        args.repetitions,
    )
    mu = torch.linspace(-0.1, 1.1, n * 50).reshape(n, 50)
    sigma = torch.full((n, 1), 0.03)
    bounds = (torch.tensor(0.0), torch.tensor(1.0))
    torch.testing.assert_close(
        truncnorm.truncnorm_mean(mu, sigma, *bounds),
        original.truncnorm.truncnorm_mean(mu, sigma, *bounds),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    row["truncated_normal"] = paired_times(
        {
            "before": lambda: original.truncnorm.truncnorm_mean(mu, sigma, *bounds),
            "after": lambda: truncnorm.truncnorm_mean(mu, sigma, *bounds),
        },
        args.repetitions,
    )
    row["speedups"] = {
        phase: row[phase]["median_seconds"]["before"]
        / row[phase]["median_seconds"]["after"]
        for phase in (
            "cached_randalo",
            "cached_bks",
            "full_api",
            "truncated_normal",
            "prediction_correction",
        )
    }
    return row


def main(args):
    original = baseline_module("randalo", args.baseline_ref)
    original.truncnorm = baseline_module("truncnorm", args.baseline_ref)
    results = []
    previous_dtype = torch.get_default_dtype()
    previous_threads = torch.get_num_threads()
    try:
        for threads in args.threads:
            torch.set_num_threads(threads)
            for dtype in (torch.float32, torch.float64):
                # The unchanged mixing-matrix helper uses Torch's default dtype.
                torch.set_default_dtype(dtype)
                row = run_case(args, original, threads, dtype)
                results.append(row)
                print(
                    json.dumps({k: row[k] for k in ("threads", "dtype", "speedups")}),
                    flush=True,
                )
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_num_threads(previous_threads)
    args.output.write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "baseline_ref": args.baseline_ref,
                "repetitions": args.repetitions,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-ref", default="HEAD")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--output", type=Path, default=Path("timings.json"))
    main(parser.parse_args())
