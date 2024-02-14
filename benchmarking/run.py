import argparse
import json
import time

import numpy as np
import torch

from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold

from alogcv.alo import ALOExact, ALORandomized
from alogcv.models import LinearMixin, LassoModel, FirstDifferenceModel


class Timer(object):
    def __enter__(self):
        self.tic = time.monotonic()
        return self

    def __exit__(self, *args):
        self.toc = time.monotonic()
        self.elapsed = self.toc - self.tic


def extract_dict_keys(d, keys):
    return tuple(d[k] for k in keys)


def get_data(data_config, rng):

    if data_config["src"] == "iid_normal_sparse_awgn":
        n_train, n_test, p, s, sigma = extract_dict_keys(
            data_config, ["n_train", "n_test", "p", "s", "sigma"]
        )
        X_train = rng.normal(size=(n_train, p))
        X_test = rng.normal(size=(n_test, p))
        beta = np.zeros(p)
        beta[:s] = rng.normal(size=s) / np.sqrt(s)
        y_train = X_train @ beta + rng.normal(scale=sigma, size=n_train)
        y_test = X_test @ beta + rng.normal(scale=sigma, size=n_test)

        gen_risk_linear = (
            lambda beta_hat: (np.linalg.norm(beta - beta_hat) ** 2 + sigma**2) / 2
        )

    elif data_config["src"] == "iid_normal_first-diff-sparse_awgn":
        n_train, n_test, p, s, sigma = extract_dict_keys(
            data_config, ["n_train", "n_test", "p", "s", "sigma"]
        )
        X_train = rng.normal(size=(n_train, p))
        X_test = rng.normal(size=(n_test, p))

        beta = np.zeros(p)
        # normalize such that after cumsum, beta has unit norm in expectation
        beta[:s] = rng.normal(size=s) * np.sqrt(2 / s / p)
        rng.shuffle(beta)
        beta = np.cumsum(beta)

        y_train = X_train @ beta + rng.normal(scale=sigma, size=n_train)
        y_test = X_test @ beta + rng.normal(scale=sigma, size=n_test)

        gen_risk_linear = (
            lambda beta_hat: (np.linalg.norm(beta - beta_hat) ** 2 + sigma**2) / 2
        )

    else:
        raise ValueError(f"Unknown data source {data_config['src']}")

    def gen_risk(model):
        if isinstance(model, LinearMixin):
            beta_hat = model.coef_
            return gen_risk_linear(beta_hat)
        else:
            return None

    def test_risk(model, risk_fun):
        y_hat = model.predict(X_test)
        return np.mean(risk_fun(y_test, y_hat))

    return X_train, y_train, gen_risk, test_risk


def model_lookup(config):

    method = config["method"]
    method_kwargs = config["method_kwargs"].copy()

    if method == "lasso":
        p = config["data"]["p"]
        lamda = method_kwargs.pop("lamda0") / np.sqrt(p)
        return LassoModel(lamda, sklearn_lasso_kwargs=method_kwargs)

    if method == "first-difference":
        p = config["data"]["p"]
        lamda = method_kwargs.pop("lamda0")
        return FirstDifferenceModel(lamda, cvxpy_kwargs=method_kwargs)


def risk_lookup(risk_name):

    if risk_name == "squared_error":
        risk_fun = lambda y, y_hat: (y - y_hat) ** 2 / 2
    else:
        raise ValueError(f"Unknown risk function {risk_name}")

    return risk_fun


def cross_val_risk(model, X, y, risk_fun, k=5):

    cv = KFold(n_splits=k, shuffle=True)
    scores = []
    for train_idx, test_idx in cv.split(X):
        X_train, y_train = X[train_idx, :], y[train_idx]
        X_test, y_test = X[test_idx, :], y[test_idx]

        new_model = model.get_new_model()
        new_model.fit(X_train, y_train)
        y_hat = new_model.predict(X_test)

        scores.append(np.mean(risk_fun(y_test, y_hat)))

    return np.mean(scores)


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to the config file")
    parser.add_argument("results_file", help="path to the results file")
    args = parser.parse_args()

    print(f"Running experiment with config file {args.config_file}")
    with open(args.config_file, "r") as f:
        config = json.load(f)
    results = {
        "id": config["id"],
        "config_path": args.config_file,
        "config": config,
    }

    # Seed the random number generators
    rng = np.random.default_rng(config["seed"])
    np.random.seed(config["seed"])
    gen = torch.Generator()
    gen.manual_seed(config["seed"])

    print("Generating data...")
    X_train, y_train, gen_risk, test_risk = get_data(config["data"], rng)
    n, p = X_train.shape

    # Fit and evaluate the model on the whole data
    model = model_lookup(config)
    risk_fun = risk_lookup(config["risk"])

    print("Fitting model...")
    with Timer() as timer:
        model.fit(X_train, y_train)
    results["full_train_time"] = timer.elapsed

    with Timer() as timer:
        results["gen_risk"] = gen_risk(model)
    results["gen_risk_time"] = timer.elapsed

    with Timer() as timer:
        results["test_risk"] = test_risk(model, risk_fun)
    results["test_risk_time"] = timer.elapsed

    # Perform cross-validation
    for k in config["cv_k"]:
        print(f"Performing {k}-fold cross-validation...")
        with Timer() as timer:
            results[f"cv_{k}_risk"] = cross_val_risk(
                model, X_train, y_train, risk_fun, k=k
            )
        results[f"cv_{k}_risk_time"] = timer.elapsed

    print("Precomputing ALO Jacobian...")
    device = torch.device(config["device"])
    with Timer() as timer:
        model.jac(device)
    results["jac_time"] = timer.elapsed

    # Perform ALO
    y_train_torch = torch.tensor(y_train, device=device)
    y_hat_torch = torch.tensor(model.predict(X_train), device=device)

    print("Performing exact ALO...")
    with Timer() as timer:
        diag_jac = model.jac(device).diag
        alo_exact = ALOExact(
            model.loss_fun,
            y_train_torch,
            y_hat_torch,
            diag_jac,
        )
        results["alo_exact_risk"] = alo_exact.eval_risk(risk_fun) / n
    results["alo_exact_time"] = timer.elapsed

    alo = None
    running_matvec_time = 0
    for m in sorted(config["alo_m"]):
        print(f"Performing randomized ALO up to {m} matvecs...")
        with Timer() as timer:
            if alo is None:
                alo = ALORandomized(
                    model.loss_fun,
                    y_train_torch,
                    y_hat_torch,
                    model.jac(device),
                    m,
                    generator=gen,
                )
            else:
                alo.do_diag_jac_estims_upto(m)
        running_matvec_time += timer.elapsed
        results[f"alo_{m}_matvec_time"] = running_matvec_time

        with Timer() as timer:
            results[f"alo_{m}_bks_risk"] = alo.eval_risk(risk_fun, order=None) / n
        results[f"alo_{m}_bks_risk_time"] = timer.elapsed

        with Timer() as timer:
            results[f"alo_{m}_poly_risk"] = alo.eval_risk(risk_fun, order=1) / n
        results[f"alo_{m}_poly_risk_time"] = timer.elapsed

    # Save the results
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=4)
