import argparse
import json
import os
import time

import numpy as np
from scipy import sparse
import torch

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from alogcv.alo import ALOExact, ALORandomized, GCV
from alogcv.models import (
    LinearMixin,
    LassoModel,
    FirstDifferenceModel,
    LogisticModel,
    RandomForestRegressorModel,
)
from alogcv.utils import GaussianGridIntegrator, sigmoid


class Timer(object):
    def __enter__(self):
        self.tic = time.monotonic()
        return self

    def __exit__(self, *args):
        self.toc = time.monotonic()
        self.elapsed = self.toc - self.tic

    @property
    def running(self):
        return time.monotonic() - self.tic


GLOBAL_TIMER = Timer().__enter__()


def seconds_to_str(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"


def log(message, *args, **kwargs):
    print(f"[{seconds_to_str(GLOBAL_TIMER.running)}] {message}", *args, **kwargs)


def extract_dict_keys(d, keys):
    return tuple(d[k] for k in keys)


def generate_sparse_normal(n, p, nz_ratio, rng=None):

    if n > p:
        p, n = n, p
        transposed = True
    else:
        transposed = False

    if rng is None:
        rng = np.random.default_rng()

    ##################################
    # Implementation based on ChatGPT
    ##################################
    rows, cols, data = [], [], []

    # Generate the sparse matrix
    for i in range(n):
        # Determine the number of non-zero elements for the current row
        non_zero_elements = rng.binomial(p, nz_ratio)

        # Choose random positions for the non-zero elements
        if non_zero_elements > 0:
            col_indices = rng.choice(p, non_zero_elements, replace=False)

            # Generate the non-zero values from a Gaussian distribution
            values = rng.normal(scale=1 / np.sqrt(nz_ratio), size=non_zero_elements)

            # Append the information to the lists
            rows.extend([i] * non_zero_elements)
            cols.extend(col_indices)
            data.extend(values)

    # Create the CSR matrix
    X = sparse.csr_matrix((data, (rows, cols)), shape=(n, p))

    if transposed:
        X = X.T
    return X


class AddMissingIndicatorAndImpute(BaseEstimator, TransformerMixin):
    """Custom transformer for adding missing indicators and imputing"""

    def __init__(self):
        self.imputer = SimpleImputer(strategy="constant", fill_value=0)
        self.indicator = MissingIndicator(features="missing-only", error_on_new=False)

    def fit(self, X, y=None):
        X = X.astype(float)
        self.imputer.fit(X, y)
        self.indicator.fit(X, y)
        return self

    def transform(self, X):
        X = X.astype(float)
        imputed_data = self.imputer.transform(X)
        missing_indicator = self.indicator.transform(X)
        return np.hstack((imputed_data, missing_indicator))


def load_kddcup09(task):

    # load and preprocess the training data
    with open(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "kddcup09",
            "orange_small_train.data",
            "orange_small_train.data",
        ),
        "r",
    ) as f:
        X_train = np.loadtxt(f, delimiter="\t", skiprows=1, dtype=object)

    X_train[X_train == ""] = np.nan

    numeric_transformer = Pipeline(
        steps=[("add_indicator_and_impute", AddMissingIndicatorAndImpute())]
    )

    # For categorical features: impute missing values and then apply one-hot encoding
    categorical_transformer = Pipeline(
        steps=[
            # ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    numeric_features = slice(0, 190)
    categorical_features = slice(190, 230)

    # Combine transformers into a ColumnTransformer
    preprocessor = Pipeline(
        steps=[
            (
                "column_transformer",
                ColumnTransformer(
                    transformers=[
                        ("num", numeric_transformer, numeric_features),
                        ("cat", categorical_transformer, categorical_features),
                    ]
                ),
            ),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    X_train = preprocessor.fit_transform(X_train)

    # load the target variable
    with open(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "kddcup09",
            f"orange_small_train_{task}.labels",
        ),
        "r",
    ) as f:
        y_train = np.loadtxt(f)

    return X_train, y_train


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

    elif data_config["src"] == "varying_norm_sparse_awgn":
        n_train, n_test, p, s, sigma = extract_dict_keys(
            data_config, ["n_train", "n_test", "p", "s", "sigma"]
        )

        # scale by offset exponential chosen to have unit second (non-central) moment
        rate = 2
        offset = (
            np.sqrt(1 - 1 / rate**2) - 1 / rate
        )  # for rate = 2, this is (sqrt(3) - 1) / 2
        assert offset > 0

        X_train = (
            rng.normal(size=(n_train, p))
            * (offset + rng.exponential(scale=1 / rate, size=n_train))[:, None]
        )
        X_test = (
            rng.normal(size=(n_test, p))
            * (offset + rng.exponential(scale=1 / rate, size=n_test))[:, None]
        )

        beta = np.zeros(p)
        beta[:s] = rng.normal(size=s) / np.sqrt(s)

        y_train = X_train @ beta + rng.normal(scale=sigma, size=n_train)
        y_test = X_test @ beta + rng.normal(scale=sigma, size=n_test)

        gen_risk_linear = (
            lambda beta_hat: (np.linalg.norm(beta - beta_hat) ** 2 + sigma**2) / 2
        )

    elif data_config["src"] == "multivariate_t_sparse_awgn":
        n_train, n_test, p, nu, s, sigma = extract_dict_keys(
            data_config, ["n_train", "n_test", "p", "nu", "s", "sigma"]
        )
        X_train = rng.normal(size=(n_train, p)) * np.sqrt(
            (nu - 2) / rng.chisquare(nu, size=n_train)[:, None]
        )
        X_test = rng.normal(size=(n_test, p)) * np.sqrt(
            (nu - 2) / rng.chisquare(nu, size=n_test)[:, None]
        )

        beta = np.zeros(p)
        beta[:s] = rng.normal(size=s) / np.sqrt(s)

        y_train = X_train @ beta + rng.normal(scale=sigma, size=n_train)
        y_test = X_test @ beta + rng.normal(scale=sigma, size=n_test)

        gen_risk_linear = (
            lambda beta_hat: (np.linalg.norm(beta - beta_hat) ** 2 + sigma**2) / 2
        )

    elif data_config["src"] == "categorical_sparse_awgn":
        n_train, n_test, p, n_categories, s, sigma = extract_dict_keys(
            data_config, ["n_train", "n_test", "p", "n_categories", "s", "sigma"]
        )
        X0_train = rng.integers(0, n_categories, size=(n_train, p))
        onehot = OneHotEncoder()
        X_train = onehot.fit_transform(X0_train) * np.sqrt(n_categories)
        X0_test = rng.integers(0, n_categories, size=(n_test, p))
        X_test = onehot.transform(X0_test) * np.sqrt(n_categories)
        beta = np.zeros(n_categories * p)
        nz_idx = rng.choice(p, s, replace=False)
        beta[nz_idx] = rng.normal(size=s) / np.sqrt(s)
        y_train = X_train @ beta + rng.normal(scale=sigma, size=n_train)
        y_test = X_test @ beta + rng.normal(scale=sigma, size=n_test)

        def gen_risk_linear(beta_hat):
            delta = beta - beta_hat
            risk = np.linalg.norm(delta) ** 2 + sigma**2
            risk += np.sum(delta) ** 2 / n_categories
            for j in range(p):
                i = n_categories * j
                risk -= np.sum(delta[i : i + n_categories]) ** 2 / n_categories
            return risk / 2

    elif data_config["src"] == "iid_normal_logistic_sparse":
        n_train, n_test, p, s, rho = extract_dict_keys(
            data_config, ["n_train", "n_test", "p", "s", "rho"]
        )
        X_train = rng.normal(size=(n_train, p))
        beta = np.zeros(p)
        beta[:s] = rng.normal(size=s) / np.sqrt(s)
        prob = sigmoid(X_train @ beta * rho)
        y_train = (rng.uniform(size=n_train) < prob).astype(float) * 2 - 1

        X_test = rng.normal(size=(n_test, p))
        prob = sigmoid(X_test @ beta * rho)
        y_test = (rng.uniform(size=n_test) < prob).astype(float) * 2 - 1

        def gen_risk_linear(beta_hat):
            Sigma = np.array(
                [[beta @ beta, beta @ beta_hat], [beta_hat @ beta, beta_hat @ beta_hat]]
            )
            integrator = GaussianGridIntegrator(
                Sigma=Sigma, n_samples_per_dim=1000, b=7
            )
            fun = lambda z: sigmoid(-np.sign(z[1, ...]) * z[0, ...] * rho)
            return integrator.integrate(fun)

    elif data_config["src"] == "iid_sparse_normal_sparse_awgn":
        n_train, n_test, p, nz_ratio, s, sigma = extract_dict_keys(
            data_config, ["n_train", "n_test", "p", "nz_ratio", "s", "sigma"]
        )

        X_train = generate_sparse_normal(n_train, p, nz_ratio, rng)
        X_test = generate_sparse_normal(n_test, p, nz_ratio, rng)
        beta = np.zeros(p)
        beta[:s] = rng.normal(size=s) / np.sqrt(s)
        y_train = X_train @ beta + rng.normal(scale=sigma, size=n_train)
        y_test = X_test @ beta + rng.normal(scale=sigma, size=n_test)

        gen_risk_linear = (
            lambda beta_hat: (np.linalg.norm(beta - beta_hat) ** 2 + sigma**2) / 2
        )

    elif data_config["src"] == "iid_normal_sparse_poly_awgn":
        n_train, n_test, p, ord, s, sigma = extract_dict_keys(
            data_config, ["n_train", "n_test", "p", "ord", "s", "sigma"]
        )
        X_train = rng.normal(size=(n_train, p))
        X_test = rng.normal(size=(n_test, p))

        X_train_poly = np.zeros((n_train, s))
        X_test_poly = np.zeros((n_test, s))
        for i in range(s):
            poly_factors = rng.integers(0, p, size=ord)
            X_train_poly[:, i] = np.prod(X_train[:, poly_factors], axis=1)
            X_test_poly[:, i] = np.prod(X_test[:, poly_factors], axis=1)

        beta = rng.normal(size=s) / np.mean(np.linalg.norm(X_train_poly, axis=1))
        y_train = X_train_poly @ beta + rng.normal(scale=sigma, size=n_train)
        y_test = X_test_poly @ beta + rng.normal(scale=sigma, size=n_test)

        gen_risk_linear = (
            lambda beta_hat: (np.linalg.norm(beta - beta_hat) ** 2 + sigma**2) / 2
        )

    elif data_config["src"] == "kddcup09_upselling":
        X_train, y_train = load_kddcup09("upselling")
        gen_risk_linear = None
        X_test = y_test = None

    else:
        raise ValueError(f"Unknown data source {data_config['src']}")

    def gen_risk(model):
        if isinstance(model, LinearMixin) and callable(gen_risk_linear):
            beta_hat = model.coef_
            return gen_risk_linear(beta_hat)
        else:
            return None

    def test_risk(model, risk_fun):
        if X_test is None or y_test is None:
            return None
        y_hat = model.predict(X_test)
        return np.mean(risk_fun(y_test, y_hat))

    return X_train, y_train, gen_risk, test_risk


def model_lookup(config):

    method = config["method"]
    method_kwargs = config["method_kwargs"].copy()

    if method == "lasso":
        p = config["data"]["p"]
        lamda = method_kwargs.pop("lamda0") / np.sqrt(p)
        direct = method_kwargs.pop("direct", None)
        return LassoModel(lamda, sklearn_lasso_kwargs=method_kwargs, direct=direct)

    if method == "first-difference":
        lamda = method_kwargs.pop("lamda0")
        return FirstDifferenceModel(lamda, cvxpy_kwargs=method_kwargs)

    if method == "logistic":
        p = config["data"]["p"]
        if method_kwargs["penalty"] == "l1":
            lamda = method_kwargs.pop("lamda0") / np.sqrt(p)
        elif method_kwargs["penalty"] == "l2":
            lamda = method_kwargs.pop("lamda0")
        direct = method_kwargs.pop("direct", None)
        return LogisticModel(
            lamda, sklearn_logistic_kwargs=method_kwargs, direct=direct
        )

    if method == "random-forest":
        return RandomForestRegressorModel(sklearn_rf_kwargs=method_kwargs)


def risk_lookup(risk_name):

    if risk_name == "squared_error":
        risk_fun = lambda y, y_hat: (y - y_hat) ** 2 / 2
    elif risk_name == "zero_one":
        risk_fun = lambda y, y_hat: y * y_hat <= 0
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


def get_centered_norm2s(X):
    X_mean = X.mean(axis=0)
    if sparse.issparse(X):
        X_mean = X_mean.A1
        X_norm2 = X.power(2).sum(axis=1).A1
    else:
        X_norm2 = (X**2).sum(axis=1)
    return X_norm2 - 2 * X @ X_mean + X_mean @ X_mean


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to the config file")
    parser.add_argument("results_file", help="path to the results file")
    args = parser.parse_args()

    log(f"Running experiment with config file {args.config_file}")
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

    log("Generating/loading data...")
    X_train, y_train, gen_risk, test_risk = get_data(config["data"], rng)
    n, p = X_train.shape

    # Fit and evaluate the model on the whole data
    model = model_lookup(config)
    risk_fun = risk_lookup(config["risk"])

    log("Fitting model...")
    with Timer() as timer:
        model.fit(X_train, y_train)
    results["full_train_time"] = timer.elapsed
    results["train_risk"] = np.mean(risk_fun(y_train, model.predict(X_train)))

    with Timer() as timer:
        results["gen_risk"] = gen_risk(model)
    results["gen_risk_time"] = timer.elapsed

    with Timer() as timer:
        results["test_risk"] = test_risk(model, risk_fun)
    results["test_risk_time"] = timer.elapsed

    # Perform cross-validation
    for k in config["cv_k"]:
        log(f"Performing {k}-fold cross-validation...")
        with Timer() as timer:
            results[f"cv_{k}_risk"] = cross_val_risk(
                model, X_train, y_train, risk_fun, k=k
            )
        results[f"cv_{k}_risk_time"] = timer.elapsed

    log("Precomputing ALO Jacobian...")
    device = torch.device(config["device"])
    with Timer() as timer:
        model.jac(device)
    results["jac_time"] = timer.elapsed

    # Perform ALO
    y_train_torch = torch.tensor(y_train, device=device)
    y_hat_torch = torch.tensor(model.predict(X_train), device=device)

    if config["alo_exact"]:
        log("Performing exact ALO...")
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

    if config.get("bks_seeds") is None:

        alo = None
        running_matvec_time = 0
        for m in sorted(config["alo_m"]):

            log(f"Performing randomized ALO up to {m} matvecs...")
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
    else:
        for bks_seed in config["bks_seeds"]:
            bks_gen = torch.Generator()
            bks_gen.manual_seed(bks_seed)
            alo = None
            running_matvec_time = 0
            for m in sorted(config["alo_m"]):
                log(f"Performing randomized ALO up to {m} matvecs...")
                with Timer() as timer:
                    alo = ALORandomized(
                        model.loss_fun,
                        y_train_torch,
                        y_hat_torch,
                        model.jac(device),
                        m,
                        generator=gen,
                    )
                running_matvec_time += timer.elapsed
                results[f"alo_{m}_{bks_seed}_matvec_time"] = running_matvec_time

                with Timer() as timer:
                    results[f"alo_{m}_{bks_seed}_bks_risk"] = (
                        alo.eval_risk(risk_fun, order=None) / n
                    )
                results[f"alo_{m}_{bks_seed}_bks_risk_time"] = timer.elapsed

                with Timer() as timer:
                    results[f"alo_{m}_{bks_seed}_poly_risk"] = (
                        alo.eval_risk(risk_fun, order=1) / n
                    )
                results[f"alo_{m}_{bks_seed}_poly_risk_time"] = timer.elapsed

    log("Performing GCV...")
    with Timer() as timer:
        X_centered_norm2s = torch.tensor(get_centered_norm2s(X_train), device=device)
        gcv = GCV(
            model.loss_fun,
            y_train_torch,
            y_hat_torch,
            model.jac(device),
            X_centered_norm2s,
            m=1,
            generator=gen,
        )
        results["gcv_risk"] = gcv.eval_risk(risk_fun) / n
    results["gcv_time"] = timer.elapsed

    if config["method"] == "random-forest":
        log("Computing OOB risk...")
        results["oob_risk"] = sum(risk_fun(y_train, model.oob_prediction())) / n

    # Save the results
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=4)
