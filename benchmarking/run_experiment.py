import csv
import uuid
import time
from statistics import mean, stdev
from pathlib import Path

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
import torch
import linops as lo

import alogcv
from alogcv.alo import ALORandomized

SEED = 0


def square_loss(y, y_hat):
    return (y - y_hat) ** 2 / 2


def square_risk(y, y_hat):
    return (y - y_hat) ** 2


def load_problem(problem):
    p = np.load(problem)
    X = torch.Tensor(p["X"])
    y = torch.Tensor(p["y"])
    n = p["n"][()]
    p = p["p"][()]
    return X, y, n, p


def sklearn(estimator, estimator_params):
    if estimator == "lasso":
        (lamda,) = estimator_params
        lamda = float(lamda)
        return Lasso(alpha=lamda, fit_intercept=False)


class CholeskyOperator(lo.LinearOperator):
    supports_operator_matrix = True

    def __init__(self, obj_hessian, X):
        n = X.shape[0]
        self._shape = (n, n)
        self.device = obj_hessian.device
        self._L = torch.linalg.cholesky(obj_hessian)
        self._X = X
        self._adjoint = self

    def _matmul_impl(self, u: torch.Tensor) -> torch.Tensor:
        v = self._X.T @ u
        w = torch.linalg.solve_triangular(self._L, v, upper=False)
        x = torch.linalg.solve_triangular(self._L.T, w, upper=True)
        return self._X @ x


def cholesky(X, y, estimator, estimator_params, k, data_dest):
    ti = time.monotonic()
    sk_estimator = sklearn(estimator, estimator_params)
    sk_estimator.fit(X.numpy(), y.numpy())
    beta_hat = torch.Tensor(sk_estimator.coef_)
    y_hat = X @ beta_hat
    n = X.shape[0]

    if estimator == "lasso":
        mask = torch.abs(beta_hat) > 1e-8
        X_lasso = X[:, mask]
        obj_hess = X_lasso.T @ X_lasso
        J = CholeskyOperator(obj_hess, X_lasso)
        alo = ALORandomized(square_loss, y, y_hat, J, k)
        risk = alo.eval_risk(square_risk, order=1) / n
    tf = time.monotonic()

    np.savez(
        data_dest,
        beta_hat=sk_estimator.coef_,
        time=tf - ti,
        risk=risk,
    )
    return tf - ti, risk


def unrolling(X, y, estimator, estimator_params, k, data_dest):
    ti = time.monotonic()
    if estimator == "lasso":
        (lamda,) = estimator_params
        beta_hat, J, solver_time, iters = alogcv.utils.lasso(X, y, lamda)
        y_hat = X @ beta_hat
        n = X.shape[0]
        alo = ALORandomized(square_loss, y, y_hat, J, k)
        risk = alo.eval_risk(square_risk, order=1) / n
    tf = time.monotonic()

    np.savez(
        data_dest,
        beta_hat=estimator.coef_,
        solver_iters=iters,
        solver_time=solver_time,
        time=tf - ti,
        risk=risk,
    )

    return tf - ti, risk


def cv(X, y, estimator, estimator_params, k, data_dest):
    ti = time.monotonic()
    sk_estimator = sklearn(estimator, estimator_params)
    sk_estimator.fit(X, y)
    scores = -cross_val_score(
        sk_estimator, X.numpy(), y.numpy(), cv=k, scoring="neg_mean_squared_error"
    )
    risk = scores.mean()
    tf = time.monotonic()

    np.savez(
        data_dest,
        beta_hat=sk_estimator.coef_,
        cv_scores=scores,
        time=tf - ti,
        risk=risk,
    )

    return tf - ti, risk


def experiment(
    problem, estimator, estimator_params, method, trials, device, data_dest_path
):
    if estimator == "lasso":
        (lamda,) = estimator_params
    else:
        raise RuntimeError("only lasso is supported")

    ts = []
    risks = []

    for i in range(trials):
        data_dest = data_dest_path.with_suffix(f".{i}.npz")
        gen = torch.manual_seed(SEED)
        X, y, n, p = load_problem(problem)
        method_split = method.split("-")
        if len(method_split) == 1:
            (method,) = method_split
            k = 1
        else:
            method, k = method_split
            k = int(k)
        if method == "exact":
            pass
        elif method == "cholesky":
            tdur, risk = cholesky(X, y, estimator, estimator_params, k, data_dest)
        elif method == "blockcg":
            pass
        elif method == "unrolling":
            tdur, risk = cholesky(X, y, estimator, estimator_params, k, data_dest)
        elif method == "cv":
            tdur, risk = cv(X, y, estimator, estimator_params, k, data_dest)
        ts.append(tdur)
        risks.append(risk)

    # return mean(ts), stdev(ts), mean(risks), stdev(risks)
    return mean(ts), mean(risks)


def main(argv):
    outdir = Path(f'./{time.strftime("%Y-%m-%d-%H_%M_%S")}')
    outdir.mkdir()

    problem_list_file = argv[1]
    methods = [s.split(":") for s in argv[2].split("|")]
    device_num = argv[3]
    if device_num == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device_num}")

    outfile = outdir / "results.csv"

    with outfile.open(mode="w") as out:
        outcsv = csv.writer(out)
        for m in methods:
            if len(m) == 1:
                method = m[0]
                trials = 1
            else:
                method, trials = m
                trials = int(trials)
            with open(problem_list_file, newline="") as fd:
                for row in csv.reader(fd):
                    problem, esitmator, *esitmator_params = row
                    data_dest_path = outdir / Path(str(uuid.uuid4()))
                    output = experiment(
                        problem,
                        esitmator,
                        esitmator_params,
                        method,
                        trials,
                        device,
                        data_dest_path,
                    )
                    outrow = [*row, method, str(data_dest_path), *output]
                    outcsv.writerow(outrow)


if __name__ == "__main__":
    import sys

    main(sys.argv)
