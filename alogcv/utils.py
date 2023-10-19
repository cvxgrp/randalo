import cvxpy as cp
import numpy as np


def robust_poly_fit(x, y, order: int):
    beta = cp.Variable(order + 1)
    r = y - np.vander(x, order + 1, True) @ beta

    prob = cp.Problem(cp.Minimize(cp.sum(cp.huber(r))))
    if not (prob.solve(cp.CLARABEL) < np.inf):
        return [np.nan for _ in range(order + 1)], np.inf
    return beta.value, np.linalg.norm(r.value)
