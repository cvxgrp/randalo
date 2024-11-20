import adelie
from dataclasses import dataclass
import linops as lo
import numpy as np
import torch

import randalo as ra

class AdelieOperator(lo.LinearOperator):
    supports_operator_matrix = True

    def __init__(self, X, adjoint=None, shape=None):
        if shape is not None:
            self._shape = shape
        else:
            m, n = X.shape
            self._shape = (m, n)
        self.X = X
        self._adjoint = adjoint if adjoint is not None else AdelieOperator(X.T, self, (n, m))

    def _matmul_impl(self, v):
        return torch.from_numpy(self.X @ v.numpy())

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = tuple(k.numpy() if isinstance(k, torch.Tensor) else k  for k in key)
        return AdelieOperator(self.X[key])

def curry(f, *args0, **kwargs0):
    return lambda *args, **kwargs: f(*args0, *args, **kwargs0, **kwargs)

class AdelieState:
    def __init__(self, state):
        self.state = state
        self.ra_lmda = ra.HyperParameter()

    def set_index(self, idx):
        self.index = idx
        self.ra_lmda.value = self.state.lmda_path[idx]


def adelie_state_to_jacobian(y, state, adelie_state):
    n, p = state.X.shape
    G, = state.groups.shape
    L, = state.lmda_path.shape

    assert p == G, "Group lasso with adelie is not supported."

    assert not state.intercept
    ell_1_term = state.alpha * ra.L1Regularizer()
    ell_2_2_term = (1 - state.alpha) / 2 * ra.SquareRegularizer()
    reg = adelie_state.ra_lmda * (ell_1_term + ell_2_2_term)

    loss = ra.MSELoss()
    J = ra.Jacobian(
        y,
        AdelieOperator(state.X),
        lambda: state.betas[adelie_state.index],
        loss,
        reg,
        'minres'
    )

    return loss, J

def adelie_state_to_randalo(y, state, adelie_state, loss, J, index, rng=None):
    y_hat = (state.X @ state.betas[index].T).squeeze()
    adelie_state.set_index(index)
    randalo = ra.RandALO(
            loss,
            J,
            y,
            y_hat,
            rng=rng)

    return randalo

def get_alo_for_sweep(y, state, risk_fun):
    L, _ = state.betas.shape
    adelie_state = AdelieState(state)
    loss, J = adelie_state_to_jacobian(y, state, adelie_state)

    output = np.empty(L)

    for i in range(L):
        randalo = adelie_state_to_randalo(y, state, adelie_state, loss, J, i)
        output[i] = randalo.evaluate(risk_fun)

    return state.lmda_path[:L], output

