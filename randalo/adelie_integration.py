import adelie
import randalo as ra
import numpy as np
from dataclasses import dataclass

def curry(f, *args0, **kwargs0):
    return lambda *args, **kwargs: f(*args0, *args, **kwargs0, **kwargs)

class AdelieState:
    def __init__(self, state):
        self.state = state
        self.ra_lmda = ra.HyperParameter()

    def set_index(self, idx):
        self.index = idx
        self.ra_lmda.value = self.state.lmda[idx]


def adelie_state_to_Jacobian(state, adelie_state):
    n, p = state.X.shape
    G, = state.groups.shape
    L, = state.lmda_path.shape

    assert p == G, "Group lasso with adelie is not supported."
    assert state.penalty == None

    assert state.offsets == None
    assert state.intercept == None
    ell_1_term = state.alpha * ra.L1Regularizer()
    ell_2_2_term = (1 - state.alpha) / 2 * ra.SquareRegularizer()
    reg = adelie_state.ra_lmda * (ell_1_term + ell_2_2_term)

    loss = ra.MSELoss()

    J = ra.Jacobian(lambda: (
        betas[adelie_state.index], # What is the type of this?
        screen_set[active_set[active_sizes[:adelie_state.index]]]),
        loss,
        reg,
    )

    return loss, J

def adelie_state_to_randalo(state, adelie_state, loss, J, index, rng):
    y_hat = state.X @ state.beta[index]
    adelie_state.set_index(index)
    randalo = ra.RandALO(
            loss,
            J,
            state.y,
            y_hat,
            rng=rng)

    return randalo

def get_alo_for_sweep(state, risk_fun):
    L, = state.lmda.shape
    adelie_state = AdelieState(state)
    loss, J = adelie_state_to_jacobian(state, adelie_state)

    output = np.empty(L)

    for i in range(L):
        randalo = adelie_state_to_randalo(state, adelie_state, loss, J, i)
        output[i] = randalo.evaluate(risk_fun)

    return output

