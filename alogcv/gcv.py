import torch

# torch.autograd.set_detect_anomaly(True)
from typing import Literal
from surecr.solver import Solver
import surecr.divergence as div_lib
import time


class GCV:
    def __init__(self, solver: Solver):
        self._solver = solver
        self._divergence_strategy: Literal[
            "default", "exact", "xtrace", "hutchinson", "hutch++"
        ] = "default"
        self._solution = None

    @property
    def solution(self):
        """
        Returns solver.solve(y) from the last compute call.
        """
        return self._solution

    def runtimes(self):
        """
        Returns how long it took for the solver to run and how long it took
        the divergence estimator to run during the last compute call.
        """
        return {
            "solver": self._t_f_solver - self._t_i_solver,
            "divergence": self._t_f_div - self._t_i_div,
        }

    def compute(self, data: torch.Tensor, divergence_parameters={}) -> torch.Tensor:
        """
        Computes and returns SURE for the estimator computed by the solver
        at the point y.

        Currently, divergence_parameters can contain the key "m" to indicate
        how many samples to use during the divergence estimation (which
        dominates the runtime at high dimensions). The default is for m to be
        102.

        In the future we may switch to A-Hutch++ and may change what options
        the divergence_parameters specifies.
        """

        shape = data.shape
        shape_ereased_data = data.reshape((-1,))
        shape_ereased_data.requires_grad_(True)

        def mu_hat(self, y):
            y = y.reshape(shape)
            self._solution = self._solver.solve(y)
            return self._solver.estimate(self._solution).reshape((-1,))

        dim_var_term = -data.numel() * self._variance

        self._t_i_solver = time.monotonic()
        mu_hat_evaled = mu_hat(self, shape_ereased_data)
        self._t_f_solver = time.monotonic()

        self._t_i_div = time.monotonic()
        divergence = div_lib.divergence(
            mu_hat_evaled,
            shape_ereased_data,
            self._divergence_strategy,
            divergence_parameters,
        )
        self._t_f_div = time.monotonic()
        self._divergence = divergence
        r = shape_ereased_data.detach() - mu_hat_evaled.deatch()

        return ((r / (1 - divergence / data.numel())) ** 2).sum()
