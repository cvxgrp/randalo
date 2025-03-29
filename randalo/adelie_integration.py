import adelie as ad
from dataclasses import dataclass
import linops as lo
import numpy as np
import scipy.sparse as sp
import time
import torch
from tqdm import tqdm

import randalo as ra


class Numpy(lo.LinearOperator):
    supports_operator_matrix = True

    def __init__(self, X, adjoint=None):
        self.X = X
        self._shape = X.shape
        if adjoint is not None:
            self._adjoint = adjoint 
        else:
            self._adjoint = Numpy(X.T, adjoint=self)

    def _matmul_impl(self, v):
        return torch.from_numpy(self.X @ v.numpy())


class NumpyMemmap(lo.LinearOperator):
    supports_operator_matrix = True

    def __init__(self, file, shape):
        self.X = np.memmap(file, dtype=np.int8, mode='r', shape=shape)
        self._shape = shape
        self._adjoint = Numpy(self.X.T)

    def _matmul_impl(self, v):
        return torch.from_numpy(self.X @ v.numpy())

class AdelieOperator(lo.LinearOperator):
    supports_operator_matrix = True

    def __init__(self, X, XT, intercept=False, adjoint=None, shape=None, sparsity=None, adj_sparsity=None):
        if intercept:
            X = ad.matrix.concatenate([X, np.ones(X.shape[0])], axis=1, n_threads=32)
            XT = ad.matrix.concatenate([XT, np.ones((1, XT.shape[1]))], axis=0, n_threads=32)

        if shape is not None:
            self._shape = shape
        else:
            n, p = X.shape
            p = p if sparsity is None else sparsity.size
            adj_p, adj_n = XT.shape
            adj_n = adj_n if adj_sparsity is None else adj_sparsity.size
            assert adj_p == p
            assert adj_n == n
            self._shape = (n, p)

        self.X = X
        self.XT = XT
        self.sparsity = sparsity
        self._adjoint = adjoint if adjoint is not None else \
                AdelieOperator(XT, X, False, self, (p, n), sparsity=adj_sparsity)

    def _matmul_impl(self, v):
        v_dtype = v.dtype
        v = v.to(torch.float64)
        assert v.dtype == torch.float64
        if len(v.shape) == 1:
            ell = 1
            squeeze = True
        else:
            ell = v.shape[1]
            squeeze = False
        v_np = np.atleast_2d(v.numpy())
 
        if self.sparsity is None:
            out = self._matmul_impl_dense(v_np, ell)
        else:
            out = self._matmul_impl_sparse(v_np, ell)
        if squeeze:
           out = out.squeeze(-1)
        return torch.from_numpy(out).to(v_dtype)

    def _matmul_impl_sparse(self, v, ell):
        rowidx = np.tile(self.sparsity, ell)
        vals = v.ravel('F')
        k = self.shape[1]
        colptr = np.arange(0, k * ell, k)
        csr = sp.csr_matrix((vals, rowidx, colptr))
        out = np.empty(ell, self.shape[0])
        self.X.sp_tmul(csr, out)
        return out

    def _matmul_impl_dense(self, v, ell):
        print("Allocating ones...", flush=True)
        ones = np.ones(v.shape[0]).ravel()
        print("Allocating destination...", flush=True)
        out = np.empty((self.shape[0], ell), order='F')
        print("Starting multiply...", flush=True)
        t0 = time.monotonic()
        for i in range(ell):
            in_ptr = v[:, i].ravel()
            out_ptr = out[:, i]
            assert in_ptr.data.contiguous, "in_ptr should be ctg"
            assert out_ptr.data.contiguous, "out_ptr should be ctg"
            self.XT.mul(in_ptr, ones, out_ptr)
        tf = time.monotonic()
        print("Took...", tf - t0, "seconds", flush=True)
        return out

    def __getitem__(self, key):
        if isinstance(key, tuple):
            left_key, right_key = tuple(k.numpy() if isinstance(k, torch.Tensor) else k  for k in key)
            if isinstance(right_key, slice) or right_key.dtype == bool:
                right_key = np.arange(self.shape[1])[right_key]
            if left_key == slice(None):
                return AdelieOperator(self.X, self.XT[right_key], sparsity=right_key)

            return AdelieOperator(self.X[left_key], self.XT[right_key], sparsity=right_key, adj_sparsity=left_key)

        if isinstance(key, slice) or key.dtype == bool:
            right_key = np.arange(self.shape[1])[key]
        return AdelieOperator(self.X[key], self.XT, adj_sparsity=key)

_i = 0

class AdelieJacobian(lo.LinearOperator):
    supports_operator_matrix = False

    def __init__(self, X, XT, indices, intercept, dtype):

        if intercept:
            X = ad.matrix.concatenate([X, np.ones(X.shape[0], dtype=dtype)], axis=1, n_threads=32)
            XT = ad.matrix.concatenate([XT, np.ones(XT.shape[1], dtype=dtype)], axis=0, n_threads=32)
        n, p = X.shape
        self._shape = (n, n)
        self.X = X
        self.XT = XT
 
        self.indices = indices
        if np.size(indices) > 0:
            self.X_S = X[:, indices]
            self.XT_S = XT[indices, :]
            self._is_zero = False
        else:
            self._is_zero = True
        self._adjoint = self

    def _matmul_impl(self, v):
        global _i
        if self._is_zero:
            return torch.zeros_like(v)
        S = self.X_S.shape[-1]
        state = ad.grpnet(
                self.X_S,
                ad.glm.gaussian(v.numpy(), dtype=np.float64),
                #ad.glm.multigaussian(v.numpy(), dtype=np.float64),
                penalty=np.zeros(S),
                lmda_path=[0], progress_bar=False, n_threads=32, intercept=False)
        import pickle
        with open(f'/scratch/groups/candes/parth/benchmark{_i}.pkl', 'wb') as fd:
            pickle.dump({
                'fit_active': state.benchmark_fit_active, 
                'fit_screen': state.benchmark_fit_screen, 
                'invariance': state.benchmark_invariance, 
                'kkt': state.benchmark_kkt, 
                'screen': state.benchmark_screen,
            }, fd)
        _i += 1
        B = np.array(
            self.X_S @ state.betas.toarray()[0] #.reshape((S, -1), order='C')
            ,
            dtype=np.float32)
        return torch.from_numpy(B)


class AdelieState:
    def __init__(self, state):
        self.state = state
        self.ra_lmda = ra.HyperParameter()

    def set_index(self, idx):
        self.index = idx
        self.ra_lmda.value = self.state.lmda_path[idx]

def adelie_state_to_jacobian(y, y_hat, weights, state, adelie_state, X_trainT):
    n, p = state.X.shape
    G, = state.groups.shape
    L, = state.lmda_path.shape

    assert p == G, "Group lasso with adelie is not supported."

    if not state.intercept:
        ell_1_term = state.alpha * ra.L1Regularizer()
        ell_2_2_term = (1 - state.alpha) / 2 * ra.SquareRegularizer()
        reg = adelie_state.ra_lmda * (ell_1_term + ell_2_2_term)
    else:
        ell_1_term = state.alpha * ra.L1Regularizer(slice(None, -1))
        ell_2_2_term = (1 - state.alpha) / 2 * ra.SquareRegularizer(slice(None, -1))
        reg = adelie_state.ra_lmda * (ell_1_term + ell_2_2_term)

    loss = ra.MSELoss(weights)
    J = ra.Jacobian(
        y,
        AdelieOperator(state.X, state.X.T if X_trainT is None else X_trainT, state.intercept),
        lambda: sp.hstack((state.betas[adelie_state.index], sp.csr_matrix(np.array([[
            state.intercepts[adelie_state.index]
        ]])))),
        loss,
        reg,
        'minres',
        lambda: y_hat[adelie_state.index]
    )

    return loss, J

def adelie_state_to_randalo(y, y_hat, state, adelie_state, loss, J, index, rng=None):
    adelie_state.set_index(index)
    randalo = ra.RandALO(
            loss,
            J,
            y,
            y_hat,
            rng=rng)

    return randalo

def get_alo_for_sweep_v2(y, state, risk_fun, step=1, X_trainT=None):
    L, _ = state.betas.shape
    adelie_state = AdelieState(state)
    loss = ra.MSELoss()
    #loss, J = adelie_state_to_jacobian(y, state, adelie_state)
    y_hat = ad.diagnostic.predict(state.X, state.betas, state.intercepts)

    lmda = state.lmda_path[:L:step]
    output = np.empty_like(lmda)
    times = np.empty_like(lmda)
    r2 = np.empty_like(lmda)

    for out_i, i in tqdm(enumerate(range(0, L, step))):
        t0 = time.monotonic()
        indices = state.betas[i].indices

        J = AdelieJacobian(state.X, indices, state.intercept, y.dtype)
        randalo = adelie_state_to_randalo(y, y_hat[i], state, adelie_state, loss, J, i)
        output[out_i] = randalo.evaluate(risk_fun)
        times[out_i] = time.monotonic() - t0
        r2[out_i] = 1 - np.square(y - y_hat[i]).sum() / np.square(y - np.mean(y)).sum()

    return state.lmda_path[:L:step], output, times, r2

def get_alo_for_sweep(y, state, risk_fun, weights, step=1, X_trainT=None):
    L, _ = state.betas.shape
    adelie_state = AdelieState(state)
    y_hat = ad.diagnostic.predict(state.X, state.betas, state.intercepts)
    loss, J = adelie_state_to_jacobian(y, y_hat, weights, state, adelie_state, X_trainT)

    lmda = state.lmda_path[:L:step]
    output = np.empty_like(lmda)
    times = np.empty_like(lmda)
    r2 = np.empty_like(lmda)

    for out_i, i in tqdm(enumerate(range(0, L, step))):
        t0 = time.monotonic()
        randalo = adelie_state_to_randalo(y, y_hat[i], state, adelie_state, loss, J, i)
        output[out_i] = randalo.evaluate(risk_fun)
        times[out_i] = time.monotonic() - t0
        r2[out_i] = 1 - np.square(y - y_hat[i]).sum() / np.square(y - np.mean(y)).sum()
        print('R^2',  r2[out_i], flush=True)

    return state.lmda_path[:L:step], output, times, r2

