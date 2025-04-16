import adelie as ad
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pickle
import pandas as pd
import pgenlib as pg


import randalo as ra
import randalo.adelie_integration as ai
import torch

import sys
if len(sys.argv) == 1 or len(sys.argv) > 3:
    raise RuntimeError()
elif len(sys.argv) == 2:
    task_id = 0xEE364A
elif len(sys.argv) == 3:
    task_id = int(sys.argv[2])

data_dir = "/oak/stanford/groups/candes/for_parth"
cache_dir = "/scratch/groups/candes/parth"

chromosomes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

print('Data loading', flush=True)
y_train = np.load(os.path.join(cache_dir, "EUR_subset_y_train.npy"))
X_train = ad.matrix.concatenate(
        [
            ad.matrix.snp_unphased(
                ad.io.snp_unphased(
                    os.path.join(cache_dir, f"EUR_subset_chr{chr}_train.snpdat"), "mmap"
                ), n_threads=32, dtype=np.float64
            )
            for chr in chromosomes] +
        [ad.matrix.dense(
            np.asfortranarray(np.load(os.path.join(cache_dir, "EUR_subset_covars_train.npy"))
        , n_threads=32)], # Unpenalize covariates
        axis=1,
        n_threads=32
)
print(f'{X_train.shape=}', flush=True)
X_trainT = ad.matrix.concatenate(
        [
            ad.matrix.snp_unphased(
                ad.io.snp_unphased(
                    os.path.join(cache_dir, f"EUR_subset_chr{chr}T_train.snpdat"), "mmap"
                ), n_threads=32, dtype=np.float64
            )
            for chr in chromosomes] +
        [ad.matrix.dense(
            np.asfortranarray(np.load(os.path.join(cache_dir, "EUR_subset_covarsT_train.npy"))
        , n_threads=32)],
        axis=0,
        n_threads=32
)
print(f'{X_trainT.shape=}', flush=True)

y_test = np.load(os.path.join(cache_dir, "EUR_subset_y_test.npy"))
X_test = ad.matrix.concatenate(
        [
            ad.matrix.snp_unphased(
                ad.io.snp_unphased(
                    os.path.join(cache_dir, f"EUR_subset_chr{chr}_test.snpdat"), "mmap"
                ), n_threads=32, dtype=np.float64
            )
            for chr in chromosomes] +
        [ad.matrix.dense(
            np.asfortranarray(np.load(os.path.join(cache_dir, "EUR_subset_covarsT_test.npy"))
        , n_threads=32)],
        axis=1,
        n_threads=32
)
print(f'{X_test.shape=}', flush=True)

penalty = np.ones(X_train.shape[1])
penalty[-10:] = 0.0

model_cache = f'/scratch/groups/candes/parth/fit_model_{task_id}_v10.pkl'

weights = np.ones_like(y_train)
if os.path.exists(model_cache):
    class fake_state:
        def __init__(self):
            with open(model_cache, 'rb') as fd:
                d = pickle.load(fd)
            self.betas = d['betas']
            self.lmda_path = d['lmda_path']
            self.intercepts = d['intercepts']
            self.penalty = penalty
            self.intercept = True
            self.X = X_train
            self.groups = np.arange(self.X.shape[1])
            self.alpha = 1.0
    state = fake_state()
    ti_solve = 0
    tf_solve = 0
else:
    ti_solve = time.monotonic()
    state = ad.grpnet(
        X=X_train,
        glm=ad.glm.gaussian(y_train, dtype=np.float64, weights=weights),
        early_exit=False,
        penalty=penalty,
        min_ratio=1e-3,
        n_threads=32,
        lmda_path_size=101,
    )
    tf_solve = time.monotonic()

    with open(model_cache, 'wb') as fd:
        pickle.dump({'betas': state.betas, 'lmda_path': state.lmda_path, 'intercepts': state.intercepts}, fd)

train_risk = lambda x, y: torch.mean((x - y)**2)
loss = torch.nn.MSELoss()
L = state.betas.shape[0]
oos = np.empty(L)
ins = np.empty(L)
size = np.empty(L)
print('Test/Train predict', flush=True)
y_hat_test = ad.diagnostic.predict(X_test, state.betas, state.intercepts)
y_hat_train = ad.diagnostic.predict(X_train, state.betas, state.intercepts)
for i in range(L):
    oos[i] = loss(torch.from_numpy(y_hat_test[i]), torch.from_numpy(y_test))
    ins[i] = loss(torch.from_numpy(y_hat_train[i]), torch.from_numpy(y_train))
    size[i] = state.betas[i].nnz
print('Final oos loss:', oos[-1])
print('Final ins loss:', ins[-1])
print('Final size:', size[-1], flush=True)

ti_alo = time.monotonic()
ld, alo, ts, r2 = ai.get_alo_for_sweep(y_train, state,  train_risk, torch.from_numpy(weights), 1, X_trainT=X_trainT)
tf_alo = time.monotonic()

np.savez(sys.argv[1], alo_lamda=ld, full_lamda=state.lmda_path, alo=alo, oos=oos, in_sample=ins, ts=ts, r2=r2, solve_time=tf_solve - ti_solve, alo_time=tf_alo - ti_alo, size=size)
