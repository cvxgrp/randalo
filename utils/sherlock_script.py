import adelie as ad
import matplotlib.pyplot as plt
import numpy as np
import time
import os
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
df = pd.read_csv(os.path.join(data_dir, "phenotypes.QC.britishonly.csv"), index_col=0)
df = df.drop('ethnicity', axis=1)
covars_dense = df[['age', 'sex'] + [f'PC{i}' for i in range(1, 11)]]
y = df['height'].to_numpy()

chromosomes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

print('Data loading')
X = ad.matrix.concatenate(
        [ad.matrix.dense(covars_dense, n_threads=32)] + 
        [
            ad.matrix.snp_unphased(
                ad.io.snp_unphased(
                    os.path.join(cache_dir, f"EUR_subset_chr{chr}.snpdat"),
                ), n_threads=32
            )
            for chr in chromosomes],
        axis=1,
)
print(f'{X.shape=}')

rng = np.random.default_rng(task_id)
P = np.random.permutation(y.shape[-1])
n_train = P.size * 9 // 10
train_mask = P[:n_train]
test_mask = P[n_train:]
X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]
print(f'{X_train.shape=}')
print(f'{X_test.shape=}')

ti_solve = time.monotonic()
state = ad.grpnet(
    X=X_train,
    glm=ad.glm.gaussian(y_train),
    early_exit=False,
    min_ratio=1e-6,
    n_threads=32,
)
tf_solve = time.monotonic()


loss = torch.nn.MSELoss()
L = state.betas.shape[0]
oos = np.empty(L)
ins = np.empty(L)
y_hat_test = ad.diagnostic.predict(X_test, state.betas, state.intercepts)
y_hat_train = ad.diagnostic.predict(X_train, state.betas, state.intercepts)
for i in range(L):
    oos[i] = loss(torch.from_numpy(y_hat_test[i]), torch.from_numpy(y_test))
    ins[i] = loss(torch.from_numpy(y_hat_train[i]), torch.from_numpy(y_train))

ti_alo = time.monotonic()
ld, alo, ts, r2 = ai.get_alo_for_sweep(y_train, state, loss, 10)
tf_alo = time.monotonic()

np.savez(sys.argv[1], lamda=ld, alo=alo, oos=oos, in_sample=ins, ts=ts, r2=r2, solve_time=tf_solve - ti_solve, alo_time=tf_alo - ti_alo)
