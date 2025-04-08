import adelie as ad
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pickle
import pandas as pd
import pgenlib as pg

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
covars_dense = np.array(
    df[['age', 'age_squared', 'sex'] + [f'PC{i}' for i in range(1, 11)]].to_numpy(),
    dtype=np.float64, order='F')
y = np.array(df['height'].to_numpy(), dtype=np.float64)

chromosomes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

rng = np.random.default_rng(0x364a)
P = rng.permutation(y.shape[-1])
n_train = P.size * 9 // 10
train_mask = np.ones(y.shape[-1], dtype=bool)
train_mask[P[n_train:]] = False

covars_dense_train = np.asfortranarray(covars_dense[train_mask])
y_train = y[train_mask]
covars_dense_test = np.asfortranarray(covars_dense[~train_mask])
y_test = y[~train_mask]


print('Data loading', flush=True)
X_train = ad.matrix.concatenate(
        [ad.matrix.dense(covars_dense_train, n_threads=32)] + 
        [
            ad.matrix.snp_unphased(
                ad.io.snp_unphased(
                    os.path.join(cache_dir, f"EUR_subset_chr{chr}_train.snpdat"), "mmap"
                ), n_threads=32, dtype=np.float64
            )
            for chr in chromosomes],
        axis=1,
        n_threads=32
)
print(f'{X_train.shape=}', flush=True)

X_test = ad.matrix.concatenate(
        [ad.matrix.dense(covars_dense_test, n_threads=32)] + 
        [
            ad.matrix.snp_unphased(
                ad.io.snp_unphased(
                    os.path.join(cache_dir, f"EUR_subset_chr{chr}_test.snpdat"), "mmap"
                ), n_threads=32, dtype=np.float64
            )
            for chr in chromosomes],
        axis=1,
        n_threads=32
)
print(f'{X_test.shape=}', flush=True)

folds = 5
rng = np.random.default_rng(0x219a)
order = rng.permutation(y_train.shape[-1])
step = y_train.shape[-1] // folds

lmdas = np.logspace(ell := 2.95916, ell - 6)

cv_risk = np.zeros(folds, 101)

ti_cv = time.monotonic()
for i in range(folds):
    weights = np.ones_like(y_train)
    weights[order[step * i: step * (i+1)]] = 0.0

    state = ad.grpnet(
        X=X_train,
        glm=ad.glm.gaussian(y_train, dtype=np.float64, weights=weights),
        early_exit=False,
        n_threads=32,
        lmda_path=lmdas,
        seed=0xEE219A,
    )
    predicts = ad.diagnostic.predict(X_train, state.betas, state.intercepts, n_threads=32)
    risks = (y_train[None, :] - predicts)**2
    cv_risk[i] = np.sum((1 - weights) * risks)

avg_cv_risk = np.sum(cv_risk, axis=0)
tf_cv = time.monotonic()

np.savez(sys.argv[1], cv_lamda=lmdas, cv=cv_risk, avg_cv=avg_cv_risk, cv_time=tf_cv - ti_cv)
