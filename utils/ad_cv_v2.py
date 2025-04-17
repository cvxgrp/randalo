import adelie as ad
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pickle
import pandas as pd

import sys
if len(sys.argv) == 1 or len(sys.argv) > 2:
    raise RuntimeError()

data_dir = "/oak/stanford/groups/candes/for_parth"
cache_dir = "/scratch/groups/candes/parth"
df = pd.read_csv(os.path.join(data_dir, "phenotypes.QC.britishonly.csv"), index_col=0)
df = df.drop('ethnicity', axis=1)
covars_dense = np.array(
    df[['age', 'age_squared', 'sex'] + [f'PC{i}' for i in range(1, 11)]].to_numpy(),
    dtype=np.float64, order='F')

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


penalty = np.ones(X_train.shape[1])
penalty[-10:] = 0.0

folds = 5
rng = np.random.default_rng(0x219a)
order = rng.permutation(y_train.shape[-1])
step = y_train.shape[-1] // folds

lmdas = np.array([0.31735175, 0.30306856, 0.28942823, 0.27640182, 0.26396168,
       0.25208145, 0.24073592, 0.22990102, 0.21955377, 0.20967222,
       0.20023541, 0.19122334, 0.18261687, 0.17439776, 0.16654856,
       0.15905264, 0.1518941 , 0.14505773, 0.13852906, 0.13229423,
       0.12634001, 0.12065377, 0.11522345, 0.11003754, 0.10508504,
       0.10035543, 0.0958387 , 0.09152524, 0.08740593, 0.08347201,
       0.07971515, 0.07612738, 0.07270108, 0.069429  , 0.06630418,
       0.06332   , 0.06047013, 0.05774852, 0.05514941, 0.05266728,
       0.05029686, 0.04803313, 0.04587128, 0.04380674, 0.04183511,
       0.03995222, 0.03815407, 0.03643686, 0.03479693, 0.03323081,
       0.03173517, 0.03030686, 0.02894282, 0.02764018, 0.02639617,
       0.02520815, 0.02407359, 0.0229901 , 0.02195538, 0.02096722,
       0.02002354, 0.01912233, 0.01826169, 0.01743978, 0.01665486,
       0.01590526, 0.01518941, 0.01450577, 0.01385291, 0.01322942,
       0.012634  , 0.01206538, 0.01152235, 0.01100375, 0.0105085 ,
       0.01003554, 0.00958387, 0.00915252, 0.00874059, 0.0083472 ,
       0.00797152, 0.00761274, 0.00727011, 0.0069429 , 0.00663042,
       0.006332  , 0.00604701, 0.00577485, 0.00551494, 0.00526673,
       0.00502969, 0.00480331, 0.00458713, 0.00438067, 0.00418351,
       0.00399522, 0.00381541, 0.00364369, 0.00347969, 0.00332308,
       0.00317352])

cv_risk = np.zeros((folds, lmdas.size))

ti_cv = time.monotonic()
for i in range(folds):
    weights = np.ones_like(y_train)
    weights[order[step * i: step * (i+1)]] = 0.0

    state = ad.grpnet(
        X=X_train,
        glm=ad.glm.gaussian(y_train, dtype=np.float64, weights=weights),
        early_exit=False,
        n_threads=32,
        penalty=penalty,
        lmda_path=lmdas,
    )
    predicts = ad.diagnostic.predict(X_train, state.betas, state.intercepts, n_threads=32)
    risks = (y_train[None, :] - predicts)**2
    cv_risk[i] = np.sum((1 - weights) * risks, axis=1)

avg_cv_risk = np.sum(cv_risk, axis=0) / y_train.size
tf_cv = time.monotonic()

np.savez(sys.argv[1], cv_lamda=lmdas, cv=cv_risk, avg_cv=avg_cv_risk, cv_time=tf_cv - ti_cv)
