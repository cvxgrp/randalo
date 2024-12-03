import adelie as ad
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pgenlib as pg

import sys
if len(sys.argv) != 2:
    raise RuntimeError()

data_dir = "/oak/stanford/groups/candes/for_parth"
cache_dir = "/scratch/candes/for_parth"
df = pd.read_csv(os.path.join(data_dir, "phenotypes.QC.britishonly.csv"), index_col=0)
covars_dense = df.loc[:, df.columns != 'height'].to_numpy()
y = df['height'].to_numpy()

chromosomes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

X = ad.matrix.concatenate(
        [ad.matrix.dense(covars_dense)] + 
        [
            ad.matrix.snp_unphased(
                ad.io.snp_unphased(
                    os.path.join(cache_dir, f"EUR_subset_chr{chr}.snpdat"),
                )
            )
            for chr in chromosomes],
        axis=1,
)
print(X.shape)

rng = np.random.default_rng(0xEE364A)
P = np.random.permutation(y.shape[-1])
n_train = P.size * 9 // 10
train_mask = P[:n_train]
test_mask = P[n_train:]
X_train = X[train_mask]
y_train = X[train_mask]
X_test = X[test_mask]
y_test = X[test_mask]


state = ad.grpnet(
    X=X_train,
    glm=ad.glm.gaussian(y_train),
    intercept=False,
)

import randalo as ra
import randalo.adelie_integration as ai
import torch

loss = torch.nn.MSELoss()
L = state.beta.shape[-1]
oos.np.empty(L)
ins.np.empty(L)
for i in range(L):
    oos[i] = loss(torch.from_numpy(X_test @ state.beta), torch.from_numpy(y_test))
    ins[i] = loss(torch.from_numpy(X_train @ state.beta), torch.from_numpy(y_train))

ld, alo = ai.get_alo_for_sweep(y, state, loss)

np.savez(sys.argv[-1], lamda=ld, alo=alo, oos=oos, in_sample=ins)
