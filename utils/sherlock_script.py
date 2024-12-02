import adelie as ad
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pgenlib as pg

data_dir = "/oak/group/candes/for_parth"
cache_dir = "/scratch//candes/for_parth"
df = pd.read_csv(os.path.join(data_dir, "master_phe.csv"), sep="\t", index_col=0)
covars_dense = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].to_numpy()

chromosomes = [17, 18, 19, 20, 21, 22]

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



state = ad.grpnet(
    X=X,
    glm=ad.glm.gaussian(y),
    intercept=False,
)

import randalo as ra
import randalo.adelie_integration as ai
import torch

ld, alo = ai.get_alo_for_sweep(y, state, torch.nn.MSELoss())

