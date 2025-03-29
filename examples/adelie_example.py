import adelie as ad
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pgenlib as pg

data_dir = "../../adelie/data"
cache_dir = "/tmp/"
df = pd.read_csv(os.path.join(data_dir, "master_phe.csv"), sep="\t", index_col=0)
covars_dense = df.iloc[:, :-1].to_numpy()
y = np.array(df.iloc[:, -1].to_numpy(), dtype=np.float64)

chromosomes = [17, 18, 19, 20, 21, 22]

X = ad.matrix.concatenate(
        [
            ad.matrix.snp_unphased(
                ad.io.snp_unphased(
                    os.path.join(cache_dir, f"EUR_subset_chr{chr}.snpdat"),
                ), dtype=np.float64
            )
            for chr in chromosomes],
        axis=1,
)
print(X.shape)



state = ad.grpnet(
    X=X,
    glm=ad.glm.gaussian(y, dtype=np.float64),
)

import randalo as ra
import randalo.adelie_integration as ai
import torch

test = ad.diagnostic.predict(X, state.betas, state.intercepts)
ld, alo, ts, r2 = ai.get_alo_for_sweep(y, state, torch.nn.MSELoss(), 5)
print(alo, r2)
