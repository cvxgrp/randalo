import adelie as ad
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pgenlib as pg

data_dir = "/oak/stanford/groups/candes/for_parth"
bedname = os.path.join(data_dir, "ukb_gen_british.bed")
bimname = os.path.join(data_dir, "ukb_gen_british.bim")
famname = os.path.join(data_dir, "ukb_gen_british.fam")

df_fam = pd.read_csv(
    famname,
    sep=" ",
    header=None,
    names=["FID", "IID", "Father", "Mother", "Sex", "Phenotype"],
)
n_samples = df_fam.shape[0]

df_bim = pd.read_csv(
    bimname,
    sep="\t",
    header=None,
    names=["chr", "variant", "pos", "base", "a1", "a2"],
)
n_snps = df_bim.shape[0]

# create bed reader
reader = pg.PgenReader(
    str.encode(bedname),
    raw_sample_ct=n_samples,
)


# define cache directory and snpdat filename
cache_dir = "/scratch/groups/candes/parth"

chromosomes = df_bim["chr"].unique()

# create bed reader
reader = pg.PgenReader(
            str.encode(bedname),
                raw_sample_ct=n_samples,
                )

rng = np.random.default_rng(task_id)
P = np.random.permutation(y.shape[-1])
n_train = P.size * 9 // 10
train_mask = np.ones(n_samples, dtype=bool)
train_mask[P[n_train:]] = False

for chr in chromosomes:
    # get 0-indexed indices for current chromosome
    df_bim_chr = df_bim[df_bim["chr"] == chr]
    variant_idxs = df_bim_chr.index.to_numpy().astype(np.uint32)

    # read the SNP matrix
    geno_out = np.empty((variant_idxs.shape[0], n_samples), dtype=np.int8)
    reader.read_list(variant_idxs, geno_out)

    # convert to sample-major
    geno_out_chr = np.asfortranarray(geno_out.T)

    # define snpdat filename
    snpdat_name_test = os.path.join(cache_dir, f"EUR_subset_chr{chr}_test.snpdat")
    snpdat_name_train = os.path.join(cache_dir, f"EUR_subset_chr{chr}_train.snpdat")
    snpdat_name_trainT = os.path.join(cache_dir, f"EUR_subset_chr{chr}T_train.snpdat")

    # create handler to convert the SNP matrix to .snpdat
    handler = ad.io.snp_unphased(snpdat_name_test)
    _ = handler.write(geno_out_chr[~train_mask])
    
    handler = ad.io.snp_unphased(snpdat_name_train)
    _ = handler.write(geno_out_chr[train_mask])
    
    handler = ad.io.snp_unphased(snpdat_name_trainT)
    _ = handler.write(np.asfortranarray(geno_out_chr[train_mask].T))
