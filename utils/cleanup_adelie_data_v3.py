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

df = pd.read_csv(os.path.join(data_dir, "phenotypes.QC.britishonly.csv"), index_col=0)
df = df.drop('ethnicity', axis=1)
covars_dense = np.array(
    df[[f'PC{i}' for i in range(1, 11)]].to_numpy(),
dtype=np.float64, order='F')

total_mask = (df['age'] >= 20) & (df['age'] < 61) & (df['sex'] == 1)

rng = np.random.default_rng(0x364a)

train_mask = total_mask & (rng.random(total_mask.shape) < 0.9)
test_mask = ~train_mask

covars_name_test = os.path.join(cache_dir, f"EUR_subset_covars_test.npy")
covars_name_train = os.path.join(cache_dir, f"EUR_subset_covars_train.npy")
covars_name_trainT = os.path.join(cache_dir, f"EUR_subset_covarsT_train.npy")
np.save(covars_name_test, covars_dense[:, ~train_mask])
np.save(covars_name_train, covars_dense[:, train_mask])
np.save(covars_name_trainT, covars_dense[:, train_mask].T)

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
    _ = handler.write(np.asfortranarray(geno_out_chr[~train_mask]))
    
    handler = ad.io.snp_unphased(snpdat_name_train)
    _ = handler.write(np.asfortranarray(geno_out_chr[train_mask]))
    
    handler = ad.io.snp_unphased(snpdat_name_trainT)
    _ = handler.write(np.asfortranarray(geno_out_chr[train_mask].T))


