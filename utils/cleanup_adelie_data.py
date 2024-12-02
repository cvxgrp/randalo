import adelie as ad
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pgenlib as pg

data_dir = "/oak/group/candes/for_parth"
bedname = os.path.join(data_dir, "EUR_subset.bed")
bimname = os.path.join(data_dir, "EUR_subset.bim")
famname = os.path.join(data_dir, "EUR_subset.fam")

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

# get 0-indexed indices for current chromosome
df_bim_chr = df_bim[df_bim["chr"] == 17]
variant_idxs = df_bim_chr.index.to_numpy().astype(np.uint32)

# read the SNP matrix
geno_out_chr = np.empty((variant_idxs.shape[0], n_samples), dtype=np.int8)
reader.read_list(variant_idxs, geno_out_chr)

# convert to sample-major
geno_out_chr = np.asfortranarray(geno_out_chr.T)

# define cache directory and snpdat filename
cache_dir = "/scratch/candes/for_parth/tmp"

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

    # define snpdat filename
    snpdat_name = os.path.join(cache_dir, f"EUR_subset_chr{chr}.snpdat")

    # create handler to convert the SNP matrix to .snpdat
    handler = ad.io.snp_unphased(snpdat_name)
    _ = handler.write(geno_out_chr)
