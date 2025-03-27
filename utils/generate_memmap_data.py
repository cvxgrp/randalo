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

# get 0-indexed indices for current chromosome
df_bim_chr = df_bim[df_bim["chr"] == 17]
variant_idxs = df_bim_chr.index.to_numpy().astype(np.uint32)

# read the SNP matrix
geno_out_chr = np.empty((variant_idxs.shape[0], n_samples), dtype=np.int8)
reader.read_list(variant_idxs, geno_out_chr)

# convert to sample-major
geno_out_chr = np.asfortranarray(geno_out_chr.T)

# define cache directory and snpdat filename
cache_dir = "/scratch/groups/candes/parth"

chromosomes = df_bim["chr"].unique()

# create bed reader
reader = pg.PgenReader(
            str.encode(bedname),
                raw_sample_ct=n_samples,
                )

total_size = 0
for chr in chromosomes:
    # get 0-indexed indices for current chromosome
    df_bim_chr = df_bim[df_bim["chr"] == chr]
    variant_idxs = df_bim_chr.index.to_numpy().astype(np.uint32)


    # read the SNP matrix
    geno_out = np.empty((variant_idxs.shape[0], n_samples), dtype=np.int8)
    reader.read_list(variant_idxs, geno_out)

    total_size += variant_idxs.shape[0]

    memmap_name = os.path.join(cache_dir, f'EUR_subset_chr{chr}.array')
    array = np.memmap(memmap_name, dtype=np.int8, mode='w+', shape=(variant_idxs.shape[0], n_samples))
    array[:] = geno_out

memmap_name = os.path.join(cache_dir, f'EUR_subset.array')
big_array = np.memmap(memmap_name, dtype=np.int8, mode='w+', shape=(total_size, n_samples))
i = 0
for chr in chromosomes:
    memmap_name = os.path.join(cache_dir, f'EUR_subset_chr{chr}.array')
    df_bim_chr = df_bim[df_bim["chr"] == chr]
    variant_idxs = df_bim_chr.index.to_numpy().astype(np.uint32)
    array = np.memmap(memmap_name, dtype=np.int8, mode='r', shape=(variant_idxs.shape[0], n_samples))
    big_array[i:(i := i + variant_idxs.shape[0])] = array

print(total_size, n_samples)
