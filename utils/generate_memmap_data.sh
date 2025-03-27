#!/bin/bash
#SBATCH --job-name=memmap_gen
#SBATCH --output=memmap_gen/output/slurm-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64GB
#SBATCH --partition=candes

ml python/3.12.1
ml py-pytorch/2.4.1_py312
. $HOME/randalo/.venv/bin/activate


python $HOME/randalo/utils/generate_memmap_data.py
