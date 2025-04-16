#!/bin/bash
#SBATCH --job-name=adelie_sweep
#SBATCH --output=adelie_sweep/output/slurm-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1024GB
#SBATCH --partition=candes,pilanci

BASE_DIR=$HOME/adelie_alo/benchmarking/lasso_sweep
RESULTS_DIR=$BASE_DIR/results
mkdir -p $RESULTS_DIR
DEST_FILE=$RESULTS_DIR/sweep.npz


ml python/3.12.1
ml py-pytorch/2.4.1_py312
. $HOME/randalo/.venv/bin/activate


python $HOME/randalo/utils/sherlock_script_v2.py $DEST_FILE
