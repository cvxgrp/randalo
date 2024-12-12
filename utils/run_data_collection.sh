#!/bin/bash
#SBATCH --job-name=adelie_data
#SBATCH --output=adelie_data/output/slurm-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=16GB
#SBATCH --partition=candes
#SBATCH --array=1-50

BASE_DIR=$HOME/adelie_alo/benchmarking/run
RESULTS_DIR=$BASE_DIR/results
mkdir -p $RESULTS_DIR
DEST_FILE=$RESULTS_DIR/sweep-$SLURM_ARRAY_TASK_ID.npz


ml python/3.12.1
ml py-pytorch/2.4.1_py312
. $HOME/randalo/.venv/bin/activate


python $HOME/randalo/utils/sherlock_script.py $DEST_FILE $SLURM_ARRAY_TASK_ID
