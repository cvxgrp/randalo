#!/bin/bash
#SBATCH --job-name=lasso_sweep
#SBATCH --output=lasso_sweep/output/slurm-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --partition=candes,normal,hns
#SBATCH --array=1-500

BASE_DIR=$HOME/alo/benchmarking/lasso_sweep
CONFIGS_DIR=$BASE_DIR/configs
RESULTS_DIR=$BASE_DIR/results
mkdir -p $RESULTS_DIR

FILES=($CONFIGS_DIR/run_*.json)

FILE=${FILES[$SLURM_ARRAY_TASK_ID-1]}
DEST_FILE=$RESULTS_DIR/$(basename $FILE)

ml python/3.9.0
ml py-pytorch/2.0.0_py39
. $HOME/alo/.venv/bin/activate

python run.py $FILE $DEST_FILE
