#!/bin/bash
#SBATCH --job-name=lasso_scaling_1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-20%10  # Adjust 1-20 to match the number of .json files, %10 limits to 10 concurrent jobs

# Assuming all your .json files are in a directory called "configs"
BASE_DIR=$HOME/alo/benchmarking/lasso_scaling_1
CONFIGS_DIR=$BASE_DIR/configs
RESULTS_DIR=$BASE_DIR/results
mkdir -p $RESULTS_DIR

FILES=($CONFIGS_DIR/run_*.json)

FILE=${FILES[$SLURM_ARRAY_TASK_ID-1]}
DEST_FILE=$RESULTS_DIR/$(basename $FILE)

python run.py $FILE $DEST_FILE

# for ((i=1; i<=50; i++))
# do
#   SLURM_ARRAY_TASK_ID=$i
#    # SLURM_ARRAY_TASK_ID is the variable that gets a unique value for each job in the array
#    FILE=${FILES[$SLURM_ARRAY_TASK_ID-1]}
#    DEST_FILE=$RESULTS_DIR/$(basename $FILE)
#    OUT_FILE=$RESULTS_DIR/$(basename $FILE .json).out

#    # Run your processing script on the file
#    python run.py $FILE $DEST_FILE 2>&1 | tee $OUT_FILE
# done


