#!/bin/bash
#SBATCH --job-name=json_processing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-100%10  # Adjust 1-100 to match the number of .json files, %10 limits to 10 concurrent jobs

# Assuming all your .json files are in a directory called "configs"
CONFIGS_DIR=lasso_scaling_1
RESULTS_DIR=$CONFIGS_DIR/results
mkdir -p $RESULTS_DIR

FILES=($CONFIGS_DIR/run_*.json)

for ((i=1; i<=50; i++))
do
  SLURM_ARRAY_TASK_ID=$i
   # SLURM_ARRAY_TASK_ID is the variable that gets a unique value for each job in the array
   FILE=${FILES[$SLURM_ARRAY_TASK_ID-1]}
   DEST_FILE=$RESULTS_DIR/$(basename $FILE)
   OUT_FILE=$RESULTS_DIR/$(basename $FILE .json).out

   # Run your processing script on the file
   python run.py $FILE $DEST_FILE 2>&1 | tee $OUT_FILE
done


