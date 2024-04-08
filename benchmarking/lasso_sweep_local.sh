#!/usr/bin/env bash

BASE_DIR=$(pwd)/lasso_sweep
CONFIGS_DIR=$BASE_DIR/configs
RESULTS_DIR=$BASE_DIR/results
mkdir -p $RESULTS_DIR

FILES=($CONFIGS_DIR/run_*.json)

for FILE in "${FILES[@]}"
do
    DEST_FILE=$RESULTS_DIR/$(basename $FILE)
    python run.py $FILE $DEST_FILE
done