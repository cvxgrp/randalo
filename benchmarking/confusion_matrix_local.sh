#!/usr/bin/env bash

BASE_DIR=$(pwd)/confusion_matrix
CONFIGS_DIR=$BASE_DIR/configs
RESULTS_DIR=$BASE_DIR/results
mkdir -p $RESULTS_DIR

FILES=($CONFIGS_DIR/run_*.json)
COUNT=${#FILES[@]}
I=0

for FILE in "${FILES[@]}"
do
    I=$((I+1))
    echo "Running $I of $COUNT"
    DEST_FILE=$RESULTS_DIR/$(basename $FILE)
    python run.py $FILE $DEST_FILE &
    if [ $((I % 2)) == 0 ]; then wait; fi
done
