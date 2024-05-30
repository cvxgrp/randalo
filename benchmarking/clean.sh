#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    exit
fi


pushd $1
rm -r configs results output
python generate_configs.py
popd
