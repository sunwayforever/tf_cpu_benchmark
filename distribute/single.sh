#!/bin/bash
export CUDA_VISIBLE_DEVICES=-1
if [[ "${NUM_WORKERS}z" == "z" ]]; then
    echo "NUM_WORKERS not set"
    exit 1
fi

./single.py
