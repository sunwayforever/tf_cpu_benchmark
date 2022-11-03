#!/bin/bash
export CUDA_VISIBLE_DEVICES=-1
if [[ "${NUM_WORKERS}z" == "z" ]]; then
    echo "NUM_WORKERS not set"
    exit 1
fi

pids=""
for i in `seq 0 ${NUM_WORKERS}`; do
    if [[ $i -eq 0 ]]; then
        ./mirror_worker.py $i &
    else
        ./mirror_worker.py $i &>/dev/null &
    fi
    pids="$pids $!"
done

for pid in $pids; do
    wait $pid
done
