#!/bin/bash
export CUDA_VISIBLE_DEVICES=-1

pids=""
for i in `seq 0 1`; do
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
