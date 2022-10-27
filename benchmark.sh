#!/bin/bash
export CUDA_VISIBLE_DEVICES='-1'

for ((a=4096;a<=8192*2;a*=2));do
    for i in `seq 1 2 50`; do
        export OMP_NUM_THREADS=$i
        python3 demo.py 1 $i dnn $a 0
    done
done

for ((a=128;a<=512;a*=2));do
    for i in `seq 1 2 50`; do
        export OMP_NUM_THREADS=$i
        python3 demo.py 1 $i cnn $a 5
    done
done

for ((a=128;a<=512;a*=2));do
    for i in `seq 1 2 50`; do
        export OMP_NUM_THREADS=$i
        python3 demo.py 1 $i dscnn $a 5
    done
done

# for ((a=32;a<=128;a*=2));do
#     for i in `seq 1 2 50`; do
#         export OMP_NUM_THREADS=$i
#         python3 demo.py 1 $i inception $a 5
#     done
# done
