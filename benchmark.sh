#!/bin/bash
export CUDA_VISIBLE_DEVICES='-1'

for ((a=4096;a<=8192*2;a*=2));do
    for i in `seq 1 2 50`; do
        export OMP_NUM_THREADS=$i
        python3 demo.py $i 1 dnn $a 0
    done
done

for ((a=128;a<=512;a*=2));do
    for i in `seq 1 2 50`; do
        export OMP_NUM_THREADS=$i
        python3 demo.py $i 1 cnn $a 5
    done
done

for ((a=128;a<=512;a*=2));do
    for i in `seq 1 2 50`; do
        export OMP_NUM_THREADS=$i
        python3 demo.py $i 1 dscnn $a 5
    done
done

# for ((a=32;a<=128;a*=2));do
#     for i in `seq 1 1 25`; do
#         if [[ $i == 1 ]]; then
#             export OMP_NUM_THREADS=1
#             python3 demo.py $i 1 inception $a 5
#         else
#             export OMP_NUM_THREADS=2
#             python3 demo.py $i 2 inception $a 5
#         fi
#     done
# done
