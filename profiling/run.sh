#!/bin/bash
num_workers_list=(4 8 16)
batch_size_list=(64 128 256)
epochs_list=(5 10 20)
pin_memory_list=(True False)
for num_workers in "${num_workers_list[@]}"; do
    for batch_size in "${batch_size_list[@]}"; do
        for epochs in "${epochs_list[@]}"; do
            for pin_memory in "${pin_memory_list[@]}"; do
                echo "Running with num_workers=${num_workers}, batch_size=${batch_size}, epochs=${epochs}, pin_memory=${pin_memory}"
                python profiling/simclr_torch.py $num_workers $batch_size $epochs $pin_memory
            done
        done
    done
done