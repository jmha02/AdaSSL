#!/bin/bash
# num_workers_list=(0 4)
# batch_size_list=(32 64)
# epochs_list=(5 10)
# pin_memory_list=(True False)
# for num_workers in "${num_workers_list[@]}"; do
#     for batch_size in "${batch_size_list[@]}"; do
#         for epochs in "${epochs_list[@]}"; do
#             for pin_memory in "${pin_memory_list[@]}"; do
#                 echo "Running with num_workers=${num_workers}, batch_size=${batch_size}, epochs=${epochs}, pin_memory=${pin_memory}"
#                 python profiling/dino.py $num_workers $batch_size $epochs $pin_memory
#             done
#         done
#     done
# done

python profiling/dino.py 0 64 500 True | tee profiling/dino_0_64_500_True.log
python profiling/AdaSSL/dino_lora.py 0 64 500 True | tee profiling/dino_lora_0_64_500_True.log