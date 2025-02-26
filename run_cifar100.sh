# python AdaSSL/main_cifar100.py --model SimCLR --epochs 100 | tee SimCLR_4_128_200_True.log
# python AdaSSL/main_cifar100.py --model DINO --epochs 5 | tee DINO_4_128_5_True.log
# python AdaSSL/main_cifar100.py --model DINO_LoRA --epochs 5 | tee DINO_LoRA_4_128_5_True.log
# python AdaSSL/main_cifar100.py --model DINO_AdaLoRA --epochs 5 | tee DINO_AdaLoRA_4_128_5_True.log
# python AdaSSL/main_cifar100.py --model DINO_IA3 --epochs 5 | tee DINO_IA3_4_128_5_True.log
python AdaSSL/main_cifar100.py --model DINO_SparseUpdate --epochs 5 | tee DINO_SparseUpdate_4_128_5_True.log
python AdaSSL/main_cifar100.py --model DINO_SparseLoRA --epochs 5 | tee DINO_SparseLoRA_4_128_5_True.log
python AdaSSL/main_cifar100.py --model DINO_SparseLoRA_Tensor --epochs 5 | tee DINO_SparseLoRA_Tensor_4_128_5_True.log
python AdaSSL/main_cifar100.py --model DINO_LoRA --epochs 5 | tee DINO_LoRA_4_128_5_True.log
python AdaSSL/main_cifar100.py --model DINO_LoRA --epochs 5 | tee DINO_LoRA_4_128_5_True.log