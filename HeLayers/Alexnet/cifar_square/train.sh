#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH -o train_gpu.o
source ../../.venv/bin/activate
module load cuda/11.8-cudnn8.6
python generate_cifar.py
