#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH -o train_plain.o
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:pascal
source ../../.venv/bin/activate
python generate_cifar.py
