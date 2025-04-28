#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH -o training.o
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:ampere
source ../../.venv/bin/activate
python generate_mnist.py
