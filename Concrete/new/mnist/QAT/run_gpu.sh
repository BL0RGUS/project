#!/bin/bash

#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=28g
#SBATCH --hint=multithread 
#SBATCH --qos="fsgf66-qos"
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH -o inference_GPU.out
lscpu
source .venv/bin/activate
python Alexnet_CIFAR.py
