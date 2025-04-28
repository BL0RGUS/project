#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --hint=multithread 
#SBATCH --gres=gpu:turing
#SBATCH --partition=ug-gpu-small
#SBATCH -o train.out
#SBATCH -t 03:00:00
source ../../../.venv/bin/activate
#python3 bnn_pynq_train.py --data ./.datasets --experiments ./experiments --gpus '0'
python Alexnet_CIFAR.py