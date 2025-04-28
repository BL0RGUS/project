#!/bin/bash

#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=28g
#SBATCH --hint=multithread 
#SBATCH --qos="fsgf66-qos"
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH -o CryptoNets_CIFAR_GPU.out
#SBATCH --job-name=mnist_gpu
lscpu
source ../../../.venv/bin/activate
python Cryptonets_CIFAR.py
