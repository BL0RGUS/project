#!/bin/bash

#SBATCH -N 1
#SBATCH -c 32
#SBATCH -o inference2.o
#SBATCH --partition=cpu
#SBATCH --hint=multithread
#SBATCH -w cpu10

source ../../.venv/bin/activate
python encrypted_cifar.py
