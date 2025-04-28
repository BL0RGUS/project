#!/bin/bash

#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=128g
#SBATCH -o inference.o
#SBATCH --partition=cpu
#SBATCH --hint=multithread
#SBATCH -w cpu10

free -g
source ../../.venv/bin/activate
python encrypted_cifar.py
