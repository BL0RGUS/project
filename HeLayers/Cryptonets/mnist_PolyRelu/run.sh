#!/bin/bash

#SBATCH -N 1
#SBATCH -c 32
#SBATCH -o inference.o
#SBATCH --partition=cpu
#
#SBATCH -w cpu9
lscpu
source ../../.venv/bin/activate
python encrypted_mnist.py
