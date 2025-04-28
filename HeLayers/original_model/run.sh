#!/bin/bash

#SBATCH -N 1
#SBATCH -c 32
#SBATCH --hint=multithread
#SBATCH -o inference.o
#SBATCH --partition=cpu

#SBATCH -w cpu6
lscpu
source ../.venv/bin/activate
python encrypted-mnist.py
