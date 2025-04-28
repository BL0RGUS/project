#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --partition=cpu
#SBATCH --qos="short"
#SBATCH -o ENCDEC_cifar.o
#SBATCH -w cpu10
source ../../../.venv/bin/activate
python Cryptonets_CIFAR.py
