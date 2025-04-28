#!/bin/bash

#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=60g
#SBATCH -t 03:00:00
#SBATCH --partition=cpu
#SBATCH --qos="short"
#SBATCH -o ENCDEC_cifar6.o
#SBATCH -w cpu10
source ../../../.venv/bin/activate
python Alexnet_CIFAR.py
