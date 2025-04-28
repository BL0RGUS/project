#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH -o train_plain.o
#SBATCH --partition=ug-gpu-small
#SBATCH -t 03:00:00
#SBATCH --qos="short"
#SBATCH --gres=gpu:turing
source ../../.venv/bin/activate
python cifar_plain.py
