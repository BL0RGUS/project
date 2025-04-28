#!/bin/bash

#SBATCH -N 1
#SBATCH -c 32
#SBATCH -o training.o
#SBATCH --partition=cpu

source ../.venv/bin/activate
python generate_original_model.py
