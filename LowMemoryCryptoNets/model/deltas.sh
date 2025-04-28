#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --hint=multithread 
#SBATCH --gres=gpu:turing
#SBATCH --partition=ug-gpu-small
#SBATCH -t 02:00:00
#SBATCH -o deltas.out
#SBATCH --job-name=training
source ../.venv/bin/activate
python findingDeltas.py
