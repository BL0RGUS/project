#!/bin/bash
# This line is required to inform the Linux
#command line to parse the script using
#the bash shell

# Instructing SLURM to locate and assign
#X number of nodes with Y number of
#cores in each node.
# X,Y are integers. Refer to table for
#various combinations
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -w cpu9
#SBATCH --hint=multithread 

# Governs the run time limit and
# resource limit for the job. Please pick values
# from the partition and QOS tables below
#for various combinations
#SBATCH --partition=cpu
#SBATCH --qos="short"


source ../../.venv/bin/activate
python mnist_in_fhe.py
