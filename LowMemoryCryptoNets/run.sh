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
#SBATCH -c 32
#SBATCH --hint=multithread
#SBATCH --mem 128g
#SBATCH -t 03:00:00
#SBATCH -w cpu10
# Governs the run time limit and
# resource limit for the job. Please pick values
# from the partition and QOS tables below
#for various combinations
#SBATCH --partition=cpu
#SBATCH -o inference.o
#SBATCH --qos="short"

export LD_LIBRARY_PATH=:/usr/local/lib:/usr/local/lib:/home2/fsgf66/project/LowMemoryCryptoNets/lib
cd build
lscpu
TOTALTIME=0
START=$(date +%s.%N)
./LowMemoryCryptoNets generate_keys 1
TIME=$(echo "$(date +%s.%N) - $START" | bc)
TOTALTIME=$(echo "$TOTALTIME+$TIME" | bc)
echo "Key Generation: $TIME seconds"

./LowMemoryCryptoNets load_keys 1
cd ..