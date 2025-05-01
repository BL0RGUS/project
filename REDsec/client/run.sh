#!/bin/bash
# This line is required to inform the Linux
#command line to parse the script using
#the bash shell

# Instructing SLURM to locate and assign
#X number of nodes with Y number of
#cores in each node.
# X,Y are integers. Refer to table for
#various combinations
#SBATCH -N 3
#SBATCH -c 32
# Governs the run time limit and
# resource limit for the job. Please pick values
# from the partition and QOS tables below
#for various combinations
#SBATCH --partition=cpu
#SBATCH --qos="short"

export LD_LIBRARY_PATH=:/usr/local/lib:/usr/local/lib:/home2/fsgf66/project/REDsec/lib

TOTALTIME=0
START=$(date +%s.%N)
make keygen
TIME=$(echo "$(date +%s.%N) - $START" | bc)
TOTALTIME=$(echo "$TOTALTIME+$TIME" | bc)
echo "Key Generation: $TIME seconds"

export format=MNIST
export image_path=mnist_test.csv
echo "Dataset: $format"
echo "Encrypting $image_path"
START=$(date +%s.%N)
make encrypt-image
TIME=$(echo "$(date +%s.%N) - $START" | bc)
TOTALTIME=$(echo "$TOTALTIME+$TIME" | bc)
echo "Image Encryption: $TIME seconds"

cd ../nets/mnist/sign1024x3

START=$(date +%s.%N)
make cpu-encrypt
TIME=$(echo "$(date +%s.%N) - $START" | bc)
TOTALTIME=$(echo "$TOTALTIME+$TIME" | bc)
echo "CPU Inference: $TIME seconds"

make clean

cd ../../../client
export format=MNIST

START=$(date +%s.%N)
make decrypt-image
TIME=$(echo "$(date +%s.%N) - $START" | bc)
TOTALTIME=$(echo "$TOTALTIME+$TIME" | bc)
echo "Classification Decryption: $TIME seconds"

echo "Total Time: $TOTALTIME seconds"