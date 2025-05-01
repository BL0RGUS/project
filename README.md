# Comparison of Secure Inference Frameworksusing Homomorphic Encryption

This repository implements 4  FHE powereed Secure Inference frameworks on two neural networks on MNIST and CIFAR-10

## Brief Instructions
### ConcreteML
1. Install the concrete-ml python library
2. install torch=2.3.1 and torchvision 0.18.1
3. First train the desired model/dataset by set the variabele training=True in the file <MODEL>_<DATASET>.py and run it, note that training on AlexNet will takes around 30 mins on ncc.
4. Run the desired model by setting training to false and running the same file.

### HElayers
1. install the pyhelayers python library
2. install requirements.txt
3. Train the desired network/dataset/activation this is usually generate_<DATASET>.py in the respective folder.
4. Run the desired network/dataset/activation this is usually encrypted_<DATASET>.py in the respective folder.


### REDsec
The REDsec code is incomplete due to unfortunate code deletion. Some results (3-FCNN) can still be run using the following steps:
1. Install [TFHE v1.1](https://github.com/tfhe/tfhe) with the packaged SPQLIOS-FMA FFT engine.
2. Install [REDCUFHE](https://github.com/TrustworthyComputing/REDcuFHE)
3. Run the slurm script run.sh, be sure to change the network to one present in nets file

### ELMO
1. Install [OpenFHE](https://github.com/TrustworthyComputing/REDcuFHE) V1.0.4, the version number is very important.
2. Train the model you want by running Model.py
3. Using the generate weights jupyter notebook enocde the weights of your chosen model, the AlexNetSmallfc is currently encoded.
4. Using the findingDeltas.py file find the ranges of values preceding relu activations in the chosen model, AlexNetSmallfc is currently encoded.
5. Select the appropriate main.cpp file for your mode.
6. replace the scale parameter in the convolution, fc and relu layers with the respective deltas found in finding deltas step.
7. run run.sh to perform secure inference.
