#Comparison of Secure Inference Frameworks using Homomorphic Encryption
This repository implements four Fully Hommorphic Encryptioon (FHE) enabled Secure Inference Frameworks. Namely we implememt:
1. [ConcreteML](https://github.com/zama-ai/concrete-ml/tree/release/1.9.x)
2. [REDsec](https://github.com/TrustworthyComputing/REDsec)
4. [HeLayers](https://ibm.github.io/helayers/)
5. [LowMemoryFHEResNet20](https://github.com/narger-ef/LowMemoryFHEResNet20)

These frameworks are applied to the Cryptonets and AlexNet architectures using MNIST and CIFAR-10.

## Usage Instructions
### ConcreteML
1. Install concrete-ml python library
```bash
pip install concrete-ml torch==2.3.1 torchvision==0.18.1
```
2. Train the desired model by setting the training variable to True and running
```
python 
```

### REDsec
1. Install [TFHE v1.1](https://github.com/tfhe/tfhe) with the packaged SPQLIOS-FMA FFT engine.
2. For gpu accelerated inference install [REDcuFHE](https://github.com/TrustworthyComputing/REDcuFHE)

## HeLayers


## LowMemoryCryptonets
