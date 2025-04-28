import torch.nn as nn
from pyhelayers.mltoolbox.model.nn_module import nn_module
from pyhelayers.mltoolbox.model.DNN_factory import DNNFactory
from pyhelayers.mltoolbox.data_loader.dataset_wrapper import DatasetWrapper
from pyhelayers.mltoolbox.data_loader.ds_factory import DSFactory
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST
import torchvision.transforms

@DNNFactory.register('Cryptonets')

class Cryptonets(nn_module):
    
    INPUT_SIZE = (1, 28, 28)
    def __init__(self, **kwargs):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 5, 3, stride=2, padding=1),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.Conv2d(5, 10, 3, stride=2, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(490, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        super().forward(x) 
        x = self.cnn(x) 
        return x
    
DNNFactory.print_supported_models()

