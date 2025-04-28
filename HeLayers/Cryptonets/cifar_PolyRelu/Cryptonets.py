import torch.nn as nn
from pyhelayers.mltoolbox.model.nn_module import nn_module
from pyhelayers.mltoolbox.model.DNN_factory import DNNFactory


@DNNFactory.register('Cryptonets')

class Cryptonets(nn_module):
    
    INPUT_SIZE = (3, 32, 32)
    def __init__(self, **kwargs):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 5, 3, stride=2, padding=1),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.Conv2d(5, 10, 3, stride=2, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(640, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        super().forward(x) 
        x = self.cnn(x) 
        return x
    
DNNFactory.print_supported_models()