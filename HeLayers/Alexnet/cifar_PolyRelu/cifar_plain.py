import os
#Printing only error debug printouts
os.environ["LOG_LEVEL"]="ERROR"
import json
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import utils # used for benchmarking
#trange used to show progress of training loops
from tqdm import trange
#magic function that renders the figure in a notebook
import torchvision
#FHE related import
import pyhelayers
#mltoolbox imports
from pyhelayers.mltoolbox.arguments import Arguments
import pyhelayers.mltoolbox.utils.util as util
from pyhelayers.mltoolbox.poly_activation_converter import starting_point
from pyhelayers.mltoolbox.data_loader.ds_factory import DSFactory
from pyhelayers.mltoolbox.model.DNN_factory import DNNFactory

from Alexnet import AlexNet



#torch.serialization.add_safe_globals([Cryptonets, torch.nn.modules.conv.Conv2d, torch.nn.modules.BatchNorm2d, torch.nn.modules.Linear, torch.nn.modules.activation.Sigmoid, types.SimpleNamespace, torch.nn.modules.container.Sequential, torch.nn.modules.activation.ReLU, torch.nn.modules.Flatten])

debug_mode = False
if debug_mode:
    num_epochs = 1
    batch_size = 500
else:
    num_epochs = 20
    batch_size = 500



args = Arguments(model="AlexNet", dataset_name="CIFAR10", num_epochs=num_epochs, classes=10, data_dir = 'cifar_data')

args.opt = "sgd"
args.lr=0.005
args.batch_size = batch_size
args.range_awareness_loss_weight=0.002
args.range_aware_train = True
args.num_epochs = num_epochs
args.save_dir = "outputs/mltoolbox/range_aware"
args.debug_mode = debug_mode

baseline_chp_location = os.path.join('outputs', 'mltoolbox', 'AlexNet_last_checkpoint.pth.tar')
args.from_checkpoint = baseline_chp_location

trainer, poly_activation_converter, epoch = starting_point(args)


print(trainer.test(args, num_epochs)[0].get_avg('accuracy'))