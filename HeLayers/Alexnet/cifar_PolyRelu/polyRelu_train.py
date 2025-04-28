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
from pyhelayers.mltoolbox.utils.util import is_cuda_available
from Alexnet import AlexNet


print(is_cuda_available())

debug_mode = False

if debug_mode:
    num_epochs = 1
    batch_size = 500
else:
    num_epochs = 20
    batch_size = 200
    
    
args = Arguments(model="AlexNet", dataset_name="CIFAR10", num_epochs=num_epochs, classes=10, data_dir = 'cifar_data')



args.opt = "sgd"
args.activation_type= "non_trainable_poly"
args.batch_size = batch_size
args.num_epochs = num_epochs
args.lr=0.005
args.from_checkpoint = "outputs/mltoolbox/range_aware/AlexNet_best_checkpoint.pth.tar"
args.range_awareness_loss_weight=0.1
args.range_aware_train = True
args.save_dir = "outputs/mltoolbox/polynomial4"

args.poly_degree = 4
    
trainer, poly_activation_converter, epoch = starting_point(args)

scheduler = ReduceLROnPlateau(trainer.get_optimizer(), factor=0.5, patience=2, min_lr=args.min_lr, verbose=True)
epochs_range = trange(1,args.num_epochs + 1)

for epoch in epochs_range:
    poly_activation_converter.replace_activations(trainer, epoch, scheduler)
    trainer.train_step(args, epoch, epochs_range)
    val_metrics, val_cf = trainer.validation(args, epoch)
    
util.save_model(trainer, poly_activation_converter, args, val_metrics, epoch, val_cf)


trainer.test(args, num_epochs)

path, model = util.save_onnx(args, poly_activation_converter, trainer)

args.batch_size = 200
trainer, poly_activation_converter, epoch = starting_point(args)

plain_samples, labels = next(iter(trainer.val_generator))
torch.save(plain_samples, 'outputs/mltoolbox/plain_samples.pt')
torch.save(labels, 'outputs/mltoolbox/labels.pt')