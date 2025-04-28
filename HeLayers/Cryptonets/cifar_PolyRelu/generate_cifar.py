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

from Cryptonets import Cryptonets



#torch.serialization.add_safe_globals([Cryptonets, torch.nn.modules.conv.Conv2d, torch.nn.modules.BatchNorm2d, torch.nn.modules.Linear, torch.nn.modules.activation.Sigmoid, types.SimpleNamespace, torch.nn.modules.container.Sequential, torch.nn.modules.activation.ReLU, torch.nn.modules.Flatten])

debug_mode = False
if debug_mode:
    num_epochs = 1
    batch_size = 500
else:
    num_epochs = 10
    batch_size = 200

train_dataloader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10(root='./data', train=True,
                               download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                (0.5, 0.5, 0.5), (0.5,0.5,0.5))
                             ])),
  batch_size=batch_size, shuffle=True)

test_dataloader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                (0.5, 0.5, 0.5), (0.5,0.5,0.5))
                             ])),
  batch_size=1, shuffle=True)


args = Arguments(model="Cryptonets", dataset_name="CIFAR10", num_epochs=num_epochs, classes=10, data_dir = 'cifar_data')

#After initializing an `Argument` object it is possible to customize its settings
args.batch_size = batch_size

#Used to do a quick test run. When performing a real run, either remove this argument, or set it to `False` 
args.debug_mode = debug_mode

#Use the following line to utilize the arguments:
trainer, poly_activation_converter, _ = starting_point(args)


model = trainer.get_model()
print(model)


# Run the training loop if the model was not loaded from checkpoint
if not args.from_checkpoint:
        epochs_range = trange(1,args.num_epochs + 1)
        scheduler = ReduceLROnPlateau(trainer.get_optimizer(), factor=0.5, patience=3, min_lr=0.000001)
        for epoch in epochs_range:
                trainer.train_step(args, epoch, epochs_range)
                val_metrics, val_cf = trainer.validation(args, epoch) # perform validation. Returns metrics (val_metrics) and confusion matrix (val_cf)
                
                scheduler.step(val_metrics.get_avg('loss'))
                
        # Saving the model
        # The save location is defined by the args.save_dir argument. The default value is set to outputs/mltoolbox.
        util.save_model(trainer, poly_activation_converter, args, val_metrics, epoch, val_cf)


trainer.test(args, num_epochs)

## RELU_range_aware
if debug_mode:
    num_epochs = 1
    batch_size = 500
else:
    num_epochs = 10
    batch_size = 200
    
print("Training done, do range aware training now")


args = Arguments(model="Cryptonets", dataset_name="CIFAR10", num_epochs=num_epochs, classes=10, data_dir = 'cifar_data')

args.activation_type= "relu_range_aware"
args.batch_size = batch_size
args.range_awareness_loss_weight=0.002
args.range_aware_train = True
args.save_dir = "outputs/mltoolbox/range_aware"
args.debug_mode = debug_mode

baseline_chp_location = os.path.join('outputs', 'mltoolbox', 'Cryptonets_last_checkpoint.pth.tar')
args.from_checkpoint = baseline_chp_location

trainer, poly_activation_converter, epoch = starting_point(args)

scheduler = ReduceLROnPlateau(trainer.get_optimizer(), factor=0.5, patience=3, min_lr=args.min_lr)
epochs_range = trange(1,args.num_epochs + 1)

for epoch in epochs_range:
    poly_activation_converter.replace_activations(trainer, epoch, scheduler)
    trainer.train_step(args, epoch, epochs_range)
    
    val_metrics, val_cf = trainer.validation(args, epoch) # perform validation. Returns metrics (val_metrics) and confusion matrix (val_cf)

util.save_model(trainer, poly_activation_converter, args, val_metrics, epoch, val_cf)

print(trainer.get_model())


## RELU_range_aware
if debug_mode:
    num_epochs = 1
    batch_size = 500
else:
    num_epochs = 10
    batch_size = 200
    
    

args.activation_type= "non_trainable_poly"
args.batch_size = batch_size
args.num_epochs = num_epochs
args.lr=0.005
args.from_checkpoint = "outputs/mltoolbox/range_aware/Cryptonets_last_checkpoint.pth.tar"
args.range_awareness_loss_weight=0.1
args.range_aware_train = True
args.save_dir = "outputs/mltoolbox/polynomial"

args.poly_degree = 10
    
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

print(model)

args.batch_size = 200
trainer, poly_activation_converter, epoch = starting_point(args)

plain_samples, labels = next(iter(trainer.val_generator))
torch.save(plain_samples, 'outputs/mltoolbox/plain_samples.pt')
torch.save(labels, 'outputs/mltoolbox/labels.pt')