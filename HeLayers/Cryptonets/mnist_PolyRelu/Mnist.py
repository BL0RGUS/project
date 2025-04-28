# MIT License
#
# Copyright (c) 2020 International Business Machines
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch
from torchvision import transforms, datasets
from pyhelayers.mltoolbox.data_loader.dataset_wrapper import DatasetWrapper
from pyhelayers.mltoolbox.he_dl_lib.my_logger import get_logger
from pyhelayers.mltoolbox.data_loader.ds_factory import DSFactory
from pyhelayers.mltoolbox.utils.util import is_cuda_available



class MNISTDataset(DatasetWrapper):
    """A wrapper to the standard Cifar10 dataset, available at torchvision.datasets.CIFAR10. The current wrapper class supplyes
    the required transformations and augmentations, and also implements the required DatasetWrapper methods
    
    """
    def __init__(self, resize=False, data_path ='data'):
        self.resize = resize
        self.logger = get_logger()

        test_transformations = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]

        if resize:
            test_transformations = [transforms.Resize(256), transforms.CenterCrop(224)] + test_transformations

        self.path = data_path




        transform_test = transforms.Compose(test_transformations)
        transform_train = transforms.Compose(test_transformations)

        self._train_data =  self.__get_dataset("train", transform_train,  path=data_path)
        self._test_data, self._val_data =  self.__test_val_dataset(transform_test, path=data_path, val_split=0.5)
        self._approximation_data = self._split_approximation_set(0.2)

    def get_class_labels_dict(self):
        return {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
    
    def is_imbalanced(self):
        """Always returns False - Cifar10 dataset is balanced"""
        return False

    def get_train_data(self):
        """Returns the training data"""
        return self._train_data

    def get_test_data(self):
        """Returns the test data"""
        return self._test_data

    def get_val_data(self):
        """Returns the validation data"""
        return self._val_data


    def get_samples_per_class(self, ds):
        """Returns the number of samples in each class. 
        The Cifar10 dataset has the same number of images in each class.

        Params:
            - dataset (VisionDataset): The dataset

        Returns:
            - list<int>: the number of samples in each class.
        """
        assert (isinstance(ds, datasets.MNIST))
        data_len = len(ds)
        return  [data_len / 10] * 10


    def get_approximation_set(self):
        """Returns data set to be used for range approximation"""
        return self._approximation_data


    def __test_val_dataset(self, transform, path, val_split=0.5):
        """Splits the data and returns validation and test sets"""
        ds = datasets.MNIST(root=path, train=False, download=True, transform=transform)

        test_idx, val_idx = train_test_split(list(range(len(ds))), test_size=val_split, random_state=42)
        val_ds = Subset(ds, val_idx)
        test_ds = Subset(ds, test_idx)

        val_ds.LABEL2NAME_DICT = ds.class_to_idx
        test_ds.LABEL2NAME_DICT = ds.class_to_idx

        return val_ds, test_ds


    def __get_dataset(self, mode, transform,  path):
        """returns the torchvision.datasets.CIFAR10 dataset"""
        ds = datasets.MNIST(root=path, train=(mode == 'train'), download=True, transform=transform)
        return ds


@DSFactory.register('MNIST')
class MNISTDataset_28(MNISTDataset):
    def __init__(self, path ='cifar_data', args = None, **kwargs):
        super().__init__(False, path)



DSFactory.print_supported_datasets()