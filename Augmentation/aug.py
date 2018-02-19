# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

class Data_Augmentation:
  
    def __init__(self,imageSize,path_to_dataset , list_of_transformations):
        self.path = path_to_dataset
        self.imageSize = imageSize
        self.transformation_list = list_of_transformations

    def data_augmentation(self):
        
        dataset_list = []
        for transformation in self.transformation_list:
            dataset_list.append(dset.ImageFolder(root = self.path, transform = transformation))
        
        return torch.utils.data.ConcatDataset(dataset_list)
    
