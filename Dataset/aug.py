# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

class Data_Augmentation:
  
    def __init__(self,imageSize,path_to_dataset):
        self.path = path_to_dataset
        self.transformation_list = []
        transform_part_one = transforms.Compose([transforms.Scale((imageSize,imageSize)), transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) 
        transform_part_two = transforms.Compose([transforms.RandomCrop((imageSize-2,imageSize-2),padding=0),
                                                 transforms.Scale((imageSize,imageSize)),transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_part_three = transforms.Compose([transforms.CenterCrop((imageSize-4,imageSize-4)),
                                                   transforms.Scale((imageSize,imageSize)), transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_part_four = transforms.Compose([transforms.Pad(padding = 1,fill=0),transforms.Scale((imageSize,imageSize)),
                                                  transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
        self.transformation_list.append(transform_part_one)
        self.transformation_list.append(transform_part_two)
        self.transformation_list.append(transform_part_three)
        self.transformation_list.append(transform_part_four)
         

    def data_augmentation(self):
        
        dataset_list = []
        for transformation in self.transformation_list:
            dataset_list.append(dset.ImageFolder(root = self.path, transform = transformation))
        
        return torch.utils.data.ConcatDataset(dataset_list)
    