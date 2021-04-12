# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import time
import os
import glob
import copy
from PIL import Image
import argparse 
from tqdm import tqdm
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

#####################################################################
# Data augmentation and normalization for training
# Just normalization for val/test
# ---------------------------------
def data_load(args):
    if 'inception' in args.net:
        args.isize = 299

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((args.isize,args.isize)),
            transforms.RandomResizedCrop(args.isize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((args.isize,args.isize)),
            transforms.CenterCrop(args.isize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((args.isize,args.isize)),
            transforms.CenterCrop(args.isize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = f'{args.dbpath}/{args.db}'

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in args.dbsplit}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch,
                                                shuffle=True, num_workers=args.workers)
                for x in args.dbsplit}
    dataset_sizes = {x: len(image_datasets[x]) for x in args.dbsplit}
    class_names = image_datasets[args.dbsplit[0]].classes
    
    # print('image_datasets: ', image_datasets)
    # print('image_datasets: ', image_datasets['test'][0][0])
    print('\tclass name: ', class_names)
    print('\tdataloaders: ', dataset_sizes)
 
    return dataloaders, dataset_sizes, class_names

#####################################################################

def numpy_loader_xglcm(filename):
    np_array = np.load(filename)['x_glcm'][0] 
    np_array = torch.from_numpy(np_array).float()
    # np_pil = Image.fromarray(np_array)
    return np_array

def numpy_loader_hlz(filename):
    np_array = np.load(filename)

    data_np_stack = np.stack((
        np_array['h_glcm'][0], 
        np_array['l_glcm'][0], 
        np_array['z_glcm'][0]))


    # numpy arrary to torch tensor float
    data_np_stack_ = torch.from_numpy(data_np_stack).float()
    
    # apply normalisation
    # norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # data_np_stack = norm(data_np_stack_)

    return data_np_stack_

#####################################################################
# Data loading from np filr - Data augmentation and normalization for training
# 
# ---------------------------------
def data_load_np(args):
    
    np_transforms = {
        'train': transforms.Compose([
            # transforms.Resize((args.isize,args.isize)),
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=(0, 90)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            # transforms.Resize((args.isize,args.isize)),
            transforms.ToPILImage(),
            # transforms.RandomRotation(degrees=(0, 90)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            # transforms.Resize((args.isize,args.isize)),
            transforms.ToPILImage(),
            # transforms.RandomRotation(degrees=(0, 90)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    data_dir = f'{args.dbpath}/{args.db}'

    if args.datatype == 'xglcm':
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                # transform=np_transforms[x],
                                                loader=numpy_loader_xglcm)
                        for x in args.dbsplit}

    elif args.datatype == 'hlz':
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                # transform=np_transforms[x],
                                                loader=numpy_loader_hlz)
                        for x in args.dbsplit}

    else:
        print('Invalid datatype.')
        exit()

    if args.mv:
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=2,
                                                shuffle=False, num_workers=args.workers)
                for x in args.dbsplit}
    else:
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch,
                                                    shuffle=True, num_workers=args.workers)
                    for x in args.dbsplit}
    dataset_sizes = {x: len(image_datasets[x]) for x in args.dbsplit}
    class_names = image_datasets[args.dbsplit[0]].classes
    
    # print('image_datasets: ', image_datasets)
    # print('image_datasets: ', image_datasets['test'][0][0])
    print('\tclass name: ', class_names)
    print('\tdataloaders: ', dataset_sizes)
 
    return dataloaders, dataset_sizes, class_names