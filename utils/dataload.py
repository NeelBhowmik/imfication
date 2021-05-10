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

##########################################################################
# read/process image and apply tranformation
def read_img(frame):
    
    if isinstance(frame, str):
        frame = cv2.imread(frame)
    
    np_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
        (0.229, 0.224, 0.225))
    ])
    
    small_frame = cv2.resize(frame, (224, 224), cv2.INTER_AREA)
    small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    small_frame = Image.fromarray(small_frame)
    small_frame = np_transforms(small_frame).float()
    small_frame = small_frame.unsqueeze(0)
    return small_frame

##########################################################################

# extract class names
def extract_clsname(cls_name):
    
    if os.path.isfile(cls_name):
        print('Read from file')
    else:
        cls_name = cls_name.split("-")

    return cls_name

##########################################################################
