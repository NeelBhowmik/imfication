# -*- coding: utf-8 -*-

from __future__ import print_function, division

import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import torchvision
# from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt

# import numpy as np
import os
import argparse
from tabulate import tabulate
# import time
# import os
# import copy
# from sklearn.metrics import confusion_matrix

import utils.dataload as dataload
import utils.models as models

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--db",
    help="specify dataset name")
parser.add_argument(
    "--dbpath",
    default='',
    help="specify the dataset directory path")
parser.add_argument(
    "--dbsplit",
    default='test',
    help="specify the dataset dataset split")
parser.add_argument(
    "--datatype",
    type=str,
    help="specify the datatype {image, video}")
parser.add_argument(
    "--net",
    default='',
    help="select the network {alexnet,resnet50,...}")
parser.add_argument(
    "--weight",
    type=str, 
    help='path to model weight file')
parser.add_argument(
    "--batch",
    type=int, 
    default=32,
    help='input testing batch size')
parser.add_argument(
    "--isize",
    type=int, 
    default=224,
    help='input data size')
parser.add_argument(
    "--cpu",
    action="store_true",
    help="if selected will run on CPU")
parser.add_argument(
    "--trt",
    action="store_true",
    help="if selected will run on TensorRT")    
parser.add_argument(
    '--workers', 
    type=int, 
    help='number of data loading workers', 
    default=2)
parser.add_argument(
    "--statf",
    type=str,
    help="a directory path to save test statistics")

args = parser.parse_args()
args = parser.parse_args()
t_val = []
for arg in vars(args):
    t_val.append([arg, getattr(args, arg)])
print(tabulate(t_val, 
    ['input', 'value'], 
    tablefmt="psql"))   
######################################################################

if args.cpu:
    args.device = torch.device('cpu')
else:
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

args.dbsplit = args.dbsplit.split("-")

print(f'\n|__Test Module: {args.net} || Dataset: {args.db}')
print('|____Start testing >>>>')

# data loading
print('\t|__Data loading >>')
dataloaders, dataset_sizes, class_names = dataload.data_load(args)

# initialise model
print('\t|__Model initilisation >>')

if args.net == 'svm':
    print('Yet to implement.')
    exit()  
else:
    # only needed for traning 
    # setting as None/False for test/inference
    args.custom_weight = None
    args.ft = False
    args.pretrained = False
    model = models.initialize_model(
        args.net, 
        len(class_names),
        args.custom_weight,
        feature_extract=args.ft, 
        use_pretrained=args.pretrained)

# load the given weight file
if args.weight:
    if os.path.isfile(args.weight):
        print('\t|__Loading model weight >>')
        model.load_state_dict(
            torch.load(args.weight, 
            map_location=args.device)['state_dict'])
    else:
        print('\t|__Model weight path not found')
        exit()
else:
    print('\t|__Model weight not found')
    exit()

model.eval()
model.to(args.device)

criterion = nn.CrossEntropyLoss()

# calculate model size
total_params = sum(p.numel() for p in model.parameters())
print(f'\t|__Model parameter: ', total_params)

# test set statistics 
models.test_model(
    args, 
    model, 
    criterion, 
    dataloaders[args.dbsplit[0]], 
    dataset_sizes
)

print('\n[Done]\n')