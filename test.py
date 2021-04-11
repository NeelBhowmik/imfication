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
import argparse
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
    default='s-trig',
    help="dataset name")
parser.add_argument(
    "--dbpath",
    default='',
    help="Specify the dataset directory path")
parser.add_argument(
    "--dbsplit",
    default='train-val-test',
    help="Specify the dataset dataset split")
parser.add_argument(
    "--datatype",
    type=str,
    help="Specify the datatype {hlz, xglcm}")
parser.add_argument(
    "--mv",
    action="store_true",
    help="Perform multiview aggregation")
parser.add_argument(
    "--net",
    default='',
    help="Select the network {simnet, alexnet,resnet50,...}")
parser.add_argument(
    "--ft",
    action="store_true",
    help="If true - only update the reshaped layer params"
          "If flase - traning from scratch")
parser.add_argument(
    "--pretrained", 
    action="store_true", 
    help='Use pretrained network.')
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
    "--ichannel",
    type=int, 
    default=3,
    help='input data channel no')
parser.add_argument(
    "--isize",
    type=int, 
    default=224,
    help='input data size')
parser.add_argument(
    "--cpu",
    action="store_true",
    help="If selected will run on CPU")
parser.add_argument(
    '--workers', 
    type=int, 
    help='number of data loading workers', 
    default=2)

parser.add_argument(
    "--statf",
    type=str,
    help="A directory path to save statistics")

args = parser.parse_args()
print('=' * 10)
print(f'\n{args}\n')
print('=' * 10)
######################################################################

if args.cpu:
    args.device = torch.device('cpu')
else:
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

args.dbsplit = args.dbsplit.split("-")

print(f'|__Test Module: {args.net} || Dataset: {args.db}')

# data loading
print('|____Data loading >>')

# dataloaders, dataset_sizes, class_names = dataload.data_load(args)
dataloaders, dataset_sizes, class_names = dataload.data_load_np(args)
args.cls_name = class_names
args.custom_weight = None
# initialise model
print('|____Model initilisation >>')

if args.net == 'simnet1':
    # model = models.simnet(len(class_names))
    model = models.simnet1(
        no_ch=args.ichannel, 
        num_classes=len(class_names))
    
elif args.net == 'simnet2':
    model = models.simnet2(
        no_ch=args.ichannel, 
        num_classes=len(class_names))

elif args.net == 'simnet3':
    model = models.simnet3(
        no_ch=args.ichannel, 
        num_classes=len(class_names))
        
else:
    model = models.initialize_model(
        args.net, 
        len(class_names),
        args.custom_weight,
        feature_extract=args.ft, 
        use_pretrained=args.pretrained)

# load pretrained weight
if args.weight:
    print("|____Loading Model Weight >>")
    model.load_state_dict(torch.load(args.weight, map_location=args.device)['state_dict'])
    # model.load_state_dict(torch.load(args.weight, map_location=args.device))
    # print("\t|__Done.\n")
else:
    print('Model weight not found')
    exit()

model.eval()
model.to(args.device)

criterion = nn.CrossEntropyLoss()

# calculate model size
total_params = sum(p.numel() for p in model.parameters())
print(f'\t|__Model Parameter: ', total_params)

# test set statistics 
models.test_model(
    args, 
    model, 
    criterion, 
    dataloaders[args.dbsplit[2]], 
    dataset_sizes
)

print('\n[Done]\n')