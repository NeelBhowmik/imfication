# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import time
import os
import glob
import copy
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import argparse 
from tqdm import tqdm
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import utils.dataload as dataload
import utils.models as models
import utils.optimisers as optimisers
       
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
    "--net",
    default='',
    help="Select the network {simnet, alexnet,resnet50,...}")
parser.add_argument(
    "--optim",
    default='SGD',
    help="Select optimizer {SGD, Adam}")
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
    "--lr",
    type=float, 
    default=0.001,
    help='initial learning rate for opimisation')
parser.add_argument(
    "--momentum",
    type=float, 
    default=0.5,
    help='momentum term of optimisation')
parser.add_argument(
    "--weight_decay",
    type=float, 
    default=0.0001,
    help='weight decay term of optimisation')
parser.add_argument(
    "--custom_weight",
    type=str, 
    help='custom weight file path to finetune')
parser.add_argument(
    "--batch",
    type=int, 
    default=32,
    help='input training batch size')
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
    "--epoch",
    type=int, 
    default=10,
    help='number of traning epoch')
parser.add_argument(
    "--save_freq",
    type=int, 
    default=100,
    help='save model weight interval')
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
    "--outf",
    help="A directory path to save model output")

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

print(f'|__Train Module: {args.net} || Dataset: {args.db}')
print('|____Start training >>>>')

# data loading
print('\t|__Data loading >>')
# dataloaders, dataset_sizes, class_names = dataload.data_load(args)
dataloaders, dataset_sizes, class_names = dataload.data_load_np(args)

# initialise model
print('\t|__Model initilisation >>')
if not(args.ft) and not(args.pretrained):
    print('\t|__Traning from scratch >>')
else:
    print('\t|__Finetuning the convnet >>')

if args.net == 'simnet1':
    # model = models.simnet(len(class_names))
    if args.custom_weight:
        model = models.simnet1(
            no_ch=args.ichannel, 
            num_classes=7)
    
    else:
        model = models.simnet1(
            no_ch=args.ichannel, 
            num_classes=len(class_names))

    model = model.to(args.device)
    print("\t",model)

    # if args.optim == 'SGD':
    #     optimizer = optim.__dict__[args.optim](
    #         model.parameters(), 
    #         lr=args.lr, momentum=args.momentum)
    # else:
    #     optimizer = optim.__dict__[args.optim](
    #         model.parameters(), 
    #         lr=args.lr, weight_decay=0.0001)
    
    if args.custom_weight:
        model.load_state_dict(torch.load(args.custom_weight)['state_dict'])
        models.set_parameter_requires_grad(model, args.ft)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(args.device)
    print("\t",model)

    params_to_update = model.parameters()
    print("Params to learn:")
    if args.ft:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    optimizer = optimisers.optimisers(
        args.optim,
        model,
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay)


elif args.net == 'simnet2':
    if args.custom_weight:
        model = models.simnet2(
            no_ch=args.ichannel, 
            num_classes=7)
    else:
        model = models.simnet2(
            no_ch=args.ichannel, 
            num_classes=len(class_names))
    # if args.optim == 'SGD':
    #     optimizer = optim.__dict__[args.optim](
    #         model.parameters(), 
    #         lr=args.lr, momentum=args.momentum)
    # else:
    #     optimizer = optim.__dict__[args.optim](
    #         model.parameters(), 
    #         lr=args.lr, weight_decay=0.0001)

    if args.custom_weight:
        model.load_state_dict(torch.load(args.custom_weight)['state_dict'])
        models.set_parameter_requires_grad(model, args.ft)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(args.device)
    print("\t",model)

    params_to_update = model.parameters()
    print("Params to learn:")
    if args.ft:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    optimizer = optimisers.optimisers(
        args.optim,
        model,
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay)

    # print("\tParams to learn:")
    # for name, param in model.named_parameters():
    #     print("\t",name)
    
elif args.net == 'simnet3':
    if args.custom_weight:
        model = models.simnet3(
            no_ch=args.ichannel, 
            num_classes=7)
    else:
        model = models.simnet3(
            no_ch=args.ichannel, 
            num_classes=len(class_names))

    if args.custom_weight:
        model.load_state_dict(torch.load(args.custom_weight)['state_dict'])
        models.set_parameter_requires_grad(model, args.ft)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(args.device)
    print("\t",model)

    params_to_update = model.parameters()
    print("Params to learn:")
    if args.ft:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    optimizer = optimisers.optimisers(
        args.optim,
        model,
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay)

else:
    model = models.initialize_model(
        args.net, 
        len(class_names), 
        args.custom_weight,
        feature_extract=args.ft, 
        use_pretrained=args.pretrained)

    # if args.custom_weight:
    #     model.load_state_dict(torch.load(args.custom_weight)['state_dict'])

    model = model.to(args.device)
    # print("\t",model)

    params_to_update = model.parameters()
    print("Params to learn:")
    if args.ft:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    optimizer = optimisers.optimisers(
        args.optim,
        model,
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay)

    # if args.optim == 'SGD':
    #     optimizer = optim.__dict__[args.optim](
    #         model.parameters(), 
    #         lr=args.lr, momentum=args.momentum)
    # else:
    #     optimizer = optim.__dict__[args.optim](
    #         model.parameters(), 
    #         lr=args.lr, weight_decay=0.0001)

# # use device - cuda/cpu
# model = model.to(args.device)

# Gather the parameters to be optimized/updated in this run. If we are
# finetuning we will be updating all parameters. However, if we are
# doing feature extract method, we will only update the parameters
# that we have just initialized, i.e. the parameters with requires_grad
# is True.

args.statf = None

# params_to_update = model.parameters()
# print("Params to learn:")
# if args.ft:
#     params_to_update = []
#     for name,param in model.named_parameters():
#         if param.requires_grad == True:
#             params_to_update.append(param)
#             print("\t",name)
# else:
#     for name,param in model.named_parameters():
#         if param.requires_grad == True:
#             print("\t",name)


# Setup the optimiser, loss fn, scheduler
# optimizer = optim.SGD(
#     params_to_update, 
#     lr=args.lr, momentum=args.momentum)

# calculate model size
total_params = sum(p.numel() for p in model.parameters())
print(f'\t|__Model Parameter: ', total_params)

criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(
    optimizer, 
    step_size=7, 
    gamma=0.1)

# start training
model, hist = models.train_model(
    args,
    model, criterion, 
    dataloaders,
    dataset_sizes,
    optimizer,
    exp_lr_scheduler, 
    num_epochs=args.epoch
)

# test set statistics 
models.test_model(
    args, 
    model, 
    criterion, 
    dataloaders['test'], 
    dataset_sizes
)

# plt.ioff()
# plt.show()

print('\n[Done]\n')