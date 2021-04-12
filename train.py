# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
# import torch.optim as optim
from torch.optim import lr_scheduler
# from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms

import time
import os
from tabulate import tabulate

import argparse 
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import utils.dataload as dataload
import utils.models as models
import utils.optimisers as optimisers
       
# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--db",
    help="specify the dataset name")
parser.add_argument(
    "--dbpath",
    help="specify the dataset directory path")
parser.add_argument(
    "--dbsplit",
    default='train-val-test',
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
    "--optim",
    default='SGD',
    help="select optimizer {SGD, Adam}")
parser.add_argument(
    "--ft",
    action="store_true",
    help="if true - only update the reshaped layer params"
         "if flase - traning from scratch")
parser.add_argument(
    "--pretrained", 
    action="store_true", 
    help='use ImageNet pretrained weight.')
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
    help='input data channel number')
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
    help="if selected will run on CPU")
parser.add_argument(
    '--workers', 
    type=int, 
    help='number of data loading workers', 
    default=2)
parser.add_argument(
    "--outf",
    type=str, 
    default='logs',
    help="a directory path to save model output")

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

print(f'\n|__Train Module: {args.net} || Dataset: {args.db}')
print('|____Start training >>>>')

# data loading
print('\t|__Data loading >>')
dataloaders, dataset_sizes, class_names = dataload.data_load(args)
# dataloaders, dataset_sizes, class_names = dataload.data_load_np(args)

# initialise model
print('\t|__Model initilisation >>')
if not(args.ft) and not(args.pretrained):
    print('\t|__Traning from scratch >>')
else:
    print('\t|__Finetuning the convnet >>')

if args.net == 'svm':
    print('Yet to implement.')
    exit()

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
    print("\n\tParams to learn:")
    if args.ft:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t\t",name)

    optimizer = optimisers.optimisers(
        args.optim,
        model,
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay)

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
print(f'\t|__Model Parameter: {total_params}\n')

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

print('\n[Done]\n')