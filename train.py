##########################################################################

# Example : Perform training on datasaet

# Copyright (c) 2024 - Neelanjan Bhowmik

# License : 
##########################################################################
from __future__ import print_function, division
import torch
import torch.nn as nn
# import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

from tabulate import tabulate
import argparse 
import utils.dataload as dataload
import utils.models as models
import utils.optimisers as optimisers

#####################################################################   
def parse_args():    
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
        "--net",
        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101',
            'vgg16', 'vgg19', 'alexnet', 'squeezenet'
            'densenet', 'shufflenet', 'mobilenet_v2', 'mnasnet'],
        help="select the network")
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
        default=1,
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
        "--work_dir",
        type=str, 
        default='./logs',
        help="a directory path to save model output")

    args = parser.parse_args()
    return args
#####################################################################

def main():
    args = parse_args()
    t_val = []
    for arg in vars(args):
        t_val.append([arg, getattr(args, arg)])
    print(tabulate(t_val, 
        ['input', 'value'], 
        tablefmt="psql"))    
    
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
    args.class_names = class_names

    # initialise model
    print('\t|__Model initilisation >>')
    if not(args.ft) and not(args.pretrained):
        print('\t|__Traning from scratch >>')
    else:
        print('\t|__Finetuning the network >>')

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

    args.statf = None

    # calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\t|__Model parameter: {total_params}\n')

    criterion = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer, 
        step_size=7, 
        gamma=0.1)

    # start training
    model = models.train_model(
        args,
        model, criterion, 
        dataloaders,
        dataset_sizes,
        optimizer,
        exp_lr_scheduler, 
        num_epochs=args.epoch
    )

    # test set statistics
    print('\n|____Start testing >>>>')
    models.test_model(
        args, 
        model, 
        criterion, 
        dataloaders['test'], 
        dataset_sizes
    )

    print('\n[Done]\n')

if __name__ == '__main__':
    main()