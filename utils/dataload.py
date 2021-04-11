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
extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.npz', '.npy')
def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    ) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)
    # both_none = extensions is None and is_valid_file is None
    # both_something = extensions is not None and is_valid_file is not None
    # if both_none or both_something:
    #     raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    # if extensions is not None:
    #     def is_valid_file(x: str) -> bool:
    #         return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    # is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                # if is_valid_file(path):
                item = path, class_index
                instances.append(item)
    return instances

# Dataloaded class for np files
class npFolder(Dataset):

    def __init__(self, root_path, transform=None):
        print(root_path)
        self.data_numpy_list = [x for x in glob.glob(os.path.join(root_path + '/anomaly', '*X*.npz'))]

        classes, class_to_idx = self.find_classes(root_path)
        print(classes, class_to_idx)
        samples = make_dataset(root_path, class_to_idx, extensions)
        print(samples)
        # print(self.data_numpy_list)

        self.transforms = transforms
        self.data_list = []
        for ind in range(len(self.data_numpy_list)):
            data_slice_file_name = self.data_numpy_list[ind]
            data_i = np.load(data_slice_file_name)
            self.data_list.append(data_i['x_glcm'])
    

    def find_classes(self, dir: str):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


    def __getitem__(self, index):

        self.data = np.asarray(self.data_list[index])
        # self.data = np.stack((self.data, self.data, self.data)) # gray to rgb 64x64 to 3x64x64
        if self.transforms:
            self.data = self.transforms(self.data)
        # print(torch.from_numpy(self.data).float())
        return torch.from_numpy(self.data).float()

    def __len__(self):
        return len(self.data_numpy_list)

def dataload_np(args):

    data_dir = f'{args.dbpath}/{args.db}'
    val_dir_path = data_dir + '/val' 
    
    val_dataset = npFolder(val_dir_path)
    print(len(val_dataset))

    dataloaders = DataLoader(
        val_dataset, batch_size=args.batch,
        shuffle=True, 
        num_workers=args.workers
    )


    dataset_sizes = len(dataloaders)
    # class_names = val_dataset.classes
    # print('image_datasets: ', image_datasets)
    # print('class name: ', class_names)
    print('dataloaders: ', dataset_sizes)

    return dataloaders, dataset_sizes, class_names


# class npFolder(Dataset):
#     def __init__(self, data, targets, transform=None):
#         self.data = data
#         self.targets = torch.LongTensor(targets)
#         self.transform = transform

#     def __getitem__(self, index):
#         x = self.data[index]
#         y = self.targets[index]

#         if self.transform:
#             x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
#             x = self.transform(x)

#         return x, y

#     def __len__(self):
#         return len(self.data)

# Let's create 10 RGB images of size 128x128 and ten labels {0, 1}
# data = list(np.random.randint(0, 255, size=(10, 3, 128, 128)))
# targets = list(np.random.randint(2, size=(10)))

# transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
# dataset = MyDataset(data, targets, transform=transform)
# dataloader = DataLoader(dataset, batch_size=5)

#####################################################################
# Data augmentation and normalization for training
# Just normalization for validation
# ---------------------------------
def data_load(args):
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