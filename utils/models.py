# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import time
import os
import glob
# import argparse
import copy
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import argparse 
from tqdm import tqdm
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


######################################################################
# define simnet 
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# class simnet(nn.Module):
#     def __init__(self, num_classes):
#         super(simnet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class Unit(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Unit,self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            kernel_size=3,
            out_channels=out_channels,
            stride=1,padding=1)
        self.bn = nn.BatchNorm2d(
            num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

class simnet2(nn.Module):
    def __init__(self, no_ch=3, num_classes=6):
        super(simnet2,self).__init__()

        #Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=no_ch,out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=64, out_channels=128)
        self.unit9 = Unit(in_channels=128, out_channels=128)
        self.unit10 = Unit(in_channels=128, out_channels=128)
        self.unit11 = Unit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=128, out_channels=128)
        self.unit13 = Unit(in_channels=128, out_channels=128)
        self.unit14 = Unit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=32)
        
        #Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, 
            self.unit3, self.pool1, 
            self.unit4, self.unit5, 
            self.unit6, self.unit7, 
            self.pool2, self.unit8, 
            self.unit9, self.unit10, 
            self.unit11, self.pool3,
            self.unit12, self.unit13, 
            self.unit14, self.avgpool)

        self.fc = nn.Linear(in_features=128,
            out_features=num_classes)
        
        self.initialize_weights()

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1,128)
        output = self.fc(output)
        return output

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    

class simnet1(nn.Module):
    def __init__(self, no_ch=3, num_classes=6):
        super(simnet1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=no_ch, 
            out_channels=12, 
            kernel_size=3, 
            stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=12, 
            out_channels=12, kernel_size=3, 
            stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=12, 
            out_channels=24, 
            kernel_size=3, 
            stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=24, 
            out_channels=24, 
            kernel_size=3, 
            stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.fc = nn.Linear(in_features=128 * 128 * 24, 
            out_features=num_classes)
        
        self.initialize_weights()

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.pool(output)

        output = self.conv3(output)
        output = self.relu3(output)

        output = self.conv4(output)
        output = self.relu4(output)

        output = output.view(-1, 128 * 128 * 24)

        output = self.fc(output)

        return output

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.xavier_normal_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

class simnet3(nn.Module):
    def __init__(self, no_ch=3, num_classes=6):
        super(simnet3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=no_ch, 
            out_channels=12, 
            kernel_size=3, 
            stride=1, padding=1)
        # self.bn = nn.BatchNorm2d(
        #     num_features=12)
        self.relu1 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=12, 
            out_channels=24, 
            kernel_size=3, 
            stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.fc = nn.Linear(in_features=128 * 128 * 24, 
            out_features=num_classes)
        
        self.initialize_weights()
        
        # self.avgpool = nn.AvgPool2d(kernel_size=128)
        # self.fc = nn.Linear(in_features=24, 
        #     out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = output.view(-1, 128 * 128 * 24)
        # output = output.view(-1, 24)

        output = self.fc(output)

        return output

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

######################################################################
# Model initialization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This helper function sets the .requires_grad attribute of the 
# parameters in the model to False when we are feature extracting. 
# By default, when we load a pretrained model all of the parameters 
# have .requires_grad=True, which is fine if we are training from 
# scratch or finetuning. However, if we are feature extracting and 
# only want to compute gradients for the newly initialized layer 
# then we want all of the other parameters to not require gradients. 

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

######################################################################
# Model initialization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
def initialize_model(model_name, num_classes, custom_weight, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    if "resnet" in model_name:
        """ Resnet18/34/50/101/152
        """
        model_ft = models.__dict__[model_name](pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # input_size = 224

        if custom_weight:
            print('\tFine tune from custom weight>>>>')
            model_ft.load_state_dict(torch.load(custom_weight)['state_dict'])
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.__dict__[model_name](pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        # input_size = 224

    elif "vgg" in model_name:
        """ VGG11_bn
        """
        model_ft = models.__dict__[model_name](pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        # input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.__dict__[model_name](pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        # input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.__dict__[model_name](pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        # input_size = 224

    # elif model_name == "inception_v3":
    #     """ Inception v3
    #     Be careful, expects (299,299) sized images and has auxiliary output
    #     """
    #     model_ft = models.__dict__[args.net](pretrained=use_pretrained)
    #     set_parameter_requires_grad(model_ft, feature_extract)
    #     # Handle the auxilary net
    #     num_ftrs = model_ft.AuxLogits.fc.in_features
    #     model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    #     # Handle the primary net
    #     num_ftrs = model_ft.fc.in_features
    #     model_ft.fc = nn.Linear(num_ftrs,num_classes)
    #     input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft

#####################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data augmentations.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

def train_model(args, model, criterion, dataloaders, dataset_sizes, optimizer, scheduler, num_epochs=10):
    since = time.time()

    test_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                # model.to(args.device)

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase]):
                # print(inputs.shape, labels.shape)
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # save model at each epoch
            if (epoch+1)%args.save_freq == 0:
                save_weights(model, epoch, args)    

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                # save best model weights
                save_weights(model, epoch, args, is_best=True)
            if phase == 'test':
                test_acc_history.append(epoch_acc)   

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc), f'Epoch: {best_epoch}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, test_acc_history

#####################################################################
# Save the best model weight
# --------------------------
def save_weights(model, epoch, args, is_best=False):
    """Save net weights for the current epoch.
    Args:
        epoch ([int]): Current epoch number.
    """
    weight_dir = f'{args.outf}/{args.db}/{args.net}_{args.lr}/train/weights'
    os.makedirs(weight_dir, exist_ok=True)
    if is_best: torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, f'{weight_dir}/net_final.pth')
    else: torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, f'{weight_dir}/net_{epoch}.pth')

######################################################################
# Test the model predictions with statistics
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def test_model(args, model, criterion, test_data, dataset_sizes):
    print('|____Start testing >>')
    test_loss = 0.0
    tn_t, fp_t, fn_t, tp_t = 0, 0, 0, 0

    labels_lst = []
    pred_lst = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_data):
            # print(path)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            test_loss += loss.item()*inputs.size(0)

            if args.mv:
                
                # print('==\nlabl---->', labels.cpu().detach().tolist())
                # print('1pre---->',preds.cpu().detach().tolist())
                l_lst = labels.cpu().detach().tolist()
                p_lst =  preds.cpu().detach().tolist()
                p_lst_n =  []
                # for i, p in enumerate(p_lst):
                if p_lst[0] == l_lst[0] or p_lst[1] == l_lst[0]:
                    p_lst_n.append(l_lst[0])
                    p_lst_n.append(l_lst[1])
                else:
                    p_lst_n = p_lst_n + p_lst

                # print('2pre---->',p_lst_n)    
                
                labels_lst = labels_lst + l_lst
                pred_lst = pred_lst + p_lst_n


            else:    
                labels_lst = labels_lst + labels.cpu().detach().tolist()
                pred_lst = pred_lst + preds.cpu().detach().tolist()

            # print('labels_lst: ',labels.cpu().detach())
            # print('pred_lst  : ',preds.cpu().detach())

            # print('Labels: ', labels.cpu().detach() )
            # print('Pred: ', preds.cpu().detach() )    
            
            # Statistics
            # tn, fp, fn, tp = confusion_matrix(
            #     labels.cpu().detach(), 
            #     preds.cpu().detach(), 
            #     labels=[0, 1]).ravel()
            
            # tn_t += tn
            # fp_t += fp
            # fn_t += fn
            # tp_t += tp
    
    print('labels_lst: ',labels_lst)
    print('pred_lst: ',pred_lst)

    print('-' * 20)
    print("Test Statistics:\n")
    # average test loss
    test_loss = test_loss/dataset_sizes[args.dbsplit[2]]
    print('Test Loss: {:.4f}\n'.format(test_loss))
    stat(args, labels_lst, pred_lst)
    print('-' * 20)

    # stats = []

    # tn = tn_t
    # fp = fp_t
    # fn = fn_t
    # tp = tp_t

    # recall = tp / (tp + fn)
    # recall = round(recall, 3)
    # fpr = fp / (fp + tn)
    # fpr = round(fpr, 3)
    # precision = tp / (tp + fp)
    # precision = round(precision,3)
    # accuracy = (tp + tn) / (tp + tn + fp + fn)
    # accuracy = round(accuracy, 3)
    # f_score = 2.0 * (precision * recall) / (precision + recall)
    # f_score = round(f_score, 3)

    # tex_stat = f'{recall} & {fpr} & {f_score} & {precision} & {accuracy}'
    # stats.append(['TPR', str(recall)])
    # stats.append(['FPR', str(fpr)])
    # stats.append(['F1', str(f_score)])
    # stats.append(['Precesion', str(precision)])
    # stats.append(['Accuracy', str(accuracy)])
    # stats.append(['Tex_stat', tex_stat])
    
    # print('TPR - Recall: ', recall)
    # print('FPR: ', fpr)
    # print('F score: ', f_score)
    # print('Precision: ', precision)
    # print('Accuracy: ', accuracy)
    # print('\nTex version:', tex_stat)
    
    # if args.statf:
    #     csv_write(args, stats)

    # # print('TPR - Recall: %.3f' % (recall))
    # # print('FPR: %.3f' % (fpr))
    # # print('F score: %.3f' % (f_score))
    # # print('Precision: %.3f' % (precision))
    # # print('Accuracy: %.3f' % (accuracy))

    # # print("Accuracy of 1:")
    # # print(tp/(tp+fn))
    # # print(tp)
    # # print(tp+fn)
    # # print("Accuracy of 0:")
    # # print(tn/(tn+fp))
    # # print(tn)
    # # print(tn+fp)
    # print('-' * 20)

######################################################################
# Save test statistics in csv file
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def csv_write(args, stats):
    stat_dir = f'{args.statf}/{args.db}/{args.net}'
    os.makedirs(stat_dir, exist_ok=True)

    with open(stat_dir + "/stat.csv", "w") as outfile:
        for s in stats:
            outfile.write(f'{s[0]},{s[1]}\n')
    outfile.close()

######################################################################
# Test statistics confusion matrix
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def stat(args, labels_lst,pred_lst):
    cnf_mtx = confusion_matrix(
                labels_lst, 
                pred_lst,
                normalize='true')

    ####TPR/FPR/TNR
    FP = cnf_mtx.sum(axis=0) - np.diag(cnf_mtx)  
    FN = cnf_mtx.sum(axis=1) - np.diag(cnf_mtx)
    TP = np.diag(cnf_mtx)
    TN = cnf_mtx.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    PPV_array = np.array(PPV)
    PPV_array = np.nan_to_num(PPV_array)
    PPV = list(PPV_array)
    
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    # f1 score
    F1 = 2.0 * (PPV * TPR) / (PPV + TPR)
    F1_array = np.array(F1)
    F1_array = np.nan_to_num(F1_array)
    F1 = list(F1_array)
    
    #Avg
    TPRav = round(sum(TPR)/(len(TPR)), 3)
    TNRav = round(sum(TNR)/(len(TNR)), 3)
    PPVav = round(sum(PPV)/(len(PPV)), 3)
    FPRav = round(sum(FPR)/(len(FPR)), 3)
    FNRav = round(sum(FNR)/(len(FNR)), 3)
    ACCav = round(sum(ACC)/(len(ACC)), 3)
    F1av  = round(sum(F1)/(len(F1)), 3)

    stats = []
    tex_stat = f'{TPRav} & {FPRav} & {F1av} & {PPVav} & {ACCav}'
    stats.append(['TPR', str(TPRav)])
    stats.append(['FPR', str(FPRav)])
    stats.append(['F1', str(F1av)])
    stats.append(['Precesion', str(PPVav)])
    stats.append(['Accuracy', str(ACCav)])
    stats.append(['Tex_stat', tex_stat])

    # print('*' * 20)
    print('Stats from confusion Matrix:')
    print('Class-wise:')
    print(f'TPR: {TPR}\nTNR: {TNR}\nFPR: {FPR}\nFNR: {FNR}\nACC: {ACC}')
    print('\nOverall:')
    # print(f'TPR: {round(TPRav,3)}\nTNR: {round(TNRav,3)}\nFPR: {round(FPRav,3)}\nFNR: {round(FNRav,3)}\nACC: {round(ACCav,3)}')
    print(f'TPR: {TPRav}\nFPR: {FPRav}\nF Score: {F1av}\nPrecesion: {PPVav} \nAccuracy: {ACCav}')
    print('\nTex version:', tex_stat)
    
    ####            

    # stats write in csv file
    # confusion matrix generation
    if args.statf:
        csv_write(args, stats)

        cnf_mtx_display = ConfusionMatrixDisplay(cnf_mtx, display_labels=args.cls_name)
        _, ax = plt.subplots(figsize=(10, 9))
        plt.rcParams.update({'font.size': 14})
        cnf_mtx_display.plot(ax=ax, values_format='.2f',xticks_rotation=45)

        stat_dir = f'{args.statf}/{args.db}/{args.net}'
        os.makedirs(stat_dir, exist_ok=True)
        stat_img = stat_dir + '/cnfmat.png'
        cnf_mtx_display.figure_.savefig(stat_img)

######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

######################################################################
