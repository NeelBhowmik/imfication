# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F
# from torch.optim import lr_scheduler
# import torchvision
from torchvision import models

import numpy as np
import time
import os
import copy
import csv
from tabulate import tabulate
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

######################################################################
# Model initialization
# --------------------
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
# --------------------
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
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # input_size = 224

        if custom_weight:
            print('\tFine tune from custom weight>>>>')
            model_ft.load_state_dict(torch.load(custom_weight, map_location=args.device)['state_dict'])
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

    elif "squeezenet" in model_name:
        """ Squeezenet
        """
        model_ft = models.__dict__[model_name](pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        # input_size = 224

    elif "densenet" in model_name:
        """ Densenet
        """
        model_ft = models.__dict__[model_name](pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        # input_size = 224

    elif model_name == "inception_v3":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.__dict__[args.net](pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
    #     input_size = 299

    # elif "googlenet" in model_name:
    #     """ GoogLeNet
    #     """
    #     model_ft = models.__dict__[model_name](pretrained=use_pretrained)
    #     if custom_weight:
    #         print('\tFine tune from custom weight>>')
    #         model_ft.load_state_dict(torch.load(custom_weight, map_location=args.device)['state_dict'])
    #     set_parameter_requires_grad(model_ft, feature_extract)
        
    #     # Handle the auxilary net
    #     # num_ftrs = model_ft.aux1.fc.in_features
    #     # model_ft.aux1.fc = nn.Linear(num_ftrs, num_classes)
    #     # num_ftrs = model_ft.aux2.fc.in_features
    #     # model_ft.aux2.fc = nn.Linear(num_ftrs, num_classes)
    #     # Handle the primary net
    #     num_ftrs = model_ft.fc.in_features
    #     model_ft.fc = nn.Linear(num_ftrs,num_classes)
        
    elif "shufflenet" in model_name:
        """ ShuffleNet
        """
        model_ft = models.__dict__[model_name](pretrained=use_pretrained)
        if custom_weight:
            print('\tFine tune from custom weight>>')
            model_ft.load_state_dict(torch.load(custom_weight, map_location=args.device)['state_dict'])
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    elif "mobilenet" in model_name:
        """ MobileNet
        """
        model_ft = models.__dict__[model_name](pretrained=use_pretrained)
        if custom_weight:
            print('\tFine tune from custom weight>>')
            model_ft.load_state_dict(torch.load(custom_weight, map_location=args.device)['state_dict'])
        set_parameter_requires_grad(model_ft, feature_extract)
        if model_name == 'mobilenet_v2':
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        # if model_name == 'mobilenet_v3_large' or model_name == 'mobilenet_v3_small':
        #     num_ftrs = model_ft.classifier[3].in_features
        #     model_ft.classifier[3] = nn.Linear(num_ftrs, num_classes)
    
    elif "mnasnet" in model_name:
        """ MNASNet
        """
        model_ft = models.__dict__[model_name](pretrained=use_pretrained)
        if custom_weight:
            print('\tFine tune from custom weight>>')
            model_ft.load_state_dict(torch.load(custom_weight, map_location=args.device)['state_dict'])
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        # input_size = 224

    else:
        print(f"{model_name}: Invalid model name, exiting...")
        exit()

    return model_ft

#####################################################################
# Visualize a few images
# ----------------------

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

def train_model(args, model, 
        criterion, dataloaders, 
        dataset_sizes, optimizer, 
        scheduler, num_epochs=10):
    since = time.time()

    train_acc_history = []
    train_ls_history = []
    test_acc_history = []
    test_ls_history = []
    args.work_dir = f'{args.work_dir}/{args.db}/{args.net}_{args.lr}/train/weights'
    os.makedirs(args.work_dir, exist_ok=True)
    log_file = f'{args.work_dir}/logs.log'
    logging.basicConfig(filename=log_file, 
                filemode='w',
                level=logging.INFO,
                format='%(asctime)s - %(message)s', 
                datefmt='%d-%m-%Y %H:%M:%S')

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        epoch_txt = f'Epoch: {epoch+1}/{num_epochs}'
        print(f'\n{epoch_txt}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        phase_txt = ''
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
                 
            phase_txt = f'{phase_txt} {phase} loss: {round(epoch_loss, 4)} {phase} acc: {round(epoch_acc.item(), 4)}'
            print(f'{phase} loss: {round(epoch_loss, 4)}, {phase} acc: {round(epoch_acc.item(), 4)}')
            # save model at each epoch
            if (epoch+1)%args.save_freq == 0:
                save_weights(model, epoch, args)    

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                bestacc_t = f'Best test Acc: {round(best_acc.item(),4)} Epoch: {best_epoch+1}'
                print(bestacc_t)    
                # save best model weights
                save_weights(model, epoch, args, is_best=True)

            if phase == 'train':
                # print(epoch_acc.cpu().detach().tolist())
                train_acc_history.append(epoch_acc.cpu().detach().tolist())
                train_ls_history.append(epoch_loss)
            if phase == 'test':
                # print(epoch_acc)
                test_acc_history.append(epoch_acc.cpu().detach().tolist())
                test_ls_history.append(epoch_loss)
            
        log_txt = f'{epoch_txt} {phase_txt} {bestacc_t}'        
        logging.info(log_txt)
       
    time_elapsed = time.time() - since
    tr_time = 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60)
    print(f'\n{tr_time}\n{bestacc_t}')
    logging.info(f'{tr_time}, {bestacc_t}')
    
    # load best model weights
    model.load_state_dict(best_model_wts)

    # plot train/val/test loss/acc
    args.plt_name = 'loss'
    stat_plot(args, train_ls_history, test_ls_history)
    args.plt_name = 'accuracy'
    stat_plot(args, train_acc_history, test_acc_history)
   
    return model

#####################################################################
# Save the best model weight
# --------------------------
def save_weights(model, epoch, args, is_best=False):
    """Save net weights for the current epoch.
    Args:
        epoch ([int]): Current epoch number.
    """
    weight_dir = args.work_dir
    # os.makedirs(weight_dir, exist_ok=True)
    if is_best: torch.save({'epoch': epoch, 
        'state_dict': model.state_dict()}, 
        f'{weight_dir}/net_final.pth')
    else: torch.save({'epoch': epoch, 
        'state_dict': model.state_dict()}, 
        f'{weight_dir}/net_{epoch}.pth')

######################################################################
# plot traning/val/test loss/accuracy
# ------------------------------------------
def stat_plot(args, train_history, test_history):

    plt.clf()
    plt.title(f"{args.plt_name} vs. epochs")
    plt.xlabel("epochs")
    plt.ylabel(args.plt_name)
    plt.plot(range(1,args.epoch +1),train_history,label="train")
    plt.plot(range(1,args.epoch+1),test_history,label="test")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, args.epoch+1, 1.0))
    plt.legend()
    # plt.show()
    plt.savefig(f'{args.work_dir}/{args.plt_name}.png')

######################################################################
# Test the model predictions with statistics
# ------------------------------------------
def test_model(args, model, criterion, test_data, dataset_sizes):
    test_loss = 0.0
    tn_t, fp_t, fn_t, tp_t = 0, 0, 0, 0

    labels_lst = []
    pred_lst = []

    with torch.no_grad():
        # for i, (inputs, labels) in enumerate(test_data):
        for (inputs, labels) in tqdm(test_data):
            # print(path)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            test_loss += loss.item()*inputs.size(0)
 
            labels_lst = labels_lst + labels.cpu().detach().tolist()
            pred_lst = pred_lst + preds.cpu().detach().tolist()
    
    print("Test Statistics:\n")
    # average test loss
    test_loss = test_loss/dataset_sizes['test']
    print('Test Loss: {:.4f}\n'.format(test_loss))
    stat(args, labels_lst, pred_lst)
    # print('-' * 20)

######################################################################
# Save test statistics in csv file
# --------------------------------
def csv_write(args, stats):
    stat_dir = f'{args.statf}/{args.db}/{args.net}'
    os.makedirs(stat_dir, exist_ok=True)

    with open(f'{stat_dir}/stat.csv', "w", newline='') as statf:
        writer = csv.writer(statf)
        for line in stats:  
            writer.writerow(line)
    statf.close()   

    # with open(stat_dir + "/stat.csv", "w") as work_dirile:
    #     for s in stats:
    #         work_dirile.write(f'{s[0]},{s[1]}\n')
    # work_dirile.close()

######################################################################
# Test statistics confusion matrix
# --------------------------------
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

    cls_stat = []
    cls_stat.append(['Class', 'TPR', 'FPR', 'F1', 'Precesion', 'Accuracy'])
    for i, cls_n in enumerate(args.class_names):
        s_c = [cls_n, 
            str(round(TPR[i],3)), 
            str(round(FPR[i],3)),
            str(round(F1[i],3)),
            str(round(PPV[i],3)),
            str(round(ACC[i],3))
        ]
        cls_stat.append(s_c)
    
    cls_stat.append(['', '', '', '', '', ''])
    cls_stat.append(['All-cls', 
        str(TPRav), 
        str(FPRav), 
        str(F1av), 
        str(PPVav), 
        str(ACCav)])
    tex_stat = f'{TPRav} & {FPRav} & {F1av} & {PPVav} & {ACCav}'

    print(tabulate(cls_stat, 
        headers="firstrow",
        tablefmt="psql"))    

    # stats write in csv file
    # confusion matrix generation
    if args.statf:
        csv_write(args, cls_stat)

        cnf_mtx_display = ConfusionMatrixDisplay(cnf_mtx, 
            display_labels=args.class_names)

        _, ax = plt.subplots(figsize=(10, 9))
        plt.rcParams.update({'font.size': 14})
        cnf_mtx_display.plot(ax=ax, 
            values_format='.2f',
            xticks_rotation=45, 
            cmap='plasma')

        stat_dir = f'{args.statf}/{args.db}/{args.net}'
        os.makedirs(stat_dir, exist_ok=True)
        stat_img = stat_dir + '/cnfmat.png'
        cnf_mtx_display.figure_.savefig(stat_img)

######################################################################
# Visualizing the model predictions
# ---------------------------------

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
# Model initialisation during test/inference
# ------------------------------------------
def init_model(args):
    # only needed for training 
    # setting as None/False for test/inference
    net_list = ['resnet18', 'resnet34', 'resnet50', 'resnet101',
        'vgg16', 'vgg16', 'alexnet', 'squeezenet'
        'densenet', 'shufflenet', 'mobilenet_v2', 'mnasnet']

    if args.net in net_list:
        args.custom_weight = None
        args.ft = False
        args.pretrained = False
        model = initialize_model(
            args.net, 
            len(args.cls_name),
            args.custom_weight,
            feature_extract=args.ft, 
            use_pretrained=args.pretrained)
    else:
        print(f'\t|__Invalid model name: {args.net}')
        exit()
    return model
    
######################################################################
# Model weight load from weight file
# during test/inference
# -----------------------------------
def load_weight(args, model):
    if args.weight:
        if os.path.isfile(args.weight):
            print('\t|__Loading model weight >>')
            model.load_state_dict(
                torch.load(args.weight, 
                map_location=args.device)['state_dict'])
            return model
        else:
            print(f'\t|__[ERROR] Invalid model weight: {args.weight}')
            exit()
    else:
        print(f'\t|__[ERROR] Model weight missing: {args.weight}')
        exit()
######################################################################
# model prediction on image
# -----------------------------------
def run_model(model, input):
    sm = nn.Softmax(dim = 1)
    with torch.no_grad():
        outputs = model(input)
        outputs = sm(outputs)
        _, preds = torch.max(outputs, 1)
        prediction = preds.cpu().detach().tolist()[0]
        output = outputs.cpu().detach().numpy()
        
    results = {'clsidx': prediction, 
        'score': output[0][prediction]}
    
    return results