##########################################################################

# Example : perform live fire detection in image/video/webcam using
# NasNet-A-OnFire, ShuffleNetV2-OnFire CNN models.

# Copyright (c) 2021 - Neelanjan Bhowmik

# License :
##########################################################################

import cv2
import os
import sys
import math
from PIL import Image
import argparse
import time
import numpy as np
import math
from tabulate import tabulate

import torch
import torchvision.transforms as transforms

import utils.dataload as dataload
import utils.models as models

##########################################################################

def data_transform(model):
    # transforms needed for shufflenetonfire
    if model == 'shufflenetonfire':
        np_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    # transforms needed for nasnetonfire
    if model == 'nasnetonfire':
        np_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    return np_transforms

##########################################################################

# read/process image and apply tranformation

def read_img(frame, np_transforms):
    small_frame = cv2.resize(frame, (224, 224), cv2.INTER_AREA)
    small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    small_frame = Image.fromarray(small_frame)
    small_frame = np_transforms(small_frame).float()
    small_frame = small_frame.unsqueeze(0)
    small_frame = small_frame.to(device)

    return small_frame

##########################################################################

# model prediction on image

def run_model_img(args, frame, model):
    output = model(frame)
    pred = torch.round(torch.sigmoid(output))
    return pred

##########################################################################

# drawing prediction on image

def draw_pred(args, frame, pred, fps_frame):
    height, width, _ = frame.shape
    if prediction == 1:
        if args.image or args.webcam:
            print(f'\t\t|____No-Fire | fps {fps_frame}')
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 2)
        cv2.putText(frame, 'No-Fire', (int(width / 16), int(height / 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        if args.image or args.webcam:
            print(f'\t\t|____Fire | fps {fps_frame}')
        cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 2)
        cv2.putText(frame, 'Fire', (int(width / 16), int(height / 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

##########################################################################
# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--image",
    help="Path to image file or image directory")
parser.add_argument(
    "--video",
    help="Path to video file or video directory")
parser.add_argument(
    "--webcam",
    action="store_true",
    help="Take inputs from webcam")
parser.add_argument(
    "--camera_to_use",
    type=int,
    default=0,
    help="Specify camera to use for webcam option")
parser.add_argument(
    "--trt",
    action="store_true",
    help="Model run on TensorRT")
parser.add_argument(
    "--net",
    type=str,
    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101',
        'vgg16', 'vgg16', 'alexnet', 'squeezenet'
        'densenet', 'shufflenet', 'mobilenet_v2', 'mnasnet'],
    help="select the network")
parser.add_argument(
    "--weight", 
    help="Model weight file path")
parser.add_argument(
    "--cls_name", 
    help="class names - accept below formats:"
        "1. - separated: n0-n1-n2"
        "2. class name file path")
parser.add_argument(
    '--activemap', 
    type=str, 
    default='gradcam',
    choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam',
            'ablationcam', 'eigencam', 'eigengradcam'],
    help='visualise class activation map using gradcam based methods')
parser.add_argument(
    "--cpu",
    action="store_true",
    help="if selected will run on CPU")
parser.add_argument(
    "--output",
    help="a directory path to save output visualisations.")
parser.add_argument(
    '--show',
    action='store_true',
    help='whether show the results on the fly on an OpenCV window.')
parser.add_argument(
    "-fs",
    "--fullscreen",
    action='store_true',
    help="run in full screen mode")
args = parser.parse_args()
t_val = []
for arg in vars(args):
    t_val.append([arg, getattr(args, arg)])
print(tabulate(t_val, 
    ['input', 'value'], 
    tablefmt="psql"))   
##########################################################################
if args.cpu:
    args.device = torch.device('cpu')
else:
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# define display window name
WINDOW_NAME = 'Prediction'

print(f'\n|__Inference Module >>>>')
print(f'\t|__Inference using: {args.net} >>')

# uses cuda if available
if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if args.cpu and args.trt:
    print(f'\n>>TensorRT runs only on gpu. Exit.')
    exit()

# initialise model
print('\t|__Model initilisation >>')

args.cls_name = dataload.extract_clsname(args.cls_name)
if args.net == 'svm':
    print('Yet to implement.')
    exit()  
else:
    model = models.init_model(args)

print(model.features)

# load the given weight file
model = models.load_weight(args, model)
model.eval()
model.to(args.device)

# calculate model size
total_params = sum(p.numel() for p in model.parameters())
print(f'\t|__Model parameter: {total_params}\n')

# TensorRT conversion
if args.trt:
    from torch2trt import TRTModule
    from torch2trt import torch2trt
    data = torch.randn((1, 3, 224, 224)).float().to(device)
    model_trt = torch2trt(model, [data], int8_mode=True)
    model_trt.to(device)
    print(f'\t|__TensorRT activated >>')

if args.output:
    os.makedirs(args.output, exist_ok=True)

# load and process input image directory or image file
if args.image:
    # list image from a directory or file
    if os.path.isdir(args.image):
        lst_img = os.listdir(args.image)
        lst_img = [os.path.join(args.image, file)
                   for file in os.listdir(args.image)]
    if os.path.isfile(args.image):
        lst_img = [args.image]

    fps = []
    # start processing image
    for im in lst_img:
        print('\t|____Image processing: ', im)
        start_t = time.time()
        # frame = cv2.imread(im)

        frame = read_img(im)
        frame  = frame.to(device)
        
        # model prediction
        if args.trt:
            prediction = run_model_img(args, frame, model_trt)
        else:
            prediction = run_model_img(args, frame, model)

        stop_t = time.time()
        fps_frame = int(1 / (stop_t - start_t))
        fps.append(fps_frame)

        # drawing prediction output
        frame = draw_pred(args, frame, prediction, fps_frame)

        # save prdiction visualisation in output path
        if args.output:
            f_name = os.path.basename(im)
            cv2.imwrite(f'{args.output}/{f_name}', frame)

        # display prdiction if output path is not provided
        # press space key to continue/next
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(0)

    avg_fps = sum(fps) / len(fps)
    print(f'\n|__Average fps {int(avg_fps)}')

# load and process input video file or webcam stream
if args.video or args.webcam:
    # define video capture object
    try:
        # to use a non-buffered camera stream (via a separate thread)
        if not(args.video):
            from models import camera_stream
            cap = camera_stream.CameraVideoStream()
        else:
            cap = cv2.VideoCapture()  # not needed for video files

    except BaseException:
        # if not then just use OpenCV default
        print("INFO: camera_stream class not found - camera input may be buffered")
        cap = cv2.VideoCapture()

    if args.output:
        os.makedirs(args.output, exist_ok=True)
    else:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    if args.video:
        if os.path.isdir(args.video):
            lst_vid = os.listdir(args.video)
            lst_vid = [os.path.join(args.video, file)
                       for file in os.listdir(args.video)]
        if os.path.isfile(args.video):
            lst_vid = [args.video]
    if args.webcam:
        lst_vid = [args.camera_to_use]

    # read from video file(s) or webcam
    for vid in lst_vid:
        keepProcessing = True
        if args.video:
            print('\t|____Video processing: ', vid)
        if args.webcam:
            print('\t|____Webcam processing: ')
        if cap.open(vid):
            # get video information
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if args.output and args.video:
                f_name = os.path.basename(vid)
                out = cv2.VideoWriter(
                    filename=f'{args.output}/{f_name}',
                    fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                    fps=float(fps),
                    frameSize=(width, height),
                    isColor=True,
                )

            while (keepProcessing):
                start_t = time.time()
                # start a timer (to see how long processing and display takes)
                start_tik = cv2.getTickCount()

                # if camera/video file successfully open then read frame
                if (cap.isOpened):
                    ret, frame = cap.read()
                    # when we reach the end of the video (file) exit cleanly
                    if (ret == 0):
                        keepProcessing = False
                        continue

                small_frame = read_img(frame, np_transforms)

                # model prediction
                if args.trt:
                    prediction = run_model_img(args, small_frame, model_trt)
                else:
                    prediction = run_model_img(args, small_frame, model)

                stop_t = time.time()
                fps_frame = int(1 / (stop_t - start_t))

                # drawing prediction output
                frame = draw_pred(args, frame, prediction, fps_frame)

                # save prdiction visualisation in output path
                # only for video input, not for webcam input
                if args.output and args.video:
                    out.write(frame)

                # display prdiction if output path is not provided
                else:
                    cv2.imshow(WINDOW_NAME, frame)
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN & args.fullscreen)

                    stop_tik = ((cv2.getTickCount() - start_tik) /
                                cv2.getTickFrequency()) * 1000
                    key = cv2.waitKey(
                        max(2, 40 - int(math.ceil(stop_tik)))) & 0xFF

                    # press "x" for exit  / press "f" for fullscreen
                    if (key == ord('x')):
                        keepProcessing = False
                    elif (key == ord('f')):
                        args.fullscreen = not(args.fullscreen)

        if args.output and args.video:
            out.release()
        else:
            cv2.destroyAllWindows()

print('\n[Done]\n')
