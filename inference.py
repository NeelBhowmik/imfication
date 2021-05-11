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
import argparse
import time
import numpy as np
import math
from tabulate import tabulate

import torch

import utils.dataload as dataload
import utils.models as models
import utils.visualise as vis

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
        " 1. - separated: n0-n1-n2"
        " 2. class name file containing: 1-class name in a line" ) 
parser.add_argument(
    "--conf_thrs",
    type=float,
    default=0.3,
    help="classification confidence threshold"
    "in between {0-1}")
parser.add_argument(
    '--activemap', 
    type=str, 
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
    args.device = torch.device('cpu')
else:
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if args.cpu and args.trt:
    print(f'\n>>TensorRT runs only on gpu. Exit.')
    exit()

# initialise model
print('\t|__Model initilisation >>')
if args.cls_name:
    args.cls_name = dataload.extract_clsname(args.cls_name)
else:
    print(f'[ERROR] Class name not provided: {args.cls_name}')
    exit()

if args.net == 'svm':
    print('Yet to implement.')
    exit()  
else:
    model = models.init_model(args)

# print(model.features)

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
    model_trt.to(args.device)
    print(f'\t|__TensorRT activated >>')

if args.output:
    os.makedirs(args.output, exist_ok=True)

if args.activemap:
    print(f'\t|__Using activation map: {args.activemap}')


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
        print('\n|__Image processing: ', im)
        start_t = time.time()
        # frame = cv2.imread(im)

        frame = dataload.read_img(im)
        frame = frame.to(args.device)
        
        # model prediction
        if args.trt:
            prediction = models.run_model(model_trt, frame)
        else:
            prediction = models.run_model(model, frame)

        stop_t = time.time()
        fps_frame = int(1 / (stop_t - start_t))
        fps.append(fps_frame)

        # drawing prediction output
        im_cv, result = vis.draw_pred(args, im, prediction)
        print(f'\t|__{result}')

        # Activation map visualisation
        cam_img = im_cv
        if args.activemap:
            cam_img = vis.activecam(args, im, model)
        # save prdiction visualisation in output path
        # display in opencv if args.show == Ture
        # display prdiction if output path is not provided
        # press space key to continue/next
        combine_img = cv2.hconcat([im_cv, cam_img])
        f_name = os.path.basename(im)
        vis.vis_write(args, f_name, im_cv, cam_img, combine_img)
        
        # if args.output:
        #     f_name = os.path.basename(im)
        #     cv2.imwrite(f'{args.output}/{f_name}', im_cv)
        #     if args.activemap:
        #         cv2.imwrite(f'{args.output}/{f_name}_{args.activemap}.png', cam_img)
        # elif args.output and args.show:
        #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        #     f_name = os.path.basename(im)
        #     cv2.imwrite(f'{args.output}/{f_name}', im_cv)
        #     if args.activemap:
        #         cv2.imwrite(f'{args.output}/{f_name}_{args.activemap}.png', cam_img) 
        #         cv2.imshow(WINDOW_NAME, combine_img)             
        #     else:
        #         cv2.imshow(WINDOW_NAME, im_cv)
        #     cv2.waitKey(0)
        # else:
        #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        #     if args.activemap:
        #         cv2.imshow(WINDOW_NAME, combine_img)             
        #     else:
        #         cv2.imshow(WINDOW_NAME, im_cv)
        #     cv2.waitKey(0)

    avg_fps = sum(fps) / len(fps)
    print(f'\n|-->>Average fps {int(avg_fps)}')

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
            print('\t|__Video processing: ', vid)
        if args.webcam:
            print('\t|__Webcam processing: ', vid)
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
