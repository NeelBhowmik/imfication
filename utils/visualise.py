import cv2
import os
import numpy as np

from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image

##########################################################################
# drawing prediction on image
#-----------------------------
def draw_pred(args, frame, prediction):
    if isinstance(frame, str):
        frame = cv2.imread(frame)
    height, width, _ = frame.shape
    score = round(prediction["score"], 2)
    
    if score >= args.conf_thrs:
        label_text = f'{args.cls_name[prediction["clsidx"]]} {str(score)}'
        cv2.putText(frame, label_text, (int(width / 24), int(height / 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        label_text = 'No prediction'

    return frame, label_text
##########################################################################
# initialisation for grad-cam
# ---------------------------
def init_activecam(args, model):
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}
    
    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # target_layers = [model.blocks[-1].norm1]
    if 'resnet' in args.net:
        target_layers = [model.layer4[-1]]
    elif 'vgg' in args.net:
        target_layers = [model.features[-1]]
    elif 'densenet' in args.net:
        target_layers = [model.features[-1]]
    elif 'mnasnet' in args.net:
        target_layers = [model.layers[-1]]
    else:
        print(f"[ERROR] Invalid model name: {args.net}")
        exit()

    cam = methods[args.activemap](model=model,
                        target_layers=target_layers,
                        use_cuda=args.device)
    # cam = methods[args.method](model=model,
    #                     target_layer=target_layer,
    #                     use_cuda=args.device)
    gb_model = GuidedBackpropReLUModel(model=model, 
                        use_cuda=args.device)    

    return cam, gb_model
##########################################################################
# visualisation for grad-cam
# ---------------------------
def activecam(args, img, cam, gb_model):
   

    if isinstance(img, str):
        rgb_img = cv2.imread(img, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224), cv2.INTER_AREA)
        # cv2.imshow('Activecam', rgb_img)
        # cv2.waitKey(0)
    else:
        rgb_img = img[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224), cv2.INTER_AREA)
            
    # rgb_img = cv2.imread(img, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32
    args.aug_smooth = False
    args.eigen_smooth = False
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=target_category,
                        aug_smooth=args.aug_smooth,
                        eigen_smooth=args.eigen_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.device)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    # cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
    # cv2.imwrite(f'{args.method}_gb.jpg', gb)
    # cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)
    return cam_image


##########################################################################
# save prdiction visualisation in output path
# display in opencv if args.show == Ture
# display prdiction if output path is not provided
# press space key to continue/next
#-------------------------------------------------
def vis_write(args, f_name, im_cv, cam_img, combine_img):
    WINDOW_NAME = 'Prediction'
    if args.output and args.show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imwrite(f'{args.output}/{f_name}', im_cv)
        if args.activemap:
            cv2.imwrite(f'{args.output}/{f_name}_{args.activemap}.png', cam_img) 
            cv2.imshow(WINDOW_NAME, combine_img)             
        else:
            cv2.imshow(WINDOW_NAME, im_cv)
        cv2.waitKey(0)
    
    elif args.output:
        cv2.imwrite(f'{args.output}/{f_name}', im_cv)
        if args.activemap:
            cv2.imwrite(f'{args.output}/{f_name}_{args.activemap}.png', cam_img)
    
    elif args.show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if args.activemap:
            cv2.imshow(WINDOW_NAME, combine_img)             
        else:
            cv2.imshow(WINDOW_NAME, im_cv)
        cv2.waitKey(0)

    else:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if args.activemap:
            cv2.imshow(WINDOW_NAME, combine_img)             
        else:
            cv2.imshow(WINDOW_NAME, im_cv)
        cv2.waitKey(0)