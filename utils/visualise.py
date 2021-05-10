import cv2
import os

##########################################################################
# drawing prediction on image
def draw_pred(args, frame, pred, cls_name):
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