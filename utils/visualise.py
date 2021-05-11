import cv2
import os

##########################################################################
# drawing prediction on image
def draw_pred(args, im, prediction):
    if isinstance(im, str):
        frame = cv2.imread(im)
    height, width, _ = frame.shape
    score = round(prediction["score"], 2)
    
    if score >= args.conf_thrs:
        label_text = f'{args.cls_name[prediction["clsidx"]]} {str(score)}'
        cv2.putText(frame, label_text, (int(width / 24), int(height / 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # cv2.putText(frame, label_text, (int(width / 16), int(height / 4)),
        #     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    return frame, label_text
##########################################################################