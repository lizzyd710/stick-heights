import cv2
import numpy as np
import imutils

def background_subtract(capture, out):
    # create the background subtractor using the K-nearest neighbors algorithm.
    back_sub = cv2.createBackgroundSubtractorKNN()

    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        # update the background model
        fg_mask = back_sub.apply(frame)

        # write the masked frame to the output file
        out.write(fg_mask)

        # this is just showing each frame/mask. eventually i'll write it instead of showing it
        # so i can use the modified capture for tracking.
        # cv2.imshow('Frame', frame)
        # cv2.imshow('FG Mask', fg_mask)

        # keyboard = cv2.waitKey(1)
        # if keyboard == 'q' or keyboard == 27:
        # break