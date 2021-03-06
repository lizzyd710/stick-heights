import cv2
import numpy as np
import imutils
from collections import deque
import sys

VIDEO_NAME = 'videos/green-tips.mp4'
WAITKEY_DELAY = 60

##################################
# All of this code is just copied from the Optical Flow tutorial site to use as a skeleton.
# From abidrahmank/OpenCV2-Python-Tutorials on github.
##################################


def optical_flow(capture):
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = capture.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while (1):
        ret, frame = capture.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        k = cv2.waitKey(WAITKEY_DELAY) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)


##################################
# This is copied from an opencv tutorial to use as a skeleton.
# https://docs.opencv.org/3.4/db/df8/tutorial_py_meanshift.html
##################################

def blob_track(capture):
    # take first frame of the video
    ret, frame = capture.read()
    # setup initial location of window
    r, h, c, w = 250, 90, 400, 125  # simply hardcoded the values
    track_window = (c, r, w, h)
    # set up the ROI for tracking
    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    while (1):
        ret, frame = capture.read()
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            # apply meanshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.intp(pts)
            img2 = cv2.polylines(frame, [pts], True, 255, 2)
            cv2.imshow('img2', img2)
            k = cv2.waitKey(WAITKEY_DELAY) & 0xff
            if k == 27:
                break
                # else:
                # cv2.imwrite(chr(k) + ".jpg", img2)
        else:
            break


##################################
# This is copied from a tutorial to use as a skeleton.
# https://www.pyimagesearch.com/2015/09/21/opencv-track-object-movement/
##################################

def track_sticks(capture, tip_hsv_lower, tip_hsv_upper, trail_length=None):
    # tip_hsv_lower = (12, 82, 169)  # (3, 79, 82)
    # tip_hsv_upper = (66, 180, 222)  # (86, 168, 138)
    pts = deque(maxlen=trail_length)

    while True:
        ret, frame = capture.read()

        if frame is None:
            break

        # resize frame, blur, convert to HSV
        frame = imutils.resize(frame, width=600)  # might play around with this later. just copied from tut right now.
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color, then perform dilations and erosions to remove small blobs
        mask = cv2.inRange(hsv, tip_hsv_lower, tip_hsv_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cv2.imshow("Mask", mask)
        # compute contour # might not have in final, but again, copying now, experiment later.
        # find contours in mask and initialize the current (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use it to compute
            # the minimum enclosing circle and centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if radius meets minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
        pts.appendleft(center)

        # loop over set of tracked points
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            # compute thickness of line (might not need to do this) and draw connecting lines
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 5)

        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", frame)

        # if (capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        # cv2.imwrite("temp name 2.jpg", frame)

        key = cv2.waitKey(WAITKEY_DELAY) & 0xFF
        if key == ord("q"):
            break


if __name__ == '__main__':
    # cap = cv2.VideoCapture('videos/orange-sticks.mp4')
    #green tips: (38,100,26) (72,183,215)
    cap = cv2.VideoCapture(VIDEO_NAME)
    print(cap.isOpened())
    lower = input("Enter lower value for HSV (format: H,S,V): ")
    lower = lower.split(",")
    lower = (int(lower[0]), int(lower[1]), int(lower[2]))
    upper = input("Enter upper value for HSV (format: H,S,V): ")
    upper = upper.split(",")
    upper = (int(upper[0]), int(upper[1]), int(upper[2]))
    track_sticks(cap, lower, upper)
    cv2.destroyAllWindows()
    cap.release()

# kalman tracking
# hough transform circles
