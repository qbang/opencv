import numpy as np
import cv2
import math
import csv
from core import make_ROI_IMG,x_to_mm
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 2

sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 100)
flann = cv2.FlannBasedMatcher(index_params, search_params)

video_path = 'linear_actuator.MTS'
# 1920.0, 1080.0
cap = cv2.VideoCapture(video_path)
# cap.set(3, 10000.0)
# cap.set(4, 5625.0)

# origin_x,origin_y = make_ROI_IMG(cap)
origin_x,origin_y = 587.0, 339.0

img1 = cv2.imread('img.jpg',0)
img1 = cv2.resize(img1, dsize=(0, 0), fx=1.0, fy=1.0, interpolation=cv2.INTER_LINEAR)

kp1, des1 = sift.detectAndCompute(img1,None)

zero = 0
one = 0
moretwo = 0

data = []
while True:
    try:
        # ret는 bool, img2는 numpy.ndarray
        ret, img2 = cap.read()
        
        if not ret: break
        kp2, des2 = sift.detectAndCompute(img2,None)
        matches = flann.knnMatch(des1,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.3*n.distance:
                good.append(m)
        if len(good)>=MIN_MATCH_COUNT:
            moretwo += 1
        else:
            if len(good) == 0:
                zero += 1
            else:
                one += 1

            print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None

        if cv2.waitKey(1) == ord('q'): break
    except:
        pass
cap.release()

print("zero ", zero)
print("one ", one)
print("moretwo",moretwo)