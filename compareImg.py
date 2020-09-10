import numpy as np
import cv2
import math
import csv

sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 100)
flann = cv2.FlannBasedMatcher(index_params, search_params)

img1 = cv2.imread('s.jpg',0)
img2 = cv2.imread('1.jpg', 0)
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

matches = flann.knnMatch(des1,des2,k=2)
matchesMask = [[0,0] for i in range(len(matches))]

for i,(m,n) in enumerate(matches):
    #강한유사도:0.5/중간유사도:0.7/약한유사도:0.8
    if m.distance < 0.7*n.distance:
        check = True
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

cv2.imwrite('0.7.jpg', img3)