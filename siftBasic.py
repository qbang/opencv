import numpy as np
import cv2 as cv
img = cv.imread('test.png')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
print(len(kp))
# img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img=cv.drawKeypoints(gray,kp,img,flags=4)

cv.imwrite('3.jpg',img)