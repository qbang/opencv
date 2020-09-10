import cv2
import numpy as np

from core import make_ROI_IMG
from matplotlib import pyplot as plt

img = cv2.imread('test.jpeg')
roi = cv2.selectROI(img)
roi_crop = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
cv2.imwrite("img.jpg",roi_crop)

gray= cv2.cvtColor(roi_crop,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

list_all = []
result = []

for i in kp:
    list_all.append([i,i.size])

list_all.sort(key=lambda list_ : list_[1])

for list_ in list_all[:100]:
    result.append(list_[0])

roi_crop = cv2.drawKeypoints(gray,result,roi_crop,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('img', roi_crop)
cv2.waitKey()
cv2.destroyAllWindows()