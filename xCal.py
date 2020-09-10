import numpy as np
import cv2
import math
import csv
from matplotlib import pyplot as plt
import datetime

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def make_ROI_IMG(cap):
    ret, img = cap.read()
    rect = cv2.selectROI("select ROI",img)
    left,top,w,h = [int(v) for v in rect]
    right = left + w
    bottom = top + h
    cv2.imwrite("img.jpg", img[top:bottom,left:right])
    middle_x = left + right
    middle_x /= 2
    middle_y = top + bottom
    middle_y /= 2
    return middle_x,middle_y

MIN_MATCH_COUNT = 2

###### 커널 매트릭스 바꾸는 부분
kernel1 = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
kernel2 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])

sift = cv2.xfeatures2d.SURF_create()
FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 100)
flann = cv2.FlannBasedMatcher(index_params, search_params)

video_path = 'video/f4-5.MTS'
cap = cv2.VideoCapture(video_path)
origin_x,origin_y = make_ROI_IMG(cap)


img1 = cv2.imread('img.jpg',0)
img1 = cv2.resize(img1, dsize=(0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        
####### 커널 매트릭스 바꾸는 부분
# img1 = cv2.filter2D(img1, -1, kernel2)
img1 = unsharp_mask(img1)

kp1, des1 = sift.detectAndCompute(img1,None)
data = []
# 동영상을 캡처하기 시작할 때부터 시간 측정
startTime = datetime.datetime.now()
while True:
    try:
        ret, img2 = cap.read()
        if not ret: break
        # img2 = img2[300:650,200:1400]
        
        ####### 이미지 확대 2배
        img2 = cv2.resize(img2, dsize=(0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        img2 = img2[700:1400,400:2800]
        ####### 커널 매트릭스 바꾸는 부분
        # img2 = cv2.filter2D(img2, -1, kernel2)
        img2 = unsharp_mask(img2)
        
        kp2, des2 = sift.detectAndCompute(img2,None)
        matches = flann.knnMatch(des1,des2,k=2)
        good = []
        for m,n in matches:
            good.append(m)
        if len(good)>=MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            x = 0
            y = 0
            for point in dst:
                point = point[0]
                x += point[0]
                y += point[1]
            x //= len(dst)
            y //= len(dst)
            gap = origin_x - x + 400
            print(gap)
            data.append([gap])
            img2 = cv2.circle(img2,(x,y),10, (0, 0, 255), -1)
            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        else:
            print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            data.append([-1])
        if cv2.waitKey(1) == ord('q'): break
    except:
        pass
cap.release()

endTime = datetime.datetime.now()
targetTime = endTime - startTime

#####여기 바꿔야
print("m3 time :", targetTime) 

#####여기 바꿔야
with open('f4-5_sharp3.csv','w', newline='') as f:
    makewrite = csv.writer(f)
    for value in data:
        makewrite.writerow(value)