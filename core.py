import cv2
import math

atan = lambda value : math.degrees(math.atan(value))

fovH = lambda angle : 2 * atan((3/(13**0.5))*tan(angle/2))
fovV = lambda angle : 2 * atan((2/(13**0.5))*tan(angle/2))

fov = lambda angle : [fovH(angle),fovV(angle)]

tan = lambda angle : math.tan(math.radians(angle))
M = 1000

from matplotlib import pyplot as plt
import matplotlib as mpl

# OPENCV_OBJECT_TRACKERS = {
#     "csrt": cv2.TrackerCSRT_create,
#     "kcf": cv2.TrackerKCF_create,
#     "boosting": cv2.TrackerBoosting_create,
#     "mil": cv2.TrackerMIL_create,
#     "tld": cv2.TrackerTLD_create,
#     "medianflow": cv2.TrackerMedianFlow_create,
#     "mosse": cv2.TrackerMOSSE_create
# }

def cv_init(cap):
    frame_count = 1
    ret, img = cap.read()
    rect = cv2.selectROI("select ROI",img)
    tracker = OPENCV_OBJECT_TRACKERS['csrt']()
    tracker.init(img,rect)
    x = 0
    y = 0
    for _ in range(frame_count):
        _, box = tracker.update(img)
        left,top,w,h = [int(v) for v in box]
        right = left + w
        bottom = top + h
        x_middle = (left + right) // 2
        y_middle = (top + bottom) // 2
        x += x_middle
        y += y_middle
    return tracker,x//frame_count,y//frame_count


def make_ROI_IMG(cap):
    ret, img = cap.read()
    rect = cv2.selectROI("select ROI",img)
    left,top,w,h = [int(v) for v in rect]
    right = left + w
    bottom = top + h
    cv2.imwrite("img.jpg",img[top:bottom,left:right])
    middle_x = left + right
    middle_x /= 2
    middle_y = top + bottom
    middle_y /= 2
    print("roi ok")
    return middle_x,middle_y

def mmPerPx(w,h,fov_x,fov_y,rate,distance = 10):
    distance *= M

    x = 2 * tan(fov_x / 2) * distance / w / rate
    y = 2 * tan(fov_y / 2) * distance / h / rate

    return x,y

def x_to_mm(gap_x):
    w = 1920
    h = 1080
    fov_value = 75
    distance = 0.549 - 0.054 - 0.075

    fov_x = fov(fov_value)[0]
    fov_y = fov(fov_value)[1]
    rate = 1
    x, y = mmPerPx(w, h, fov_x, fov_y, rate, distance)
    return x * gap_x