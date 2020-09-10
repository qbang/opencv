import numpy as np
import cv2
import math
import csv
from matplotlib import pyplot as plt
import datetime

img2 = cv2.imread('img1.png')
img2 = cv2.bilateralFilter(img2,9,75,75)
cv2.imwrite("img11.png", img2)