import cv2

src = cv2.imread("test.png", cv2.IMREAD_COLOR)

dst = src.copy() 
dst = src[300:650, 200:1400]

cv2.imwrite("cropImg9.png", dst)
cv2.destroyAllWindows()