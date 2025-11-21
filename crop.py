import cv2 

img = cv2.imread('1.jpg')

crop = img[100:400, 200:500]

cv2.imshow('Cropped Image', crop)
cv2.waitKey(0)
cv2.destroyAllWindows()