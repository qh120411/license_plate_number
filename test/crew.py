import cv2 

img = cv2.imread('1.jpg')

cv2.putText(img, 'Hello', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

cv2.rectangle(img, (100,100), (300, 300), (255,0,0), 2)

cv2.imshow('Drew',img)
cv2.waitKey(0)