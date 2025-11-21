import cv2

jpg = cv2.imread('1.jpg')

cv2.imshow('image', jpg)
cv2.waitKey(0) # Đợi nhấn phím bất kỳ để đóng cửa sổ
cv2.destroyAllWindows()