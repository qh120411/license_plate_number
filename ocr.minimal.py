import cv2
import easyocr

reader = easyocr.Reader(['en'])
img = cv2.imread('image/1.jpg')

res = reader.readtext(img)

print(res)

