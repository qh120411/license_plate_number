import cv2
import easyocr

reader = easyocr.Reader(['en'])
img = cv2.imread('image/2.jpg')

res = reader.readtext(img)

for bbox, text, prob in res :
    print(text, prob)
    
