import cv2
import easyocr

reader = easyocr.Reader(['en'])

img = cv2.imread('1.jpg')

if img is None:
    print("Lỗi không nhận dạng được biển số xe")
else :
    res = reader.readtext(img)
    for bbox, text, prob in res :
        print(text, prob)