import cv2
import easyocr
import re

img = cv2.imread('image/2.jpg')
reader = easyocr.Reader(['en'])
res = reader.readtext(img)

res_sort = sorted(res, key = lambda x: x[0][0][1])

texts = [text.replace(" ", "") for bbox, text, prob in res_sort]
plate_raw = ''.join(texts)

plate = re.sub(r'[^A-Za-z0-9]', '', plate_raw)

print("Biển số xe nhận dạng được:", plate)