import cv2
import easyocr
import re

ALLOWED = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
reader = easyocr.Reader(['en'])
REGEX = "^[0-9]{2}[A-Z]{2}[0-9]{5,6}$"

def clean_text(text: str) -> str:
    text = text.upper().replace(" ", "").replace("-", "")
    text = "".join(ch for ch in text if ch in ALLOWED)
    return text

def preprocess(img) :
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    return thresh

def ocr_plate(img) :
    pre = preprocess(img)
    res = reader.readtext(pre)
    
    candidates = []
    for _, text, conf in res : 
        t = clean_text(text)
        if ALLOWED and len(t) >= 6 :
            candidates.append((t, float(conf)))
    
    if not candidates :
        return None
    
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    
    
    best = candidates[0]
    return best 