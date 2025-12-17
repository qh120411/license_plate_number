import easyocr
import cv2
import re

reader = easyocr.Reader(['en'])
ALLOWED = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

def preprocess_image(frame) :
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold( blur, 255 ,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2)
    return gray, blur, thresh

def normalize_bsx(text):
    text = text.replace(" ", "").upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

def valid_bsx(text):
    return ( re.match(r'\d{2}[A-M]\d{5,6}$', text) or
            re.match(r'\d{2}[A-M]{2}\d{5}', text)
    )

def read_plate(plate_img, reader) :
    
    gray, blur, thresh = preprocess_image(plate_img)
    
    candidates = [] #tao 1 mang chua cac phuong an ma he thong doc duoc
    for img in (gray, blur, thresh) : 
        res = reader.readtext(img, allowlist=ALLOWED)
        
    for (_, text, conf, ) in res :
        text = normalize_bsx(text)
        
        if valid_bsx(text) :
            candidates.append(text, conf)
            
    if not candidates :
        return None
    
    best_text, best_conf = max(candidates, key=lambda x: x[1])
    return best_text