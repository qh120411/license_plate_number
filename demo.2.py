import cv2
import easyocr
import re
import datetime


reader = easyocr.Reader(['en'])

def preprocess_image(frame) :
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    return blur

def read_plate(frame) :
    res = reader.readtext(frame)
    res_sort = sorted(res, key = lambda x : x[0][0][1])
    
    text = [text.replace(" ", "") for bbox, text, prob in res_sort]
    plate_raw = "".join(text)
    plate = re.sub(r'[^A-Za-z0-9]', '', plate_raw)
    return plate

cap = cv2.VideoCapture(0)
count = 0
last_plate = ""
plate = ""

while True :
    ret, frame = cap.read()
    if not ret : 
        break
    count += 1
    
    if ( count % 60 == 0 ) :
        preprocess = preprocess_image(frame)
        plate = read_plate(preprocess)
    
    if len(plate) == 9 and plate != "" and plate != last_plate:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        last_plate = plate
        last_time = now
        
        print("Xe nhận diện được biển số: ", plate)
        print("Thời gian nhận diện: ", now)
        
    cv2.imshow('Webcam', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break
    
cap.release()
cv2.destroyAllWindows()
        
        
    
    