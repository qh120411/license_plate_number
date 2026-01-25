from ultralytics import YOLO

model = YOLO("models/license_plate_detector.pt")

def detect_plate(frame, conf = 0.35) :
    res = model(frame, conf = conf, verbose = False )
    
    plates = []
    
    for r in res :
        for box in r.boxes :
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plates.append((x1,y1,x2,y2))
            
    return plates

