from ultralytics import YOLO

model = YOLO("models/license_plate_detector.pt")

def detect_plate(frame) :
    results = model(frame, conf=0.35, verbose = False)
    detections = []
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
        
            plate_img = frame[y1:y2, x1:x2]
            detections.append({
                "img" : plate_img,
                "bbox": (x1,y1, x2, y2),
                "conf": conf
            })
            
    return detections    