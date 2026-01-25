import cv2
from ocr import ocr_plate
from detect import detect_plate

def run_camera() :
    url = "http://192.168.31.102:4747/video" 
    cap = cv2.VideoCapture(url)
    if not cap.isOpened() :
        print("He thong khong ket noi duoc cam")
        return
    while True : 
        ret, frame = cap.read()
        if not ret :
            print("Khong the doc frame")
            break
        
        plates = detect_plate(frame)
        for ( x1, y1, x2, y2) in plates : 
            plate_img = frame[y1:y2, x1:x2]
            res = ocr_plate(plate_img)
            if res : 
                text, conf = res
                print("", text) #print bien so
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{text} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == ord('q') :
            break
    cap.release()
    cv2.destroyAllWindows()