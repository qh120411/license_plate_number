import os
from ultralytics import YOLO


class PlateDetector:
    def __init__(self, model_path=None, conf=0.35):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "..", "best.pt")
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        plates = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = frame[y1:y2, x1:x2]
                plates.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(box.conf[0]),
                    "plate_img": plate_img,
                })

        return plates
