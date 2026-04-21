import cv2
import csv
import os
from datetime import datetime
from detect import PlateDetector
from ocr import PlateOCR


CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "scan_history.csv")
PLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "plates")
CSV_HEADERS = ["plate", "ocr_confidence", "yolo_confidence", "time", "image_path"]


class CameraStream:

    def __init__(self, source=0, skip_frames=3):
        self.source = source
        self.skip_frames = skip_frames
        self.detector = PlateDetector()
        self.ocr = PlateOCR()
        self.best_records = {}
        os.makedirs(PLATES_DIR, exist_ok=True)
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(CSV_PATH):
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writeheader()

    def _save_plate_image(self, plate_img, text: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{text}_{timestamp}.jpg"
        filepath = os.path.join(PLATES_DIR, filename)
        cv2.imwrite(filepath, plate_img)
        return os.path.join("plates", filename)

    def _save_to_csv(self, record: dict):
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writerow(record)

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print("[ERROR] Khong the ket noi camera:", self.source)
            return
        
        print("[INFO] Camera da ket noi. Nhan 'q' de thoat.")
        print(f"[INFO] Ket qua luu tai: {os.path.abspath(CSV_PATH)}")
        print(f"[INFO] Anh bien so luu tai: {os.path.abspath(PLATES_DIR)}")
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Khong the doc frame, thu ket noi lai...")
                cap.release()
                cap = cv2.VideoCapture(self.source)
                continue

            frame_count += 1

            #Skip frame để giảm tải — chỉ detect mỗi N frame
            if frame_count % self.skip_frames != 0:
                cv2.imshow("License Plate Detection", frame)
                if cv2.waitKey(1) == ord("q"):
                    break
                continue
            
            plates = self.detector.detect(frame)

            for plate in plates:
                x1, y1, x2, y2 = plate["bbox"]
                # OCR đọc biển
                result = self.ocr.read_plate(plate["plate_img"])
                
                if result:
                    text = result["text"]
                    conf = result["confidence"]
                    # Chỉ lưu nếu: biển mới HOẶC confidence cao hơn lần trước
                    prev = self.best_records.get(text)
                    if prev is None:
                        # Biển mới → lưu ảnh + ghi CSV
                        img_path = self._save_plate_image(plate["plate_img"], text)
                        record = {
                            "plate": text,
                            "ocr_confidence": round(conf, 4),
                            "yolo_confidence": round(plate["confidence"], 4),
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "image_path": img_path,
                        }
                        self.best_records[text] = record
                        self._save_to_csv(record)
                        print(f"[NEW]   {text}  (OCR: {conf:.2f}, YOLO: {plate['confidence']:.2f})  -> {img_path}")

                    elif conf > prev["ocr_confidence"]:
                        # Confidence cao hơn → xoá ảnh cũ, lưu ảnh mới
                        old_path = os.path.join(os.path.dirname(__file__), "..", prev["image_path"])
                        if os.path.exists(old_path):
                            os.remove(old_path)

                        img_path = self._save_plate_image(plate["plate_img"], text)
                        record = {
                            "plate": text,
                            "ocr_confidence": round(conf, 4),
                            "yolo_confidence": round(plate["confidence"], 4),
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "image_path": img_path,
                        }
                        self.best_records[text] = record
                        self._rewrite_csv()
                        print(f"[UPDATE] {text}  conf {prev['ocr_confidence']:.2f} -> {conf:.2f}  -> {img_path}")

                    # Vẽ lên frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{text} ({conf:.0%})"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.imshow("License Plate Detection", frame)
            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[INFO] Ket thuc. Tong so bien da quet: {len(self.best_records)}")
        print(f"[INFO] File CSV: {os.path.abspath(CSV_PATH)}")
        print(f"[INFO] Anh crop: {os.path.abspath(PLATES_DIR)}")

    def _rewrite_csv(self):
        """Ghi đè toàn bộ CSV với best_records hiện tại."""
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            for record in self.best_records.values():
                writer.writerow(record)

    def get_history(self) -> list[dict]:
        """Trả về danh sách biển đã quét (mỗi biển 1 record tốt nhất)."""
        return list(self.best_records.values())