import cv2
import easyocr
import re


class PlateOCR:
    """EasyOCR-based Vietnamese license plate reader."""

    # Ký tự hợp lệ trên biển số VN
    ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    # Regex biển số VN:
    #   - 2 số tỉnh + 1-2 chữ seri + (có thể 1 số seri) + 4-5 số thân
    #   - Ô tô 1 dòng: 51F97022 (8-9 ký tự)
    #   - Xe máy 2 dòng: 99E122268 (9 ký tự)
    PLATE_PATTERNS = [
        r"^[0-9]{2}[A-Z]{1,2}[0-9]{4,6}$",   # 51F97022, 30A12345
        r"^[0-9]{2}[A-Z][0-9]{1}[0-9]{4,5}$", # 99E122268 (seri có số)
    ]

    def __init__(self, langs=None):
        if langs is None:
            langs = ["en"]
        self.reader = easyocr.Reader(langs, gpu=False)

    # ------------------------------------------------------------------ #
    #  Tiền xử lý ảnh
    # ------------------------------------------------------------------ #
    @staticmethod
    def preprocess(img):
        """Chuyển ảnh biển số sang binary để OCR đọc tốt hơn."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize nếu ảnh quá nhỏ — giúp OCR chính xác hơn
        h, w = gray.shape
        if w < 200:
            scale = 200 / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        return thresh

    # ------------------------------------------------------------------ #
    #  Làm sạch & validate text
    # ------------------------------------------------------------------ #
    @classmethod
    def clean_text(cls, text: str) -> str | None:
        """Chuẩn hóa text OCR → biển số VN hoặc None."""
        text = text.upper()
        # Loại bỏ các ký tự phân cách phổ biến trên biển VN
        for ch in (" ", "-", ".", "·"):
            text = text.replace(ch, "")
        # Chỉ giữ ký tự hợp lệ
        text = "".join(ch for ch in text if ch in cls.ALLOWED)
        # Validate theo pattern
        if cls.is_valid_plate(text):
            return text
        return None

    @classmethod
    def is_valid_plate(cls, text: str) -> bool:
        """Kiểm tra text có khớp format biển số VN không."""
        return any(re.match(p, text) for p in cls.PLATE_PATTERNS)

    # ------------------------------------------------------------------ #
    #  Đọc biển số
    # ------------------------------------------------------------------ #
    def read_plate(self, img) -> dict | None:
        """
        Đọc biển số từ ảnh crop biển.

        Returns: dict với:
            - text: str (biển số đã chuẩn hóa, VD: "51F97022")
            - confidence: float (0-1)
        Hoặc None nếu không đọc được.
        """
        pre = self.preprocess(img)
        results = self.reader.readtext(pre)

        if not results:
            return None

        # --- Chiến lược 1: Ghép tất cả dòng lại (biển 2 dòng xe máy) ---
        all_texts = []
        total_conf = 0.0
        for _, text, conf in results:
            all_texts.append(text)
            total_conf += conf

        merged = "".join(all_texts)
        merged_clean = self.clean_text(merged)
        if merged_clean:
            avg_conf = total_conf / len(all_texts)
            return {"text": merged_clean, "confidence": round(avg_conf, 4)}

        # --- Chiến lược 2: Thử từng dòng riêng (biển 1 dòng ô tô) ---
        candidates = []
        for _, text, conf in results:
            t = self.clean_text(text)
            if t:
                candidates.append({"text": t, "confidence": round(float(conf), 4)})

        if candidates:
            # Trả về candidate có confidence cao nhất
            candidates.sort(key=lambda x: x["confidence"], reverse=True)
            return candidates[0]

        return None