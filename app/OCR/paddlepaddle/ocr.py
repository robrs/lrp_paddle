from paddleocr import PaddleOCR


class OCR:
    @staticmethod
    def create_ocr():
        return PaddleOCR(
            use_angle_cls=True,
            lang="pt"
        )
