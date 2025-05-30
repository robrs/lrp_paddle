import cv2
import numpy as np
from app.services.detector import Detector
from utils.image_utils import crop_with_margin, save_image
from app.OCR.paddlepaddle.ocr import OCR
from app.OCR.paddlepaddle.veriry_plate_type import VerifyPlateType
from app.OCR.paddlepaddle.engine_ocr import EngineOCR


class ProcessImage:
    def __init__(self, image):
        self.image = image
        self.ocr = OCR.create_ocr()

    def exec(self):
        detector = Detector()

        npimg = np.frombuffer(self.image.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        plates = detector.detect_plate(frame)
        results = []
        if plates is not None and len(plates) > 0:
            for box in plates:
                img_cropped = crop_with_margin(frame, box)
                save_image(img_cropped, 'image_cropped-')
                gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                #if laplacian_var < 80:
                    #print("Imagem desfocada, ignorando...")
                    #continue

                verify_plate_type = VerifyPlateType(self.ocr, img_cropped)
                is_mercosul = verify_plate_type.detect_br_flag_or_letters()

                engine_ocr = EngineOCR(self.ocr, is_mercosul)
                text = engine_ocr.run(img_cropped)
                if text:
                    results.append({"plate": text})

        return results
