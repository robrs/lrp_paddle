import cv2

class VerifyPlateType:
    def __init__(self, OCR, image):
        self.image = image
        self.ocr = OCR

    def detect_br_flag_or_letters(self):
        """
        Detecta a bandeira do Brasil (via template matching) ou as letras 'BR' (via OCR) na imagem.
        Retorna True se encontrar, False caso contrário.
        """
        # 1. Tenta localizar as letras 'BR' na parte inferior esquerda da imagem
        # Se não encontrar, tenta localizar a bandeira do Brasil
        br_template = "./assets/brasil.jpeg"  # Caminho para o template da bandeira do Brasil /assets/brasil.jpeg
        h, w = self.image.shape[:2]
        roi = self.image[0 : int(h * 0.3), :]  # região superior da placa
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        roi_bgr = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        ocr_result = self.ocr.ocr(roi_bgr, cls=True)
        # for line in ocr_result:
        #   for det in line:
        #      if det and len(det) > 1:
        #         text, conf = det[1]
        #        if "BR" in text.replace(" ", "").upper() and conf > 0.5:
        #            print("Letras 'BR' detectadas na placa.")
        #           return True

        # 2. Se fornecido, tenta localizar a bandeira via template matching
        if br_template is not None:
            template = cv2.imread(br_template)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(roi_gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.7
            loc = (res >= threshold).any()
            if loc:
                print("Bandeira do Brasil detectada na placa.")
                return True

        print("Nenhuma bandeira ou letras 'BR' detectadas.")
        return False