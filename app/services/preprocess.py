import cv2

class Preprocess:

    @staticmethod
    def preprocess_image(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 11, 17, 17)
        _, thresholded = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded

    @staticmethod
    def process_image_negative(img):
        # Inverte as cores para que a fonte branca vire preta
        inverted = cv2.bitwise_not(img)
        # Ajuste de contraste e brilho para destacar a fonte
        adjusted = cv2.convertScaleAbs(inverted, alpha=1.2, beta=-40)
        #aplica blur
        #adjusted = cv2.GaussianBlur(adjusted, (5, 5), 0)
        return Preprocess.preprocess_image(adjusted)