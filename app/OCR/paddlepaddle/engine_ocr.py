import uuid
import re
import cv2
import numpy as np

class EngineOCR:

    def __init__(self, OCR, is_mercosul=False):
        self.is_mercosul = is_mercosul
        self.ocr = OCR

    def run(self, image):

        # Verifica se a imagem é válida
        if image is None or not isinstance(image, (np.ndarray, list)):
            print("Imagem inválida ou não carregada.")
            return ""

        import os
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../static/outputs"))
        os.makedirs(output_dir, exist_ok=True)
        file_name = str(uuid.uuid4()) + ".jpg"
        cv2.imwrite(os.path.join(output_dir, file_name), image)

        result = self.ocr.ocr(image, cls=True)
        print("Resultado do OCR:", result)
        plate_text = self.get_plates(result)
        if plate_text:
            return plate_text
        else:
            print("Nenhuma placa detectada.")
            return ""
 
    def get_plates(self, result):
        # Extrai todas as detecções do resultado
        plates = []

        for line in result:
            if line == [] or line == None:
                continue
            for det in line:
                if det == [] or det == None:
                    continue
                text, conf = det[1]
                plate_regex = r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$|^[A-Z]{3}[0-9]{4}$"
                text = (
                    text.replace(" ", "")
                    .replace("-", "")
                    .replace(".", "")
                    .replace(",", "")
                    .replace(":", "")
                )
                final_text = self.clean_plate_text(text)
                if (
                    conf >= 0.7
                    and len(final_text) >= 7
                    and re.match(plate_regex, final_text)
                ):
                    plates.append((final_text, conf))
            # Ordena por maior confiança
            plates.sort(key=lambda x: x[1], reverse=True)
            if plates:
                print("Placa(s) detectada(s):", plates)
                return plates[0][0]  # retorna a placa com maior confiança
            else:
                print("Nenhuma placa detectada com confiança suficiente.")
                return ""

   
    def clean_plate_text(self, text):

        # apenas numeros e letras
        text = re.sub(r"[^A-Z0-9]", "", text)
        # Remove espaços em branco
        text = text.replace(" ", "")

        replace_dict = {
            "0": "O",
            "1": "I",
            "2": "Z",
            "3": "E",
            "4": "A",
            "5": "S",
            "6": "G",
            "7": "T",
            "8": "B",
            "9": "P",
        }

        # os 3 primeiros caracteres devem ser letras fazer replace com o dicionario
        for k, v in replace_dict.items():
            text = re.sub(r"(?<=^.{0})" + k, v, text)

        print("Texto após limpeza:", text)

        if not self.is_mercosul:
            return text

        """
        Template Mercosul: LLLNLNN
        """
        # verifica o 5º caractere e troca por 'O' se for '0', 'I' se for '1', 'Z' se for '2'....
        # use replace_dict para substituir os caracteres na quinta posição
        for k, v in replace_dict.items():
            text = re.sub(r"(?<=^.{4})" + k, v, text)

        return text


