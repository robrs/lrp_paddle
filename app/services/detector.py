from ultralytics import YOLO
import cv2
import uuid

class Detector:
    def __init__(self, model_path="license_plate_detector.pt"):
        self.model = YOLO(model_path)

    def detect_plate(self, frame):
        results = self.model(frame)
        detections = results[0].boxes.xyxy.cpu().numpy()
        if len(detections) > 0:
            frame_with_boxes = frame.copy()
            for box in detections:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            file_name = f"frame_{uuid.uuid4()}.jpg"
            cv2.imwrite(f"plates-detected/{file_name}", frame_with_boxes)
            return detections  # lista de [x1, y1, x2, y2]
        return None
    
    def detect_vehicle(self, frame):
        # salva o frame original
        file_name = f"frame_{uuid.uuid4()}.jpg"
        cv2.imwrite(f"frames/{file_name}", frame)
        results = YOLO("yolov8n.pt")(frame, conf=0.5)
        print("Resultados da detecção de veículos:", results)
        detections = []
        for box, cls in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.cls.cpu().numpy()):
            # Supondo que 0 = carro, 1 = moto (ajuste conforme seu modelo)
            label = "carro" if int(cls) == 0 else "moto"
            detections.append({"box": box, "label": label, "cls": cls})
        return detections
