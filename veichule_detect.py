import cv2
from ultralytics import YOLO

# Carregue o modelo YOLOv8 pré-treinado
model = YOLO('yolov8n.pt')  # Use o caminho do seu modelo

# Abra o vídeo (0 para webcam, ou caminho do arquivo)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realize a inferência
    results = model(frame)[0]

    veiculo_detectado = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label in ["car", "bus", "motorcycle", "truck"]:
            veiculo_detectado = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Salve o frame para futura detecção de placa
            cv2.imwrite("frame_veiculo.jpg", frame)
            print("Veículo detectado! Frame salvo para detecção de placa.")
            break  # Salva apenas o primeiro veículo detectado no frame

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
