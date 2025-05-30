import cv2
import os
import uuid


def crop_with_margin(image, box, margin=0.1):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)
    x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
    x2, y2 = min(w, x2 + dx), min(h, y2 + dy)
    return image[y1:y2, x1:x2]


def save_image(image, prefix=""):
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../static/outputs")
    )
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{prefix}{str(uuid.uuid4())}.jpg"
    cv2.imwrite(os.path.join(output_dir, file_name), image)
