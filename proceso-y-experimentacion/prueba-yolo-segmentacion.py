import cv2
import numpy as np
from ultralytics import YOLO

# Modelo de segmentación
modelo = YOLO("yolov8s-seg.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("SEGMENTACION", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # INFERENCIA SIN STREAM → DEVUELVE UNA LISTA NORMAL
    results = modelo(
        frame,
        imgsz=640,
        conf=0.15,
        iou=0.6,
        retina_masks=True  # imprescindible para que no se desplace
    )

    r = results[0]

    frame_out = frame.copy()

    if r.masks is not None:
        masks = r.masks.data.cpu().numpy()    # (N,H,W)
        clases = r.boxes.cls.cpu().numpy()    # (N,)

        # Filtrar SOLO PERSON (class=0)
        idx_personas = np.where(clases == 0)[0]

        if len(idx_personas) > 0:

            # Seleccionar la persona más grande
            best_idx = max(idx_personas, key=lambda i: masks[i].sum())
            mask = masks[best_idx]

            # Máscara binaria
            mask_bin = (mask * 255).astype("uint8")

            # Colorear máscara
            color_mask = np.zeros_like(frame_out)
            color_mask[:, :, 2] = mask_bin  # canal rojo

            overlay = cv2.addWeighted(frame_out, 1.0, color_mask, 0.5, 0)

            # Bounding box exacta
            ys, xs = np.where(mask_bin > 0)
            if len(xs) > 0:
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), 3)

            frame_out = overlay

    cv2.imshow("SEGMENTACION", frame_out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
