"""
Detección integrada HUMANO (YOLO) + FOLIO (OpenCV clásico)
sobre WEBCAM en tiempo real, y cálculo aproximado de la altura
del humano usando un folio A4 como referencia.

Salida:
    - Ventana en tiempo real con:
         * Caja verde = humano detectado
         * Caja azul = folio detectado
         * Texto con altura estimada
    - Terminal imprime la altura estimada ocasionalmente.

Uso:
    python detector-altura-webcam.py

Requisitos:
    - ultralytics (YOLOv8)
    - opencv-python
    - numpy
    - best.pt en el mismo directorio
"""

import cv2
import numpy as np
from ultralytics import YOLO

PESOS_YOLO = "best.pt"

# ============================================================
#  DETECTOR DE PERSONA (YOLO)
# ============================================================

def detectar_persona(frame, modelo):
    """
    Devuelve:
      - bbox_persona = (x1,y1,x2,y2)  o None
      - frame con caja dibujada
    """
    r = modelo(frame, conf=0.55, iou=0.7)[0]

    if r.boxes is None or len(r.boxes) == 0:
        return None, frame

    # Tomamos la primera persona
    x1, y1, x2, y2 = map(int, r.boxes.xyxy[0].tolist())

    frame_out = frame.copy()
    cv2.rectangle(frame_out, (x1,y1), (x2,y2), (0,255,0), 3)

    return (x1, y1, x2, y2), frame_out


# ============================================================
#  DETECTOR DE FOLIO (Color + forma A4)
# ============================================================

def detectar_folio_en_roi(roi):
    """Versión optimizada para webcam: rápido y robusto."""

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Color blanco neutro
    mask_A = cv2.inRange(A, 120, 136)
    mask_B = cv2.inRange(B, 120, 136)
    mask_color = cv2.bitwise_and(mask_A, mask_B)

    # Luminosidad alta
    _, mask_L = cv2.threshold(L, 180, 255, cv2.THRESH_BINARY)

    # Candidato a folio
    mask_folio = cv2.bitwise_and(mask_color, mask_L)

    # Bordes solo donde hay folio
    L_folio = cv2.bitwise_and(L, L, mask=mask_folio)
    edges = cv2.Canny(L_folio, 50, 150)

    # Morfología: unir bordes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mejor_rect = None
    mejor_puntuacion = 0

    for c in contornos:
        if len(c) < 5:
            continue

        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect

        if w == 0 or h == 0:
            continue

        area_rect = w*h
        if area_rect < 5000:
            continue

        # Ratio A4
        ratio = max(w,h) / min(w,h)
        if not (1.3 < ratio < 1.5):
            continue

        # Media de L interna
        box = cv2.boxPoints(rect).astype(np.int32)
        mask_rect = np.zeros(L.shape, dtype=np.uint8)
        cv2.drawContours(mask_rect, [box], -1, 255, -1)
        mean_L = cv2.mean(L, mask=mask_rect)[0]

        puntuacion = area_rect * (mean_L/255.0)

        if puntuacion > mejor_puntuacion:
            mejor_puntuacion = puntuacion
            mejor_rect = rect

    return mejor_rect


# ============================================================
#  ALTURA (regla de 3)
# ============================================================

def calcular_altura(bbox_persona, rect_folio):
    x1,y1,x2,y2 = bbox_persona
    altura_px = y2 - y1

    (_, _), (w,h), _ = rect_folio
    folio_px = max(w,h)

    if folio_px == 0:
        return None

    # Lado largo A4 = 29.7 cm
    altura_cm = (altura_px / folio_px) * 29.7
    return altura_cm


# ============================================================
#  WEBCAM
# ============================================================

def main():
    print("Cargando YOLO...")
    modelo = YOLO(PESOS_YOLO)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webcam iniciada. Pulsa 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturando frame.")
            break

        frame_out = frame.copy()

        # -------------------------------------------------------
        # 1) Detectar persona
        # -------------------------------------------------------
        bbox, frame_out = detectar_persona(frame_out, modelo)

        if bbox is not None:
            x1,y1,x2,y2 = bbox
            roi = frame[y1:y2, x1:x2]

            # -------------------------------------------------------
            # 2) Detectar folio
            # -------------------------------------------------------
            rect_folio = detectar_folio_en_roi(roi)

            if rect_folio is not None:
                box = cv2.boxPoints(rect_folio).astype(np.int32)
                box[:,0] += x1
                box[:,1] += y1

                cv2.drawContours(frame_out, [box], 0, (255,0,0), 3)

                # -------------------------------------------------------
                # 3) Calcular altura
                # -------------------------------------------------------
                altura_cm = calcular_altura(bbox, rect_folio)
                if altura_cm is not None:
                    cv2.putText(
                        frame_out,
                        f"Altura: {altura_cm:.1f} cm",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 255),
                        3
                    )

                    print(f"Altura estimada: {altura_cm:.1f} cm")

        # Mostrar frame final
        cv2.imshow("Detección Humano + Folio + Altura (Webcam)", frame_out)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
