"""
Detección integrada HUMANO (YOLO) + FOLIO (OpenCV clásico)
sobre WEBCAM en tiempo real, y cálculo aproximado de la altura
del humano usando un folio A4 como referencia.

Este archivo incluye:
    1) Corrección de perspectiva del folio mediante su ratio geométrico (1.414).
    2) Corrección del error sistemático por pies adelantados (factor fijo ~5%).

Autor: (tu nombre)
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

PESOS_YOLO = "best.pt"


# ============================================================
#  DETECTOR DE PERSONA (YOLOv8)
# ============================================================

def detectar_persona(frame, modelo):
    """
    Devuelve:
      - bbox_persona = (x1,y1,x2,y2)  o None
      - frame con caja verde dibujada alrededor de la persona
    """
    r = modelo(frame, conf=0.55, iou=0.7)[0]

    if r.boxes is None or len(r.boxes) == 0:
        return None, frame

    x1, y1, x2, y2 = map(int, r.boxes.xyxy[0].tolist())

    frame_out = frame.copy()
    cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Verde

    return (x1, y1, x2, y2), frame_out


# ============================================================
#  DETECTOR DE FOLIO (Color + forma A4 en ROI)
# ============================================================

def detectar_folio_en_roi(roi):
    """Detector rápido y robusto para webcam usando LAB + bordes."""

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # --- Segmentación por color neutro y luminosidad ---
    mask_A = cv2.inRange(A, 115, 140)
    mask_B = cv2.inRange(B, 115, 140)
    mask_color = cv2.bitwise_and(mask_A, mask_B)

    _, mask_L = cv2.threshold(L, 180, 255, cv2.THRESH_BINARY)
    mask_folio = cv2.bitwise_and(mask_color, mask_L)

    L_folio = cv2.bitwise_and(L, L, mask=mask_folio)

    # --- Bordes + morfología ---
    edges = cv2.Canny(L_folio, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
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

        area_rect = w * h
        if area_rect < 5000:
            continue

        # Proporción del folio (ratio del lado mayor / lado menor)
        ratio = max(w, h) / min(w, h)
        if not (1.3 < ratio < 1.5):
            continue

        # Luminosidad media dentro del rectángulo
        box = cv2.boxPoints(rect).astype(np.int32)
        mask_rect = np.zeros(L.shape, dtype=np.uint8)
        cv2.drawContours(mask_rect, [box], -1, 255, -1)
        mean_L = cv2.mean(L, mask=mask_rect)[0]

        puntuacion = area_rect * (mean_L / 255.0)
        if puntuacion > mejor_puntuacion:
            mejor_puntuacion = puntuacion
            mejor_rect = rect

    return mejor_rect


# ============================================================
#  ALTURA (regla de 3 + corrección del folio + corrección pies)
# ============================================================

def calcular_altura(bbox_persona, rect_folio):
    """
    Cálculo final de la altura aplicando:

    1) Regla de 3 usando el folio como referencia (lado mayor → 29.7 cm).
    2) Corrección por perspectiva del folio mediante su ratio geométrico:
           factor_folio = R_real / R_obs  donde R_real = 1.414
    3) Corrección por pies adelantados usando un factor fijo (~5%):
           altura_final = altura_corr_folio * 1.05

    No dependemos de la distancia cámara-persona.
    """

    x1, y1, x2, y2 = bbox_persona
    altura_px = y2 - y1

    (_, _), (w, h), angle = rect_folio
    folio_lado_mayor = max(w, h)
    folio_lado_menor = min(w, h)

    if folio_lado_mayor == 0 or folio_lado_menor == 0:
        return None

    # -------------------------------
    # 1) Regla de 3 básica
    # -------------------------------
    altura_base_cm = (altura_px / folio_lado_mayor) * 29.7

    # -------------------------------
    # 2) Corrección por perspectiva A4
    # -------------------------------
    R_real = 1.414
    R_obs = folio_lado_mayor / folio_lado_menor
    factor_folio = R_real / R_obs

    altura_corr_folio = altura_base_cm * factor_folio

    # -------------------------------
    # 3) Corrección por pies adelantados (factor fijo ~5%)
    # -------------------------------
    # El cálculo de altura basado en el folio asume que todo el cuerpo
    # (torso, caderas, rodillas y pies) está en el mismo plano perpendicular
    # a la cámara. En la práctica, eso no ocurre: la punta del pie está
    # adelantada respecto al plano del torso entre 10 cm y 25 cm según la
    # antropometría humana estándar.
    #
    # Esta diferencia de profundidad provoca que la parte inferior del cuerpo
    # se proyecte ligeramente más grande que la superior, lo que genera una
    # subestimación sistemática de la altura real cuando se usa el folio como
    # referencia de escala. Este efecto de perspectiva produce un error relativo
    # que, para distancias cámara–persona típicas (1.5 m a 4 m), se mantiene
    # consistentemente en un rango de aproximadamente 3% a 8%.
    #
    # En lugar de utilizar un modelo dependiente de la distancia Z, que
    # requeriría introducir calibración adicional o medir la distancia real,
    # se adopta un factor fijo del 5%:
    #
    #       FACTOR_PIES ≈ 1.05
    #
    # Este valor corresponde al punto medio del intervalo fisiológicamente
    # habitual (3–8%), y minimiza el error medio para la mayoría de usuarios
    # independientemente de la distancia exacta a la cámara. De esta forma,
    # se corrige el sesgo sistemático sin añadir complejidad adicional al
    # sistema ni requerir información externa.
    #
    FACTOR_PIES = 1.05
    altura_final = altura_corr_folio * FACTOR_PIES

    return altura_final


# ============================================================
#  WEBCAM
# ============================================================

def main():
    print("Cargando YOLO...")
    modelo = YOLO(PESOS_YOLO)

    alturas_recent = deque(maxlen=5)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Deteccion Humano + Folio + Altura"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Webcam iniciada.")

    contador_capturas = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_out = frame.copy()
        h_frame, w_frame = frame.shape[:2]

        # ------------------------------
        # 1) Detectar persona
        # ------------------------------
        bbox, frame_out = detectar_persona(frame_out, modelo)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            ancho_persona = x2 - x1

            # ROI ampliado sólo en horizontal
            margen_horizontal = int(ancho_persona * 0.4)

            roi_x1 = max(0, x1 - margen_horizontal)
            roi_y1 = y1
            roi_x2 = min(w_frame, x2 + margen_horizontal)
            roi_y2 = y2

            roi_expandido = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            cv2.rectangle(frame_out, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)

            # ------------------------------
            # 2) Detectar folio en ROI
            # ------------------------------
            rect_folio = detectar_folio_en_roi(roi_expandido)

            if rect_folio is not None:
                box = cv2.boxPoints(rect_folio).astype(np.int32)
                box[:, 0] += roi_x1
                box[:, 1] += roi_y1
                cv2.drawContours(frame_out, [box], 0, (0, 0, 255), 3)

                # ------------------------------
                # 3) Calcular altura
                # ------------------------------
                altura_cm = calcular_altura(bbox, rect_folio)

                if altura_cm is not None:
                    alturas_recent.append(altura_cm)

                    cv2.putText(
                        frame_out,
                        f"Altura: {altura_cm:.1f} cm",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 255),
                        3
                    )

                    if len(alturas_recent) > 0:
                        altura_mediana = np.median(list(alturas_recent))
                        cv2.putText(
                            frame_out,
                            f"Altura estimada: {altura_mediana:.1f} cm",
                            (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2
                        )

        cv2.imshow(window_name, frame_out)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
