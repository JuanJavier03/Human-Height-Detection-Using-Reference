"""
Detección integrada HUMANO (YOLO-SEG) + FOLIO (OpenCV clásico)
sobre WEBCAM en tiempo real, y cálculo aproximado de la altura
del humano usando un folio A4 como referencia.

Este archivo incluye:
    1) Corrección de perspectiva del folio mediante su ratio geométrico (1.414).
    2) Corrección por pies adelantados basada en geometría de cámara (factor ≈ 1.03).
    3) Ajuste de precisión final por suela/puntera (~ -1.0 cm).

Autor: (tu nombre)
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Modelo de segmentación
PESOS_YOLO = "yolov8s-seg.pt"


# ============================================================
#  DETECTOR DE PERSONA (SEGMENTACIÓN)
# ============================================================

def detectar_persona_segmentado(frame, modelo):
    """
    Devuelve:
      - bbox_persona = (x1, y1, x2, y2) EXACTA gracias a la máscara
      - mask binaria
      - frame con bounding dibujada
    """

    r = modelo(frame, conf=0.55, iou=0.7)[0]

    if r.masks is None or len(r.masks) == 0:
        return None, None, frame

    mask = r.masks.data[0].cpu().numpy()
    mask_bin = (mask * 255).astype("uint8")

    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None, frame

    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())

    frame_out = frame.copy()
    cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return (x1, y1, x2, y2), mask_bin, frame_out


# ============================================================
#  DETECTOR DE FOLIO
# ============================================================

def detectar_folio_en_roi(roi):

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    mask_A = cv2.inRange(A, 115, 140)
    mask_B = cv2.inRange(B, 115, 140)
    mask_color = cv2.bitwise_and(mask_A, mask_B)

    _, mask_L = cv2.threshold(L, 180, 255, cv2.THRESH_BINARY)
    mask_folio = cv2.bitwise_and(mask_color, mask_L)

    L_folio = cv2.bitwise_and(L, L, mask=mask_folio)

    edges = cv2.Canny(L_folio, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, 2)

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

        ratio = max(w, h) / min(w, h)
        if not (1.3 < ratio < 1.5):
            continue

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
# ALTURA COMPLETA (regla de 3 + perspectiva + pies + suela)
# ============================================================

def calcular_altura(bbox_persona, rect_folio):
    """
    Correcciones aplicadas:

    1) Regla de 3 usando el lado mayor del folio → 29.7 cm.

    2) Corrección por perspectiva del A4:
         factor_folio = R_real / R_obs
         Corrige compresión del folio cuando está inclinado.

    3) CORRECCIÓN POR PIES ADELANTADOS (factor geométrico calculado):
         Derivación:
            La punta del pie está adelantada una distancia L respecto al torso.
            El folio está en el plano del torso (distancia Z a la cámara).
            La punta del pie está en Z_pie = Z - L.
            En pinhole:
                 H_medido ∝ 1/Z_pie
                 H_real   ∝ 1/Z
            Por tanto:
                 factor = H_medido / H_real = Z / (Z - L)

         Valores típicos:
            L ≈ 0.15 m (adelanto promedio del pie humano)
            Z entre 2 y 4 m
            factor entre 1.04 y 1.065

         El usuario puede modificar Z desde esta función.

    4) Ajuste por suela/puntera (~1 cm de media)
    """

    # ----------------------------------------------
    # 1) Altura medida directamente de segmentación
    # ----------------------------------------------
    x1, y1, x2, y2 = bbox_persona
    altura_px = y2 - y1

    # ----------------------------------------------
    # 2) Dimensiones del folio detectado
    # ----------------------------------------------
    (_, _), (w, h), angle = rect_folio
    folio_lado_mayor = max(w, h)
    folio_lado_menor = min(w, h)

    if folio_lado_mayor == 0 or folio_lado_menor == 0:
        return None

    # ----------------------------------------------
    # 3) Regla de 3
    # ----------------------------------------------
    altura_base_cm = (altura_px / folio_lado_mayor) * 29.7

    # ----------------------------------------------
    # 4) Corrección por perspectiva del folio
    # ----------------------------------------------
    R_real = 1.414
    R_obs = folio_lado_mayor / folio_lado_menor
    factor_folio = R_real / R_obs
    altura_corr_folio = altura_base_cm * factor_folio

    # ----------------------------------------------
    # 5) CORRECCIÓN por pies adelantados (factor físico)
    # ----------------------------------------------
    Z = 3.0  # metros (distancia estimada cámara - torso)
    L = 0.15  # metros (adelanto medio del pie)
    if Z <= L:
        Z = L + 0.01   # seguridad matemática

    factor_pies = Z / (Z - L)
    altura_corr_pies = altura_corr_folio * factor_pies

    # ----------------------------------------------
    # 6) Ajuste por suela / puntera (+1 cm de calzado)
    # ----------------------------------------------
    ajuste_suela = 1.0
    altura_final = altura_corr_pies - ajuste_suela

    return altura_final



# ============================================================
# WEBCAM
# ============================================================

def main():

    print("Cargando YOLO (segmentación)...")
    modelo = YOLO(PESOS_YOLO)

    alturas_recent = deque(maxlen=5)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Detección Humano (SEG) + Folio + Altura"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Webcam iniciada.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_out = frame.copy()
        h_frame, w_frame = frame.shape[:2]

        # 1) Persona (segmentación)
        bbox, mask_persona, frame_out = detectar_persona_segmentado(frame_out, modelo)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            ancho_persona = x2 - x1

            margen_horizontal = int(ancho_persona * 0.4)

            roi_x1 = max(0, x1 - margen_horizontal)
            roi_y1 = y1
            roi_x2 = min(w_frame, x2 + margen_horizontal)
            roi_y2 = y2

            roi_expandido = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            cv2.rectangle(frame_out, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)

            # 2) Folio
            rect_folio = detectar_folio_en_roi(roi_expandido)

            if rect_folio is not None:
                box = cv2.boxPoints(rect_folio).astype(np.int32)
                box[:, 0] += roi_x1
                box[:, 1] += roi_y1
                cv2.drawContours(frame_out, [box], 0, (0, 0, 255), 3)

                # 3) Altura
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
