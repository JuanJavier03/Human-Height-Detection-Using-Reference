"""
Detección integrada HUMANO (YOLO-SEG) + FOLIO (OpenCV clásico)
sobre WEBCAM en tiempo real, y cálculo aproximado de la altura
del humano usando un folio A4 como referencia.

Incluye:
    - Segmentación precisa de persona (bounding box real derivada de máscara)
    - Detección del folio dentro de la bounding box de la persona
    - Corrección de perspectiva del folio (ratio geométrico 1.414)
    - Corrección geométrica por adelantamiento del pie (factor = Z/(Z-L))
    - Ajuste final por suela/calzado

Este fichero está completamente parametrizado:
Desde la terminal puedes indicar:
    --Z      → distancia cámara-torso (en metros)
    --L      → adelantamiento de la punta del pie respecto al torso (en metros)
    --suela  → corrección por calzado en cm (positivo o negativo)

Ejemplo:
    python detector-altura-webcam.py --Z 2.15 --L 0.10 --suela 1.2
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import argparse


# ============================================================
# PARÁMETROS POR TERMINAL
# ============================================================

def get_arguments():
    """
    Parser de argumentos.
    Permite modificar parámetros geométricos sin tocar el código.
    """
    parser = argparse.ArgumentParser(
        description="Medidor de altura con YOLO-SEG + folio parametrizable."
    )

    parser.add_argument(
        "--Z",
        type=float,
        required=True,
        help="Distancia cámara-torso en metros."
    )

    parser.add_argument(
        "--L",
        type=float,
        default=0.15,
        help="Diferencia en profundidad entre torso y punta del pie (en metros)."
    )

    parser.add_argument(
        "--suela",
        type=float,
        default=1.0,
        help="Corrección por calzado en cm (default: 1.0 cm)."
    )

    return parser.parse_args()



# ============================================================
# CONFIGURACIÓN
# ============================================================

PESOS_YOLO = "yolov8s-seg.pt"



# ============================================================
# DETECTOR DE PERSONA POR SEGMENTACIÓN
# ============================================================

def detectar_persona_segmentado(frame, modelo):
    """
    Segmentación de persona mediante YOLO-Seg.
    Extrae bounding box REAL desde la máscara.
    """
    results = modelo(
        frame,
        imgsz=640,
        conf=0.15,
        iou=0.6,
        retina_masks=True
    )

    r = results[0]

    if r.masks is None or len(r.masks) == 0:
        return None, None

    masks = r.masks.data.cpu().numpy()
    clases = r.boxes.cls.cpu().numpy()

    # Filtrar SOLO personas (class=0)
    idx = np.where(clases == 0)[0]
    if len(idx) == 0:
        return None, None

    # Seleccionar la máscara más grande
    best = max(idx, key=lambda i: masks[i].sum())
    mask = masks[best]

    mask_bin = (mask * 255).astype("uint8")
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return None, None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    return (x1, y1, x2, y2), mask_bin



# ============================================================
# DETECTOR DE FOLIO
# ============================================================

def detectar_folio_en_roi(roi):
    """
    Detecta un folio A4 dentro de un ROI usando LAB + Canny + análisis geométrico.
    Devuelve rectángulo mínimo (cv2.minAreaRect).
    """

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Umbrales robustos para color blanco del folio
    mask_A = cv2.inRange(A, 115, 140)
    mask_B = cv2.inRange(B, 115, 140)
    mask_color = cv2.bitwise_and(mask_A, mask_B)

    # Umbral en canal L (luminosidad)
    _, mask_L = cv2.threshold(L, 180, 255, cv2.THRESH_BINARY)

    # Intersección de criterios de color
    mask_folio = cv2.bitwise_and(mask_color, mask_L)

    # Extraer solo píxeles blancos
    L_folio = cv2.bitwise_and(L, L, mask=mask_folio)

    # Bordes Canny + dilatación
    edges = cv2.Canny(L_folio, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, 2)

    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mejor_rect = None
    mejor_score = 0

    for c in contornos:
        if len(c) < 5:
            continue

        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect

        # Filtro geométrico mínimo
        if w == 0 or h == 0 or w * h < 5000:
            continue

        ratio = max(w, h) / min(w, h)

        # Ratio de folio A4 = 1.414
        if not (1.3 < ratio < 1.5):
            continue

        # Puntuación basada en luminosidad y tamaño
        box = cv2.boxPoints(rect).astype(np.int32)
        mask_rect = np.zeros(L.shape, dtype=np.uint8)
        cv2.drawContours(mask_rect, [box], -1, 255, -1)
        mean_L = cv2.mean(L, mask=mask_rect)[0]

        score = (w * h) * (mean_L / 255.0)

        if score > mejor_score:
            mejor_score = score
            mejor_rect = rect

    return mejor_rect



# ============================================================
# CÁLCULO FINAL DE ALTURA (PARAMETRIZADO)
# ============================================================

def calcular_altura(bbox_persona, rect_folio, Z, L, suela_cm):
    """
    Cálculo profesional de altura:
        1) Regla de 3 con el folio A4 (lado mayor 29.7 cm)
        2) Corrección por perspectiva (ratio folio)
        3) Corrección geométrica por pie adelantado Z/(Z-L)
        4) Ajuste final de suela en cm

    Devuelve altura en cm.
    """

    x1, y1, x2, y2 = bbox_persona
    altura_px = y2 - y1

    (_, _), (w, h), _ = rect_folio
    mayor = max(w, h)
    menor = min(w, h)

    if mayor == 0 or menor == 0:
        return None

    # 1) Regla de tres usando lado mayor de folio (29.7 cm)
    altura_base_cm = (altura_px / mayor) * 29.7

    # 2) Corrección por perspectiva del folio
    R_real = 1.414
    R_obs = mayor / menor
    factor_folio = R_real / R_obs
    altura_corr_folio = altura_base_cm * factor_folio

    # 3) Corrección geométrica por pies: factor = Z / (Z - L)
    factor_pies = Z / (Z - L)
    altura_corr_pies = altura_corr_folio * factor_pies

    # 4) Ajuste por calzado
    altura_final = altura_corr_pies - suela_cm

    return altura_final




# ============================================================
# WEBCAM
# ============================================================

def main(Z, L, suela_cm):
    """
    Lógica principal del sistema:
    - Captura webcam
    - Detecta persona
    - Detecta folio en bounding box exacta
    - Calcula altura corregida
    """

    modelo = YOLO(PESOS_YOLO)
    alturas = deque(maxlen=5)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Altura", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_out = frame.copy()

        # ---------------------------
        # 1) Persona
        # ---------------------------
        bbox, mask_persona = detectar_persona_segmentado(frame, modelo)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0,255,0), 3)

            # ---------------------------
            # 2) ROI exacto
            # ---------------------------
            roi = frame[y1:y2, x1:x2]

            # ---------------------------
            # 3) Folio dentro del ROI
            # ---------------------------
            rect_folio = detectar_folio_en_roi(roi)

            if rect_folio is not None:
                box = cv2.boxPoints(rect_folio).astype(np.int32)
                box[:,0] += x1
                box[:,1] += y1
                cv2.drawContours(frame_out, [box], 0, (0,0,255), 3)

                # ---------------------------
                # 4) Altura ya parametrizada
                # ---------------------------
                altura = calcular_altura(bbox, rect_folio, Z, L, suela_cm)

                if altura is not None:
                    alturas.append(altura)

                    cv2.putText(
                        frame_out,
                        f"Altura: {altura:.1f} cm",
                        (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0,255,255),
                        3
                    )

                    media = np.median(list(alturas))
                    cv2.putText(
                        frame_out,
                        f"Media: {media:.1f} cm",
                        (10,100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0,255,0),
                        2
                    )

        cv2.imshow("Altura", frame_out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":
    args = get_arguments()
    main(args.Z, args.L, args.suela)
