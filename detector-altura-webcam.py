"""
Detección integrada HUMANO (YOLO-SEG) + FOLIO (OpenCV clásico)
sobre WEBCAM en tiempo real, y cálculo aproximado de la altura
del humano usando un folio A4 como referencia.

Incluye:
    - Segmentación precisa de persona (bounding box real, derivada de máscara)
    - Corrección de perspectiva del folio (ratio geométrico 1.414)
    - Corrección geométrica por pies adelantados (factor = Z/(Z-L))
    - Ajuste final por calzado (~2 cm)

El folio se detecta ÚNICAMENTE dentro de la bounding box de la persona.
No hay expansión lateral ni superior.
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO


# ============================================================
# CONFIGURACIÓN
# ============================================================

PESOS_YOLO = "yolov8s-seg.pt"


# ============================================================
# DETECTOR DE PERSONA POR SEGMENTACIÓN
# ============================================================

def detectar_persona_segmentado(frame, modelo):
    """
    Detección exacta de persona usando segmentación.
    - YOLO trabaja sobre el frame original.
    - retina_masks=True evita desplazamientos.
    - La bounding box se calcula directamente de la máscara.
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

    # Filtrar SOLO "persona" (class=0)
    idx = np.where(clases == 0)[0]
    if len(idx) == 0:
        return None, None

    # elegir la máscara más grande
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
    Detecto el folio usando LAB + bordes.
    Esta función trabaja únicamente dentro del ROI (bounding de la persona).
    """

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
    mejor_score = 0

    for c in contornos:
        if len(c) < 5:
            continue

        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect

        if w == 0 or h == 0 or w * h < 5000:
            continue

        ratio = max(w, h) / min(w, h)
        if not (1.3 < ratio < 1.5):
            continue

        box = cv2.boxPoints(rect).astype(np.int32)
        mask_rect = np.zeros(L.shape, dtype=np.uint8)
        cv2.drawContours(mask_rect, [box], -1, 255, -1)
        mean_L = cv2.mean(L, mask=mask_rect)[0]

        score = (w*h) * (mean_L/255.0)
        if score > mejor_score:
            mejor_score = score
            mejor_rect = rect

    return mejor_rect



# ============================================================
# CÁLCULO FINAL DE ALTURA
# ============================================================

def calcular_altura(bbox_persona, rect_folio):
    x1, y1, x2, y2 = bbox_persona
    altura_px = y2 - y1

    (_, _), (w, h), _ = rect_folio
    mayor = max(w, h)
    menor = min(w, h)

    if mayor == 0 or menor == 0:
        return None

    # 1) regla de 3 usando lado mayor del folio (29.7 cm)
    altura_base_cm = (altura_px / mayor) * 29.7

    # 2) corrección de perspectiva A4
    R_real = 1.414
    R_obs = mayor / menor
    factor_folio = R_real / R_obs
    altura_corr_folio = altura_base_cm * factor_folio

    # 3) corrección por pies adelantados (modelo geométrico)
    Z = 3.20   # distancia cámara–torso
    L = 0.10   # adelanto del pie

    factor_pies = Z / (Z - L)
    altura_corr_pies = altura_corr_folio * factor_pies

    # 4) ajuste por suela/puntera
    altura_final = altura_corr_pies - 2.0

    return altura_final



# ============================================================
# WEBCAM
# ============================================================

def main():
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

        # ===========================================
        # 1) PERSONA
        # ===========================================
        bbox, mask_persona = detectar_persona_segmentado(frame, modelo)

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            # bbox persona
            cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0,255,0), 3)

            # ===========================================
            # 2) ROI = bounding box EXACTA de la persona
            # ===========================================
            roi_x1, roi_x2 = x1, x2
            roi_y1, roi_y2 = y1, y2

            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            # ===========================================
            # 3) DETECTAR FOLIO dentro del bbox y nada más
            # ===========================================
            rect_folio = detectar_folio_en_roi(roi)

            if rect_folio is not None:
                box = cv2.boxPoints(rect_folio).astype(np.int32)
                box[:,0] += roi_x1
                box[:,1] += roi_y1
                cv2.drawContours(frame_out, [box], 0, (0,0,255), 3)

                # ===========================================
                # 4) ALTURA
                # ===========================================
                altura = calcular_altura(bbox, rect_folio)

                if altura is not None:
                    alturas.append(altura)

                    cv2.putText(frame_out,
                                f"Altura: {altura:.1f} cm",
                                (10,50),
                                cv2.FONT_HERSHEY_SIMPLEX,1.2,
                                (0,255,255),3)

                    media = np.median(list(alturas))
                    cv2.putText(frame_out,
                                f"Media: {media:.1f} cm",
                                (10,100),
                                cv2.FONT_HERSHEY_SIMPLEX,1.0,
                                (0,255,0),2)

        cv2.imshow("Altura", frame_out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
