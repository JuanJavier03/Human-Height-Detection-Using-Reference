"""
Detección integrada de HUMANO (YOLO) + FOLIO (OpenCV clásico)
y cálculo aproximado de la altura en cm usando el folio como referencia.

Uso:
    python detectar_humano_folio.py foto.jpg

Salida:
    - Ventana con la imagen final: humano con caja verde y folio con caja azul.
    - Impresión en terminal de la altura estimada en cm.

Requisitos:
    - ultralytics (YOLOv8)
    - opencv-python
    - numpy
    - best.pt en el mismo directorio
"""

import sys
import cv2
import numpy as np
from ultralytics import YOLO

PESOS_YOLO = "best.pt"

# ============================================================
#  DETECTOR DE PERSONA (YOLO)
# ============================================================

def detectar_persona(imagen_bgr):
    """
    Detecta a la persona con YOLO y devuelve:
    - bounding box (x1, y1, x2, y2)
    - imagen con la caja dibujada
    """
    modelo = YOLO(PESOS_YOLO)

    resultados = modelo(imagen_bgr, conf=0.55, iou=0.7)
    r = resultados[0]

    if r.boxes is None or len(r.boxes) == 0:
        return None, imagen_bgr

    # Por simplicidad, tomamos la primera persona detectada.
    x1, y1, x2, y2 = map(int, r.boxes.xyxy[0].tolist())

    # Dibujar caja
    img = imagen_bgr.copy()
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)

    return (x1, y1, x2, y2), img


# ============================================================
#  DETECTOR DE FOLIO (Color + forma A4)
#  (versión compacta y optimizada del pipeline clásico)
# ============================================================

def detectar_folio_en_roi(roi):
    """
    Detecta el folio en el ROI usando:
    - Tema 3: LAB
    - Tema 4: Umbralización simple
    - Tema 4: Canny
    - Tema 4: Morfología
    - Tema 4: minAreaRect + ratio A4

    Devuelve:
      rect (minAreaRect)  ó None
    """

    # ----------- Tema 3: Convertir BGR → LAB -----------
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # ----------- Color: blanco neutro en A y B -----------
    mask_A = cv2.inRange(A, 120, 136)
    mask_B = cv2.inRange(B, 120, 136)
    mask_color = cv2.bitwise_and(mask_A, mask_B)

    # ----------- Luminosidad: muy brillante en L -----------
    _, mask_L = cv2.threshold(L, 180, 255, cv2.THRESH_BINARY)

    # ----------- Combinar máscaras (potencial folio) -----------
    mask_folio = cv2.bitwise_and(mask_color, mask_L)

    # ----------- Canny solo en la región candidata -----------
    L_focus = cv2.bitwise_and(L, L, mask=mask_folio)
    edges = cv2.Canny(L_focus, 50, 150)

    # ----------- Morfología: dilatación para unir bordes -----------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)

    # ----------- Contornos -----------
    contornos, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        if area_rect < 5000:   # filtro de tamaño mínimo
            continue

        # ----------- Tema 4: Ratio geométrico A4 -----------
        lado_largo = max(w, h)
        lado_corto = min(w, h)
        ratio = lado_largo / lado_corto

        if not (1.3 < ratio < 1.5):
            continue

        # ----------- Comprobación: el interior debe ser brillante -----------
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
#  CÁLCULO DE ALTURA (usando folio A4)
# ============================================================

def calcular_altura_px_y_cm(bbox_persona, rect_folio):
    """
    Calcula la altura aproximada del humano.
    - Altura persona = distancia vertical de su bounding box YOLO.
    - Altura folio = lado largo detectado en píxeles.
    - Folio A4 real: 29.7 cm lado largo.
    """

    x1, y1, x2, y2 = bbox_persona
    altura_px = y2 - y1

    # Folio
    (_, _), (w, h), _ = rect_folio
    lado_folio_px = max(w, h)

    if lado_folio_px == 0:
        return None, None

    # Regla de 3
    altura_cm = (altura_px / lado_folio_px) * 29.7

    return altura_px, altura_cm


# ============================================================
#  PROGRAMA PRINCIPAL
# ============================================================

def main():

    if len(sys.argv) != 2:
        print("Uso:\n  python detector-altura.py foto.jpg")
        sys.exit(1)

    ruta = sys.argv[1]
    imagen = cv2.imread(ruta)
    if imagen is None:
        print("No se pudo cargar la imagen.")
        sys.exit(1)

    # -----------------------------------------------
    # 1) Detectar persona
    # -----------------------------------------------
    bbox_humano, img_humano = detectar_persona(imagen)

    if bbox_humano is None:
        print("No se detectó ninguna persona.")
        sys.exit(0)

    x1, y1, x2, y2 = bbox_humano
    roi = imagen[y1:y2, x1:x2]

    # -----------------------------------------------
    # 2) Detectar folio en ROI
    # -----------------------------------------------
    rect_folio = detectar_folio_en_roi(roi)

    img_final = imagen.copy()

    if rect_folio is not None:
        # Convertir a coords globales
        box = cv2.boxPoints(rect_folio).astype(np.int32)
        box[:,0] += x1
        box[:,1] += y1

        cv2.drawContours(img_final, [box], 0, (255,0,0), 3)

        # -----------------------------------------------
        # 3) Calcular altura aproximada
        # -----------------------------------------------
        altura_px, altura_cm = calcular_altura_px_y_cm(bbox_humano, rect_folio)

        print(f"Altura estimada del humano: {altura_cm:.1f} cm")
    else:
        print("Folio NO detectado. No se puede calcular la altura.")

    # Dibujar la caja del humano
    cv2.rectangle(img_final, (x1,y1), (x2,y2), (0,255,0), 3)

    # Mostrar imagen final
    cv2.imshow("Humano + Folio detectados", img_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
