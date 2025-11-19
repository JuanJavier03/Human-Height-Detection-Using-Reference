"""
Visualización PASO A PASO de la detección del folio usando
únicamente técnicas tradicionales vistas en clase.

Cada vez que cierres una ventana, se abrirá la siguiente,
permitiendo observar cómo evoluciona la imagen en cada etapa.

CONTENIDOS USADOS:
- Tema 3: Transformaciones de color (BGR → LAB)
- Tema 4: Umbralización global y de Otsu
- Tema 4: Operaciones morfológicas (apertura)
- Tema 4: Detectores de bordes (Canny)
- Tema 4: Detección de contornos, bounding boxes rotados (minAreaRect)
- Tema 4: Aprox. poligonal y análisis geométrico de formas

Uso:
    python folio-detector-pasos.py imagen.jpg x1 y1 x2 y2
(El ROI corresponde a la zona del cuerpo detectada previamente por YOLO.)
"""

import sys
import cv2
import numpy as np


def mostrar(nombre, imagen):
    """Muestra una imagen y espera a que se cierre la ventana."""
    cv2.imshow(nombre, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def procesar_roi(roi):
    """Realiza paso a paso la detección del folio, mostrando cada transformación."""

    # -------------------------------------------------------------
    # PASO 1 — Imagen original del ROI
    # -------------------------------------------------------------
    mostrar("PASO 1: ROI original (recorte del cuerpo)", roi)

    # -------------------------------------------------------------
    # PASO 2 — Tema 3: Conversión BGR → LAB
    # LAB separa bien luminosidad (L) y cromaticidad (A,B)
    # -------------------------------------------------------------
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    mostrar("PASO 2: Conversión a LAB", lab)

    L, A, B = cv2.split(lab)
    mostrar("PASO 3: Canal L (luminosidad)", L)

    # -------------------------------------------------------------
    # PASO 4 — Tema 4: Umbralización global sobre L
    # Preselecciona zonas muy brillantes (probable folio)
    # -------------------------------------------------------------
    _, mask_luminosa = cv2.threshold(L, 180, 255, cv2.THRESH_BINARY)
    mostrar("PASO 4: Máscara de zonas luminosas (umbral en L)", mask_luminosa)

    # -------------------------------------------------------------
    # PASO 5 — Aplicar máscara de luminancia
    # -------------------------------------------------------------
    L_focus = cv2.bitwise_and(L, L, mask=mask_luminosa)
    mostrar("PASO 5: Canal L restringido a zonas brillantes", L_focus)

    # -------------------------------------------------------------
    # PASO 6 — Tema 4: Otsu aplicado SOLO en la región luminosa
    # -------------------------------------------------------------
    _, th_otsu = cv2.threshold(L_focus, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mostrar("PASO 6: Umbralización de Otsu sobre región luminosa", th_otsu)

    # Corrección por si Otsu sale invertido
    if np.sum(th_otsu == 255) < np.sum(th_otsu == 0):
        th_otsu = cv2.bitwise_not(th_otsu)
        mostrar("PASO 7: Otsu invertido (corrección)", th_otsu)

    # -------------------------------------------------------------
    # PASO 8 — Tema 4: Morfología (Apertura)
    # Limpieza del ruido fino
    # -------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    th_open = cv2.morphologyEx(th_otsu, cv2.MORPH_OPEN, kernel)
    mostrar("PASO 8: Morfología — Apertura", th_open)

    # -------------------------------------------------------------
    # PASO 9 — Tema 4: Canny (visualización de bordes)
    # -------------------------------------------------------------
    edges = cv2.Canny(th_open, 50, 150)
    mostrar("PASO 9: Bordes (Canny)", edges)

    # -------------------------------------------------------------
    # PASO 10 — Tema 4: Contornos externos
    # -------------------------------------------------------------
    contornos, _ = cv2.findContours(th_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug_contornos = roi.copy()
    cv2.drawContours(debug_contornos, contornos, -1, (0, 255, 0), 2)
    mostrar("PASO 10: Contornos detectados", debug_contornos)

    # -------------------------------------------------------------
    # PASO 11 — DETECCIÓN REAL DEL FOLIO (robusta ante inclinación)
    #
    # Usamos minAreaRect del Tema 4 para obtener un rectángulo rotado.
    # Luego validamos la geometría (ratio A4 = 1.414).
    # -------------------------------------------------------------
    mejor_rect = None
    mejor_area = 0

    for c in contornos:

        area = cv2.contourArea(c)
        if area < 5000:     # filtro de área fuerte
            continue

        # Rectángulo mínimo rotado (Tema 4)
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect

        # Evitar división por cero
        if w == 0 or h == 0:
            continue

        # Ratio del folio A4 (independiente de la orientación)
        ratio = max(w, h) / min(w, h)

        # Rango aceptable para A4 real (1.414 ± tolerancia)
        if not (1.35 < ratio < 1.47):
            continue

        # Filtro extra: rectitud del contorno (approxPolyDP)
        per = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * per, True)

        ratio_rectitud = len(c) / len(approx)
        if ratio_rectitud > 12:
            continue

        # Mantener el mejor
        if area > mejor_area:
            mejor_area = area
            mejor_rect = rect

    # -------------------------------------------------------------
    # PASO 12 — Dibujo final del folio detectado (rectángulo rotado)
    # -------------------------------------------------------------
    debug_final = roi.copy()

    if mejor_rect:
        box = cv2.boxPoints(mejor_rect)
        box = np.int0(box)
        cv2.drawContours(debug_final, [box], 0, (255, 0, 0), 3)
        mostrar("PASO 11: Folio detectado (rectángulo rotado A4)", debug_final)
    else:
        mostrar("PASO 11: No se detectó folio", roi)


def main():
    if len(sys.argv) != 6:
        print("Uso:")
        print(" python folio-detector-pasos.py imagen.jpg x1 y1 x2 y2")
        print("Las coordenadas corresponden al ROI detectado con YOLO.")
        sys.exit(1)

    ruta = sys.argv[1]
    x1, y1, x2, y2 = map(int, sys.argv[2:])

    imagen = cv2.imread(ruta)
    if imagen is None:
        print("Error: no se pudo cargar la imagen.")
        sys.exit(1)

    roi = imagen[y1:y2, x1:x2]
    procesar_roi(roi)


if __name__ == "__main__":
    main()
