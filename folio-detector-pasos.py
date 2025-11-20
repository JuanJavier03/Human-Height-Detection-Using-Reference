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
    mostrar("PASO 2: Conversión a LAB (ROI en LAB)", lab)

    # Separamos canales
    L, A, B = cv2.split(lab)
    mostrar("PASO 3: Canal L (luminosidad)", L)
    mostrar("PASO 4: Canal A (croma verde–rojo)", A)
    mostrar("PASO 5: Canal B (croma azul–amarillo)", B)

    # -------------------------------------------------------------
    # PASO 6 — Tema 4: Máscara por luminosidad (candidatos a folio)
    # Folio = muy brillante en L
    # -------------------------------------------------------------
    _, mask_luminosa = cv2.threshold(L, 180, 255, cv2.THRESH_BINARY)
    mostrar("PASO 6: Máscara de zonas luminosas (umbral en L)", mask_luminosa)

    # -------------------------------------------------------------
    # PASO 7 — Tema 3/4: Máscara por color neutro en A y B
    # Folio blanco ≈ neutro en A y B (valores alrededor de 128 en LAB).
    # La piel, ropa, fondo suelen alejarse de 128.
    # -------------------------------------------------------------
    # Estos rangos se pueden ajustar según tus imágenes reales.
    mask_A = cv2.inRange(A, 120, 136)
    mask_B = cv2.inRange(B, 120, 136)
    mask_neutra = cv2.bitwise_and(mask_A, mask_B)
    mostrar("PASO 7: Máscara de color neutro (A y B ≈ 128)", mask_neutra)

    # -------------------------------------------------------------
    # PASO 8 — Combinación de ambas máscaras:
    #   - Muy luminoso
    #   - Casi sin croma (blanco neutro)
    # Esto apunta prácticamente solo al folio.
    # -------------------------------------------------------------
    mask_folio = cv2.bitwise_and(mask_luminosa, mask_neutra)
    mostrar("PASO 8: Máscara combinada (luminoso + neutro)", mask_folio)

    # Aplicamos esta máscara al canal L para concentrar Otsu en esa región
    L_focus = cv2.bitwise_and(L, L, mask=mask_folio)
    mostrar("PASO 9: Canal L restringido a zonas de posible folio", L_focus)

    # -------------------------------------------------------------
    # PASO 10 — Tema 4: Otsu sobre la región candidata a folio
    # Ahora Otsu trabaja SOLO sobre píxeles compatibles con folio blanco.
    # -------------------------------------------------------------
    _, th_otsu = cv2.threshold(L_focus, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mostrar("PASO 10: Umbralización de Otsu sobre región de folio", th_otsu)

    # Corrección por si Otsu sale invertido
    if np.sum(th_otsu == 255) < np.sum(th_otsu == 0):
        th_otsu = cv2.bitwise_not(th_otsu)
        mostrar("PASO 11: Otsu invertido (corrección)", th_otsu)

    # -------------------------------------------------------------
    # PASO 12 — Tema 4: Morfología (Apertura)
    # Limpieza del ruido fino que aún pueda quedar.
    # -------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    th_open = cv2.morphologyEx(th_otsu, cv2.MORPH_OPEN, kernel)
    mostrar("PASO 12: Morfología — Apertura (limpieza)", th_open)

    # -------------------------------------------------------------
    # PASO 13 — Tema 4: Canny (visualización de bordes)
    # -------------------------------------------------------------
    edges = cv2.Canny(th_open, 50, 150)
    mostrar("PASO 13: Bordes (Canny)", edges)

    # -------------------------------------------------------------
    # PASO 14 — Tema 4: Contornos externos
    # -------------------------------------------------------------
    contornos, _ = cv2.findContours(th_open, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    debug_contornos = roi.copy()
    cv2.drawContours(debug_contornos, contornos, -1, (0, 255, 0), 2)
    mostrar("PASO 14: Contornos detectados", debug_contornos)

    # -------------------------------------------------------------
    # PASO 15 — DETECCIÓN REAL DEL FOLIO (robusta ante inclinación)
    #
    # Usamos minAreaRect (Tema 4) para obtener el rectángulo ROTADO
    # que mejor encaja, y lo validamos por:
    #   - área mínima
    #   - proporción A4 ≈ 1.414
    #   - rectitud del contorno (approxPolyDP)
    # -------------------------------------------------------------
    mejor_rect = None
    mejor_area = 0

    for c in contornos:
        area = cv2.contourArea(c)
        if area < 5000:     # filtro de área fuerte (ajustable)
            continue

        # Rectángulo mínimo rotado
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect

        if w == 0 or h == 0:
            continue

        # Proporción A4 independientemente de la orientación
        ratio = max(w, h) / min(w, h)
        if not (1.35 < ratio < 1.47):   # A4 ≈ 1.414 ± tolerancia
            continue

        # Rectitud del contorno (cuántos puntos elimina approxPolyDP)
        per = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * per, True)
        ratio_rectitud = len(c) / len(approx)
        if ratio_rectitud > 12:
            continue

        if area > mejor_area:
            mejor_area = area
            mejor_rect = rect

    # -------------------------------------------------------------
    # PASO 16 — Dibujo final del folio detectado (rectángulo rotado)
    # -------------------------------------------------------------
    debug_final = roi.copy()

    if mejor_rect:
        box = cv2.boxPoints(mejor_rect)
        box = np.int0(box)
        cv2.drawContours(debug_final, [box], 0, (255, 0, 0), 3)
        mostrar("PASO 15: Folio detectado (rectángulo rotado A4)", debug_final)
    else:
        mostrar("PASO 15: No se detectó folio", roi)


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
