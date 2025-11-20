"""
Visualización PASO A PASO de la detección del folio usando
únicamente técnicas tradicionales vistas en clase.

Cada vez que cierres una ventana, se abrirá la siguiente,
permitiendo observar cómo evoluciona la imagen en cada etapa.

CONTENIDOS USADOS:
- Tema 3: Transformaciones de color (BGR → LAB)
- Tema 4: Umbralización simple (inRange / threshold)
- Tema 4: Operaciones morfológicas (dilatación)
- Tema 4: Detectores de bordes (Canny)
- Tema 4: Contornos y bounding boxes rotados (minAreaRect)
- Tema 4: Análisis geométrico (ratio A4)

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
    # PASO 1 — ROI original (zona del cuerpo)
    # -------------------------------------------------------------
    mostrar("PASO 1: ROI original (cuerpo + folio)", roi)

    # -------------------------------------------------------------
    # PASO 2 — Tema 3: BGR → LAB
    # LAB separa bien la luminosidad (L) del color (A,B).
    # -------------------------------------------------------------
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    mostrar("PASO 2: ROI en espacio LAB", lab)

    L, A, B = cv2.split(lab)
    mostrar("PASO 3: Canal L (luminosidad)", L)
    mostrar("PASO 4: Canal A (croma verde–rojo)", A)
    mostrar("PASO 5: Canal B (croma azul–amarillo)", B)

    # -------------------------------------------------------------
    # PASO 6 — Máscara por color: buscar "blanco neutro"
    # Tema 3 + Tema 4: inRange
    # El folio blanco en LAB tiene A y B cerca de un valor neutro.
    # Suponemos valores aproximados en torno a 128 (esto se puede ajustar).
    # -------------------------------------------------------------
    # Rango de "neutralidad" en A y B (ajustable según tus imágenes)
    mask_A = cv2.inRange(A, 120, 136)
    mask_B = cv2.inRange(B, 120, 136)
    mask_neutra = cv2.bitwise_and(mask_A, mask_B)
    mostrar("PASO 6: Máscara de color neutro (A y B ≈ blanco)", mask_neutra)

    # -------------------------------------------------------------
    # PASO 7 — Máscara por luminosidad: zonas muy claras
    # Tema 4: umbralización simple
    # -------------------------------------------------------------
    _, mask_L = cv2.threshold(L, 180, 255, cv2.THRESH_BINARY)
    mostrar("PASO 7: Máscara de zonas luminosas (umbral en L)", mask_L)

    # -------------------------------------------------------------
    # PASO 8 — Máscara combinada: "posible folio"
    # (blanco y luminoso)
    # -------------------------------------------------------------
    mask_folio = cv2.bitwise_and(mask_neutra, mask_L)
    mostrar("PASO 8: Máscara combinada (blanco + luminoso)", mask_folio)

    # -------------------------------------------------------------
    # PASO 9 — Aplicar máscara al canal L
    # Trabajaremos sólo en las zonas con color/luz de folio.
    # -------------------------------------------------------------
    L_folio = cv2.bitwise_and(L, L, mask=mask_folio)
    mostrar("PASO 9: Canal L restringido a región tipo folio", L_folio)

    # -------------------------------------------------------------
    # PASO 10 — Tema 4: Canny sobre la zona candidata a folio
    # Buscamos bordes fuertes allí donde hay blanco neutro.
    # -------------------------------------------------------------
    edges = cv2.Canny(L_folio, 50, 150)
    mostrar("PASO 10: Bordes (Canny) en la región de posible folio", edges)

    # -------------------------------------------------------------
    # PASO 11 — Morfología sobre los bordes (dilatación)
    # Tema 4: Operaciones morfológicas
    # Dilatamos los bordes para cerrar huecos y formar contornos más claros.
    # -------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    mostrar("PASO 11: Bordes dilatados (para contornos más robustos)", edges_dilated)

    # -------------------------------------------------------------
    # PASO 12 — Tema 4: Contornos sobre los bordes dilatados
    # -------------------------------------------------------------
    contornos, _ = cv2.findContours(
        edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    debug_contornos = roi.copy()
    cv2.drawContours(debug_contornos, contornos, -1, (0, 255, 0), 2)
    mostrar("PASO 12: Contornos detectados en bordes dilatados", debug_contornos)

    # -------------------------------------------------------------
    # PASO 13 — Detección del folio por COLOR + FORMA (ratio A4)
    #
    # Aquí es donde de verdad usamos la forma:
    #   - minAreaRect → rectángulo rotado
    #   - ratio lado_largo/lado_corto ≈ 1.414
    #   - área mínima para evitar ruido.
    # -------------------------------------------------------------
    mejor_rect = None
    mejor_puntuacion = 0.0

    for c in contornos:
        if len(c) < 5:
            continue

        # Rectángulo mínimo rotado alrededor del contorno
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect

        if w == 0 or h == 0:
            continue

        # Ratio geométrico
        lado_largo = max(w, h)
        lado_corto = min(w, h)
        ratio = lado_largo / lado_corto

        # Esperamos algo muy cercano a A4 ≈ 1.414
        # Usamos una tolerancia relativamente estrecha.
        if not (1.3 < ratio < 1.5):
            continue

        # Área del rectángulo (para descartar cosas pequeñas).
        area_rect = w * h
        if area_rect < 5000:  # AJUSTAR según tamaño del ROI
            continue

        # Comprobamos la "blancura" interior:
        # Creamos una máscara del rectángulo y calculamos media de L.
        box = cv2.boxPoints(rect)
        box_int = box.astype(np.int32)

        mask_rect = np.zeros(L.shape, dtype=np.uint8)
        cv2.drawContours(mask_rect, [box_int], -1, 255, -1)

        # Media de L dentro del rectángulo → debe ser alta si es folio
        mean_L = cv2.mean(L, mask=mask_rect)[0]

        # Puntuación combinada: área grande + buena luminosidad
        puntuacion = area_rect * (mean_L / 255.0)

        if puntuacion > mejor_puntuacion:
            mejor_puntuacion = puntuacion
            mejor_rect = rect

    # -------------------------------------------------------------
    # PASO 14 — Dibujar el folio detectado (si existe)
    # -------------------------------------------------------------
    debug_final = roi.copy()

    if mejor_rect is not None:
        box = cv2.boxPoints(mejor_rect)
        box = box.astype(np.int32)
        cv2.drawContours(debug_final, [box], 0, (255, 0, 0), 3)
        mostrar("PASO 13: Folio detectado (color + ratio A4)", debug_final)
    else:
        mostrar("PASO 13: No se detectó folio", roi)


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
