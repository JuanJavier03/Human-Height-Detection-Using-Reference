"""
Detección de PERSONA + FOLIO en una imagen fija.
------------------------------------------------

Este script detecta primero a la persona mediante YOLOv8
y posteriormente detecta un FOLIO usando ÚNICAMENTE técnicas clásicas
del temario de la asignatura de Procesamiento de Imágenes Digitales:

- Tema 3: Transformaciones y espacios de color (BGR → LAB)
- Tema 4: Umbralización global + Otsu
- Tema 4: Filtros y suavizado
- Tema 4: Operaciones morfológicas (apertura, cierre)
- Tema 4: Canny (detección de bordes)
- Tema 4: Contornos + aproximación poligonal
- Tema 4: Detección de formas rectangulares vía approxPolyDP

Salida del script:
- Se muestra en pantalla la imagen con ambas detecciones dibujadas.
- En terminal se imprimen las coordenadas del humano y del folio.
"""

import sys
import cv2
import numpy as np
from ultralytics import YOLO


PESOS_YOLO = "best.pt"


# =====================================================================
# === DETECTOR DE FOLIO (TÉCNICAS TRADICIONALES DE LA ASIGNATURA) ====
# =====================================================================

def detectar_folio(roi):
    """
    Detecta un folio dentro del ROI usando EXCLUSIVAMENTE técnicas vistas en clase.

    Parámetros:
        roi (np.ndarray): recorte de la imagen correspondiente al cuerpo detectado.

    Devuelve:
        (x1, y1, x2, y2) del folio dentro del ROI.
        Si no detecta folio, devuelve None.
    """

    # -------------------------------------------------------------
    # 1. Tema 3 - Conversión de espacio de color BGR → LAB
    # -------------------------------------------------------------
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # -------------------------------------------------------------
    # 2. Tema 4 - Suavizado para eliminar ruido (Gaussiano)
    # -------------------------------------------------------------
    L_blur = cv2.GaussianBlur(L, (5, 5), 0)

    # -------------------------------------------------------------
    # 3. Tema 4 - Umbralización de Otsu (ideal para separar folio)
    # -------------------------------------------------------------
    _, th = cv2.threshold(L_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # A veces Otsu invierte la lógica (folio negro/blanco),
    # aseguramos que el folio es la región BLANCA (mayor área blanca).
    if np.sum(th == 255) < np.sum(th == 0):
        th = cv2.bitwise_not(th)

    # -------------------------------------------------------------
    # 4. Tema 4 - Morfología (apertura para limpiar ruido)
    # -------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    th_clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    # -------------------------------------------------------------
    # 5. Tema 4 - Detección de contornos externos
    # -------------------------------------------------------------
    contornos, _ = cv2.findContours(th_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mejor_folio = None
    mejor_area = 0

    # -------------------------------------------------------------
    # 6. Tema 4 - Filtrar contornos buscando un RECTÁNGULO
    #    mediante approxPolyDP → 4 vértices
    # -------------------------------------------------------------
    for c in contornos:

        area = cv2.contourArea(c)
        if area < 1000:   # filtro mínimo
            continue

        perimetro = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimetro, True)

        if len(approx) == 4:
            # Es un rectángulo → buen candidato a folio
            x, y, w, h = cv2.boundingRect(approx)

            # Comprobar proporción aproximada de un folio (A4: 1:1.41)
            ratio = h / w if w > 0 else 0
            if 1.2 < ratio < 1.7:   # rango flexible por perspectiva
                if area > mejor_area:
                    mejor_area = area
                    mejor_folio = (x, y, x+w, y+h)

    return mejor_folio  # None si no hay folio


# =====================================================================
# === DETECTOR DE PERSONA (YOLOV8) + INTEGRACIÓN DEL FOLIO =============
# =====================================================================

def procesar_imagen(ruta_imagen):
    """
    Detecta persona con YOLO + detecta folio con técnicas de clase.
    """

    # -------------------------------
    # Cargar YOLO
    # -------------------------------
    modelo = YOLO(PESOS_YOLO)

    # -------------------------------
    # Cargar imagen original
    # -------------------------------
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print("Error: la imagen no existe.")
        sys.exit(1)

    # -------------------------------
    # Detección de persona (igual que human-detector.py)
    # -------------------------------
    resultados = modelo(ruta_imagen, conf=0.55, iou=0.7)
    resultado = resultados[0]

    cajas = resultado.boxes.xyxy.tolist() if resultado.boxes is not None else []
    if len(cajas) == 0:
        print("No se detectó ninguna persona.")
        return

    # Asumimos que solo hay una persona principal
    x1, y1, x2, y2 = map(int, cajas[0])

    print("\n--- DETECCIÓN DE PERSONA ---")
    print(f"Humano: ({x1}, {y1}) -> ({x2}, {y2})")

    # Dibujar persona
    imagen_con_cajas = resultado.plot()

    # -------------------------------
    # Recorte del cuerpo (ROI)
    # -------------------------------
    roi = imagen[y1:y2, x1:x2]

    # -------------------------------
    # DETECCIÓN DEL FOLIO (OpenCV)
    # -------------------------------
    folio = detectar_folio(roi)

    print("\n--- DETECCIÓN DE FOLIO ---")
    if folio is None:
        print("No se ha detectado folio.")
    else:
        fx1, fy1, fx2, fy2 = folio
        print(f"Folio en ROI: ({fx1}, {fy1}) -> ({fx2}, {fy2})")

        # Ajustar a coordenadas globales de la imagen
        gx1 = x1 + fx1
        gy1 = y1 + fy1
        gx2 = x1 + fx2
        gy2 = y1 + fy2

        print(f"Folio (coordenadas globales): ({gx1}, {gy1}) -> ({gx2}, {gy2})")

        cv2.rectangle(imagen_con_cajas, (gx1, gy1), (gx2, gy2), (255, 0, 0), 3)

    # -------------------------------
    # Mostrar resultado final
    # -------------------------------
    cv2.imshow("Detección Humano + Folio", imagen_con_cajas)
    print("\nPulsa una tecla en la ventana para cerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =====================================================================
# === MAIN =============================================================
# =====================================================================

def main():
    if len(sys.argv) != 2:
        print("Uso: python human-folio-detector.py imagen.jpg")
        sys.exit(1)

    procesar_imagen(sys.argv[1])


if __name__ == "__main__":
    main()
