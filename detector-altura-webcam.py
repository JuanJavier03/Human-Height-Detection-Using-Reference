"""
Detección integrada HUMANO (YOLO) + FOLIO (OpenCV clásico)
sobre WEBCAM en tiempo real, y cálculo aproximado de la altura
del humano usando un folio A4 como referencia.
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

PESOS_YOLO = "best.pt"

# ============================================================
#  DETECTOR DE PERSONA (YOLO)
# ============================================================

def detectar_persona(frame, modelo):
    r = modelo(frame, conf=0.55, iou=0.7)[0]

    if r.boxes is None or len(r.boxes) == 0:
        return None, frame

    x1, y1, x2, y2 = map(int, r.boxes.xyxy[0].tolist())
    frame_out = frame.copy()
    cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return (x1, y1, x2, y2), frame_out


# ============================================================
#  DETECTOR DE FOLIO (Color + forma A4)
# ============================================================

def detectar_folio_en_roi(roi):
    """Versión optimizada para webcam: rápido y robusto."""

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Color blanco neutro
    mask_A = cv2.inRange(A, 115, 140)
    mask_B = cv2.inRange(B, 115, 140)
    mask_color = cv2.bitwise_and(mask_A, mask_B)

    # Luminosidad alta
    _, mask_L = cv2.threshold(L, 180, 255, cv2.THRESH_BINARY)

    mask_folio = cv2.bitwise_and(mask_color, mask_L)
    L_folio = cv2.bitwise_and(L, L, mask=mask_folio)

    # Detectar bordes
    edges = cv2.Canny(L_folio, 50, 150)

    # Morfología
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contornos, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

        # Verificar luminosidad
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
#  ALTURA (regla de 3)
# ============================================================

def calcular_altura(bbox_persona, rect_folio):
    x1, y1, x2, y2 = bbox_persona
    altura_px = y2 - y1

    (_, _), (w, h), angle = rect_folio
    folio_px = max(w, h)

    if folio_px == 0:
        return None

    altura_cm = (altura_px / folio_px) * 29.7

    return altura_cm


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
    print("Controles:")
    print("  's' - Guardar captura en archivo")
    print("  'c' - Copiar al portapapeles (requiere xclip)")
    print("  'q' - Salir")

    contador_capturas = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturando frame.")
            break

        frame_out = frame.copy()
        h_frame, w_frame = frame.shape[:2]

        # 1) Detectar persona
        bbox, frame_out = detectar_persona(frame_out, modelo)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            
            # -------------------------------------------------------
            # EXPANSIÓN: ROI lateral (40% a cada lado)
            # -------------------------------------------------------
            ancho_persona = x2 - x1
            
            # SOLO laterales (sin expandir arriba/abajo)
            margen_horizontal = int(ancho_persona * 0.4)
            
            # Calcular ROI expandido
            roi_x1 = max(0, x1 - margen_horizontal)
            roi_y1 = y1
            roi_x2 = min(w_frame, x2 + margen_horizontal)
            roi_y2 = y2
            
            # Extraer ROI expandido
            roi_expandido = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            # 2) Detectar folio en ROI expandido
            rect_folio = detectar_folio_en_roi(roi_expandido)

            if rect_folio is not None:
                # Ajustar coordenadas al frame completo
                box = cv2.boxPoints(rect_folio).astype(np.int32)
                box[:, 0] += roi_x1
                box[:, 1] += roi_y1

                cv2.drawContours(frame_out, [box], 0, (0, 0, 255), 3)

                # 3) Calcular altura
                altura_cm = calcular_altura(bbox, rect_folio)
                if altura_cm is not None:
                    alturas_recent.append(altura_cm)
                    
                    # Altura del frame actual
                    cv2.putText(
                        frame_out,
                        f"Altura: {altura_cm:.1f} cm",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 255),
                        3
                    )

                    # MEDIANA de las últimas 5 mediciones
                    if len(alturas_recent) > 0:
                        altura_mediana = np.median(list(alturas_recent))
                        cv2.putText(
                            frame_out,
                            f"Altura estimada: {altura_mediana:.1f} cm ({len(alturas_recent)} muestras)",
                            (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2
                        )

        cv2.imshow(window_name, frame_out)

        # CONTROLES DE TECLADO
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        
        elif key == ord('s'):
            nombre_archivo = f"captura_{contador_capturas:03d}.png"
            cv2.imwrite(nombre_archivo, frame_out)
            print(f"✓ Captura guardada: {nombre_archivo}")
            contador_capturas += 1
        
        elif key == ord('c'):
            temp_file = "/tmp/captura_temp.png"
            cv2.imwrite(temp_file, frame_out)
            
            import subprocess
            try:
                subprocess.run(
                    ["xclip", "-selection", "clipboard", "-t", "image/png", "-i", temp_file],
                    check=True
                )
                print("✓ Captura copiada al portapapeles")
            except FileNotFoundError:
                print("✗ Error: xclip no instalado. Instala con: sudo apt install xclip")
            except subprocess.CalledProcessError:
                print("✗ Error al copiar al portapapeles")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
