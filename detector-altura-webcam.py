"""
Detección integrada HUMANO (YOLO) + FOLIO (OpenCV clásico)
sobre WEBCAM en tiempo real, y cálculo aproximado de la altura
del humano usando un folio A4 como referencia.

Salida:
    - Ventana en tiempo real con:
         * Caja verde = humano detectado
         * Caja roja = folio detectado
         * Texto con altura estimada
    - Terminal imprime la altura estimada ocasionalmente.

Uso:
    python detector-altura-webcam.py

Requisitos:
    - ultralytics (YOLOv8)
    - opencv-python
    - numpy
    - best.pt en el mismo directorio
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
    """
    Devuelve:
      - bbox_persona = (x1,y1,x2,y2)  o None
      - frame con caja dibujada
    """
    r = modelo(frame, conf=0.55, iou=0.7)[0]

    if r.boxes is None or len(r.boxes) == 0:
        return None, frame

    # Tomamos la primera persona
    x1, y1, x2, y2 = map(int, r.boxes.xyxy[0].tolist())

    frame_out = frame.copy()
    cv2.rectangle(frame_out, (x1,y1), (x2,y2), (0,255,0), 3)

    return (x1, y1, x2, y2), frame_out


# ============================================================
#  DETECTOR DE FOLIO (Color + forma A4)
# ============================================================

def detectar_folio_en_roi(roi):
    """Versión optimizada para webcam 1280x720."""

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Color blanco neutro
    mask_A = cv2.inRange(A, 115, 140)
    mask_B = cv2.inRange(B, 115, 140)
    mask_color = cv2.bitwise_and(mask_A, mask_B)

    # Luminosidad alta
    _, mask_L = cv2.threshold(L, 150, 255, cv2.THRESH_BINARY)

    # Candidato a folio
    mask_folio = cv2.bitwise_and(mask_color, mask_L)

    # Escala de grises
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_folio = cv2.bitwise_and(gray, gray, mask=mask_folio)
    
    # Ecualización CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray_folio)
    
    # Detectar bordes
    edges = cv2.Canny(gray_eq, 30, 120)

    # Morfología
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    edges = cv2.dilate(edges, kernel, iterations=3)
    edges = cv2.erode(edges, kernel, iterations=1)

    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mejor_rect = None
    mejor_puntuacion = 0
    
    h_roi, w_roi = roi.shape[:2]

    for c in contornos:
        if len(c) < 5:
            continue

        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect

        if w == 0 or h == 0:
            continue

        # ===== FILTROS AJUSTADOS PARA 1280x720 =====
        
        # Lados del rectángulo detectado
        lado_min = min(w, h)
        lado_max = max(w, h)
        
        # Folio A4 en cámara 1280x720:
        # - A 1m: lado corto ~100-200px, lado largo ~140-300px
        # - A 3m: lado corto ~35-70px, lado largo ~50-100px
        
        # Rango válido del lado CORTO (21 cm real)
        if lado_min < 30 or lado_min > 250:
            continue
            
        # Rango válido del lado LARGO (29.7 cm real)  
        if lado_max < 45 or lado_max > 400:
            continue

        # Área razonable
        area_rect = w * h
        if area_rect < 1500 or area_rect > 100000:
            continue

        # Ratio A4 estricto (29.7/21 = 1.414)
        ratio = lado_max / lado_min
        if not (1.3 < ratio < 1.55):
            continue

        # Posición superior (pecho, no pies)
        if cy > h_roi * 0.6:
            continue

        # Luminosidad mínima
        box = cv2.boxPoints(rect).astype(np.int32)
        mask_rect = np.zeros(L.shape, dtype=np.uint8)
        cv2.drawContours(mask_rect, [box], -1, 255, -1)
        mean_L = cv2.mean(L, mask=mask_rect)[0]
        
        if mean_L < 130:
            continue

        # Puntuación
        factor_posicion = 1.5 * (1.0 - cy/h_roi)
        puntuacion = area_rect * (mean_L/255.0) * factor_posicion

        if puntuacion > mejor_puntuacion:
            mejor_puntuacion = puntuacion
            mejor_rect = recto

    return mejor_rect


# ============================================================
#  ALTURA (regla de 3)
# ============================================================

def calcular_altura(bbox_persona, rect_folio):
    x1, y1, x2, y2 = bbox_persona
    
    # Sin recorte de padding
    altura_px = y2 - y1

    (_, _), (w, h), angle = rect_folio
    
    # DETECTAR ORIENTACIÓN DEL FOLIO
    # minAreaRect devuelve (w, h) donde w es siempre <= h
    # Necesitamos determinar qué dimensión corresponde al lado vertical
    
    # Dimensiones reales A4
    LADO_LARGO_A4_CM = 29.7
    LADO_CORTO_A4_CM = 21.0
    
    # Si h > w, el folio está orientado con el lado largo en VERTICAL
    # Si w > h, el folio está orientado con el lado largo en HORIZONTAL
    
    if h > w:
        # VERTICAL: lado largo (29.7cm) está en posición vertical
        folio_px_vertical = h
        referencia_cm = LADO_LARGO_A4_CM
        orientacion = "VERTICAL (alto)"
    else:
        # HORIZONTAL: lado corto (21cm) está en posición vertical
        folio_px_vertical = h
        referencia_cm = LADO_CORTO_A4_CM
        orientacion = "HORIZONTAL (bajo)"

    if folio_px_vertical == 0:
        return None

    # DEBUG: ver qué dimensiones estamos usando
    print(f"DEBUG - Orientación folio: {orientacion}")
    print(f"DEBUG - Altura persona: {altura_px:.1f} px")
    print(f"DEBUG - Folio detectado w={w:.1f}, h={h:.1f}")
    print(f"DEBUG - Usando dimensión vertical: {folio_px_vertical:.1f} px = {referencia_cm} cm")
    print(f"DEBUG - Ratio persona/foli+o: {altura_px/folio_px_vertical:.2f}")

    # Regla de tres: altura_persona_px / folio_vertical_px = altura_cm / referencia_cm
    altura_cm = (altura_px / folio_px_vertical) * referencia_cm
    
    print(f"DEBUG - Altura calculada: {altura_cm:.1f} cm")
    print("="*50)
    
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
    
    print("Webcam iniciada. Pulsa 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturando frame.")
            break

        frame_out = frame.copy()

        # -------------------------------------------------------
        # 1) Detectar persona
        # -------------------------------------------------------
        bbox, frame_out = detectar_persona(frame_out, modelo)

        if bbox is not None:
            x1,y1,x2,y2 = bbox
            roi = frame[y1:y2, x1:x2]

            # -------------------------------------------------------
            # 2) Detectar folio
            # -------------------------------------------------------
            rect_folio = detectar_folio_en_roi(roi)

            if rect_folio is not None:
                box = cv2.boxPoints(rect_folio).astype(np.int32)
                box[:,0] += x1
                box[:,1] += y1

                cv2.drawContours(frame_out, [box], 0, (0,0,255), 3)

                # -------------------------------------------------------
                # 3) Calcular altura
                # -------------------------------------------------------
                altura_cm = calcular_altura(bbox, rect_folio)
                if altura_cm is not None:
                    alturas_recent.append(altura_cm)
                    cv2.putText(
                        frame_out,
                        f"Altura: {altura_cm:.1f} cm",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),  # Rojo
                        3
                    )

                    if alturas_recent:
                        altura_mediana = np.median(alturas_recent)
                        cv2.putText(
                            frame_out,
                            f"Altura media: {altura_mediana:.1f} cm",
                            (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),  # Rojo
                            2
                        )

                    print(f"Altura estimada: {altura_cm:.1f} cm")

        # Mostrar frame final
        cv2.imshow(window_name, frame_out)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
