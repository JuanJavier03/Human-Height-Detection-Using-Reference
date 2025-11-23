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
    """Versión mejorada: maneja luz mixta (sombra arriba, luz abajo)."""

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Color blanco neutro (RELAJADO para diferentes condiciones de luz)
    mask_A = cv2.inRange(A, 115, 140)  # Más amplio
    mask_B = cv2.inRange(B, 115, 140)  # Más amplio
    mask_color = cv2.bitwise_and(mask_A, mask_B)

    # Luminosidad alta (REDUCIDO para detectar folio en sombra)
    _, mask_L = cv2.threshold(L, 150, 255, cv2.THRESH_BINARY)  # Era 180

    # Candidato a folio
    mask_folio = cv2.bitwise_and(mask_color, mask_L)

    # Usar escala de grises para mejor detección de bordes
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_folio = cv2.bitwise_and(gray, gray, mask=mask_folio)
    
    # Ecualización CLAHE para mejorar contraste en zonas sombreadas
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray_folio)
    
    # Detectar bordes con umbrales más sensibles
    edges = cv2.Canny(gray_eq, 30, 120)

    # Morfología para conectar bordes
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

        area_rect = w*h
        # Área mínima baja para detectar folio más lejos o pequeño
        if area_rect < 800:
            continue

        # Ratio A4 (más tolerante)
        ratio = max(w,h) / min(w,h)
        if not (1.2 < ratio < 1.6):
            continue

        # FILTRO CLAVE: El folio debe estar en la mitad SUPERIOR del ROI
        # Esto descarta rectángulos en pies/suelo
        if cy > h_roi * 0.6:  # Si está por debajo del 60%, descartarlo
            continue

        # Verificar luminosidad mínima (folio blanco, no sombra oscura)
        box = cv2.boxPoints(rect).astype(np.int32)
        mask_rect = np.zeros(L.shape, dtype=np.uint8)
        cv2.drawContours(mask_rect, [box], -1, 255, -1)
        mean_L = cv2.mean(L, mask=mask_rect)[0]
        
        # Umbral bajo para permitir folio en sombra
        if mean_L < 130:
            continue

        # Puntuación: favorece área grande + luminosidad + posición superior
        # cy/h_roi = posición vertical normalizada (0=arriba, 1=abajo)
        factor_posicion = 1.5 * (1.0 - cy/h_roi)  # Bonus por estar arriba
        puntuacion = area_rect * (mean_L/255.0) * factor_posicion

        if puntuacion > mejor_puntuacion:
            mejor_puntuacion = puntuacion
            mejor_rect = rect

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
                        (255, 0, 0),  # Azul
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
                            (255, 0, 0),  # Azul
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
