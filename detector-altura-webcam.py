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
    # Ejecutar la detección sobre el vídeo.
    # - conf: umbral de confianza mínimo para dibujar una detección.
    # - iou: umbral de solapamiento para filtrar cajas muy parecidas.
    r = modelo(frame, conf=0.55, iou=0.7)[0]

    if r.boxes is None or len(r.boxes) == 0:
        return None, frame

    # Tomamos la primera persona
    x1, y1, x2, y2 = map(int, r.boxes.xyxy[0].tolist())

    frame_out = frame.copy()
    cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 3)    # Verde

    return (x1, y1, x2, y2), frame_out


# ============================================================
#  DETECTOR DE FOLIO (Color + forma A4)
# ============================================================

def detectar_folio_en_roi(roi):
    """Versión optimizada para webcam: rápido y robusto."""

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Color blanco neutro (RELAJADO para diferentes condiciones de luz)
    mask_A = cv2.inRange(A, 115, 140)  # Más amplio (acercarse a 128 = neutro)
    mask_B = cv2.inRange(B, 115, 140)  # Más amplio
    # Solo deja pasar los píxeles neutros de ambos canales
    mask_color = cv2.bitwise_and(mask_A, mask_B)

    # Luminosidad alta (REDUCIDO para detectar folio en sombra)
    _, mask_L = cv2.threshold(L, 180, 255, cv2.THRESH_BINARY)  # Era 180

    # Candidato a folio:
    # máscara con los píxeles neutros (mask_color) y luminosos (mask_L)
    mask_folio = cv2.bitwise_and(mask_color, mask_L)

    # Usar escala de grises sobre el ROI y mostrar mejor el folio (zona candidata)
    #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    L_folio = cv2.bitwise_and(L, L, mask=mask_folio)

    # Ecualización CLAHE para mejorar contraste en zonas sombreadas
   # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
   # gray_eq = clahe.apply(gray_folio)

    # Detectar bordes con umbrales más sensibles
    edges = cv2.Canny(L_folio, 50, 150)

    # Morfología para conectar bordes
    # Los bordes del folio pueden tener pequeños huecos
    # Dilatación los conecta
    # Erosión elimina ruido sin borrar los bordes principales
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
   # edges = cv2.erode(edges, kernel, iterations=1)

    # Encontrar todas las figuras cerradas en la imagen de bordes
    # RETR_EXTERNAL: solo contornos externos (se ignoran agujeros internos)
    # CHAIN_APPROX_SIMPLE: comprime contornos (solo guarda puntos clave)
    # A --- B --- C
    #             |
    #             D --- E
    # Podemos eliminar B y quedarnos con A - C
    contornos, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mejor_rect = None
    mejor_puntuacion = 0

   # h_roi, w_roi = roi.shape[:2]    # Solo guardamos alto x ancho

    for c in contornos:
        if len(c) < 5:  # Número mínimo de puntos para minAreaRect (menos, es ruido)
            continue

        # Cálculo del rectángulo mínimo que encierra el contorno
            #  Usamos cv2.minAreaRect:
            # - Calcular el rectángulo rotado de área mínima que encierra el contorno
            # - Es perfecto para detectar papeles/folios que pueden estar inclinados

            # Devuelve una tupla con:
            # - (cx, cy): coordenadas del centro del rectángulo
            # - (w, h): ancho y alto del rectángulo
            # - angle: ángulo de rotación (en grados)

        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect

        # Si alguna de las dimensiones es cero, se descarta:
        if w == 0 or h == 0:
            continue

        area_rect = w*h
        # Área mínima baja para detectar folio más lejos o pequeño
            # 800 px ≈ rectángulo de 28×28 píxeles (muy pequeño)
            # Folio muy lejano o parcialmente visible pasa el filtro
            # Eliminamos ruidos pequeños
        if area_rect < 5000:    #800 antes
            continue

        # Validar la proporción aproximada del folio A4 (21x29.7 cm)
            # Calculamos la proporción entre el lado mayor y el menor
            # Proporción real de un A4 = 29.7 / 21 ≈ 1.414
        ratio = max(w, h) / min(w, h)   # Siempre >= 1
        # 1.2 es casi un cuadrado, 1.6 es muy alargado
        if not (1.3 < ratio < 1.5):     # Se garantiza un margen
            continue

        # El folio debe estar en la mitad SUPERIOR del ROI
        # Esto descarta rectángulos en pies/suelo (zonas brillantes como el sol)
       # if cy > h_roi * 0.6:  # Si está por debajo del 60%, descartarlo
        #    continue

        # Verificar luminosidad mínima (folio blanco, no sombra oscura)
        box = cv2.boxPoints(rect).astype(np.int32)  # Convierte el rectángulo a 4 puntos (esquinas, 4 coordenadas enteras)
        mask_rect = np.zeros(L.shape, dtype=np.uint8)   # Máscara negra del tamaño del canal L
        cv2.drawContours(mask_rect, [box], -1, 255, -1) # Rellenar el rectángulo en la máscara con blanco
        mean_L = cv2.mean(L, mask=mask_rect)[0]  # Media de L dentro del rectángulo

        # Umbral bajo para permitir folio en sombra
            # 130 en escala 0 - 255 ≈ 51% de brillo
            # Folio en sombra tiene luminosidad baja pero no tanto
            # Sombras muy oscuras/paredes grises se descartan
        # if mean_L < 130:
        #     continue

        # Puntuación: favorece área grande + luminosidad + posición superior
        # cy/h_roi = posición vertical normalizada (0=arriba, 1=abajo)
            # cy/h_roi cerca de 0 (por ejemplo, 0.2)
            # cy/h_roi cerca de 1 (por ejemplo, 0.8)
            # 1.0 - cy/h_roi invierte la escala (1=arriba, 0=abajo)
                # 1.0 - 0.2 = 0.8 (más alto, mejor)
            # 1.5 * (1.0 - cy/h_roi) escala el factor para dar más peso (elegir el "mejor folio")
     #   factor_posicion = 1.5 * (1.0 - cy/h_roi)  # Bonus por estar arriba
            # area_rect: favorece folios grandes
            # mean_L/255.0: favorece folios luminosos
            # factor_posicion: favorece folios en zona superior
       # puntuacion = area_rect * (mean_L/255.0) * factor_posicion

        puntuacion = area_rect * (mean_L/255.0)

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
   # LADO_LARGO_A4_CM = 29.7
   # LADO_CORTO_A4_CM = 21.0

    # Si h > w, el folio está orientado con el lado largo en VERTICAL
    # Si w > h, el folio está orientado con el lado largo en HORIZONTAL

    # if h > w:
    #     # VERTICAL: lado largo (29.7cm) está en posición vertical
    #     folio_px_vertical = h
    #     referencia_cm = LADO_LARGO_A4_CM
    #     orientacion = "VERTICAL (alto)"
    # else:
    #     # HORIZONTAL: lado corto (21cm) está en posición vertical
    #     folio_px_vertical = h
    #     referencia_cm = LADO_CORTO_A4_CM
    #     orientacion = "HORIZONTAL (bajo)"

    # if folio_px_vertical == 0:
    #     return None

    # # DEBUG: ver qué dimensiones estamos usando
    # print(f"DEBUG - Orientación folio: {orientacion}")
    # print(f"DEBUG - Altura persona: {altura_px:.1f} px")
    # print(f"DEBUG - Folio detectado w={w:.1f}, h={h:.1f}")
    # print(
    #     f"DEBUG - Usando dimensión vertical: {folio_px_vertical:.1f} px = {referencia_cm} cm")
    # print(f"DEBUG - Ratio persona/foli+o: {altura_px/folio_px_vertical:.2f}")
    folio_px = max(w,h)

    if folio_px == 0:
        return None



    # Regla de tres: altura_persona_px / folio_vertical_px = altura_cm / referencia_cm
   # altura_cm = (altura_px / folio_px_vertical) * referencia_cm

    altura_cm = (altura_px / folio_px) * 29.7

    # print(f"DEBUG - Altura calculada: {altura_cm:.1f} cm")
    # print("="*50)

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

        # -------------------------------------------------------
        # 1) Detectar persona
        # -------------------------------------------------------
        bbox, frame_out = detectar_persona(frame_out, modelo)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            
            # -------------------------------------------------------
            # EXPANSIÓN: crear ROI ampliado (alcance del brazo)
            # -------------------------------------------------------
            ancho_persona = x2 - x1
            
            # SOLO EXPANDIR LATERALES (40% del ancho a cada lado = mitad del 80% anterior)
            # NO expandir arriba ni abajo
            margen_horizontal = int(ancho_persona * 0.4)
            
            # Calcular ROI expandido (SOLO en horizontal)
            roi_x1 = max(0, x1 - margen_horizontal)
            roi_y1 = y1  # SIN EXPANSIÓN arriba
            roi_x2 = min(w_frame, x2 + margen_horizontal)
            roi_y2 = y2  # SIN EXPANSIÓN abajo
            
            # Extraer ROI expandido
            roi_expandido = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # DEBUG: Dibujar ROI expandido (opcional, comentar si molesta)
            cv2.rectangle(frame_out, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)  # Cyan

            # -------------------------------------------------------
            # 2) Detectar folio en ROI expandido
            # -------------------------------------------------------
            rect_folio = detectar_folio_en_roi(roi_expandido)

            if rect_folio is not None:
                # Ajustar coordenadas al frame completo
                box = cv2.boxPoints(rect_folio).astype(np.int32)
                box[:, 0] += roi_x1
                box[:, 1] += roi_y1

                cv2.drawContours(frame_out, [box], 0, (0, 0, 255), 3)

                # -------------------------------------------------------
                # 3) Calcular altura
                # -------------------------------------------------------
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

        # Mostrar frame final
        cv2.imshow(window_name, frame_out)

        # -------------------------------------------------------
        # CONTROLES DE TECLADO
        # -------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        
        elif key == ord('s'):
            # Guardar captura en archivo
            nombre_archivo = f"captura_{contador_capturas:03d}.png"
            cv2.imwrite(nombre_archivo, frame_out)
            print(f"✓ Captura guardada: {nombre_archivo}")
            contador_capturas += 1
        
        elif key == ord('c'):
            # Copiar al portapapeles (requiere xclip en Linux)
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
