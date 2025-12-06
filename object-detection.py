"""
Prueba de concepto: Inferencia en tiempo real con Webcam.

Objetivo:
- Validar el rendimiento de YOLOv8 sobre un flujo de video en vivo (Webcam).
- Probar el modelo generalista `yolov8n.pt` (Nano) capaz de detectar 80 clases 
  (personas, móviles, sillas, etc.) del dataset COCO.
- Implementar una interfaz básica de usuario con un botón interactivo de STOP.

Uso:
    python object-detection.py

Requisitos:
    - Cámara web conectada.
    - ultralytics, opencv-python.
    - Conexión a internet (primera vez) para descargar `yolov8n.pt`.
"""

import cv2
from ultralytics import YOLO

# Variable global para controlar el estado del bucle desde el evento del ratón.
stop_requested = False


def add_stop_button(window_name):
    """
    Configura el manejador de eventos del ratón para detectar clics
    en el botón virtual de 'STOP'.
    """
    def mouse_callback(event, x, y, flags, param):
        global stop_requested
        # Si se hace clic izquierdo (LBUTTONDOWN)
        if event == cv2.EVENT_LBUTTONDOWN:
            # Coordenadas definidas para el botón (x: 10-110, y: 10-50)
            if 10 <= x <= 110 and 10 <= y <= 50:
                print("Botón STOP presionado. Cerrando aplicación...")
                stop_requested = True

    # Vincular la función callback a la ventana especificada
    cv2.setMouseCallback(window_name, mouse_callback)


def main():
    global stop_requested

    print("Cargando modelo YOLOv8 Nano (generalista)...")
    # Usamos 'yolov8n.pt': Versión 'Nano', más rápida y ligera, ideal para CPU/Webcam.
    model = YOLO('yolov8n.pt')

    # Iniciar captura de video (0 suele ser la webcam integrada)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo acceder a la webcam.")
        return

    # Crear la ventana antes del bucle para poder asignarle el botón
    nombre_ventana = "YOLOv8 Inferencia en Tiempo Real"
    cv2.namedWindow(nombre_ventana)
    add_stop_button(nombre_ventana)

    print("Iniciando bucle de video. Pulsa el botón STOP en pantalla o 'q' para salir.")

    while cap.isOpened():
        # Leer un frame de la cámara
        success, frame = cap.read()

        # Si falla la lectura o se pidió parar, rompemos el bucle
        if not success or stop_requested:
            break

        # Ejecutar la detección sobre el frame actual
        # YOLOv8 devuelve una lista de resultados (uno por frame)
        results = model(frame, verbose=False) # verbose=False para no saturar la consola

        # `plot()` dibuja las cajas y etiquetas de todas las clases detectadas
        annotated_frame = results[0].plot()

        # --- DIBUJAR INTERFAZ (Botón STOP) ---
        # Rectángulo rojo de fondo
        cv2.rectangle(annotated_frame, (10, 10), (110, 50), (0, 0, 255), -1)
        # Texto "STOP" en blanco
        cv2.putText(annotated_frame, 'STOP', (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Mostrar el resultado final
        cv2.imshow(nombre_ventana, annotated_frame)

        # Mecanismo de seguridad: Salir también con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Liberar recursos al terminar
    cap.release()
    cv2.destroyAllWindows()
    print("Aplicación finalizada correctamente.")


if __name__ == "__main__":
    main()