"""
Script para realizar detección de personas en tiempo real usando la webcam
con un modelo YOLOv8 previamente entrenado.

Objetivos:
- Abrir la webcam del sistema y capturar vídeo en directo.
- Aplicar detección de personas fotograma a fotograma.
- Mostrar en pantalla los resultados con las cajas dibujadas.
- Mostrar el número total de personas detectadas en cada frame.
- Imprimir por consola los vértices de las cajas detectadas (x1, y1, x2, y2),
  igual que en el script human-detector.py.
- Cerrar el programa pulsando la tecla 'q'.

Uso:
    python human-detector-webcam.py

Requisitos:
- ultralytics (YOLOv8)
- opencv-python
- Archivo de pesos YOLO: `best.pt` en la misma carpeta.
"""

import cv2
from ultralytics import YOLO

# Ruta del modelo
PESOS_YOLO = "best.pt"


def detectar_webcam() -> None:
    """
    Ejecuta la detección de personas en tiempo real usando la webcam.
    """

    # Cargar el modelo YOLO entrenado
    modelo = YOLO(PESOS_YOLO)

    # Inicializar webcam
    cap = cv2.VideoCapture(0)

    # Ajustar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: no se pudo acceder a la webcam.")
        return

    print("Detección iniciada. Pulsa 'q' para salir.\n")

    # Bucle principal de vídeo
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar un frame de la webcam.")
            break

        # Predicción frame a frame, con los mismos parámetros de precisión del script original
        resultados = modelo(frame, conf=0.55, iou=0.7, stream=True)

        # Procesar las detecciones obtenidas en este frame
        for resultado in resultados:
            frame_anotado = resultado.plot()

            # Extraer las cajas (x1, y1, x2, y2)
            cajas_xyxy = resultado.boxes.xyxy.tolist() if resultado.boxes is not None else []
            cajas = [(x1, y1, x2, y2) for (x1, y1, x2, y2) in cajas_xyxy]

            # Imprimir las detecciones por consola igual que en human-detector.py
            if cajas:
                print("---- DETECCIONES EN ESTE FRAME ----")
                for i, (x1, y1, x2, y2) in enumerate(cajas, start=1):
                    print(f"Persona {i}: ({x1:.2f}, {y1:.2f}) -> ({x2:.2f}, {y2:.2f})")
            else:
                print("Sin detecciones en este frame.")

            # Contar personas detectadas
            num_personas = len(cajas)

            # Mostrar contador en pantalla
            cv2.putText(
                frame_anotado,
                f"Personas: {num_personas}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            # Mostrar frame
            cv2.imshow("YOLOv8 - Detección de Humanos (Webcam)", frame_anotado)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detección finalizada.")


def main() -> None:
    """
    Punto de entrada del script. No recibe argumentos.
    """
    print("Cargando modelo y preparando webcam...")
    detectar_webcam()


if __name__ == "__main__":
    main()
