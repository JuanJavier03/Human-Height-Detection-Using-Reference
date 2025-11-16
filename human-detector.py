"""
Script sencillo para detectar personas en una imagen usando YOLOv8.

Objetivo:
- Recibir solo la ruta de una imagen como argumento.
- Detectar personas en la imagen con un modelo ya entrenado (`best.pt`).
- Dibujar las cajas de detección sobre la imagen.
- Mostrar el resultado en una ventana, sin guardar nada en disco.
- Guardar en una variable las coordenadas de las cajas de detección
  e imprimirlas por consola (para usarlas en una segunda fase).

Uso desde la terminal (ejemplos):
    python human-detector.py foto_persona.jpg
    python human-detector.py ruta/a/mi_imagen.png

Requisitos:
- Tener instalado el paquete `ultralytics` (YOLOv8).
- Tener instalado `opencv-python` para mostrar la ventana.
- Tener el archivo de pesos `best.pt` en la misma carpeta que este script,
  o ajustar la constante PESOS_YOLO más abajo.
"""

import sys
from typing import List, Tuple

import cv2
from ultralytics import YOLO


# Ruta de los pesos del modelo YOLO que se van a usar.
# Se asume que el archivo `best.pt` está en la misma carpeta que este script.
PESOS_YOLO = "best.pt"


def detectar_personas_y_mostrar(ruta_imagen: str) -> List[Tuple[float, float, float, float]]:
    """
    Carga una imagen, ejecuta YOLOv8 para detectar personas y muestra
    el resultado en una ventana.

    Además, devuelve y escribe por consola las coordenadas de las cajas
    de detección en formato (x1, y1, x2, y2).

    Parámetros
    ----------
    ruta_imagen : str
        Ruta de la imagen sobre la que se quiere hacer la detección.

    Devuelve
    --------
    List[Tuple[float, float, float, float]]
        Lista de cajas de detección (una por persona detectada).
    """

    # Cargar el modelo YOLO con los pesos entrenados.
    modelo = YOLO(PESOS_YOLO)

    # Ejecutar la detección sobre la imagen.
    # - conf: umbral de confianza mínimo para dibujar una detección.
    # - iou: umbral de solapamiento para filtrar cajas muy parecidas.
    resultados = modelo(ruta_imagen, conf=0.55, iou=0.7)

    # `resultados` es una lista; para una sola imagen usamos el primer elemento.
    resultado = resultados[0]

    # Extraer las cajas de detección en formato (x1, y1, x2, y2).
    # `boxes.xyxy` es un tensor; lo convertimos a lista de listas de floats.
    cajas_xyxy: List[List[float]] = resultado.boxes.xyxy.tolist() if resultado.boxes is not None else []

    # Convertimos cada caja a tupla para que sea más cómoda de manejar.
    cajas: List[Tuple[float, float, float, float]] = [
        (x1, y1, x2, y2) for (x1, y1, x2, y2) in cajas_xyxy
    ]

    # Imprimir por consola las cajas detectadas (bordes de detección).
    # Esto será útil para la segunda parte (detección del folio / referencia).
    if cajas:
        print("Cajas de detección encontradas (x1, y1, x2, y2) en píxeles:")
        for i, (x1, y1, x2, y2) in enumerate(cajas, start=1):
            print(f"  Persona {i}: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
    else:
        print("No se han detectado personas en la imagen.")

    # Dibujar las cajas y etiquetas de detección directamente sobre la imagen.
    # `plot()` devuelve un array de imagen (tipo NumPy, en BGR) con las cajas pintadas.
    imagen_con_cajas = resultado.plot()

    # Mostrar la imagen en una ventana usando OpenCV.
    # La ejecución se detiene hasta que pulses una tecla dentro de la ventana.
    cv2.imshow("Detección de personas (YOLOv8)", imagen_con_cajas)
    print("Ventana abierta: pulsa cualquier tecla sobre la ventana para cerrarla.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cajas


def main() -> None:
    """
    Punto de entrada del script.

    Lógica de argumentos:
    - No se aceptan opciones ni flags.
    - Solo se espera UN argumento posicional: la ruta de la imagen.
    """

    # `sys.argv` contiene: [nombre_script, ruta_imagen]
    if len(sys.argv) != 2:
        print("Uso incorrecto.\n")
        print("Forma correcta de ejecutar:")
        print("    python human-detector.py ruta/de/la_imagen.jpg")
        sys.exit(1)

    ruta_imagen = sys.argv[1]

    try:
        _ = detectar_personas_y_mostrar(ruta_imagen)
    except FileNotFoundError:
        print(f"Error: no se encontró la imagen en la ruta: {ruta_imagen}")
        sys.exit(1)
    except Exception as e:
        print("Ocurrió un error inesperado durante la detección:")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()

