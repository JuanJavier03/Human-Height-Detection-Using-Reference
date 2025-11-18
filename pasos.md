### Pasos a implementar
1.- Detectar a la persona y al objeto de referencia en la imagen (YOLOv8)
    
  - [ ] TODO: Para esto es necesario un modelo entrenado que detecte ambos objetos.

2.- Calcula la profundidad de cada píxel en la imagen (MiDaS)

3.- Calcula la media de todos los píxeles de cada región detectada, eliminando los bordes (Segmentación Semántica)

4.- Quedarse con los frames del vídeo cuyos valores de profundidad entre la persona y el objeto de referencia sean casi idénticos (fórmula del umbral)
$$\left| P_d^{(k)} - O_d^{(k)} \right| \leq \delta$$

5.- Calcular la altura de la persona usando la fórmula del ratio:
$$\hat{H}^{(k)} = h \frac{P_h^{(k)}}{O_h^{(k)}}$$
