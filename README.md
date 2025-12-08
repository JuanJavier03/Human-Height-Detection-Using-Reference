# Human Height Detection Using A4 Reference – Webcam Version
### YOLOv8-Seg + OpenCV + Geometría en tiempo real

Este proyecto implementa un sistema en tiempo real capaz de **medir la altura real de una persona utilizando un folio A4 como referencia**, aplicando técnicas avanzadas de visión artificial y segmentación mediante **YOLOv8-Seg**.

El sistema combina:
- Detección humana por **segmentación**
- Detección de folio A4 con corrección de perspectiva
- Análisis geométrico avanzado (Z y L)
- Corrección por suela del calzado
- Filtro por mediana para estabilidad en tiempo real
- Funcionamiento en tiempo real con webcam

---

# 1. Características principales

✔ Detección precisa del contorno del cuerpo gracias a **YOLOv8-Seg**  
✔ Extracción del **punto más alto** y **más bajo** de la silueta  
✔ Detección y corrección del **folio A4 (ratio 1.414)**  
✔ Cálculo de altura real mediante:
- Distancia persona–cámara (**Z**)
- Corrección de adelanto del pie (**L**)
- Corrección de la suela  
✔ Estimación estable gracias a **mediana móvil**  
✔ Funciona con cualquier webcam

---

# 2. Requisitos

## 2.1 Software necesario
- **Python 3.12.x** (versión probada: 3.12.6)  
- **pip**
- (**Opcional**) **git** para clonar repositorio

## 2.2 Dependencias principales
- `opencv-python`
- `ultralytics` (YOLOv8)
- `numpy`

Instalación automática desde `requirements.txt` (ver sección 3).

## 2.3 Modelo requerido (IMPORTANTE)

El sistema necesita obligatoriamente un archivo de pesos de **YOLOv8-Seg**.  
El utilizado por defecto es:

```
yolov8s-seg.pt
```

Debe estar **en la raíz del proyecto**.  

Si se desea usar otro modelo (por ejemplo `yolov8n-seg.pt`, `yolov8m-seg.pt`, etc.) debe **modificarse la línea del script**:

```python
PESOS_YOLO = "yolov8s-seg.pt"
```

---

# 3. Instalación

## 3.1 Crear entorno virtual
### Windows (PowerShell/CMD)
```bash
py -3.12 -m venv env
```

### Linux / macOS
```bash
python3.12 -m venv env
```

## 3.2 Activar entorno

### Windows PowerShell
```bash
.\env\Scripts\Activate.ps1
```

### Windows CMD
```bash
env\Scripts\activate.bat
```

### Linux / macOS
```bash
source env/bin/activate
```

## 3.3 Instalar dependencias
```bash
pip install -r requirements.txt
```

---

# 4. Ejecución del sistema

El script principal es:

```
detector-altura-webcam.py
```

Ejemplo básico:

```bash
python detector-altura-webcam.py --Z 2.10
```

Ejemplo completo con parámetros:

```bash
python detector-altura-webcam.py --Z 2.10 --L 0.10 --suela 2.0
```

---

# ⚙️ 5. Parámetros del script

| Parámetro | Tipo  | Obligatorio | Descripción |
|----------|-------|-------------|-------------|
| `--Z`    | float | ✔ Sí        | Distancia cámara–torso (m). |
| `--L`    | float | ✖ No        | Adelanto del pie respecto al torso (m). Default = 0.15 |
| `--suela`| float | ✖ No        | Corrección por suela del calzado (cm). Default = 1.0 |

---

# 6. Funcionamiento en tiempo real

Al ejecutarse, el sistema:

1. Abre la webcam
2. Detecta la persona mediante **YOLOv8-Seg**
3. Obtiene la máscara completa del cuerpo
4. Extrae:
   - Punto más alto real  
   - Punto más bajo real  
5. Detecta el folio A4 dentro del área filtrada
6. Corrige perspectiva para recuperar ratio 1.414
7. Calcula la altura usando:
   - Z  
   - Modelo geométrico Z/(Z – L)  
   - Corrección de suela  
8. Filtra la altura mediante mediana móvil
9. Muestra en pantalla:
   - Bounding box real del cuerpo  
   - Folio detectado  
   - Altura estimada  
   - Altura filtrada

Salir: **tecla Q**

---

# 7. Notas importantes

- El archivo `yolov8s-seg.pt` **debe estar en la raíz o el script no funcionará**  
- Si se usa otro modelo YOLOv8-Seg, **editar el código**  
- La cámara debe ser estática  
- El folio debe estar recto y visible  
- Una medición incorrecta de Z dará errores sistemáticos  
- Evitar iluminación amarilla o sombras intensas

---

# 8. Solución de Problemas

### No detecta persona
✔ Aumenta la iluminación  
✔ Asegúrate de estar completamente dentro del encuadre  

### No detecta folio
✔ No usar folios doblados  
✔ Evitar superficies blancas muy cercanas  
✔ Revisar iluminación

### La altura varía mucho
✔ Revisa que Z esté bien medida  
✔ Reduce el movimiento corporal  
✔ Coloca la cámara fija  
✔ Mantén el folio recto y perpendicular

---

# 9. Estructura recomendada del proyecto

```
/Human-Height-Detection
│
├── detector-altura-webcam.py
├── yolov8s-seg.pt
├── requirements.txt
├── README.md
└── imgs/ (opcional)
```

---

# 10. Créditos
- Juan Javier Sánchez Portillo
- Darío López Villegas
- Paulo Felipe Rezende da Silva

Proyecto desarrollado para la asignatura de Procesamiento de Imágenes Digitales.  
Utiliza tecnología YOLOv8-Seg de **Ultralytics** y técnicas avanzadas de visión artificial con **OpenCV**.

---

# 11. Licencia
Este proyecto es de uso académico.  
Consulta la licencia del repositorio original de YOLOv8 para uso comercial o redistribución.
