# Human Height Detection (YOLOv8)

Proyecto sencillo para detectar personas en una imagen usando YOLOv8 y mostrar el resultado en una ventana.

---

## 1. Requisitos previos

- Python **3.12.1** instalado.
- `git` (opcional, solo si clonas el repositorio).

---

## 2. Crear entorno virtual `env`

En la raíz del proyecto, ejecuta en PowerShell o terminal:

```bash
py -3.12 -m venv env
```

Activa el entorno virtual:

- En **Windows (PowerShell)**:
  ```bash
  .\env\Scripts\Activate.ps1
  ```
- En **Windows (cmd)**:
  ```bash
  env\Scripts\activate.bat
  ```
- En **Linux / macOS**:
  ```bash
  source env/bin/activate
  ```

---

## 3. Instalar dependencias

Con el entorno `env` activado, instala los requisitos:

```bash
pip install -r requirements.txt
```

Asegúrate también de que el archivo de pesos `best.pt` está en la raíz del proyecto (ya incluido en este repositorio).

---

## 4. Probar el detector

El script principal es `human-detector.py`. Solo necesita la ruta de una imagen como argumento y abrirá una ventana con las detecciones dibujadas.

Ejemplo de uso (con entorno activado):

```bash
python human-detector.py fotos-prueba/person4.jpg
```

Puedes sustituir `fotos-prueba/person4.jpg` por la ruta a cualquier otra imagen donde aparezca al menos una persona.

