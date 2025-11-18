# Cálculo de la altura de una persona usando un objeto de referencia

Proyecto basado en el paper de Takeda *et al.* [Calibration-Free Height Estimation for Person](https://onlinelibrary.wiley.com/doi/full/10.1002/tee.24077) para calcular la altura de una persona en una vídeo tomado por una webcam, utilizando un objeto de referencia de altura conocida (en nuestro caso, un folio A4).

## Contenidos

### 1. Requisitos previos

- Python **3.12.x** instalado. (Versiones más recientes pueden no ser compatibles con las dependencias en las versiones especificadas)
- `git` (opcional, solo si clonas el repositorio).


### 2–4. Instalación y prueba

<details>
<summary>Instrucciones para probar el detector de personas en imágenes</summary>

#### 2. Crear entorno virtual `env`

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

#### 3. Instalar dependencias

Con el entorno `env` activado, instala los requisitos:

```bash
pip install -r requirements.txt
```

Asegúrate también de que el archivo de pesos `best.pt` está en la raíz del proyecto (ya incluido en este repositorio).

#### 4. Probar el detector

El script principal es `human-detector.py`. Solo necesita la ruta de una imagen como argumento y abrirá una ventana con las detecciones dibujadas.

Ejemplo de uso (con entorno activado):

```bash
python human-detector.py fotos-prueba/person4.jpg
```

Puedes sustituir `fotos-prueba/person4.jpg` por la ruta a cualquier otra imagen donde aparezca al menos una persona.

</details>

