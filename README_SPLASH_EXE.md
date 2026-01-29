## Splash al inicio (PySide6) + build a EXE

### Qué se cambió
- **`bootstrap.py`**: entrypoint liviano que muestra un splash y **luego** importa `0sec.py` (que es lo que tarda).
- **Splash**:
  - imagen: **`SlpashScreen_General.png`** (sin fondo adicional)
  - esquinas: redondeadas (transparencia real en las esquinas)
- **`app_paths.py`**: helper `resource_path()` para que los assets funcionen igual en dev y en EXE.
- **`0sec.py`**: el icono ahora usa `resource_path("xebec_logo_88.png")`.

### Cómo correr en PyCharm (para ver el splash)
Ejecuta **`bootstrap.py`** (en vez de `0sec.py`).

### Build EXE (PowerShell)
Desde esta carpeta (la raíz del proyecto):

- **onedir (recomendado por velocidad de arranque)**:
  - `./build_exe_onedir.ps1`

- **onefile**:
  - `./build_exe_onefile.ps1`

Ambos scripts:
- usan `Scripts/python.exe` (tu venv local)
- instalan/actualizan PyInstaller automáticamente (puedes evitarlo con `-NoInstall`)
- incluyen estos assets mínimos: `SlpashScreen_General.png`, `SplashScreen.scale-100.png` (fallback), `xebec_logo_88.png`, `loading.gif` y (si existe) `xebec.ico`

### Salida
- onedir: `dist/0_00sec/0_00sec.exe`
- onefile: `dist/0_00sec.exe`


