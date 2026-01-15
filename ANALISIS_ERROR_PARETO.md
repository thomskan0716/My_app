# Análisis Detallado del Error en Análisis de Pareto

## 📋 Resumen del Problema

Cuando ejecutas "yosoku" (predicción) desde la pantalla de gráficos del análisis no lineal, ocurre el siguiente error:

- ✅ **02_prediction.py**: Se ejecuta correctamente
- ❌ **03_pareto_analyzer.py**: Falla en la línea 159 al intentar crear un `ExcelWriter` con el engine `xlsxwriter`

### Error Específico
```
File: 03_pareto_analyzer.py, Line 159
with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
```

El error ocurre en la inicialización de `xlsxwriter` dentro de `pandas.io.excel._xlsxwriter.py` línea 197.

---

## 🔍 Causas Posibles

### 1. **Problema de Entorno en Subproceso** ⚠️ (MÁS PROBABLE)
Cuando `nonlinear_worker.py` ejecuta el script mediante `subprocess.Popen`, el entorno puede diferir del entorno principal:

- **PYTHONPATH**: Puede no incluir correctamente las rutas donde está instalado `xlsxwriter`
- **Variables de entorno**: Las variables configuradas en `nonlinear_worker.py` pueden interferir
- **Directorio de trabajo**: El script se ejecuta en `working_dir` que puede no tener acceso a las librerías del venv

### 2. **Conflicto de DLLs** 🔴
El código ya tiene detección de conflictos de DLLs (`dll_debug.py`). Posibles conflictos:
- Múltiples runtimes de OpenMP cargados simultáneamente
- DLLs de Qt/PySide6 interfiriendo con xlsxwriter
- DLLs de MKL/OpenBLAS causando problemas de inicialización

### 3. **Permisos de Escritura** 📝
El archivo Excel se intenta guardar en:
```
{OUTPUT_FOLDER}/04_パレート解/pareto_frontier.xlsx
```
- La carpeta puede no existir o no tener permisos de escritura
- El archivo puede estar abierto en otro programa (Excel)

### 4. **Instalación Corrupta de xlsxwriter** 🔧
Aunque funciona en el entorno principal, puede no estar correctamente instalado para el subproceso:
- Instalación incompleta en el venv
- Versión incompatible con pandas
- Módulos faltantes o corruptos

### 5. **Problema de Inicialización de xlsxwriter** ⚙️
El error en `pandas.io.excel._xlsxwriter.py` línea 197 sugiere:
- Error al importar `Workbook` de xlsxwriter
- Problema al inicializar el objeto Workbook
- Falta de recursos del sistema al crear el writer

---

## ✅ Soluciones (SIN MODIFICAR 03_pareto_analyzer.py)

### Solución 0: Mejora Implementada en nonlinear_worker.py ⭐⭐ (YA APLICADA)

**Se ha mejorado `nonlinear_worker.py`** para solucionar el problema:

1. **Inclusión de site-packages en PYTHONPATH**: Ahora el subproceso incluye automáticamente las rutas de `site-packages` del venv, asegurando que `xlsxwriter` y otras librerías se encuentren correctamente.

2. **Variable KMP_DUPLICATE_LIB_OK**: Se agregó para evitar conflictos de DLLs OpenMP que pueden interferir con la inicialización de xlsxwriter.

**Esta mejora debería resolver el problema sin necesidad de modificar `03_pareto_analyzer.py`.**

Si el problema persiste después de esta mejora, prueba las siguientes soluciones:

### Solución 1: Verificar y Reinstalar xlsxwriter ⭐ (RECOMENDADO)

```powershell
# Activar el entorno virtual
.\.venv\Scripts\Activate.ps1

# Desinstalar y reinstalar xlsxwriter
pip uninstall xlsxwriter -y
pip install xlsxwriter --upgrade

# Verificar instalación
python -c "import xlsxwriter; print('OK:', xlsxwriter.__version__)"
python -c "import pandas as pd; writer = pd.ExcelWriter('test.xlsx', engine='xlsxwriter'); writer.close(); import os; os.remove('test.xlsx'); print('ExcelWriter OK')"
```

### Solución 2: Verificar Permisos y Rutas

1. **Verificar que la carpeta de salida existe y tiene permisos:**
   - Navega a la carpeta del proyecto donde se ejecuta el análisis
   - Verifica que la carpeta `04_パレート解` puede crearse/escribirse
   - Asegúrate de que no hay archivos Excel abiertos en esa ubicación

2. **Verificar el directorio de trabajo:**
   - El script se ejecuta desde `working_dir` (output_folder)
   - Asegúrate de que ese directorio tiene acceso al venv

### Solución 3: Ajustar Variables de Entorno en nonlinear_worker.py

El problema puede estar en cómo se configuran las variables de entorno. Aunque no quieres modificar `03_pareto_analyzer.py`, puedes ajustar `nonlinear_worker.py`:

**Ubicación**: `nonlinear_worker.py`, método `_run_script()` (líneas 283-302)

**Ajustes sugeridos:**
- Asegurar que `PYTHONPATH` incluye el sitio-packages del venv
- Agregar `KMP_DUPLICATE_LIB_OK=TRUE` para evitar conflictos de DLLs
- Verificar que `sys.executable` apunta al Python correcto del venv

### Solución 4: Verificar Conflictos de DLLs

Ejecuta un diagnóstico de DLLs antes de ejecutar el análisis:

```python
# Agregar al inicio de nonlinear_worker.py (solo para diagnóstico)
from dll_debug import print_dll_report
print_dll_report("Before Pareto Analysis")
```

Esto te ayudará a identificar si hay conflictos de DLLs que puedan estar causando el problema.

### Solución 5: Usar Python del Venv Explícitamente

En `nonlinear_worker.py`, línea 326, verifica que `sys.executable` apunta al Python del venv:

```python
# En lugar de:
[sys.executable, script_path]

# Asegurar que es el Python del venv:
python_exe = Path(sys.executable).resolve()
# Verificar que está en .venv
```

### Solución 6: Verificar Versiones Compatibles

Asegúrate de tener versiones compatibles:

```powershell
pip list | findstr -i "pandas xlsxwriter"
```

Versiones recomendadas:
- `pandas >= 1.3.0`
- `xlsxwriter >= 3.0.0`

### Solución 7: Ejecutar en Modo Debug

Para obtener más información del error, puedes modificar temporalmente `nonlinear_worker.py` para capturar el stderr completo:

```python
# En _run_script(), después de process.wait():
if returncode != 0:
    # Leer stderr completo
    stderr_output = process.stderr.read() if process.stderr else ""
    print(f"STDERR: {stderr_output}")
```

---

## 🔧 Diagnóstico Rápido

Ejecuta estos comandos para diagnosticar:

```powershell
# 1. Verificar xlsxwriter
python -c "import xlsxwriter; print('xlsxwriter:', xlsxwriter.__version__)"

# 2. Verificar pandas
python -c "import pandas as pd; print('pandas:', pd.__version__)"

# 3. Probar ExcelWriter directamente
python -c "import pandas as pd; import os; writer = pd.ExcelWriter('test_pareto.xlsx', engine='xlsxwriter'); writer.close(); os.remove('test_pareto.xlsx'); print('✅ ExcelWriter funciona')"

# 4. Verificar permisos en la carpeta de salida
# Navega a: Archivos_de_salida\Proyecto_XX\04_非線形回帰\XXX_YYYYMMDD_HHMMSS
# Intenta crear un archivo Excel manualmente
```

---

## 📊 Orden de Prioridad para Solucionar

1. **Primero**: Solución 1 (Reinstalar xlsxwriter) - Más probable que resuelva el problema
2. **Segundo**: Solución 3 (Ajustar variables de entorno) - Si el problema persiste
3. **Tercero**: Solución 4 (Diagnóstico de DLLs) - Para identificar conflictos
4. **Cuarto**: Solución 2 (Verificar permisos) - Si hay problemas de acceso
5. **Último recurso**: Solución 7 (Modo debug) - Para obtener más información

---

## ⚠️ Notas Importantes

- **NO se modifica `03_pareto_analyzer.py`** como solicitaste
- Todas las soluciones son externas al script problemático
- Si ninguna solución funciona, el problema puede requerir modificar `nonlinear_worker.py` para mejorar el entorno de ejecución
- El error sugiere un problema de inicialización de xlsxwriter, no un problema de lógica del código

---

## 🆘 Si Nada Funciona

Si después de intentar todas las soluciones el problema persiste:

1. **Captura el error completo**: Modifica temporalmente `nonlinear_worker.py` para mostrar el traceback completo
2. **Verifica el entorno del subproceso**: Compara `sys.path` y variables de entorno entre el proceso principal y el subproceso
3. **Considera usar openpyxl como alternativa**: Aunque esto requeriría modificar `03_pareto_analyzer.py` (que no quieres hacer), podría ser una solución temporal

---

## 📝 Checklist de Verificación

- [ ] xlsxwriter está instalado y funciona en el entorno principal
- [ ] La carpeta de salida tiene permisos de escritura
- [ ] No hay archivos Excel abiertos en la carpeta de destino
- [ ] Las versiones de pandas y xlsxwriter son compatibles
- [ ] El Python del venv se está usando correctamente en el subproceso
- [ ] No hay conflictos de DLLs detectados
- [ ] PYTHONPATH incluye el sitio-packages del venv

