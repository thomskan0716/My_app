# Análisis: Archivo Intermedio para Importación a BD de Yosoku

## 🔍 Flujo Actual

### Ubicación del código:
- **Método principal**: `import_nonlinear_pareto_to_database()` (línea 10324)
- **Método de preparación**: `prepare_dataframe_for_import()` (línea 10271)
- **Método de importación**: `import_yosoku_results_to_database()` (línea 10135)

### Flujo paso a paso:

1. **Usuario presiona "データベースにインポート"** en la pantalla de gráficos de Pareto
   - Se llama: `import_nonlinear_pareto_to_database(excel_path)`
   - `excel_path` = ruta completa a `Prediction_output.xlsx`
   - Ejemplo: `C:\Users\...\03_予測\Prediction_output.xlsx`

2. **Mostrar diálogo de parámetros** (línea 10328)
   - `selected_params = self.show_yosoku_parameters_dialog()`
   - Usuario ingresa: diámetro, material, longitud de alambre, tipo de cepillo

3. **Leer archivo Excel** (líneas 10335-10337)
   ```python
   df = pd.read_excel(excel_path)
   ```
   - Lee `Prediction_output.xlsx`
   - Contiene las predicciones del análisis de Pareto

4. **Preparar DataFrame** (línea 10340)
   ```python
   df_prepared = self.prepare_dataframe_for_import(df, selected_params)
   ```
   - **Agrega columnas de tipo de cepillo**:
     - `A13`, `A11`, `A21`, `A32` (binarias según el tipo seleccionado)
   - **Agrega columnas de usuario**:
     - `直径` (diámetro)
     - `材料` (material)
     - `線材長` (longitud de alambre)
   - **Renombra columnas** si tienen prefijo `prediction_`
   - **Calcula `加工時間`** si no existe (fórmula: 100 / 送り速度 * 60)

5. **Guardar en archivo temporal** (líneas 10342-10347)
   ```python
   import tempfile
   temp_dir = tempfile.gettempdir()
   temp_file = os.path.join(temp_dir, f"pareto_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
   df_prepared.to_excel(temp_file, index=False)
   ```
   - **Ubicación actual**: Carpeta temporal del sistema (ej: `C:\Users\...\AppData\Local\Temp\`)
   - **Nombre**: `pareto_import_YYYYMMDD_HHMMSS.xlsx`
   - **Problema**: Se guarda en carpeta temporal y se elimina después de 5 segundos

6. **Importar a BD** (línea 10350)
   ```python
   self.import_yosoku_results_to_database(temp_file)
   ```
   - Usa el archivo temporal para importar

7. **Limpiar archivo temporal** (líneas 10352-10361)
   - Se elimina después de 5 segundos

## 📋 Requerimiento del Usuario

**Quiere que cuando se crea el DataFrame preparado (`df_prepared`), se guarde un archivo intermedio en la misma carpeta que `Prediction_output.xlsx`.**

### Especificaciones:
- **Ubicación**: Misma carpeta que `Prediction_output.xlsx`
  - Si `excel_path` = `C:\Users\...\03_予測\Prediction_output.xlsx`
  - Entonces el archivo intermedio debe estar en: `C:\Users\...\03_予測\`
- **Contenido**: El DataFrame `df_prepared` (unión de `Prediction_output.xlsx` + datos adicionales)
- **Momento**: Después de preparar el DataFrame, antes de importar a BD
- **Propósito**: Tener un registro del DataFrame que se está importando

## 🔧 Cambios Necesarios

### 1. Obtener la carpeta del archivo original
```python
import os
from pathlib import Path

# Obtener carpeta donde está Prediction_output.xlsx
excel_folder = os.path.dirname(excel_path)
# O usar Path:
excel_folder = Path(excel_path).parent
```

### 2. Crear nombre para el archivo intermedio
```python
from datetime import datetime

# Opción 1: Con timestamp
intermediate_filename = f"Prediction_output_prepared_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

# Opción 2: Nombre fijo (sobrescribe si existe)
intermediate_filename = "Prediction_output_prepared.xlsx"

# Opción 3: Con sufijo antes de la extensión
base_name = Path(excel_path).stem  # "Prediction_output"
intermediate_filename = f"{base_name}_prepared.xlsx"
```

### 3. Guardar archivo intermedio
```python
intermediate_path = os.path.join(excel_folder, intermediate_filename)
# O con Path:
intermediate_path = excel_folder / intermediate_filename

df_prepared.to_excel(intermediate_path, index=False)
print(f"📁 Archivo intermedio guardado: {intermediate_path}")
```

### 4. Ubicación en el código

**Lugar exacto**: Después de la línea 10340 (después de `df_prepared = self.prepare_dataframe_for_import(df, selected_params)`)

**Código actual (líneas 10342-10347)**:
```python
# 4. Guardar DataFrame preparado en archivo temporal
import tempfile
temp_dir = tempfile.gettempdir()
temp_file = os.path.join(temp_dir, f"pareto_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
df_prepared.to_excel(temp_file, index=False)
print(f"📁 Archivo temporal creado: {temp_file}")
```

**Código modificado**:
```python
# 4. Guardar DataFrame preparado en archivo intermedio (misma carpeta que Prediction_output.xlsx)
from pathlib import Path
excel_folder = Path(excel_path).parent
intermediate_filename = f"Prediction_output_prepared_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
intermediate_path = excel_folder / intermediate_filename
df_prepared.to_excel(intermediate_path, index=False)
print(f"📁 Archivo intermedio guardado: {intermediate_path}")

# 5. Guardar también en archivo temporal para la importación
import tempfile
temp_dir = tempfile.gettempdir()
temp_file = os.path.join(temp_dir, f"pareto_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
df_prepared.to_excel(temp_file, index=False)
print(f"📁 Archivo temporal creado: {temp_file}")
```

## ⚠️ Consideraciones

1. **Archivo temporal**: Se mantiene porque `import_yosoku_results_to_database()` necesita un archivo para importar. El archivo intermedio es adicional, no reemplaza al temporal.

2. **Nombre del archivo**: 
   - Con timestamp: Evita sobrescribir si se importa múltiples veces
   - Sin timestamp: Más simple, pero sobrescribe el anterior

3. **Manejo de errores**: Si falla al guardar el archivo intermedio, no debería detener el proceso de importación.

4. **Limpieza**: El archivo intermedio NO se elimina automáticamente (a diferencia del temporal), queda como registro.

## 📝 Resumen

**Cambio requerido**: 
- Agregar código después de preparar `df_prepared` para guardar un archivo Excel en la misma carpeta que `Prediction_output.xlsx`
- El archivo contendrá el DataFrame con los datos originales + datos adicionales del usuario
- El archivo temporal se mantiene para la importación, el intermedio es adicional





