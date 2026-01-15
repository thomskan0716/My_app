# 📋 Plan de Implementación: Análisis No Lineal

## 🎯 Objetivos Divididos en Fases

### **FASE 1: Preparación y Habilitación del Botón** ✅
**Objetivo:** Activar el botón 非線形解析 y conectarlo al handler básico

**Tareas:**
1. Habilitar el botón 非線形解析 (línea ~2935 de 0sec.py)
2. Conectar el click al handler `on_nonlinear_analysis_clicked`
3. Crear función básica que muestre mensaje de "en desarrollo"

**Archivos a modificar:**
- `0sec.py` (línea 2935-2937 y crear handler básico)

---

### **FASE 2: Estructura de Salida y Gestión de Carpetas** ✅
**Objetivo:** Crear la lógica para generar carpetas de salida `04_非線形回帰\NUM_FECHA_HORA`

**Tareas:**
1. Función para obtener el número correlativo más alto de carpetas existentes
2. Función para crear carpeta con formato `NUM_FECHA_HORA` (ejemplo: `01_20250115_143022`)
3. Estructura dentro de la carpeta:
   ```
   NUM_FECHA_HORA/
   ├── 01_model_builder_output/
   │   ├── models/
   │   ├── results/
   │   └── graphs/
   ├── 02_prediction_output/
   └── 03_pareto_output/
   ```

**Archivos a crear:**
- `nonlinear_folder_manager.py` (nuevo módulo)

**Archivos a modificar:**
- `0sec.py` (importar y usar el folder manager)

---

### **FASE 3: Guardar Datos Filtrados en Formato Correcto** ✅
**Objetivo:** Convertir los datos filtrados a Excel con las columnas que esperan los scripts

**Tareas:**
1. Extraer datos filtrados de la BBDD
2. Mapear columnas de la BBDD a las columnas esperadas por config.py:
   - `送り速度`, `切込量`, `突出し量`, `載せ率`, `回転速度`, `パス数`, `UPカット`
3. Agregar las columnas objetivo: `摩耗量`, `上面ダレ量`, `側面ダレ量`
4. Guardar como `20250925_総実験データ.xlsx` en la carpeta de salida

**Archivos a crear:**
- `nonlinear_data_preparer.py` (convierte datos filtrados a formato esperado)

**Archivos a modificar:**
- `0sec.py` (usar data preparer antes de ejecutar scripts)

---

### **FASE 4: Configuración Mínima - Dynamizar config.py** ✅
**Objetivo:** Hacer que config.py acepte parámetros dinámicos de entrada

**Tareas:**
1. Modificar `config.py` para que las rutas sean dinámicas:
   ```python
   # Antes:
   DATA_FOLDER = '01_データセット'
   INPUT_FILE = '20250925_総実験データ.xlsx'
   RESULT_FOLDER = '回帰_0817_DCV_shap'
   
   # Después: Acceso dinámico que puede ser cambiado
   _DATA_FOLDER = None  # Se setea dinámicamente
   _INPUT_FILE = None
   _RESULT_FOLDER = None
   ```
2. Crear funciones helper para setear paths dinámicamente
3. Mantener compatibilidad con uso existente

**Archivos a modificar:**
- `config.py` (minimal changes - solo rutas)

---

### **FASE 5: Worker para Ejecución en Background** ✅
**Objetivo:** Crear worker que ejecuta 01_model_builder.py en background

**Tareas:**
1. Crear clase `NonlinearWorker(QThread)` similar a `LinearAnalysisWorker`
2. Worker debe:
   - Preparar datos filtrados
   - Configurar paths en config.py dinámicamente
   - Ejecutar `01_model_builder.py` usando `subprocess` o importar y ejecutar
   - Capturar 3 gráficos de salida
   - Emitir progreso
3. Manejar errores y logging

**Archivos a crear:**
- `nonlinear_worker.py`

**Consideraciones:**
- Los scripts originales NO se modifican
- Se ejecutan como subproceso o se importan dinámicamente
- Rutas temporales para ejecución

---

### **FASE 6: Diálogo de Configuración de Parámetros** ✅
**Objetivo:** UI para configurar parámetros de config.py (líneas 24-50, 96-183, 228-262)

**Parámetros a configurar:**

**Grupo 1: Modelos (Líneas 24-50)**
- `MODELS_TO_USE`: Checkboxes para cada modelo
- `FALLBACK_FINAL_MODEL`: Combo box
- `N_TRIALS`: Spinbox (número de trials de Optuna)

**Grupo 2: Configuración General (Líneas 96-183)**
- `TARGET_COLUMNS`: Checkboxes
- `USE_CORRELATION_REMOVAL`: Checkbox
- `CORRELATION_THRESHOLD`: Spinbox
- `DEFAULT_TOP_K`: Spinbox
- `USE_MANDATORY_FEATURES`: Checkbox
- `TRANSFORM_METHOD`: Combo box
- `OUTER_SPLITS`: Spinbox
- `INNER_SPLITS`: Spinbox
- `SHAP_MODE`: Combo box
- `SHAP_MAX_SAMPLES`: Spinbox

**Grupo 3: Pareto (Líneas 228-262)**
- `PARETO_OBJECTIVES`: Lista de checkboxes con optimización dir (min/max)
- `PARETO_PLOT_*`: Parámetros de visualización

**Archivos a crear:**
- `nonlinear_config_dialog.py` (UI completa con tabs por grupos)

**Archivos a modificar:**
- `0sec.py` (llamar diálogo antes de ejecutar worker)

---

### **FASE 7: Visor de Gráficos con OK/NG** ✅
**Objetivo:** Pantalla que muestra los 3 gráficos con navegación y botones OK/NG

**Características:**
- Imagen grande centrada del gráfico actual
- Flechas ← → para navegar entre gráficos
- Indicador "Gráfico 1 de 3"
- Botones:
  - **OK**: Continúa con 02 y 03
  - **NG**: Cancela y termina

**Archivos a crear:**
- `graph_viewer_dialog.py`

**Archivos a modificar:**
- `0sec.py` (mostrar diálogo después de ejecutar 01)

---

### **FASE 8: Ejecución de 02_prediction.py** ✅
**Objetivo:** Ejecutar script de predicción con datos de entrada preparados

**Tareas:**
1. Preparar archivo `Prediction_input.xlsx` en carpeta `03_予測`
2. Ejecutar `02_prediction.py` con los paths configurados
3. Capturar archivo de salida `Prediction_output.xlsx`

**Archivos a modificar:**
- `nonlinear_worker.py` (agregar método para ejecutar 02)
- `0sec.py` (llamar ejecución de 02 después de OK)

---

### **FASE 9: Ejecución de 03_pareto_analyzer.py** ✅
**Objetivo:** Ejecutar análisis de Pareto y generar resultados finales

**Tareas:**
1. Ejecutar `03_pareto_analyzer.py` usando el output de 02
2. Guardar gráficos y Excel de Pareto en carpeta correspondiente
3. Mostrar mensaje de finalización

**Archivos a modificar:**
- `nonlinear_worker.py` (agregar método para ejecutar 03)
- `0sec.py` (llamar ejecución de 03, mostrar mensaje final)

---

### **FASE 10: Integración Completa y Testing** ✅
**Objetivo:** Integrar todo el flujo y probar end-to-end

**Flujo completo:**
1. Usuario aplica filtros en pantalla de filtros
2. Click en 非線形解析
3. Aparece diálogo de configuración
4. Usuario configura parámetros y click "Continuar"
5. Se crea carpeta de salida
6. Se preparan datos filtrados
7. Se ejecuta 01_model_builder.py en background con progreso
8. Al terminar, aparece visor de gráficos
9. Usuario navega entre gráficos
10. Click OK → Se ejecuta 02_prediction.py
11. Se ejecuta 03_pareto_analyzer.py
12. Mensaje de finalización con ubicación de resultados

**Archivos a modificar:**
- `0sec.py` (integración completa del flujo)

---

## 📊 Resumen de Archivos

### Nuevos archivos a crear:
1. `nonlinear_folder_manager.py` - Gestión de carpetas
2. `nonlinear_data_preparer.py` - Preparación de datos
3. `nonlinear_worker.py` - Ejecución en background
4. `nonlinear_config_dialog.py` - UI de configuración
5. `graph_viewer_dialog.py` - Visor de gráficos
6. `PLAN_IMPLEMENTACION_NONLINEAR.md` - Este archivo

### Archivos a modificar:
1. `0sec.py` - Integración principal
2. `config.py` - Paths dinámicos (mínimos cambios)

### Archivos NO modificados:
1. `01_model_builder.py` - Se ejecuta tal cual
2. `02_prediction.py` - Se ejecuta tal cual
3. `03_pareto_analyzer.py` - Se ejecuta tal cual

---

## 🚀 Orden Recomendado de Ejecución

**Empezar con:**
1. FASE 1 (Habilitar botón) - **Más fácil, da feedback inmediato**
2. FASE 2 (Gestión de carpetas) - **Fundamental para todo lo demás**
3. FASE 3 (Preparación de datos) - **Necesario para ejecutar scripts**

**Continuar con:**
4. FASE 4 (Config dinámico) - **Necesario para rutas**
5. FASE 5 (Worker básico) - **Permite ejecución en background**
6. FASE 6 (Diálogo de config) - **Agrega configurabilidad**

**Finalizar con:**
7. FASE 7 (Visor de gráficos) - **UX importante**
8. FASE 8 y 9 (Ejecución 02 y 03) - **Completa el flujo**
9. FASE 10 (Integración y testing) - **Prueba final**

---

## ⚠️ Notas Importantes

1. **Mínimos cambios a scripts originales**: Solo config.py necesita cambios menores para paths dinámicos
2. **Ejecución como subproceso**: Alternativa a modificar scripts - ejecutar como subprocess
3. **Thread safety**: Los workers deben manejar QThread correctamente
4. **Paths absolutos**: Evitar rutas relativas, usar paths absolutos
5. **Temp cleanup**: Limpiar archivos temporales después de ejecución
6. **Error handling**: Manejar errores en cada fase del flujo







