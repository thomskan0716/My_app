# Análisis: Importación a Base de Datos para Análisis No Lineal

## 📋 Resumen del Requerimiento

Después de que termine `03_pareto_analyzer.py`, se debe mostrar:
1. **Pantalla con gráficos** de los resultados de Pareto
2. **Botón "戻る"** (Volver)
3. **Botón "データベースにインポート"** (Importar a Base de Datos)

Al presionar "データベースにインポート", debe:
- Usar la misma lógica que el análisis lineal (Yosoku)
- Usar el archivo: `\04_非線形回帰\100_20251111_165705\03_予測\Prediction_output.xlsx`
- Pedir al usuario: **線材長**, **直径**, **tipo de cepillo** (A13/A11/A21/A32), **材料**
- Insertar los datos en la base de datos `yosoku_predictions.db`

---

## 🔍 Análisis del Flujo Actual

### 1. Flujo del Análisis No Lineal (Actual)

**Ubicación**: `nonlinear_worker.py`

```python
# Después de Stage 03 (Pareto Analyzer)
def run_stage2_and_3(self):
    # ...
    success_03 = self._run_script("03_pareto_analyzer.py", self.output_folder)
    
    if not success_03:
        self.error.emit("❌ Error en Stage 03: Pareto Analyzer")
        return
    
    # Análisis completado
    results_final = {
        'stage': 'completed',
        'output_folder': self.output_folder,
        'all_stages_completed': True
    }
    
    self.finished.emit(results_final)  # ← Aquí termina actualmente
```

**Problema**: Actualmente solo emite `finished` pero no muestra gráficos ni opción de importar.

---

### 2. Flujo del Análisis Lineal (Yosoku) - Referencia

**Ubicación**: `0sec.py`

#### 2.1. Diálogo de Parámetros (Líneas 9297-9392)

```python
def show_yosoku_parameters_dialog(self):
    """Muestra diálogo para seleccionar parámetros antes de importar"""
    dialog = QDialog(self)
    dialog.setWindowTitle("予測パラメーター選択")
    
    # Campos del formulario:
    # - brush_combo: A13, A11, A21, A32
    # - diameter_combo: 6, 15, 25, 40, 60, 100
    # - material_combo: Steel, Alum
    # - wire_length_combo: 30-75 (intervalos de 5)
    
    if result == QDialog.Accepted:
        selected_params = {
            'brush': brush_combo.currentData(),
            'diameter': diameter_combo.currentData(),
            'material': material_combo.currentData(),
            'wire_length': wire_length_combo.currentData()
        }
        return selected_params
```

#### 2.2. Importación a Base de Datos (Líneas 9753-9830)

```python
def import_yosoku_results_to_database(self, excel_path):
    """Importa resultados de Yosoku a la base de datos"""
    # 1. Muestra diálogo de parámetros
    selected_params = self.show_yosoku_parameters_dialog()
    
    # 2. Crea worker para importación
    worker = YosokuImportWorker(excel_path, self)
    
    # 3. Ejecuta importación
    # ...
```

#### 2.3. Worker de Importación (Líneas 986-1340)

**Clase**: `YosokuImportWorker`

**Proceso**:
1. Crea carpeta temporal y copia el Excel
2. Convierte fórmulas a valores (usando xlwings o openpyxl)
3. Lee datos del Excel
4. Conecta a `yosoku_predictions.db`
5. Crea/actualiza tabla `yosoku_predictions`
6. Inserta datos con `INSERT OR REPLACE` (sobreescribe duplicados)

**Estructura de la tabla**:
```sql
CREATE TABLE IF NOT EXISTS yosoku_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    A13 INTEGER,
    A11 INTEGER,
    A21 INTEGER,
    A32 INTEGER,
    直径 REAL,
    材料 TEXT,
    線材長 REAL,
    回転速度 REAL,
    送り速度 REAL,
    UPカット INTEGER,
    切込量 REAL,
    突出量 REAL,
    載せ率 REAL,
    パス数 INTEGER,
    加工時間 REAL,
    上面ダレ量 REAL,
    側面ダレ量 REAL,
    摩耗量 REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Índice único** (para evitar duplicados):
```sql
CREATE UNIQUE INDEX idx_unique_yosoku 
ON yosoku_predictions (
    A13, A11, A21, A32, 直径, 材料, 線材長, 回転速度, 
    送り速度, UPカット, 切込量, 突出量, 載せ率, パス数, 加工時間
)
```

**Datos que se insertan**:
- Los parámetros del usuario (A13/A11/A21/A32, 直径, 材料, 線材長) se agregan a cada fila
- Los datos operacionales (回転速度, 送り速度, etc.) vienen del Excel
- Las predicciones (上面ダレ量, 側面ダレ量, 摩耗量) vienen del Excel

---

## 🎯 Componentes a Implementar

### 1. Diálogo de Resultados de Pareto

**Ubicación sugerida**: Crear nuevo archivo `pareto_results_dialog.py` o agregar a `graph_viewer_dialog.py`

**Componentes necesarios**:
- Visualizador de gráficos de Pareto (ya existe en `03_pareto_analyzer.py`)
- Botón "戻る" (Volver)
- Botón "データベースにインポート" (Importar a Base de Datos)

**Gráficos a mostrar**:
- Los gráficos se generan en: `{OUTPUT_FOLDER}/04_パレート解/pareto_plots/`
- Formato: `pareto_{x_logical}__vs__{y_logical}.png`

### 2. Integración con NonlinearWorker

**Modificar**: `nonlinear_worker.py`, método `run_stage2_and_3()`

**Después de Stage 03**:
```python
# En lugar de solo emitir finished, emitir resultados con información de gráficos
results_final = {
    'stage': 'completed',
    'output_folder': self.output_folder,
    'all_stages_completed': True,
    'pareto_plots_folder': os.path.join(self.output_folder, '04_パレート解', 'pareto_plots'),
    'prediction_output_file': os.path.join(self.output_folder, '03_予測', 'Prediction_output.xlsx')
}
```

### 3. Reutilización de Código

#### 3.1. Diálogo de Parámetros
**Reutilizar**: `show_yosoku_parameters_dialog()` de `0sec.py` (líneas 9297-9392)
- ✅ Ya existe y funciona correctamente
- ✅ Pide exactamente los datos necesarios: brush, diameter, material, wire_length

#### 3.2. Worker de Importación
**Reutilizar**: `YosokuImportWorker` de `0sec.py` (líneas 986-1340)
- ✅ Ya existe y funciona correctamente
- ✅ Usa el mismo archivo Excel (`Prediction_output.xlsx`)
- ✅ Inserta en la misma base de datos (`yosoku_predictions.db`)
- ⚠️ **Modificación necesaria**: El worker actual espera que el Excel ya tenga las columnas A13, A11, A21, A32, 直径, 材料, 線材長 pre-llenadas. Para el análisis no lineal, estas columnas NO existen en `Prediction_output.xlsx`, por lo que hay que agregarlas antes de importar.

#### 3.3. Lógica de Agregar Columnas de Usuario

**Ubicación**: `YosokuImportWorker.run()` (líneas 1004-1340)

**Problema**: El Excel del análisis no lineal (`Prediction_output.xlsx`) tiene:
- ✅ Columnas operacionales: 回転速度, 送り速度, UPカット, 切込量, 突出量, 載せ率, パス数
- ✅ Columnas de predicción: prediction_上面ダレ量, prediction_側面ダレ量, prediction_摩耗量
- ❌ **NO tiene**: A13, A11, A21, A32, 直径, 材料, 線材長

**Solución**: Antes de insertar en la BD, agregar estas columnas con los valores del usuario:
```python
# En YosokuImportWorker.run(), después de leer el DataFrame:
# Agregar columnas de usuario a cada fila
df['A13'] = 1 if selected_params['brush'] == 'A13' else 0
df['A11'] = 1 if selected_params['brush'] == 'A11' else 0
df['A21'] = 1 if selected_params['brush'] == 'A21' else 0
df['A32'] = 1 if selected_params['brush'] == 'A32' else 0
df['直径'] = selected_params['diameter']
df['材料'] = selected_params['material']
df['線材長'] = selected_params['wire_length']
```

**También necesitamos**:
- Renombrar columnas de predicción: `prediction_上面ダレ量` → `上面ダレ量`
- Calcular `加工時間` si no existe: `100 / 送り速度 * 60`

---

## 📝 Plan de Implementación

### Paso 1: Crear Diálogo de Resultados de Pareto

**Archivo**: `pareto_results_dialog.py` (nuevo)

**Estructura**:
```python
class ParetoResultsDialog(QDialog):
    def __init__(self, pareto_plots_folder, prediction_output_file, parent=None):
        # Mostrar gráficos de Pareto
        # Botón "戻る"
        # Botón "データベースにインポート"
        
    def import_to_database(self):
        # 1. Llamar a show_yosoku_parameters_dialog()
        # 2. Crear worker de importación
        # 3. Ejecutar importación
```

### Paso 2: Modificar NonlinearWorker

**Archivo**: `nonlinear_worker.py`

**Cambios**:
- Después de Stage 03, emitir información de gráficos y archivo de predicción
- En el handler de `finished` en `0sec.py`, mostrar el diálogo de resultados

### Paso 3: Modificar YosokuImportWorker (Opcional - Mejora)

**Archivo**: `0sec.py`, clase `YosokuImportWorker`

**Cambios**:
- Agregar parámetro `selected_params` al constructor
- Si `selected_params` está presente, agregar columnas de usuario al DataFrame
- Renombrar columnas de predicción si tienen prefijo `prediction_`
- Calcular `加工時間` si no existe

**Alternativa** (sin modificar YosokuImportWorker):
- Crear función helper que prepare el DataFrame antes de pasarlo al worker
- Esta función agrega las columnas de usuario y renombra columnas

### Paso 4: Integrar en 0sec.py

**Archivo**: `0sec.py`

**Cambios**:
- En el handler de `nonlinear_worker.finished`, verificar si `stage == 'completed'`
- Si es así, mostrar `ParetoResultsDialog`
- El diálogo maneja la importación usando los métodos existentes

---

## 🔧 Archivos a Modificar/Crear

### Nuevos Archivos
1. `pareto_results_dialog.py` - Diálogo para mostrar resultados de Pareto

### Archivos a Modificar
1. `nonlinear_worker.py` - Agregar información de gráficos en `results_final`
2. `0sec.py` - Handler de `nonlinear_worker.finished` para mostrar diálogo
3. `0sec.py` - (Opcional) Modificar `YosokuImportWorker` para aceptar `selected_params`

---

## 📊 Estructura de Datos

### Archivo de Entrada: `Prediction_output.xlsx`

**Columnas esperadas**:
- Columnas operacionales: 回転速度, 送り速度, UPカット, 切込量, 突出量, 載せ率, パス数
- Columnas de predicción: `prediction_上面ダレ量`, `prediction_側面ダレ量`, `prediction_摩耗量` (o sin prefijo)
- **NO tiene**: A13, A11, A21, A32, 直径, 材料, 線材長, 加工時間

### Archivo de Salida: Base de Datos `yosoku_predictions.db`

**Datos insertados**:
- A13, A11, A21, A32: Del usuario (valores 0/1)
- 直径, 材料, 線材長: Del usuario
- 回転速度, 送り速度, UPカット, 切込量, 突出量, 載せ率, パス数: Del Excel
- 加工時間: Calculado (`100 / 送り速度 * 60`) si no existe
- 上面ダレ量, 側面ダレ量, 摩耗量: Del Excel (renombradas si tienen prefijo `prediction_`)

---

## ✅ Checklist de Implementación

- [ ] Crear `pareto_results_dialog.py` con visualizador de gráficos
- [ ] Agregar botones "戻る" y "データベースにインポート"
- [ ] Modificar `nonlinear_worker.py` para incluir información de gráficos
- [ ] Modificar handler en `0sec.py` para mostrar diálogo después de Stage 03
- [ ] Reutilizar `show_yosoku_parameters_dialog()` para pedir datos al usuario
- [ ] Crear función helper para preparar DataFrame (agregar columnas de usuario)
- [ ] Reutilizar `YosokuImportWorker` para importar a BD
- [ ] Probar flujo completo: Stage 03 → Diálogo → Importación

---

## 🔍 Puntos de Atención

1. **Ruta del archivo**: El archivo está en `{output_folder}/03_予測/Prediction_output.xlsx`
2. **Gráficos**: Están en `{output_folder}/04_パレート解/pareto_plots/`
3. **Renombrado de columnas**: Verificar si las columnas de predicción tienen prefijo `prediction_`
4. **Cálculo de 加工時間**: Solo calcular si no existe en el Excel
5. **Validación**: Verificar que el archivo Excel existe antes de importar

---

## 💡 Optimizaciones

1. **Reutilizar código existente**: Usar `show_yosoku_parameters_dialog()` y `YosokuImportWorker` sin modificar
2. **Función helper**: Crear `prepare_dataframe_for_import(df, selected_params)` que:
   - Agrega columnas de usuario
   - Renombra columnas de predicción
   - Calcula 加工時間
3. **Mismo flujo**: Mantener el mismo flujo que el análisis lineal para consistencia





