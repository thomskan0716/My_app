# Integración del Análisis Lineal con la Aplicación

## Resumen

Se ha conectado exitosamente el archivo `linear_analysis_advanced.py` a la aplicación principal `0sec.py` para que se ejecute cuando se presione el botón "線形解析" (Análisis Lineal) en la pantalla de filtros.

## Funcionalidades Implementadas

### 1. Conexión Principal
- **Función**: `run_advanced_linear_analysis_from_db(db_manager, filters, output_folder)`
- **Ubicación**: `linear_analysis_advanced.py` (líneas 2044-2301)
- **Propósito**: Ejecuta el análisis lineal completo usando datos filtrados de la base de datos

### 2. Manejo de Filtros
- **Filtros de cepillo**: A13, A11, A21, A32 (columnas booleanas)
  - **Filtro específico**: Selecciona registros con valor 1 en la columna específica
  - **Filtro "すべて"**: Selecciona registros con valor 1 en cualquiera de las columnas A13, A11, A21 o A32
- **Filtros de rango**: Valores numéricos con rangos min-max
- **Filtros de fecha**: Experimentos por fecha

### 3. Gestión de Base de Datos
- **Tabla principal**: `main_results` (568 registros disponibles)
- **Columnas de cepillo**: A13, A11, A21, A32 (valores 0/1)
- **Método de consulta**: `fetch_filtered()` con SQL dinámico
- **Verificación**: Detecta automáticamente tablas con datos disponibles

### 4. Estructura de Salida
```
output_folder/
├── 01_学習モデル/
│   ├── filtered_data.xlsx
│   └── best_model_*.pkl
├── 02_パラメーター/
│   ├── preprocessing_params.json
│   ├── prediction_formulas.json
│   └── encoders.pkl
├── 03_評価スコア/
│   ├── evaluation_scores.xlsx
│   └── 01_チャート/
│       └── regression_enhanced_*.png
├── 04_予測計算/
│   └── XEBEC_予測計算機_逆変換対応.xlsx
```

### 5. Características del Análisis
- **Transformaciones automáticas**: log, sqrt, boxcox, yeo-johnson
- **Selección de características**: Basada en importancia de Random Forest
- **Modelos lineales**: LinearRegression, Ridge, Lasso, ElasticNet
- **Validación cruzada**: Doble CV para robustez
- **Excel Calculator**: Con fórmulas de predicción e inversión

## Flujo de Ejecución

1. **Usuario presiona "線形解析"**
2. **Verificación de vista de filtros**
3. **Obtención de filtros aplicados**
4. **Consulta a base de datos con filtros**
5. **Verificación de tablas disponibles**
6. **Ejecución del pipeline de análisis**
7. **Generación de resultados y Excel Calculator**
8. **Visualización de resultados en la aplicación**

## Archivos Modificados

### `linear_analysis_advanced.py`
- ✅ Agregada función `run_advanced_linear_analysis_from_db()`
- ✅ Agregada función `_create_sample_data_and_analyze()`
- ✅ Agregada función `create_analysis_summary_table()`
- ✅ Manejo robusto de errores y casos edge

### `0sec.py`
- ✅ Actualizada función `show_linear_analysis_results()`
- ✅ Mejorada visualización de resultados
- ✅ Agregada tabla de resumen
- ✅ Información de Excel Calculator

### `db_manager.py`
- ✅ Agregado método `fetch_filtered()`
- ✅ Soporte para consultas SQL con parámetros

## Casos de Uso

### Caso 1: Datos Existentes
- Usuario aplica filtros
- Se ejecuta análisis con datos filtrados
- Se generan modelos y Excel Calculator

### Caso 2: Sin Datos
- Base de datos vacía o sin registros válidos
- Se muestra mensaje de error informativo
- Se solicita al usuario verificar datos disponibles

### Caso 3: Errores de Filtros
- Filtros inválidos
- Se manejan errores graciosamente
- Se muestran mensajes informativos

## Resultados Esperados

### Métricas de Modelos
- **Regresión**: R², MAE, RMSE
- **Clasificación**: Accuracy, F1-Score
- **Transformaciones**: log, sqrt, boxcox, yeo-johnson

### Archivos Generados
- **Excel Calculator**: Para predicciones en tiempo real
- **Gráficos**: Visualizaciones de resultados
- **Reportes**: Métricas detalladas
- **Modelos**: Archivos .pkl para reutilización

## Notas Técnicas

- **Compatibilidad**: Funciona con datos reales y simulados
- **Robustez**: Manejo de errores en múltiples niveles
- **Escalabilidad**: Soporta diferentes tamaños de datos
- **Mantenibilidad**: Código modular y documentado

## Próximos Pasos

1. **Optimización**: Mejorar rendimiento con datasets grandes
2. **Validación**: Pruebas con datos reales de producción
3. **Documentación**: Manual de usuario detallado
4. **Monitoreo**: Logs de ejecución y métricas de uso

