# Integración del Optimizador D-óptimo + I-óptimo

## Descripción

Se ha integrado exitosamente el nuevo optimizador que combina D-óptimo e I-óptimo en un solo botón, reemplazando la lógica separada de Dsaitekika e iSaitekika. El nuevo sistema ofrece funcionalidades mejoradas y visualizaciones más avanzadas.

## Archivos Nuevos

### 1. `integrated_optimizer.py`
- **Función principal**: `run_integrated_optimizer()`
- **Características**:
  - Combina D-óptimo e I-óptimo en una sola ejecución
  - Manejo avanzado de datos experimentales existentes
  - Reducción jerárquica de candidatos para grandes conjuntos de datos
  - Visualizaciones mejoradas con histogramas de características y UMAP
  - Optimización automática de hiperparámetros UMAP

### 2. `integrated_optimizer_worker.py`
- **Clase**: `IntegratedOptimizerWorker`
- **Propósito**: Worker para ejecutar el optimizador en hilos separados
- **Parámetros configurables**:
  - `sample_file`: Archivo de combinaciones de muestras
  - `existing_file`: Archivo de datos experimentales existentes (opcional)
  - `output_folder`: Carpeta de salida
  - `num_points`: Número de experimentos a seleccionar
  - `sample_size`: Tamaño máximo de muestra para reducción
  - `enable_hyperparameter_tuning`: Habilitar optimización de hiperparámetros
  - `force_reoptimization`: Forzar reoptimización

### 3. `integrated_config.py`
- **Propósito**: Configuración centralizada de parámetros
- **Parámetros principales**:
  - `DEFAULT_NUM_EXPERIMENTS = 15`
  - `DEFAULT_SAMPLE_SIZE = 5000`
  - `DEFAULT_ENABLE_HYPERPARAMETER_TUNING = True`
  - `DEFAULT_FORCE_REOPTIMIZATION = False`

## Modificaciones en Archivos Existentes

### 1. `0sec.py` (Archivo principal de GUI)

#### Importaciones Actualizadas
```python
from integrated_optimizer_worker import IntegratedOptimizerWorker
```

#### Función `on_combined_optimizer_clicked()` Actualizada
- Ahora usa `IntegratedOptimizerWorker` en lugar de `DIOptimizerWorker`
- Parámetros configurados para el nuevo optimizador
- Carpeta de salida cambiada a `_IntegratedResults`

#### Nueva Función `on_integrated_optimizer_finished()`
- Maneja los resultados del optimizador integrado
- Guarda tanto resultados D-óptimo como I-óptimo
- Muestra información detallada de resultados
- Genera visualizaciones mejoradas

#### Función `on_ok_clicked()` Mejorada
- Maneja tanto optimizador integrado como original
- Permite guardar D-óptimo, I-óptimo o ambos
- Interfaz de usuario mejorada con opciones múltiples

#### Función `on_ng_clicked()` Mejorada
- Opciones para reejecutar optimización integrada o individual
- Manejo diferenciado según el tipo de optimización ejecutada

## Características del Nuevo Optimizador

### 1. **Procesamiento de Datos Experimentales Existentes**
- Carga y validación automática de datos existentes
- Emparejamiento de alta precisión con puntos candidatos
- Verificación de calidad de datos (valores faltantes, duplicados)
- Diagnóstico automático de problemas de compatibilidad

### 2. **Reducción Jerárquica de Candidatos**
- Clustering automático para grandes conjuntos de datos
- Protección de puntos experimentales existentes
- Reducción inteligente manteniendo representatividad

### 3. **Algoritmos de Optimización Mejorados**
- **D-óptimo**: Cálculo numéricamente estable con métodos SVD/QR
- **I-óptimo**: Selección basada en distancias máximas
- Manejo de puntos experimentales existentes en ambos algoritmos

### 4. **Visualizaciones Avanzadas**
- **Histogramas de características**: Distribución de cada variable con colores diferenciados
- **UMAP mejorado**: Reducción de dimensionalidad con optimización automática de hiperparámetros
- **Comparación PCA vs UMAP**: Visualización lado a lado
- **Información detallada**: Estadísticas y parámetros de optimización

### 5. **Gestión de Archivos Mejorada**
- Múltiples archivos de salida:
  - `D_optimal_新規実験点.xlsx`: Nuevos puntos D-óptimo
  - `I_optimal_新規実験点.xlsx`: Nuevos puntos I-óptimo
  - `D_optimal_全実験点.xlsx`: Todos los puntos D-óptimo (existentes + nuevos)
  - `I_optimal_全実験点.xlsx`: Todos los puntos I-óptimo (existentes + nuevos)
  - `候補点一覧_v2.xlsx`: Lista completa de candidatos
- Gráficos de visualización en alta resolución

## Uso del Sistema Integrado

### 1. **Preparación de Datos**
- Generar archivo `sample_combinations.xlsx` con combinaciones de muestras
- Opcionalmente, preparar archivo `既存実験データ.xlsx` con datos experimentales existentes

### 2. **Ejecución**
- Cargar archivo de combinaciones de muestras
- Hacer clic en "最適化を実行" (Optimización integrada)
- El sistema ejecutará automáticamente D-óptimo e I-óptimo

### 3. **Resultados**
- Visualización automática de gráficos
- Tabla con resultados D-óptimo (por defecto)
- Navegación entre visualizaciones
- Opciones para guardar resultados

### 4. **Guardado de Resultados**
- **OK**: Guardar resultados (D-óptimo, I-óptimo o ambos)
- **NG**: Reejecutar optimización (integrada o individual)

## Parámetros Configurables

### En `integrated_config.py`:
```python
DEFAULT_NUM_EXPERIMENTS = 15          # Número de experimentos
DEFAULT_SAMPLE_SIZE = 5000            # Tamaño máximo de muestra
DEFAULT_ENABLE_HYPERPARAMETER_TUNING = True  # Optimización UMAP
DEFAULT_FORCE_REOPTIMIZATION = False  # Reoptimización forzada
```

### En la GUI:
- Los parámetros se pueden modificar en `on_combined_optimizer_clicked()`
- Actualmente configurados con valores por defecto optimizados

## Ventajas del Sistema Integrado

1. **Eficiencia**: Una sola ejecución para ambos algoritmos
2. **Consistencia**: Mismo conjunto de datos para ambas optimizaciones
3. **Comparabilidad**: Resultados directamente comparables
4. **Flexibilidad**: Opciones para guardar uno o ambos resultados
5. **Visualización**: Gráficos mejorados con información detallada
6. **Robustez**: Manejo avanzado de datos y errores
7. **Escalabilidad**: Reducción automática para grandes conjuntos de datos

## Compatibilidad

- **Mantiene compatibilidad** con el sistema original
- **Funciones originales** siguen disponibles
- **Migración gradual** posible
- **Configuración flexible** para diferentes necesidades

## Próximos Pasos

1. **Pruebas exhaustivas** con diferentes conjuntos de datos
2. **Optimización de parámetros** basada en resultados reales
3. **Documentación adicional** para usuarios finales
4. **Interfaz de configuración** para parámetros avanzados 