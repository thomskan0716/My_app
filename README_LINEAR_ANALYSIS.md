# Sistema de Análisis Lineal para 0.00sec

## 📋 Descripción

Este sistema integra análisis lineal de machine learning en la aplicación 0.00sec, permitiendo analizar datos de experimentos de procesamiento de materiales.

## 🚀 Funcionalidades

### ✅ **Análisis Lineal Integrado**
- **Regresión**: Para variables continuas (摩耗量, 上面ダレ量, 側面ダレ量)
- **Clasificación**: Para variables binarias (バリ除去)
- **Transformaciones automáticas**: Log, Box-Cox, etc.
- **Validación cruzada**: Para robustez del modelo

### ✅ **Filtrado Inteligente**
- Filtros por fecha, material, parámetros de proceso
- Aplicación de filtros antes del análisis
- Query dinámica a la base de datos

### ✅ **Resultados Completos**
- Modelos entrenados guardados como `.pkl`
- Gráficos de regresión y residuales
- Métricas de rendimiento (R², MAE, RMSE, F1)
- Reportes en Excel y JSON

## 🏗️ Arquitectura Modular

### **1. Módulo Principal (`linear_analysis_module.py`)**
- Clase `LinearAnalysisPipeline`: Pipeline completo de análisis
- Clase `LinearAnalysisConfig`: Configuración del sistema
- Función `run_linear_analysis_from_db`: Interfaz con la BD

### **2. Integración en 0.00sec (`0sec.py`)**
- Botón "線形解析" en el panel izquierdo
- Navegación automática a filtros
- Ejecución del análisis con filtros aplicados
- Visualización de resultados en la interfaz

## 📊 Variables del Sistema

### **Variables Objetivo (Target)**
| Variable | Tipo | Descripción |
|----------|------|-------------|
| バリ除去 | Clasificación | Eliminación de rebabas (0/1) |
| 摩耗量 | Regresión | Cantidad de desgaste |
| 上面ダレ量 | Regresión | Deformación de la superficie superior |
| 側面ダレ量 | Regresión | Deformación de la superficie lateral |

### **Variables de Características (Features)**
| Variable | Descripción |
|----------|-------------|
| 送り速度 | Velocidad de avance |
| UPカット | Corte superior |
| 切込量 | Profundidad de corte |
| 突出し量 | Cantidad de protrusión |
| 載せ率 | Tasa de carga |
| 回転速度 | Velocidad de rotación |
| パス数 | Número de pasadas |

## 🔧 Instalación y Configuración

### **1. Dependencias Requeridas**
```bash
pip install scikit-learn scipy matplotlib seaborn pandas numpy joblib
```

### **2. Archivos del Sistema**
- `linear_analysis_module.py` - Módulo de análisis
- `0sec.py` - Aplicación principal (ya modificada)
- `output_analysis/` - Directorio de salida

### **3. Estructura de Directorios**
```
0.00sec/
├── linear_analysis_module.py
├── 0sec.py
├── output_analysis/
│   ├── model_*.pkl
│   ├── regression_*.png
│   ├── analysis_results.xlsx
│   └── analysis_results.json
└── Archivos_pruebas/
    └── 線形モデル_回帰分離混合_Ver2_noA11A21A32.py
```

## 📱 Uso del Sistema

### **1. Acceso al Análisis Lineal**
1. Abrir la aplicación 0.00sec
2. Hacer clic en el botón "線形解析" (Análisis Lineal)
3. El sistema navegará automáticamente a la pantalla de filtros

### **2. Configuración de Filtros**
1. **実験日**: Rango de fechas de experimentos
2. **バリ除去**: Filtro por eliminación de rebabas
3. **上面ダレ量**: Rango de deformación superior
4. **側面ダレ量**: Rango de deformación lateral
5. **材料**: Tipo de material (Steel/Alumi)
6. **A13, A11, A21, A32**: Parámetros de herramienta

### **3. Ejecución del Análisis**
1. Configurar filtros deseados
2. Hacer clic en "線形解析" nuevamente
3. Confirmar la ejecución
4. Esperar a que se complete el análisis

### **4. Visualización de Resultados**
- **Resumen**: Estadísticas generales del análisis
- **Modelos**: Estado de cada modelo entrenado
- **Métricas**: R², MAE, RMSE para regresión; Accuracy, F1 para clasificación
- **Gráficos**: Predicción vs Real, Análisis de residuales

## 📁 Archivos de Salida

### **Modelos Entrenados**
- `model_バリ除去.pkl` - Modelo de clasificación
- `model_摩耗量.pkl` - Modelo de regresión
- `model_上面ダレ量.pkl` - Modelo de regresión
- `model_側面ダレ量.pkl` - Modelo de regresión

### **Gráficos**
- `regression_摩耗量.png` - Resultados de regresión
- `regression_上面ダレ量.png` - Resultados de regresión
- `regression_側面ダレ量.png` - Resultados de regresión

### **Reportes**
- `analysis_results.xlsx` - Resumen en Excel
- `analysis_results.json` - Datos técnicos en JSON

## 🔍 Mapeo de Nombres

### **Base de Datos → Análisis**
El sistema mapea automáticamente los nombres de columnas de la BD a los nombres del análisis:

```python
DB_TO_ANALYSIS_MAPPING = {
    '送り速度': '送り速度',
    'UPカット': 'UPカット', 
    '切込量': '切込量',
    '突出し量': '突出し量',
    '載せ率': '載せ率',
    '回転速度': '回転速度',
    'パス数': 'パス数'
}
```

## ⚠️ Consideraciones Importantes

### **1. Datos Mínimos**
- **Regresión**: Mínimo 10 muestras por objetivo
- **Clasificación**: Mínimo 5 muestras por clase

### **2. Valores Faltantes**
- Se rellenan automáticamente con la mediana
- Se excluyen columnas con >50% de valores faltantes

### **3. Transformaciones**
- Se aplican automáticamente según la distribución de datos
- Log, Box-Cox, Yeo-Johnson según sea apropiado

### **4. Rendimiento**
- Validación cruzada con 5 folds para velocidad
- Modelos lineales para interpretabilidad
- Guardado automático de resultados

## 🐛 Solución de Problemas

### **Error: "Módulo no encontrado"**
```bash
# Verificar que linear_analysis_module.py esté en el directorio correcto
ls -la linear_analysis_module.py
```

### **Error: "Dependencias faltantes"**
```bash
pip install scikit-learn scipy matplotlib seaborn
```

### **Error: "No hay datos válidos"**
- Verificar que los filtros no sean demasiado restrictivos
- Comprobar que la BD tenga datos en las columnas requeridas

### **Error: "Memoria insuficiente"**
- Reducir el número de folds de validación cruzada
- Usar filtros más específicos para reducir el dataset

## 📈 Mejoras Futuras

### **1. Funcionalidades Planificadas**
- Análisis no lineal (Random Forest, SVM)
- Selección automática de características
- Validación cruzada anidada
- Exportación a Excel con fórmulas

### **2. Optimizaciones**
- Paralelización del entrenamiento
- Caché de modelos pre-entrenados
- Interfaz web para resultados

### **3. Integración**
- API REST para análisis remoto
- Base de datos en la nube
- Reportes automáticos por email

## 📞 Soporte

Para problemas o preguntas sobre el sistema de análisis lineal:

1. **Revisar logs** en la consola de la aplicación
2. **Verificar archivos** de salida en `output_analysis/`
3. **Comprobar dependencias** con `pip list`
4. **Revisar mapeo** de nombres de columnas

---

**Desarrollado para 0.00sec - Sistema de Optimización de Muestras**
**Versión**: 1.0.0
**Fecha**: 2025-01-29

