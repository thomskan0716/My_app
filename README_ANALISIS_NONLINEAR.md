# 📖 README: Análisis No Lineal - Sistema Completo

## 🎯 Descripción

Sistema completo de análisis no lineal integrado en la aplicación 0.00sec. Permite ejecutar análisis de regresión no lineal con configuración personalizada, visualización de resultados, predicción y análisis de Pareto.

## ✨ Características Principales

- ✅ **Integración completa** con el sistema existente
- ✅ **Configuración personalizada** de parámetros
- ✅ **Visor de gráficos** con navegación OK/NG
- ✅ **Análisis automático** en 3 stages (01, 02, 03)
- ✅ **Gestión inteligente** de carpetas con numeración
- ✅ **Reutilización** de datos filtrados sin duplicación
- ✅ **Sin modificaciones** a scripts originales

## 🚀 Inicio Rápido

### 1. Preparar Datos
```
1. Abrir aplicación: python 0sec.py
2. Click "データベースにインポート"
3. Seleccionar archivo de resultados
4. Aplicar filtros deseados
5. Click "分析" para filtrar
```

### 2. Ejecutar Análisis No Lineal
```
1. Click en botón "非線形解析"
2. Configurar parámetros en el diálogo:
   - Tab 1: Seleccionar modelos
   - Tab 2: Configurar CV, SHAP, etc.
   - Tab 3: Configurar objetivos Pareto
3. Click "続行"
4. Confirmar ejecución
```

### 3. Revisar y Continuar
```
1. Revisar gráficos generados (visor)
2. Navegar con flechas ← →
3. Decidir: OK o NG
4. Si OK: Se ejecutan stages 02 y 03
5. Ver resultados finales
```

## 📂 Estructura del Sistema

### **Módulos de Integración**
- `nonlinear_folder_manager.py` - Gestión de carpetas
- `nonlinear_worker.py` - Worker de ejecución
- `nonlinear_config_dialog.py` - Diálogo de configuración
- `graph_viewer_dialog.py` - Visor de gráficos

### **Módulos de Soporte** (00_Pythonコード/)
- `feature_aware_augmentor.py` - Aumento de datos
- `data_analyzer.py` - Análisis exploratorio
- `core/preprocessing.py` - Preprocesamiento
- `core/utils.py` - Utilidades
- `models/model_factory.py` - Factory de modelos
- `shap_analysis/complete_shap.py` - Análisis SHAP

### **Scripts Originales** (sin modificar)
- `01_model_builder.py` - Construcción de modelos
- `02_prediction.py` - Predicción
- `03_pareto_analyzer.py` - Análisis Pareto

## 📊 Flujo de Datos

```
Usuario
  ↓
Aplica Filtros → self.filtered_df
  ↓
Click "非線形解析"
  ↓
Obtiene datos de BBDD (con filtros)
  ↓
Muestra diálogo de configuración
  ↓
Crea carpeta: 04_非線形回帰\NUM_FECHA_HORA
  ↓
Guarda datos filtrados
  ↓
Ejecuta 01_model_builder.py
  ↓
Muestra visor de gráficos
  ↓
Usuario hace OK → Ejecuta 02_prediction.py
  ↓
Ejecuta 03_pareto_analyzer.py
  ↓
Resultados finales
```

## 🎛️ Configuración Disponible

### **Modelos (Tab 1)**
- Random Forest
- LightGBM
- XGBoost
- Gradient Boost
- Ridge
- Lasso
- Elastic Net

### **General (Tab 2)**
- Características: top_k, corr_threshold
- Transformación: método de transformación
- CV: outer_splits, inner_splits
- SHAP: modo y max_samples

### **Pareto (Tab 3)**
- Objetivos: 摩耗量, 切削時間, 上面ダレ量, 側面ダレ量
- Direcciones: min/max por objetivo

## 📁 Estructura de Resultados

```
04_非線形回帰\
└── 01_20250115_143022\
    ├── 01_データセット\      (datos de entrada)
    ├── 01_学習モデル\        (modelos entrenados)
    ├── 02_結果\             (resultados y gráficos)
    ├── 03_グラフ\           (gráficos adicionales)
    ├── 04_予測\             (predicciones)
    └── 05_パレート解\       (análisis Pareto)
```

## ⚠️ Notas Importantes

### Dependencias
- ✅ scikit-learn
- ✅ numpy
- ✅ pandas
- ✅ optuna (para optimización)
- ⚠️ lightgbm (opcional)
- ⚠️ xgboost (opcional)
- ⚠️ shap (opcional)

### Compatibilidad
- ✅ Scripts originales sin cambios
- ✅ Compatible con análisis lineal
- ✅ Mismos datos para ambos análisis
- ✅ Sin duplicación de código

## 🐛 Solución de Problemas

### Error: "ModuleNotFoundError"
**Causa:** Módulos faltantes en `00_Pythonコード/`
**Solución:** Verificar que todos los módulos estén creados

### Error: "Script not found"
**Causa:** 01, 02, 03 no están en la carpeta actual
**Solución:** Los scripts deben estar en el directorio raíz del proyecto

### Error: "No filtered data"
**Causa:** Filtros muy restrictivos
**Solución:** Ajustar filtros o verificar datos en BBDD

## 📚 Documentación Adicional

- `INSTRUCCIONES_USO.md` - Instrucciones detalladas
- `MODULOS_CREADOS.md` - Documentación de módulos
- `IMPLEMENTACION_COMPLETA_FASE_1-10.md` - Detalles técnicos
- `RESUMEN_FINAL.md` - Resumen completo

## 🎉 ¡Listo para Usar!

El sistema está completamente funcional y listo para análisis no lineal en producción.

---

**Versión:** 1.0  
**Última actualización:** 2025-10-27  
**Estado:** ✅ Completo y funcional







