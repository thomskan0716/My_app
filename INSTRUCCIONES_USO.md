# 📖 Instrucciones de Uso: Análisis No Lineal

## 🚀 Inicio Rápido

### 1. Importar Datos
1. Abrir la aplicación `0sec.py`
2. Click en **"データベースにインポート"** (botón izquierdo)
3. Seleccionar archivo de resultados Excel

### 2. Aplicar Filtros
1. La aplicación automáticamente muestra vista de filtros
2. Configurar filtros deseados:
   - 実験日 (rango de fechas)
   - バリ除去 (0/1)
   - 上面ダレ量, 側面ダレ量 (rangos)
   - 摩耗量, 面粗度 (rangos)
   - Cepillos (A13, A11, A21, A32)
   - Parámetros de máquina
3. Click en **"分析"** para aplicar filtros
4. Verificar número de registros filtrados

### 3. Ejecutar Análisis No Lineal
1. Click en **"非線形解析"** (botón azul en vista de filtros)
2. Se abre diálogo de configuración con 3 pestañas:

#### **Pestaña 1: モデル設定 (Modelos)**
- Seleccionar modelos a usar (checkboxes):
  - Random Forest ✅
  - LightGBM ✅
  - XGBoost □
  - Gradient Boost □
  - Ridge □
  - Lasso □
  - Elastic Net □
- Configurar número de trials (Optuna): 50 (default)
- Seleccionar modelo de respaldo: ridge (default)

#### **Pestaña 2: 一般設定 (Configuración General)**
- **特徴量設定:**
  - top_k: 20 (número de características)
  - 相関閾値: 0.95
  - 相関除去機能: ✅ activado
- **変数変換:**
  - auto (automático)
- **クロスバリデーション:**
  - 外側分割数: 10
  - 内側分割数: 10
- **SHAP分析:**
  - detailed (modo detallado)
  - 最大サンプル数: 200

#### **Pestaña 3: パレート設定 (Pareto)**
- Seleccionar objetivos con checkboxes:
  - 摩耗量: min ✅
  - 切削時間: min ✅
  - 上面ダレ量: min ✅
  - 側面ダレ量: min ✅
- Cambiar dirección (min/max) si es necesario

3. Click en **"続行"** para continuar
4. Confirmar ejecución en diálogo

### 4. Observar Progreso
- Se muestra diálogo de progreso con chibi Xebec
- Progreso en tiempo real:
  - 10%: Preparando...
  - 20%: Preparando datos...
  - 40%: Iniciando entrenamiento de modelos...
  - 70%: Buscando gráficos...
  - 100%: ¡Completado!

### 5. Revisar Gráficos
- Aparece visor de gráficos automáticamente
- Navegar con flechas (← →)
- Contador muestra: "1 / 3", "2 / 3", "3 / 3"
- Botones:
  - **OK** (verde): Continuar con predicción y Pareto
  - **NG** (rojo): Detener y guardar resultados hasta aquí

### 6. Ver Resultados Finales
- Si OK: Se ejecutan automáticamente:
  - 02_prediction.py (predicciones)
  - 03_pareto_analyzer.py (análisis Pareto)
- Mensaje de finalización muestra:
  - Ubicación completa de resultados
  - Estructura de carpetas generada
  - Archivos creados

---

## 📂 Ubicación de Resultados

Los resultados se guardan en:
```
NOMBRE_DEL_PROYECTO/
└── 04_非線形回帰/
    └── 01_20250115_143022/
```

### Estructura de Carpetas

**01_データセット/**
- `20250925_総実験データ.xlsx` - Datos filtrados de entrada

**01_学習モデル/**
- `final_model_摩耗量.pkl` - Modelo para 摩耗量
- `final_model_上面ダレ量.pkl` - Modelo para 上面ダレ量
- `final_model_側面ダレ量.pkl` - Modelo para 側面ダレ量

**02_結果/**
- `摩耗量_results.png` - Gráfico de resultados
- `上面ダレ量_results.png` - Gráfico de resultados
- `側面ダレ量_results.png` - Gráfico de resultados
- `dcv_results.pkl` - Resultados completos

**03_グラフ/**
- Gráficos adicionales de SHAP

**04_予測/**
- `Prediction_input.xlsx` - Datos de entrada para predicción
- `Prediction_output.xlsx` - Resultados de predicción

**05_パレート解/**
- `pareto_frontier.xlsx` - Soluciones de Pareto
- `pareto_plots/` - Gráficos de Pareto

---

## ⚙️ Configuración Avanzada

### Ajustar Parámetros de Modelo
En el diálogo de configuración, puedes ajustar:

**Modelos:**
- Más modelos = más tiempo de ejecución
- Menos modelos = más rápido
- Recomendado: Random Forest + LightGBM

**Características:**
- top_k más alto = más características
- top_k más bajo = menos características
- Valor por defecto: 20

**CV Splits:**
- Más splits = más tiempo pero más robusto
- Menos splits = más rápido
- Por defecto: 10 outer / 10 inner

**SHAP:**
- detailed: Análisis completo (lento)
- summary: Análisis rápido
- none: Sin análisis SHAP

---

## 🐛 Solución de Problemas

### Error: "モジュールが見つかりません"
**Causa:** Archivos faltantes
**Solución:** Asegurar que existan:
- `nonlinear_worker.py`
- `nonlinear_folder_manager.py`
- `nonlinear_config_dialog.py`
- `graph_viewer_dialog.py`
- `config.py` (modificado)

### Error: "フィルタリングされたデータがありません"
**Causa:** No se aplicaron filtros
**Solución:** 
1. Ir a vista de filtros
2. Click en "分析" para aplicar filtros
3. Verificar número de registros

### Error: "Timeout"
**Causa:** Scripts tardan mucho tiempo
**Solución:**
- Reducir número de trials
- Seleccionar menos modelos
- Reducir top_k

### Gráficos no aparecen
**Causa:** 01_model_builder.py no generó gráficos
**Solución:**
- Verificar que config.py esté configurado correctamente
- Revisar logs en consola
- Verificar que los scripts originales funcionan

---

## 💡 Consejos de Uso

### Para Resultados Óptimos
1. **Datos:** Asegurar suficientes datos filtrados (≥ 50 registros)
2. **Filtros:** Aplicar filtros razonables, no muy restrictivos
3. **Modelos:** Usar Random Forest + LightGBM como mínimo
4. **Trials:** 50+ trials para mejor optimización
5. **SHAP:** Activar para entender importancia de características

### Para Resultados Rápidos
1. **Modelos:** Seleccionar solo 1 modelo (LightGBM)
2. **Trials:** 20-30 trials
3. **CV:** 5 outer / 5 inner splits
4. **SHAP:** Modo "summary" o "none"
5. **Features:** Reducir top_k a 10-15

### Para Análisis Detallado
1. **Modelos:** Seleccionar todos los modelos
2. **Trials:** 100+ trials
3. **CV:** 10 outer / 10 inner (default)
4. **SHAP:** Modo "detailed"
5. **Features:** Aumentar top_k a 30-50

---

## 📞 Soporte

Si tienes problemas:
1. Revisar logs en consola
2. Verificar que todos los archivos existen
3. Verificar que la carpeta de proyecto está configurada
4. Verificar que hay suficiente espacio en disco

---

## ✨ ¡Disfruta del Análisis No Lineal!

El sistema está completamente integrado y listo para usar. ¡Experimenta con diferentes configuraciones para obtener los mejores resultados!







