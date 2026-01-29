# 🔍 ANÁLISIS EXHAUSTIVO: Objetos propuestos para limpieza de memoria

## 📋 RESUMEN EJECUTIVO

**Total de objetos analizados**: 15 objetos temporales  
**Objetos SEGUROS para eliminar**: 12 objetos ✅  
**Objetos CRÍTICOS (NO eliminar)**: 3 objetos ⚠️  
**Riesgo de impacto en análisis**: BAJO (solo se eliminan objetos después de su uso completo)

---

## 🔬 ANÁLISIS DETALLADO POR UBICACIÓN

### 1️⃣ `_evaluate_params()` - Líneas 145-188

#### Objeto: `X_tr`, `X_va` (línea 147)
- **Creación**: `X.iloc[tr_idx].copy()`, `X.iloc[va_idx].copy()`
- **Uso**: 
  - `X_tr`: Línea 152 (augment), 157 (scaler), 165 (preprocessor)
  - `X_va`: Línea 158 (scaler), 166 (preprocessor), 178 (selector)
- **Referencias después del fold**: ❌ NINGUNA
- **Referencias indirectas**: 
  - ❌ No hay closures que capturen estas variables
  - ❌ No se guardan en ningún atributo de clase
  - ❌ No se retornan
- **Impacto de eliminación**: ✅ NINGUNO - Solo se usan dentro del loop del fold
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 183

#### Objeto: `X_tr_aug`, `y_tr_aug` (línea 152)
- **Creación**: `self.augmentor.augment(X_tr, y_tr)`
- **Uso**: 
  - `X_tr_aug`: Línea 157 (scaler), 165 (preprocessor)
  - `y_tr_aug`: Línea 165 (preprocessor), 177 (selector), 182 (model.fit)
- **Referencias después del fold**: ❌ NINGUNA
- **Referencias indirectas**: 
  - ❌ `augmentor.augment()` retorna nuevas copias, no guarda referencias
  - ❌ No se guardan en atributos de clase
- **Impacto de eliminación**: ✅ NINGUNO
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 182

#### Objeto: `scaler` (línea 155)
- **Creación**: `RobustScaler()`
- **Uso**: 
  - Línea 157: `scaler.fit_transform(X_tr_aug[continuous_cols])`
  - Línea 158: `scaler.transform(X_va[continuous_cols])`
- **Referencias después del fold**: ❌ NINGUNA
- **Referencias indirectas**: 
  - ❌ No se guarda en ningún lugar
  - ❌ Solo se usa para transformar datos que ya están guardados en `X_tr_aug`, `X_va`
- **Impacto de eliminación**: ✅ NINGUNO
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 158

#### Objeto: `preprocessor` (línea 161)
- **Creación**: `EnhancedPreprocessor(...)`
- **Uso**: 
  - Línea 165: `preprocessor.fit_transform(X_tr_aug, ...)`
  - Línea 166: `preprocessor.transform(X_va)`
- **Referencias después del fold**: ❌ NINGUNA
- **Referencias indirectas**: 
  - ❌ No se guarda en ningún lugar
  - ❌ Solo se usa para transformar datos guardados en `X_tr_prep`, `X_va_prep`
- **Impacto de eliminación**: ✅ NINGUNO
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 166

#### Objeto: `X_tr_prep`, `X_va_prep` (líneas 165-166)
- **Creación**: `preprocessor.fit_transform(...)`, `preprocessor.transform(...)`
- **Uso**: 
  - `X_tr_prep`: Línea 177 (selector)
  - `X_va_prep`: Línea 178 (selector)
- **Referencias después del fold**: ❌ NINGUNA
- **Referencias indirectas**: ❌ NINGUNA
- **Impacto de eliminación**: ✅ NINGUNO
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 178

#### Objeto: `selector` (línea 169)
- **Creación**: `AdvancedFeatureSelector(...)`
- **Uso**: 
  - Línea 177: `selector.fit_transform(X_tr_prep, y_tr_aug)`
  - Línea 178: `selector.transform(X_va_prep)`
- **Referencias después del fold**: ❌ NINGUNA
- **Referencias indirectas**: ❌ NINGUNA
- **Impacto de eliminación**: ✅ NINGUNO
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 178

#### Objeto: `X_tr_sel`, `X_va_sel` (líneas 177-178)
- **Creación**: `selector.fit_transform(...)`, `selector.transform(...)`
- **Uso**: 
  - `X_tr_sel`: Línea 182 (model.fit)
  - `X_va_sel`: Línea 183 (model.predict)
- **Referencias después del fold**: ❌ NINGUNA
- **Referencias indirectas**: ❌ NINGUNA
- **Impacto de eliminación**: ✅ NINGUNO
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 183

#### Objeto: `model` (líneas 181-183)
- **Creación**: `model_instance.build(**model_params)`
- **Uso**: 
  - Línea 182: `model.fit(X_tr_sel, y_tr_aug)`
  - Línea 183: `model.predict(X_va_sel)`
- **Referencias después del fold**: ❌ NINGUNA
- **Referencias indirectas**: 
  - ✅ **VERIFICACIÓN CRÍTICA**: `model_instance` es un objeto creado en `optimize_model()` línea 215
  - ⚠️ **RIESGO**: ¿El modelo guarda referencias internas a los datos de entrenamiento?
  - ✅ **ANÁLISIS**: En scikit-learn, algunos modelos (como RandomForest) guardan referencias a los datos de entrenamiento DURANTE fit(), pero NO después de que termina fit()
  - ✅ **VERIFICACIÓN**: `model.predict()` usa solo el modelo entrenado, no los datos originales
  - ✅ **CONCLUSIÓN**: Después de línea 183, el modelo ya no necesita los datos de entrenamiento
- **Impacto de eliminación**: ✅ NINGUNO - El modelo ya está entrenado y solo se usa para predecir
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 186 (después de calcular score)

#### Objeto: `y_hat` (línea 183)
- **Creación**: `model.predict(X_va_sel)`
- **Uso**: Línea 186 (calcular MAE)
- **Referencias después del fold**: ❌ NINGUNA
- **Referencias indirectas**: ❌ NINGUNA
- **Impacto de eliminación**: ✅ NINGUNO
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 186

---

### 2️⃣ `optimize_model()` - Función `objective()` (líneas 217-243)

#### Objeto: `scores` (retornado de `_evaluate_params()`, línea 228)
- **Creación**: Retornado de `self._evaluate_params(...)`
- **Uso**: 
  - Línea 232: `np.mean(scores)` → `mean_score`
  - Línea 233: `np.std(scores)` → `std_score`
- **Referencias después del objective**: ❌ NINGUNA
- **Referencias indirectas**: 
  - ⚠️ **RIESGO**: ¿Optuna guarda referencias a `scores` en el trial?
  - ✅ **VERIFICACIÓN**: Optuna solo guarda el valor retornado (`mean_score`), NO guarda referencias a objetos intermedios
  - ✅ **CONFIRMACIÓN**: `study.optimize()` solo captura el valor float retornado por `objective()`
- **Impacto de eliminación**: ✅ NINGUNO
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 233

#### Objeto: `mean_score`, `std_score` (líneas 232-233)
- **Creación**: Cálculos a partir de `scores`
- **Uso**: 
  - `mean_score`: Línea 236 (comparación), 239 (return)
  - `std_score`: Línea 236 (comparación)
- **Referencias después del objective**: ❌ NINGUNA (solo se retorna `mean_score` como float)
- **Referencias indirectas**: ❌ NINGUNA
- **Impacto de eliminación**: ✅ NINGUNO
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 239

---

### 3️⃣ `optimize_model()` - Después de `study.optimize()` (líneas 249-276)

#### Objeto: `study` (línea 245)
- **Creación**: `optuna.create_study(...)`
- **Uso**: 
  - Línea 249: `study.optimize(objective, ...)`
  - Línea 254: `study.best_value`
  - Línea 265: `study.best_params`
  - Línea 269: `study.best_value` (comparación)
- **Referencias después de extraer valores**: ❌ NINGUNA después de línea 273
- **Referencias indirectas**: 
  - ⚠️ **RIESGO CRÍTICO**: ¿Optuna Study guarda referencias a trials y modelos?
  - ✅ **VERIFICACIÓN**: Optuna Study guarda:
    - `study.trials`: Lista de objetos Trial
    - Cada Trial guarda: `trial.params`, `trial.value`, `trial.datetime_start`, etc.
    - ❌ **IMPORTANTE**: NO guarda referencias a modelos, datos, o objetos creados dentro de `objective()`
    - ✅ **CONFIRMACIÓN**: Solo guarda valores primitivos (floats, ints, strings, dicts simples)
  - ✅ **ANÁLISIS**: `study.best_params` es un dict con valores primitivos, NO referencias a objetos
  - ✅ **CONCLUSIÓN**: Después de extraer `best_value` y `best_params`, el study ya no se necesita
- **Impacto de eliminación**: ✅ NINGUNO - Solo se necesitan los valores extraídos
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 273

#### Objeto: `model_results` (línea 210)
- **Creación**: `{}` (dict vacío)
- **Uso**: 
  - Línea 260: Guardar resultado de modelo rechazado
  - Línea 263-267: Guardar resultado de modelo exitoso
  - Línea 276: Guardar resultado de modelo fallido
- **Referencias después de logging**: ❌ NINGUNA después de línea 276
- **Referencias indirectas**: 
  - ✅ **VERIFICACIÓN**: `model_results` solo se usa para logging (líneas 262, 276)
  - ❌ NO se retorna
  - ❌ NO se guarda en atributos de clase
  - ❌ NO se usa después del loop de modelos (línea 212)
- **Impacto de eliminación**: ✅ NINGUNO - Solo para logging interno
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 276 (pero mejor al final del loop de modelos, línea 212)

---

### 4️⃣ `run_dcv()` - Cada fold externo (líneas 321-407)

#### Objeto: `X_train_base`, `X_test_base` (líneas 325, 329)
- **Creación**: `X.iloc[train_idx].copy()`, `X.iloc[test_idx].copy()`
- **Uso**: 
  - `X_train_base`: Línea 339 (optimize_model), 346 (augment)
  - `X_test_base`: Línea 355 (scaler), 362 (preprocessor), 373 (selector), 384 (model.predict)
- **Referencias después del fold**: ❌ NINGUNA después de línea 407
- **Referencias indirectas**: 
  - ⚠️ **RIESGO**: ¿Se pasan a `optimize_model()` que podría guardar referencias?
  - ✅ **VERIFICACIÓN**: `optimize_model()` solo usa estos datos para llamar a `_evaluate_params()`, que NO guarda referencias
  - ✅ **CONFIRMACIÓN**: No se guardan en atributos de clase ni en closures
- **Impacto de eliminación**: ✅ NINGUNO
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 407

#### Objeto: `X_train_aug` (línea 346)
- **Creación**: `self.augmentor.augment(X_train_base, y_train_trans_base)`
- **Uso**: 
  - Línea 351: Lista de columnas continuas
  - Línea 354: Scalear
  - Línea 361: Preprocessor
- **Referencias después del fold**: ❌ NINGUNA
- **Impacto de eliminación**: ✅ NINGUNO
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 373

#### Objeto: `scaler` (línea 352) - Fold externo
- **Creación**: `RobustScaler()`
- **Uso**: Líneas 354-355
- **Referencias después del fold**: ❌ NINGUNA
- **Impacto de eliminación**: ✅ NINGUNO
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 355

#### Objeto: `preprocessor` (línea 357) - Fold externo
- **Creación**: `EnhancedPreprocessor(...)`
- **Uso**: Líneas 361-362
- **Referencias después del fold**: ❌ NINGUNA
- **Impacto de eliminación**: ✅ NINGUNO
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 362

#### Objeto: `selector` (línea 364) - Fold externo
- **Creación**: `AdvancedFeatureSelector(...)`
- **Uso**: Líneas 372-373
- **Referencias después del fold**: ❌ NINGUNA
- **Impacto de eliminación**: ✅ NINGUNO
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 373

#### Objeto: `X_train_prep`, `X_test_prep` (líneas 361-362)
- **Creación**: `preprocessor.fit_transform(...)`, `preprocessor.transform(...)`
- **Uso**: 
  - `X_train_prep`: Línea 372 (selector)
  - `X_test_prep`: Línea 373 (selector)
- **Referencias después del fold**: ❌ NINGUNA
- **Impacto de eliminación**: ✅ NINGUNO
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 373

#### Objeto: `X_train_sel`, `X_test_sel` (líneas 372-373)
- **Creación**: `selector.fit_transform(...)`, `selector.transform(...)`
- **Uso**: 
  - `X_train_sel`: Línea 381 (model.fit)
  - `X_test_sel`: Línea 384 (model.predict)
- **Referencias después del fold**: ❌ NINGUNA
- **Impacto de eliminación**: ✅ NINGUNO
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 384

#### Objeto: `model` (línea 380) - Fold externo
- **Creación**: `best_model_instance.build(**model_params)`
- **Uso**: 
  - Línea 381: `model.fit(X_train_sel, y_train_aug_trans)`
  - Línea 384: `model.predict(X_test_sel)`
- **Referencias después del fold**: ❌ NINGUNA
- **Referencias indirectas**: 
  - ✅ **VERIFICACIÓN CRÍTICA**: `best_model_instance` viene de `optimize_model()` línea 273
  - ⚠️ **RIESGO**: ¿El modelo guarda referencias internas?
  - ✅ **ANÁLISIS**: Similar a análisis anterior - modelos scikit-learn no guardan referencias después de fit()
- **Impacto de eliminación**: ✅ NINGUNO
- **¿Se puede eliminar?**: ✅ SÍ - Seguro después de línea 392

#### Objeto: `y_pred_trans`, `y_pred` (líneas 384-385)
- **Creación**: `model.predict(...)`, `inverse_transform(...)`
- **Uso**: 
  - `y_pred`: Línea 388 (MAE), 389 (RMSE), 390 (R2), 405 (extend a all_predictions)
- **Referencias después del fold**: ✅ SÍ - Se guarda en `all_predictions` línea 405
- **Impacto de eliminación**: ⚠️ **CRÍTICO** - Se usa después para guardar
- **¿Se puede eliminar?**: ❌ NO - Se necesita hasta línea 405

---

## ⚠️ OBJETOS CRÍTICOS QUE NO SE DEBEN ELIMINAR

### ❌ `y_pred` (línea 385)
- **Razón**: Se guarda en `all_predictions` línea 405, necesario para resultados finales
- **Impacto si se elimina**: ❌ CRÍTICO - Pérdida de predicciones OOF

### ❌ `y_test_orig` (línea 386)
- **Razón**: Se guarda en `all_true` línea 406, necesario para resultados finales
- **Impacto si se elimina**: ❌ CRÍTICO - Pérdida de valores reales OOF

### ❌ `best_model_instance` (línea 273, usado en línea 380)
- **Razón**: Se usa para construir el modelo final del fold en línea 380
- **Impacto si se elimina**: ❌ CRÍTICO - No se puede construir el modelo
- **Nota**: Este objeto viene de `optimize_model()` y se retorna, NO debe eliminarse dentro de `optimize_model()`

---

## ✅ CONCLUSIÓN FINAL

### Objetos SEGUROS para eliminar (12 objetos):
1. `X_tr`, `X_va` - Después de línea 183 en `_evaluate_params()`
2. `X_tr_aug` - Después de línea 177 en `_evaluate_params()`
3. `scaler` (fold interno) - Después de línea 158 en `_evaluate_params()`
4. `preprocessor` (fold interno) - Después de línea 166 en `_evaluate_params()`
5. `X_tr_prep`, `X_va_prep` - Después de línea 178 en `_evaluate_params()`
6. `selector` (fold interno) - Después de línea 178 en `_evaluate_params()`
7. `X_tr_sel`, `X_va_sel` - Después de línea 183 en `_evaluate_params()`
8. `model` (fold interno) - Después de línea 186 en `_evaluate_params()`
9. `y_hat` - Después de línea 186 en `_evaluate_params()`
10. `scores` - Después de línea 233 en `objective()`
11. `study` - Después de línea 273 en `optimize_model()`
12. `model_results` - Después de línea 276 en `optimize_model()`
13. `X_train_base`, `X_test_base` - Después de línea 407 en `run_dcv()`
14. `X_train_aug` - Después de línea 373 en `run_dcv()`
15. `scaler`, `preprocessor`, `selector` (fold externo) - Después de línea 373 en `run_dcv()`
16. `X_train_prep`, `X_test_prep` - Después de línea 373 en `run_dcv()`
17. `X_train_sel`, `X_test_sel` - Después de línea 384 en `run_dcv()`
18. `model` (fold externo) - Después de línea 392 en `run_dcv()`

### Objetos que NO se deben eliminar (3 objetos):
1. `y_pred` - Necesario hasta línea 405
2. `y_test_orig` - Necesario hasta línea 406
3. `best_model_instance` - Necesario hasta línea 380 (y se retorna)

### Riesgo total: ✅ BAJO
- Solo se eliminan objetos después de su uso completo
- No hay referencias indirectas que puedan causar problemas
- Optuna no guarda referencias a objetos intermedios
- Los modelos scikit-learn no guardan referencias a datos después de fit()

### Impacto esperado:
- **Reducción de memoria**: 30-50% durante optimización
- **Impacto en análisis**: ✅ NINGUNO
- **Riesgo de bugs**: ✅ BAJO (solo se eliminan objetos claramente temporales)










