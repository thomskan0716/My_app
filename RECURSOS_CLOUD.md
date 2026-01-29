# Requisitos de Recursos para Ejecución en la Nube

## Análisis de la Aplicación

Esta aplicación Python es un sistema de análisis de regresión con interfaz gráfica (PySide6) que incluye:
- Análisis lineal y no lineal
- Modelos de Machine Learning (LightGBM, Random Forest, XGBoost)
- Optimización de hiperparámetros con Optuna
- Cross-Validation doble (Outer + Inner splits)
- Análisis SHAP para importancia de características
- Generación de gráficos y visualizaciones

---

## 📊 Estimación de Recursos

### **Memoria RAM (Recomendada)**

#### **Mínimo:**
- **4 GB RAM** - Para ejecuciones básicas con datos pequeños (< 10,000 filas)
- Solo análisis lineal
- Sin análisis SHAP detallado

#### **Recomendado:**
- **8-16 GB RAM** - Para uso normal con configuración por defecto
  - 50 trials de Optuna
  - 10 outer splits + 10 inner splits
  - 2 modelos (Random Forest + LightGBM)
  - 3 targets (variables objetivo)
  - Análisis SHAP básico

#### **Óptimo:**
- **16-32 GB RAM** - Para análisis completos y datos grandes
  - Múltiples modelos simultáneos
  - Análisis SHAP detallado
  - Datos con > 50,000 filas
  - Múltiples proyectos en paralelo

#### **Desglose de Uso de Memoria:**
- **Aplicación base (PySide6 GUI)**: ~200-500 MB
- **Librerías (NumPy, Pandas, Scikit-learn)**: ~300-500 MB
- **Modelos ML en memoria**: ~100-500 MB por modelo
- **Datos en memoria**: ~50-200 MB por 10,000 filas (depende de columnas)
- **Optuna (50 trials)**: ~500 MB - 1 GB (acumula historial)
- **Cross-Validation**: ~1-2 GB (múltiples folds en memoria)
- **SHAP Analysis**: ~500 MB - 2 GB (depende de SHAP_MAX_SAMPLES)
- **Matplotlib/Seaborn**: ~100-300 MB (gráficos en memoria)

**Total estimado (uso normal)**: 3-6 GB RAM

---

### **CPU (Procesador)**

#### **Mínimo:**
- **2 cores** - Funcional pero lento
- Tiempo de ejecución: 2-4 horas para análisis completo

#### **Recomendado:**
- **4-8 cores** - Balance entre costo y rendimiento
- Tiempo de ejecución: 30 minutos - 2 horas para análisis completo
- Permite paralelización de:
  - Optuna trials (si está habilitado)
  - Cross-validation folds
  - Modelos múltiples

#### **Óptimo:**
- **8-16 cores** - Para análisis rápidos y producción
- Tiempo de ejecución: 10-30 minutos para análisis completo
- Mejor aprovechamiento de:
  - LightGBM (paralelización nativa)
  - Random Forest (n_jobs)
  - NumPy/SciPy (BLAS/LAPACK multi-threaded)

#### **Configuración de Paralelización:**
- **LightGBM**: Usa múltiples threads automáticamente
- **Random Forest**: `n_jobs=-1` usa todos los cores disponibles
- **Optuna**: Puede paralelizar trials si se configura
- **NumPy/SciPy**: Usa OpenMP/BLAS multi-threaded (MKL, OpenBLAS)

**Nota**: La aplicación usa `ThreadPoolExecutor` para algunas operaciones, pero la mayoría del procesamiento pesado está en las librerías de ML que aprovechan múltiples cores automáticamente.

---

### **Almacenamiento (Disco)**

#### **Mínimo:**
- **5-10 GB** - Para instalación y datos básicos

#### **Recomendado:**
- **20-50 GB** - Para proyectos múltiples y resultados
  - Instalación de Python + librerías: ~3-5 GB
  - Datos de entrada: ~100 MB - 1 GB por proyecto
  - Modelos guardados: ~50-200 MB por modelo
  - Resultados y gráficos: ~500 MB - 2 GB por análisis
  - Base de datos SQLite: ~10-100 MB

#### **Óptimo:**
- **50-100 GB** - Para múltiples proyectos y backups
  - Historial de análisis
  - Modelos entrenados
  - Visualizaciones de alta resolución

---

### **GPU (Opcional)**

- **No requerida** - La aplicación no usa GPU actualmente
- Los modelos (LightGBM, Random Forest) pueden usar GPU pero no está configurado
- Si se implementa soporte GPU:
  - **NVIDIA GPU con CUDA** (mínimo 4 GB VRAM)
  - Aceleraría LightGBM/XGBoost significativamente

---

## ⚙️ Configuración Actual (config.py)

Basado en la configuración por defecto:

```python
N_TRIALS = 50              # Trials de Optuna
OUTER_SPLITS = 10          # Folds externos
INNER_SPLITS = 10          # Folds internos
MODELS_TO_USE = ['random_forest', 'lightgbm']  # 2 modelos
TARGET_COLUMNS = ['摩耗量', '上面ダレ量', '側面ダレ量']  # 3 targets
SHAP_MAX_SAMPLES = 200     # Muestras para SHAP
```

**Cálculo de operaciones:**
- Total de entrenamientos: 50 trials × 10 outer × 10 inner × 2 modelos × 3 targets = **300,000 entrenamientos** (en el peor caso)
- En práctica, Optuna optimiza y reduce esto significativamente

---

## ☁️ Recomendaciones por Proveedor Cloud

### **AWS (EC2 / SageMaker)**
- **Mínimo**: `t3.medium` (2 vCPU, 4 GB RAM) - ~$0.04/hora
- **Recomendado**: `t3.xlarge` (4 vCPU, 16 GB RAM) - ~$0.17/hora
- **Óptimo**: `m5.2xlarge` (8 vCPU, 32 GB RAM) - ~$0.38/hora

### **Google Cloud (Compute Engine)**
- **Mínimo**: `e2-medium` (2 vCPU, 4 GB RAM) - ~$0.03/hora
- **Recomendado**: `e2-standard-4` (4 vCPU, 16 GB RAM) - ~$0.13/hora
- **Óptimo**: `e2-standard-8` (8 vCPU, 32 GB RAM) - ~$0.26/hora

### **Azure (Virtual Machines)**
- **Mínimo**: `Standard_B2s` (2 vCPU, 4 GB RAM) - ~$0.04/hora
- **Recomendado**: `Standard_D4s_v3` (4 vCPU, 16 GB RAM) - ~$0.19/hora
- **Óptimo**: `Standard_D8s_v3` (8 vCPU, 32 GB RAM) - ~$0.38/hora

### **Heroku / Railway / Render**
- **Mínimo**: 4 GB RAM - ~$25-50/mes
- **Recomendado**: 8-16 GB RAM - ~$50-100/mes
- **Nota**: Estas plataformas son más caras pero más fáciles de desplegar

---

## 🔧 Optimizaciones para Reducir Recursos

### **Reducir Memoria:**
1. Reducir `N_TRIALS` de 50 a 20-30
2. Reducir `OUTER_SPLITS` / `INNER_SPLITS` de 10 a 5
3. Usar solo 1 modelo en lugar de 2
4. Desactivar SHAP (`SHAP_MODE = 'none'`)
5. Procesar targets secuencialmente en lugar de paralelo

### **Reducir CPU:**
1. Limitar threads: `OMP_NUM_THREADS=2`
2. Usar modelos más simples (Ridge/Lasso en lugar de Random Forest)
3. Reducir número de trials

### **Reducir Almacenamiento:**
1. Limpiar modelos antiguos
2. Comprimir resultados
3. Usar almacenamiento externo (S3, GCS) para resultados

---

## 📝 Variables de Entorno Recomendadas

Para optimizar el uso de recursos en la nube:

```bash
# Limitar threads de OpenMP
export OMP_NUM_THREADS=4

# Limitar threads de MKL (Intel Math Kernel Library)
export MKL_NUM_THREADS=4

# Limitar threads de OpenBLAS
export OPENBLAS_NUM_THREADS=4

# Backend de matplotlib sin GUI (importante para servidores)
export MPLBACKEND=Agg

# Backend de Qt sin GUI (para PySide6 en servidor)
export QT_QPA_PLATFORM=offscreen
# o
export QT_QPA_PLATFORM=vnc  # Si necesitas GUI remota
```

---

## ⚠️ Consideraciones Importantes

1. **GUI en la Nube**: PySide6 requiere un servidor X o VNC para mostrar la interfaz gráfica. Considera:
   - Usar modo headless (sin GUI) si es posible
   - Usar Xvfb para GUI virtual
   - Usar VNC para acceso remoto

2. **Tiempo de Ejecución**: Los análisis completos pueden tardar horas. Considera:
   - Usar instancias spot/preemptibles para ahorrar costos
   - Implementar checkpoints para reanudar análisis
   - Usar colas de trabajo (Celery, RQ)

3. **Escalabilidad**: Para múltiples usuarios/proyectos:
   - Usar contenedores (Docker)
   - Orquestación (Kubernetes)
   - Load balancing

4. **Costos**: 
   - Análisis completos pueden costar $5-20 por ejecución en instancias recomendadas
   - Considera instancias reservadas para uso continuo (hasta 70% descuento)

---

## 📊 Resumen Ejecutivo

| Recurso | Mínimo | Recomendado | Óptimo |
|---------|--------|-------------|--------|
| **RAM** | 4 GB | 8-16 GB | 16-32 GB |
| **CPU** | 2 cores | 4-8 cores | 8-16 cores |
| **Disco** | 10 GB | 20-50 GB | 50-100 GB |
| **GPU** | No requerida | No requerida | Opcional (4+ GB VRAM) |
| **Costo/hora** | $0.03-0.04 | $0.13-0.19 | $0.26-0.38 |
| **Tiempo análisis** | 2-4 horas | 30 min - 2 horas | 10-30 min |

**Recomendación final**: Comienza con **8 GB RAM y 4 cores** para evaluar el rendimiento real con tus datos, luego ajusta según necesidad.



