# integrated_config.py
# Configuración para el optimizador integrado D-óptimo + I-óptimo

# === Parámetros de optimización ===
DEFAULT_NUM_EXPERIMENTS = 15  # Número de experimentos a seleccionar
DEFAULT_SAMPLE_SIZE = 5000    # Tamaño máximo de muestra para reducción
DEFAULT_ENABLE_HYPERPARAMETER_TUNING = True  # Habilitar optimización de hiperparámetros UMAP
DEFAULT_FORCE_REOPTIMIZATION = False  # No forzar reoptimización

# === Parámetros de tolerancia para emparejamiento ===
DEFAULT_TOLERANCE_RELATIVE = 1e-4  # Tolerancia relativa para emparejamiento
DEFAULT_TOLERANCE_ABSOLUTE = 1e-6  # Tolerancia absoluta para emparejamiento

# === Parámetros de reducción de candidatos ===
CANDIDATE_REDUCTION_THRESHOLD = 10000  # Umbral para activar reducción
MAX_REDUCED_CANDIDATES = 5000          # Máximo número de candidatos después de reducción

# === Parámetros UMAP ===
DEFAULT_UMAP_N_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_UMAP_N_COMPONENTS = 2

# === Configuración de archivos ===
DEFAULT_EXISTING_DATA_FILE = "既存実験データ.xlsx"  # Archivo de datos existentes por defecto
DEFAULT_OUTPUT_FOLDER_SUFFIX = "_IntegratedResults"  # Sufijo para carpeta de resultados

# === Configuración de visualización ===
DEFAULT_FIGURE_SIZE = (15, 10)
DEFAULT_DPI = 300
DEFAULT_FONT_SIZE = 12

# === Configuración de procesamiento ===
USE_NUMERICAL_STABLE_METHOD = True  # Usar métodos numéricamente estables
VERBOSE = True  # Mostrar mensajes detallados 