# integrated_config.py
# ES: Configuración para el optimizador integrado D-óptimo + I-óptimo
# EN: Configuration for the integrated optimizer (D-optimal + I-optimal)
# JA: 統合最適化（D最適 + I最適）用設定

# ES: Parámetros de optimización | EN: Optimization parameters | JA: 最適化パラメータ
DEFAULT_NUM_EXPERIMENTS = 15  # Number of experiments to select
DEFAULT_SAMPLE_SIZE = 5000    # Max sample size for candidate reduction
DEFAULT_ENABLE_HYPERPARAMETER_TUNING = True  # Enable UMAP hyperparameter tuning
DEFAULT_FORCE_REOPTIMIZATION = False  # Do not force re-optimization

# ES: Parámetros de tolerancia para emparejamiento | EN: Matching tolerance parameters | JA: マッチング許容誤差
DEFAULT_TOLERANCE_RELATIVE = 1e-4  # Relative tolerance for matching
DEFAULT_TOLERANCE_ABSOLUTE = 1e-6  # Absolute tolerance for matching

# ES: Parámetros de reducción de candidatos | EN: Candidate-reduction parameters | JA: 候補削減パラメータ
CANDIDATE_REDUCTION_THRESHOLD = 10000  # Threshold to activate reduction
MAX_REDUCED_CANDIDATES = 5000          # Max candidates after reduction

# === Parámetros UMAP ===
DEFAULT_UMAP_N_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_UMAP_N_COMPONENTS = 2

# ES: Configuración de archivos | EN: File settings | JA: ファイル設定
DEFAULT_EXISTING_DATA_FILE = "既存実験データ.xlsx"  # Default existing-data filename
DEFAULT_OUTPUT_FOLDER_SUFFIX = "_IntegratedResults"  # Results folder suffix

# ES: Configuración de visualización | EN: Visualization settings | JA: 可視化設定
DEFAULT_FIGURE_SIZE = (15, 10)
DEFAULT_DPI = 300
DEFAULT_FONT_SIZE = 12

# ES: Configuración de procesamiento | EN: Processing settings | JA: 処理設定
USE_NUMERICAL_STABLE_METHOD = True  # Use numerically stable methods
VERBOSE = True  # Show verbose output