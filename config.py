"""
プロジェクト設定ファイル
特徴量タイプの明確な定義と管理
"""
import os, math


class Config:
    PIPELINE_SCHEMA_VERSION = "2.1"
    PIPELINE_FORMAT_LABEL  = "dict+sk_pipeline"

    """回帰分析の設定クラス"""
    
    # ========== 共通設定 ==========
    # ファイル設定 (con soporte dinámico)
    _dynamic_base_folder = None  # Para paths dinámicos
    _dynamic_data_folder = None
    _dynamic_result_folder = None
    
    # Valores por defecto
    DATA_FOLDER = '01_データセット'
    INPUT_FILE = '20250925_総実験データ.xlsx'
    RESULT_FOLDER = '03_学習結果'
    MODEL_FOLDER = '02_学習モデル'
    PREDICTION_FOLDER = '04_予測'
    PREDICTION_DATA = 'Prediction_input.xlsx'
    FINAL_MODEL_PREFIX = 'final_model'
    PREDICTION_COLUMN_PREFIX = 'prediction'
    PREDICTION_OUTPUT_FILE = 'Prediction_output.xlsx'

    # EDA（探索的データ分析設定）
    RUN_DATA_ANALYSIS = True  # Falseにすると分析をスキップ
    
    # --- 並列実行の最小ポリシー ---
    CPU_FRACTION = 0.9          # 使うCPU比率
    MIN_RESERVE_CORES = 1       # 予備で空けるコア数
    OPTUNA_JOBS_CAP = 8         # Optunaの上限制限
    MODEL_JOBS_CAP  = 32        # モデル側の上限制限

    @classmethod
    def set_dynamic_paths(cls, base_folder, data_folder=None, result_folder=None):
        """
        Configura paths dinámicos para ejecución programática
        
        Parameters
        ----------
        base_folder : str
            Carpeta base del proyecto (donde están las subcarpetas)
        data_folder : str, optional
            Subcarpeta de datos (por defecto '01_データセット')
        result_folder : str, optional
            Subcarpeta de resultados (por defecto '02_結果')
        """
        cls._dynamic_base_folder = base_folder
        cls._dynamic_data_folder = data_folder or '01_データセット'
        cls._dynamic_result_folder = result_folder or '02_結果'
        print(f"✅ Config paths actualizados: base={base_folder}")
    
    @classmethod
    def get_base_folder(cls):
        """Obtiene la carpeta base (dinámica o por defecto)"""
        if cls._dynamic_base_folder:
            return cls._dynamic_base_folder
        return os.getcwd() if 'os' in dir() else '.'
        
    @classmethod
    def get_data_folder(cls):
        """Obtiene la carpeta de datos (dinámica o por defecto)"""
        base = cls.get_base_folder()
        data = cls._dynamic_data_folder or cls.DATA_FOLDER
        return os.path.join(base, data)
    
    @classmethod
    def get_result_folder(cls):
        """Obtiene la carpeta de resultados (dinámica o por defecto)"""
        base = cls.get_base_folder()
        result = cls._dynamic_result_folder or cls.RESULT_FOLDER
        return os.path.join(base, result)
    
    # 日本語フォント設定
    JAPANESE_FONT_FAMILY = ['Yu Gothic']
    JAPANESE_FONT_UNICODE_MINUS = False
    
    @classmethod
    def _auto_jobs(cls) -> int:
        cpu = os.cpu_count() or 1
        use = max(1, math.floor(cpu * cls.CPU_FRACTION) - cls.MIN_RESERVE_CORES)
        return use

    @classmethod
    def get_n_jobs(cls, kind: str = "optuna") -> int:
        base = cls._auto_jobs()
        if kind == "optuna":
            return max(1, min(base, cls.OPTUNA_JOBS_CAP))
        if kind == "model":
            return max(1, min(base, cls.MODEL_JOBS_CAP))
        return max(1, base)

    # ========== 01_モデル学習用設定 ==========  --------------------------UI--------------------------------------------------------
    # EDA（探索的データ分析設定）
    RUN_DATA_ANALYSIS = True  # Falseにすると分析をスキップ
        # フォールバック設定
    FALLBACK_MODEL_ORDER = ['random_forest', 'lightgbm', 'xgboost', 'gradient_boost', 'ridge', 'elastic_net', 'lasso','catboost',
        'gaussian_process',]
    FALLBACK_FINAL_MODEL = 'ridge'
    FALLBACK_DEFAULT_PARAMS = {
        'alpha': 1.0,
        'top_k': 20,
        'corr_threshold': 0.95,
        'use_interactions': False,
        'use_polynomial': False
    }
    

    # ========== モデル設定 ==========
    # ★命名を ModelFactory 側のキーに完全一致させる / 追加: catboost, gaussian_process
    MODELS_TO_USE = [
        'random_forest',
        'lightgbm',
        'catboost',
        #'gaussian_process',
        # 'xgboost',
        # 'ridge', 'lasso', 'elastic_net', 'gradient_boost'
    ]

    # モデル毎の詳細設定（キー名も完全一致に修正）
    MODEL_CONFIGS = {
        'lightgbm': {
            'n_estimators_range': (100, 500),
            'num_leaves_range': (20, 150),
            'learning_rate_range': (0.01, 0.3),
            'enable': True
        },
        'xgboost': {
            'n_estimators_range': (100, 400),
            'max_depth_range': (3, 12),
            'learning_rate_range': (0.01, 0.3),
            'enable': True
        },
        'random_forest': {  # ← randomforest → random_forest（バグ修正）
            'n_estimators_range': (50, 300),
            'max_depth_range': (5, 20),
            'enable': True
        },
        'ridge': {
            'alpha_range': (0.001, 100),
            'enable': True
        },
        'lasso': {
            'alpha_range': (0.001, 100),
            'enable': True
        },
        'elastic_net': {  # ← elasticnet → elastic_net（バグ修正）
            'alpha_range': (0.001, 100),
            'l1_ratio_range': (0.1, 0.9),
            'enable': True
        },
        'gradient_boost': {  # ← gradientboost → gradient_boスト（バグ修正）
            'n_estimators_range': (50, 300),
            'max_depth_range': (3, 10),
            'enable': True
        },
        'catboost': {  # 追加
            'depth_range': (4, 10),
            'learning_rate_range': (0.01, 0.3),
            'iterations_range': (200, 800),
            'enable': True
        },
        'gaussian_process': {  # 追加
            'alpha_range': (1e-8, 1e-2),
            'n_restarts_optimizer_range': (0, 5),
            'enable': True
        }
    }

    # デフォルトモデル（全モデルの最適化に失敗した場合）
    DEFAULT_MODEL = 'random_forest'

    # ========== 目的変数設定 ==========
    TARGET_COLUMNS = ['摩耗量', '上面ダレ量', '側面ダレ量']

    # ========== 特徴量設定 ==========
    USE_MANDATORY_FEATURES = True
    MANDATORY_FEATURES = [
        '送り速度',    # 切削条件の基本
        '切込量',      # 切削条件の基本
        '回転速度',    # 切削条件の基本
        '突出量',
        '載せ率',
        'パス数',
        'UPカット',
        'A32',         # 工具情報
        'A11',         # 工具情報
        'A21',         # 工具情報
        'A13'          # 工具情報 (cepillo)
    ]
    
    # 特徴選択設定
    # 説明変数同士の相関が高い場合、重要度が低い方を除去する機能
    USE_CORRELATION_REMOVAL = True  # 相関除去機能を有効にするか
    CORRELATION_THRESHOLD = 0.95    # 相関閾値（この値以上の相関を持つ特徴量ペアから1つを除去）
    DEFAULT_TOP_K = 20              # デフォルトの特徴選択数
    DEFAULT_CORR_THRESHOLD = 0.95   # デフォルトの相関閾値
    DEFAULT_USE_INTERACTIONS = False  # デフォルトの交互作用項生成
    DEFAULT_USE_POLYNOMIAL = False    # デフォルトの多項式特徴生成
    
    # ========== 説明変数設定 ==========
    FEATURE_COLUMNS = [
        #'A32', 'A11', 'A21',           
        '送り速度', '切込量', '突出量', '載せ率', '回転速度', 'パス数', 'UPカット', 'A32', 'A11', 'A21', 'A13', #'切削時間',                      
        ##'v_m_per_s', 'k_N_per_m', 'Fn_N', 'Ff_N', 'CuttingTime_s', 'FrictionHeat_J', 'FrictionWork_J',
        #'Material_鋼', 'Material_アルミ', 'Material_SUS', 'Material_チタン', 'Material_銅', 'Material_ジュラルミン'
    ]
    
    # ========== 特徴量タイプ定義 ==========
    CONTINUOUS_FEATURES = [
        '送り速度', '切込量', '突出量', '載せ率', '回転速度', #'切削時間',
        #'v_m_per_s', 'k_N_per_m', 'Fn_N', 'Ff_N', 'CuttingTime_s', 'FrictionHeat_J', 'FrictionWork_J'
    ]
    
    DISCRETE_FEATURES = ['A32', 'A11', 'A21', 'A13']  # A13: cepillo
    
    BINARY_FEATURES = [
        'UPカット',
        #'Material_鋼', 'Material_アルミ', 'Material_SUS', 'Material_チタン', 'Material_銅', 'Material_ジュラルミン'
    ]
    
    INTEGER_FEATURES = ['パス数']

    # ========== 目的変数変換方法 ==========
    TRANSFORM_METHOD = 'auto'  # 'auto','log','sqrt','none' 等（auto推奨）

    # ========== PPM設定 ==========
    PPM_LEVELS = [1, 5, 10]
    AUGMENT_RATIO = 0.3
    PPM_AUGMENT_PER_LEVEL = 1
    USE_PPM_AUGMENTATION = True

    # ========== Cross-Validation設定 ==========
    OUTER_SPLITS = 10
    INNER_SPLITS = 10

    # ========== 最適化設定 ==========
    N_TRIALS = 50
    OPTIMIZATION_METRIC = 'mae'

    # ========== その他設定 ==========
    RANDOM_STATE = 42
    GROUP_COLUMN = None

    # ========== SHAP設定 ==========
    SHAP_MODE = 'detailed'  # 'none','summary','detailed','full'
    SHAP_MAX_SAMPLES = 200
    SHAP_TOP_FEATURES = 10

    @classmethod
    def validate(cls):
        """設定の整合性チェック"""
        # 特徴量チェック
        all_features = (cls.CONTINUOUS_FEATURES + cls.DISCRETE_FEATURES + 
                       cls.BINARY_FEATURES + cls.INTEGER_FEATURES)
        
        if len(all_features) != len(set(all_features)):
            raise ValueError("特徴量タイプ定義に重複があります")
        
        if set(all_features) != set(cls.FEATURE_COLUMNS):
            missing = set(cls.FEATURE_COLUMNS) - set(all_features)
            extra = set(all_features) - set(cls.FEATURE_COLUMNS)
            if missing:
                print(f"⚠ 未定義の特徴量: {missing}")
            if extra:
                print(f"⚠ 余分な特徴量定義: {extra}")
            raise ValueError("特徴量定義が一致しません")
        
        # モデル設定チェック
        if not cls.MODELS_TO_USE:
            raise ValueError("MODELS_TO_USEが空です。少なくとも1つのモデルを指定してください")
        return True

    @classmethod
    def get_feature_info(cls):
        """特徴量情報の取得"""
        return {
            'continuous': cls.CONTINUOUS_FEATURES,
            'discrete': cls.DISCRETE_FEATURES,
            'binary': cls.BINARY_FEATURES,
            'integer': cls.INTEGER_FEATURES,
            'total': len(cls.FEATURE_COLUMNS),
            'ppm_target': len(cls.CONTINUOUS_FEATURES)
        }

    # ========== 日本語フォント設定 ==========
    JAPANESE_FONT_FAMILY = ['Yu Gothic']
    JAPANESE_FONT_UNICODE_MINUS = False

    # ========== ログ設定 ==========
    SHOW_OPTUNA_PROGRESS = True
    VERBOSE_LOGGING = False
    SHOW_DATA_ANALYSIS_DETAILS = True
    SHOW_AUGMENTATION_LOGS = False

    # ========== パレート分析設定 ==========
    PARETO_OUTPUT_FOLDER = '05_パレート解'
    PARETO_OBJECTIVES = {
        '摩耗量': 'min',
        '切削時間': 'min',
        '上面ダレ量': 'min',
        '側面ダレ量': 'min',
    }
    PARETO_PLOT_FIGSIZE = (7, 6)
    PARETO_PLOT_ALPHA_ALL = 0.35
    PARETO_PLOT_SIZE_ALL = 18
    PARETO_PLOT_SIZE_PARETO = 36
    PARETO_PLOT_EDGECOLORS = 'r'
    PARETO_PLOT_FACECOLORS = 'none'
    PARETO_PLOT_LINEWIDTHS = 1.2
    PARETO_PLOT_DPI = 300
    PARETO_PLOT_GRID_ALPHA = 0.3
    PARETO_PLOTS_FOLDER = 'pareto_plots'
    PARETO_EXCEL_FILENAME = 'pareto_frontier.xlsx'
    PARETO_PLOT_FILENAME_FORMAT = 'pareto_{x_logical}__vs__{y_logical}.png'
    PARETO_SHEET_PARETO_ONLY = 'pareto_only'
    PARETO_SHEET_META = 'meta'
    PARETO_LABEL_MIN_SUFFIX = '（↓良い）'
    PARETO_LABEL_MAX_SUFFIX = '（↑良い）'
    # パレートフロントの間引きオプション
    PARETO_USE_EPSILON_PLOT = True  # 図用にε-支配で間引きするかどうか
    PARETO_EPSILON_REL = 0.02  # ε-支配の相対閾値（例: 0.02 = 2%）
    # パフォーマンス・可視化オプション
    PARETO_CHUNK_SIZE = 10000  # パレート計算時のチャンクサイズ（メモリ効率化）
    PARETO_VERBOSE = True  # パレート計算の進捗ログを表示するか
    PARETO_MAX_ROWS_FOR_STRICT = None  # 厳密パレート計算の行数上限（None=制限なし）
    PARETO_ENABLE_PLOTTING = True  # プロット生成を有効にするか
    PARETO_PLOT_SHOW_PAIRWISE_FRONT = True  # 2Dペア専用前面（黒★）を表示するか
    PARETO_PLOT_MAX_POINTS = 1500  # プロット上のパレート点の最大表示数（超過時はK-meansで間引き）
    # Excel出力設定
    PARETO_SHEET_PARETO_ONLY_PLOT = 'pareto_only_plotmask'  # 図用パレートフロントのシート名
    # 階層化パレート設定（Lexicographic）
    LEXI_CASES = [
        {"primary": ["摩耗量", "切削時間"], "secondary": ["上面ダレ量", "側面ダレ量"], "tag": "pri_摩耗量_切削時間"},
        {"primary": ["切削時間", "上面ダレ量"], "secondary": ["摩耗量", "側面ダレ量"], "tag": "pri_切削時間_上面ダレ量"},
    ]

    # ========== 切削時間計算設定 ==========
    CUTTING_DISTANCE_MM = 100
    CUTTING_TIME_FORMULA = 'CUTTING_DISTANCE_MM / 送り速度 * 60'
    CUTTING_TIME_COLUMN_NAME = '切削時間'

    @classmethod
    def get_model_info(cls):
        return {
            'models_to_use': cls.MODELS_TO_USE,
            'default_model': cls.DEFAULT_MODEL,
            'model_configs': cls.MODEL_CONFIGS
        }
