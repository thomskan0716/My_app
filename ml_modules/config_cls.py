from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Literal, Union, Set
import numpy as np

class ConfigCLS:
    """
    分類タスク用 設定クラス（全定数に型ヒント付き）
    - trainer_dcv_cls.py / main_train_cls.py / main_predict_cls.py が参照
    - LightGBM警告対策のためパラメータ範囲を調整
    """

    # ====== 1. ファイル入出力設定 ======
    DATA_FOLDER: str = "データセット"              # データフォルダ名
    INPUT_FILE: str = "20250925_総実験データ.xlsx"
    PREDICT_INPUT_FILE: str = "Prediction_input.xlsx"
    PARENT_FOLDER_TEMPLATE: str = "分類解析結果_{timestamp}"  # 親フォルダ名のテンプレート
    RESULT_FOLDER: str = "01_多目的最適化"
    RESULT_FOLDER_TEMPLATE: str = "02_本学習結果"  # 本学習結果フォルダ
    LOG_FOLDER: str = "03_実行ログ"                # 実行ログフォルダ名
    SAVE_EXECUTION_LOG: bool = True               # 実行ログの保存を有効化
    EXECUTION_LOG_FILENAME: str = "execution_log.txt"  # 実行ログファイル名
    
    # 02_本学習結果のサブフォルダ設定
    MODEL_FOLDER: str = "01_モデル"                    # モデル関連フォルダ
    EVALUATION_FOLDER: str = "02_評価結果"            # 評価結果フォルダ
    PREDICTION_FOLDER: str = "03_予測結果"            # 予測結果フォルダ
    DIAGNOSTIC_FOLDER: str = "04_診断情報"            # 診断情報フォルダ
    
    # モデル情報ファイル設定
    SAVE_MODEL_INFO: bool = True                      # モデル情報の保存を有効化
    MODEL_INFO_FILENAME: str = "model_info.json"      # モデル情報ファイル名
    
    # フォルダパス設定（実行時に動的に設定される絶対パス）
    MODEL_FOLDER_PATH: str = ""                       # モデルフォルダの絶対パス
    EVALUATION_FOLDER_PATH: str = ""                  # 評価結果フォルダの絶対パス
    PREDICTION_FOLDER_PATH: str = ""                  # 予測結果フォルダの絶対パス
    DIAGNOSTIC_FOLDER_PATH: str = ""                   # 診断情報フォルダの絶対パス
    
    @classmethod
    def set_folder_paths(cls, parent_folder: str):
        """フォルダパスを動的に設定"""
        import os
        main_folder = os.path.join(parent_folder, cls.RESULT_FOLDER_TEMPLATE)
        cls.MODEL_FOLDER_PATH = os.path.join(main_folder, cls.MODEL_FOLDER)
        cls.EVALUATION_FOLDER_PATH = os.path.join(main_folder, cls.EVALUATION_FOLDER)
        cls.PREDICTION_FOLDER_PATH = os.path.join(main_folder, cls.PREDICTION_FOLDER)
        cls.DIAGNOSTIC_FOLDER_PATH = os.path.join(main_folder, cls.DIAGNOSTIC_FOLDER)
    
    # ====== 2. データスキーマ設定 ======
    # ターゲット/スキーマ
    TARGET_COLUMN: str = "バリ除去"  # 2値ラベルの列名（実データに合わせてください）
    POSITIVE_LABELS: List[Union[int, str, bool]] = [1, "P", "Yes", "OK", True]
    NEGATIVE_LABELS: List[Union[int, str, bool]] = [0, "N", "No", "NG", False]

    ID_COLUMNS_CANDIDATES: List[str] = ["ID", "試験ID", "EntryID"]
    DATE_COLUMNS_CANDIDATES: List[str] = ["日付", "測定日", "Date"]
    GROUP_COLUMN: Optional[str] = None  # 例: "試験ロット" があるなら列名を指定

    # --- 強制除外リスト（上の集合に加え、ここに書いた列は常に説明変数から外す）---
    EXCLUDE_COLUMNS: Set[str] = set(["Index","Unnamed: 0", "Unnamed: 1"])  # Excel由来のゴミ列想定

    # --- 使用可能性のある変数候補（ALLOWED_FEATURES以外は使用されない）---
    ALLOWED_FEATURES: Set[str] = set([
        'A32',
        'A11',
        'A21',
        '送り速度',
        '切込量',
        '突出し量',
        '載せ率',
        '回転速度',
        'UPカット',
        'パス数',
    ])

    # --- 強制保持（必ず残す）プロセス変数（回帰/分類 共通）---
    MUST_KEEP_FEATURES: Set[str] = set([
        'A32',
        'A11',
        'A21',
        '送り速度',
        '切込量',
        '突出し量',
        '載せ率',
        '回転速度',
        'UPカット',
        'パス数',
    ])

    # ========== 特徴量タイプ定義 ==========
    CONTINUOUS_FEATURES = [
        '送り速度', '切込量', '突出し量',
        '載せ率', '回転速度', 
        #'切削時間',
        #'v_m_per_s','k_N_per_m','Fn_N','Ff_N',
        #'CuttingTime_s','FrictionHeat_J','FrictionWork_J'
    ]
    
    DISCRETE_FEATURES = ['A32', 'A11', 'A21']
    
    BINARY_FEATURES = [
        'UPカット',
        #'Material_鋼','Material_アルミ','Material_SUS',
        #'Material_チタン','Material_銅','Material_ジュラルミン'
    ]
    
    INTEGER_FEATURES = ['パス数']

    # ====== 3. モデル比較設定 ======
    MODELS_TO_USE: List[str] = ["lightgbm", "xgboost", "random_forest", "logistic"]
    COMPARE_MODELS: bool = True  # モデル比較を有効化
    MODEL_COMPARISON_CV_SPLITS: int = 5         # モデル比較時のCV分割数
    MODEL_COMPARISON_SCORING: str = 'roc_auc'   # モデル比較時の評価指標
    # 評価指標の選択肢:
    # 'accuracy'   - 正解率（バリ除去0or1が均衡している場合）
    # 'roc_auc'    - ROC曲線下面積（バリ除去0or1が不均衡でも使える、確率出力必要）
    # 'f1'         - F1スコア（Positive検出重視）
    # 'precision'  - 適合率（誤検出FP（偽陽性）を減らしたい）
    # 'recall'     - 再現率（見逃しFN（偽陰性）を減らしたい）

    # ====== 4. 多目的最適化設定 ======
    N_TRIALS_MULTI_OBJECTIVE: int = 100  # 多目的最適化の試行回数
    
    # 最適解選択時の重み（合計1.0になるように）
    FP_WEIGHT: float = 0.3        # FP率の重み（低いほど良い）
    COVERAGE_WEIGHT: float = 0.5  # カバレッジの重み（高いほど良い）
    AUC_WEIGHT: float = 0.2       # AUCの重み（高いほど良い）
    
    # NP_ALPHA探索範囲
    NP_ALPHA_RANGE: Tuple[float, float] = (0.001, 0.05)

    # ====== 5. DCV学習設定 ======
    # クロスバリデーション
    OUTER_SPLITS: int = 10
    INNER_SPLITS: int = 10
    RANDOM_STATE: int = 42

    # 内側最適化（Optuna）
    N_TRIALS_INNER: int = 50  # 実務50〜100, 研究100〜300推奨

    # Innerのみノイズ付与（リーク対策）
    USE_INNER_NOISE: bool = True
    NOISE_PPM: int = 50              # ノイズレベル（ppm単位）
    NOISE_RATIO: float = 0.3         # ノイズ付きサンプルの追加比率（0.3 = 30%）

    # ====== 6. 閾値決定設定 ======
    # Neyman–Pearson（τ+）設定
    # 「負例スコアの (1-α) 分位」を超える確率のみを P と宣言（FP上限=α）
    NP_ALPHA: float = 0.05
    USE_UPPER_CI_ADJUST: bool = True
    CI_METHOD: Literal["wilson", "normal", "jeffreys"] = "wilson"
    CI_CONFIDENCE: float = 0.95

    # τ− 探索（グレーゾーン縮小）
    TAU_NEG_GRID: List[float] = np.linspace(0.0, 0.8, 161).tolist()  # 0.4→0.8に拡張
    TAU_NEG_FALLBACK_RATIO: float = 0.3 

    # ====== 7. 評価・検証設定 ======
    # 固定HP評価設定
    FINAL_EVALUATION_CV_SPLITS: int = 5         # 固定HP評価時のCV分割数
    FINAL_EVALUATION_SHUFFLE: bool = True       # 固定HP評価時のシャッフル有無
    FINAL_EVALUATION_RANDOM_STATE: int = 42     # 固定HP評価時の乱数シード

    # ホールドアウト評価設定
    HOLDOUT_TEST_SIZE: float = 0.2              # ホールドアウトセットのサイズ（20%）
    HOLDOUT_STRATIFY: bool = True               # クラス分布を保持するか
    HOLDOUT_RANDOM_STATE: int = 42              # ホールドアウト分割時の乱数シード

    # グレー領域診断設定
    GRAY_ZONE_MIN_WIDTH: float = 0.05              # グレー領域の最小幅（警告閾値）
    GRAY_ZONE_MAX_WIDTH: float = 0.5               # グレー領域の最大幅（警告閾値）

    # ====== 8. 出力・可視化設定 ======

    # 可視化設定
    # 共通設定
    PLOT_FONT_FAMILY: List[str] = ['Yu Gothic', 'sans-serif']  # フォントファミリー
    PLOT_UNICODE_MINUS: bool = False                 # マイナス記号のUnicode表示
    PLOT_DPI: int = 150                              # 図の解像度（DPI）
    PLOT_BBOX_INCHES: str = 'tight'                  # 保存時の余白調整
    PLOT_BACKEND: str = 'Agg'                        # 非表示バックエンド（ポップアップ防止）
    
    # 固定HP評価の混同行列
    FIXED_HP_PLOT_FIGSIZE: Tuple[int, int] = (15, 10)        # 図のサイズ
    FIXED_HP_PLOT_LAYOUT: Tuple[int, int] = (2, 3)           # レイアウト（2行3列）
    
    # OOF予測の混同行列
    OOF_PLOT_FIGSIZE: Tuple[int, int] = (14, 6)              # 図のサイズ
    OOF_PLOT_LAYOUT: Tuple[int, int] = (1, 2)                # レイアウト（1行2列）
    
    # DCVフォールド詳細分析
    DCV_PLOT_COLS: int = 3                                   # 列数（行数は動的）
    
    # 特徴量重要度可視化設定
    SAVE_FEATURE_IMPORTANCE: bool = True                     # 特徴量重要度の可視化を有効化
    FEATURE_IMPORTANCE_TOP_K: int = 20                       # 表示する上位特徴量数
    FEATURE_IMPORTANCE_FIGSIZE: Tuple[int, int] = (12, 8)    # 特徴量重要度図のサイズ

    # ====== 9. モデル設定 ======
    # モデルごとの探索レンジ等（データ量を考慮して調整）
    MODEL_CONFIGS: Dict[str, Dict[str, Union[bool, Tuple[int, int], Tuple[float, float]]]] = {
        # ===== LightGBM: データ量184件を考慮して緩和 =====
        "lightgbm": {
            "enable": True,
            "n_estimators_range": (100, 500),      # 200-1200 → 100-500（過学習防止）
            "learning_rate_range": (0.01, 0.2),    # そのまま
            "num_leaves_range": (15, 63),          # 31-255 → 15-63（単純化）
            "min_data_in_leaf_range": (5, 20),     # 10-60 → 5-20（緩和：警告対策）
            "min_gain_to_split_range": (0.0, 0.01), # 0.02 → 0.01（分割しやすく）
            "feature_fraction_range": (0.6, 1.0),   # そのまま
            "bagging_fraction_range": (0.7, 1.0),   # そのまま
            "bagging_freq_range": (0, 5),           # そのまま
            "lambda_l1_range": (0.0, 3.0),          # そのまま
            "lambda_l2_range": (0.0, 3.0),          # そのまま
            "max_depth_range": (3, 8),              # 3-12 → 3-8（浅めに）
        },

        # ===== XGBoost: 同様に調整 =====
        "xgboost": {
            "enable": True,
            "n_estimators_range": (50, 300),        # 100-600 → 50-300
            "learning_rate_range": (0.01, 0.3),     # そのまま
            "max_depth_range": (3, 7),              # 3-10 → 3-7
            "subsample_range": (0.6, 1.0),          # そのまま
            "colsample_bytree_range": (0.6, 1.0),   # そのまま
            "reg_alpha_range": (0.0, 5.0),          # そのまま
            "reg_lambda_range": (0.0, 5.0),         # そのまま
            "min_child_weight_range": (1.0, 10.0),  # 0.0 → 1.0（最小値調整）
            "gamma_range": (0.0, 5.0),              # そのまま
        },

        # ===== RandomForest: 調整 =====
        "random_forest": {
            "enable": True,
            "n_estimators_range": (100, 400),       # 200-800 → 100-400
            "max_depth_range": (3, 12),             # 3-20 → 3-12
            "min_samples_split_range": (2, 20),     # そのまま
            "min_samples_leaf_range": (1, 10),      # 1-20 → 1-10
            "max_features_range": (0.4, 1.0),       # そのまま
        },

        # ===== Logistic（saga + elasticnet 前提）=====
        "logistic": {
            "enable": True,
            "C_range": (1e-3, 1e2),
            "l1_ratio_range": (0.0, 1.0),
        },
    }

    # ====== 10. 特徴選択・校正設定 ======
    # 特徴選択
    SELECT_TOP_K_RANGE: Tuple[int, int] = (20, 80)        # 重要度上位を何本残すか
    CORR_THRESHOLD_RANGE: Tuple[float, float] = (0.90, 0.99)  # 高相関のカット閾値

    # 校正（Calibrator）
    #CALIBRATION_CANDIDATES: Tuple[str, ...] = ("temperature", "isotonic")
    CALIBRATION_CANDIDATES: Tuple[str, ...] = ("isotonic",)  # isotonicのみ使用
    CALIBRATION_SELECTION_METRIC_WEIGHTS: Dict[str, float] = {
        "ece": 0.7,   # Expected Calibration Error
        "brier": 0.3, # Brier score
    }

    # ====== 11. OOD検出設定 ======
    # OOD 検出（Mahalanobis）
    USE_OOD: bool = True
    OOD_PERCENTILE: float = 99  # 上位パーセンタイルを「OOD候補」に

    # ====== 12. その他設定 ======
    # 追加: ログの詳細度など（必要なら）

    # ====== 簡易バリデーション ======
    @classmethod
    def validate(cls) -> bool:
        # MODELS_TO_USE が空でない
        if not cls.MODELS_TO_USE:
            raise ValueError("MODELS_TO_USE が空です。少なくとも1つのモデルを指定してください。")
        # 分位の整合性
        if not (0.0 < cls.NP_ALPHA < 0.5):
            raise ValueError("NP_ALPHA は (0, 0.5) の範囲で指定してください。")
        # CI 信頼係数
        if not (0.5 < cls.CI_CONFIDENCE < 0.999):
            raise ValueError("CI_CONFIDENCE は (0.5, 0.999) の範囲で指定してください。")
        # グリッド
        if not cls.TAU_NEG_GRID or not all(isinstance(v, float) for v in cls.TAU_NEG_GRID):
            raise ValueError("TAU_NEG_GRID が不正です。float のリストで与えてください。")
        return True

    @classmethod
    def get_tau_neg_grid(cls, tau_pos: float = None) -> List[float]:
        """τ+に応じた動的なグリッド生成"""
        if tau_pos is not None:
            # τ+より小さい範囲でグリッドを生成
            max_val = min(0.8, tau_pos - 0.01)
            return np.linspace(0.0, max_val, 161).tolist()
        else:
            return cls.TAU_NEG_GRID

