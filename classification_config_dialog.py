"""
ES: Diálogo de configuración para análisis de clasificación (bunrui kaiseki).
EN: Configuration dialog for classification analysis (bunrui kaiseki).
JA: 分類解析（bunrui kaiseki）の設定ダイアログ。

ES: Permite configurar parámetros de config_cls.py antes de ejecutar.
EN: Lets the user configure config_cls.py parameters before running.
JA: 実行前に config_cls.py のパラメータを設定できます。
"""
import os
import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTabWidget, QWidget, QCheckBox, QSpinBox, QComboBox,
    QGroupBox, QFormLayout, QMessageBox, QDoubleSpinBox, QListWidget,
    QAbstractItemView, QListWidgetItem, QLineEdit, QTextEdit, QSplitter,
    QFileDialog
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

# ES: Intentar importar config_cls | EN: Try to import config_cls | JA: config_cls のインポートを試行
try:
    # ES: Buscar ml_modules/config_cls.py | EN: Look for ml_modules/config_cls.py | JA: ml_modules/config_cls.py を探索
    current_dir = Path(__file__).parent.absolute()
    ml_modules_path = current_dir / "ml_modules" / "config_cls.py"
    
    if ml_modules_path.exists():
        if str(ml_modules_path.parent) not in sys.path:
            sys.path.insert(0, str(ml_modules_path.parent))
        from config_cls import ConfigCLS
    else:
        # ES: Buscar en otras ubicaciones | EN: Search in other locations | JA: 他の場所も探索
        potential_paths = [
            current_dir.parent / "ml_modules" / "config_cls.py",
            Path.cwd() / "ml_modules" / "config_cls.py",
        ]
        config_found = False
        for config_path in potential_paths:
            if config_path.exists():
                if str(config_path.parent) not in sys.path:
                    sys.path.insert(0, str(config_path.parent))
                try:
                    from config_cls import ConfigCLS
                    config_found = True
                    break
                except ImportError:
                    continue
        
        if not config_found:
            # ES: Crear un módulo dummy | EN: Create a dummy module | JA: ダミーモジュールを作成
            import types
            ConfigCLS = types.SimpleNamespace()
            print("⚠️ Warning: config_cls.py could not be imported; using default values")
except ImportError as e:
    print(f"⚠️ config_cls のインポート中にエラー: {e}")
    import types
    ConfigCLS = types.SimpleNamespace()


class ClassificationConfigDialog(QDialog):
    """ES: Diálogo para configurar parámetros del análisis de clasificación
    EN: Dialog to configure classification analysis parameters
    JA: 分類解析パラメータ設定ダイアログ
    """
    
    def __init__(self, parent=None, filtered_df=None):
        super().__init__(parent)
        self.setWindowTitle("分類分析設定")
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        
        self.filtered_df = filtered_df
        self.config_values = {}
        
        # ES: Variables para almacenar la validación de carpeta existente
        # EN: State for validating an existing folder
        # JA: 既存フォルダ検証用の状態
        self.validated_folder_path = None
        self.project_folder_path = None
        self.is_folder_valid = False
        
        self.setup_ui()
        
        # ES: Si hay datos filtrados, mostrar información en el diálogo
        # EN: If filtered data is provided, show summary info in the dialog
        # JA: フィルタ済みデータがあれば概要情報を表示
        if filtered_df is not None and not filtered_df.empty:
            self._show_data_info()
    
    def setup_ui(self):
        """ES: Configura la interfaz de usuario
        EN: Build the UI
        JA: UIを構築
        """
        layout = QVBoxLayout()
        
        # ES: Título | EN: Title | JA: タイトル
        title = QLabel("分類分析パラメータ設定")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # ES: Crear pestañas | EN: Create tabs | JA: タブを作成
        tabs = QTabWidget()
        
        # Tab 1: Características
        features_tab = self.create_features_tab()
        tabs.addTab(features_tab, "特徴量設定")
        
        # Tab 2: Modelos
        models_tab = self.create_models_tab()
        tabs.addTab(models_tab, "モデル設定")
        
        # Tab 3: Optimización multiobjetivo
        multiobj_tab = self.create_multiobjective_tab()
        tabs.addTab(multiobj_tab, "多目的最適化")
        
        # Tab 4: DCV y Aprendizaje
        dcv_tab = self.create_dcv_tab()
        tabs.addTab(dcv_tab, "DCV学習設定")
        
        # Tab 5: Umbrales
        thresholds_tab = self.create_thresholds_tab()
        tabs.addTab(thresholds_tab, "閾値決定")
        
        # Tab 6: Evaluación
        evaluation_tab = self.create_evaluation_tab()
        tabs.addTab(evaluation_tab, "評価設定")
        
        # ES: Tab 7: Cargar existente | EN: Tab 7: Load existing | JA: タブ7：既存読み込み
        load_existing_tab = self.create_load_existing_tab()
        tabs.addTab(load_existing_tab, "既存結果読み込み")
        
        layout.addWidget(tabs)
        
        # Botones
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = QPushButton("キャンセル")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("続行")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setDefault(True)
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def create_features_tab(self):
        """ES: Crear tab de configuración de características
        EN: Create the feature-configuration tab
        JA: 特徴量設定タブを作成
        """
        tab = QWidget()
        layout = QVBoxLayout()
        
        # ES: Obtener lista de todas las características disponibles
        # EN: Get the list of all available features
        # JA: 利用可能な特徴量一覧を取得
        try:
            all_features = sorted(list(getattr(ConfigCLS, 'ALLOWED_FEATURES', set([
                'A32', 'A11', 'A21', '送り速度', '切込量', '突出し量',
                '載せ率', '回転速度', 'UPカット', 'パス数'
            ]))))
        except:
            all_features = sorted(['A32', 'A11', 'A21', '送り速度', '切込量', '突出し量',
                                  '載せ率', '回転速度', 'UPカット', 'パス数'])
        
        # ALLOWED_FEATURES - Lista con checkboxes
        allowed_group = QGroupBox("使用可能な特徴量 (ALLOWED_FEATURES)")
        allowed_layout = QVBoxLayout()
        
        self.allowed_features_list = QListWidget()
        self.allowed_features_list.setSelectionMode(QAbstractItemView.MultiSelection)
        
        try:
            default_allowed = getattr(ConfigCLS, 'ALLOWED_FEATURES', set(all_features))
            if isinstance(default_allowed, set):
                default_allowed = default_allowed
            else:
                default_allowed = set(default_allowed)
        except:
            default_allowed = set(all_features)
        
        for feature in all_features:
            item = QListWidgetItem(feature)
            item.setCheckState(Qt.Checked if feature in default_allowed else Qt.Unchecked)
            self.allowed_features_list.addItem(item)
        
        allowed_layout.addWidget(QLabel("特徴量を選択してください:"))
        allowed_layout.addWidget(self.allowed_features_list)
        allowed_group.setLayout(allowed_layout)
        layout.addWidget(allowed_group)
        
        # MUST_KEEP_FEATURES - Lista con checkboxes
        must_keep_group = QGroupBox("強制保持特徴量 (MUST_KEEP_FEATURES)")
        must_keep_layout = QVBoxLayout()
        
        self.must_keep_features_list = QListWidget()
        self.must_keep_features_list.setSelectionMode(QAbstractItemView.MultiSelection)
        
        try:
            default_must_keep = getattr(ConfigCLS, 'MUST_KEEP_FEATURES', set(all_features))
            if isinstance(default_must_keep, set):
                default_must_keep = default_must_keep
            else:
                default_must_keep = set(default_must_keep)
        except:
            default_must_keep = set(all_features)
        
        for feature in all_features:
            item = QListWidgetItem(feature)
            item.setCheckState(Qt.Checked if feature in default_must_keep else Qt.Unchecked)
            self.must_keep_features_list.addItem(item)
        
        must_keep_layout.addWidget(QLabel("特徴量を選択してください:"))
        must_keep_layout.addWidget(self.must_keep_features_list)
        must_keep_group.setLayout(must_keep_layout)
        layout.addWidget(must_keep_group)
        
        # ES: Tipo de características - Listas con checkboxes
        # EN: Feature types - checkbox lists
        # JA: 特徴量タイプ（チェックボックス一覧）
        types_group = QGroupBox("特徴量タイプ定義")
        types_layout = QVBoxLayout()
        
        # ES: Crear un splitter para organizar las listas
        # EN: Create a splitter to lay out the lists
        # JA: リスト配置用のスプリッターを作成
        splitter = QSplitter(Qt.Horizontal)
        
        # CONTINUOUS_FEATURES
        continuous_group = QGroupBox("連続特徴量 (CONTINUOUS_FEATURES)")
        continuous_layout = QVBoxLayout()
        self.continuous_features_list = QListWidget()
        self.continuous_features_list.setSelectionMode(QAbstractItemView.MultiSelection)
        
        try:
            default_continuous = getattr(ConfigCLS, 'CONTINUOUS_FEATURES', [
                '送り速度', '切込量', '突出し量', '載せ率', '回転速度'
            ])
        except:
            default_continuous = ['送り速度', '切込量', '突出し量', '載せ率', '回転速度']
        
        for feature in all_features:
            item = QListWidgetItem(feature)
            item.setCheckState(Qt.Checked if feature in default_continuous else Qt.Unchecked)
            self.continuous_features_list.addItem(item)
        
        continuous_layout.addWidget(self.continuous_features_list)
        continuous_group.setLayout(continuous_layout)
        splitter.addWidget(continuous_group)
        
        # DISCRETE_FEATURES
        discrete_group = QGroupBox("離散特徴量 (DISCRETE_FEATURES)")
        discrete_layout = QVBoxLayout()
        self.discrete_features_list = QListWidget()
        self.discrete_features_list.setSelectionMode(QAbstractItemView.MultiSelection)
        
        try:
            default_discrete = getattr(ConfigCLS, 'DISCRETE_FEATURES', ['A32', 'A11', 'A21'])
        except:
            default_discrete = ['A32', 'A11', 'A21']
        
        for feature in all_features:
            item = QListWidgetItem(feature)
            item.setCheckState(Qt.Checked if feature in default_discrete else Qt.Unchecked)
            self.discrete_features_list.addItem(item)
        
        discrete_layout.addWidget(self.discrete_features_list)
        discrete_group.setLayout(discrete_layout)
        splitter.addWidget(discrete_group)
        
        # BINARY_FEATURES
        binary_group = QGroupBox("2値特徴量 (BINARY_FEATURES)")
        binary_layout = QVBoxLayout()
        self.binary_features_list = QListWidget()
        self.binary_features_list.setSelectionMode(QAbstractItemView.MultiSelection)
        
        try:
            default_binary = getattr(ConfigCLS, 'BINARY_FEATURES', ['UPカット'])
        except:
            default_binary = ['UPカット']
        
        for feature in all_features:
            item = QListWidgetItem(feature)
            item.setCheckState(Qt.Checked if feature in default_binary else Qt.Unchecked)
            self.binary_features_list.addItem(item)
        
        binary_layout.addWidget(self.binary_features_list)
        binary_group.setLayout(binary_layout)
        splitter.addWidget(binary_group)
        
        # INTEGER_FEATURES
        integer_group = QGroupBox("整数特徴量 (INTEGER_FEATURES)")
        integer_layout = QVBoxLayout()
        self.integer_features_list = QListWidget()
        self.integer_features_list.setSelectionMode(QAbstractItemView.MultiSelection)
        
        try:
            default_integer = getattr(ConfigCLS, 'INTEGER_FEATURES', ['パス数'])
        except:
            default_integer = ['パス数']
        
        for feature in all_features:
            item = QListWidgetItem(feature)
            item.setCheckState(Qt.Checked if feature in default_integer else Qt.Unchecked)
            self.integer_features_list.addItem(item)
        
        integer_layout.addWidget(self.integer_features_list)
        integer_group.setLayout(integer_layout)
        splitter.addWidget(integer_group)
        
        splitter.setSizes([200, 200, 200, 200])  # Distribute space evenly
        types_layout.addWidget(splitter)
        types_group.setLayout(types_layout)
        layout.addWidget(types_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_models_tab(self):
        """ES: Crear tab de configuración de modelos
        EN: Create the model-configuration tab
        JA: モデル設定タブを作成
        """
        tab = QWidget()
        layout = QVBoxLayout()
        
        # MODELS_TO_USE
        models_group = QGroupBox("使用するモデル (MODELS_TO_USE)")
        models_layout = QVBoxLayout()
        
        self.model_checkboxes = {}
        models = [
            ('lightgbm', 'LightGBM'),
            ('xgboost', 'XGBoost'),
            ('random_forest', 'Random Forest'),
            ('logistic', 'Logistic Regression')
        ]
        
        try:
            default_models = getattr(ConfigCLS, 'MODELS_TO_USE', ['lightgbm', 'xgboost', 'random_forest', 'logistic'])
        except:
            default_models = ['lightgbm', 'xgboost', 'random_forest', 'logistic']
        
        for model_key, model_name in models:
            checkbox = QCheckBox(model_name)
            checkbox.setChecked(model_key in default_models)
            self.model_checkboxes[model_key] = checkbox
            models_layout.addWidget(checkbox)
        
        models_group.setLayout(models_layout)
        layout.addWidget(models_group)
        
        # COMPARE_MODELS
        compare_group = QGroupBox("モデル比較設定")
        compare_layout = QFormLayout()
        
        self.compare_models = QCheckBox()
        try:
            self.compare_models.setChecked(getattr(ConfigCLS, 'COMPARE_MODELS', True))
        except:
            self.compare_models.setChecked(True)
        compare_layout.addRow("モデル比較を有効化 (COMPARE_MODELS):", self.compare_models)
        
        self.model_comparison_cv_splits = QSpinBox()
        self.model_comparison_cv_splits.setMinimum(3)
        self.model_comparison_cv_splits.setMaximum(20)
        try:
            self.model_comparison_cv_splits.setValue(getattr(ConfigCLS, 'MODEL_COMPARISON_CV_SPLITS', 5))
        except:
            self.model_comparison_cv_splits.setValue(5)
        compare_layout.addRow("モデル比較CV分割数 (MODEL_COMPARISON_CV_SPLITS):", self.model_comparison_cv_splits)
        
        self.model_comparison_scoring = QComboBox()
        self.model_comparison_scoring.addItems(['roc_auc', 'accuracy', 'f1', 'precision', 'recall'])
        try:
            current_scoring = getattr(ConfigCLS, 'MODEL_COMPARISON_SCORING', 'roc_auc')
            index = self.model_comparison_scoring.findText(current_scoring)
            if index >= 0:
                self.model_comparison_scoring.setCurrentIndex(index)
        except:
            pass
        compare_layout.addRow("評価指標 (MODEL_COMPARISON_SCORING):", self.model_comparison_scoring)
        
        compare_group.setLayout(compare_layout)
        layout.addWidget(compare_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_multiobjective_tab(self):
        """ES: Crear tab de optimización multiobjetivo
        EN: Create the multi-objective optimization tab
        JA: 多目的最適化タブを作成
        """
        tab = QWidget()
        layout = QVBoxLayout()
        
        # N_TRIALS_MULTI_OBJECTIVE
        trials_group = QGroupBox("試行回数")
        trials_layout = QFormLayout()
        
        self.n_trials_multi_objective = QSpinBox()
        self.n_trials_multi_objective.setMinimum(10)
        self.n_trials_multi_objective.setMaximum(500)
        try:
            self.n_trials_multi_objective.setValue(getattr(ConfigCLS, 'N_TRIALS_MULTI_OBJECTIVE', 100))
        except:
            self.n_trials_multi_objective.setValue(100)
        trials_layout.addRow("多目的最適化試行回数 (N_TRIALS_MULTI_OBJECTIVE):", self.n_trials_multi_objective)
        
        trials_group.setLayout(trials_layout)
        layout.addWidget(trials_group)
        
        # Pesos
        weights_group = QGroupBox("最適解選択時の重み（合計1.0になるように）")
        weights_layout = QFormLayout()
        
        self.fp_weight = QDoubleSpinBox()
        self.fp_weight.setMinimum(0.0)
        self.fp_weight.setMaximum(1.0)
        self.fp_weight.setSingleStep(0.1)
        self.fp_weight.setDecimals(2)
        try:
            self.fp_weight.setValue(getattr(ConfigCLS, 'FP_WEIGHT', 0.3))
        except:
            self.fp_weight.setValue(0.3)
        weights_layout.addRow("FP率の重み (FP_WEIGHT):", self.fp_weight)
        
        self.coverage_weight = QDoubleSpinBox()
        self.coverage_weight.setMinimum(0.0)
        self.coverage_weight.setMaximum(1.0)
        self.coverage_weight.setSingleStep(0.1)
        self.coverage_weight.setDecimals(2)
        try:
            self.coverage_weight.setValue(getattr(ConfigCLS, 'COVERAGE_WEIGHT', 0.5))
        except:
            self.coverage_weight.setValue(0.5)
        weights_layout.addRow("カバレッジの重み (COVERAGE_WEIGHT):", self.coverage_weight)
        
        self.auc_weight = QDoubleSpinBox()
        self.auc_weight.setMinimum(0.0)
        self.auc_weight.setMaximum(1.0)
        self.auc_weight.setSingleStep(0.1)
        self.auc_weight.setDecimals(2)
        try:
            self.auc_weight.setValue(getattr(ConfigCLS, 'AUC_WEIGHT', 0.2))
        except:
            self.auc_weight.setValue(0.2)
        weights_layout.addRow("AUCの重み (AUC_WEIGHT):", self.auc_weight)
        
        weights_group.setLayout(weights_layout)
        layout.addWidget(weights_group)
        
        # NP_ALPHA_RANGE
        alpha_range_group = QGroupBox("NP_ALPHA探索範囲")
        alpha_range_layout = QFormLayout()
        
        self.np_alpha_range_min = QDoubleSpinBox()
        self.np_alpha_range_min.setMinimum(0.0001)
        self.np_alpha_range_min.setMaximum(0.1)
        self.np_alpha_range_min.setSingleStep(0.001)
        self.np_alpha_range_min.setDecimals(4)
        try:
            default_range = getattr(ConfigCLS, 'NP_ALPHA_RANGE', (0.001, 0.05))
            self.np_alpha_range_min.setValue(default_range[0])
        except:
            self.np_alpha_range_min.setValue(0.001)
        alpha_range_layout.addRow("最小値:", self.np_alpha_range_min)
        
        self.np_alpha_range_max = QDoubleSpinBox()
        self.np_alpha_range_max.setMinimum(0.0001)
        self.np_alpha_range_max.setMaximum(0.1)
        self.np_alpha_range_max.setSingleStep(0.001)
        self.np_alpha_range_max.setDecimals(4)
        try:
            default_range = getattr(ConfigCLS, 'NP_ALPHA_RANGE', (0.001, 0.05))
            self.np_alpha_range_max.setValue(default_range[1])
        except:
            self.np_alpha_range_max.setValue(0.05)
        alpha_range_layout.addRow("最大値:", self.np_alpha_range_max)
        
        alpha_range_group.setLayout(alpha_range_layout)
        layout.addWidget(alpha_range_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_dcv_tab(self):
        """ES: Crear tab de configuración DCV
        EN: Create the DCV configuration tab
        JA: DCV設定タブを作成
        """
        tab = QWidget()
        layout = QVBoxLayout()
        
        # CV splits
        cv_group = QGroupBox("クロスバリデーション設定")
        cv_layout = QFormLayout()
        
        self.outer_splits = QSpinBox()
        self.outer_splits.setMinimum(3)
        self.outer_splits.setMaximum(20)
        try:
            self.outer_splits.setValue(getattr(ConfigCLS, 'OUTER_SPLITS', 10))
        except:
            self.outer_splits.setValue(10)
        cv_layout.addRow("外側分割数 (OUTER_SPLITS):", self.outer_splits)
        
        self.inner_splits = QSpinBox()
        self.inner_splits.setMinimum(3)
        self.inner_splits.setMaximum(20)
        try:
            self.inner_splits.setValue(getattr(ConfigCLS, 'INNER_SPLITS', 10))
        except:
            self.inner_splits.setValue(10)
        cv_layout.addRow("内側分割数 (INNER_SPLITS):", self.inner_splits)
        
        self.random_state = QSpinBox()
        self.random_state.setMinimum(0)
        self.random_state.setMaximum(9999)
        try:
            self.random_state.setValue(getattr(ConfigCLS, 'RANDOM_STATE', 42))
        except:
            self.random_state.setValue(42)
        cv_layout.addRow("乱数シード (RANDOM_STATE):", self.random_state)
        
        cv_group.setLayout(cv_layout)
        layout.addWidget(cv_group)
        
        # Optuna
        optuna_group = QGroupBox("Optuna最適化設定")
        optuna_layout = QFormLayout()
        
        self.n_trials_inner = QSpinBox()
        self.n_trials_inner.setMinimum(10)
        self.n_trials_inner.setMaximum(500)
        try:
            self.n_trials_inner.setValue(getattr(ConfigCLS, 'N_TRIALS_INNER', 50))
        except:
            self.n_trials_inner.setValue(50)
        optuna_layout.addRow("内側最適化試行回数 (N_TRIALS_INNER):", self.n_trials_inner)
        
        optuna_group.setLayout(optuna_layout)
        layout.addWidget(optuna_group)
        
        # Noise
        noise_group = QGroupBox("ノイズ付加設定")
        noise_layout = QFormLayout()
        
        self.use_inner_noise = QCheckBox()
        try:
            self.use_inner_noise.setChecked(getattr(ConfigCLS, 'USE_INNER_NOISE', True))
        except:
            self.use_inner_noise.setChecked(True)
        noise_layout.addRow("Inner CVでノイズ付加 (USE_INNER_NOISE):", self.use_inner_noise)
        
        self.noise_ppm = QSpinBox()
        self.noise_ppm.setMinimum(1)
        self.noise_ppm.setMaximum(1000)
        try:
            self.noise_ppm.setValue(getattr(ConfigCLS, 'NOISE_PPM', 50))
        except:
            self.noise_ppm.setValue(50)
        noise_layout.addRow("ノイズレベル [ppm] (NOISE_PPM):", self.noise_ppm)
        
        self.noise_ratio = QDoubleSpinBox()
        self.noise_ratio.setMinimum(0.0)
        self.noise_ratio.setMaximum(1.0)
        self.noise_ratio.setSingleStep(0.1)
        self.noise_ratio.setDecimals(2)
        try:
            self.noise_ratio.setValue(getattr(ConfigCLS, 'NOISE_RATIO', 0.3))
        except:
            self.noise_ratio.setValue(0.3)
        noise_layout.addRow("ノイズ付きサンプル追加比率 (NOISE_RATIO):", self.noise_ratio)
        
        noise_group.setLayout(noise_layout)
        layout.addWidget(noise_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_thresholds_tab(self):
        """ES: Crear tab de configuración de umbrales
        EN: Create the threshold-configuration tab
        JA: 閾値設定タブを作成
        """
        tab = QWidget()
        layout = QVBoxLayout()
        
        # NP_ALPHA
        np_alpha_group = QGroupBox("Neyman-Pearson設定")
        np_alpha_layout = QFormLayout()
        
        self.np_alpha = QDoubleSpinBox()
        self.np_alpha.setMinimum(0.0001)
        self.np_alpha.setMaximum(0.5)
        self.np_alpha.setSingleStep(0.001)
        self.np_alpha.setDecimals(4)
        try:
            self.np_alpha.setValue(getattr(ConfigCLS, 'NP_ALPHA', 0.05))
        except:
            self.np_alpha.setValue(0.05)
        np_alpha_layout.addRow("NP_ALPHA:", self.np_alpha)
        
        self.use_upper_ci_adjust = QCheckBox()
        try:
            self.use_upper_ci_adjust.setChecked(getattr(ConfigCLS, 'USE_UPPER_CI_ADJUST', True))
        except:
            self.use_upper_ci_adjust.setChecked(True)
        np_alpha_layout.addRow("信頼区間調整を使用 (USE_UPPER_CI_ADJUST):", self.use_upper_ci_adjust)
        
        self.ci_method = QComboBox()
        self.ci_method.addItems(['wilson', 'normal', 'jeffreys'])
        try:
            current_method = getattr(ConfigCLS, 'CI_METHOD', 'wilson')
            index = self.ci_method.findText(current_method)
            if index >= 0:
                self.ci_method.setCurrentIndex(index)
        except:
            pass
        np_alpha_layout.addRow("信頼区間方法 (CI_METHOD):", self.ci_method)
        
        self.ci_confidence = QDoubleSpinBox()
        self.ci_confidence.setMinimum(0.5)
        self.ci_confidence.setMaximum(0.999)
        self.ci_confidence.setSingleStep(0.01)
        self.ci_confidence.setDecimals(3)
        try:
            self.ci_confidence.setValue(getattr(ConfigCLS, 'CI_CONFIDENCE', 0.95))
        except:
            self.ci_confidence.setValue(0.95)
        np_alpha_layout.addRow("信頼係数 (CI_CONFIDENCE):", self.ci_confidence)
        
        np_alpha_group.setLayout(np_alpha_layout)
        layout.addWidget(np_alpha_group)
        
        # TAU_NEG
        tau_neg_group = QGroupBox("τ-探索設定")
        tau_neg_layout = QFormLayout()
        
        self.tau_neg_fallback_ratio = QDoubleSpinBox()
        self.tau_neg_fallback_ratio.setMinimum(0.0)
        self.tau_neg_fallback_ratio.setMaximum(1.0)
        self.tau_neg_fallback_ratio.setSingleStep(0.1)
        self.tau_neg_fallback_ratio.setDecimals(2)
        try:
            self.tau_neg_fallback_ratio.setValue(getattr(ConfigCLS, 'TAU_NEG_FALLBACK_RATIO', 0.3))
        except:
            self.tau_neg_fallback_ratio.setValue(0.3)
        tau_neg_layout.addRow("フォールバック比率 (TAU_NEG_FALLBACK_RATIO):", self.tau_neg_fallback_ratio)
        
        tau_neg_group.setLayout(tau_neg_layout)
        layout.addWidget(tau_neg_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_evaluation_tab(self):
        """ES: Crear tab de configuración de evaluación
        EN: Create the evaluation-configuration tab
        JA: 評価設定タブを作成
        """
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Fixed HP evaluation
        fixed_hp_group = QGroupBox("固定HP評価設定")
        fixed_hp_layout = QFormLayout()
        
        self.final_evaluation_cv_splits = QSpinBox()
        self.final_evaluation_cv_splits.setMinimum(3)
        self.final_evaluation_cv_splits.setMaximum(20)
        try:
            self.final_evaluation_cv_splits.setValue(getattr(ConfigCLS, 'FINAL_EVALUATION_CV_SPLITS', 5))
        except:
            self.final_evaluation_cv_splits.setValue(5)
        fixed_hp_layout.addRow("CV分割数 (FINAL_EVALUATION_CV_SPLITS):", self.final_evaluation_cv_splits)
        
        self.final_evaluation_shuffle = QCheckBox()
        try:
            self.final_evaluation_shuffle.setChecked(getattr(ConfigCLS, 'FINAL_EVALUATION_SHUFFLE', True))
        except:
            self.final_evaluation_shuffle.setChecked(True)
        fixed_hp_layout.addRow("シャッフル (FINAL_EVALUATION_SHUFFLE):", self.final_evaluation_shuffle)
        
        self.final_evaluation_random_state = QSpinBox()
        self.final_evaluation_random_state.setMinimum(0)
        self.final_evaluation_random_state.setMaximum(9999)
        try:
            self.final_evaluation_random_state.setValue(getattr(ConfigCLS, 'FINAL_EVALUATION_RANDOM_STATE', 42))
        except:
            self.final_evaluation_random_state.setValue(42)
        fixed_hp_layout.addRow("乱数シード (FINAL_EVALUATION_RANDOM_STATE):", self.final_evaluation_random_state)
        
        fixed_hp_group.setLayout(fixed_hp_layout)
        layout.addWidget(fixed_hp_group)
        
        # Holdout evaluation
        holdout_group = QGroupBox("ホールドアウト評価設定")
        holdout_layout = QFormLayout()
        
        self.holdout_test_size = QDoubleSpinBox()
        self.holdout_test_size.setMinimum(0.1)
        self.holdout_test_size.setMaximum(0.5)
        self.holdout_test_size.setSingleStep(0.05)
        self.holdout_test_size.setDecimals(2)
        try:
            self.holdout_test_size.setValue(getattr(ConfigCLS, 'HOLDOUT_TEST_SIZE', 0.2))
        except:
            self.holdout_test_size.setValue(0.2)
        holdout_layout.addRow("テストセットサイズ (HOLDOUT_TEST_SIZE):", self.holdout_test_size)
        
        self.holdout_stratify = QCheckBox()
        try:
            self.holdout_stratify.setChecked(getattr(ConfigCLS, 'HOLDOUT_STRATIFY', True))
        except:
            self.holdout_stratify.setChecked(True)
        holdout_layout.addRow("層化分割 (HOLDOUT_STRATIFY):", self.holdout_stratify)
        
        self.holdout_random_state = QSpinBox()
        self.holdout_random_state.setMinimum(0)
        self.holdout_random_state.setMaximum(9999)
        try:
            self.holdout_random_state.setValue(getattr(ConfigCLS, 'HOLDOUT_RANDOM_STATE', 42))
        except:
            self.holdout_random_state.setValue(42)
        holdout_layout.addRow("乱数シード (HOLDOUT_RANDOM_STATE):", self.holdout_random_state)
        
        holdout_group.setLayout(holdout_layout)
        layout.addWidget(holdout_group)
        
        # Gray zone
        gray_zone_group = QGroupBox("グレー領域診断設定")
        gray_zone_layout = QFormLayout()
        
        self.gray_zone_min_width = QDoubleSpinBox()
        self.gray_zone_min_width.setMinimum(0.0)
        self.gray_zone_min_width.setMaximum(1.0)
        self.gray_zone_min_width.setSingleStep(0.01)
        self.gray_zone_min_width.setDecimals(2)
        try:
            self.gray_zone_min_width.setValue(getattr(ConfigCLS, 'GRAY_ZONE_MIN_WIDTH', 0.05))
        except:
            self.gray_zone_min_width.setValue(0.05)
        gray_zone_layout.addRow("最小幅 (GRAY_ZONE_MIN_WIDTH):", self.gray_zone_min_width)
        
        self.gray_zone_max_width = QDoubleSpinBox()
        self.gray_zone_max_width.setMinimum(0.0)
        self.gray_zone_max_width.setMaximum(1.0)
        self.gray_zone_max_width.setSingleStep(0.01)
        self.gray_zone_max_width.setDecimals(2)
        try:
            self.gray_zone_max_width.setValue(getattr(ConfigCLS, 'GRAY_ZONE_MAX_WIDTH', 0.5))
        except:
            self.gray_zone_max_width.setValue(0.5)
        gray_zone_layout.addRow("最大幅 (GRAY_ZONE_MAX_WIDTH):", self.gray_zone_max_width)
        
        gray_zone_group.setLayout(gray_zone_layout)
        layout.addWidget(gray_zone_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_load_existing_tab(self):
        """ES: Crear tab para cargar análisis existente
        EN: Create the tab for loading an existing analysis
        JA: 既存解析を読み込むタブを作成
        """
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Instrucciones
        info_label = QLabel(
            "既存の分類解析結果を読み込みます。\n\n"
            "以下の構造を持つフォルダを選択してください:\n"
            "05_分類/分類解析結果_YYYYMMDD_HHMMSS/\n\n"
            "必要な構造:\n"
            "• 02_本学習結果/\n"
            "  - 01_モデル/\n"
            "    - final_bundle_cls.pkl\n"
            "  - 02_評価結果/\n"
            "    - (グラフPNGファイル)\n"
            "  - 04_診断情報/\n"
            "    - diagnostic_report.txt\n"
            "• 00_データセット/\n"
            "  - (データファイル)"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 12px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(info_label)
        
        # ES: Botón para seleccionar carpeta
        # EN: Button to select a folder
        # JA: フォルダ選択ボタン
        select_button = QPushButton("📁 フォルダを選択")
        select_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        select_button.clicked.connect(self.on_select_folder_clicked)
        layout.addWidget(select_button)
        
        # ES: Label para mostrar la ruta seleccionada
        # EN: Label to display the selected path
        # JA: 選択パス表示ラベル
        self.selected_folder_label = QLabel("選択されていません")
        self.selected_folder_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #7f8c8d;
                padding: 8px;
                background-color: #ecf0f1;
                border-radius: 5px;
                border: 1px solid #bdc3c7;
            }
        """)
        self.selected_folder_label.setWordWrap(True)
        layout.addWidget(self.selected_folder_label)
        
        # ES: Label para mostrar estado de validación
        # EN: Label to display validation status
        # JA: 検証ステータス表示ラベル
        self.validation_status_label = QLabel("")
        self.validation_status_label.setWordWrap(True)
        layout.addWidget(self.validation_status_label)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def on_select_folder_clicked(self):
        """ES: Maneja el clic en el botón de seleccionar carpeta
        EN: Handle the click on the folder selection button
        JA: フォルダ選択ボタンのクリックを処理
        """
        folder = QFileDialog.getExistingDirectory(
            self,
            "既存結果フォルダを選択",
            "",
            QFileDialog.ShowDirsOnly
        )
        
        if not folder:
            return
        
        # ES: Validar estructura | EN: Validate structure | JA: 構造を検証
        validation_result = self.validate_folder_structure(folder)
        
        if validation_result['is_valid']:
            self.selected_folder_label.setText(f"✅ {validation_result['validated_path']}")
            self.selected_folder_label.setStyleSheet("""
                QLabel {
                    font-size: 11px;
                    color: #27ae60;
                    padding: 8px;
                    background-color: #d5f4e6;
                    border-radius: 5px;
                    border: 1px solid #27ae60;
                    font-weight: bold;
                }
            """)
            self.validation_status_label.setText("✅ フォルダ構造が正しく検証されました。")
            self.validation_status_label.setStyleSheet("color: #27ae60; font-weight: bold; padding: 5px;")
            self.validated_folder_path = validation_result['validated_path']
            self.project_folder_path = validation_result['project_folder']
            self.is_folder_valid = True
        else:
            self.selected_folder_label.setText(f"❌ {folder}")
            self.selected_folder_label.setStyleSheet("""
                QLabel {
                    font-size: 11px;
                    color: #e74c3c;
                    padding: 8px;
                    background-color: #fadbd8;
                    border-radius: 5px;
                    border: 1px solid #e74c3c;
                }
            """)
            self.validation_status_label.setText(f"❌ {validation_result['error_message']}")
            self.validation_status_label.setStyleSheet("color: #e74c3c; font-weight: bold; padding: 5px;")
            self.validated_folder_path = None
            self.project_folder_path = None
            self.is_folder_valid = False
    
    def validate_folder_structure(self, folder_path):
        """
        ES: Valida la estructura de carpetas del análisis de clasificación existente.
        EN: Validate the folder structure of an existing classification analysis.
        JA: 既存の分類解析フォルダ構造を検証する。
        
        Returns:
            dict: {
                'is_valid': bool,
                'error_message': str,
                'validated_path': str,  # Path to analysis folder (分類解析結果_YYYYMMDD_HHMMSS)
                'project_folder': str   # Project folder path
            }
        """
        import re
        
        current_path = Path(folder_path)
        analysis_folder = None  # Classification analysis folder (分類解析結果_YYYYMMDD_HHMMSS)
        project_folder = None
        pattern = re.compile(r'^分類解析結果_\d{8}_\d{6}$')
        
        # ES: Archivos requeridos en 02_本学習結果/01_モデル
        # EN: Required files in 02_本学習結果/01_モデル
        # JA: 02_本学習結果/01_モデル の必須ファイル
        required_model_files = [
            'final_bundle_cls.pkl'
        ]
        
        # ES: Archivos requeridos en 02_本学習結果/04_診断情報
        # EN: Required files in 02_本学習結果/04_診断情報
        # JA: 02_本学習結果/04_診断情報 の必須ファイル
        required_diagnostic_files = [
            'diagnostic_report.txt'
        ]
        
        # ES: Caso 1: El usuario seleccionó directamente la carpeta 分類解析結果_YYYYMMDD_HHMMSS
        # EN: Case 1: The user directly selected the 分類解析結果_YYYYMMDD_HHMMSS folder
        # JA: ケース1：分類解析結果_YYYYMMDD_HHMMSS フォルダを直接選択
        if pattern.match(current_path.name):
            analysis_folder = current_path
            # ES: Buscar hacia arriba para encontrar 05_分類 y el proyecto
            # EN: Walk up to find 05_分類 and the project folder
            # JA: 上方向に辿って 05_分類 とプロジェクトを探す
            for parent in current_path.parents:
                if parent.name == "05_分類":
                    project_folder = parent.parent
                    break
        
        # ES: Caso 2: El usuario seleccionó una subcarpeta (02_本学習結果, 01_モデル, etc.)
        # EN: Case 2: The user selected a subfolder (02_本学習結果, 01_モデル, etc.)
        # JA: ケース2：サブフォルダ（02_本学習結果/01_モデル等）を選択
        elif current_path.name in ["02_本学習結果", "01_モデル", "02_評価結果", "04_診断情報", "00_データセット"]:
            # ES: La carpeta del análisis es el padre
            # EN: The analysis folder is the parent
            # JA: 解析フォルダは親ディレクトリ
            analysis_folder = current_path.parent
            # ES: Verificar que el nombre del padre coincida con el patrón
            # EN: Verify the parent name matches the expected pattern
            # JA: 親フォルダ名がパターンに一致するか確認
            if not pattern.match(analysis_folder.name):
                analysis_folder = None
            else:
                # ES: Buscar hacia arriba para encontrar 05_分類
                # EN: Walk up to find 05_分類
                # JA: 上方向に辿って 05_分類 を探す
                for parent in analysis_folder.parents:
                    if parent.name == "05_分類":
                        project_folder = parent.parent
                        break
        
        # ES: Caso 3: El usuario seleccionó 05_分類 o carpeta del proyecto
        # EN: Case 3: The user selected 05_分類 or the project folder
        # JA: ケース3：05_分類 またはプロジェクトフォルダを選択
        else:
            # ES: Buscar 05_分類 desde cualquier nivel
            # EN: Look for 05_分類 from any level
            # JA: どの階層からでも 05_分類 を探索
            classification_folder = None
            
            # ES: Buscar hacia arriba
            # EN: Search upwards
            # JA: 上方向に探索
            for parent in [current_path] + list(current_path.parents):
                classification_candidate = parent / "05_分類"
                if classification_candidate.exists() and classification_candidate.is_dir():
                    classification_folder = classification_candidate
                    project_folder = parent
                    break
            
            # ES: Si no se encuentra hacia arriba, buscar en el folder seleccionado
            # EN: If not found upwards, check within the selected folder
            # JA: 上方向で見つからなければ選択フォルダ内を確認
            if classification_folder is None:
                if current_path.name == "05_分類":
                    classification_folder = current_path
                    project_folder = current_path.parent
                elif (current_path / "05_分類").exists():
                    classification_folder = current_path / "05_分類"
                    project_folder = current_path
            
            if classification_folder is None:
                return {
                    'is_valid': False,
                    'error_message': '05_分類 フォルダが見つかりません。',
                    'validated_path': None,
                    'project_folder': None
                }
            
            # ES: Buscar carpeta con patrón 分類解析結果_YYYYMMDD_HHMMSS
            # EN: Find a folder matching 分類解析結果_YYYYMMDD_HHMMSS
            # JA: 分類解析結果_YYYYMMDD_HHMMSS に一致するフォルダを探索
            for item in classification_folder.iterdir():
                if item.is_dir() and pattern.match(item.name):
                    analysis_folder = item
                    break
            
            if analysis_folder is None:
                return {
                    'is_valid': False,
                    'error_message': '分類解析結果_YYYYMMDD_HHMMSS 形式のフォルダが見つかりません。',
                    'validated_path': None,
                    'project_folder': str(project_folder) if project_folder else None
                }
        
        # ES: Verificar que se encontró la carpeta del análisis
        # EN: Verify the analysis folder was found
        # JA: 解析フォルダが見つかったか確認
        if analysis_folder is None or not analysis_folder.exists():
            return {
                'is_valid': False,
                'error_message': '分析フォルダ (分類解析結果_YYYYMMDD_HHMMSS) が見つかりません。',
                'validated_path': None,
                'project_folder': str(project_folder) if project_folder else None
            }
        
        # ES: Verificar carpeta 02_本学習結果
        # EN: Verify 02_本学習結果 folder
        # JA: 02_本学習結果 フォルダを確認
        learning_result_folder = analysis_folder / "02_本学習結果"
        if not learning_result_folder.exists() or not learning_result_folder.is_dir():
            return {
                'is_valid': False,
                'error_message': '02_本学習結果 フォルダが見つかりません。',
                'validated_path': None,
                'project_folder': str(project_folder) if project_folder else None
            }
        
        # ES: Verificar carpeta 02_本学習結果/01_モデル
        # EN: Verify 02_本学習結果/01_モデル folder
        # JA: 02_本学習結果/01_モデル フォルダを確認
        model_folder = learning_result_folder / "01_モデル"
        if not model_folder.exists() or not model_folder.is_dir():
            return {
                'is_valid': False,
                'error_message': '02_本学習結果/01_モデル フォルダが見つかりません。',
                'validated_path': None,
                'project_folder': str(project_folder) if project_folder else None
            }
        
        # ES: Verificar archivos en 01_モデル
        # EN: Verify files in 01_モデル
        # JA: 01_モデル 内のファイルを確認
        missing_model_files = []
        for file_name in required_model_files:
            file_path = model_folder / file_name
            if not file_path.exists():
                missing_model_files.append(file_name)
        
        if missing_model_files:
            return {
                'is_valid': False,
                'error_message': f'01_モデル に以下のファイルが見つかりません: {", ".join(missing_model_files)}',
                'validated_path': None,
                'project_folder': str(project_folder) if project_folder else None
            }
        
        # ES: Verificar carpeta 02_本学習結果/04_診断情報
        # EN: Verify 02_本学習結果/04_診断情報 folder
        # JA: 02_本学習結果/04_診断情報 フォルダを確認
        diagnostic_folder = learning_result_folder / "04_診断情報"
        if not diagnostic_folder.exists() or not diagnostic_folder.is_dir():
            return {
                'is_valid': False,
                'error_message': '02_本学習結果/04_診断情報 フォルダが見つかりません。',
                'validated_path': None,
                'project_folder': str(project_folder) if project_folder else None
            }
        
        # ES: Verificar archivos en 04_診断情報
        # EN: Verify files in 04_診断情報
        # JA: 04_診断情報 内のファイルを確認
        missing_diagnostic_files = []
        for file_name in required_diagnostic_files:
            file_path = diagnostic_folder / file_name
            if not file_path.exists():
                missing_diagnostic_files.append(file_name)
        
        if missing_diagnostic_files:
            return {
                'is_valid': False,
                'error_message': f'04_診断情報 に以下のファイルが見つかりません: {", ".join(missing_diagnostic_files)}',
                'validated_path': None,
                'project_folder': str(project_folder) if project_folder else None
            }
        
        # ES: Si no se encontró project_folder, intentar buscarlo desde analysis_folder
        # EN: If project_folder was not found, try to infer it from analysis_folder
        # JA: project_folder が見つからなければ analysis_folder から推定
        if project_folder is None:
            for parent in analysis_folder.parents:
                if parent.name == "05_分類":
                    project_folder = parent.parent
                    break
        
        # ES: Todo está correcto
        # EN: Everything looks good
        # JA: 問題なし
        return {
            'is_valid': True,
            'error_message': '',
            'validated_path': str(analysis_folder),
            'project_folder': str(project_folder) if project_folder else None
        }
    
    def get_config_values(self):
        """Obtiene los valores configurados"""
        config_vals = {}
        
        # Características - obtener de las listas con checkboxes
        allowed_features = []
        for i in range(self.allowed_features_list.count()):
            item = self.allowed_features_list.item(i)
            if item.checkState() == Qt.Checked:
                allowed_features.append(item.text())
        config_vals['ALLOWED_FEATURES'] = set(allowed_features)
        
        must_keep_features = []
        for i in range(self.must_keep_features_list.count()):
            item = self.must_keep_features_list.item(i)
            if item.checkState() == Qt.Checked:
                must_keep_features.append(item.text())
        config_vals['MUST_KEEP_FEATURES'] = set(must_keep_features)
        
        continuous_features = []
        for i in range(self.continuous_features_list.count()):
            item = self.continuous_features_list.item(i)
            if item.checkState() == Qt.Checked:
                continuous_features.append(item.text())
        config_vals['CONTINUOUS_FEATURES'] = continuous_features
        
        discrete_features = []
        for i in range(self.discrete_features_list.count()):
            item = self.discrete_features_list.item(i)
            if item.checkState() == Qt.Checked:
                discrete_features.append(item.text())
        config_vals['DISCRETE_FEATURES'] = discrete_features
        
        binary_features = []
        for i in range(self.binary_features_list.count()):
            item = self.binary_features_list.item(i)
            if item.checkState() == Qt.Checked:
                binary_features.append(item.text())
        config_vals['BINARY_FEATURES'] = binary_features
        
        integer_features = []
        for i in range(self.integer_features_list.count()):
            item = self.integer_features_list.item(i)
            if item.checkState() == Qt.Checked:
                integer_features.append(item.text())
        config_vals['INTEGER_FEATURES'] = integer_features
        
        # Modelos
        config_vals['MODELS_TO_USE'] = [
            model for model, checkbox in self.model_checkboxes.items()
            if checkbox.isChecked()
        ]
        if not config_vals['MODELS_TO_USE']:
            config_vals['MODELS_TO_USE'] = ['lightgbm']  # Default
        
        config_vals['COMPARE_MODELS'] = self.compare_models.isChecked()
        config_vals['MODEL_COMPARISON_CV_SPLITS'] = self.model_comparison_cv_splits.value()
        config_vals['MODEL_COMPARISON_SCORING'] = self.model_comparison_scoring.currentText()
        
        # Optimización multiobjetivo
        config_vals['N_TRIALS_MULTI_OBJECTIVE'] = self.n_trials_multi_objective.value()
        config_vals['FP_WEIGHT'] = self.fp_weight.value()
        config_vals['COVERAGE_WEIGHT'] = self.coverage_weight.value()
        config_vals['AUC_WEIGHT'] = self.auc_weight.value()
        config_vals['NP_ALPHA_RANGE'] = (self.np_alpha_range_min.value(), self.np_alpha_range_max.value())
        
        # DCV
        config_vals['OUTER_SPLITS'] = self.outer_splits.value()
        config_vals['INNER_SPLITS'] = self.inner_splits.value()
        config_vals['RANDOM_STATE'] = self.random_state.value()
        config_vals['N_TRIALS_INNER'] = self.n_trials_inner.value()
        config_vals['USE_INNER_NOISE'] = self.use_inner_noise.isChecked()
        config_vals['NOISE_PPM'] = self.noise_ppm.value()
        config_vals['NOISE_RATIO'] = self.noise_ratio.value()
        
        # Umbrales
        config_vals['NP_ALPHA'] = self.np_alpha.value()
        config_vals['USE_UPPER_CI_ADJUST'] = self.use_upper_ci_adjust.isChecked()
        config_vals['CI_METHOD'] = self.ci_method.currentText()
        config_vals['CI_CONFIDENCE'] = self.ci_confidence.value()
        config_vals['TAU_NEG_FALLBACK_RATIO'] = self.tau_neg_fallback_ratio.value()
        
        # Evaluación
        config_vals['FINAL_EVALUATION_CV_SPLITS'] = self.final_evaluation_cv_splits.value()
        config_vals['FINAL_EVALUATION_SHUFFLE'] = self.final_evaluation_shuffle.isChecked()
        config_vals['FINAL_EVALUATION_RANDOM_STATE'] = self.final_evaluation_random_state.value()
        config_vals['HOLDOUT_TEST_SIZE'] = self.holdout_test_size.value()
        config_vals['HOLDOUT_STRATIFY'] = self.holdout_stratify.isChecked()
        config_vals['HOLDOUT_RANDOM_STATE'] = self.holdout_random_state.value()
        config_vals['GRAY_ZONE_MIN_WIDTH'] = self.gray_zone_min_width.value()
        config_vals['GRAY_ZONE_MAX_WIDTH'] = self.gray_zone_max_width.value()
        
        # ES: Cargar existente | EN: Load existing | JA: 既存読み込み
        config_vals['load_existing'] = self.is_folder_valid
        config_vals['selected_folder_path'] = self.validated_folder_path
        config_vals['project_folder'] = self.project_folder_path
        
        return config_vals
    
    def _show_data_info(self):
        """ES: Muestra información de los datos filtrados en el diálogo
        EN: Show information about the filtered data in the dialog
        JA: ダイアログにフィルタ済みデータ情報を表示
        """
        if self.filtered_df is None or self.filtered_df.empty:
            return
        
        # ES: Obtener información de los datos
        # EN: Collect data info
        # JA: データ情報を収集
        df = self.filtered_df
        info_lines = []
        
        # Información básica
        info_lines.append(f"📊 データ件数: {len(df)} 件")
        
        # Información de parámetros si están disponibles
        if '材料' in df.columns:
            materials = df['材料'].dropna().unique()
            if len(materials) > 0:
                info_lines.append(f"材料: {', '.join(map(str, materials))}")
        
        if '回転速度' in df.columns:
            rot_speeds = df['回転速度'].dropna()
            if len(rot_speeds) > 0:
                info_lines.append(f"回転速度: {rot_speeds.min():.0f} - {rot_speeds.max():.0f}")
        
        if '送り速度' in df.columns:
            feed_speeds = df['送り速度'].dropna()
            if len(feed_speeds) > 0:
                info_lines.append(f"送り速度: {feed_speeds.min():.0f} - {feed_speeds.max():.0f}")
        
        if '切込量' in df.columns:
            cut_depths = df['切込量'].dropna()
            if len(cut_depths) > 0:
                info_lines.append(f"切込量: {cut_depths.min():.2f} - {cut_depths.max():.2f}")
        
        if '突出量' in df.columns:
            protrusions = df['突出量'].dropna()
            if len(protrusions) > 0:
                info_lines.append(f"突出量: {protrusions.min():.0f} - {protrusions.max():.0f}")
        
        if '載せ率' in df.columns:
            load_ratios = df['載せ率'].dropna()
            if len(load_ratios) > 0:
                info_lines.append(f"載せ率: {load_ratios.min():.2f} - {load_ratios.max():.2f}")
        
        if 'パス数' in df.columns:
            passes = df['パス数'].dropna()
            if len(passes) > 0:
                info_lines.append(f"パス数: {passes.min():.0f} - {passes.max():.0f}")
        
        if '加工時間' in df.columns:
            times = df['加工時間'].dropna()
            if len(times) > 0:
                info_lines.append(f"加工時間: {times.min():.2f} - {times.max():.2f}")
        
        # Mostrar información en consola
        if info_lines:
            info_text = "\n".join(info_lines)
            print(f"📋 フィルタ済みデータ情報:\n{info_text}")

