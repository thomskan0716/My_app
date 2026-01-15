"""
Diálogo de configuración para análisis no lineal
Permite configurar parámetros de config.py antes de ejecutar
"""
import os
import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTabWidget, QWidget, QCheckBox, QSpinBox, QComboBox,
    QGroupBox, QFormLayout, QMessageBox, QDoubleSpinBox, QListWidget,
    QAbstractItemView, QListWidgetItem, QSplitter, QFileDialog
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

# ★ Asegurar que config.py se pueda importar
# Añadir el directorio actual y el directorio del script al sys.path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Intentar importar config
try:
    import config
except ImportError:
    # Si no se encuentra, intentar desde el directorio raíz del proyecto
    # Buscar config.py en el directorio padre o en el directorio actual
    config_paths = [
        current_dir / "config.py",
        current_dir.parent / "config.py",
        Path.cwd() / "config.py",
    ]
    
    config_found = False
    for config_path in config_paths:
        if config_path.exists():
            # Añadir el directorio del config.py al sys.path
            if str(config_path.parent) not in sys.path:
                sys.path.insert(0, str(config_path.parent))
            try:
                import config
                config_found = True
                break
            except ImportError:
                continue
    
    if not config_found:
        # Si aún no se encuentra, crear un módulo config dummy
        import types
        config = types.ModuleType('config')
        config.Config = types.SimpleNamespace()
        print("⚠️ Advertencia: No se pudo importar config.py, usando valores por defecto")


class NonlinearConfigDialog(QDialog):
    """Diálogo para configurar parámetros del análisis no lineal"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("非線形解析設定")
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)
        
        self.config_values = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        layout = QVBoxLayout()
        
        # Título
        title = QLabel("非線形解析パラメータ設定")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Crear pestañas
        tabs = QTabWidget()
        
        # Tab 1: Modelos
        models_tab = self.create_models_tab()
        tabs.addTab(models_tab, "モデル設定")
        
        # Tab 2: Hyperparameters
        hyperparams_tab = self.create_hyperparams_tab()
        tabs.addTab(hyperparams_tab, "ハイパーパラメータ")
        
        # Tab 3: Feature Selection
        features_tab = self.create_features_tab()
        tabs.addTab(features_tab, "特徴量選択")
        
        # Tab 4: Configuración General
        general_tab = self.create_general_tab()
        tabs.addTab(general_tab, "一般設定")
        
        # Tab 5: Pareto
        pareto_tab = self.create_pareto_tab()
        tabs.addTab(pareto_tab, "パレート設定")
        
        # Tab 6: Cargar existente
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
    
    def create_models_tab(self):
        """Crear tab de configuración de modelos"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Grupo: Modelos a usar
        models_group = QGroupBox("使用するモデル")
        models_layout = QVBoxLayout()
        
        self.model_checkboxes = {}
        models = [
            ('random_forest', 'Random Forest'),
            ('lightgbm', 'LightGBM'),
            ('xgboost', 'XGBoost'),
            ('gradient_boost', 'Gradient Boost'),
            ('ridge', 'Ridge'),
            ('lasso', 'Lasso'),
            ('elastic_net', 'Elastic Net')
        ]
        
        # Valores por defecto (checked)
        default_models = ['random_forest', 'lightgbm']
        
        for model_key, model_name in models:
            checkbox = QCheckBox(model_name)
            checkbox.setChecked(model_key in default_models)
            self.model_checkboxes[model_key] = checkbox
            models_layout.addWidget(checkbox)
        
        models_group.setLayout(models_layout)
        layout.addWidget(models_group)
        
        # Grupo: Configuración adicional
        misc_group = QGroupBox("その他の設定")
        misc_layout = QFormLayout()
        
        # Número de trials
        self.n_trials = QSpinBox()
        self.n_trials.setMinimum(10)
        self.n_trials.setMaximum(200)
        self.n_trials.setValue(50)
        misc_layout.addRow("Optuna試行回数:", self.n_trials)
        
        # Fallback model
        self.fallback_combo = QComboBox()
        self.fallback_combo.addItems(['random_forest', 'lightgbm', 'ridge', 'lasso'])
        self.fallback_combo.setCurrentText('ridge')
        misc_layout.addRow("フォールバックモデル:", self.fallback_combo)
        
        misc_group.setLayout(misc_layout)
        layout.addWidget(misc_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_hyperparams_tab(self):
        """Crear tab de configuración de hiperparámetros"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Cargar configuración actual
        try:
            current_config = config.Config.MODEL_CONFIGS
        except AttributeError:
            current_config = {}
        
        # Group: Default Model
        default_group = QGroupBox("デフォルトモデル")
        default_layout = QFormLayout()
        
        self.default_model_combo = QComboBox()
        models_list = ['random_forest', 'lightgbm', 'xgboost', 'gradient_boost', 'ridge', 'lasso', 'elastic_net']
        self.default_model_combo.addItems(models_list)
        try:
            current_default = getattr(config.Config, 'DEFAULT_MODEL', 'random_forest')
            index = models_list.index(current_default) if current_default in models_list else 0
            self.default_model_combo.setCurrentIndex(index)
        except (AttributeError, ValueError):
            pass
        default_layout.addRow("デフォルトモデル:", self.default_model_combo)
        
        default_group.setLayout(default_layout)
        layout.addWidget(default_group)
        
        # Group: Logging Settings
        logging_group = QGroupBox("ログ設定")
        logging_layout = QFormLayout()
        
        self.show_optuna_progress = QCheckBox()
        self.show_optuna_progress.setChecked(getattr(config.Config, 'SHOW_OPTUNA_PROGRESS', True))
        logging_layout.addRow("Optuna進捗表示:", self.show_optuna_progress)
        
        self.verbose_logging = QCheckBox()
        self.verbose_logging.setChecked(getattr(config.Config, 'VERBOSE_LOGGING', False))
        logging_layout.addRow("詳細ログ:", self.verbose_logging)
        
        self.show_data_analysis = QCheckBox()
        self.show_data_analysis.setChecked(getattr(config.Config, 'SHOW_DATA_ANALYSIS_DETAILS', True))
        logging_layout.addRow("データ分析詳細:", self.show_data_analysis)
        
        logging_group.setLayout(logging_layout)
        layout.addWidget(logging_group)
        
        # Hyperparameters configuration (will be expanded based on selected models)
        info_label = QLabel("各モデルのハイパーパラメータ範囲を設定します。\n使用するモデルは「モデル設定」タブで選択してください。")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_features_tab(self):
        """Crear tab de selección de características"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        info_label = QLabel("分析に使用する特徴量を選択します。\n青いチェックマークは推奨必須特徴量です。")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Single list for all features
        features_group = QGroupBox("説明変数選択")
        features_layout = QVBoxLayout()
        self.features_list = QListWidget()
        
        # Get all features from config
        try:
            feature_columns = list(config.Config.FEATURE_COLUMNS)
            mandatory_features = set(config.Config.MANDATORY_FEATURES)
            
            # Add all features, marking mandatory ones
            for feature in feature_columns:
                item = QListWidgetItem(feature)
                item.setCheckState(Qt.Checked)
                
                # Mark mandatory features (but allow them to be unchecked)
                if feature in mandatory_features:
                    # Color the text blue to indicate recommended
                    blue_color = QColor(0, 0, 255)  # Blue color
                    item.setForeground(blue_color)
                self.features_list.addItem(item)
        except Exception as e:
            print(f"Error loading features: {e}")
        
        features_layout.addWidget(self.features_list)
        features_group.setLayout(features_layout)
        layout.addWidget(features_group)
        
        tab.setLayout(layout)
        return tab
    
    def create_general_tab(self):
        """Crear tab de configuración general"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Grupo: Características
        features_group = QGroupBox("特徴量設定")
        features_layout = QFormLayout()
        
        # Top K
        self.top_k = QSpinBox()
        self.top_k.setMinimum(5)
        self.top_k.setMaximum(100)
        self.top_k.setValue(20)
        features_layout.addRow("特徴選択数 (top_k):", self.top_k)
        
        # Correlación threshold
        from PySide6.QtWidgets import QDoubleSpinBox
        self.corr_threshold = QDoubleSpinBox()
        self.corr_threshold.setMinimum(0.5)
        self.corr_threshold.setMaximum(1.0)
        self.corr_threshold.setSingleStep(0.05)
        self.corr_threshold.setValue(0.95)
        features_layout.addRow("相関閾値:", self.corr_threshold)
        
        # Use correlation removal
        self.use_corr_removal = QCheckBox()
        self.use_corr_removal.setChecked(True)
        features_layout.addRow("相関除去機能:", self.use_corr_removal)
        
        features_group.setLayout(features_layout)
        layout.addWidget(features_group)
        
        # Grupo: Transformación
        transform_group = QGroupBox("変数変換")
        transform_layout = QFormLayout()
        
        # Transform method
        self.transform_method = QComboBox()
        self.transform_method.addItems(['auto', 'yeo-johnson', 'quantile', 'robust', 'log', 'sqrt', 'none'])
        self.transform_method.setCurrentText('auto')
        transform_layout.addRow("変換方法:", self.transform_method)
        
        transform_group.setLayout(transform_layout)
        layout.addWidget(transform_group)
        
        # Grupo: CV
        cv_group = QGroupBox("クロスバリデーション")
        cv_layout = QFormLayout()
        
        self.outer_splits = QSpinBox()
        self.outer_splits.setMinimum(3)
        self.outer_splits.setMaximum(20)
        self.outer_splits.setValue(10)
        cv_layout.addRow("外側分割数:", self.outer_splits)
        
        self.inner_splits = QSpinBox()
        self.inner_splits.setMinimum(3)
        self.inner_splits.setMaximum(20)
        self.inner_splits.setValue(10)
        cv_layout.addRow("内側分割数:", self.inner_splits)
        
        cv_group.setLayout(cv_layout)
        layout.addWidget(cv_group)
        
        # Grupo: SHAP
        shap_group = QGroupBox("SHAP分析")
        shap_layout = QFormLayout()
        
        self.shap_mode = QComboBox()
        self.shap_mode.addItems(['none', 'summary', 'detailed', 'full'])
        self.shap_mode.setCurrentText('detailed')
        shap_layout.addRow("SHAPモード:", self.shap_mode)
        
        self.shap_max_samples = QSpinBox()
        self.shap_max_samples.setMinimum(50)
        self.shap_max_samples.setMaximum(500)
        self.shap_max_samples.setValue(200)
        shap_layout.addRow("最大サンプル数:", self.shap_max_samples)
        
        shap_group.setLayout(shap_layout)
        layout.addWidget(shap_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_pareto_tab(self):
        """Crear tab de configuración de Pareto"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        info_label = QLabel("パレート分析の目的変数と最適化方向を設定します。")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Grupo: Objetivos Pareto
        objectives_group = QGroupBox("目的変数")
        objectives_layout = QVBoxLayout()
        
        self.pareto_objectives = {}
        
        # Lista de objetivos con sus direcciones por defecto
        objective_configs = [
            ('摩耗量', 'min'),
            ('切削時間', 'min'),
            ('上面ダレ量', 'min'),
            ('側面ダレ量', 'min')
        ]
        
        for obj_name, default_dir in objective_configs:
            row = QHBoxLayout()
            
            checkbox = QCheckBox(obj_name)
            checkbox.setChecked(True)
            self.pareto_objectives[f"{obj_name}_checkbox"] = checkbox
            row.addWidget(checkbox)
            
            dir_combo = QComboBox()
            dir_combo.addItems(['min', 'max'])
            dir_combo.setCurrentText(default_dir)
            self.pareto_objectives[f"{obj_name}_direction"] = dir_combo
            row.addWidget(dir_combo)
            
            row.addStretch()
            objectives_layout.addLayout(row)
        
        objectives_group.setLayout(objectives_layout)
        layout.addWidget(objectives_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_load_existing_tab(self):
        """Crear tab para cargar análisis existente"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Instrucciones
        info_label = QLabel(
            "既存の非線形解析結果を読み込みます。\n\n"
            "以下の構造を持つフォルダを選択してください:\n"
            "04_非線形回帰/NUM_YYYYMMDD_HHMMSS/\n\n"
            "必要な構造:\n"
            "• 02_学習モデル/\n"
            "  - final_model_上面ダレ量.pkl\n"
            "  - final_model_側面ダレ量.pkl\n"
            "  - final_model_摩耗量.pkl\n"
            "• 03_学習結果/\n"
            "  - dcv_results.pkl\n"
            "  - analysis_results.json\n"
            "  - 上面ダレ量_results.png\n"
            "  - 側面ダレ量_results.png\n"
            "  - 摩耗量_results.png\n"
            "  - data_analysis/ (フォルダ)\n"
            "    - analysis_report.json\n"
            "    - correlation_heatmap.png\n"
            "    - data_overview.png\n"
            "    - features_distribution.png\n"
            "    - statistics.csv\n"
            "    - target_*.png"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 12px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(info_label)
        
        # Botón para seleccionar carpeta
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
        
        # Label para mostrar la ruta seleccionada
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
        
        # Label para mostrar estado de validación
        self.validation_status_label = QLabel("")
        self.validation_status_label.setWordWrap(True)
        layout.addWidget(self.validation_status_label)
        
        # Variables para almacenar la validación
        self.validated_folder_path = None
        self.project_folder_path = None
        self.is_folder_valid = False
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def on_select_folder_clicked(self):
        """Maneja el clic en el botón de seleccionar carpeta"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "既存結果フォルダを選択",
            "",
            QFileDialog.ShowDirsOnly
        )
        
        if not folder:
            return
        
        # Validar estructura
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
        Valida la estructura de carpetas del análisis no lineal existente
        Basado en la nueva estructura:
        - 02_学習モデル: debe tener final_model_*.pkl
        - 03_学習結果: debe tener dcv_results.pkl, analysis_results.json, y PNGs
        - 03_学習結果/data_analysis: debe tener archivos de análisis
        
        Returns:
            dict: {
                'is_valid': bool,
                'error_message': str,
                'validated_path': str,  # Ruta a la carpeta del análisis (NUM_YYYYMMDD_HHMMSS)
                'project_folder': str   # Carpeta del proyecto
            }
        """
        import re
        
        current_path = Path(folder_path)
        analysis_folder = None  # Carpeta NUM_YYYYMMDD_HHMMSS
        project_folder = None
        pattern = re.compile(r'^\d+_\d{8}_\d{6}$')
        
        # Archivos requeridos en 02_学習モデル
        required_model_files = [
            'final_model_上面ダレ量.pkl',
            'final_model_側面ダレ量.pkl',
            'final_model_摩耗量.pkl'
        ]
        
        # Archivos requeridos en 03_学習結果
        required_result_files = [
            'dcv_results.pkl',
            'analysis_results.json',
            '上面ダレ量_results.png',
            '側面ダレ量_results.png',
            '摩耗量_results.png'
        ]
        
        # Archivos requeridos en 03_学習結果/data_analysis
        required_data_analysis_files = [
            'analysis_report.json',
            'correlation_heatmap.png',  # Nota: sin guión bajo (heatmap, no heat_map)
            'data_overview.png',
            'features_distribution.png',
            'statistics.csv',
            'target_上面ダレ量.png',
            'target_側面ダレ量.png',
            'target_摩耗量.png'
        ]
        
        # Caso 1: El usuario seleccionó directamente la carpeta NUM_YYYYMMDD_HHMMSS
        if pattern.match(current_path.name):
            analysis_folder = current_path
            # Buscar hacia arriba para encontrar 04_非線形回帰 y el proyecto
            for parent in current_path.parents:
                if parent.name == "04_非線形回帰":
                    project_folder = parent.parent
                    break
        
        # Caso 2: El usuario seleccionó 02_学習モデル o 03_学習結果
        elif current_path.name in ["02_学習モデル", "03_学習結果"]:
            # La carpeta del análisis es el padre
            analysis_folder = current_path.parent
            # Verificar que el nombre del padre coincida con el patrón
            if not pattern.match(analysis_folder.name):
                analysis_folder = None
            else:
                # Buscar hacia arriba para encontrar 04_非線形回帰
                for parent in analysis_folder.parents:
                    if parent.name == "04_非線形回帰":
                        project_folder = parent.parent
                        break
        
        # Caso 3: El usuario seleccionó 04_非線形回帰 o carpeta del proyecto
        else:
            # Buscar 04_非線形回帰 desde cualquier nivel
            nonlinear_folder = None
            
            # Buscar hacia arriba
            for parent in [current_path] + list(current_path.parents):
                nonlinear_candidate = parent / "04_非線形回帰"
                if nonlinear_candidate.exists() and nonlinear_candidate.is_dir():
                    nonlinear_folder = nonlinear_candidate
                    project_folder = parent
                    break
            
            # Si no se encuentra hacia arriba, buscar en el folder seleccionado
            if nonlinear_folder is None:
                if current_path.name == "04_非線形回帰":
                    nonlinear_folder = current_path
                    project_folder = current_path.parent
                elif (current_path / "04_非線形回帰").exists():
                    nonlinear_folder = current_path / "04_非線形回帰"
                    project_folder = current_path
            
            if nonlinear_folder is None:
                return {
                    'is_valid': False,
                    'error_message': '04_非線形回帰 フォルダが見つかりません。',
                    'validated_path': None,
                    'project_folder': None
                }
            
            # Buscar carpeta con patrón NUM_YYYYMMDD_HHMMSS
            for item in nonlinear_folder.iterdir():
                if item.is_dir() and pattern.match(item.name):
                    analysis_folder = item
                    break
            
            if analysis_folder is None:
                return {
                    'is_valid': False,
                    'error_message': 'NUM_YYYYMMDD_HHMMSS 形式のフォルダが見つかりません。',
                    'validated_path': None,
                    'project_folder': str(project_folder) if project_folder else None
                }
        
        # Verificar que se encontró la carpeta del análisis
        if analysis_folder is None or not analysis_folder.exists():
            return {
                'is_valid': False,
                'error_message': '分析フォルダ (NUM_YYYYMMDD_HHMMSS) が見つかりません。',
                'validated_path': None,
                'project_folder': str(project_folder) if project_folder else None
            }
        
        # Verificar carpeta 02_学習モデル
        model_folder = analysis_folder / "02_学習モデル"
        if not model_folder.exists() or not model_folder.is_dir():
            return {
                'is_valid': False,
                'error_message': '02_学習モデル フォルダが見つかりません。',
                'validated_path': None,
                'project_folder': str(project_folder) if project_folder else None
            }
        
        # Verificar archivos en 02_学習モデル
        missing_model_files = []
        for file_name in required_model_files:
            file_path = model_folder / file_name
            if not file_path.exists():
                missing_model_files.append(file_name)
        
        if missing_model_files:
            return {
                'is_valid': False,
                'error_message': f'02_学習モデル に以下のファイルが見つかりません: {", ".join(missing_model_files)}',
                'validated_path': None,
                'project_folder': str(project_folder) if project_folder else None
            }
        
        # Verificar carpeta 03_学習結果
        result_folder = analysis_folder / "03_学習結果"
        if not result_folder.exists() or not result_folder.is_dir():
            return {
                'is_valid': False,
                'error_message': '03_学習結果 フォルダが見つかりません。',
                'validated_path': None,
                'project_folder': str(project_folder) if project_folder else None
            }
        
        # Verificar archivos en 03_学習結果
        missing_result_files = []
        for file_name in required_result_files:
            file_path = result_folder / file_name
            if not file_path.exists():
                missing_result_files.append(file_name)
        
        if missing_result_files:
            return {
                'is_valid': False,
                'error_message': f'03_学習結果 に以下のファイルが見つかりません: {", ".join(missing_result_files)}',
                'validated_path': None,
                'project_folder': str(project_folder) if project_folder else None
            }
        
        # Verificar carpeta data_analysis dentro de 03_学習結果
        data_analysis_folder = result_folder / "data_analysis"
        if not data_analysis_folder.exists() or not data_analysis_folder.is_dir():
            return {
                'is_valid': False,
                'error_message': '03_学習結果/data_analysis フォルダが見つかりません。',
                'validated_path': None,
                'project_folder': str(project_folder) if project_folder else None
            }
        
        # Verificar archivos en data_analysis
        missing_data_analysis_files = []
        for file_name in required_data_analysis_files:
            file_path = data_analysis_folder / file_name
            if not file_path.exists():
                missing_data_analysis_files.append(file_name)
        
        if missing_data_analysis_files:
            return {
                'is_valid': False,
                'error_message': f'data_analysis に以下のファイルが見つかりません: {", ".join(missing_data_analysis_files)}',
                'validated_path': None,
                'project_folder': str(project_folder) if project_folder else None
            }
        
        # Si no se encontró project_folder, intentar buscarlo desde analysis_folder
        if project_folder is None:
            for parent in analysis_folder.parents:
                if parent.name == "04_非線形回帰":
                    project_folder = parent.parent
                    break
        
        # Todo está correcto
        return {
            'is_valid': True,
            'error_message': '',
            'validated_path': str(analysis_folder),
            'project_folder': str(project_folder) if project_folder else None
        }
    
    def get_config_values(self):
        """Obtiene los valores configurados"""
        config_vals = {}
        
        # Modelos seleccionados
        config_vals['models_to_use'] = [
            model for model, checkbox in self.model_checkboxes.items()
            if checkbox.isChecked()
        ]
        
        if not config_vals['models_to_use']:
            config_vals['models_to_use'] = ['random_forest']  # Default
        
        # Configuración adicional (usar N_TRIALS en mayúsculas para consistencia)
        config_vals['N_TRIALS'] = self.n_trials.value()
        config_vals['n_trials'] = self.n_trials.value()  # Mantener también en minúsculas para compatibilidad
        config_vals['fallback_model'] = self.fallback_combo.currentText()
        
        # Default model and logging settings
        try:
            config_vals['default_model'] = self.default_model_combo.currentText()
        except:
            pass
        
        try:
            config_vals['show_optuna_progress'] = self.show_optuna_progress.isChecked()
        except:
            pass
        
        try:
            config_vals['verbose_logging'] = self.verbose_logging.isChecked()
        except:
            pass
        
        try:
            config_vals['show_data_analysis'] = self.show_data_analysis.isChecked()
        except:
            pass
        
        # Feature selection
        try:
            selected_features = []
            
            # Add all checked features
            for i in range(self.features_list.count()):
                item = self.features_list.item(i)
                if item.checkState() == Qt.Checked:
                    selected_features.append(item.text())
            
            config_vals['selected_features'] = selected_features
        except Exception as e:
            print(f"Error getting selected features: {e}")
        
        # Características
        config_vals['top_k'] = self.top_k.value()
        config_vals['corr_threshold'] = self.corr_threshold.value()
        config_vals['use_correlation_removal'] = self.use_corr_removal.isChecked()
        
        # Transformación
        config_vals['transform_method'] = self.transform_method.currentText()
        
        # CV
        config_vals['outer_splits'] = self.outer_splits.value()
        config_vals['inner_splits'] = self.inner_splits.value()
        
        # SHAP
        config_vals['shap_mode'] = self.shap_mode.currentText()
        config_vals['shap_max_samples'] = self.shap_max_samples.value()
        
        # Pareto
        config_vals['pareto_objectives'] = {}
        for obj_name in ['摩耗量', '切削時間', '上面ダレ量', '側面ダレ量']:
            checkbox = self.pareto_objectives.get(f"{obj_name}_checkbox")
            direction = self.pareto_objectives.get(f"{obj_name}_direction")
            
            if checkbox and checkbox.isChecked():
                config_vals['pareto_objectives'][obj_name] = direction.currentText()
        
        # Cargar existente
        config_vals['load_existing'] = self.is_folder_valid
        config_vals['selected_folder_path'] = self.validated_folder_path
        config_vals['project_folder'] = self.project_folder_path
        
        return config_vals



