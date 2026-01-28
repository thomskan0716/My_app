"""
ES: Worker para ejecutar análisis de clasificación (bunrui kaiseki) en un thread separado.
EN: Worker to run classification analysis (bunrui kaiseki) in a separate thread.
JA: 分類解析（bunrui kaiseki）を別スレッドで実行するワーカー。

ES: Ejecuta Run_pipeline_ver3.3_20250914.py.
EN: Runs Run_pipeline_ver3.3_20250914.py.
JA: Run_pipeline_ver3.3_20250914.py を実行する。
"""
import sys
import os
import subprocess
import pandas as pd
import json
import time
import threading
import re
import shutil
from pathlib import Path
from PySide6.QtCore import QThread, Signal, QMetaObject, Qt


class ClassificationWorker(QThread):
    """ES: Worker que ejecuta el análisis de clasificación en un thread separado
    EN: Worker that runs the classification analysis in a separate thread
    JA: 分類解析を別スレッドで実行するワーカー
    """
    
    # ES: Señales para comunicación con la GUI | EN: Signals for GUI communication | JA: GUI通信用シグナル
    progress_updated = Signal(int, str)  # (value, message)
    status_updated = Signal(str)  # message
    finished = Signal(dict)  # results dict
    error = Signal(str)  # error message
    console_output = Signal(str)  # console output (for IDE/terminal)
    file_selection_requested = Signal(str)  # (initial_path) - request file selection
    
    def __init__(self, filtered_df, project_folder, parent=None, config_values=None, selected_brush=None, selected_material=None, selected_wire_length=None):
        """
        ES: Inicializa el worker.
        EN: Initialize the worker.
        JA: ワーカーを初期化する。
        
        Parameters
        ----------
        filtered_df : pd.DataFrame
            ES: DataFrame con los datos filtrados
            EN: DataFrame containing filtered data
            JA: フィルタ済みデータのDataFrame
        project_folder : str
            ES: Carpeta base del proyecto
            EN: Project base folder
            JA: プロジェクトのベースフォルダ
        parent : QWidget, optional
            ES: Widget padre
            EN: Parent widget
            JA: 親ウィジェット
        config_values : dict, optional
            ES: Valores de configuración del diálogo
            EN: Configuration values from the dialog
            JA: ダイアログからの設定値
        selected_brush : str, optional
            ES: Tipo de cepillo seleccionado (A11, A21, o A32) para Prediction_input.xlsx
            EN: Selected brush type (A11, A21, or A32) for Prediction_input.xlsx
            JA: Prediction_input.xlsx 用に選択したブラシタイプ（A11/A21/A32）
        selected_material : str, optional
            ES: Material seleccionado (Steel, Alum) para Prediction_input.xlsx
            EN: Selected material (Steel, Alum) for Prediction_input.xlsx
            JA: Prediction_input.xlsx 用に選択した材料（Steel/Alum）
        selected_wire_length : int, optional
            ES: Longitud de alambre seleccionada (30-75mm) para Prediction_input.xlsx
            EN: Selected wire length (30–75mm) for Prediction_input.xlsx
            JA: Prediction_input.xlsx 用に選択した線材長（30–75mm）
        """
        super().__init__(parent)
        self.filtered_df = filtered_df
        self.project_folder = project_folder
        self.config_values = config_values or {}
        self.selected_brush = selected_brush or "A13"  # Default: A13
        self.selected_material = selected_material or "Steel"  # Default: Steel
        self.selected_wire_length = selected_wire_length or 75  # Default: 75
        self.output_folder = None
        self._cancelled = False
        self._current_process = None
        self._json_reader_stop = threading.Event()
        self._stop_reading = None
        self._selected_file_path = None  # Selected file path (set by the user)
        self._file_selection_event = threading.Event()  # File-selection synchronization event
        
        # ES: Estado del progreso para parsing (similar a nonlinear_worker) | EN: Parsing progress state (similar to nonlinear_worker) | JA: パース進捗状態（nonlinear_workerと同様）
        self.current_fold = 0
        self.total_folds = self.config_values.get('OUTER_SPLITS', 10)
        self.current_trial = 0
        self.total_trials = self.config_values.get('N_TRIALS_INNER', 50)
        self.current_model = 0
        self.total_models = len(self.config_values.get('MODELS_TO_USE', ['lightgbm']))
        
        # ES: Estados de tareas | EN: Task states | JA: タスク状態
        self.model_comparison_completed = False
        self.multiobjective_completed = False
        self.dcv_training = False
        self.prediction_completed = False
        self.evaluation_completed = False
        self.current_task = 'initialization'  # initialization, model_comparison, multiobjective, dcv, prediction, evaluation
    
    def cancel(self):
        """ES: Cancela la ejecución del análisis
        EN: Cancel the analysis execution
        JA: 解析実行をキャンセル
        """
        self._cancelled = True
        if self._current_process:
            try:
                self._current_process.terminate()
                self._current_process.wait(timeout=5)
            except:
                try:
                    self._current_process.kill()
                except:
                    pass
        self._json_reader_stop.set()
    
    def run(self):
        """ES: Ejecuta el análisis de clasificación
        EN: Run the classification analysis
        JA: 分類解析を実行
        """
        start_time = time.time()
        
        try:
            # ES: Verificar si es carga de carpeta existente | EN: Check if loading an existing folder | JA: 既存フォルダ読み込みか確認
            load_existing = self.config_values.get('load_existing', False)
            selected_folder_path = self.config_values.get('selected_folder_path', '')
            
            if load_existing and selected_folder_path:
                # ES: Cargar carpeta existente sin ejecutar análisis | EN: Load existing folder without running analysis | JA: 解析せず既存フォルダを読み込み
                self.status_updated.emit("📁 既存結果を読み込み中...")
                self.progress_updated.emit(50, "既存結果を読み込み中...")
                
                # ES: Usar la carpeta seleccionada como output_folder | EN: Use selected folder as output_folder | JA: 選択フォルダを output_folder に設定
                self.output_folder = selected_folder_path
                
                # ES: Buscar resultados generados | EN: Find generated results | JA: 生成された結果を探索
                results = self._find_results()
                
                # ES: Emitir resultados como carga existente | EN: Emit results as an existing-load run | JA: 既存読み込みとして結果を送信
                results_existing = {
                    'output_folder': self.output_folder,
                    'analysis_duration': 0,  # No duration for existing analysis
                    'project_folder': self.config_values.get('project_folder', self.project_folder),
                    'load_existing': True,
                    'existing_folder_path': selected_folder_path,
                    'result_folders': results.get('result_folders', []),
                    'graph_paths': results.get('graph_paths', []),
                    'model_files': results.get('model_files', []),
                    'evaluation_files': results.get('evaluation_files', [])
                }
                
                self.progress_updated.emit(100, "既存結果読み込み完了")
                self.status_updated.emit("✅ 既存結果を読み込みました。")
                
                # ES: Emitir finished para que la GUI muestre los resultados existentes | EN: Emit finished so the GUI can show existing results | JA: GUI表示のため finished を送信
                self.finished.emit(results_existing)
                return
            
            # ES: Verificar cancelación | EN: Check cancellation | JA: キャンセル確認
            if self._cancelled:
                return
            
            # ES: Crear carpeta de salida 05_分類 | EN: Create output folder 05_分類 | JA: 出力フォルダ 05_分類 を作成
            self.status_updated.emit("📁 Creando carpeta de salida...")
            classification_folder = os.path.join(self.project_folder, "05_分類")
            os.makedirs(classification_folder, exist_ok=True)
            
            # ES: Crear subcarpeta con timestamp (carpeta de salida directa) | EN: Create timestamp subfolder (direct output folder) | JA: タイムスタンプ付きサブフォルダ（直接の出力先）を作成
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = os.path.join(classification_folder, f"分類解析結果_{timestamp}")
            os.makedirs(self.output_folder, exist_ok=True)
            
            # ES: No copiar ml_modules ni Run_pipeline; usar los del .venv directamente | EN: Do not copy ml_modules/Run_pipeline; use the ones from .venv | JA: ml_modules/Run_pipeline はコピーせず .venv のものを使用
            # ES: Buscar ml_modules en .venv | EN: Locate ml_modules in .venv | JA: .venv 内の ml_modules を探索
            script_dir = Path(__file__).parent.absolute()
            venv_ml_modules = script_dir / "ml_modules"
            
            # ES: Si no está en el directorio del script, buscar en el directorio padre (.venv) | EN: If not in the script dir, search parent dir (.venv) | JA: スクリプト直下になければ親ディレクトリ（.venv）を探索
            if not venv_ml_modules.exists() or not (venv_ml_modules / "models_cls.py").exists():
                venv_ml_modules = script_dir.parent / "ml_modules"
            
            # ES: Verificar que ml_modules existe | EN: Verify ml_modules exists | JA: ml_modules の存在確認
            if not venv_ml_modules.exists() or not (venv_ml_modules / "models_cls.py").exists():
                self.error.emit(f"❌ ml_modules が見つかりません: {venv_ml_modules}")
                return
            
            print(f"✅ ml_modules を見つけました: {venv_ml_modules}")
            
            # ES: Buscar Run_pipeline_ver3.3_20250914.py en .venv | EN: Locate Run_pipeline_ver3.3_20250914.py in .venv | JA: .venv 内の Run_pipeline_ver3.3_20250914.py を探索
            venv_pipeline_script = script_dir / "Run_pipeline_ver3.3_20250914.py"
            if not venv_pipeline_script.exists():
                venv_pipeline_script = script_dir.parent / "Run_pipeline_ver3.3_20250914.py"
            
            if not venv_pipeline_script.exists():
                self.error.emit(f"❌ Run_pipeline_ver3.3_20250914.py が見つかりません: {venv_pipeline_script}")
                return
            
            print(f"✅ パイプラインスクリプトを見つけました: {venv_pipeline_script}")
            
            # ES: Crear carpeta 00_データセット en la carpeta de salida | EN: Create 00_データセット under output folder | JA: 出力先に 00_データセット を作成
            data_folder = os.path.join(self.output_folder, "00_データセット")
            os.makedirs(data_folder, exist_ok=True)
            
            # ES: Guardar datos filtrados en 00_データセット | EN: Save filtered data into 00_データセット | JA: フィルタ済みデータを 00_データセット に保存
            self.status_updated.emit("💾 Guardando datos filtrados...")
            # ES: Usar fecha actual para el nombre del archivo | EN: Use current date for the filename | JA: ファイル名に現在日付を使用
            from datetime import datetime
            date_str = datetime.now().strftime("%Y%m%d")
            input_filename = f"{date_str}_総実験データ.xlsx"
            input_file = os.path.join(data_folder, input_filename)
            self.filtered_df.to_excel(input_file, index=False)
            print(f"✅ データを保存しました: {input_file}")
            
            # ES: Guardar el nombre del archivo para usarlo en la configuración | EN: Store the filename for config generation | JA: 設定生成のためファイル名を保持
            self.input_filename = input_filename
            
            # ES: Buscar y procesar archivo 未実験データ.xlsx del proyecto | EN: Find and process project's 未実験データ.xlsx | JA: プロジェクトの未実験データ.xlsxを探索・処理
            self.status_updated.emit("📋 Procesando archivo de predicción...")
            predict_input_file = self._create_prediction_input_file(data_folder)
            if not predict_input_file:
                self.error.emit("❌ Prediction_input.xlsx を作成できませんでした")
                return
            
            print(f"✅ 予測用ファイルを作成しました: {predict_input_file}")
            
            # ES: Verificar cancelación | EN: Check cancellation | JA: キャンセル確認
            if self._cancelled:
                return
            
            # ES: Crear archivo de configuración temporal en la carpeta de salida | EN: Create temporary config file in the output folder | JA: 出力先に一時設定ファイルを作成
            self.status_updated.emit("⚙️ Creando configuración temporal...")
            # ES: El archivo de configuración se guarda directamente en output_folder/config_cls.py
            # EN: The config file is saved directly to output_folder/config_cls.py
            # JP: 設定ファイルは output_folder/config_cls.py に直接保存される
            config_file = self._create_temp_config()
            
            if config_file and os.path.exists(config_file):
                print(f"✅ 設定ファイルを作成しました: {config_file}")
            
            # ES: Usar el script original del .venv (no copiado) | EN: Use the original .venv script (not copied) | JA: .venv の元スクリプトを使用（コピーしない）
            pipeline_script = str(venv_pipeline_script)
            
            # ES: Verificar cancelación | EN: Check cancellation | JA: キャンセル確認
            if self._cancelled:
                return
            
            # ES: Ejecutar el pipeline | EN: Run the pipeline | JA: パイプラインを実行
            self.status_updated.emit("🔧 Ejecutando pipeline de clasificación...")
            self.progress_updated.emit(20, "Pipeline実行中...")
            
            success = self._run_pipeline(pipeline_script, self.output_folder, config_file)
            
            if self._cancelled:
                return
            
            if not success:
                self.error.emit("❌ Error ejecutando el pipeline de clasificación")
                return
            
            # ES: Buscar resultados generados | EN: Find generated results | JA: 生成された結果を探索
            self.status_updated.emit("📊 Buscando resultados...")
            results = self._find_results()
            
            # ES: Calcular tiempo total | EN: Compute total time | JA: 総時間を計算
            end_time = time.time()
            analysis_duration = end_time - start_time
            
            results['output_folder'] = self.output_folder
            results['analysis_duration'] = analysis_duration
            results['project_folder'] = self.project_folder
            results['load_existing'] = False  # Not an existing-load; it's a new analysis
            
            self.progress_updated.emit(100, "分析完了")
            self.status_updated.emit("✅ 分類分析が完了しました")
            
            self.finished.emit(results)
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error en análisis de clasificación: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.error.emit(error_msg)
    
    def _create_temp_config(self):
        """ES: Crea un archivo de configuración temporal basado en config_values
        EN: Create a temporary config file from config_values
        JA: config_values から一時設定ファイルを作成"""
        # ES: El pipeline busca config_cls.py en ml_modules, así que creamos
        # EN: The pipeline looks for config_cls.py in ml_modules, so we create
        # JP: パイプラインはml_modules内のconfig_cls.pyを探すため、作成する
        # ES: un 99_ml_modules en la carpeta de salida solo con config_cls.py
        # EN: a 99_ml_modules folder under the output folder containing only config_cls.py
        # JP: 出力先にconfig_cls.pyだけ入った99_ml_modulesを作る
        ml_modules_dst = Path(self.output_folder) / "99_ml_modules"
        ml_modules_dst.mkdir(parents=True, exist_ok=True)
        
        config_file = ml_modules_dst / "config_cls.py"
        
        # ES: Crear carpeta 99_-----------------
        # EN: Create folder 99_-----------------
        # JP: 99_----------------- フォルダを作成
        separator_folder = Path(self.output_folder) / "99_-----------------"
        separator_folder.mkdir(parents=True, exist_ok=True)
        
        # ES: También crear ml_modules como symlink a 99_ml_modules para compatibilidad con el pipeline
        # EN: Also create ml_modules as a symlink to 99_ml_modules for pipeline compatibility
        # JP: パイプライン互換のため、ml_modulesを99_ml_modulesへのシンボリックリンクとして作成する
        # ES: El pipeline busca BASE / "ml_modules", así que necesitamos crear este symlink
        # EN: The pipeline looks for BASE / "ml_modules", so we need this symlink
        # JP: パイプラインはBASE / \"ml_modules\"を参照するため、このリンクが必要
        ml_modules_alias = Path(self.output_folder) / "ml_modules"
        if not ml_modules_alias.exists():
            try:
                # ES: En Windows, intentar crear symlink (puede requerir privilegios)
                # EN: On Windows, try to create a symlink (may require privileges)
                # JP: Windowsではシンボリックリンク作成を試す（権限が必要な場合あり）
                if hasattr(os, 'symlink'):
                    os.symlink("99_ml_modules", ml_modules_alias, target_is_directory=True)
                    print(f"✅ シンボリックリンクを作成しました: {ml_modules_alias} -> 99_ml_modules")
                else:
                    # ES: Si no hay symlink, copiar solo config_cls.py a ml_modules también
                    # EN: If symlinks are not available, also copy only config_cls.py into ml_modules
                    # JP: シンボリックリンク不可なら、ml_modulesにもconfig_cls.pyだけコピーする
                    ml_modules_fallback = Path(self.output_folder) / "ml_modules"
                    ml_modules_fallback.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(config_file, ml_modules_fallback / "config_cls.py")
                    print(f"✅ 互換性のため config_cls.py も ml_modules にコピーしました")
            except Exception as e:
                # ES: Si falla el symlink, copiar solo config_cls.py
                # EN: If creating the symlink fails, copy only config_cls.py
                # JP: リンク作成に失敗した場合はconfig_cls.pyのみコピーする
                ml_modules_fallback = Path(self.output_folder) / "ml_modules"
                ml_modules_fallback.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(config_file, ml_modules_fallback / "config_cls.py")
                print(f"⚠️ シンボリックリンクを作成できません。config_cls.py を ml_modules にコピーします: {e}")
        
        # ES: Leer el archivo config_cls.py original como plantilla
        # EN: Read the original config_cls.py as a template
        # JP: 元のconfig_cls.pyをテンプレートとして読み込む
        config_cls_path = self._find_config_cls()
        config_content = ""
        
        if config_cls_path and os.path.exists(config_cls_path):
            with open(config_cls_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
        
        # ES: Si no se encuentra, crear uno básico
        # EN: If it's not found, create a basic one
        # JP: 見つからない場合は基本版を作成する
        if not config_content:
            config_content = self._get_default_config_content()
        
        # ES: Modificar los valores según config_values
        # EN: Modify values according to config_values
        # JP: config_valuesに従って値を変更する
        modified_content = self._modify_config_content(config_content, self.config_values)
        
        # ES: Escribir archivo temporal
        # EN: Write temporary file
        # JP: 一時ファイルを書き込む
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"✅ 一時設定ファイルを作成しました: {config_file}")
        return str(config_file)
    
    def _find_config_cls(self):
        """Busca el archivo config_cls.py"""
        potential_paths = [
            Path(__file__).parent / "ml_modules" / "config_cls.py",
            Path(__file__).parent.parent / "ml_modules" / "config_cls.py",
            Path.cwd() / "ml_modules" / "config_cls.py",
        ]
        
        for path in potential_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def _get_default_config_content(self):
        """Retorna contenido por defecto de config_cls.py"""
        return '''from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Literal, Union, Set
import numpy as np

class ConfigCLS:
    """ES: Configuración temporal para clasificación
    EN: Temporary configuration for classification
    JA: 分類用の一時設定"""
    pass
'''
    
    def _modify_config_content(self, content, config_values):
        """ES: Modifica el contenido de config_cls.py según config_values
        EN: Modify config_cls.py content according to config_values
        JA: config_values に従い config_cls.py の内容を変更"""
        # ES: Esta función modifica los valores en el contenido del archivo
        # EN: This function modifies values in the file content
        # JP: この関数はファイル内容内の値を変更する
        # ES: Por simplicidad, crearemos un archivo que sobrescriba los valores
        # EN: For simplicity, we generate content that overwrites the values
        # JP: 単純化のため、値を上書きする内容を生成する
        
        modifications = []
        
        # Características
        if 'ALLOWED_FEATURES' in config_values:
            features_str = ', '.join([f"'{f}'" for f in sorted(config_values['ALLOWED_FEATURES'])])
            modifications.append(f"    ALLOWED_FEATURES: Set[str] = set([{features_str}])")
        
        if 'MUST_KEEP_FEATURES' in config_values:
            features_str = ', '.join([f"'{f}'" for f in sorted(config_values['MUST_KEEP_FEATURES'])])
            modifications.append(f"    MUST_KEEP_FEATURES: Set[str] = set([{features_str}])")
        
        if 'CONTINUOUS_FEATURES' in config_values:
            features_str = ', '.join([f"'{f}'" for f in config_values['CONTINUOUS_FEATURES']])
            modifications.append(f"    CONTINUOUS_FEATURES = [{features_str}]")
        
        if 'DISCRETE_FEATURES' in config_values:
            features_str = ', '.join([f"'{f}'" for f in config_values['DISCRETE_FEATURES']])
            modifications.append(f"    DISCRETE_FEATURES = [{features_str}]")
        
        if 'BINARY_FEATURES' in config_values:
            features_str = ', '.join([f"'{f}'" for f in config_values['BINARY_FEATURES']])
            modifications.append(f"    BINARY_FEATURES = [{features_str}]")
        
        if 'INTEGER_FEATURES' in config_values:
            features_str = ', '.join([f"'{f}'" for f in config_values['INTEGER_FEATURES']])
            modifications.append(f"    INTEGER_FEATURES = [{features_str}]")
        
        # Modelos
        if 'MODELS_TO_USE' in config_values:
            models_str = ', '.join([f'"{m}"' for m in config_values['MODELS_TO_USE']])
            modifications.append(f"    MODELS_TO_USE: List[str] = [{models_str}]")
        
        if 'COMPARE_MODELS' in config_values:
            modifications.append(f"    COMPARE_MODELS: bool = {config_values['COMPARE_MODELS']}")
        
        if 'MODEL_COMPARISON_CV_SPLITS' in config_values:
            modifications.append(f"    MODEL_COMPARISON_CV_SPLITS: int = {config_values['MODEL_COMPARISON_CV_SPLITS']}")
        
        if 'MODEL_COMPARISON_SCORING' in config_values:
            modifications.append(f"    MODEL_COMPARISON_SCORING: str = '{config_values['MODEL_COMPARISON_SCORING']}'")
        
        # Optimización multiobjetivo
        if 'N_TRIALS_MULTI_OBJECTIVE' in config_values:
            modifications.append(f"    N_TRIALS_MULTI_OBJECTIVE: int = {config_values['N_TRIALS_MULTI_OBJECTIVE']}")
        
        if 'FP_WEIGHT' in config_values:
            modifications.append(f"    FP_WEIGHT: float = {config_values['FP_WEIGHT']}")
        
        if 'COVERAGE_WEIGHT' in config_values:
            modifications.append(f"    COVERAGE_WEIGHT: float = {config_values['COVERAGE_WEIGHT']}")
        
        if 'AUC_WEIGHT' in config_values:
            modifications.append(f"    AUC_WEIGHT: float = {config_values['AUC_WEIGHT']}")
        
        if 'NP_ALPHA_RANGE' in config_values:
            min_val, max_val = config_values['NP_ALPHA_RANGE']
            modifications.append(f"    NP_ALPHA_RANGE: Tuple[float, float] = ({min_val}, {max_val})")
        
        # DCV
        if 'OUTER_SPLITS' in config_values:
            modifications.append(f"    OUTER_SPLITS: int = {config_values['OUTER_SPLITS']}")
        
        if 'INNER_SPLITS' in config_values:
            modifications.append(f"    INNER_SPLITS: int = {config_values['INNER_SPLITS']}")
        
        if 'RANDOM_STATE' in config_values:
            modifications.append(f"    RANDOM_STATE: int = {config_values['RANDOM_STATE']}")
        
        if 'N_TRIALS_INNER' in config_values:
            modifications.append(f"    N_TRIALS_INNER: int = {config_values['N_TRIALS_INNER']}")
        
        if 'USE_INNER_NOISE' in config_values:
            modifications.append(f"    USE_INNER_NOISE: bool = {config_values['USE_INNER_NOISE']}")
        
        if 'NOISE_PPM' in config_values:
            modifications.append(f"    NOISE_PPM: int = {config_values['NOISE_PPM']}")
        
        if 'NOISE_RATIO' in config_values:
            modifications.append(f"    NOISE_RATIO: float = {config_values['NOISE_RATIO']}")
        
        # Umbrales
        if 'NP_ALPHA' in config_values:
            modifications.append(f"    NP_ALPHA: float = {config_values['NP_ALPHA']}")
        
        if 'USE_UPPER_CI_ADJUST' in config_values:
            modifications.append(f"    USE_UPPER_CI_ADJUST: bool = {config_values['USE_UPPER_CI_ADJUST']}")
        
        if 'CI_METHOD' in config_values:
            modifications.append(f"    CI_METHOD: Literal['wilson', 'normal', 'jeffreys'] = '{config_values['CI_METHOD']}'")
        
        if 'CI_CONFIDENCE' in config_values:
            modifications.append(f"    CI_CONFIDENCE: float = {config_values['CI_CONFIDENCE']}")
        
        if 'TAU_NEG_FALLBACK_RATIO' in config_values:
            modifications.append(f"    TAU_NEG_FALLBACK_RATIO: float = {config_values['TAU_NEG_FALLBACK_RATIO']}")
        
        # Evaluación
        if 'FINAL_EVALUATION_CV_SPLITS' in config_values:
            modifications.append(f"    FINAL_EVALUATION_CV_SPLITS: int = {config_values['FINAL_EVALUATION_CV_SPLITS']}")
        
        if 'FINAL_EVALUATION_SHUFFLE' in config_values:
            modifications.append(f"    FINAL_EVALUATION_SHUFFLE: bool = {config_values['FINAL_EVALUATION_SHUFFLE']}")
        
        if 'FINAL_EVALUATION_RANDOM_STATE' in config_values:
            modifications.append(f"    FINAL_EVALUATION_RANDOM_STATE: int = {config_values['FINAL_EVALUATION_RANDOM_STATE']}")
        
        if 'HOLDOUT_TEST_SIZE' in config_values:
            modifications.append(f"    HOLDOUT_TEST_SIZE: float = {config_values['HOLDOUT_TEST_SIZE']}")
        
        if 'HOLDOUT_STRATIFY' in config_values:
            modifications.append(f"    HOLDOUT_STRATIFY: bool = {config_values['HOLDOUT_STRATIFY']}")
        
        if 'HOLDOUT_RANDOM_STATE' in config_values:
            modifications.append(f"    HOLDOUT_RANDOM_STATE: int = {config_values['HOLDOUT_RANDOM_STATE']}")
        
        if 'GRAY_ZONE_MIN_WIDTH' in config_values:
            modifications.append(f"    GRAY_ZONE_MIN_WIDTH: float = {config_values['GRAY_ZONE_MIN_WIDTH']}")
        
        if 'GRAY_ZONE_MAX_WIDTH' in config_values:
            modifications.append(f"    GRAY_ZONE_MAX_WIDTH: float = {config_values['GRAY_ZONE_MAX_WIDTH']}")
        
        # Actualizar rutas de salida (relativas al directorio de trabajo)
        # El pipeline espera DATA_FOLDER = "00_データセット" (carpeta que creamos)
        # ES: Usar el nombre del archivo con fecha actual
        # EN: Use a filename with the current date
        # JP: 現在日付を含むファイル名を使用する
        input_filename = getattr(self, 'input_filename', None)
        if not input_filename:
            from datetime import datetime
            date_str = datetime.now().strftime("%Y%m%d")
            input_filename = f"{date_str}_総実験データ.xlsx"
        
        modifications.append(f'    DATA_FOLDER: str = "00_データセット"')
        modifications.append(f'    INPUT_FILE: str = "{input_filename}"')
        modifications.append(f'    PREDICT_INPUT_FILE: str = "Prediction_input.xlsx"')
        # ES: Cambiar PARENT_FOLDER_TEMPLATE a "." para que no cree carpeta intermedia
        # EN: Set PARENT_FOLDER_TEMPLATE to \".\" so it does not create an intermediate folder
        # JP: 中間フォルダを作らないようPARENT_FOLDER_TEMPLATEを\".\"にする
        modifications.append(f'    PARENT_FOLDER_TEMPLATE: str = "."')
        
        # ES: Crear contenido final
        # EN: Build final content
        # JP: 最終内容を生成
        # ES: Reemplazar valores existentes en lugar de solo agregar
        # EN: Replace existing values instead of only appending
        # JP: 追記だけでなく既存値を置換する
        final_content = content
        
        # Reemplazar DATA_FOLDER si existe
        import re
        # ES: Buscar y reemplazar DATA_FOLDER
        # EN: Find and replace DATA_FOLDER
        # JP: DATA_FOLDERを検索して置換
        final_content = re.sub(
            r'(\s+DATA_FOLDER:\s*str\s*=\s*)"[^"]*"',
            r'\1"00_データセット"',
            final_content
        )
        
        # ES: Reemplazar INPUT_FILE si existe (usar el nombre del archivo con fecha actual)
        # EN: Replace INPUT_FILE if it exists (use the current-date filename)
        # JP: INPUT_FILEがあれば置換（現在日付のファイル名を使用）
        input_filename = getattr(self, 'input_filename', None)
        if not input_filename:
            from datetime import datetime
            date_str = datetime.now().strftime("%Y%m%d")
            input_filename = f"{date_str}_総実験データ.xlsx"
        
        final_content = re.sub(
            r'(\s+INPUT_FILE:\s*str\s*=\s*)"[^"]*"',
            f'\\1"{input_filename}"',
            final_content
        )
        
        # Reemplazar PREDICT_INPUT_FILE si existe
        final_content = re.sub(
            r'(\s+PREDICT_INPUT_FILE:\s*str\s*=\s*)"[^"]*"',
            r'\1"Prediction_input.xlsx"',
            final_content
        )
        
        # ES: Reemplazar PARENT_FOLDER_TEMPLATE para que no cree carpeta intermedia
        # EN: Replace PARENT_FOLDER_TEMPLATE so it does not create an intermediate folder
        # JP: 中間フォルダを作らないようPARENT_FOLDER_TEMPLATEを置換
        final_content = re.sub(
            r'(\s+PARENT_FOLDER_TEMPLATE:\s*str\s*=\s*)"[^"]*"',
            r'\1"."',
            final_content
        )
        
        # ES: Reemplazar PARENT_FOLDER_TEMPLATE para que no cree carpeta intermedia
        # EN: Replace PARENT_FOLDER_TEMPLATE so it does not create an intermediate folder
        # JP: 中間フォルダを作らないようPARENT_FOLDER_TEMPLATEを置換
        final_content = re.sub(
            r'(\s+PARENT_FOLDER_TEMPLATE:\s*str\s*=\s*)"[^"]*"',
            r'\1"."',
            final_content
        )
        
        # ES: Agregar modificaciones al final de la clase
        # EN: Append modifications at the end of the class
        # JP: クラス末尾に変更を追加
        if "class ConfigCLS:" in final_content:
            # Insertar modificaciones antes del último método o al final de la clase
            # ES: Buscar el último @classmethod o método y agregar antes
            # EN: Find the last @classmethod or method and insert before it
            # JP: 最後の@classmethod/メソッドを探し、その前に挿入する
            lines = final_content.split('\n')
            insert_pos = len(lines)
            
            # ES: Buscar el final de la clase (última línea antes de una línea vacía o fuera de la clase)
            # EN: Find the end of the class (last line before a blank line or leaving the class)
            # JP: クラス終端を探す（空行/クラス外に出る直前の最終行）
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().startswith('@classmethod') or lines[i].strip().startswith('def '):
                    # Encontrar el final de este método
                    j = i + 1
                    indent_level = len(lines[i]) - len(lines[i].lstrip())
                    while j < len(lines):
                        if lines[j].strip() and not lines[j].startswith(' ' * (indent_level + 1)) and not lines[j].startswith('\t'):
                            if not lines[j].strip().startswith('#'):
                                insert_pos = j
                                break
                        j += 1
                    break
            
            # Insertar modificaciones
            modifications_text = "\n    # === Modificaciones temporales ===\n"
            for mod in modifications:
                modifications_text += "    " + mod + "\n"
            
            lines.insert(insert_pos, modifications_text)
            final_content = '\n'.join(lines)
        else:
            # ES: Si no hay clase, crear una básica
            # EN: If there is no class, create a basic one
            # JP: クラスが無い場合は基本クラスを作成する
            final_content += "\n\n# === Modificaciones temporales ===\n"
            for mod in modifications:
                final_content += mod + "\n"
        
        return final_content
    
    def _create_prediction_input_file(self, data_folder):
        """
        Crea el archivo Prediction_input.xlsx basado en el archivo 未実験データ.xlsx del proyecto
        Agrega las columnas A11, A21, A32 según la selección del usuario
        Si no encuentra el archivo, pide al usuario que lo seleccione manualmente
        """
        try:
            # Buscar archivo 未実験データ.xlsx en la carpeta del proyecto
            project_path = Path(self.project_folder)
            
            # Buscar archivo con patrón *_未実験データ.xlsx
            unexperimented_files = list(project_path.glob("*_未実験データ.xlsx"))
            
            unexperimented_file = None
            
            if not unexperimented_files:
                # ES: No se encontró el archivo; pedir al usuario que lo seleccione
                # EN: File not found; ask the user to select it
                # JP: ファイルが見つからないため、ユーザーに選択してもらう
                self.console_output.emit(f"⚠️ *_未実験データ.xlsx が見つかりません: {project_path}")
                self.status_updated.emit("ファイル選択待ち...")
                
                # Resetear variables de selección
                self._selected_file_path = None
                self._file_selection_event.clear()
                
                # Emitir señal para que la GUI muestre el diálogo
                self.file_selection_requested.emit(str(project_path))
                
                # ES: Esperar a que el usuario seleccione el archivo (máximo 5 minutos)
                # EN: Wait for the user to select a file (max 5 minutes)
                # JP: ユーザーのファイル選択を待つ（最大5分）
                max_wait = 300  # 5 minutos en segundos
                if self._file_selection_event.wait(timeout=max_wait):
                    # ES: El usuario seleccionó un archivo
                    # EN: User selected a file
                    # JP: ユーザーがファイルを選択した
                    if self._selected_file_path:
                        unexperimented_file = Path(self._selected_file_path)
                        print(f"📋 ユーザーが選択したファイル: {unexperimented_file}")
                    else:
                        self.error.emit("❌ ファイルが選択されませんでした。")
                        return None
                else:
                    # ES: Timeout: el usuario no seleccionó el archivo a tiempo
                    # EN: Timeout: user did not select a file in time
                    # JP: タイムアウト: ユーザーが時間内にファイルを選択しなかった
                    self.error.emit("❌ ファイル選択がタイムアウトしました。")
                    return None
            else:
                # ES: Usar el primer archivo encontrado
                # EN: Use the first found file
                # JP: 見つかった最初のファイルを使用する
                unexperimented_file = unexperimented_files[0]
                print(f"📋 未実験データ ファイルを見つけました: {unexperimented_file}")
            
            # ES: Leer el archivo
            # EN: Read the file
            # JP: ファイルを読み込む
            self.status_updated.emit("ファイル読み込み中...")
            df_predict = pd.read_excel(unexperimented_file)
            
            # ES: Validar que el archivo tiene las columnas necesarias
            # EN: Validate that the file contains the required columns
            # JP: 必要な列があるか検証する
            required_columns = ['回転速度', '送り速度', 'UPカット', '切込量', '突出量', '載せ率', 'パス数']
            missing_columns = [col for col in required_columns if col not in df_predict.columns]
            
            if missing_columns:
                error_msg = (
                    f"❌ 選択されたファイルに必要な列がありません:\n\n"
                    f"不足している列: {', '.join(missing_columns)}\n\n"
                    f"必要な列: {', '.join(required_columns)}\n\n"
                    f"ファイル: {unexperimented_file}"
                )
                self.error.emit(error_msg)
                return None
            
            # ES: Validar que el archivo tiene al menos una fila de datos
            # EN: Validate that the file has at least one data row
            # JP: 少なくとも1行のデータがあるか検証する
            if len(df_predict) == 0:
                self.error.emit(f"❌ 選択されたファイルにデータがありません: {unexperimented_file}")
                return None
            
            print(f"✅ ファイルを検証しました。列: {list(df_predict.columns)}")
            print(f"✅ 行数: {len(df_predict)}")
            
            # ES: Agregar columnas A13, A11, A21, A32
            # EN: Add columns A13, A11, A21, A32
            # JP: A13/A11/A21/A32列を追加
            # La columna seleccionada será 1, las otras 0
            # A13 debe estar en la primera posición (columna A)
            df_predict['A13'] = 0
            df_predict['A11'] = 0
            df_predict['A21'] = 0
            df_predict['A32'] = 0
            
            # Establecer la columna seleccionada en 1
            if self.selected_brush == "A13":
                df_predict['A13'] = 1
            elif self.selected_brush == "A11":
                df_predict['A11'] = 1
            elif self.selected_brush == "A21":
                df_predict['A21'] = 1
            elif self.selected_brush == "A32":
                df_predict['A32'] = 1
            
            # Agregar columnas 材料 y 線材長 con los valores seleccionados
            df_predict['材料'] = self.selected_material
            df_predict['線材長'] = self.selected_wire_length
            
            # Reordenar columnas para que A13 esté primero (columna A)
            # Obtener todas las columnas
            all_columns = list(df_predict.columns)
            # Remover A13, A11, A21, A32, 材料, 線材長 de la lista
            brush_columns = ['A13', 'A11', 'A21', 'A32']
            param_columns = ['材料', '線材長']
            other_columns = [col for col in all_columns if col not in brush_columns + param_columns]
            # Crear nuevo orden: A13 primero, luego A11, A21, A32, luego 材料, 線材長, luego el resto
            new_column_order = brush_columns + param_columns + other_columns
            # Reordenar DataFrame
            df_predict = df_predict[new_column_order]
            
            # Guardar como Prediction_input.xlsx en 00_データセット
            output_file = os.path.join(data_folder, "Prediction_input.xlsx")
            df_predict.to_excel(output_file, index=False)
            
            return output_file
            
        except Exception as e:
            self.console_output.emit(f"❌ Prediction_input.xlsx 作成中にエラー: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _find_pipeline_script(self):
        """Busca el archivo Run_pipeline_ver3.3_20250914.py en .venv"""
        script_dir = Path(__file__).parent.absolute()
        potential_paths = [
            script_dir / "Run_pipeline_ver3.3_20250914.py",
            script_dir.parent / "Run_pipeline_ver3.3_20250914.py",
        ]
        
        for path in potential_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def _run_pipeline(self, script_path, working_dir, config_file):
        """ES: Ejecuta el pipeline de clasificación
        EN: Run the classification pipeline
        JA: 分類パイプラインを実行"""
        try:
            # Configurar variables de entorno
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"
            env["MPLBACKEND"] = "Agg"
            env["QT_QPA_PLATFORM"] = "offscreen"
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            
            # ES: Buscar ml_modules en .venv (no copiado)
            # EN: Locate ml_modules in .venv (not copied)
            # JP: .venv内のml_modulesを探す（コピーしない）
            script_dir = Path(__file__).parent.absolute()
            venv_ml_modules = script_dir / "ml_modules"
            if not venv_ml_modules.exists() or not (venv_ml_modules / "models_cls.py").exists():
                venv_ml_modules = script_dir.parent / "ml_modules"
            
            if venv_ml_modules.exists():
                env["ML_MODULES_PATH"] = str(venv_ml_modules)
            
            # Configurar PYTHONPATH
            # El pipeline se ejecuta desde working_dir
            # BASE = Path("./") en el script se refiere al directorio de trabajo actual
            python_paths = []
            
            # 1. 99_ml_modules dentro de working_dir (PRIMERO) - donde está config_cls.py modificado
            # El pipeline busca config_cls desde ml_modules, así que esto debe tener prioridad
            ml_modules_in_workdir = Path(working_dir) / "99_ml_modules"
            if ml_modules_in_workdir.exists():
                python_paths.append(str(ml_modules_in_workdir))
            
            # ES: También crear un symlink o alias ml_modules -> 99_ml_modules para compatibilidad
            # EN: Also create a symlink/alias ml_modules -> 99_ml_modules for compatibility
            # JP: 互換性のためml_modules -> 99_ml_modulesのリンク/エイリアスも作成する
            # ES: En Windows, creamos un symlink si es posible; si no, al menos usamos 99_ml_modules en PYTHONPATH
            # EN: On Windows, create a symlink if possible; otherwise at least use 99_ml_modules in PYTHONPATH
            # JP: Windowsでは可能ならシンボリックリンク、不可なら99_ml_modulesをPYTHONPATHに入れる
            ml_modules_alias = Path(working_dir) / "ml_modules"
            if not ml_modules_alias.exists() and ml_modules_in_workdir.exists():
                try:
                    # ES: Intentar crear symlink (requiere permisos en Windows)
                    # EN: Try to create a symlink (requires permissions on Windows)
                    # JP: シンボリックリンク作成を試す（Windowsでは権限が必要）
                    if hasattr(os, 'symlink'):
                        os.symlink(ml_modules_in_workdir, ml_modules_alias, target_is_directory=True)
                        python_paths.append(str(ml_modules_alias))
                except:
                    # ES: Si falla, al menos agregar 99_ml_modules al path
                    # EN: If it fails, at least add 99_ml_modules to the path
                    # JP: 失敗した場合は最低限99_ml_modulesをパスに追加する
                    pass
            
            # 2. working_dir - directorio de trabajo actual
            python_paths.append(str(working_dir))
            
            # 3. ml_modules del .venv (para que encuentre models_cls.py, etc.)
            if venv_ml_modules.exists():
                python_paths.append(str(venv_ml_modules))
            
            # 4. Directorio donde está el script del pipeline
            script_dir = Path(script_path).parent
            if script_dir.exists():
                python_paths.append(str(script_dir))
            
            # ES: 6. Agregar site-packages
            # EN: 6. Add site-packages
            # JP: 6. site-packages を追加
            import site
            for site_pkg in site.getsitepackages():
                if os.path.exists(site_pkg):
                    python_paths.append(site_pkg)
            
            # ES: 7. Agregar PYTHONPATH existente si hay
            # EN: 7. Add existing PYTHONPATH if present
            # JP: 7. 既存のPYTHONPATHがあれば追加
            existing_pythonpath = env.get("PYTHONPATH", "")
            if existing_pythonpath:
                python_paths.append(existing_pythonpath)
            
            # Eliminar duplicados manteniendo el orden
            seen = set()
            unique_paths = []
            for path in python_paths:
                if path not in seen:
                    seen.add(path)
                    unique_paths.append(path)
            
            env["PYTHONPATH"] = os.pathsep.join(unique_paths)
            
            # Ejecutar script
            self.console_output.emit(f"🔧 Ejecutando: {script_path}")
            self.console_output.emit(f"📁 Directorio de trabajo: {working_dir}")
            self.console_output.emit(f"📁 PYTHONPATH: {env['PYTHONPATH']}")
            
            # ES: Verificar que config_cls.py existe en 99_ml_modules dentro de working_dir
            # EN: Verify that config_cls.py exists under 99_ml_modules in working_dir
            # JP: working_dir内の99_ml_modulesにconfig_cls.pyがあるか確認
            ml_modules_in_workdir = Path(working_dir) / "99_ml_modules"
            config_check = ml_modules_in_workdir / "config_cls.py"
            if not config_check.exists():
                self.console_output.emit(f"❌ エラー: config_cls.py が見つかりません: {ml_modules_in_workdir}")
                return False
            
            # ES: Verificar que ml_modules del .venv existe
            # EN: Verify that the .venv ml_modules exists
            # JP: .venv側のml_modulesが存在するか確認
            if not venv_ml_modules.exists() or not (venv_ml_modules / "models_cls.py").exists():
                self.console_output.emit(f"❌ エラー: ml_modules が見つかりません: {venv_ml_modules}")
                return False
            
            # Usar el script original del .venv (no copiado)
            script_to_run = script_path
            self.console_output.emit(f"📝 Usando script del .venv: {script_to_run}")
            
            # ES: Debug: Verificar estructura antes de ejecutar
            # EN: Debug: Check structure before running
            # JP: Debug: 実行前に構造を確認
            self.console_output.emit(f"📋 Verificando estructura en {working_dir}:")
            workdir_path = Path(working_dir)
            if workdir_path.exists():
                try:
                    for item in workdir_path.iterdir():
                        if item.is_dir():
                            self.console_output.emit(f"  📁 {item.name}/")
                            if item.name == "ml_modules":
                                try:
                                    for subitem in item.iterdir():
                                        if subitem.is_file():
                                            self.console_output.emit(f"    📄 {subitem.name}")
                                except:
                                    pass
                        elif item.is_file():
                            self.console_output.emit(f"  📄 {item.name}")
                except Exception as e:
                    self.console_output.emit(f"⚠️ Error verificando estructura: {e}")
            
            # Asegurar que working_dir es un string para subprocess
            working_dir_str = str(working_dir) if isinstance(working_dir, Path) else working_dir
            
            # ES: Ejecutar script con Popen para poder leer salida en tiempo real
            # EN: Run the script with Popen so we can read output in real time
            # JP: リアルタイムで出力を読むためPopenでスクリプトを実行
            # IMPORTANTE: cwd debe ser working_dir para que BASE = Path("./") funcione
            process = subprocess.Popen(
                [sys.executable, script_to_run],
                cwd=working_dir_str,  # Ejecutar desde working_dir para que BASE = Path("./") funcione
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1  # Line buffered
            )
            
            # ES: Guardar referencia al proceso para poder cancelarlo
            # EN: Store a reference to the process so we can cancel it
            # JP: キャンセルできるようプロセス参照を保持
            self._current_process = process
            
            # Event para detener los threads de lectura de forma segura
            stop_reading = threading.Event()
            self._stop_reading = stop_reading
            
            # ES: Leer stdout y stderr en tiempo real usando threads
            # EN: Read stdout and stderr in real time using threads
            # JP: スレッドでstdout/stderrをリアルタイム読取
            def read_output(pipe, is_stderr=False):
                try:
                    while not stop_reading.is_set():
                        line = pipe.readline()
                        if not line:  # EOF o pipe cerrado
                            break
                        # Emitir directamente a consola
                        line_clean = line.rstrip('\n\r')
                        if line_clean:
                            prefix = "[STDERR] " if is_stderr else ""
                            self.console_output.emit(f"{prefix}{line_clean}")
                            # Parsear progreso para extraer información detallada
                            self._parse_progress(line_clean)
                except (ValueError, OSError):
                    # Pipe ya cerrado — salir silenciosamente
                    pass
                except Exception:
                    # Cualquier otro error — ignorar
                    pass
            
            stdout_thread = threading.Thread(target=read_output, args=(process.stdout, False), daemon=True)
            stderr_thread = threading.Thread(target=read_output, args=(process.stderr, True), daemon=True)
            stdout_thread.start()
            stderr_thread.start()
            
            # Esperar a que termine el proceso o sea cancelado
            # Usar polling para poder cancelar
            while process.poll() is None:
                if self._cancelled:
                    print("🛑 処理をキャンセル中...")
                    self.console_output.emit("🛑 処理をキャンセル中...")
                    try:
                        process.terminate()  # Intentar terminar suavemente
                        # Esperar un poco para que termine (polling)
                        for _ in range(20):  # Esperar hasta 2 segundos (20 * 0.1)
                            if process.poll() is not None:
                                break
                            time.sleep(0.1)
                        
                        # Si aún no terminó, forzar kill
                        if process.poll() is None:
                            print("⚠️ プロセスが終了しません。強制終了します...")
                            self.console_output.emit("⚠️ プロセスが終了しません。強制終了します...")
                            process.kill()
                            process.wait()
                    except Exception as e:
                        print(f"⚠️ キャンセル処理中にエラー: {e}")
                        try:
                            if process.poll() is None:
                                process.kill()
                                process.wait()
                        except:
                            pass
                    return False
                time.sleep(0.1)  # Wait a bit before checking again
            
            returncode = process.returncode
            
            # ES: Detener los threads de lectura antes de cerrar pipes
            # EN: Stop reader threads before closing pipes
            # JP: パイプを閉じる前に読取スレッドを停止
            stop_reading.set()
            stdout_thread.join(timeout=1.0)  # Esperar máximo 1 segundo
            stderr_thread.join(timeout=1.0)  # Esperar máximo 1 segundo
            
            # ES: Limpiar referencia al proceso
            # EN: Clear process references
            # JP: プロセス参照をクリア
            self._current_process = None
            self._stop_reading = None
            
            # ES: Cerrar pipes de forma segura (ya no hay threads leyendo)
            # EN: Close pipes safely (no threads are reading anymore)
            # JP: パイプを安全に閉じる（読取スレッドは停止済み）
            try:
                if process.stdout:
                    process.stdout.close()
                if process.stderr:
                    process.stderr.close()
            except:
                pass
            
            if returncode == 0:
                self.console_output.emit(f"✅ Pipeline ejecutado exitosamente")
                return True
            else:
                self.console_output.emit(f"❌ Pipeline falló con código {returncode}")
                # ES: Intentar leer cualquier salida restante de stderr para ver el error
                # EN: Try to read any remaining stderr output to see the error
                # JP: エラー確認のためstderrの残り出力を読んでみる
                try:
                    if process.stderr:
                        remaining_stderr = process.stderr.read()
                        if remaining_stderr:
                            for line in remaining_stderr.split('\n'):
                                line_clean = line.rstrip('\n\r')
                                if line_clean:
                                    self.console_output.emit(f"[STDERR] {line_clean}")
                except:
                    pass
                return False
                
        except Exception as e:
            import traceback
            error_msg = f"❌ Error ejecutando pipeline: {str(e)}\n{traceback.format_exc()}"
            self.console_output.emit(error_msg)
            return False
    
    def _parse_progress(self, line):
        """
        Parsea el output del pipeline para extraer información de progreso
        y actualizar la barra de progreso con información detallada
        """
        try:
            # ES: Detectar modelo comparación
            # EN: Detect model comparison
            # JP: モデル比較を検出
            if 'モデル比較評価' in line or 'モデル比較' in line:
                self.current_task = 'model_comparison'
                self.progress_updated.emit(5, "モデル比較中...")
                return
            
            if '選択されたモデル' in line or '最適モデル' in line:
                self.model_comparison_completed = True
                self.progress_updated.emit(10, "モデル比較完了")
                return
            
            # Detectar multi-objective optimization
            if '[Step 1]' in line and '多目的最適化' in line:
                self.current_task = 'multiobjective'
                self.progress_updated.emit(15, "多目的最適化中...")
                return
            
            if '最適α値発見' in line or '多目的最適化' in line and '完了' in line:
                self.multiobjective_completed = True
                self.progress_updated.emit(20, "多目的最適化完了")
                return
            
            # Detectar DCV学習開始
            if '[Step 2]' in line and '本学習' in line:
                self.current_task = 'dcv'
                self.dcv_training = True
                self.progress_updated.emit(25, "DCV学習開始...")
                return
            
            # Detectar Outer Fold (patrón: "--- Outer Fold X/Y ---" o similar)
            fold_match = re.search(r'Outer\s+Fold\s+(\d+)/(\d+)', line, re.IGNORECASE)
            if not fold_match:
                fold_match = re.search(r'外側.*?(\d+)/(\d+)', line)
            if fold_match:
                self.current_fold = int(fold_match.group(1))
                self.total_folds = int(fold_match.group(2))
                # Calcular progreso: 25% (inicio DCV) + (fold/total_folds) * 50% (DCV)
                progress = 25 + int((self.current_fold / self.total_folds) * 50)
                self.progress_updated.emit(progress, f"DCV学習中... Fold {self.current_fold}/{self.total_folds}")
                return
            
            # Detectar Inner Fold
            inner_fold_match = re.search(r'Inner\s+Fold\s+(\d+)/(\d+)', line, re.IGNORECASE)
            if inner_fold_match:
                inner_fold = int(inner_fold_match.group(1))
                inner_total = int(inner_fold_match.group(2))
                # Progreso más detallado dentro del fold actual
                fold_progress = 25 + int((self.current_fold / self.total_folds) * 50)
                inner_progress = int((inner_fold / inner_total) * 5)  # 5% por fold interno
                total_progress = fold_progress + inner_progress
                self.progress_updated.emit(total_progress, f"DCV学習中... Outer {self.current_fold}/{self.total_folds}, Inner {inner_fold}/{inner_total}")
                return
            
            # Detectar Trial de Optuna
            trial_match = re.search(r'Trial\s+(\d+)', line, re.IGNORECASE)
            if trial_match:
                self.current_trial = int(trial_match.group(1))
                # Actualizar progreso basado en trial
                if self.total_trials > 0:
                    trial_progress = int((self.current_trial / self.total_trials) * 5)  # 5% por trial
                    fold_progress = 25 + int((self.current_fold / self.total_folds) * 50)
                    total_progress = min(75, fold_progress + trial_progress)
                    self.progress_updated.emit(total_progress, f"DCV学習中... Fold {self.current_fold}/{self.total_folds}, Trial {self.current_trial}/{self.total_trials}")
                return
            
            # Detectar aprendizaje completado
            if '学習完了' in line or '学習が完了' in line:
                self.dcv_training = False
                self.current_task = 'prediction'
                self.progress_updated.emit(75, "学習完了、予測準備中...")
                return
            
            # ES: Detectar predicción
            # EN: Detect prediction
            # JP: 予測を検出
            if '予測実行' in line or '予測処理開始' in line or 'predict' in line.lower():
                self.current_task = 'prediction'
                self.progress_updated.emit(80, "予測実行中...")
                return
            
            if '予測処理完了' in line or '予測完了' in line:
                self.prediction_completed = True
                self.progress_updated.emit(85, "予測完了")
                return
            
            # Detectar OOF予測分析
            if '[OOF予測分析]' in line or 'OOF予測' in line:
                self.current_task = 'evaluation'
                self.progress_updated.emit(86, "OOF予測分析中...")
                return
            
            # Detectar evaluación final
            if '[最終モデル性能評価]' in line or '固定HP評価' in line or '評価中' in line:
                self.current_task = 'evaluation'
                self.progress_updated.emit(88, "最終評価中...")
                return
            
            # ES: Detectar análisis de características
            # EN: Detect feature analysis
            # JP: 特徴量解析を検出
            if '[特徴量重要度分析]' in line or '特徴量重要度' in line:
                self.progress_updated.emit(92, "特徴量重要度分析中...")
                return
            
            # Detectar diagnóstico
            if '診断レポート' in line or 'diagnostic' in line.lower():
                self.progress_updated.emit(95, "診断レポート生成中...")
                return
            
            # Detectar finalización
            if 'すべての処理が完了しました' in line or '処理完了' in line or '完了しました' in line:
                self.evaluation_completed = True
                self.progress_updated.emit(98, "処理完了...")
                return
            
        except Exception as e:
            # ES: Si hay error en el parsing, no hacer nada (no es crítico)
            # EN: If parsing fails, do nothing (not critical)
            # JP: パースでエラーが出ても何もしない（致命的ではない）
            pass
    
    def _find_results(self):
        """Busca los resultados generados por el pipeline"""
        results = {
            'result_folders': [],
            'graph_paths': [],
            'model_files': [],
            'evaluation_files': []
        }
        
        # ES: El pipeline crea una carpeta con timestamp
        # EN: The pipeline creates a timestamped folder
        # JP: パイプラインはタイムスタンプ付きフォルダを作成する
        # ES: Buscar en el directorio de trabajo
        # EN: Search in the working directory
        # JP: 作業ディレクトリ内で検索する
        if not os.path.exists(self.output_folder):
            return results
        
        # ES: Buscar carpetas de resultados
        # EN: Search for result folders
        # JP: 結果フォルダを探す
        for item in os.listdir(self.output_folder):
            item_path = os.path.join(self.output_folder, item)
            if os.path.isdir(item_path):
                # ES: Verificar si es una carpeta de resultados del pipeline
                # EN: Check whether this is a pipeline results folder
                # JP: パイプライン結果フォルダか確認する
                if "分類解析結果" in item or "分類" in item:
                    results['result_folders'].append(item_path)
        
        # ES: Buscar archivos de gráficos
        # EN: Search for chart files
        # JP: グラフファイルを探す
        for root, dirs, files in os.walk(self.output_folder):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    results['graph_paths'].append(os.path.join(root, file))
                elif file.endswith('.pkl'):
                    results['model_files'].append(os.path.join(root, file))
                elif file.endswith(('.xlsx', '.csv', '.json')):
                    if 'evaluation' in file.lower() or '評価' in file:
                        results['evaluation_files'].append(os.path.join(root, file))
        
        return results

