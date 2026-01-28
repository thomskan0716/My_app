"""
ES: Worker para ejecutar análisis no lineal en un thread separado.
EN: Worker to run the non-linear analysis in a separate thread.
JA: 非線形解析を別スレッドで実行するワーカー。

ES: Ejecuta los scripts 01_model_builder.py, 02_prediction.py, 03_pareto_analyzer.py.
EN: Runs the scripts 01_model_builder.py, 02_prediction.py, 03_pareto_analyzer.py.
JA: 01_model_builder.py / 02_prediction.py / 03_pareto_analyzer.py を実行する。
"""
import sys
import os
import subprocess
import pandas as pd
import json
import time
import threading
import re
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from nonlinear_folder_manager import NonlinearFolderManager


class NonlinearWorker(QThread):
    """ES: Worker que ejecuta el análisis no lineal en un thread separado
    EN: Worker that runs the non-linear analysis in a separate thread
    JA: 非線形解析を別スレッドで実行するワーカー
    """
    
    # ES: Señales para comunicación con la GUI | EN: Signals for GUI communication | JA: GUI通信用シグナル
    progress_updated = Signal(int, str)  # (value, message)
    progress_detailed = Signal(int, int, int, int, int, int, str, bool, bool, bool, int, int)  # (trial_current, trial_total, fold_current, fold_total, pass_current, pass_total, current_task, data_analysis_completed, final_model_training, shap_analysis, model_current, model_total)
    status_updated = Signal(str)  # message
    finished = Signal(dict)  # results dict
    error = Signal(str)  # error message
    console_output = Signal(str)  # console output (for IDE/terminal)
    
    def __init__(self, filtered_df, project_folder, parent=None, config_values=None):
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
        """
        super().__init__(parent)
        self.filtered_df = filtered_df
        self.project_folder = project_folder
        self.config_values = config_values or {}
        self.output_folder = None
        self.current_stage = None
        self._json_reader_stop = threading.Event()  # Stop flag for the JSON reader
        self._cancelled = False  # Cancel flag
        self._current_process = None  # Current subprocess reference
        self._stop_reading = None  # Stop flag for output reading
        self._cached_script_base_dir = None  # Cache script_base_dir to avoid repeated checks
        self._cached_python_code_folder = None  # Cache python_code_folder
        
        # ES: Estado del progreso para parsing | EN: Parsing progress state | JA: パース進捗状態
        self.current_fold = 0
        self.total_folds = self.config_values.get('outer_splits', self.config_values.get('OUTER_SPLITS', 10))
        self.current_trial = 0  # Completed-trial counter in current fold (incremental: 1, 2, 3...)
        # ES: Normalizar nombre: puede venir como 'n_trials' o 'N_TRIALS' | EN: Normalize key name: it may be 'n_trials' or 'N_TRIALS' | JA: キー名を正規化（n_trials / N_TRIALS の可能性）
        self.total_trials = self.config_values.get('N_TRIALS', self.config_values.get('n_trials', 50))
        self.current_model = 0
        self.total_models = len(self.config_values.get('MODELS_TO_USE', ['random_forest', 'lightgbm']))
        self.current_pass = 0  # Current pass (current target)
        self.total_passes = len(self.config_values.get('TARGET_COLUMNS', []))  # Total passes (targets)
        # ES: Si no hay TARGET_COLUMNS en config, usar un valor por defecto (normalmente 3) | EN: If TARGET_COLUMNS is missing, use a default (usually 3) | JA: TARGET_COLUMNS が無い場合はデフォルト（通常3）を使用
        if self.total_passes == 0:
            self.total_passes = 3  # Default value
        self.last_detected_target = None  # Avoid detecting the same target twice
        
        # ES: ✅ Variables para progreso acumulado (para cálculo lineal de porcentaje) | EN: ✅ Accumulated progress variables (for linear percent computation) | JA: ✅ 累積進捗変数（割合を線形計算するため）
        self.accumulated_trial_current = 0  # Total accumulated completed trials (across passes/folds/models)
        self.accumulated_trial_total = 0  # Total accumulated trials (passes * folds * trials_per_fold * models)
        
        # ES: ✅ Set para rastrear qué trials ya fueron contados (evitar contar el mismo trial dos veces)
        # EN: ✅ Set to track which trials were already counted (avoid double-counting)
        # JA: ✅ 既にカウント済みtrialを追跡（重複カウント防止）
        self.completed_trials_in_current_fold = set()  # IDs of trials completed in the current fold
        
        # ES: Estados adicionales para tareas dentro de 01_model_builder
        # EN: Additional state for tasks inside 01_model_builder
        # JA: 01_model_builder 内タスク用の追加状態
        self.data_analysis_completed = False  # Data analysis completed
        self.current_task = 'initialization'  # Current task: initialization, data_analysis, dcv, final_model, shap, saving
        self.final_model_training = False  # Final model training
        self.shap_analysis = False  # SHAP analysis
        self.saving_completed = False  # Saving completed
        
    def run(self):
        """ES: Ejecuta el análisis no lineal
        EN: Run the non-linear analysis
        JA: 非線形解析を実行
        """
        import time
        start_time = time.time()  # Record start time
        self.analysis_start_time = start_time
        
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
                
                # ES: Buscar gráficos generados | EN: Find generated graphs | JA: 生成グラフを探索
                graph_paths = self._find_graphs(self.output_folder)
                
                # ES: Buscar carpeta de resultados para obtener subfolders | EN: Locate results folder to collect subfolders | JA: サブフォルダ取得のため結果フォルダを探索
                result_folder = os.path.join(self.output_folder, '03_学習結果')
                subfolders = {}
                if os.path.exists(result_folder):
                    subfolders['03_学習結果'] = result_folder
                
                # ES: Emitir resultados como carga existente | EN: Emit results as an existing-load run | JA: 既存読み込みとして結果を送信
                results_existing = {
                    'stage': 'completed',
                    'output_folder': self.output_folder,
                    'graph_paths': graph_paths,
                    'subfolders': subfolders,
                    'all_stages_completed': False,  # Puede que no tenga stages 2 y 3
                    'load_existing': True,
                    'existing_folder_path': selected_folder_path
                }
                
                self.progress_updated.emit(100, "既存結果読み込み完了")
                self.status_updated.emit("✅ 既存結果を読み込みました。")
                
                # ES: Emitir finished para que la GUI muestre los resultados existentes | EN: Emit finished so the GUI can show existing results | JA: GUI表示のため finished を送信
                self.finished.emit(results_existing)
                return
            
            # ES: Si no es carga existente, ejecutar análisis normal | EN: If not loading existing, run normal analysis | JA: 既存読み込みでなければ通常解析を実行
            # ES: Crear carpeta de salida | EN: Create output folder | JA: 出力フォルダ作成
            self.status_updated.emit("📁 Creando carpeta de salida...")
            
            # ES: Verificar cancelación antes de crear carpetas | EN: Check cancellation before creating folders | JA: フォルダ作成前にキャンセル確認
            if self._cancelled:
                print("🛑 フォルダー作成前に解析がキャンセルされました")
                return
            
            folder_manager = NonlinearFolderManager(self.project_folder)
            self.output_folder = folder_manager.create_output_folder()
            subfolders = folder_manager.create_subfolder_structure(self.output_folder)
            
            # ES: Verificar cancelación después de crear carpetas | EN: Check cancellation after creating folders | JA: フォルダ作成後にキャンセル確認
            if self._cancelled:
                print("🛑 フォルダー作成後に解析がキャンセルされました")
                return
            
            # ES: Guardar datos filtrados | EN: Save filtered data | JA: フィルタ済みデータを保存
            self.status_updated.emit("💾 Guardando datos filtrados...")
            data_folder = os.path.join(self.output_folder, "01_データセット")
            os.makedirs(data_folder, exist_ok=True)
            
            input_file = os.path.join(data_folder, "filtered_data.xlsx")
            df_to_save = self.filtered_df.copy()
            # ES: Mantener el comportamiento actual de guardado de filtered_data.xlsx
            # EN: Keep the current behavior for saving filtered_data.xlsx
            # JA: filtered_data.xlsx の保存挙動は現状維持
            df_to_save.to_excel(input_file, index=False)
            print(f"✅ データを保存しました: {input_file}")

            # ES: Crear un segundo archivo para el análisis del modelo: analysis_df.xlsx | EN: Create a second file for model analysis: analysis_df.xlsx | JA: モデル解析用に analysis_df.xlsx を作成
            # ES: A partir de filtered_data, eliminar columnas no deseadas como '材料' y '実験日' | EN: From filtered_data, drop unwanted columns like '材料' and '実験日' | JA: filtered_data から不要列（材料/実験日など）を削除
            analysis_df = df_to_save.copy()
            cols_to_drop = ['材料', '実験日']
            try:
                drop_cols = [c for c in cols_to_drop if c in analysis_df.columns]
                if drop_cols:
                    analysis_df = analysis_df.drop(columns=drop_cols)
                    print(f"ℹ️ analysis_df.xlsx で削除した列: {drop_cols}")
                # ES: Forzar que columnas enteras no sean int64 al leerlas en Stage 01 | EN: Force integer cols to be read as float64 in Stage 01 | JA: Stage01でint列がfloat64として読まれるように調整
                # ES: Convertir columnas int a float para que pd.read_excel las infiera como float64 | EN: Convert int columns to float so pd.read_excel infers float64 | JA: int列をfloatへ変換し pd.read_excel の推論をfloat64にする
                int_cols_analysis = analysis_df.select_dtypes(include=["int64", "int32", "int"]).columns
                if len(int_cols_analysis) > 0:
                    analysis_df[int_cols_analysis] = analysis_df[int_cols_analysis].astype("float64")
                    print(f"ℹ️ analysis_df.xlsx の整数列を float に変換しました: {list(int_cols_analysis)}")
            except Exception as e:
                print(f"⚠️ analysis_df.xlsx の列準備に失敗しました: {e}")

            analysis_file = os.path.join(data_folder, "analysis_df.xlsx")
            analysis_df.to_excel(analysis_file, index=False)
            print(f"✅ 解析用データを保存しました: {analysis_file}")
            
            # ES: Verificar cancelación después de guardar datos | EN: Check cancellation after saving data | JA: データ保存後にキャンセル確認
            if self._cancelled:
                print("🛑 データ保存後に解析がキャンセルされました")
                return
            
            # ES: Guardar configuración personalizada directamente como config.py | EN: Save custom config directly as config.py | JA: カスタム設定を config.py として保存
            # ES: (En esta carpeta solo existirá este config.py, modificado) | EN: (Only this modified config.py will exist in this folder) | JA: （このフォルダには変更済みの config.py のみ置く）
            config_file = os.path.join(self.output_folder, "config.py")
            self._save_config_file(config_file)
            
            # ES: Verificar cancelación después de guardar configuración | EN: Check cancellation after saving config | JA: 設定保存後にキャンセル確認
            if self._cancelled:
                print("🛑 設定保存後に解析がキャンセルされました")
                return
            
            # ES: Copiar scripts necesarios a la carpeta de salida | EN: Copy required scripts to the output folder | JA: 必要スクリプトを出力フォルダへコピー
            self.status_updated.emit("📋 Copiando scripts...")
            # ES: Ya no copiamos el config.py genérico; usamos el config.py generado arriba | EN: We no longer copy the generic config.py; we use the generated one above | JA: 汎用config.pyはコピーせず、上で生成したconfig.pyを使用
            scripts_to_copy = ["01_model_builder.py", "02_prediction.py", "03_pareto_analyzer.py"]
            
            # ES: ✅ Buscar scripts en el directorio donde está 0sec.py (directorio del proyecto) | EN: ✅ Locate scripts in the directory containing 0sec.py | JA: ✅ 0sec.py があるディレクトリでスクリプトを探索
            # ES: project_folder es la carpeta base, pero los scripts están en el directorio padre | EN: project_folder is the project base; scripts live in the parent directory | JA: project_folder はプロジェクト基点だがスクリプトは親ディレクトリ
            script_base_dir = None
            if self.project_folder:
                # ES: project_folder es algo como "Archivos_de_salida/Proyecto_79"
                # EN: project_folder looks like "Archivos_de_salida/Proyecto_79"
                # JA: project_folder は例："Archivos_de_salida/Proyecto_79"
                # ES: Los scripts están en el directorio padre (donde está 0sec.py)
                # EN: Scripts live in the parent directory (where 0sec.py is)
                # JA: スクリプトは親ディレクトリ（0sec.py がある場所）にある
                potential_base = Path(self.project_folder).parent.parent
                if (potential_base / "0sec.py").exists():
                    script_base_dir = potential_base
                else:
                    # ES: Intentar buscar desde el directorio actual
                    # EN: Try searching from the current directory
                    # JP: 現在のディレクトリから探す
                    current_dir = Path.cwd()
                    if (current_dir / "0sec.py").exists():
                        script_base_dir = current_dir
                    elif (current_dir / "01_model_builder.py").exists():
                        script_base_dir = current_dir
            
            if script_base_dir is None:
                script_base_dir = Path.cwd()  # Fallback al directorio actual
            
            for script in scripts_to_copy:
                # ES: Verificar cancelación durante copia de scripts
                # EN: Check cancellation during script copy
                # JP: スクリプトコピー中にキャンセルを確認
                if self._cancelled:
                    print("🛑 スクリプトコピー中に解析がキャンセルされました")
                    return
                
                script_path = script_base_dir / script
                if script_path.exists():
                    import shutil
                    dest = os.path.join(self.output_folder, script)
                    shutil.copy2(str(script_path), dest)
                    print(f"✅ スクリプトをコピーしました: {script_path} → {dest}")
                else:
                    print(f"⚠️ スクリプトが見つかりません: {script_path}")
            
            # ES: Verificar cancelación antes de ejecutar Stage 01
            # EN: Check cancellation before running Stage 01
            # JP: Stage 01実行前にキャンセルを確認
            if self._cancelled:
                print("🛑 Stage 01 実行前に解析がキャンセルされました")
                return
            
            # Ejecutar Stage 01: Model Builder
            self.current_stage = '01_model_builder'
            self.status_updated.emit("🔧 モデル構築中...")
            self.progress_updated.emit(10, "モデル構築中...")
            
            # ES: Verificar cancelación antes de ejecutar
            # EN: Check cancellation before running
            # JP: 実行前にキャンセルを確認
            if self._cancelled:
                print("🛑 Stage 01 実行前に解析がキャンセルされました")
                return
            
            success_01 = self._run_script("01_model_builder.py", self.output_folder)
            
            # Si fue cancelado, no emitir error
            if self._cancelled:
                print("🛑 Stage 01 実行中に解析がキャンセルされました")
                return
            
            if not success_01:
                self.error.emit("❌ Error en Stage 01: Model Builder")
                return
            
            # ES: Calcular tiempo total de análisis
            # EN: Compute total analysis time
            # JP: 解析の総時間を計算
            end_time = time.time()
            analysis_duration = end_time - start_time
            self.analysis_duration = analysis_duration
            
            # ES: Guardar resultados en JSON antes de mostrar la pantalla de resumen
            # EN: Save results to JSON before showing the summary screen
            # JP: サマリー画面表示前に結果をJSONに保存
            self._save_analysis_results_json()
            
            # ES: Buscar gráficos generados (para referencia, pero no se mostrarán)
            # EN: Find generated charts (for reference, but they won't be shown)
            # JP: 生成されたグラフを探す（参照用、表示はしない）
            graph_paths = self._find_graphs(self.output_folder)
            
            # Emitir resultados del Stage 01 como 'completed' para ir directamente a la pantalla de resumen
            results_01 = {
                'stage': 'completed',  # Cambiar a 'completed' para que vaya directamente a _show_final_results
                'output_folder': self.output_folder,
                'graph_paths': graph_paths,
                'subfolders': subfolders,
                'all_stages_completed': False,  # Indicar que solo se completó el stage 01
                'load_existing': False  # Not an existing-load; it's a new analysis
            }
            
            self.progress_updated.emit(100, "Stage 01 完了")
            self.status_updated.emit("✅ Stage 01 完了。結果を表示します...")
            
            # Emitir finished para que la GUI muestre directamente la pantalla de resumen
            self.finished.emit(results_01)
            
        except Exception as e:
            import traceback
            error_msg = f"❌ 非線形解析中にエラー: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.error.emit(error_msg)
    
    def run_stage2_and_3(self):
        """
        Continúa con los stages 02 y 03 después de que el usuario confirme
        Este método se llama desde la GUI cuando el usuario hace OK en el visor de gráficos
        """
        print("🔍 デバッグ run_stage2_and_3: メソッド呼び出し", flush=True)
        print(f"🔍 DEBUG run_stage2_and_3: output_folder = {self.output_folder}", flush=True)
        try:
            # ES: Verificar cancelación antes de continuar
            # EN: Check cancellation before continuing
            # JP: 続行前にキャンセルを確認
            if self._cancelled:
                print("🛑 Stage 02 実行前に解析がキャンセルされました")
                return
            
            # Ejecutar Stage 02: Prediction
            self.current_stage = '02_prediction'
            self.status_updated.emit("🔧 Ejecutando Stage 02: Prediction...")
            self.progress_updated.emit(60, "Stage 02: Prediction")
            
            success_02 = self._run_script("02_prediction.py", self.output_folder)
            print(f"🔍 DEBUG run_stage2_and_3: success_02 = {success_02}")
            
            # Si fue cancelado, no emitir error
            if self._cancelled:
                print("🛑 Stage 02 実行中に解析がキャンセルされました")
                return
            
            if not success_02:
                print("🔍 デバッグ run_stage2_and_3: Stage 02 に失敗。error を送信します")
                self.error.emit("❌ Error en Stage 02: Prediction")
                return
            
            # ES: Verificar cancelación antes de Stage 03
            # EN: Check cancellation before Stage 03
            # JP: Stage 03前にキャンセルを確認
            if self._cancelled:
                print("🛑 Stage 03 実行前に解析がキャンセルされました")
                return
            
            # Ejecutar Stage 03: Pareto Analyzer
            self.current_stage = '03_pareto_analyzer'
            self.status_updated.emit("🔧 Ejecutando Stage 03: Pareto Analyzer...")
            self.progress_updated.emit(90, "Stage 03: Pareto Analyzer")
            
            success_03 = self._run_script("03_pareto_analyzer.py", self.output_folder)
            print(f"🔍 DEBUG run_stage2_and_3: success_03 = {success_03}")
            
            # Si fue cancelado, no emitir error
            if self._cancelled:
                print("🛑 Stage 03 実行中に解析がキャンセルされました")
                return
            
            if not success_03:
                print("🔍 デバッグ run_stage2_and_3: Stage 03 に失敗。error を送信します")
                self.error.emit("❌ Error en Stage 03: Pareto Analyzer")
                return
            
            # ES: Análisis completado | EN: Analysis completed | JA: 解析完了
            self.progress_updated.emit(100, "Análisis completado")
            self.status_updated.emit("✅ Análisis no lineal completado exitosamente")
            
            # ES: Guardar datos de resultados en JSON
            # EN: Save results data to JSON
            # JP: 結果データをJSONに保存
            self._save_analysis_results_json()
            
            # ES: Buscar gráficos de Pareto
            # EN: Find Pareto charts
            # JP: パレートのグラフを探す
            pareto_plots_folder = os.path.join(self.output_folder, "05_パレート解", "pareto_plots")
            prediction_output_file = os.path.join(self.output_folder, "04_予測", "Prediction_output.xlsx")
            
            # ES: DEBUG: Verificar rutas
            # EN: DEBUG: Check paths
            # JP: DEBUG: パスを確認
            print(f"🔍 DEBUG nonlinear_worker: output_folder = {self.output_folder}", flush=True)
            print(f"🔍 DEBUG nonlinear_worker: pareto_plots_folder = {pareto_plots_folder}", flush=True)
            print(f"🔍 DEBUG nonlinear_worker: prediction_output_file = {prediction_output_file}", flush=True)
            print(f"🔍 DEBUG nonlinear_worker: pareto_plots_folder exists = {os.path.exists(pareto_plots_folder)}", flush=True)
            print(f"🔍 DEBUG nonlinear_worker: prediction_output_file exists = {os.path.exists(prediction_output_file)}", flush=True)
            
            # ES: Verificar si existen archivos en la carpeta de gráficos
            # EN: Check whether there are files in the charts folder
            # JP: グラフフォルダにファイルがあるか確認
            if os.path.exists(pareto_plots_folder):
                graph_files = [f for f in os.listdir(pareto_plots_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
                print(f"🔍 デバッグ nonlinear_worker: 検出したグラフ数 = {len(graph_files)}", flush=True)
                if graph_files:
                    print(f"🔍 デバッグ nonlinear_worker: 先頭のグラフ = {graph_files[:3]}", flush=True)
            
            results_final = {
                'stage': 'completed',
                'output_folder': self.output_folder,
                'all_stages_completed': True,
                'pareto_plots_folder': pareto_plots_folder,
                'prediction_output_file': prediction_output_file
            }
            
            print("🔍 デバッグ run_stage2_and_3: finished シグナルを送信中", flush=True)
            print(f"🔍 DEBUG run_stage2_and_3: results_final = {results_final}", flush=True)
            self.finished.emit(results_final)
            print("🔍 デバッグ run_stage2_and_3: finished シグナル送信完了", flush=True)
            
        except Exception as e:
            import traceback
            error_msg = f"❌ 解析の続行中にエラー: {str(e)}\n{traceback.format_exc()}"
            print("🔍 デバッグ run_stage2_and_3: 例外を捕捉")
            print(error_msg)
            print(f"🔍 デバッグ run_stage2_and_3: error シグナルを送信中")
            self.error.emit(error_msg)
    
    def _get_json_log_path(self, working_dir):
        """
        Obtiene la ruta del archivo JSON de log basándose en la estructura de carpetas
        
        Parameters
        ----------
        working_dir : str
            Directorio de trabajo (output_folder)
        
        Returns
        -------
        str
            Ruta completa al archivo console_output.jsonl
        """
        # El JSON se guarda en RESULT_FOLDER (03_学習結果)
        # Según config_custom.py, RESULT_FOLDER = '03_学習結果'
        result_folder = os.path.join(working_dir, '03_学習結果')
        json_path = os.path.join(result_folder, 'console_output.jsonl')
        return json_path
    
    def _read_json_log(self, json_path):
        """
        Lee el archivo JSON de log en tiempo real y emite mensajes a consola
        
        Parameters
        ----------
        json_path : str
            Ruta al archivo console_output.jsonl
        """
        last_position = 0
        max_wait_time = 300  # Max 5 minutes waiting for the file to appear
        wait_interval = 0.5  # Check every 0.5 seconds
        elapsed_time = 0
        
        # ES: Esperar a que el archivo exista
        # EN: Wait for the file to exist
        # JP: ファイルが存在するまで待つ
        while not os.path.exists(json_path) and elapsed_time < max_wait_time:
            time.sleep(wait_interval)
            elapsed_time += wait_interval
        
        if not os.path.exists(json_path):
            self.console_output.emit(f"⚠️ Archivo JSON no encontrado: {json_path}")
            return
        
        # ES: Leer el archivo en tiempo real (reabriendo cada vez para evitar problemas de bloqueo)
        # EN: Read the file in real time (reopen each time to avoid file-lock issues)
        # JP: ロック問題回避のため毎回開き直してリアルタイムで読む
        try:
            # ES: Primero, leer todo el contenido existente
            # EN: First, read all existing content
            # JP: まず既存内容をすべて読む
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            if 'message' in entry:
                                self.console_output.emit(entry['message'])
                        except json.JSONDecodeError:
                            continue
                # ES: Obtener el tamaño actual del archivo después de leerlo
                # EN: Get the current file size after reading it
                # JP: 読み込み後に現在のファイルサイズを取得
                last_position = os.path.getsize(json_path)
            
            # ES: Leer nuevas líneas mientras el proceso está corriendo
            # EN: Read new lines while the process is running
            # JP: プロセス稼働中に新しい行を読む
            while not self._json_reader_stop.is_set():
                time.sleep(0.1)  # Polling cada 100ms
                
                # ES: Verificar si el archivo creció
                # EN: Check whether the file grew
                # JP: ファイルが増えたか確認
                if os.path.exists(json_path):
                    current_size = os.path.getsize(json_path)
                    if current_size > last_position:
                        # ES: Reabrir el archivo y leer solo las nuevas líneas
                        # EN: Reopen the file and read only the new lines
                        # JP: ファイルを開き直し、新しい行だけ読む
                        with open(json_path, 'r', encoding='utf-8') as f:
                            f.seek(last_position)
                            new_content = f.read(current_size - last_position)
                            last_position = current_size
                            
                            # Procesar nuevas líneas
                            for line in new_content.split('\n'):
                                line = line.strip()
                                if line:
                                    try:
                                        entry = json.loads(line)
                                        if 'message' in entry:
                                            self.console_output.emit(entry['message'])
                                    except json.JSONDecodeError:
                                        # Si no es JSON válido, puede ser contenido parcial
                                        continue
        except Exception as e:
            self.console_output.emit(f"⚠️ JSON 読み込み中にエラー: {e}")
            import traceback
            self.console_output.emit(f"Traceback: {traceback.format_exc()}")
    
    def _run_script(self, script_name, working_dir):
        """
        Ejecuta un script Python en un subproceso y lee el JSON de log en tiempo real
        
        Parameters
        ----------
        script_name : str
            Nombre del script a ejecutar
        working_dir : str
            Directorio de trabajo
        
        Returns
        -------
        bool
            True si el script se ejecutó exitosamente, False en caso contrario
        """
        script_path = os.path.join(working_dir, script_name)
        
        # ES: Si el script no está en la carpeta de salida, usar el del directorio actual
        # EN: If the script is not in the output folder, use the one in the current directory
        # JP: 出力先に無ければ、カレントディレクトリのものを使用する
        if not os.path.exists(script_path):
            script_path = script_name
            if not os.path.exists(script_path):
                print(f"❌ スクリプトが見つかりません: {script_name}")
                self.console_output.emit(f"❌ スクリプトが見つかりません: {script_name}")
                return False
        
        try:
            # Configurar variables de entorno para evitar conflictos de DLLs
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"
            env["MPLBACKEND"] = "Agg"
            env["QT_QPA_PLATFORM"] = "offscreen"
            # Permitir múltiples DLLs OpenMP si es necesario (evita conflictos)
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            
            # Configurar PYTHONPATH para que encuentre los módulos
            # ✅ Usar la misma lógica que para script_base_dir (donde está 0sec.py)
            # Esto asegura que encontremos el directorio raíz donde está 00_Pythonコード
            if self._cached_script_base_dir is None:
                # Calcular script_base_dir si no está cacheado (solo la primera vez)
                script_base_dir = None
                if self.project_folder:
                    potential_base = Path(self.project_folder).parent.parent
                    if (potential_base / "0sec.py").exists():
                        script_base_dir = potential_base
                    else:
                        current_dir = Path.cwd()
                        if (current_dir / "0sec.py").exists():
                            script_base_dir = current_dir
                        elif (current_dir / "01_model_builder.py").exists():
                            script_base_dir = current_dir
                
                if script_base_dir is None:
                    script_base_dir = Path.cwd()  # Fallback al directorio actual
                
                self._cached_script_base_dir = script_base_dir
            else:
                script_base_dir = self._cached_script_base_dir
            
            # Cachear python_code_folder también
            if self._cached_python_code_folder is None:
                python_code_folder = script_base_dir / "00_Pythonコード"
                self._cached_python_code_folder = python_code_folder
            else:
                python_code_folder = self._cached_python_code_folder
            
            # Incluir site-packages del venv para que encuentre librerías como xlsxwriter
            import site
            site_packages_paths = []
            try:
                # Obtener todos los site-packages del venv actual
                for site_pkg in site.getsitepackages():
                    if os.path.exists(site_pkg):
                        site_packages_paths.append(site_pkg)
            except:
                # ES: Fallback: buscar site-packages manualmente
                # EN: Fallback: search site-packages manually
                # JP: フォールバック: site-packagesを手動で探す
                venv_lib = Path(sys.executable).parent.parent / "Lib" / "site-packages"
                if venv_lib.exists():
                    site_packages_paths.append(str(venv_lib))
            
            # Construir PYTHONPATH
            pythonpath_parts = [str(python_code_folder)]
            pythonpath_parts.extend(site_packages_paths)
            
            # ES: Agregar PYTHONPATH existente si hay
            # EN: Add existing PYTHONPATH if present
            # JP: 既存のPYTHONPATHがあれば追加
            existing_pythonpath = env.get("PYTHONPATH", "")
            if existing_pythonpath:
                pythonpath_parts.append(existing_pythonpath)
            
            # Usar separador correcto según el sistema operativo
            separator = ";" if sys.platform == "win32" else ":"
            pythonpath = separator.join(pythonpath_parts)
            
            env["PYTHONPATH"] = pythonpath
            
            # ES: Obtener ruta del JSON de log
            # EN: Get JSON log path
            # JP: JSONログのパスを取得
            json_log_path = self._get_json_log_path(working_dir)
            
            # ES: Ejecutar script
            # EN: Run the script
            # JP: スクリプトを実行
            self.console_output.emit(f"🔧 Ejecutando: {script_path}")
            self.console_output.emit(f"📁 Working directory: {working_dir}")
            self.console_output.emit(f"📁 PYTHONPATH: {pythonpath}")
            self.console_output.emit(f"📝 JSON log: {json_log_path}")
            
            # Reiniciar el evento de parada del lector JSON
            self._json_reader_stop.clear()
            
            # ES: Iniciar hilo para leer JSON en tiempo real
            # EN: Start a thread to read JSON in real time
            # JP: JSONをリアルタイムで読むスレッドを開始
            json_reader_thread = threading.Thread(
                target=self._read_json_log,
                args=(json_log_path,),
                daemon=True
            )
            json_reader_thread.start()
            
            # ES: Ejecutar script con Popen para poder leer salida en tiempo real
            # EN: Run the script with Popen so we can read output in real time
            # JP: リアルタイムで出力を読むためPopenで実行
            process = subprocess.Popen(
                [sys.executable, script_path],
                cwd=working_dir,
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
            
            # ES: Leer stdout y stderr en tiempo real (el script original no genera JSON)
            # EN: Read stdout and stderr in real time (the original script does not generate JSON)
            # JP: stdout/stderrをリアルタイムで読む（元スクリプトはJSONを生成しない）
            # ✅ ACTIVADO: El script original imprime directamente a stdout/stderr
            def read_output(pipe, is_stderr=False):
                try:
                    while not stop_reading.is_set():
                        line = pipe.readline()
                        if not line:  # EOF o pipe cerrado
                            break
                        # ✅ Emitir directamente a consola (sin depender de JSON)
                        line_clean = line.rstrip('\n\r')
                        if line_clean:
                            self.console_output.emit(line_clean)
                            # ✅ Parsear progreso para extraer fold y trial
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
                    print("🛑 プロセスをキャンセル中...")
                    self.console_output.emit("🛑 プロセスをキャンセル中...")
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
                        print(f"⚠️ Error al cancelar proceso: {e}")
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
            
            # Detener el lector JSON
            self._json_reader_stop.set()
            json_reader_thread.join(timeout=1.0)  # Esperar máximo 1 segundo
            
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
                self.console_output.emit(f"✅ Script ejecutado exitosamente: {script_name}")
                return True
            else:
                self.console_output.emit(f"❌ Script falló con código {returncode}: {script_name}")
                # ES: ✅ Intentar leer cualquier salida restante de stderr para ver el error
                # EN: ✅ Try to read any remaining stderr output to see the error
                # JP: ✅ エラー確認のためstderrの残り出力を読んでみる
                try:
                    if process.stderr:
                        remaining_stderr = process.stderr.read()
                        if remaining_stderr:
                            for line in remaining_stderr.decode('utf-8', errors='replace').split('\n'):
                                line_clean = line.rstrip('\n\r')
                                if line_clean:
                                    self.console_output.emit(f"[STDERR] {line_clean}")
                except:
                    pass
                return False
                
        except Exception as e:
            self.console_output.emit(f"❌ Error ejecutando script {script_name}: {e}")
            import traceback
            error_trace = traceback.format_exc()
            self.console_output.emit(error_trace)
            self._json_reader_stop.set()  # Asegurar que el lector se detenga
            return False
    
    def _parse_progress(self, line):
        """
        Parsea el output del script para extraer información de progreso (fold y trial)
        y emite la señal progress_detailed
        """
        try:
            # ES: Solo parsear si estamos en el stage 01 (model_builder)
            # EN: Only parse when we are in stage 01 (model_builder)
            # JP: Stage 01（model_builder）のときのみ解析する
            if self.current_stage != '01_model_builder':
                return
            
            # ES: Detectar análisis de datos completado
            # EN: Detect completed data analysis
            # JP: データ分析完了を検出
            if 'データ分析完了' in line or 'データ分析が完了しました' in line:
                self.data_analysis_completed = True
                self.current_task = 'dcv'
                # ES: Emitir progreso actualizado
                # EN: Emit updated progress
                # JP: 更新した進捗を送信
                self.progress_detailed.emit(
                    self.current_trial,
                    self.total_trials,
                    self.current_fold,
                    self.total_folds,
                    self.current_pass,
                    self.total_passes,
                    self.current_task,
                    self.data_analysis_completed,
                    self.final_model_training,
                    self.shap_analysis,
                    self.current_model,
                    self.total_models
                )
                return
            
            # ES: Detectar inicio de análisis de datos
            # EN: Detect start of data analysis
            # JP: データ分析開始を検出
            if 'データ分析開始' in line:
                self.current_task = 'data_analysis'
                # ES: Emitir progreso actualizado
                # EN: Emit updated progress
                # JP: 更新した進捗を送信
                self.progress_detailed.emit(
                    self.current_trial,
                    self.total_trials,
                    self.current_fold,
                    self.total_folds,
                    self.current_pass,
                    self.total_passes,
                    self.current_task,
                    self.data_analysis_completed,
                    self.final_model_training,
                    self.shap_analysis,
                    self.current_model,
                    self.total_models
                )
                return
            
            # ES: Detectar entrenamiento del modelo final
            # EN: Detect final model training
            # JP: 最終モデル学習を検出
            if '最終モデル訓練' in line or '最終モデル訓練（全データ' in line:
                self.final_model_training = True
                self.current_task = 'final_model'
                # ES: Emitir progreso actualizado
                # EN: Emit updated progress
                # JP: 更新した進捗を送信
                self.progress_detailed.emit(
                    self.current_trial,
                    self.total_trials,
                    self.current_fold,
                    self.total_folds,
                    self.current_pass,
                    self.total_passes,
                    self.current_task,
                    self.data_analysis_completed,
                    self.final_model_training,
                    self.shap_analysis,
                    self.current_model,
                    self.total_models
                )
                return
            
            # ES: Detectar análisis SHAP
            # EN: Detect SHAP analysis
            # JP: SHAP解析を検出
            if 'SHAP' in line and ('分析' in line or 'analyze' in line.lower()):
                self.shap_analysis = True
                self.current_task = 'shap'
                # ES: Emitir progreso actualizado
                # EN: Emit updated progress
                # JP: 更新した進捗を送信
                self.progress_detailed.emit(
                    self.current_trial,
                    self.total_trials,
                    self.current_fold,
                    self.total_folds,
                    self.current_pass,
                    self.total_passes,
                    self.current_task,
                    self.data_analysis_completed,
                    self.final_model_training,
                    self.shap_analysis,
                    self.current_model,
                    self.total_models
                )
                return
            
            # ES: Detectar guardado completado
            # EN: Detect completed saving
            # JP: 保存完了を検出
            if '推論用バンドル保存' in line or '✅ 推論用バンドル保存' in line:
                self.saving_completed = True
                self.current_task = 'saving'
                # ES: Emitir progreso actualizado
                # EN: Emit updated progress
                # JP: 更新した進捗を送信
                self.progress_detailed.emit(
                    self.current_trial,
                    self.total_trials,
                    self.current_fold,
                    self.total_folds,
                    self.current_pass,
                    self.total_passes,
                    self.current_task,
                    self.data_analysis_completed,
                    self.final_model_training,
                    self.shap_analysis,
                    self.current_model,
                    self.total_models
                )
                return
            
            # ES: Detectar inicio de nueva pasada (target): "Double Cross-Validation: {target_name}" o "処理中: {target}"
            # EN: Detect the start of a new pass (target): "Double Cross-Validation: {target_name}" or "処理中: {target}"
            # JP: 新しいパス（目的変数）の開始を検出: "Double Cross-Validation: {target_name}" または "処理中: {target}"
            # ES: Priorizar "Double Cross-Validation" porque aparece después de "処理中"
            # EN: Prefer "Double Cross-Validation" because it appears after "処理中"
            # JP: "処理中"の後に出るため "Double Cross-Validation" を優先
            pass_match = re.search(r'Double\s+Cross-Validation:\s+(\w+)', line, re.IGNORECASE)
            target_name = None
            if pass_match:
                target_name = pass_match.group(1)
            else:
                # ES: Si no se encuentra "Double Cross-Validation", buscar "処理中"
                # EN: If "Double Cross-Validation" is not found, look for "処理中"
                # JP: "Double Cross-Validation"が見つからない場合は"処理中"を探す
                pass_match = re.search(r'処理中:\s+(\w+)', line)
                if pass_match:
                    target_name = pass_match.group(1)
            
            if target_name and target_name != self.last_detected_target:
                # ES: Nuevo target detectado: incrementar pasada
                # EN: New target detected: increment pass counter
                # JP: 新しい目的変数を検出: パス数を増やす
                self.last_detected_target = target_name
                self.current_pass += 1
                self.current_fold = 0  # Reset fold when the pass changes
                self.current_trial = 0  # ✅ Reset completed-trials counter when the pass changes
                self.current_model = 0  # Reset model when the pass changes
                self.completed_trials_in_current_fold = set()  # ✅ Reset the set of completed trials
                self.final_model_training = False  # Reset for a new pass
                self.shap_analysis = False  # Reset for a new pass
                self.saving_completed = False  # Reset for a new pass
                self.current_task = 'dcv'  # Back to DCV for the new pass
                # ES: Emitir progreso actualizado con la pasada correcta
                # EN: Emit updated progress with the correct pass
                # JP: 正しいパスで更新した進捗を送信
                self.progress_detailed.emit(
                    self.current_trial,
                    self.total_trials,
                    self.current_fold,
                    self.total_folds,
                    self.current_pass,
                    self.total_passes,
                    self.current_task,
                    self.data_analysis_completed,
                    self.final_model_training,
                    self.shap_analysis,
                    self.current_model,
                    self.total_models
                )
                return
            
            # ES: Detectar Outer Fold: "--- Outer Fold X/Y ---"
            # EN: Detect Outer Fold: "--- Outer Fold X/Y ---"
            # JP: Outer Foldを検出: "--- Outer Fold X/Y ---"
            fold_match = re.search(r'---\s*Outer\s+Fold\s+(\d+)/(\d+)\s*---', line, re.IGNORECASE)
            if fold_match:
                self.current_fold = int(fold_match.group(1))
                self.total_folds = int(fold_match.group(2))
                self.current_trial = 0  # ✅ Reset completed-trials counter when the fold changes
                self.current_model = 0  # Reset model when the fold changes
                self.completed_trials_in_current_fold = set()  # ✅ Reset the set of completed trials
                # ES: Emitir progreso actualizado
                # EN: Emit updated progress
                # JP: 更新した進捗を送信
                self.progress_detailed.emit(
                    self.current_trial,
                    self.total_trials,
                    self.current_fold,
                    self.total_folds,
                    self.current_pass,
                    self.total_passes,
                    self.current_task,
                    self.data_analysis_completed,
                    self.final_model_training,
                    self.shap_analysis,
                    self.current_model,
                    self.total_models
                )
                return
            
            # ES: Detectar inicio de optimización de modelo: "🔍 {model_name} 最適化中..."
            # EN: Detect start of model optimization: "🔍 {model_name} 最適化中..."
            # JP: モデル最適化開始を検出: "🔍 {model_name} 最適化中..."
            model_match = re.search(r'🔍\s+(\w+)\s+最適化中', line)
            if model_match:
                self.current_model += 1
                # ES: ✅ NO resetear contador de trials cuando cambia el modelo dentro del mismo fold
                # EN: ✅ Do NOT reset the trial counter when the model changes within the same fold
                # JP: ✅ 同一fold内でモデルが変わってもtrialカウンタはリセットしない
                # ES: El contador de trials debe continuar a través de todos los modelos en el mismo fold
                # EN: The trial counter must continue across all models within the same fold
                # JP: trialカウンタは同一fold内の全モデルを通して継続する必要がある
                # ES: Solo se resetea cuando cambia el fold
                # EN: It is only reset when the fold changes
                # JP: foldが変わるときだけリセットする
                # ES: Emitir progreso actualizado para mostrar el cambio de modelo
                # EN: Emit updated progress to reflect the model change
                # JP: モデル変更を反映するため進捗を更新して送信
                self.progress_detailed.emit(
                    self.current_trial,  # Mantener el contador actual (no resetear)
                    self.total_trials,
                    self.current_fold,
                    self.total_folds,
                    self.current_pass,
                    self.total_passes,
                    self.current_task,
                    self.data_analysis_completed,
                    self.final_model_training,
                    self.shap_analysis,
                    self.current_model,
                    self.total_models
                )
                return
            
            # ES: ✅ Formato de barra de progreso de Optuna: buscar "X/Y" (prioritario porque muestra trials completados)
            # EN: ✅ Optuna progress-bar format: look for "X/Y" (preferred because it shows completed trials)
            # JP: ✅ Optuna進捗バー形式: "X/Y"を探す（完了trial数が分かるため優先）
            # Ejemplo: "Best trial: 34. Best value: 4.04966: 100%|██████████| 50/50 [04:34<00:00,  2.34s/it]"
            # El formato "X/Y" muestra: X = trials completados, Y = total trials
            trial_progress_match = re.search(r'(\d+)/(\d+)\s*\[', line)
            if trial_progress_match:
                trials_completed = int(trial_progress_match.group(1))  # Número de trials completados (contador incremental)
                trial_total = int(trial_progress_match.group(2))  # Total trials
                
                # ✅ Usar el contador de trials completados (no el número del trial)
                self.current_trial = trials_completed
                self.total_trials = trial_total
                
                # ✅ Calcular valores acumulados para porcentaje lineal
                if self.current_pass > 0 and self.total_folds > 0:
                    trials_per_fold = trial_total
                    # Trials completados en passes anteriores
                    trials_in_previous_passes = (self.current_pass - 1) * self.total_folds * trials_per_fold
                    # Trials completados en folds anteriores del pass actual
                    trials_in_previous_folds = (self.current_fold - 1) * trials_per_fold
                    # Trials completados en el fold actual
                    self.accumulated_trial_current = trials_in_previous_passes + trials_in_previous_folds + trials_completed
                    # ES: Total de trials acumulados | EN: Total accumulated trials | JA: 累積trial総数
                    self.accumulated_trial_total = self.total_passes * self.total_folds * trials_per_fold
                else:
                    # Fallback: usar valores locales si no hay suficiente información
                    self.accumulated_trial_current = trials_completed
                    self.accumulated_trial_total = trial_total
                
                # Emitir progreso actualizado
                self.progress_detailed.emit(
                    self.current_trial,  # Trials completados en fold actual (para mostrar: 1/50, 2/50, etc.)
                    self.total_trials,   # Total trials per fold
                    self.current_fold,
                    self.total_folds,
                    self.current_pass,
                    self.total_passes,
                    self.current_task,
                    self.data_analysis_completed,
                    self.final_model_training,
                    self.shap_analysis,
                    self.current_model,
                    self.total_models
                )
                return
            
            # Detectar trial de Optuna: buscar patrones como "[I ...] Trial X finished" o "Trial X finished"
            # ✅ Estos mensajes indican que un trial se completó, incrementar contador
            trial_finished_match = re.search(r'\[I\s+\d+:\d+:\d+\.\d+\]\s+Trial\s+(\d+)\s+finished', line)
            if trial_finished_match:
                trial_id = int(trial_finished_match.group(1))  # ID del trial completado (puede ser 8, 13, 2, etc.)
                
                # ✅ Solo incrementar contador si este trial no fue contado antes
                if trial_id not in self.completed_trials_in_current_fold:
                    self.completed_trials_in_current_fold.add(trial_id)
                    self.current_trial += 1  # Incrementar contador de trials completados
                    
                    # ✅ Actualizar valores acumulados
                    if self.current_pass > 0 and self.total_folds > 0 and self.total_trials > 0:
                        trials_per_fold = self.total_trials
                        trials_in_previous_passes = (self.current_pass - 1) * self.total_folds * trials_per_fold
                        trials_in_previous_folds = (self.current_fold - 1) * trials_per_fold
                        self.accumulated_trial_current = trials_in_previous_passes + trials_in_previous_folds + self.current_trial
                        self.accumulated_trial_total = self.total_passes * self.total_folds * trials_per_fold
                
                # Emitir progreso actualizado
                self.progress_detailed.emit(
                    self.current_trial,  # Contador incremental de trials completados
                    self.total_trials,
                    self.current_fold,
                    self.total_folds,
                    self.current_pass,
                    self.total_passes,
                    self.current_task,
                    self.data_analysis_completed,
                    self.final_model_training,
                    self.shap_analysis,
                    self.current_model,
                    self.total_models
                )
                return
            
            # Otro formato: "Trial X finished with value..."
            trial_finished_match2 = re.search(r'Trial\s+(\d+)\s+finished', line, re.IGNORECASE)
            if trial_finished_match2:
                trial_id = int(trial_finished_match2.group(1))  # ID del trial completado
                
                # ✅ Solo incrementar contador si este trial no fue contado antes
                if trial_id not in self.completed_trials_in_current_fold:
                    self.completed_trials_in_current_fold.add(trial_id)
                    self.current_trial += 1  # Incrementar contador de trials completados
                    
                    # ✅ Actualizar valores acumulados
                    if self.current_pass > 0 and self.total_folds > 0 and self.total_trials > 0:
                        trials_per_fold = self.total_trials
                        trials_in_previous_passes = (self.current_pass - 1) * self.total_folds * trials_per_fold
                        trials_in_previous_folds = (self.current_fold - 1) * trials_per_fold
                        self.accumulated_trial_current = trials_in_previous_passes + trials_in_previous_folds + self.current_trial
                        self.accumulated_trial_total = self.total_passes * self.total_folds * trials_per_fold
                
                # Emitir progreso actualizado
                self.progress_detailed.emit(
                    self.current_trial,  # Contador incremental de trials completados
                    self.total_trials,
                    self.current_fold,
                    self.total_folds,
                    self.current_pass,
                    self.total_passes,
                    self.current_task,
                    self.data_analysis_completed,
                    self.final_model_training,
                    self.shap_analysis,
                    self.current_model,
                    self.total_models
                )
                return
                
        except Exception as e:
            # Silenciar errores de parsing para no interrumpir el flujo
            pass
    def cancel(self):
        """ES: Cancela la ejecución del análisis
        EN: Cancel the analysis execution
        JA: 解析の実行をキャンセル"""
        print("🛑 非線形解析をキャンセル中...")
        self._cancelled = True
        
        # Terminar proceso subprocess si está corriendo
        if self._current_process is not None:
            try:
                print("🛑 Terminando proceso subprocess...")
                self._current_process.terminate()
                # Esperar un poco (polling)
                for _ in range(20):  # Esperar hasta 2 segundos
                    if self._current_process.poll() is not None:
                        break
                    time.sleep(0.1)
                
                # Si aún no terminó, forzar kill
                if self._current_process.poll() is None:
                    print("⚠️ プロセスが終了しません。kill します...")
                    self._current_process.kill()
                    self._current_process.wait()
                else:
                    print("✅ Proceso subprocess terminado correctamente")
            except Exception as e:
                print(f"⚠️ Error al terminar proceso: {e}")
                try:
                    if self._current_process and self._current_process.poll() is None:
                        self._current_process.kill()
                        self._current_process.wait()
                except:
                    pass
        
        # Detener lectura de output
        if self._stop_reading is not None:
            self._stop_reading.set()
            print("✅ Threads de lectura detenidos")
        
        # Detener lector JSON
        self._json_reader_stop.set()
        print("✅ Lector JSON detenido")
        
        # Solicitar que el thread termine
        if self.isRunning():
            print("🛑 worker スレッドの終了を要求中...")
            self.quit()
        
        print("✅ キャンセル完了")
    
    def _save_config_file(self, config_file_path):
        """
        Guarda el archivo de configuración personalizada.
        Copia config.py completo y reemplaza solo los valores modificados desde la UI.
        """
        # ES: Buscar config.py en el directorio actual o en el directorio del script
        # EN: Look for config.py in the current directory or the script directory
        # JP: 現在ディレクトリまたはスクリプトディレクトリでconfig.pyを探す
        config_py_path = None
        possible_paths = [
            Path.cwd() / 'config.py',
            Path(__file__).parent / 'config.py',
            Path(self.output_folder).parent / 'config.py',
        ]
        
        for path in possible_paths:
            if path.exists():
                config_py_path = path
                break
        
        if not config_py_path:
            raise FileNotFoundError("No se encontró config.py. Asegúrate de que existe en el directorio de trabajo.")
        
        # ES: Leer config.py completo
        # EN: Read the full config.py
        # JP: config.pyを全文読み込む
        with open(config_py_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # Mapa de normalización de nombres de modelos
        model_name_map = {
            'random_forest': 'RandomForest',
            'lightgbm': 'LightGBM',
            'xgboost': 'XGBoost',
            'gradient_boost': 'GradientBoost',
            'ridge': 'Ridge',
            'lasso': 'Lasso',
            'elastic_net': 'ElasticNet'
        }
        
        # Función auxiliar para reemplazar valores en config.py
        def replace_config_value(content, pattern, new_value, is_string=True, is_list=False, is_dict=False, is_raw_string=False):
            """
            Reemplaza un valor en config.py usando regex.
            Mantiene la indentación original del archivo y preserva comentarios.
            Siempre agrega un espacio antes del comentario si existe.
            
            Args:
                is_raw_string: Si es True, usa r'...' en lugar de '...' para strings
            """
            pattern_clean = pattern.strip()
            
            if is_dict:
                # ES: Para diccionarios multilínea, buscar desde el patrón hasta el cierre de llaves
                # EN: For multi-line dicts, search from the pattern to the closing brace
                # JP: 複数行辞書はパターンから閉じカッコまで検索
                # Capturar la indentación original y comentario si existe
                # El patrón debe capturar todo el diccionario, incluyendo las llaves
                dict_pattern = rf'^(\s*)({re.escape(pattern_clean)}\s*=\s*{{)(.*?)(^\s*}})(\s*#.*)?$'
                def dict_replacer(match):
                    indent = match.group(1)
                    comment = match.group(5) if match.group(5) else ''
                    if comment:
                        comment = ' ' + comment.strip()  # Asegurar espacio antes del comentario
                    # new_value ya contiene el diccionario completo con llaves {}
                    # ES: Solo necesitamos agregar la indentación a cada línea
                    # EN: We only need to add the indentation to each line
                    # JP: 各行にインデントを付けるだけでよい
                    dict_lines = new_value.split('\n')
                    formatted_dict = '\n'.join([f"{indent}    {line}" if line.strip() else line for line in dict_lines])
                    # Si new_value es un string simple como "{'key': 'value'}", formatearlo mejor
                    if new_value.startswith('{') and new_value.endswith('}') and '\n' not in new_value:
                        # Es un diccionario en una línea, formatearlo en múltiples líneas
                        try:
                            import ast
                            dict_obj = ast.literal_eval(new_value)
                            formatted_items = []
                            for k, v in dict_obj.items():
                                formatted_items.append(f"{indent}    '{k}': '{v}',")
                            formatted_dict = '\n'.join(formatted_items)
                        except:
                            # Si falla el parsing, usar el valor tal cual pero con indentación
                            formatted_dict = f"{indent}    {new_value}"
                    return f"{indent}{pattern_clean} = {{\n{formatted_dict}\n{indent}}}{comment}"
                content = re.sub(dict_pattern, dict_replacer, content, flags=re.MULTILINE | re.DOTALL)
            elif is_list:
                # ES: Para listas multilínea, buscar desde el patrón hasta el cierre de corchetes
                # EN: For multi-line lists, search from the pattern to the closing bracket
                # JP: 複数行リストはパターンから閉じカッコまで検索
                # Capturar la indentación original y comentario si existe
                list_pattern = rf'^(\s*)({re.escape(pattern_clean)}\s*=\s*\[)(.*?)(\])(\s*#.*)?$'
                def list_replacer(match):
                    indent = match.group(1)
                    comment = match.group(5) if match.group(5) else ''
                    if comment:
                        comment = ' ' + comment.strip()  # Asegurar espacio antes del comentario
                    return f"{indent}{pattern_clean} = {new_value}{comment}"
                content = re.sub(list_pattern, list_replacer, content, flags=re.MULTILINE | re.DOTALL)
            else:
                # Para valores simples
                if is_string:
                    # ES: String: buscar el patrón y reemplazar el valor entre comillas
                    # EN: String: find the pattern and replace the value inside quotes
                    # JP: 文字列: パターンを探して引用符内の値を置換
                    # Capturar la indentación original, comillas y comentario si existe
                    # Manejar también raw strings (r'...' o r"...")
                    pattern_regex = rf'^(\s*)({re.escape(pattern_clean)}\s*=\s*)(r?)([\'"])([^\'"]*)(\4)(\s*#.*)?$'
                    def string_replacer(match):
                        indent = match.group(1)
                        raw_prefix = match.group(3)  # 'r' o vacío
                        quote = match.group(4)
                        comment = match.group(7) if match.group(7) else ''
                        if comment:
                            comment = ' ' + comment.strip()  # Asegurar espacio antes del comentario
                        # Asegurar que new_value no tenga comillas dobles incorrectas ni prefijos r
                        clean_value = new_value.strip("'\"")
                        # Si new_value ya tiene r' o r", quitarlo
                        if clean_value.startswith("r'") or clean_value.startswith('r"'):
                            clean_value = clean_value[2:]
                        elif clean_value.startswith("r"):
                            clean_value = clean_value[1:]
                        # Usar raw string si se especificó
                        if is_raw_string:
                            return f"{indent}{pattern_clean} = r{quote}{clean_value}{quote}{comment}"
                        else:
                            return f"{indent}{pattern_clean} = {quote}{clean_value}{quote}{comment}"
                    content = re.sub(pattern_regex, string_replacer, content, flags=re.MULTILINE)
                else:
                    # ES: Número o booleano: buscar el patrón y reemplazar el valor
                    # EN: Number/boolean: find the pattern and replace the value
                    # JP: 数値/真偽値: パターンを探して値を置換
                    # Capturar la indentación original y comentario si existe
                    # Manejar casos como "50#" (sin espacio) o "50 # comentario" (con espacio)
                    pattern_regex = rf'^(\s*)({re.escape(pattern_clean)}\s*=\s*)([^\n]+)$'
                    def value_replacer(match):
                        indent = match.group(1)
                        full_line = match.group(3).strip()
                        
                        # Separar el valor del comentario
                        # ES: Buscar # que puede estar pegado o con espacio
                        # EN: Look for # which may be attached or separated by a space
                        # JP: #がくっついている/空白ありの両方を考慮して探す
                        if '#' in full_line:
                            # Dividir por #, pero mantener el comentario
                            parts = full_line.split('#', 1)
                            old_value = parts[0].strip()
                            comment_text = parts[1].strip() if len(parts) > 1 else ''
                            
                            # Reconstruir con espacio antes del comentario
                            if comment_text:
                                comment = f" # {comment_text}"
                            else:
                                comment = ""
                        else:
                            # No hay comentario
                            comment = ""
                        
                        return f"{indent}{pattern_clean} = {new_value}{comment}"
                    content = re.sub(pattern_regex, value_replacer, content, flags=re.MULTILINE)
            
            return content
        
        # Reemplazar rutas (siempre se reemplazan)
        # Nota: En config.py estas son atributos de clase Config
        data_folder = os.path.join(self.output_folder, '01_データセット')
        result_folder = os.path.join(self.output_folder, '03_学習結果')
        model_folder = os.path.join(self.output_folder, '02_学習モデル')
        
        # Reemplazar atributos de clase Config
        # Para rutas, usar r'' para manejar correctamente las barras invertidas en Windows
        # Pasar solo el path sin comillas ni r, la función agregará r'...' correctamente
        config_content = replace_config_value(config_content, 'DATA_FOLDER', data_folder, is_string=True, is_raw_string=True)
        config_content = replace_config_value(config_content, 'RESULT_FOLDER', result_folder, is_string=True, is_raw_string=True)
        config_content = replace_config_value(config_content, 'MODEL_FOLDER', model_folder, is_string=True, is_raw_string=True)
        # ES: Usar analysis_df.xlsx como archivo de entrada para 01_model_builder
        # EN: Use analysis_df.xlsx as the input file for 01_model_builder
        # JP: 01_model_builderの入力ファイルとしてanalysis_df.xlsxを使用
        config_content = replace_config_value(config_content, 'INPUT_FILE', 'analysis_df.xlsx', is_string=True)
        
        # Reemplazar MODELS_TO_USE si está en config_values
        if 'models_to_use' in self.config_values and self.config_values['models_to_use']:
            normalized_models = []
            for model in self.config_values['models_to_use']:
                # Mantener el formato original de config.py (nombres en minúsculas con guiones bajos)
                normalized_models.append(f"'{model}'")
            models_str = f"[{', '.join(normalized_models)}]"
            config_content = replace_config_value(config_content, 'MODELS_TO_USE', models_str, is_string=False, is_list=True)
        
        # Reemplazar N_TRIALS
        if 'N_TRIALS' in self.config_values or 'n_trials' in self.config_values:
            n_trials = self.config_values.get('N_TRIALS', self.config_values.get('n_trials', 50))
            print(f"🔧 Reemplazando N_TRIALS con valor: {n_trials}")
            config_content = replace_config_value(config_content, 'N_TRIALS', str(n_trials), is_string=False)
            # ES: Verificar que el reemplazo funcionó
            # EN: Verify that the replacement worked
            # JP: 置換が成功したか確認
            if f"N_TRIALS = {n_trials}" in config_content or f"N_TRIALS = {n_trials} #" in config_content:
                print(f"✅ N_TRIALS reemplazado correctamente en config_custom.py")
            else:
                print(f"⚠️ 警告: N_TRIALS が正しく置換されていない可能性があります")
        
        # Reemplazar OUTER_SPLITS e INNER_SPLITS
        if 'outer_splits' in self.config_values or 'OUTER_SPLITS' in self.config_values:
            outer_splits = self.config_values.get('outer_splits', self.config_values.get('OUTER_SPLITS', 10))
            config_content = replace_config_value(config_content, 'OUTER_SPLITS', str(outer_splits), is_string=False)
        
        if 'inner_splits' in self.config_values or 'INNER_SPLITS' in self.config_values:
            inner_splits = self.config_values.get('inner_splits', self.config_values.get('INNER_SPLITS', 10))
            config_content = replace_config_value(config_content, 'INNER_SPLITS', str(inner_splits), is_string=False)
        
        # Reemplazar DEFAULT_TOP_K
        if 'top_k' in self.config_values:
            config_content = replace_config_value(config_content, 'DEFAULT_TOP_K', str(self.config_values['top_k']), is_string=False)
        
        # Reemplazar DEFAULT_CORR_THRESHOLD
        if 'corr_threshold' in self.config_values:
            config_content = replace_config_value(config_content, 'DEFAULT_CORR_THRESHOLD', str(self.config_values['corr_threshold']), is_string=False)
        
        # Reemplazar USE_CORRELATION_REMOVAL
        if 'use_correlation_removal' in self.config_values:
            use_corr = str(self.config_values['use_correlation_removal'])
            config_content = replace_config_value(config_content, 'USE_CORRELATION_REMOVAL', use_corr, is_string=False)
        
        # Reemplazar TRANSFORM_METHOD
        if 'transform_method' in self.config_values:
            config_content = replace_config_value(config_content, 'TRANSFORM_METHOD', self.config_values['transform_method'], is_string=True)
        
        # Reemplazar SHAP_MODE
        if 'shap_mode' in self.config_values:
            config_content = replace_config_value(config_content, 'SHAP_MODE', self.config_values['shap_mode'], is_string=True)
        
        # Reemplazar SHAP_MAX_SAMPLES
        if 'shap_max_samples' in self.config_values:
            config_content = replace_config_value(config_content, 'SHAP_MAX_SAMPLES', str(self.config_values['shap_max_samples']), is_string=False)
        
        # Reemplazar DEFAULT_MODEL
        if 'default_model' in self.config_values:
            default_model = self.config_values['default_model']
            # Mantener el formato original (minúsculas con guiones bajos)
            # Pasar solo el valor sin comillas, la función agregará las comillas correctas
            config_content = replace_config_value(config_content, 'DEFAULT_MODEL', default_model, is_string=True)
        
        # Reemplazar SHOW_OPTUNA_PROGRESS
        if 'show_optuna_progress' in self.config_values:
            show_progress = str(self.config_values['show_optuna_progress'])
            config_content = replace_config_value(config_content, 'SHOW_OPTUNA_PROGRESS', show_progress, is_string=False)
        
        # Reemplazar VERBOSE_LOGGING
        if 'verbose_logging' in self.config_values:
            verbose = str(self.config_values['verbose_logging'])
            config_content = replace_config_value(config_content, 'VERBOSE_LOGGING', verbose, is_string=False)
        
        # Reemplazar SHOW_DATA_ANALYSIS_DETAILS
        if 'show_data_analysis' in self.config_values:
            show_details = str(self.config_values['show_data_analysis'])
            config_content = replace_config_value(config_content, 'SHOW_DATA_ANALYSIS_DETAILS', show_details, is_string=False)
        
        # Reemplazar FEATURE_COLUMNS (selected_features)
        if 'selected_features' in self.config_values and self.config_values['selected_features']:
            features_list = self.config_values['selected_features']
            features_str = '[' + ', '.join([f"'{f}'" for f in features_list]) + ']'
            config_content = replace_config_value(config_content, 'FEATURE_COLUMNS', features_str, is_string=False, is_list=True)
            
            # También actualizar las listas de tipos de características para que solo contengan las seleccionadas
            # Esto es necesario para que la validación de Config.validate() pase
            # ES: Leer las listas originales de config.py para determinar el tipo de cada característica
            # EN: Read the original lists from config.py to determine each feature's type
            # JP: 各特徴量のタイプ判定のため、元のconfig.pyのリストを読む
            from config import Config as OriginalConfig
            
            # Filtrar cada lista de tipos para que solo contenga características seleccionadas
            continuous_selected = [f for f in OriginalConfig.CONTINUOUS_FEATURES if f in features_list]
            discrete_selected = [f for f in OriginalConfig.DISCRETE_FEATURES if f in features_list]
            binary_selected = [f for f in OriginalConfig.BINARY_FEATURES if f in features_list]
            integer_selected = [f for f in OriginalConfig.INTEGER_FEATURES if f in features_list]
            
            print(f"🔍 選択した特徴量: {features_list}")
            print(f"🔍 CONTINUOUS_FEATURES filtradas: {continuous_selected}")
            print(f"🔍 DISCRETE_FEATURES filtradas: {discrete_selected}")
            print(f"🔍 BINARY_FEATURES filtradas: {binary_selected}")
            print(f"🔍 INTEGER_FEATURES filtradas: {integer_selected}")
            
            # Reemplazar las listas de tipos
            if continuous_selected:
                continuous_str = '[' + ', '.join([f"'{f}'" for f in continuous_selected]) + ']'
                config_content = replace_config_value(config_content, 'CONTINUOUS_FEATURES', continuous_str, is_string=False, is_list=True)
            else:
                continuous_str = '[]'
                config_content = replace_config_value(config_content, 'CONTINUOUS_FEATURES', continuous_str, is_string=False, is_list=True)
            
            if discrete_selected:
                discrete_str = '[' + ', '.join([f"'{f}'" for f in discrete_selected]) + ']'
                config_content = replace_config_value(config_content, 'DISCRETE_FEATURES', discrete_str, is_string=False, is_list=True)
            else:
                discrete_str = '[]'
                config_content = replace_config_value(config_content, 'DISCRETE_FEATURES', discrete_str, is_string=False, is_list=True)
            
            if binary_selected:
                binary_str = '[' + ', '.join([f"'{f}'" for f in binary_selected]) + ']'
                config_content = replace_config_value(config_content, 'BINARY_FEATURES', binary_str, is_string=False, is_list=True)
            else:
                binary_str = '[]'
                config_content = replace_config_value(config_content, 'BINARY_FEATURES', binary_str, is_string=False, is_list=True)
            
            if integer_selected:
                integer_str = '[' + ', '.join([f"'{f}'" for f in integer_selected]) + ']'
                config_content = replace_config_value(config_content, 'INTEGER_FEATURES', integer_str, is_string=False, is_list=True)
            else:
                integer_str = '[]'
                config_content = replace_config_value(config_content, 'INTEGER_FEATURES', integer_str, is_string=False, is_list=True)
        
        # Reemplazar TARGET_COLUMNS si está en config_values
        if 'TARGET_COLUMNS' in self.config_values and self.config_values['TARGET_COLUMNS']:
            targets_list = self.config_values['TARGET_COLUMNS']
            if isinstance(targets_list, list):
                targets_str = '[' + ', '.join([f"'{t}'" for t in targets_list]) + ']'
                config_content = replace_config_value(config_content, 'TARGET_COLUMNS', targets_str, is_string=False, is_list=True)
        
        # Reemplazar MANDATORY_FEATURES
        # Si hay características seleccionadas, filtrar MANDATORY_FEATURES para que solo contenga las seleccionadas
        if 'selected_features' in self.config_values and self.config_values['selected_features']:
            features_list = self.config_values['selected_features']
            # ES: Leer MANDATORY_FEATURES original de config.py
            # EN: Read the original MANDATORY_FEATURES from config.py
            # JP: 元のconfig.pyのMANDATORY_FEATURESを読む
            from config import Config as OriginalConfig
            # Filtrar MANDATORY_FEATURES para que solo contenga características seleccionadas
            mandatory_filtered = [f for f in OriginalConfig.MANDATORY_FEATURES if f in features_list]
            if mandatory_filtered:
                mandatory_str = '[' + ', '.join([f"'{m}'" for m in mandatory_filtered]) + ']'
                config_content = replace_config_value(config_content, 'MANDATORY_FEATURES', mandatory_str, is_string=False, is_list=True)
                print(f"🔍 MANDATORY_FEATURES filtradas: {mandatory_filtered}")
            else:
                # Si no hay características obligatorias seleccionadas, dejar la lista vacía
                mandatory_str = '[]'
                config_content = replace_config_value(config_content, 'MANDATORY_FEATURES', mandatory_str, is_string=False, is_list=True)
                print(f"🔍 MANDATORY_FEATURES が空です（必須特徴量が選択されていません）")
        elif 'MANDATORY_FEATURES' in self.config_values and self.config_values['MANDATORY_FEATURES']:
            # Si se proporciona explícitamente en config_values, usarlo
            mandatory_list = self.config_values['MANDATORY_FEATURES']
            if isinstance(mandatory_list, list):
                mandatory_str = '[' + ', '.join([f"'{m}'" for m in mandatory_list]) + ']'
                config_content = replace_config_value(config_content, 'MANDATORY_FEATURES', mandatory_str, is_string=False, is_list=True)
        
        # Reemplazar PARETO_OBJECTIVES si está en config_values
        if 'pareto_objectives' in self.config_values and self.config_values['pareto_objectives']:
            pareto_dict = self.config_values['pareto_objectives']
            if isinstance(pareto_dict, dict):
                # Formatear como diccionario Python válido, una línea por item
                pareto_lines = [f"'{k}': '{v}'," for k, v in pareto_dict.items()]
                pareto_str = '\n'.join(pareto_lines)
                config_content = replace_config_value(config_content, 'PARETO_OBJECTIVES', pareto_str, is_string=False, is_dict=True)
        
        # ES: Agregar comentario al inicio indicando que es un archivo generado
        # EN: Add a header comment indicating this file is generated
        # JP: 生成ファイルであることを示すヘッダーコメントを追加
        header_comment = "# Configuración personalizada para análisis no lineal\n# Generado automáticamente - Basado en config.py\n# Solo se modifican los valores configurados desde la UI\n\n"
        
        # ES: Verificar si ya tiene el comentario
        # EN: Check whether it already has the header comment
        # JP: 既にヘッダーコメントがあるか確認
        if not config_content.startswith("# Configuración personalizada"):
            config_content = header_comment + config_content
        
        # ES: Escribir archivo
        # EN: Write file
        # JP: ファイルを書き込む
        with open(config_file_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"✅ 設定を保存しました: {config_file_path}")
        
        # ES: Debug: Verificar que N_TRIALS está en el archivo guardado
        # EN: Debug: Verify N_TRIALS is present in the saved file
        # JP: Debug: 保存したファイルにN_TRIALSが含まれるか確認
        if 'N_TRIALS' in config_content:
            # ES: Buscar la línea de N_TRIALS
            # EN: Find the N_TRIALS line
            # JP: N_TRIALSの行を探す
            for line in config_content.split('\n'):
                if 'N_TRIALS' in line and '=' in line:
                    print(f"🔍 config_custom.py の N_TRIALS 行: {line.strip()}")
                    break
        else:
            print(f"⚠️ 警告: 保存後の config_custom.py に N_TRIALS が見つかりません")
    
    def _find_graphs(self, output_folder):
        """ES: Busca gráficos generados en la carpeta de salida
        EN: Search for generated graphs in the output folder
        JA: 出力フォルダ内の生成グラフを検索"""
        graph_paths = []
        
        # ES: Buscar en subcarpetas comunes
        # EN: Search in common subfolders
        # JP: よくあるサブフォルダ内を検索
        search_folders = [
            os.path.join(output_folder, "03_学習結果"),
            output_folder
        ]
        
        image_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.pdf']
        
        for folder in search_folders:
            if os.path.exists(folder):
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            full_path = os.path.join(root, file)
                            graph_paths.append(full_path)
        
        # Ordenar por nombre
        graph_paths.sort()
        
        print(f"📊 グラフを {len(graph_paths)} 件検出")
        return graph_paths
    
    def _save_analysis_results_json(self):
        """
        Guarda los datos de resultados del análisis en un archivo JSON
        para facilitar la lectura posterior
        """
        try:
            # ES: Ruta donde guardar el JSON (directamente en la carpeta de resultados)
            # EN: Path to save the JSON (directly in the results folder)
            # JP: JSON保存先（結果フォルダ直下）
            result_folder = os.path.join(self.output_folder, '03_学習結果')
            
            if not os.path.exists(result_folder):
                print(f"⚠️ 結果フォルダーが見つかりません: {result_folder}")
                return
            
            json_path = os.path.join(result_folder, 'analysis_results.json')
            
            # Extraer datos del DataFrame filtrado
            data_count = len(self.filtered_df) if self.filtered_df is not None else 0
            
            # Calcular data_range (min y max de columnas numéricas)
            data_range = "N/A"
            if self.filtered_df is not None and len(self.filtered_df) > 0:
                numeric_cols = self.filtered_df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    df_numeric = self.filtered_df[numeric_cols]
                    min_vals = df_numeric.min()
                    max_vals = df_numeric.max()
                    # ES: Crear string con rango de algunas columnas principales
                    # EN: Build a range string for some main columns
                    # JP: 主な列の範囲文字列を作る
                    range_parts = []
                    for col in numeric_cols[:5]:  # Primeras 5 columnas numéricas
                        range_parts.append(f"{col}: [{min_vals[col]:.2f}, {max_vals[col]:.2f}]")
                    data_range = "; ".join(range_parts)
                    if len(numeric_cols) > 5:
                        data_range += f" ... (+{len(numeric_cols) - 5} más)"
            
            # Obtener filters_applied desde config_values
            # ES: Guardar como lista para que pueda ser leída después
            # EN: Save as a list so it can be read later
            # JP: 後で読めるようリストとして保存
            filters_applied = self.config_values.get('filters_applied', [])
            if not filters_applied or filters_applied == []:
                filters_applied = []
            
            # ES: Extraer información de modelos y métricas CV desde dcv_results.pkl
            # EN: Extract model info and CV metrics from dcv_results.pkl
            # JP: dcv_results.pklからモデル情報とCV指標を抽出
            # dcv_results.pkl está directamente en 03_学習結果 (sin subcarpeta)
            models_trained = 0
            models = {}
            pickle_path = os.path.join(result_folder, 'dcv_results.pkl')
            
            if os.path.exists(pickle_path):
                try:
                    import pickle
                    import numpy as np
                    with open(pickle_path, 'rb') as f:
                        pickle_data = pickle.load(f)
                    
                    # La estructura de dcv_results.pkl es un diccionario donde las claves son los nombres de los targets
                    # Cada valor es un diccionario con los resultados del DCV para ese target
                    if isinstance(pickle_data, dict):
                        # Iterar sobre cada target (摩耗量, 上面ダレ量, 側面ダレ量)
                        for target_name, result_data in pickle_data.items():
                            if isinstance(result_data, dict):
                                # ES: Extraer información del modelo
                                # EN: Extract model information
                                # JP: モデル情報を抽出
                                model_entry = {
                                    'model_name': result_data.get('final_model_name', 'Unknown'),
                                    'target_name': target_name
                                }
                                
                                # Extraer métricas CV (estas son las métricas principales)
                                cv_mae = result_data.get('cv_mae')
                                cv_rmse = result_data.get('cv_rmse')
                                cv_r2 = result_data.get('cv_r2')
                                
                                # Convertir a float si es necesario (puede ser numpy scalar o None)
                                def safe_float(value):
                                    if value is None:
                                        return None
                                    if isinstance(value, (int, float)):
                                        return float(value)
                                    if hasattr(value, 'item'):
                                        try:
                                            return float(value.item())
                                        except:
                                            return None
                                    try:
                                        return float(value)
                                    except:
                                        return None
                                
                                model_entry['cv_mae'] = safe_float(cv_mae)
                                model_entry['cv_rmse'] = safe_float(cv_rmse)
                                model_entry['cv_r2'] = safe_float(cv_r2)
                                
                                # ES: Extraer parámetros del modelo
                                # EN: Extract model parameters
                                # JP: モデルパラメータを抽出
                                best_params = result_data.get('best_params', {})
                                if best_params:
                                    # Convertir parámetros a tipos básicos
                                    clean_params = {}
                                    for param_name, param_value in best_params.items():
                                        if isinstance(param_value, (int, float, str, bool, type(None))):
                                            clean_params[param_name] = param_value
                                        elif hasattr(param_value, 'item'):
                                            try:
                                                clean_params[param_name] = float(param_value.item())
                                            except:
                                                clean_params[param_name] = str(param_value)
                                        else:
                                            clean_params[param_name] = str(param_value)
                                    model_entry['best_params'] = clean_params
                                
                                # ES: Extraer información de fold_results si está disponible
                                # EN: Extract fold_results information if available
                                # JP: fold_resultsがあれば情報を抽出
                                fold_results = result_data.get('fold_results', [])
                                if fold_results:
                                    # Calcular estadísticas de los folds
                                    fold_maes = [fr.get('mae') for fr in fold_results if fr.get('mae') is not None]
                                    fold_rmses = [fr.get('rmse') for fr in fold_results if fr.get('rmse') is not None]
                                    fold_r2s = [fr.get('r2') for fr in fold_results if fr.get('r2') is not None]
                                    
                                    if fold_maes:
                                        model_entry['fold_mae_mean'] = safe_float(np.mean(fold_maes))
                                        model_entry['fold_mae_std'] = safe_float(np.std(fold_maes))
                                    if fold_rmses:
                                        model_entry['fold_rmse_mean'] = safe_float(np.mean(fold_rmses))
                                        model_entry['fold_rmse_std'] = safe_float(np.std(fold_rmses))
                                    if fold_r2s:
                                        model_entry['fold_r2_mean'] = safe_float(np.mean(fold_r2s))
                                        model_entry['fold_r2_std'] = safe_float(np.std(fold_r2s))
                                
                                models[target_name] = model_entry
                                models_trained += 1
                                
                        print(f"✅ dcv_results.pkl から CV 指標付きモデルを {models_trained} 件抽出しました")
                except Exception as e:
                    print(f"⚠️ dcv_results.pkl の読み込み中にエラー（モデル抽出）: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ES: Calcular tiempo de análisis (si está disponible)
            # EN: Compute analysis duration (if available)
            # JP: 解析時間を計算（利用可能なら）
            analysis_duration = getattr(self, 'analysis_duration', None)
            if analysis_duration is not None:
                # Convertir a formato legible (horas:minutos:segundos)
                hours = int(analysis_duration // 3600)
                minutes = int((analysis_duration % 3600) // 60)
                seconds = int(analysis_duration % 60)
                milliseconds = int((analysis_duration % 1) * 1000)
                
                if hours > 0:
                    duration_str = f"{hours}時間{minutes}分{seconds}秒"
                elif minutes > 0:
                    duration_str = f"{minutes}分{seconds}秒"
                else:
                    duration_str = f"{seconds}.{milliseconds:03d}秒"
                
                analysis_duration_seconds = round(analysis_duration, 3)
            else:
                duration_str = "N/A"
                analysis_duration_seconds = None
            
            # ES: Crear diccionario con los datos
            # EN: Build a dictionary with the data
            # JP: データ辞書を作成
            results_data = {
                'data_count': data_count,
                'models_trained': models_trained,
                'filters_applied': filters_applied if filters_applied else [],
                'data_range': data_range,
                'output_folder': self.output_folder,
                'models': models if models else {},
                'analysis_duration_seconds': analysis_duration_seconds,
                'analysis_duration_formatted': duration_str
            }
            
            # ES: Guardar en JSON
            # EN: Save as JSON
            # JP: JSONとして保存
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"✅ 解析データを保存しました: {json_path}")
            
        except Exception as e:
            print(f"⚠️ 解析データのJSON保存中にエラー: {e}")
            import traceback
            traceback.print_exc()
