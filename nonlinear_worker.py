"""
Worker para ejecutar análisis no lineal en un thread separado
Ejecuta los scripts 01_model_builder.py, 02_prediction.py, 03_pareto_analyzer.py
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
    """Worker que ejecuta el análisis no lineal en un thread separado"""
    
    # Señales para comunicación con la GUI
    progress_updated = Signal(int, str)  # (value, message)
    progress_detailed = Signal(int, int, int, int, int, int, str, bool, bool, bool, int, int)  # (trial_current, trial_total, fold_current, fold_total, pass_current, pass_total, current_task, data_analysis_completed, final_model_training, shap_analysis, model_current, model_total)
    status_updated = Signal(str)  # message
    finished = Signal(dict)  # results dict
    error = Signal(str)  # error message
    console_output = Signal(str)  # mensaje de consola (para mostrar en PyCharm/consola)
    
    def __init__(self, filtered_df, project_folder, parent=None, config_values=None):
        """
        Inicializa el worker
        
        Parameters
        ----------
        filtered_df : pd.DataFrame
            DataFrame con los datos filtrados
        project_folder : str
            Carpeta base del proyecto
        parent : QWidget, optional
            Widget padre
        config_values : dict, optional
            Valores de configuración del diálogo
        """
        super().__init__(parent)
        self.filtered_df = filtered_df
        self.project_folder = project_folder
        self.config_values = config_values or {}
        self.output_folder = None
        self.current_stage = None
        self._json_reader_stop = threading.Event()  # Event para detener el lector JSON
        self._cancelled = False  # Flag para cancelación
        self._current_process = None  # Referencia al proceso subprocess actual
        self._stop_reading = None  # Event para detener lectura de output
        self._cached_script_base_dir = None  # Cachear script_base_dir para evitar verificaciones repetidas
        self._cached_python_code_folder = None  # Cachear python_code_folder
        
        # Estado del progreso para parsing
        self.current_fold = 0
        self.total_folds = self.config_values.get('outer_splits', self.config_values.get('OUTER_SPLITS', 10))
        self.current_trial = 0  # Contador de trials completados en el fold actual (incremental: 1, 2, 3...)
        # Normalizar nombre: puede venir como 'n_trials' o 'N_TRIALS'
        self.total_trials = self.config_values.get('N_TRIALS', self.config_values.get('n_trials', 50))
        self.current_model = 0
        self.total_models = len(self.config_values.get('MODELS_TO_USE', ['random_forest', 'lightgbm']))
        self.current_pass = 0  # Pasada actual (target actual)
        self.total_passes = len(self.config_values.get('TARGET_COLUMNS', []))  # Total de pasadas (targets)
        # Si no hay TARGET_COLUMNS en config, usar un valor por defecto (normalmente 3)
        if self.total_passes == 0:
            self.total_passes = 3  # Valor por defecto
        self.last_detected_target = None  # Para evitar detectar el mismo target dos veces
        
        # ✅ Variables para progreso acumulado (para cálculo de porcentaje lineal)
        self.accumulated_trial_current = 0  # Total de trials completados acumulados (a través de todos los passes, folds y trials)
        self.accumulated_trial_total = 0  # Total de trials acumulado (calculado dinámicamente: passes * folds * trials_per_fold)
        
        # ✅ Set para rastrear qué trials ya fueron contados (evitar contar el mismo trial dos veces)
        self.completed_trials_in_current_fold = set()  # IDs de trials completados en el fold actual
        
        # Estados adicionales para tareas dentro de 01_model_builder
        self.data_analysis_completed = False  # Análisis de datos completado
        self.current_task = 'initialization'  # Tarea actual: initialization, data_analysis, dcv, final_model, shap, saving
        self.final_model_training = False  # Entrenamiento del modelo final
        self.shap_analysis = False  # Análisis SHAP
        self.saving_completed = False  # Guardado completado
        
    def run(self):
        """Ejecuta el análisis no lineal"""
        import time
        start_time = time.time()  # Registrar tiempo de inicio
        self.analysis_start_time = start_time
        
        try:
            # Verificar si es carga de carpeta existente
            load_existing = self.config_values.get('load_existing', False)
            selected_folder_path = self.config_values.get('selected_folder_path', '')
            
            if load_existing and selected_folder_path:
                # Cargar carpeta existente sin ejecutar análisis
                self.status_updated.emit("📁 既存結果を読み込み中...")
                self.progress_updated.emit(50, "既存結果を読み込み中...")
                
                # Usar la carpeta seleccionada como output_folder
                self.output_folder = selected_folder_path
                
                # Buscar gráficos generados
                graph_paths = self._find_graphs(self.output_folder)
                
                # Buscar carpeta de resultados para obtener subfolders
                result_folder = os.path.join(self.output_folder, '03_学習結果')
                subfolders = {}
                if os.path.exists(result_folder):
                    subfolders['03_学習結果'] = result_folder
                
                # Emitir resultados como carga existente
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
                
                # Emitir finished para que la GUI muestre los resultados existentes
                self.finished.emit(results_existing)
                return
            
            # Si no es carga existente, ejecutar análisis normal
            # Crear carpeta de salida
            self.status_updated.emit("📁 Creando carpeta de salida...")
            
            # Verificar cancelación antes de crear carpetas
            if self._cancelled:
                print("🛑 Análisis cancelado antes de crear carpetas")
                return
            
            folder_manager = NonlinearFolderManager(self.project_folder)
            self.output_folder = folder_manager.create_output_folder()
            subfolders = folder_manager.create_subfolder_structure(self.output_folder)
            
            # Verificar cancelación después de crear carpetas
            if self._cancelled:
                print("🛑 Análisis cancelado después de crear carpetas")
                return
            
            # Guardar datos filtrados
            self.status_updated.emit("💾 Guardando datos filtrados...")
            data_folder = os.path.join(self.output_folder, "01_データセット")
            os.makedirs(data_folder, exist_ok=True)
            
            input_file = os.path.join(data_folder, "filtered_data.xlsx")
            df_to_save = self.filtered_df.copy()
            # (Mantenemos el comportamiento actual de guardado de filtered_data.xlsx)
            df_to_save.to_excel(input_file, index=False)
            print(f"✅ Datos guardados: {input_file}")

            # Crear un segundo archivo para el análisis del modelo: analysis_df.xlsx
            # A partir de filtered_data, eliminar columnas no deseadas como '材料' y '実験日'
            analysis_df = df_to_save.copy()
            cols_to_drop = ['材料', '実験日']
            try:
                drop_cols = [c for c in cols_to_drop if c in analysis_df.columns]
                if drop_cols:
                    analysis_df = analysis_df.drop(columns=drop_cols)
                    print(f"ℹ️ Columnas eliminadas en analysis_df.xlsx: {drop_cols}")
                # Forzar que las columnas enteras no sean int64 al leerlas en 01:
                # convertir columnas int a float para que pd.read_excel las infiera como float64.
                int_cols_analysis = analysis_df.select_dtypes(include=["int64", "int32", "int"]).columns
                if len(int_cols_analysis) > 0:
                    analysis_df[int_cols_analysis] = analysis_df[int_cols_analysis].astype("float64")
                    print(f"ℹ️ Columnas enteras convertidas a float en analysis_df.xlsx: {list(int_cols_analysis)}")
            except Exception as e:
                print(f"⚠️ No se pudieron preparar columnas en analysis_df.xlsx: {e}")

            analysis_file = os.path.join(data_folder, "analysis_df.xlsx")
            analysis_df.to_excel(analysis_file, index=False)
            print(f"✅ Datos de análisis guardados: {analysis_file}")
            
            # Verificar cancelación después de guardar datos
            if self._cancelled:
                print("🛑 Análisis cancelado después de guardar datos")
                return
            
            # Guardar configuración personalizada directamente como config.py
            # (en esta carpeta SOLO existirá este config.py, que es el modificado)
            config_file = os.path.join(self.output_folder, "config.py")
            self._save_config_file(config_file)
            
            # Verificar cancelación después de guardar configuración
            if self._cancelled:
                print("🛑 Análisis cancelado después de guardar configuración")
                return
            
            # Copiar scripts necesarios a la carpeta de salida
            self.status_updated.emit("📋 Copiando scripts...")
            # Ya no copiamos el config.py genérico del proyecto; usamos el config.py generado arriba
            scripts_to_copy = ["01_model_builder.py", "02_prediction.py", "03_pareto_analyzer.py"]
            
            # ✅ Buscar scripts en el directorio donde está 0sec.py (directorio del proyecto)
            # project_folder es la carpeta base, pero los scripts están en el directorio padre
            script_base_dir = None
            if self.project_folder:
                # project_folder es algo como "Archivos_de_salida/Proyecto_79"
                # Los scripts están en el directorio padre (donde está 0sec.py)
                potential_base = Path(self.project_folder).parent.parent
                if (potential_base / "0sec.py").exists():
                    script_base_dir = potential_base
                else:
                    # Intentar buscar desde el directorio actual
                    current_dir = Path.cwd()
                    if (current_dir / "0sec.py").exists():
                        script_base_dir = current_dir
                    elif (current_dir / "01_model_builder.py").exists():
                        script_base_dir = current_dir
            
            if script_base_dir is None:
                script_base_dir = Path.cwd()  # Fallback al directorio actual
            
            for script in scripts_to_copy:
                # Verificar cancelación durante copia de scripts
                if self._cancelled:
                    print("🛑 Análisis cancelado durante copia de scripts")
                    return
                
                script_path = script_base_dir / script
                if script_path.exists():
                    import shutil
                    dest = os.path.join(self.output_folder, script)
                    shutil.copy2(str(script_path), dest)
                    print(f"✅ Script copiado: {script_path} → {dest}")
                else:
                    print(f"⚠️ Script no encontrado: {script_path}")
            
            # Verificar cancelación antes de ejecutar Stage 01
            if self._cancelled:
                print("🛑 Análisis cancelado antes de ejecutar Stage 01")
                return
            
            # Ejecutar Stage 01: Model Builder
            self.current_stage = '01_model_builder'
            self.status_updated.emit("🔧 モデル構築中...")
            self.progress_updated.emit(10, "モデル構築中...")
            
            # Verificar cancelación antes de ejecutar
            if self._cancelled:
                print("🛑 Análisis cancelado antes de ejecutar Stage 01")
                return
            
            success_01 = self._run_script("01_model_builder.py", self.output_folder)
            
            # Si fue cancelado, no emitir error
            if self._cancelled:
                print("🛑 Análisis cancelado durante Stage 01")
                return
            
            if not success_01:
                self.error.emit("❌ Error en Stage 01: Model Builder")
                return
            
            # Calcular tiempo total de análisis
            end_time = time.time()
            analysis_duration = end_time - start_time
            self.analysis_duration = analysis_duration
            
            # Guardar resultados en JSON antes de mostrar la pantalla de resumen
            self._save_analysis_results_json()
            
            # Buscar gráficos generados (para referencia, pero no se mostrarán)
            graph_paths = self._find_graphs(self.output_folder)
            
            # Emitir resultados del Stage 01 como 'completed' para ir directamente a la pantalla de resumen
            results_01 = {
                'stage': 'completed',  # Cambiar a 'completed' para que vaya directamente a _show_final_results
                'output_folder': self.output_folder,
                'graph_paths': graph_paths,
                'subfolders': subfolders,
                'all_stages_completed': False,  # Indicar que solo se completó el stage 01
                'load_existing': False  # No es carga existente, es análisis nuevo
            }
            
            self.progress_updated.emit(100, "Stage 01 completado")
            self.status_updated.emit("✅ Stage 01 completado. Mostrando resultados...")
            
            # Emitir finished para que la GUI muestre directamente la pantalla de resumen
            self.finished.emit(results_01)
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error en análisis no lineal: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.error.emit(error_msg)
    
    def run_stage2_and_3(self):
        """
        Continúa con los stages 02 y 03 después de que el usuario confirme
        Este método se llama desde la GUI cuando el usuario hace OK en el visor de gráficos
        """
        print("🔍 DEBUG run_stage2_and_3: MÉTODO LLAMADO", flush=True)
        print(f"🔍 DEBUG run_stage2_and_3: output_folder = {self.output_folder}", flush=True)
        try:
            # Verificar cancelación antes de continuar
            if self._cancelled:
                print("🛑 Análisis cancelado antes de ejecutar Stage 02")
                return
            
            # Ejecutar Stage 02: Prediction
            self.current_stage = '02_prediction'
            self.status_updated.emit("🔧 Ejecutando Stage 02: Prediction...")
            self.progress_updated.emit(60, "Stage 02: Prediction")
            
            success_02 = self._run_script("02_prediction.py", self.output_folder)
            print(f"🔍 DEBUG run_stage2_and_3: success_02 = {success_02}")
            
            # Si fue cancelado, no emitir error
            if self._cancelled:
                print("🛑 Análisis cancelado durante Stage 02")
                return
            
            if not success_02:
                print("🔍 DEBUG run_stage2_and_3: Stage 02 falló, emitiendo error")
                self.error.emit("❌ Error en Stage 02: Prediction")
                return
            
            # Verificar cancelación antes de Stage 03
            if self._cancelled:
                print("🛑 Análisis cancelado antes de ejecutar Stage 03")
                return
            
            # Ejecutar Stage 03: Pareto Analyzer
            self.current_stage = '03_pareto_analyzer'
            self.status_updated.emit("🔧 Ejecutando Stage 03: Pareto Analyzer...")
            self.progress_updated.emit(90, "Stage 03: Pareto Analyzer")
            
            success_03 = self._run_script("03_pareto_analyzer.py", self.output_folder)
            print(f"🔍 DEBUG run_stage2_and_3: success_03 = {success_03}")
            
            # Si fue cancelado, no emitir error
            if self._cancelled:
                print("🛑 Análisis cancelado durante Stage 03")
                return
            
            if not success_03:
                print("🔍 DEBUG run_stage2_and_3: Stage 03 falló, emitiendo error")
                self.error.emit("❌ Error en Stage 03: Pareto Analyzer")
                return
            
            # Análisis completado
            self.progress_updated.emit(100, "Análisis completado")
            self.status_updated.emit("✅ Análisis no lineal completado exitosamente")
            
            # Guardar datos de resultados en JSON
            self._save_analysis_results_json()
            
            # Buscar gráficos de Pareto
            pareto_plots_folder = os.path.join(self.output_folder, "05_パレート解", "pareto_plots")
            prediction_output_file = os.path.join(self.output_folder, "04_予測", "Prediction_output.xlsx")
            
            # DEBUG: Verificar rutas
            print(f"🔍 DEBUG nonlinear_worker: output_folder = {self.output_folder}", flush=True)
            print(f"🔍 DEBUG nonlinear_worker: pareto_plots_folder = {pareto_plots_folder}", flush=True)
            print(f"🔍 DEBUG nonlinear_worker: prediction_output_file = {prediction_output_file}", flush=True)
            print(f"🔍 DEBUG nonlinear_worker: pareto_plots_folder exists = {os.path.exists(pareto_plots_folder)}", flush=True)
            print(f"🔍 DEBUG nonlinear_worker: prediction_output_file exists = {os.path.exists(prediction_output_file)}", flush=True)
            
            # Verificar si existen archivos en la carpeta de gráficos
            if os.path.exists(pareto_plots_folder):
                graph_files = [f for f in os.listdir(pareto_plots_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
                print(f"🔍 DEBUG nonlinear_worker: gráficos encontrados = {len(graph_files)}", flush=True)
                if graph_files:
                    print(f"🔍 DEBUG nonlinear_worker: primeros gráficos = {graph_files[:3]}", flush=True)
            
            results_final = {
                'stage': 'completed',
                'output_folder': self.output_folder,
                'all_stages_completed': True,
                'pareto_plots_folder': pareto_plots_folder,
                'prediction_output_file': prediction_output_file
            }
            
            print("🔍 DEBUG run_stage2_and_3: Emitiendo señal finished", flush=True)
            print(f"🔍 DEBUG run_stage2_and_3: results_final = {results_final}", flush=True)
            self.finished.emit(results_final)
            print("🔍 DEBUG run_stage2_and_3: Señal finished emitida", flush=True)
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error continuando análisis: {str(e)}\n{traceback.format_exc()}"
            print("🔍 DEBUG run_stage2_and_3: EXCEPCIÓN CAPTURADA")
            print(error_msg)
            print(f"🔍 DEBUG run_stage2_and_3: Emitiendo señal error")
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
        max_wait_time = 300  # Máximo 5 minutos esperando que aparezca el archivo
        wait_interval = 0.5  # Verificar cada 0.5 segundos
        elapsed_time = 0
        
        # Esperar a que el archivo exista
        while not os.path.exists(json_path) and elapsed_time < max_wait_time:
            time.sleep(wait_interval)
            elapsed_time += wait_interval
        
        if not os.path.exists(json_path):
            self.console_output.emit(f"⚠️ Archivo JSON no encontrado: {json_path}")
            return
        
        # Leer el archivo en tiempo real (reabriendo cada vez para evitar problemas de bloqueo)
        try:
            # Primero, leer todo el contenido existente
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            if 'message' in entry:
                                self.console_output.emit(entry['message'])
                        except json.JSONDecodeError:
                            continue
                # Obtener el tamaño actual del archivo después de leerlo
                last_position = os.path.getsize(json_path)
            
            # Leer nuevas líneas mientras el proceso está corriendo
            while not self._json_reader_stop.is_set():
                time.sleep(0.1)  # Polling cada 100ms
                
                # Verificar si el archivo creció
                if os.path.exists(json_path):
                    current_size = os.path.getsize(json_path)
                    if current_size > last_position:
                        # Reabrir el archivo y leer solo las nuevas líneas
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
            self.console_output.emit(f"⚠️ Error leyendo JSON: {e}")
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
        
        # Si el script no está en la carpeta de salida, usar el del directorio actual
        if not os.path.exists(script_path):
            script_path = script_name
            if not os.path.exists(script_path):
                print(f"❌ Script no encontrado: {script_name}")
                self.console_output.emit(f"❌ Script no encontrado: {script_name}")
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
                # Fallback: buscar site-packages manualmente
                venv_lib = Path(sys.executable).parent.parent / "Lib" / "site-packages"
                if venv_lib.exists():
                    site_packages_paths.append(str(venv_lib))
            
            # Construir PYTHONPATH
            pythonpath_parts = [str(python_code_folder)]
            pythonpath_parts.extend(site_packages_paths)
            
            # Agregar PYTHONPATH existente si hay
            existing_pythonpath = env.get("PYTHONPATH", "")
            if existing_pythonpath:
                pythonpath_parts.append(existing_pythonpath)
            
            # Usar separador correcto según el sistema operativo
            separator = ";" if sys.platform == "win32" else ":"
            pythonpath = separator.join(pythonpath_parts)
            
            env["PYTHONPATH"] = pythonpath
            
            # Obtener ruta del JSON de log
            json_log_path = self._get_json_log_path(working_dir)
            
            # Ejecutar script
            self.console_output.emit(f"🔧 Ejecutando: {script_path}")
            self.console_output.emit(f"📁 Working directory: {working_dir}")
            self.console_output.emit(f"📁 PYTHONPATH: {pythonpath}")
            self.console_output.emit(f"📝 JSON log: {json_log_path}")
            
            # Reiniciar el evento de parada del lector JSON
            self._json_reader_stop.clear()
            
            # Iniciar hilo para leer JSON en tiempo real
            json_reader_thread = threading.Thread(
                target=self._read_json_log,
                args=(json_log_path,),
                daemon=True
            )
            json_reader_thread.start()
            
            # Ejecutar script con Popen para poder leer salida en tiempo real
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
            
            # Guardar referencia al proceso para poder cancelarlo
            self._current_process = process
            
            # Event para detener los threads de lectura de forma segura
            stop_reading = threading.Event()
            self._stop_reading = stop_reading
            
            # Leer stdout y stderr en tiempo real (el script original no genera JSON)
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
                    print("🛑 Cancelando proceso...")
                    self.console_output.emit("🛑 Cancelando proceso...")
                    try:
                        process.terminate()  # Intentar terminar suavemente
                        # Esperar un poco para que termine (polling)
                        for _ in range(20):  # Esperar hasta 2 segundos (20 * 0.1)
                            if process.poll() is not None:
                                break
                            time.sleep(0.1)
                        
                        # Si aún no terminó, forzar kill
                        if process.poll() is None:
                            print("⚠️ Proceso no terminó, forzando cierre...")
                            self.console_output.emit("⚠️ Proceso no terminó, forzando cierre...")
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
                time.sleep(0.1)  # Esperar un poco antes de verificar de nuevo
            
            returncode = process.returncode
            
            # Detener los threads de lectura antes de cerrar pipes
            stop_reading.set()
            stdout_thread.join(timeout=1.0)  # Esperar máximo 1 segundo
            stderr_thread.join(timeout=1.0)  # Esperar máximo 1 segundo
            
            # Limpiar referencia al proceso
            self._current_process = None
            self._stop_reading = None
            
            # Detener el lector JSON
            self._json_reader_stop.set()
            json_reader_thread.join(timeout=1.0)  # Esperar máximo 1 segundo
            
            # Cerrar pipes de forma segura (ya no hay threads leyendo)
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
                # ✅ Intentar leer cualquier salida restante de stderr para ver el error
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
            # Solo parsear si estamos en el stage 01 (model_builder)
            if self.current_stage != '01_model_builder':
                return
            
            # Detectar análisis de datos completado
            if 'データ分析完了' in line or 'データ分析が完了しました' in line:
                self.data_analysis_completed = True
                self.current_task = 'dcv'
                # Emitir progreso actualizado
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
            
            # Detectar inicio de análisis de datos
            if 'データ分析開始' in line:
                self.current_task = 'data_analysis'
                # Emitir progreso actualizado
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
            
            # Detectar entrenamiento del modelo final
            if '最終モデル訓練' in line or '最終モデル訓練（全データ' in line:
                self.final_model_training = True
                self.current_task = 'final_model'
                # Emitir progreso actualizado
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
            
            # Detectar análisis SHAP
            if 'SHAP' in line and ('分析' in line or 'analyze' in line.lower()):
                self.shap_analysis = True
                self.current_task = 'shap'
                # Emitir progreso actualizado
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
            
            # Detectar guardado completado
            if '推論用バンドル保存' in line or '✅ 推論用バンドル保存' in line:
                self.saving_completed = True
                self.current_task = 'saving'
                # Emitir progreso actualizado
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
            
            # Detectar inicio de nueva pasada (target): "Double Cross-Validation: {target_name}" o "処理中: {target}"
            # Priorizar "Double Cross-Validation" porque aparece después de "処理中"
            pass_match = re.search(r'Double\s+Cross-Validation:\s+(\w+)', line, re.IGNORECASE)
            target_name = None
            if pass_match:
                target_name = pass_match.group(1)
            else:
                # Si no se encuentra "Double Cross-Validation", buscar "処理中"
                pass_match = re.search(r'処理中:\s+(\w+)', line)
                if pass_match:
                    target_name = pass_match.group(1)
            
            if target_name and target_name != self.last_detected_target:
                # Nuevo target detectado - incrementar pasada
                self.last_detected_target = target_name
                self.current_pass += 1
                self.current_fold = 0  # Reset fold cuando cambia la pasada
                self.current_trial = 0  # ✅ Reset contador de trials completados cuando cambia la pasada
                self.current_model = 0  # Reset model cuando cambia la pasada
                self.completed_trials_in_current_fold = set()  # ✅ Reset set de trials completados
                self.final_model_training = False  # Reset para nueva pasada
                self.shap_analysis = False  # Reset para nueva pasada
                self.saving_completed = False  # Reset para nueva pasada
                self.current_task = 'dcv'  # Volver a DCV para nueva pasada
                # Emitir progreso actualizado con la pasada correcta
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
            
            # Detectar Outer Fold: "--- Outer Fold X/Y ---"
            fold_match = re.search(r'---\s*Outer\s+Fold\s+(\d+)/(\d+)\s*---', line, re.IGNORECASE)
            if fold_match:
                self.current_fold = int(fold_match.group(1))
                self.total_folds = int(fold_match.group(2))
                self.current_trial = 0  # ✅ Reset contador de trials completados cuando cambia el fold
                self.current_model = 0  # Reset model cuando cambia el fold
                self.completed_trials_in_current_fold = set()  # ✅ Reset set de trials completados
                # Emitir progreso actualizado
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
            
            # Detectar inicio de optimización de modelo: "🔍 {model_name} 最適化中..."
            model_match = re.search(r'🔍\s+(\w+)\s+最適化中', line)
            if model_match:
                self.current_model += 1
                # ✅ NO resetear contador de trials cuando cambia el modelo dentro del mismo fold
                # El contador de trials debe continuar a través de todos los modelos en el mismo fold
                # Solo se resetea cuando cambia el fold
                # Emitir progreso actualizado para mostrar el cambio de modelo
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
            
            # ✅ Formato de barra de progreso de Optuna: buscar "X/Y" (prioritario porque muestra trials completados)
            # Ejemplo: "Best trial: 34. Best value: 4.04966: 100%|██████████| 50/50 [04:34<00:00,  2.34s/it]"
            # El formato "X/Y" muestra: X = trials completados, Y = total trials
            trial_progress_match = re.search(r'(\d+)/(\d+)\s*\[', line)
            if trial_progress_match:
                trials_completed = int(trial_progress_match.group(1))  # Número de trials completados (contador incremental)
                trial_total = int(trial_progress_match.group(2))  # Total de trials
                
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
                    # Total de trials acumulados
                    self.accumulated_trial_total = self.total_passes * self.total_folds * trials_per_fold
                else:
                    # Fallback: usar valores locales si no hay suficiente información
                    self.accumulated_trial_current = trials_completed
                    self.accumulated_trial_total = trial_total
                
                # Emitir progreso actualizado
                self.progress_detailed.emit(
                    self.current_trial,  # Trials completados en fold actual (para mostrar: 1/50, 2/50, etc.)
                    self.total_trials,   # Total de trials por fold
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
        """Cancela la ejecución del análisis"""
        print("🛑 Cancelando análisis no lineal...")
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
                    print("⚠️ Proceso no terminó, forzando kill...")
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
            print("🛑 Solicitando terminación del thread del worker...")
            self.quit()
        
        print("✅ Cancelación completada")
    
    def _save_config_file(self, config_file_path):
        """
        Guarda el archivo de configuración personalizada.
        Copia config.py completo y reemplaza solo los valores modificados desde la UI.
        """
        # Buscar config.py en el directorio actual o en el directorio del script
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
        
        # Leer config.py completo
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
                # Para diccionarios multilínea, buscar desde el patrón hasta el cierre de llaves
                # Capturar la indentación original y comentario si existe
                # El patrón debe capturar todo el diccionario, incluyendo las llaves
                dict_pattern = rf'^(\s*)({re.escape(pattern_clean)}\s*=\s*{{)(.*?)(^\s*}})(\s*#.*)?$'
                def dict_replacer(match):
                    indent = match.group(1)
                    comment = match.group(5) if match.group(5) else ''
                    if comment:
                        comment = ' ' + comment.strip()  # Asegurar espacio antes del comentario
                    # new_value ya contiene el diccionario completo con llaves {}
                    # Solo necesitamos agregar la indentación a cada línea
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
                # Para listas multilínea, buscar desde el patrón hasta el cierre de corchetes
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
                    # String: buscar el patrón y reemplazar el valor entre comillas
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
                    # Número o booleano: buscar el patrón y reemplazar el valor
                    # Capturar la indentación original y comentario si existe
                    # Manejar casos como "50#" (sin espacio) o "50 # comentario" (con espacio)
                    pattern_regex = rf'^(\s*)({re.escape(pattern_clean)}\s*=\s*)([^\n]+)$'
                    def value_replacer(match):
                        indent = match.group(1)
                        full_line = match.group(3).strip()
                        
                        # Separar el valor del comentario
                        # Buscar # que puede estar pegado o con espacio
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
        # Usar analysis_df.xlsx como archivo de entrada para 01_model_builder
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
            # Verificar que el reemplazo funcionó
            if f"N_TRIALS = {n_trials}" in config_content or f"N_TRIALS = {n_trials} #" in config_content:
                print(f"✅ N_TRIALS reemplazado correctamente en config_custom.py")
            else:
                print(f"⚠️ ADVERTENCIA: N_TRIALS podría no haberse reemplazado correctamente")
        
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
            # Leer las listas originales de config.py para determinar el tipo de cada característica
            from config import Config as OriginalConfig
            
            # Filtrar cada lista de tipos para que solo contenga características seleccionadas
            continuous_selected = [f for f in OriginalConfig.CONTINUOUS_FEATURES if f in features_list]
            discrete_selected = [f for f in OriginalConfig.DISCRETE_FEATURES if f in features_list]
            binary_selected = [f for f in OriginalConfig.BINARY_FEATURES if f in features_list]
            integer_selected = [f for f in OriginalConfig.INTEGER_FEATURES if f in features_list]
            
            print(f"🔍 Características seleccionadas: {features_list}")
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
            # Leer MANDATORY_FEATURES original de config.py
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
                print(f"🔍 MANDATORY_FEATURES vacía (ninguna característica obligatoria seleccionada)")
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
        
        # Agregar comentario al inicio indicando que es un archivo generado
        header_comment = "# Configuración personalizada para análisis no lineal\n# Generado automáticamente - Basado en config.py\n# Solo se modifican los valores configurados desde la UI\n\n"
        
        # Verificar si ya tiene el comentario
        if not config_content.startswith("# Configuración personalizada"):
            config_content = header_comment + config_content
        
        # Escribir archivo
        with open(config_file_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"✅ Configuración guardada: {config_file_path}")
        
        # Debug: Verificar que N_TRIALS está en el archivo guardado
        if 'N_TRIALS' in config_content:
            # Buscar la línea de N_TRIALS
            for line in config_content.split('\n'):
                if 'N_TRIALS' in line and '=' in line:
                    print(f"🔍 Línea N_TRIALS en config_custom.py: {line.strip()}")
                    break
        else:
            print(f"⚠️ ADVERTENCIA: N_TRIALS no encontrado en config_custom.py después de guardar")
    
    def _find_graphs(self, output_folder):
        """Busca gráficos generados en la carpeta de salida"""
        graph_paths = []
        
        # Buscar en subcarpetas comunes
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
        
        print(f"📊 Encontrados {len(graph_paths)} gráficos")
        return graph_paths
    
    def _save_analysis_results_json(self):
        """
        Guarda los datos de resultados del análisis en un archivo JSON
        para facilitar la lectura posterior
        """
        try:
            # Ruta donde guardar el JSON (directamente en la carpeta de resultados)
            result_folder = os.path.join(self.output_folder, '03_学習結果')
            
            if not os.path.exists(result_folder):
                print(f"⚠️ Carpeta de resultados no encontrada: {result_folder}")
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
                    # Crear string con rango de algunas columnas principales
                    range_parts = []
                    for col in numeric_cols[:5]:  # Primeras 5 columnas numéricas
                        range_parts.append(f"{col}: [{min_vals[col]:.2f}, {max_vals[col]:.2f}]")
                    data_range = "; ".join(range_parts)
                    if len(numeric_cols) > 5:
                        data_range += f" ... (+{len(numeric_cols) - 5} más)"
            
            # Obtener filters_applied desde config_values
            # Guardar como lista para que pueda ser leída después
            filters_applied = self.config_values.get('filters_applied', [])
            if not filters_applied or filters_applied == []:
                filters_applied = []
            
            # Extraer información de modelos y métricas CV desde dcv_results.pkl
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
                                # Extraer información del modelo
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
                                
                                # Extraer parámetros del modelo
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
                                
                                # Extraer información de fold_results si está disponible
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
                                
                        print(f"✅ Extraídos {models_trained} modelos con métricas CV desde dcv_results.pkl")
                except Exception as e:
                    print(f"⚠️ Error leyendo dcv_results.pkl para extraer modelos: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Calcular tiempo de análisis (si está disponible)
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
            
            # Crear diccionario con los datos
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
            
            # Guardar en JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"✅ Datos de análisis guardados en: {json_path}")
            
        except Exception as e:
            print(f"⚠️ Error guardando datos de análisis en JSON: {e}")
            import traceback
            traceback.print_exc()
