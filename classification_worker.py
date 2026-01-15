"""
Worker para ejecutar análisis de clasificación (bunrui kaiseki) en un thread separado
Ejecuta Run_pipeline_ver3.3_20250914.py
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
    """Worker que ejecuta el análisis de clasificación en un thread separado"""
    
    # Señales para comunicación con la GUI
    progress_updated = Signal(int, str)  # (value, message)
    status_updated = Signal(str)  # message
    finished = Signal(dict)  # results dict
    error = Signal(str)  # error message
    console_output = Signal(str)  # mensaje de consola
    file_selection_requested = Signal(str)  # (initial_path) - solicita selección de archivo
    
    def __init__(self, filtered_df, project_folder, parent=None, config_values=None, selected_brush=None, selected_material=None, selected_wire_length=None):
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
        selected_brush : str, optional
            Tipo de cepillo seleccionado (A11, A21, o A32) para Prediction_input.xlsx
        selected_material : str, optional
            Material seleccionado (Steel, Alum) para Prediction_input.xlsx
        selected_wire_length : int, optional
            Longitud de alambre seleccionada (30-75mm) para Prediction_input.xlsx
        """
        super().__init__(parent)
        self.filtered_df = filtered_df
        self.project_folder = project_folder
        self.config_values = config_values or {}
        self.selected_brush = selected_brush or "A13"  # Por defecto A13
        self.selected_material = selected_material or "Steel"  # Por defecto Steel
        self.selected_wire_length = selected_wire_length or 75  # Por defecto 75
        self.output_folder = None
        self._cancelled = False
        self._current_process = None
        self._json_reader_stop = threading.Event()
        self._stop_reading = None
        self._selected_file_path = None  # Para almacenar el archivo seleccionado por el usuario
        self._file_selection_event = threading.Event()  # Evento para sincronizar selección de archivo
        
        # Estado del progreso para parsing (similar a nonlinear_worker)
        self.current_fold = 0
        self.total_folds = self.config_values.get('OUTER_SPLITS', 10)
        self.current_trial = 0
        self.total_trials = self.config_values.get('N_TRIALS_INNER', 50)
        self.current_model = 0
        self.total_models = len(self.config_values.get('MODELS_TO_USE', ['lightgbm']))
        
        # Estados de tareas
        self.model_comparison_completed = False
        self.multiobjective_completed = False
        self.dcv_training = False
        self.prediction_completed = False
        self.evaluation_completed = False
        self.current_task = 'initialization'  # initialization, model_comparison, multiobjective, dcv, prediction, evaluation
    
    def cancel(self):
        """Cancela la ejecución del análisis"""
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
        """Ejecuta el análisis de clasificación"""
        start_time = time.time()
        
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
                
                # Buscar resultados generados
                results = self._find_results()
                
                # Emitir resultados como carga existente
                results_existing = {
                    'output_folder': self.output_folder,
                    'analysis_duration': 0,  # No hay duración para análisis existente
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
                
                # Emitir finished para que la GUI muestre los resultados existentes
                self.finished.emit(results_existing)
                return
            
            # Verificar cancelación
            if self._cancelled:
                return
            
            # Crear carpeta de salida 05_分類
            self.status_updated.emit("📁 Creando carpeta de salida...")
            classification_folder = os.path.join(self.project_folder, "05_分類")
            os.makedirs(classification_folder, exist_ok=True)
            
            # Crear subcarpeta con timestamp - esta será la carpeta de salida directa
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = os.path.join(classification_folder, f"分類解析結果_{timestamp}")
            os.makedirs(self.output_folder, exist_ok=True)
            
            # NO copiar ml_modules ni Run_pipeline - usar los del .venv directamente
            # Buscar ml_modules en C:\Users\xebec0176\Desktop\0.00sec\.venv\ml_modules
            script_dir = Path(__file__).parent.absolute()
            venv_ml_modules = script_dir / "ml_modules"
            
            # Si no está en el directorio del script, buscar en el directorio padre (.venv)
            if not venv_ml_modules.exists() or not (venv_ml_modules / "models_cls.py").exists():
                venv_ml_modules = script_dir.parent / "ml_modules"
            
            # Verificar que ml_modules existe
            if not venv_ml_modules.exists() or not (venv_ml_modules / "models_cls.py").exists():
                self.error.emit(f"❌ ml_modules no encontrado en {venv_ml_modules}")
                return
            
            print(f"✅ ml_modules encontrado: {venv_ml_modules}")
            
            # Buscar Run_pipeline_ver3.3_20250914.py en .venv
            venv_pipeline_script = script_dir / "Run_pipeline_ver3.3_20250914.py"
            if not venv_pipeline_script.exists():
                venv_pipeline_script = script_dir.parent / "Run_pipeline_ver3.3_20250914.py"
            
            if not venv_pipeline_script.exists():
                self.error.emit(f"❌ Run_pipeline_ver3.3_20250914.py no encontrado en {venv_pipeline_script}")
                return
            
            print(f"✅ Pipeline script encontrado: {venv_pipeline_script}")
            
            # Crear carpeta 00_データセット en la carpeta de salida
            data_folder = os.path.join(self.output_folder, "00_データセット")
            os.makedirs(data_folder, exist_ok=True)
            
            # Guardar datos filtrados en 00_データセット
            self.status_updated.emit("💾 Guardando datos filtrados...")
            # Usar fecha actual para el nombre del archivo
            from datetime import datetime
            date_str = datetime.now().strftime("%Y%m%d")
            input_filename = f"{date_str}_総実験データ.xlsx"
            input_file = os.path.join(data_folder, input_filename)
            self.filtered_df.to_excel(input_file, index=False)
            print(f"✅ Datos guardados: {input_file}")
            
            # Guardar el nombre del archivo para usarlo en la configuración
            self.input_filename = input_filename
            
            # Buscar y procesar archivo 未実験データ.xlsx del proyecto
            self.status_updated.emit("📋 Procesando archivo de predicción...")
            predict_input_file = self._create_prediction_input_file(data_folder)
            if not predict_input_file:
                self.error.emit("❌ No se pudo crear Prediction_input.xlsx")
                return
            
            print(f"✅ Archivo de predicción creado: {predict_input_file}")
            
            # Verificar cancelación
            if self._cancelled:
                return
            
            # Crear archivo de configuración temporal en la carpeta de salida
            self.status_updated.emit("⚙️ Creando configuración temporal...")
            # El archivo de configuración se guarda directamente en output_folder/config_cls.py
            config_file = self._create_temp_config()
            
            if config_file and os.path.exists(config_file):
                print(f"✅ Configuración creada: {config_file}")
            
            # Usar el script original del .venv (no copiado)
            pipeline_script = str(venv_pipeline_script)
            
            # Verificar cancelación
            if self._cancelled:
                return
            
            # Ejecutar el pipeline
            self.status_updated.emit("🔧 Ejecutando pipeline de clasificación...")
            self.progress_updated.emit(20, "Pipeline実行中...")
            
            success = self._run_pipeline(pipeline_script, self.output_folder, config_file)
            
            if self._cancelled:
                return
            
            if not success:
                self.error.emit("❌ Error ejecutando el pipeline de clasificación")
                return
            
            # Buscar resultados generados
            self.status_updated.emit("📊 Buscando resultados...")
            results = self._find_results()
            
            # Calcular tiempo total
            end_time = time.time()
            analysis_duration = end_time - start_time
            
            results['output_folder'] = self.output_folder
            results['analysis_duration'] = analysis_duration
            results['project_folder'] = self.project_folder
            results['load_existing'] = False  # No es carga existente, es análisis nuevo
            
            self.progress_updated.emit(100, "分析完了")
            self.status_updated.emit("✅ 分類分析が完了しました")
            
            self.finished.emit(results)
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error en análisis de clasificación: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.error.emit(error_msg)
    
    def _create_temp_config(self):
        """Crea un archivo de configuración temporal basado en config_values"""
        # El pipeline busca config_cls.py en ml_modules, así que creamos
        # un 99_ml_modules en la carpeta de salida solo con config_cls.py
        ml_modules_dst = Path(self.output_folder) / "99_ml_modules"
        ml_modules_dst.mkdir(parents=True, exist_ok=True)
        
        config_file = ml_modules_dst / "config_cls.py"
        
        # Crear carpeta 99_-----------------
        separator_folder = Path(self.output_folder) / "99_-----------------"
        separator_folder.mkdir(parents=True, exist_ok=True)
        
        # También crear ml_modules como symlink a 99_ml_modules para compatibilidad con el pipeline
        # El pipeline busca BASE / "ml_modules", así que necesitamos crear este symlink
        ml_modules_alias = Path(self.output_folder) / "ml_modules"
        if not ml_modules_alias.exists():
            try:
                # En Windows, intentar crear symlink (puede requerir privilegios)
                if hasattr(os, 'symlink'):
                    os.symlink("99_ml_modules", ml_modules_alias, target_is_directory=True)
                    print(f"✅ Symlink creado: {ml_modules_alias} -> 99_ml_modules")
                else:
                    # Si no hay symlink, copiar solo config_cls.py a ml_modules también
                    ml_modules_fallback = Path(self.output_folder) / "ml_modules"
                    ml_modules_fallback.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(config_file, ml_modules_fallback / "config_cls.py")
                    print(f"✅ config_cls.py copiado también a ml_modules para compatibilidad")
            except Exception as e:
                # Si falla el symlink, copiar solo config_cls.py
                ml_modules_fallback = Path(self.output_folder) / "ml_modules"
                ml_modules_fallback.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(config_file, ml_modules_fallback / "config_cls.py")
                print(f"⚠️ No se pudo crear symlink, copiando config_cls.py a ml_modules: {e}")
        
        # Leer el archivo config_cls.py original como plantilla
        config_cls_path = self._find_config_cls()
        config_content = ""
        
        if config_cls_path and os.path.exists(config_cls_path):
            with open(config_cls_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
        
        # Si no se encuentra, crear uno básico
        if not config_content:
            config_content = self._get_default_config_content()
        
        # Modificar los valores según config_values
        modified_content = self._modify_config_content(config_content, self.config_values)
        
        # Escribir archivo temporal
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"✅ Configuración temporal creada: {config_file}")
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
    """Configuración temporal para clasificación"""
    pass
'''
    
    def _modify_config_content(self, content, config_values):
        """Modifica el contenido de config_cls.py según config_values"""
        # Esta función modifica los valores en el contenido del archivo
        # Por simplicidad, crearemos un archivo que sobrescriba los valores
        
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
        # Usar el nombre del archivo con fecha actual
        input_filename = getattr(self, 'input_filename', None)
        if not input_filename:
            from datetime import datetime
            date_str = datetime.now().strftime("%Y%m%d")
            input_filename = f"{date_str}_総実験データ.xlsx"
        
        modifications.append(f'    DATA_FOLDER: str = "00_データセット"')
        modifications.append(f'    INPUT_FILE: str = "{input_filename}"')
        modifications.append(f'    PREDICT_INPUT_FILE: str = "Prediction_input.xlsx"')
        # Cambiar PARENT_FOLDER_TEMPLATE a "." para que no cree carpeta intermedia
        modifications.append(f'    PARENT_FOLDER_TEMPLATE: str = "."')
        
        # Crear contenido final
        # Reemplazar valores existentes en lugar de solo agregar
        final_content = content
        
        # Reemplazar DATA_FOLDER si existe
        import re
        # Buscar y reemplazar DATA_FOLDER
        final_content = re.sub(
            r'(\s+DATA_FOLDER:\s*str\s*=\s*)"[^"]*"',
            r'\1"00_データセット"',
            final_content
        )
        
        # Reemplazar INPUT_FILE si existe (usar el nombre del archivo con fecha actual)
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
        
        # Reemplazar PARENT_FOLDER_TEMPLATE para que no cree carpeta intermedia
        final_content = re.sub(
            r'(\s+PARENT_FOLDER_TEMPLATE:\s*str\s*=\s*)"[^"]*"',
            r'\1"."',
            final_content
        )
        
        # Reemplazar PARENT_FOLDER_TEMPLATE para que no cree carpeta intermedia
        final_content = re.sub(
            r'(\s+PARENT_FOLDER_TEMPLATE:\s*str\s*=\s*)"[^"]*"',
            r'\1"."',
            final_content
        )
        
        # Agregar modificaciones al final de la clase
        if "class ConfigCLS:" in final_content:
            # Insertar modificaciones antes del último método o al final de la clase
            # Buscar el último @classmethod o método y agregar antes
            lines = final_content.split('\n')
            insert_pos = len(lines)
            
            # Buscar el final de la clase (última línea antes de una línea vacía o fuera de la clase)
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
            # Si no hay clase, crear una básica
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
                # No se encontró el archivo, pedir al usuario que lo seleccione
                self.console_output.emit(f"⚠️ No se encontró archivo *_未実験データ.xlsx en {project_path}")
                self.status_updated.emit("ファイル選択待ち...")
                
                # Resetear variables de selección
                self._selected_file_path = None
                self._file_selection_event.clear()
                
                # Emitir señal para que la GUI muestre el diálogo
                self.file_selection_requested.emit(str(project_path))
                
                # Esperar a que el usuario seleccione el archivo (máximo 5 minutos)
                max_wait = 300  # 5 minutos en segundos
                if self._file_selection_event.wait(timeout=max_wait):
                    # Usuario seleccionó archivo
                    if self._selected_file_path:
                        unexperimented_file = Path(self._selected_file_path)
                        print(f"📋 Archivo seleccionado por usuario: {unexperimented_file}")
                    else:
                        self.error.emit("❌ ファイルが選択されませんでした。")
                        return None
                else:
                    # Timeout - usuario no seleccionó archivo a tiempo
                    self.error.emit("❌ ファイル選択がタイムアウトしました。")
                    return None
            else:
                # Usar el primer archivo encontrado
                unexperimented_file = unexperimented_files[0]
                print(f"📋 Archivo 未実験データ encontrado: {unexperimented_file}")
            
            # Leer el archivo
            self.status_updated.emit("ファイル読み込み中...")
            df_predict = pd.read_excel(unexperimented_file)
            
            # Validar que el archivo tiene las columnas necesarias
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
            
            # Validar que el archivo tiene al menos una fila de datos
            if len(df_predict) == 0:
                self.error.emit(f"❌ 選択されたファイルにデータがありません: {unexperimented_file}")
                return None
            
            print(f"✅ Archivo validado correctamente. Columnas: {list(df_predict.columns)}")
            print(f"✅ Filas de datos: {len(df_predict)}")
            
            # Agregar columnas A13, A11, A21, A32
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
            self.console_output.emit(f"❌ Error creando Prediction_input.xlsx: {str(e)}")
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
        """Ejecuta el pipeline de clasificación"""
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
            
            # Buscar ml_modules en .venv (no copiado)
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
            
            # También crear un symlink o alias ml_modules -> 99_ml_modules para compatibilidad
            # En Windows, creamos un symlink si es posible, sino copiamos solo config_cls.py
            ml_modules_alias = Path(working_dir) / "ml_modules"
            if not ml_modules_alias.exists() and ml_modules_in_workdir.exists():
                try:
                    # Intentar crear symlink (requiere permisos en Windows)
                    if hasattr(os, 'symlink'):
                        os.symlink(ml_modules_in_workdir, ml_modules_alias, target_is_directory=True)
                        python_paths.append(str(ml_modules_alias))
                except:
                    # Si falla, al menos agregar 99_ml_modules al path
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
            
            # 6. Agregar site-packages
            import site
            for site_pkg in site.getsitepackages():
                if os.path.exists(site_pkg):
                    python_paths.append(site_pkg)
            
            # 7. Agregar PYTHONPATH existente si hay
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
            
            # Verificar que config_cls.py existe en 99_ml_modules dentro de working_dir
            ml_modules_in_workdir = Path(working_dir) / "99_ml_modules"
            config_check = ml_modules_in_workdir / "config_cls.py"
            if not config_check.exists():
                self.console_output.emit(f"❌ Error: config_cls.py no encontrado en {ml_modules_in_workdir}")
                return False
            
            # Verificar que ml_modules del .venv existe
            if not venv_ml_modules.exists() or not (venv_ml_modules / "models_cls.py").exists():
                self.console_output.emit(f"❌ Error: ml_modules no encontrado en {venv_ml_modules}")
                return False
            
            # Usar el script original del .venv (no copiado)
            script_to_run = script_path
            self.console_output.emit(f"📝 Usando script del .venv: {script_to_run}")
            
            # Debug: Verificar estructura antes de ejecutar
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
            
            # Ejecutar script con Popen para poder leer salida en tiempo real
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
            
            # Guardar referencia al proceso para poder cancelarlo
            self._current_process = process
            
            # Event para detener los threads de lectura de forma segura
            stop_reading = threading.Event()
            self._stop_reading = stop_reading
            
            # Leer stdout y stderr en tiempo real usando threads
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
            
            # Cerrar pipes de forma segura (ya no hay threads leyendo)
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
                # Intentar leer cualquier salida restante de stderr para ver el error
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
            # Detectar modelo comparación
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
            
            # Detectar predicción
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
            
            # Detectar análisis de características
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
            # Si hay error en el parsing, no hacer nada (no es crítico)
            pass
    
    def _find_results(self):
        """Busca los resultados generados por el pipeline"""
        results = {
            'result_folders': [],
            'graph_paths': [],
            'model_files': [],
            'evaluation_files': []
        }
        
        # El pipeline crea una carpeta con timestamp
        # Buscar en el directorio de trabajo
        if not os.path.exists(self.output_folder):
            return results
        
        # Buscar carpetas de resultados
        for item in os.listdir(self.output_folder):
            item_path = os.path.join(self.output_folder, item)
            if os.path.isdir(item_path):
                # Verificar si es una carpeta de resultados del pipeline
                if "分類解析結果" in item or "分類" in item:
                    results['result_folders'].append(item_path)
        
        # Buscar archivos de gráficos
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

