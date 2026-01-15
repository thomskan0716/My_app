from PySide6.QtCore import QObject, QThread, Signal
import os
import shutil
from datetime import datetime
import sqlite3

class ShowResultsWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, project_folder, results_file_path, selected_brush, diameter, material, backup_function, process_function, experiment_info=None):
        super().__init__()
        self.project_folder = project_folder
        self.results_file_path = results_file_path
        self.selected_brush = selected_brush
        self.diameter = diameter
        self.material = material
        self.backup_function = backup_function
        self.process_function = process_function
        self.experiment_info = experiment_info  # Información del experimento encontrado
        # ✅ Instalación profesional: usar ProgramData\\...\\data\\results.db (y migrar legacy si existe)
        from app_paths import migrate_legacy_db_if_needed
        self.db_path = migrate_legacy_db_if_needed("results.db", shared=True)

    def run(self):
        try:
            # Create database connection in this thread
            print(f"🔍 Debug - ShowResultsWorker usando base de datos: {os.path.abspath(self.db_path)}")
            self.conn = sqlite3.connect(self.db_path, timeout=10)
            
            # ✅ NUEVO: Usar información del experimento si está disponible, o buscar si no
            if self.experiment_info:
                print("✅ Debug - Usando información de experimento proporcionada")
                folder_name = self.experiment_info['folder_name']
                optimization_type = self.experiment_info['optimization_type']
                
                # Extraer número de la carpeta
                import re
                number_patterns = [
                    r'(\d{3,})',  # Números de 3 o más dígitos
                    r'(\d{2,})',  # Números de 2 o más dígitos
                    r'(\d+)'      # Cualquier número
                ]
                
                folder_num = 1  # Número por defecto
                for pattern in number_patterns:
                    number_match = re.search(pattern, folder_name)
                    if number_match:
                        extracted_number = number_match.group(1)
                        folder_num = int(extracted_number)
                        break
                
                print(f"📊 Número de carpeta extraído: {folder_num}")
                print(f"📁 Carpeta encontrada: {folder_name}")
                print(f"🔧 Tipo de optimización: {optimization_type}")
            else:
                print("⚠️ Debug - No se proporcionó información de experimento, usando valores por defecto")
                folder_num = 1
                optimization_type = "D最適化"
                print(f"📊 Usando número de carpeta por defecto: {folder_num}")
                print(f"🔧 Usando tipo de optimización por defecto: {optimization_type}")

            # ✅ NUEVO: AHORA verificar si ya existe un archivo idéntico en 02_実験データ
            print("🔍 Debug - Verificando si ya existe un archivo idéntico en 02_実験データ...")
            experiment_data_folder = os.path.join(self.project_folder, "02_実験データ")
            
            if os.path.exists(experiment_data_folder):
                # Leer el archivo de resultados actual
                import pandas as pd
                try:
                    current_results_df = pd.read_excel(self.results_file_path)
                    print(f"🔍 Debug - Archivo actual contiene {len(current_results_df)} filas")
                    
                    # Buscar archivos Excel en todas las subcarpetas de 02_実験データ
                    identical_file_found = False
                    identical_folder = None
                    
                    for subfolder in os.listdir(experiment_data_folder):
                        subfolder_path = os.path.join(experiment_data_folder, subfolder)
                        if os.path.isdir(subfolder_path):
                            print(f"🔍 Debug - Revisando carpeta: {subfolder}")
                            
                            # Buscar archivos Excel en la subcarpeta
                            for file in os.listdir(subfolder_path):
                                if file.endswith(('.xlsx', '.xls', '.xlsm', '.xlsb')):
                                    file_path = os.path.join(subfolder_path, file)
                                    try:
                                        existing_df = pd.read_excel(file_path)
                                        print(f"🔍 Debug - Comparando con: {file} ({len(existing_df)} filas)")
                                        
                                        # Comparar si son idénticos
                                        if current_results_df.equals(existing_df):
                                            print(f"✅ Archivo idéntico encontrado en: {subfolder}/{file}")
                                            identical_file_found = True
                                            identical_folder = subfolder
                                            break
                                    except Exception as e:
                                        print(f"🔍 Debug - Error leyendo {file}: {e}")
                                        continue
                            
                            if identical_file_found:
                                break
                    
                    if identical_file_found:
                        print(f"⚠️ Ya existe un archivo idéntico en: {identical_folder}")
                        print("🛑 No se procederá con la creación de carpeta")
                        
                        # Retornar resultado indicando que ya existe
                        result = {
                            'results_file_path': self.results_file_path,
                            'backup_result': {'backup_path': '', 'remaining_rows': 0, 'removed_rows': 0},
                            # ✅ Importante: en este early-exit NO se importa nada a la BBDD.
                            # Incluir db_upsert_result evita que la UI haga fallback contando filas del Excel.
                            'db_upsert_result': {'inserted': 0, 'updated': 0, 'db_backup_path': None},
                            'optimization_type': 'EXISTING',
                            'identical_folder': identical_folder,
                            'skipped_reason': 'identical_file_found',
                        }
                        
                        self.finished.emit(result)
                        return
                    else:
                        print("✅ No se encontró archivo idéntico, procediendo con la creación de carpeta...")
                        
                except Exception as e:
                    print(f"🔍 Debug - Error leyendo archivo de resultados: {e}")
                    print("⚠️ Continuando con la creación de carpeta...")
            else:
                print("🔍 Debug - Carpeta 02_実験データ no existe, procediendo con la creación...")
            
            # ✅ NUEVO: SOLO SI NO EXISTE ARCHIVO IDÉNTICO, crear la carpeta en 02_実験データ
            print("🔍 Debug - Creando carpeta en 02_実験データ...")
            now = datetime.now()
            fecha_hora = now.strftime('%Y%m%d_%H%M%S')
            
            # Crear carpeta en 02_実験データ
            experiment_data_folder = os.path.join(self.project_folder, "02_実験データ")
            os.makedirs(experiment_data_folder, exist_ok=True)
            
            new_folder_name = f"{folder_num:03d}_{optimization_type}_{fecha_hora}"
            new_folder_path = os.path.join(experiment_data_folder, new_folder_name)
            os.makedirs(new_folder_path, exist_ok=True)
            print(f"✅ Creando nueva carpeta: {new_folder_path}")

            # ✅ NUEVO: Hacer backup y actualizar archivo de muestreo
            print("🔄 Iniciando proceso de backup y actualización del archivo de muestreo...")
            backup_result = self.backup_function(self.results_file_path, self.project_folder)

            # Procesar archivo de resultados (線材長 viene del archivo)
            # Pass the connection created in this thread
            db_upsert_result = self.process_function(
                self.results_file_path, 
                self.selected_brush, 
                self.diameter, 
                self.material,
                self.conn  # Pass the connection created in this thread
            )
            
            # ✅ NUEVO: Verificar el contenido de la base de datos después de importar
            print("🔍 Debug - Verificando contenido de la base de datos después de importar...")
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM main_results")
            total_count = cursor.fetchone()[0]
            print(f"🔍 Debug - Total de registros en la base de datos: {total_count}")
            
            if total_count > 0:
                print("🔍 Debug - Últimos 5 registros insertados:")
                cursor.execute("SELECT * FROM main_results ORDER BY id DESC LIMIT 5")
                recent_records = cursor.fetchall()
                
                # Obtener nombres de columnas
                cursor.execute("PRAGMA table_info(main_results)")
                columns_info = cursor.fetchall()
                column_names = [col[1] for col in columns_info]
                
                for i, record in enumerate(recent_records, 1):
                    print(f"  Registro {i}:")
                    for j, value in enumerate(record):
                        if j < len(column_names):
                            print(f"    {column_names[j]}: {value}")
                    print()
            else:
                print("🔍 Debug - No hay registros en la base de datos")

            # Guardar archivo de resultados con el nombre especificado
            fecha = now.strftime('%Y%m%d')
            results_filename = f"実験結果_{optimization_type}_{fecha}.xlsx"
            results_file_path = os.path.join(new_folder_path, results_filename)
            
            # Copiar el archivo de resultados original
            shutil.copy2(self.results_file_path, results_file_path)

            # Retornar resultado
            result = {
                'results_file_path': results_file_path,
                'backup_result': backup_result,
                'db_upsert_result': db_upsert_result,
                'optimization_type': optimization_type
            }
            
            # ✅ NUEVO: Emitir señal y también retornar el resultado
            self.finished.emit(result)
            return result
            
        except Exception as e:
            import traceback
            self.error.emit(f"❌ 処理中にエラーが発生しました:\n{str(e)}\n\n{traceback.format_exc()}")
        finally:
            # Cerrar la conexión a la base de datos
            if hasattr(self, 'conn'):
                self.conn.close() 