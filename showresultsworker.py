from PySide6.QtCore import QObject, QThread, Signal
import os
import shutil
from datetime import datetime
import sqlite3

class ShowResultsWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, project_folder, results_file_path, diameter, material, backup_function, process_function, experiment_info=None):
        super().__init__()
        self.project_folder = project_folder
        self.results_file_path = results_file_path
        self.diameter = diameter
        self.material = material
        self.backup_function = backup_function
        self.process_function = process_function
        self.experiment_info = experiment_info  # Experiment info found
        # ES: Instalación profesional: usar ProgramData\\...\\data\\results.db (y migrar legacy si existe)
        # EN: Pro install: use ProgramData\\...\\data\\results.db (and migrate legacy DB if present)
        # JA: 製品版：ProgramData\\...\\data\\results.db を使用（必要なら旧DBを移行）
        from app_paths import migrate_legacy_db_if_needed
        self.db_path = migrate_legacy_db_if_needed("results.db", shared=True)

    def run(self):
        try:
            # Create database connection in this thread
            print(f"🔍 デバッグ - ShowResultsWorker が使用するDB: {os.path.abspath(self.db_path)}")
            self.conn = sqlite3.connect(self.db_path, timeout=10)
            
            # ES: ✅ NUEVO: Usar información del experimento si está disponible, o buscar si no
            # EN: ✅ NEW: Use provided experiment info if available; otherwise use defaults
            # JA: ✅ 新規：実験情報があれば使用、無ければデフォルトを使用
            if self.experiment_info:
                print("✅ デバッグ - 提供された実験情報を使用します")
                folder_name = self.experiment_info['folder_name']
                optimization_type = self.experiment_info['optimization_type']
                
                # ES: Extraer número de la carpeta
                # EN: Extract folder number
                # JA: フォルダ番号を抽出
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
                
                print(f"📊 抽出したフォルダー番号: {folder_num}")
                print(f"📁 検出したフォルダー: {folder_name}")
                print(f"🔧 最適化タイプ: {optimization_type}")
            else:
                print("⚠️ デバッグ - 実験情報が提供されていないため、デフォルトを使用します")
                folder_num = 1
                optimization_type = "D最適化"
                print(f"📊 デフォルトのフォルダー番号を使用: {folder_num}")
                print(f"🔧 デフォルトの最適化タイプを使用: {optimization_type}")

            # ES: ✅ NUEVO: Verificar si ya existe un archivo idéntico en 02_実験データ
            # EN: ✅ NEW: Check whether an identical file already exists in 02_実験データ
            # JA: ✅ 新規：02_実験データ に同一ファイルがあるか確認
            print("🔍 デバッグ - 02_実験データ に同一ファイルがあるか確認中...")
            experiment_data_folder = os.path.join(self.project_folder, "02_実験データ")
            
            if os.path.exists(experiment_data_folder):
                # ES: Leer el archivo de resultados actual
                # EN: Read the current results file
                # JA: 現在の結果ファイルを読み込み
                import pandas as pd
                try:
                    current_results_df = pd.read_excel(self.results_file_path)
                    print(f"🔍 デバッグ - 現在のファイルの行数: {len(current_results_df)}")
                    
                    # ES: Buscar archivos Excel en todas las subcarpetas de 02_実験データ
                    # EN: Search Excel files in all subfolders of 02_実験データ
                    # JA: 02_実験データ 配下の全サブフォルダでExcelを検索
                    identical_file_found = False
                    identical_folder = None
                    
                    for subfolder in os.listdir(experiment_data_folder):
                        subfolder_path = os.path.join(experiment_data_folder, subfolder)
                        if os.path.isdir(subfolder_path):
                            print(f"🔍 デバッグ - フォルダー確認: {subfolder}")
                            
                            # ES: Buscar archivos Excel en la subcarpeta
                            # EN: Search Excel files within the subfolder
                            # JA: サブフォルダ内のExcelを検索
                            for file in os.listdir(subfolder_path):
                                if file.endswith(('.xlsx', '.xls', '.xlsm', '.xlsb')):
                                    file_path = os.path.join(subfolder_path, file)
                                    try:
                                        existing_df = pd.read_excel(file_path)
                                        print(f"🔍 デバッグ - 比較: {file}（{len(existing_df)} 行）")
                                        
                                        # Comparar si son idénticos
                                        if current_results_df.equals(existing_df):
                                            print(f"✅ 同一ファイルを検出: {subfolder}/{file}")
                                            identical_file_found = True
                                            identical_folder = subfolder
                                            break
                                    except Exception as e:
                                        print(f"🔍 デバッグ - {file} の読み込み中にエラー: {e}")
                                        continue
                            
                            if identical_file_found:
                                break
                    
                    if identical_file_found:
                        print(f"⚠️ 同一ファイルがすでに存在します: {identical_folder}")
                        print("🛑 フォルダー作成を中止します")
                        
                        # ES: Retornar resultado indicando que ya existe
                        # EN: Return result indicating it already exists
                        # JA: 既存のため結果を返す
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
                        print("✅ 同一ファイルは見つかりませんでした。フォルダー作成を続行します...")
                        
                except Exception as e:
                    print(f"🔍 デバッグ - 結果ファイルの読み込み中にエラー: {e}")
                    print("⚠️ フォルダー作成を続行します...")
            else:
                print("🔍 デバッグ - 02_実験データ が存在しないため作成します...")
            
            # ES: ✅ NUEVO: Solo si NO existe archivo idéntico, crear la carpeta en 02_実験データ
            # EN: ✅ NEW: Only if no identical file exists, create the folder under 02_実験データ
            # JA: ✅ 新規：同一ファイルが無い場合のみ 02_実験データ にフォルダ作成
            print("🔍 デバッグ - 02_実験データ にフォルダーを作成中...")
            now = datetime.now()
            fecha_hora = now.strftime('%Y%m%d_%H%M%S')
            
            # ES: Crear carpeta en 02_実験データ
            # EN: Create folder in 02_実験データ
            # JA: 02_実験データ にフォルダ作成
            experiment_data_folder = os.path.join(self.project_folder, "02_実験データ")
            os.makedirs(experiment_data_folder, exist_ok=True)
            
            new_folder_name = f"{folder_num:03d}_{optimization_type}_{fecha_hora}"
            new_folder_path = os.path.join(experiment_data_folder, new_folder_name)
            os.makedirs(new_folder_path, exist_ok=True)
            print(f"✅ 新しいフォルダーを作成しました: {new_folder_path}")

            # ES: ✅ NUEVO: Hacer backup y actualizar archivo de muestreo
            # EN: ✅ NEW: Backup and update the sampling file
            # JA: ✅ 新規：サンプルファイルをバックアップ＆更新
            print("🔄 バックアップとサンプルファイル更新を開始...")
            backup_result = self.backup_function(self.results_file_path, self.project_folder)

            # ES: Procesar archivo de resultados (線材長 viene del archivo)
            # EN: Process results file (線材長 comes from the file)
            # JA: 結果ファイルを処理（線材長 はファイル由来）
            # Pass the connection created in this thread
            db_upsert_result = self.process_function(
                self.results_file_path, 
                None,  # Brush is always from file (A13/A11/A21/A32)
                self.diameter, 
                self.material,
                self.conn  # Pass the connection created in this thread
            )
            
            # ES: ✅ NUEVO: Verificar el contenido de la base de datos después de importar
            # EN: ✅ NEW: Verify database contents after import
            # JA: ✅ 新規：インポート後にDB内容を確認
            print("🔍 デバッグ - インポート後にDB内容を確認中...")
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM main_results")
            total_count = cursor.fetchone()[0]
            print(f"🔍 デバッグ - DBの総レコード数: {total_count}")
            
            if total_count > 0:
                print("🔍 デバッグ - 直近5件の挿入レコード:")
                cursor.execute("SELECT * FROM main_results ORDER BY id DESC LIMIT 5")
                recent_records = cursor.fetchall()
                
                # ES: Obtener nombres de columnas
                # EN: Get column names
                # JA: 列名を取得
                cursor.execute("PRAGMA table_info(main_results)")
                columns_info = cursor.fetchall()
                column_names = [col[1] for col in columns_info]
                
                for i, record in enumerate(recent_records, 1):
                    print(f"  レコード {i}:")
                    for j, value in enumerate(record):
                        if j < len(column_names):
                            print(f"    {column_names[j]}: {value}")
                    print()
            else:
                print("🔍 デバッグ - DBにレコードがありません")

            # ES: Guardar archivo de resultados con el nombre especificado
            # EN: Save results file using the specified name
            # JA: 指定名で結果ファイルを保存
            fecha = now.strftime('%Y%m%d')
            results_filename = f"実験結果_{optimization_type}_{fecha}.xlsx"
            results_file_path = os.path.join(new_folder_path, results_filename)
            
            # ES: Copiar el archivo de resultados original
            # EN: Copy the original results file
            # JA: 元の結果ファイルをコピー
            shutil.copy2(self.results_file_path, results_file_path)

            # ES: Retornar resultado
            # EN: Return result
            # JA: 結果を返す
            result = {
                'results_file_path': results_file_path,
                'backup_result': backup_result,
                'db_upsert_result': db_upsert_result,
                'optimization_type': optimization_type
            }
            
            # ES: ✅ NUEVO: Emitir señal y también retornar el resultado
            # EN: ✅ NEW: Emit signal and also return the result
            # JA: ✅ 新規：シグナル送信＆戻り値として返す
            self.finished.emit(result)
            return result
            
        except Exception as e:
            import traceback
            self.error.emit(f"❌ 処理中にエラーが発生しました:\n{str(e)}\n\n{traceback.format_exc()}")
        finally:
            # ES: Cerrar la conexión a la base de datos
            # EN: Close database connection
            # JA: DB接続を閉じる
            if hasattr(self, 'conn'):
                self.conn.close() 