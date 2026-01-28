"""ES: Gestor de carpetas para análisis no lineal. Crea carpetas con numeración correlativa y timestamp.
EN: Folder manager for non-linear analysis. Creates folders with sequential numbering and timestamp.
JA: 非線形解析用フォルダ管理。連番＋タイムスタンプでフォルダを作成。"""
import os
import re
from datetime import datetime
from pathlib import Path


class NonlinearFolderManager:
    """ES: Gestiona la creación de carpetas para análisis no lineal
    EN: Manages folder creation for non-linear analysis
    JA: 非線形解析用フォルダ作成を管理"""
    
    def __init__(self, project_folder):
        """ES: Inicializa el gestor de carpetas
        EN: Initialize the folder manager
        JA: フォルダマネージャを初期化
        
        Parameters
        ----------
        project_folder : str
            ES: Carpeta base del proyecto (donde está NOMBRE_DEL_PROYECTO)
            EN: Project base folder (where NOMBRE_DEL_PROYECTO is)
            JA: プロジェクトのベースフォルダ（NOMBRE_DEL_PROYECTO の所在）
        """
        self.project_folder = project_folder
        self.base_folder = os.path.join(project_folder, "04_非線形回帰")
    
    def create_output_folder(self):
        """
        Crea una carpeta con número correlativo y timestamp
        Formato: NUM_FECHA_HORA (ejemplo: 01_20250115_143022)
        
        Returns
        -------
        str
            Ruta completa de la carpeta creada
        """
        # ES: Crear carpeta base si no existe | EN: Create base folder if missing | JA: ベースフォルダが無ければ作成
        os.makedirs(self.base_folder, exist_ok=True)
        
        # ES: Obtener siguiente número correlativo | EN: Get next sequential number | JA: 次の連番を取得
        next_number = self._get_next_correlative_number()
        
        # ES: Obtener timestamp | EN: Get timestamp | JA: タイムスタンプ取得
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ES: Crear nombre de carpeta: NUM_timestamp | EN: Build folder name: NUM_timestamp | JA: フォルダ名（NUM_timestamp）を作成
        folder_name = f"{next_number:02d}_{timestamp}"
        full_path = os.path.join(self.base_folder, folder_name)
        
        # ES: Crear carpeta | EN: Create folder | JA: フォルダ作成
        os.makedirs(full_path, exist_ok=True)
        
        print(f"📁 フォルダーを作成しました: {full_path}")
        return full_path
    
    def _get_next_correlative_number(self):
        """
        Obtiene el siguiente número correlativo basándose en las carpetas existentes
        
        Returns
        -------
        int
            Siguiente número correlativo
        """
        if not os.path.exists(self.base_folder):
            return 1
        
        existing_numbers = []
        
        # ES: Buscar todas las carpetas con patrón NUM_* | EN: Find all folders matching NUM_* | JA: NUM_* パターンのフォルダを探索
        for item in os.listdir(self.base_folder):
            item_path = os.path.join(self.base_folder, item)
            if os.path.isdir(item_path):
                # ES: Buscar patrones como "01_", "02_", etc. | EN: Match patterns like "01_", "02_", etc. | JA: 「01_」「02_」などのパターンをマッチ
                match = re.match(r'^(\d{2})_', item)
                if match:
                    number = int(match.group(1))
                    existing_numbers.append(number)
        
        if not existing_numbers:
            return 1
        
        return max(existing_numbers) + 1
    
    def create_subfolder_structure(self, base_output_folder):
        """
        Crea la estructura de subcarpetas dentro de la carpeta de salida
        
        Parameters
        ----------
        base_output_folder : str
            Carpeta base de salida
        
        Returns
        -------
        dict
            Diccionario con las rutas de las subcarpetas creadas
        """
        subfolders = {
            'models': os.path.join(base_output_folder, "02_学習モデル"),
            'results': os.path.join(base_output_folder, "03_学習結果"),
            'predictions': os.path.join(base_output_folder, "04_予測"),
            'pareto': os.path.join(base_output_folder, "05_パレート解"),
        }
        
        for folder_path in subfolders.values():
            os.makedirs(folder_path, exist_ok=True)
            print(f"📁 サブフォルダーを作成しました: {folder_path}")
        
        return subfolders
    
    def get_all_existing_folders(self):
        """
        Obtiene todas las carpetas existentes en orden
        
        Returns
        -------
        list
            Lista de rutas de carpetas ordenadas por número
        """
        if not os.path.exists(self.base_folder):
            return []
        
        folders = []
        for item in os.listdir(self.base_folder):
            item_path = os.path.join(self.base_folder, item)
            if os.path.isdir(item_path):
                match = re.match(r'^(\d{2})_', item)
                if match:
                    number = int(match.group(1))
                    folders.append((number, item_path))
        
        # Ordenar por número
        folders.sort(key=lambda x: x[0])
        return [path for _, path in folders]







