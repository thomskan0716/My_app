"""
Diálogo para mostrar resultados de análisis de Pareto
Muestra gráficos y permite importar a base de datos
"""
import os
import glob
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap


class ParetoResultsDialog(QDialog):
    """Diálogo para mostrar resultados de Pareto con opción de importar a BD"""
    
    # Señal emitida cuando se solicita importar a BD
    import_requested = Signal(str)  # excel_path
    
    def __init__(self, pareto_plots_folder, prediction_output_file, parent=None):
        super().__init__(parent)
        self.pareto_plots_folder = pareto_plots_folder
        self.prediction_output_file = prediction_output_file
        self.graph_paths = []
        self.current_index = 0
        
        print(f"🔍 DEBUG ParetoResultsDialog.__init__: pareto_plots_folder = {pareto_plots_folder}")
        print(f"🔍 DEBUG ParetoResultsDialog.__init__: prediction_output_file = {prediction_output_file}")
        print(f"🔍 DEBUG ParetoResultsDialog.__init__: pareto_plots_folder exists = {os.path.exists(pareto_plots_folder) if pareto_plots_folder else False}")
        
        self.setWindowTitle("パレート分析結果")
        self.setMinimumSize(900, 700)
        
        self.setup_ui()
        self.load_graphs()
        self.update_display()
        
        print(f"🔍 DEBUG ParetoResultsDialog.__init__: gráficos cargados = {len(self.graph_paths)}")
    
    def setup_ui(self):
        """Configura la interfaz"""
        layout = QVBoxLayout()
        
        # Título
        self.title_label = QLabel("パレート分析結果")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px; color: #2c3e50;")
        layout.addWidget(self.title_label)
        
        # Contenedor para imagen
        image_container = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(500)
        self.image_label.setStyleSheet("background-color: white; border: 2px solid #3498db; border-radius: 5px;")
        image_container.addWidget(self.image_label)
        layout.addLayout(image_container)
        
        # Información del gráfico
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 12px; color: #7f8c8d; margin: 5px;")
        layout.addWidget(self.info_label)
        
        # Navegación
        nav_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("← 前へ")
        self.prev_button.setMinimumWidth(120)
        self.prev_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self.prev_button.clicked.connect(self.show_previous)
        nav_layout.addWidget(self.prev_button)
        
        nav_layout.addStretch()
        
        self.counter_label = QLabel()
        self.counter_label.setAlignment(Qt.AlignCenter)
        self.counter_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        nav_layout.addWidget(self.counter_label)
        
        nav_layout.addStretch()
        
        self.next_button = QPushButton("次へ →")
        self.next_button.setMinimumWidth(120)
        self.next_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self.next_button.clicked.connect(self.show_next)
        nav_layout.addWidget(self.next_button)
        
        layout.addLayout(nav_layout)
        
        # Botones de acción
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        
        # Botón 戻る (Volver)
        self.back_button = QPushButton("戻る")
        self.back_button.setMinimumWidth(120)
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        self.back_button.clicked.connect(self.reject)
        buttons_layout.addWidget(self.back_button)
        
        # Botón データベースにインポート
        self.import_button = QPushButton("データベースにインポート")
        self.import_button.setMinimumWidth(200)
        self.import_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        self.import_button.clicked.connect(self.import_to_database)
        buttons_layout.addWidget(self.import_button)
        
        layout.addLayout(buttons_layout)
        self.setLayout(layout)
    
    def load_graphs(self):
        """Carga los gráficos de Pareto desde la carpeta"""
        print(f"🔍 DEBUG load_graphs: carpeta = {self.pareto_plots_folder}")
        print(f"🔍 DEBUG load_graphs: existe = {os.path.exists(self.pareto_plots_folder) if self.pareto_plots_folder else False}")
        
        if not os.path.exists(self.pareto_plots_folder):
            print(f"⚠️ Carpeta de gráficos no encontrada: {self.pareto_plots_folder}")
            return
        
        # Buscar archivos PNG en la carpeta
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.svg']
        for ext in image_extensions:
            pattern = os.path.join(self.pareto_plots_folder, ext)
            found = glob.glob(pattern)
            print(f"🔍 DEBUG load_graphs: buscando {pattern}, encontrados = {len(found)}")
            self.graph_paths.extend(found)
        
        # Ordenar por nombre
        self.graph_paths.sort()
        
        print(f"📊 Encontrados {len(self.graph_paths)} gráficos de Pareto")
        if self.graph_paths:
            print(f"🔍 DEBUG load_graphs: primeros gráficos = {[os.path.basename(p) for p in self.graph_paths[:3]]}")
    
    def update_display(self):
        """Actualiza la visualización del gráfico actual"""
        if not self.graph_paths:
            self.image_label.setText("グラフが見つかりません")
            self.info_label.setText("")
            self.counter_label.setText("0 / 0")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return
        
        # Actualizar índice
        if self.current_index < 0:
            self.current_index = 0
        elif self.current_index >= len(self.graph_paths):
            self.current_index = len(self.graph_paths) - 1
        
        # Cargar imagen
        current_graph = self.graph_paths[self.current_index]
        if os.path.exists(current_graph):
            pixmap = QPixmap(current_graph)
            if not pixmap.isNull():
                # Escalar manteniendo aspecto
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText(f"画像を読み込めませんでした:\n{current_graph}")
        else:
            self.image_label.setText(f"ファイルが見つかりません:\n{current_graph}")
        
        # Actualizar contador
        total = len(self.graph_paths)
        current = self.current_index + 1
        self.counter_label.setText(f"{current} / {total}")
        
        # Actualizar información
        graph_name = os.path.basename(current_graph)
        self.info_label.setText(f"📊 {graph_name}")
        
        # Actualizar estado de botones
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.graph_paths) - 1)
    
    def show_previous(self):
        """Muestra el gráfico anterior"""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
    
    def show_next(self):
        """Muestra el siguiente gráfico"""
        if self.current_index < len(self.graph_paths) - 1:
            self.current_index += 1
            self.update_display()
    
    def import_to_database(self):
        """Solicita importar a base de datos"""
        # Verificar que el archivo existe
        if not os.path.exists(self.prediction_output_file):
            QMessageBox.warning(
                self,
                "エラー",
                f"❌ 予測結果ファイルが見つかりません:\n\n{self.prediction_output_file}"
            )
            return
        
        # Emitir señal para que el padre maneje la importación
        self.import_requested.emit(self.prediction_output_file)
    
    def resizeEvent(self, event):
        """Redimensionar imagen cuando se redimensiona el diálogo"""
        super().resizeEvent(event)
        self.update_display()

