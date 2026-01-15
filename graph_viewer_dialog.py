"""
Visor de gráficos con navegación y botones OK/NG
Muestra los 3 gráficos generados por 01_model_builder.py
"""
import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap


class GraphViewerDialog(QDialog):
    """Diálogo para ver gráficos con navegación"""
    
    def __init__(self, graph_paths, parent=None):
        super().__init__(parent)
        self.graph_paths = graph_paths
        self.current_index = 0
        self.ok_pressed = False
        
        self.setWindowTitle("グラフ確認")
        self.setMinimumSize(800, 600)
        
        self.setup_ui()
        self.update_display()
    
    def setup_ui(self):
        """Configura la interfaz"""
        layout = QVBoxLayout()
        
        # Título
        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(self.title_label)
        
        # Contenedor para imagen
        image_container = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(400)
        self.image_label.setStyleSheet("background-color: white; border: 1px solid #ddd;")
        image_container.addWidget(self.image_label)
        layout.addLayout(image_container)
        
        # Navegación
        nav_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("← 前へ")
        self.prev_button.setMinimumWidth(100)
        self.prev_button.clicked.connect(self.show_previous)
        nav_layout.addWidget(self.prev_button)
        
        nav_layout.addStretch()
        
        self.counter_label = QLabel()
        self.counter_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.counter_label)
        
        nav_layout.addStretch()
        
        self.next_button = QPushButton("次へ →")
        self.next_button.setMinimumWidth(100)
        self.next_button.clicked.connect(self.show_next)
        nav_layout.addWidget(self.next_button)
        
        layout.addLayout(nav_layout)
        
        # Botones OK/NG
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        
        self.ng_button = QPushButton("NG")
        self.ng_button.setMinimumWidth(100)
        self.ng_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.ng_button.clicked.connect(self.reject)
        buttons_layout.addWidget(self.ng_button)
        
        self.ok_button = QPushButton("OK")
        self.ok_button.setMinimumWidth(100)
        self.ok_button.setDefault(True)
        self.ok_button.setStyleSheet("""
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
        self.ok_button.clicked.connect(self.accept)
        buttons_layout.addWidget(self.ok_button)
        
        layout.addLayout(buttons_layout)
        self.setLayout(layout)
    
    def update_display(self):
        """Actualiza la visualización del gráfico actual"""
        if not self.graph_paths:
            self.image_label.setText("グラフが見つかりません")
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
        
        # Actualizar título
        graph_name = os.path.basename(current_graph)
        self.title_label.setText(f"グラフ: {graph_name}")
        
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
    
    def resizeEvent(self, event):
        """Redimensionar imagen cuando se redimensiona el diálogo"""
        super().resizeEvent(event)
        self.update_display()







