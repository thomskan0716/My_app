"""
Diálogo para seleccionar el tipo de cepillo (A13, A11, A21, A32) para el análisis de clasificación
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QButtonGroup, QRadioButton
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class BrushSelectionDialog(QDialog):
    """Diálogo para seleccionar el tipo de cepillo"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ブラシタイプ選択")
        self.setMinimumWidth(400)
        self.setMinimumHeight(200)
        
        self.selected_brush = None
        self.setup_ui()
    
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Título
        title = QLabel("予測データ用のブラシタイプを選択してください")
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Descripción
        description = QLabel("Prediction_input.xlsx に使用するブラシタイプを選択します。\n選択したブラシタイプは 1、その他は 0 になります。")
        description.setWordWrap(True)
        description.setStyleSheet("color: #666; margin-bottom: 15px;")
        layout.addWidget(description)
        
        # Grupo de botones de radio
        self.button_group = QButtonGroup(self)
        
        radio_a13 = QRadioButton("A13")
        radio_a13.setChecked(True)  # Por defecto A13
        self.selected_brush = "A13"
        radio_a13.toggled.connect(lambda checked: self._on_brush_selected("A13", checked))
        
        radio_a11 = QRadioButton("A11")
        radio_a11.toggled.connect(lambda checked: self._on_brush_selected("A11", checked))
        
        radio_a21 = QRadioButton("A21")
        radio_a21.toggled.connect(lambda checked: self._on_brush_selected("A21", checked))
        
        radio_a32 = QRadioButton("A32")
        radio_a32.toggled.connect(lambda checked: self._on_brush_selected("A32", checked))
        
        self.button_group.addButton(radio_a13, 0)
        self.button_group.addButton(radio_a11, 1)
        self.button_group.addButton(radio_a21, 2)
        self.button_group.addButton(radio_a32, 3)
        
        layout.addWidget(radio_a13)
        layout.addWidget(radio_a11)
        layout.addWidget(radio_a21)
        layout.addWidget(radio_a32)
        
        layout.addStretch()
        
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
    
    def _on_brush_selected(self, brush_type, checked):
        """Maneja la selección de un tipo de cepillo"""
        if checked:
            self.selected_brush = brush_type
    
    def get_selected_brush(self):
        """Retorna el tipo de cepillo seleccionado"""
        return self.selected_brush

