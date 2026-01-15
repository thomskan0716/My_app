#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Loading Window with Progress Bar for Excel Formula Processing
"""

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QProgressBar, QPushButton, QTextEdit)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QPalette, QColor

class ExcelProcessingDialog(QDialog):
    """Dialog for showing Excel formula processing progress"""
    
    def __init__(self, parent=None, total_rows=0):
        super().__init__(parent)
        self.total_rows = total_rows
        self.processed_rows = 0
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components"""
        self.setWindowTitle("Procesando fórmulas Excel")
        self.setFixedSize(500, 300)
        self.setModal(True)
        
        # Main layout
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("🔄 Procesando fórmulas Excel")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Status label
        self.status_label = QLabel(f"Preparando procesamiento de {self.total_rows} filas...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)
        
        # Details text area
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(100)
        self.details_text.setReadOnly(True)
        self.details_text.append("Iniciando procesamiento de fórmulas...")
        layout.addWidget(self.details_text)
        
        # Cancel button
        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.clicked.connect(self.cancel_processing)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Style the dialog
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333333;
            }
            QProgressBar {
                border: 2px solid #cccccc;
                border-radius: 5px;
                text-align: center;
                background-color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 5px;
                background-color: #ffffff;
                font-family: 'Consolas', monospace;
                font-size: 10px;
            }
            QPushButton {
                background-color: #ff6b6b;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff5252;
            }
        """)
        
    def update_progress(self, processed_rows, current_task=""):
        """Update the progress bar and status"""
        self.processed_rows = processed_rows
        
        if self.total_rows > 0:
            percentage = int((processed_rows / self.total_rows) * 100)
            self.progress_bar.setValue(percentage)
            
            remaining_rows = self.total_rows - processed_rows
            self.status_label.setText(
                f"Procesadas: {processed_rows}/{self.total_rows} filas "
                f"({remaining_rows} restantes)"
            )
            
            if current_task:
                self.details_text.append(f"Fila {processed_rows}: {current_task}")
                # Keep only last 10 lines
                lines = self.details_text.toPlainText().split('\n')
                if len(lines) > 10:
                    self.details_text.setPlainText('\n'.join(lines[-10:]))
                
                # Scroll to bottom
                cursor = self.details_text.textCursor()
                cursor.movePosition(cursor.End)
                self.details_text.setTextCursor(cursor)
        
    def set_total_rows(self, total_rows):
        """Set the total number of rows to process"""
        self.total_rows = total_rows
        self.progress_bar.setMaximum(100)
        
    def add_detail(self, message):
        """Add a detail message"""
        self.details_text.append(message)
        
    def cancel_processing(self):
        """Cancel the processing"""
        self.reject()
        
    def processing_complete(self):
        """Mark processing as complete"""
        self.progress_bar.setValue(100)
        self.status_label.setText("✅ Procesamiento completado")
        self.details_text.append("✅ Todas las fórmulas han sido procesadas exitosamente")
        
        # Change cancel button to close
        self.cancel_button.setText("Cerrar")
        self.cancel_button.clicked.disconnect()
        self.cancel_button.clicked.connect(self.accept)
        
    def processing_error(self, error_message):
        """Mark processing as failed"""
        self.status_label.setText("❌ Error en el procesamiento")
        self.details_text.append(f"❌ Error: {error_message}")
        
        # Change cancel button to close
        self.cancel_button.setText("Cerrar")
        self.cancel_button.clicked.disconnect()
        self.cancel_button.clicked.connect(self.accept)


