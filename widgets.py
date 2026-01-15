from PySide6.QtWidgets import (
    QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFrame
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

from app_paths import resource_path


def create_logo_widget(logo_path: str = "xebec.jpg"):
    """Crea un widget QLabel para mostrar el logo escalado manteniendo el aspecto."""
    logo_label = QLabel()
    pixmap = QPixmap(resource_path(logo_path))

    # Escalar la imagen a un tamaño máximo, manteniendo la relación de aspecto
    max_width = 220
    max_height = 220
    scaled_pixmap = pixmap.scaled(
        max_width, max_height,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )

    logo_label.setPixmap(scaled_pixmap)
    logo_label.setAlignment(Qt.AlignCenter)
    return logo_label

def create_load_sample_button():
    """Botón para cargar archivo de muestreo."""
    button = QPushButton("Load Sample File")
    return button

def create_load_results_button():
    """Botón para cargar archivo de resultados."""
    button = QPushButton("Load Results File")
    return button

def create_dsaitekika_button():
    """Botón para ejecutar Dsaitekika."""
    button = QPushButton("Run Dsaitekika")
    button.setEnabled(False)  # Inicialmente desactivado
    return button

def create_isaitekika_button():
    """Botón para ejecutar iSaitekika."""
    button = QPushButton("Run iSaitekika")
    button.setEnabled(False)  # Inicialmente desactivado
    return button

def create_show_results_button():
    """Botón para mostrar resultados de regresión."""
    button = QPushButton("Show Regression Results")
    button.setEnabled(False)  # Inicialmente desactivado
    return button

def create_regression_labels():
    """Crea dos etiquetas para 決定係数 (R²) y 外れ値 (hazurechi)."""
    r2_label = QLabel("決定係数 (R²): ---")
    r2_label.setAlignment(Qt.AlignCenter)
    r2_label.setStyleSheet("font-size: 16px; font-weight: bold;")

    hazurechi_label = QLabel("外れ値 (Outliers): ---")
    hazurechi_label.setAlignment(Qt.AlignCenter)
    hazurechi_label.setStyleSheet("font-size: 16px; font-weight: bold;")

    return r2_label, hazurechi_label

def create_ok_ng_buttons():
    """Crea un frame con los botones OK y NG."""
    frame = QFrame()
    layout = QHBoxLayout()
    frame.setLayout(layout)

    ok_button = QPushButton("OK")
    ok_button.setEnabled(False)  # Inicialmente desactivado

    ng_button = QPushButton("NG")
    ng_button.setEnabled(False)  # Inicialmente desactivado

    layout.addWidget(ok_button)
    layout.addWidget(ng_button)

    return frame, ok_button, ng_button

def create_load_sample_block():
    """Crea un bloque con botón de cargar sample + etiqueta de archivo."""
    frame = QFrame()
    layout = QVBoxLayout()
    frame.setLayout(layout)

    button = QPushButton("Load Sample File")
    label = QLabel("No file loaded")
    label.setAlignment(Qt.AlignCenter)
    label.setStyleSheet("color: gray; font-size: 12px;")

    layout.addWidget(button)
    layout.addWidget(label)

    return frame, button, label

def create_load_results_block():
    """Crea un bloque con botón de cargar resultados + etiqueta de archivo."""
    frame = QFrame()
    layout = QVBoxLayout()
    frame.setLayout(layout)

    button = QPushButton("Load Results File")
    label = QLabel("No file loaded")
    label.setAlignment(Qt.AlignCenter)
    label.setStyleSheet("color: gray; font-size: 12px;")

    layout.addWidget(button)
    layout.addWidget(label)

    return frame, button, label

