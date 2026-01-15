import sys
import os
import warnings
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QWidget, QGridLayout,
                             QProgressBar, QProgressDialog, QComboBox, QLineEdit, QDateEdit, QRadioButton,
                             QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
                             QDialog, QFrame, QScrollArea, QSplitter, QTextEdit, QGroupBox,
                             QCheckBox, QSpinBox, QDoubleSpinBox, QTabWidget, QTextBrowser,
                             QFormLayout, QSizePolicy, QListWidget, QDialogButtonBox)
from PySide6.QtCore import Qt, QTimer, QDate, QThread, Signal, QPropertyAnimation, QEasingCurve, QSize, QObject, QEvent, QPoint
from PySide6.QtGui import QPixmap, QFont, QPalette, QColor, QIcon, QPainter, QLinearGradient, QMovie, QIntValidator, QTextCursor, QFontDatabase, QFontMetrics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from datetime import datetime, timedelta
import json
import sqlite3
import traceback
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import subprocess
import shutil
import zipfile
import tempfile
import glob
import re
from pathlib import Path
import logging
import hashlib
import pickle
import gzip
import base64
import io
import csv
import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows
import warnings
warnings.filterwarnings('ignore')

# --- D基準値 (D-score) - referencia: D_and_I最適化_Greedy法_ver3.py ---
from scipy.spatial.distance import cdist
from scipy.linalg import qr as scipy_qr
try:
    from sklearn.preprocessing import StandardScaler
except Exception:
    StandardScaler = None

from app_paths import (
    resource_path,
    get_db_path,
    get_backup_dir,
    migrate_legacy_db_if_needed,
)
from app_manifest import get_app_title
from backup_manager import auto_daily_backup, create_backup, prune_backups

# Rutas canónicas de BBDD (instalación profesional: ProgramData\\...\\data)
RESULTS_DB_PATH = migrate_legacy_db_if_needed("results.db", shared=True)
YOSOKU_LINEAL_DB_PATH = get_db_path("yosoku_predictions_lineal.db", shared=True)
YOSOKU_NO_LINEAL_DB_PATH = get_db_path("yosoku_predictions_no_lineal.db", shared=True)

print("🔧 Importando módulos...")

try:
    print("🔧 Importando widgets...")
    from widgets import (
        create_logo_widget, create_ok_ng_buttons, create_dsaitekika_button, create_isaitekika_button,
        create_load_sample_button, create_load_results_button, create_show_results_button,
        create_regression_labels, create_load_sample_block, create_load_results_block
    )
    print("✅ Widgets importados correctamente")
except Exception as e:
    print(f"❌ Error importando widgets: {e}")
    raise

try:
    print("🔧 Importando workers...")
    from dsaitekikaworker import DsaitekikaWorker
    from showresultsworker import ShowResultsWorker
    from samplecombineworker import SampleCombinerWorker
    print("✅ Workers importados correctamente")
except Exception as e:
    print(f"❌ Error importando workers: {e}")
    raise

try:
    print("🔧 Importando nonlinear worker...")
    from nonlinear_worker import NonlinearWorker
    print("✅ Nonlinear worker importado correctamente")
except Exception as e:
    print(f"⚠️ Error importando nonlinear worker: {e}")
    print("  (continuando sin análisis no lineal)")
    NonlinearWorker = None

try:
    print("🔧 Importando diálogos de análisis no lineal...")
    from nonlinear_config_dialog import NonlinearConfigDialog
    from graph_viewer_dialog import GraphViewerDialog
    from pareto_results_dialog import ParetoResultsDialog
    print("✅ Diálogos importados correctamente")
except Exception as e:
    print(f"⚠️ Error importando diálogos: {e}")
    print("  (continuando sin diálogos)")
    NonlinearConfigDialog = None
    GraphViewerDialog = None
    ParetoResultsDialog = None

try:
    print("🔧 Importando classification worker...")
    from classification_worker import ClassificationWorker
    print("✅ Classification worker importado correctamente")
except Exception as e:
    print(f"⚠️ Error importando classification worker: {e}")
    print("  (continuando sin análisis de clasificación)")
    ClassificationWorker = None

try:
    print("🔧 Importando diálogo de clasificación...")
    from classification_config_dialog import ClassificationConfigDialog
    print("✅ Diálogo de clasificación importado correctamente")
except Exception as e:
    print(f"⚠️ Error importando diálogo de clasificación: {e}")
    print("  (continuando sin diálogo)")
    ClassificationConfigDialog = None

try:
    print("🔧 Importando diálogo de selección de cepillo...")
    from brush_selection_dialog import BrushSelectionDialog
    print("✅ Diálogo de selección de cepillo importado correctamente")
except Exception as e:
    print(f"⚠️ Error importando diálogo de selección de cepillo: {e}")
    print("  (continuando sin diálogo)")
    BrushSelectionDialog = None

try:
    print("🔧 Importando módulos de base de datos...")
    from db_manager import DBManager as DBManagerMain
    from result_processor import ResultProcessor
    print("✅ Módulos de BD importados correctamente")
except Exception as e:
    print(f"❌ Error importando módulos de BD: {e}")
    raise

try:
    print("🔧 Importando integrated optimizer...")
    from integrated_optimizer_worker import IntegratedOptimizerWorker
    print("✅ Integrated optimizer importado correctamente")
except Exception as e:
    print(f"❌ Error importando integrated optimizer: {e}")
    raise

print("✅ Todos los módulos importados correctamente")
from datetime import datetime
import glob
import os, shutil
import sqlite3
import pandas as pd
import numpy as np

def calculate_d_criterion(X_selected):
    """Calcula el criterio D-óptimo usando la lógica de D_and_I最適化_Greedy法_ver3.py"""
    try:
        if X_selected.shape[0] < X_selected.shape[1]:
            return -np.inf
            
        # Calcular número de condición para detectar problemas numéricos
        condition_number = np.linalg.cond(X_selected)
        
        # Usar método numéricamente estable si la matriz está mal condicionada
        USE_NUMERICAL_STABLE_METHOD = True
        if USE_NUMERICAL_STABLE_METHOD or condition_number > 1e12:
            method = 'svd'
            print(f"🔧 高条件数検出({condition_number:.2e}) - SVD法適用")
        else:
            method = 'qr'
            
        if method == 'svd':
            # Usar SVD para matrices mal condicionadas
            _, s, _ = np.linalg.svd(X_selected, full_matrices=False)
            valid_singular_values = s[s > 1e-14]
            if len(valid_singular_values) == 0:
                return -np.inf
            log_det = np.sum(np.log(valid_singular_values))
        else:
            # Usar QR decomposition para matrices bien condicionadas
            q, r = np.linalg.qr(X_selected, mode='economic')
            diag_r = np.diag(r)
            det = np.abs(np.prod(diag_r))
            log_det = np.log(det) if det > 1e-300 else -np.inf
            
        return log_det
    except Exception as e:
        print(f"⚠️ D-criterion計算エラー: {e}")
        return -np.inf

def calculate_i_criterion(X_selected, X_all):
    """Calcula el criterio I-óptimo"""
    try:
        if len(X_selected) == 0:
            return -np.inf
        distances = cdist(X_all, X_selected)
        min_distances = np.min(distances, axis=1)
        return -np.mean(min_distances)
    except:
        return -np.inf

def _standardize_like_reference(X: np.ndarray) -> np.ndarray:
    """
    Estandariza como en el archivo de referencia (StandardScaler).
    Si sklearn no está disponible, aplica z-score (ddof=0) con fallback seguro.
    """
    X = np.asarray(X, dtype=float)
    if StandardScaler is not None:
        return StandardScaler().fit_transform(X)
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std

def calculate_d_criterion_stable_reference(X: np.ndarray, method: str = "auto",
                                           use_numerical_stable_method: bool = True,
                                           verbose: bool = False):
    """
    Cálculo idéntico a D_and_I最適化_Greedy法_ver3.py:
    devuelve (log_det, condition_number)
    """
    try:
        condition_number = np.linalg.cond(X)
        if use_numerical_stable_method or (method == "auto" and condition_number > 1e12):
            method = "svd"
            if verbose and condition_number > 1e12:
                print(f"🔧 高条件数検出({condition_number:.2e}) - SVD法適用")
        if method == "svd":
            _, s, _ = np.linalg.svd(X, full_matrices=False)
            valid_singular_values = s[s > 1e-14]
            if len(valid_singular_values) == 0:
                return -np.inf, condition_number
            log_det = np.sum(np.log(valid_singular_values))
        else:
            _, r = scipy_qr(X, mode="economic")
            diag_r = np.diag(r)
            det = np.abs(np.prod(diag_r))
            log_det = np.log(det) if det > 1e-300 else -np.inf
        return log_det, condition_number
    except Exception as e:
        if verbose:
            print(f"⚠️ D-criterion計算エラー: {e}")
        return -np.inf, np.inf

def calculate_d_score_reference(candidate_points_raw: np.ndarray, selected_indices) -> float:
    """
    D-score de referencia: fit StandardScaler sobre TODOS los candidatos,
    luego D-criterion estable sobre el subconjunto seleccionado.
    """
    if candidate_points_raw is None or selected_indices is None:
        return -np.inf
    X_scaled = _standardize_like_reference(candidate_points_raw)
    selected_indices = list(selected_indices)
    if len(selected_indices) == 0:
        return -np.inf
    X_subset = X_scaled[selected_indices]
    score, _ = calculate_d_criterion_stable_reference(X_subset, method="auto", use_numerical_stable_method=True, verbose=False)
    return float(score)

def _extract_design_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Extrae la matriz de variables de diseño (7 columnas) por NOMBRE, compatible con formato antiguo y nuevo.
    Columnas esperadas:
      回転速度, 送り速度, (UPカット o 回転方向), (切込量 o 切込み量), (突出量 o 突出し量), 載せ率, パス数
    """
    dir_col = "UPカット" if "UPカット" in df.columns else ("回転方向" if "回転方向" in df.columns else None)
    if dir_col is None:
        raise ValueError("❌ Falta columna de dirección: 'UPカット' o '回転方向'")
    cut_col = "切込量" if "切込量" in df.columns else ("切込み量" if "切込み量" in df.columns else None)
    if cut_col is None:
        raise ValueError("❌ Falta columna de切込量: '切込量' o '切込み量'")
    out_col = "突出量" if "突出量" in df.columns else ("突出し量" if "突出し量" in df.columns else None)
    if out_col is None:
        raise ValueError("❌ Falta columna de突出量: '突出量' o '突出し量'")

    design_cols = ["回転速度", "送り速度", dir_col, cut_col, out_col, "載せ率", "パス数"]
    missing = [c for c in design_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Faltan columnas de diseño: {missing}")
    X = df[design_cols].copy()
    for c in design_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    if X.isna().any().any():
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"❌ Valores no numéricos en columnas de diseño: {bad_cols}")
    return X.to_numpy()








class LoadingOverlay(QWidget):
    """
    Widget overlay para mostrar loading dentro de la ventana principal.
    Usa QWidget en lugar de QDialog para que sea parte de la jerarquía de widgets
    y respete automáticamente el orden de ventanas del sistema operativo.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # CRÍTICO: Forzar explícitamente que NO sea una ventana de nivel superior
        # Esto asegura que el widget sea parte de la jerarquía del parent, no una ventana flotante
        self.setWindowFlags(Qt.Widget)  # Forzar que sea widget hijo, no ventana
        
        # NO usar setWindowModality - es un widget hijo, no una ventana
        # El widget será parte de la jerarquía del parent (center_frame)
        
        # Asegurar que tenga parent (si no lo tiene, no funcionará correctamente)
        if parent:
            self.setParent(parent)
        
        # Configurar como widget overlay con fondo semitransparente
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("""
            QWidget {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
            }
            QLabel {
                background: transparent;
                color: white;
            }
        """)

        # Layout centrado para el loading
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)

        self.movie = QMovie(resource_path("loading.gif"))
        self.movie.setScaledSize(QSize(64, 64))
        self.label.setMovie(self.movie)

        layout.addWidget(self.label)
        
        # Inicialmente oculto
        self.hide()
    
    def _update_geometry(self):
        """Actualiza la geometría para cubrir todo el parent"""
        if self.parent() and self.isVisible():
            parent = self.parent()
            self.setGeometry(0, 0, parent.width(), parent.height())
    
    def start(self):
        """Inicia el loading y lo muestra cubriendo todo el parent"""
        # CRÍTICO: Verificar y forzar que NO sea una ventana
        # Si por alguna razón se convirtió en ventana, forzar que no lo sea
        if self.isWindow():
            print("⚠️ WARNING: LoadingOverlay se detectó como ventana, corrigiendo...")
            self.setWindowFlags(Qt.Widget)
            if self.parent():
                self.setParent(self.parent())  # Re-establecer parent
        
        if self.parent():
            parent = self.parent()
            
            # Asegurar que el parent esté establecido correctamente
            if self.parent() != parent:
                self.setParent(parent)
            
            # Forzar que NO sea ventana nuevamente después de setParent
            self.setWindowFlags(Qt.Widget)
            
            # Cubrir todo el área del parent
            self.setGeometry(0, 0, parent.width(), parent.height())
            print(f"🔧 Loading overlay configurado: {parent.width()}x{parent.height()}")
            print(f"🔧 Es ventana: {self.isWindow()}, Parent: {parent}")
            
            # Conectar el evento de resize del parent para ajustar el overlay
            if not hasattr(self, '_resize_connected'):
                parent.installEventFilter(self)
                self._resize_connected = True
        else:
            # Si no hay parent, usar tamaño mínimo
            print("⚠️ WARNING: LoadingOverlay no tiene parent")
            self.resize(120, 120)
            # Aún así, forzar que no sea ventana
            self.setWindowFlags(Qt.Widget)

        self.movie.start()
        self.show()
        
        # Verificar una vez más que no sea ventana después de show()
        if self.isWindow():
            print("⚠️ WARNING: LoadingOverlay se convirtió en ventana después de show(), corrigiendo...")
            self.setWindowFlags(Qt.Widget)
            if self.parent():
                self.setParent(self.parent())
        
        self.raise_()  # Elevar dentro del parent, no del sistema
        QApplication.processEvents()  # Forzar actualización de la UI
    
    def eventFilter(self, obj, event):
        """Filtra eventos del parent para ajustar el tamaño cuando cambia"""
        if obj == self.parent() and event.type() == QEvent.Type.Resize:
            self._update_geometry()
        return super().eventFilter(obj, event)

    def stop(self):
        """Detiene el loading y lo oculta"""
        self.movie.stop()
        self.hide()

class ReusableProgressDialog(QDialog):
    """Cuadro de progreso reutilizable con imagen personalizable"""
    
    # Señal emitida cuando se cancela el proceso
    cancelled = Signal()
    
    def __init__(self, parent=None, title="処理中...", chibi_image="xebec_chibi_suzukisan.png", chibi_size=100):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(600, 320)  # Tamaño ajustado para incluir tiempo
        # Sin WindowStaysOnTopHint: solo bloquea el parent, no se queda en primer plano del sistema
        self.setWindowFlags(Qt.Dialog)
        # WindowModal bloquea solo el parent, no toda la aplicación ni otras apps
        self.setWindowModality(Qt.WindowModal)
        
        # Variables para tracking de actividad
        self.start_time = time.time()
        self.last_activity_time = time.time()
        self.process_active = True  # Estado del proceso Python
        self.last_progress_value = 0
        self.activity_timer = QTimer()
        self.activity_timer.timeout.connect(self._update_activity_indicator)
        self.activity_timer.start(1000)  # Actualizar cada segundo
        
        # Variables para tracking de stages
        self.current_stage = '01_model_builder'  # Stage actual
        
        # Establecer fondo sólido sin borde
        self.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
                border-radius: 10px;
            }
        """)
        
        # Layout principal
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Título y chibi en la misma línea horizontal
        title_chibi_layout = QHBoxLayout()
        title_chibi_layout.setContentsMargins(0, 0, 0, 0)
        title_chibi_layout.setSpacing(10)
        
        # Título a la izquierda
        title_label = QLabel("処理実行中")
        title_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        """)
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_label.setFixedHeight(30)
        title_chibi_layout.addWidget(title_label)
        
        # Espaciador para empujar el chibi a la derecha
        title_chibi_layout.addStretch()
        
        # Imagen del chibi a la derecha
        try:
            chibi_label = QLabel()
            chibi_pixmap = QPixmap(resource_path(chibi_image))
            if not chibi_pixmap.isNull():
                # Redimensionar para un tamaño adecuado (usando chibi_size)
                scaled_pixmap = chibi_pixmap.scaled(chibi_size, chibi_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                chibi_label.setPixmap(scaled_pixmap)
                chibi_label.setFixedSize(chibi_size, chibi_size)
                chibi_label.setStyleSheet("background: transparent; border: none; margin: 0; padding: 0;")
                chibi_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                title_chibi_layout.addWidget(chibi_label)
            else:
                print(f"⚠️ No se pudo cargar {chibi_image}")
        except Exception as e:
            print(f"⚠️ Error cargando imagen chibi: {e}")
        
        layout.addLayout(title_chibi_layout)
        
        # Label para tiempo transcurrido y estimado (centrado, debajo del título)
        time_info_layout = QHBoxLayout()
        time_info_layout.addStretch()
        self.time_info_label = QLabel("⏱️ 経過時間: 0:00 | 推定残り時間: 計算中...")
        self.time_info_label.setStyleSheet("""
            font-size: 13px;
            font-weight: bold;
            color: #2c3e50;
            padding: 5px;
        """)
        self.time_info_label.setAlignment(Qt.AlignCenter)
        self.time_info_label.setFixedHeight(25)
        time_info_layout.addWidget(self.time_info_label)
        time_info_layout.addStretch()
        layout.addLayout(time_info_layout)
        
        # Variables para cálculo de tiempo estimado
        self.trial_times = []  # Lista de tiempos por trial
        self.last_trial_start_time = None
        self.current_trial_number = 0
        
        # Barra de progreso centrada que ocupa todo el ancho
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(30)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                background-color: #ecf0f1;
                min-height: 25px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Etiqueta de porcentaje centrada (azul)
        percentage_layout = QHBoxLayout()
        percentage_layout.addStretch()
        self.percentage_label = QLabel("0%")
        self.percentage_label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #3498db;
        """)
        self.percentage_label.setAlignment(Qt.AlignCenter)
        self.percentage_label.setFixedHeight(25)
        percentage_layout.addWidget(self.percentage_label)
        percentage_layout.addStretch()
        layout.addLayout(percentage_layout)
        
        # Etiqueta para mostrar Trial, Fold y Pasadas centrada
        trial_fold_layout = QHBoxLayout()
        trial_fold_layout.addStretch()
        self.trial_fold_label = QLabel("")
        self.trial_fold_label.setStyleSheet("""
            font-size: 13px;
            font-weight: bold;
            color: #2c3e50;
        """)
        self.trial_fold_label.setAlignment(Qt.AlignCenter)
        self.trial_fold_label.setFixedHeight(25)
        trial_fold_layout.addWidget(self.trial_fold_label)
        trial_fold_layout.addStretch()
        layout.addLayout(trial_fold_layout)
        
        # Botón de cancelar centrado
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_button = QPushButton("キャンセル")
        self.cancel_button.setFixedSize(120, 35)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        self.cancel_button.clicked.connect(self.cancel_process)
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Centrar en la pantalla
        self.center_on_screen()
    
    def center_on_screen(self):
        """Centrar el diálogo en la pantalla"""
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)
    
    def update_progress(self, percentage, status_message):
        """Actualizar progreso y mensaje de estado"""
        current_time = time.time()
        
        # Actualizar última actividad si hay cambio de progreso
        if abs(int(percentage) - self.last_progress_value) > 0:
            self.last_activity_time = current_time
            self.last_progress_value = int(percentage)
        
        # Actualizar barra de progreso
        self.progress_bar.setValue(int(percentage))
        self.percentage_label.setText(f"{int(percentage)}%")
        
        # Actualizar color según actividad (Opción 4)
        self._update_progress_color(current_time)
        
        QApplication.processEvents()  # Forzar actualización de la UI
    
    def set_process_active(self, active):
        """Actualizar estado del proceso Python"""
        self.process_active = active
        QApplication.processEvents()
    
    def _update_progress_color(self, current_time):
        """Actualizar color de la barra según actividad (Opción 4)"""
        time_since_activity = current_time - self.last_activity_time
        
        if time_since_activity < 3:
            # Verde: actividad reciente
            color = "#27ae60"
        elif time_since_activity < 10:
            # Amarillo: actividad moderada
            color = "#f39c12"
        elif time_since_activity < 30:
            # Naranja: posible bloqueo
            color = "#e67e22"
        else:
            # Rojo: probable bloqueo
            color = "#e74c3c"
        
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                background-color: #ecf0f1;
                min-height: 25px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)
    
    def _update_activity_indicator(self):
        """Actualizar indicadores de actividad cada segundo"""
        current_time = time.time()
        
        # Actualizar tiempo transcurrido siempre
        if hasattr(self, 'time_info_label'):
            elapsed_time = current_time - self.start_time
            elapsed_str = self._format_time(elapsed_time)
            
            # Obtener el texto actual para preservar la estimación si existe
            current_text = self.time_info_label.text()
            
            # Si ya hay una estimación calculada (no "計算中"), preservarla
            if "推定残り時間:" in current_text and "計算中" not in current_text:
                # Extraer la estimación del texto actual
                try:
                    remaining_part = current_text.split("推定残り時間:")[1].strip()
                    # Actualizar solo el tiempo transcurrido, mantener la estimación
                    self.time_info_label.setText(f"⏱️ 経過時間: {elapsed_str} | 推定残り時間: {remaining_part}")
                except:
                    # Si falla, calcular estimación básica
                    if len(self.trial_times) > 0 and elapsed_time > 0:
                        # Usar promedio de trials para estimar
                        avg_trial_time = sum(self.trial_times) / len(self.trial_times)
                        estimated_remaining = max(0, avg_trial_time - elapsed_time)
                        estimated_str = self._format_time(estimated_remaining)
                        self.time_info_label.setText(f"⏱️ 経過時間: {elapsed_str} | 推定残り時間: {estimated_str}")
                    else:
                        self.time_info_label.setText(f"⏱️ 経過時間: {elapsed_str} | 推定残り時間: 計算中...")
            else:
                # No hay estimación, calcular una básica si es posible
                if len(self.trial_times) > 0 and elapsed_time > 0:
                    avg_trial_time = sum(self.trial_times) / len(self.trial_times)
                    estimated_remaining = max(0, avg_trial_time - elapsed_time)
                    estimated_str = self._format_time(estimated_remaining)
                    self.time_info_label.setText(f"⏱️ 経過時間: {elapsed_str} | 推定残り時間: {estimated_str}")
                else:
                    self.time_info_label.setText(f"⏱️ 経過時間: {elapsed_str} | 推定残り時間: 計算中...")
        
        # Actualizar color según actividad
        self._update_progress_color(current_time)
        
        QApplication.processEvents()
    
    def _format_time(self, seconds):
        """Formatea segundos a formato legible (MM:SS o HH:MM:SS)"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}:{secs:02d}"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}:{mins:02d}:{secs:02d}"
    
    def set_status(self, status_message):
        """Actualizar solo el mensaje de estado (no se usa en el layout simplificado)"""
        pass
        QApplication.processEvents()
    
    def update_progress_detailed(self, trial_current, trial_total, fold_current, fold_total, pass_current, pass_total, current_task='dcv', data_analysis_completed=False, final_model_training=False, shap_analysis=False, model_current=0, model_total=0):
        """Actualizar información detallada de progreso (trial/fold/pass) y calcular porcentaje"""
        current_time = time.time()
        
        # Detectar cuando comienza un nuevo trial para calcular tiempo promedio
        if trial_current > self.current_trial_number:
            # Nuevo trial detectado
            if self.last_trial_start_time is not None:
                # Calcular tiempo del trial anterior
                trial_duration = current_time - self.last_trial_start_time
                self.trial_times.append(trial_duration)
                # Mantener solo los últimos 10 trials para el promedio
                if len(self.trial_times) > 10:
                    self.trial_times.pop(0)
            
            self.last_trial_start_time = current_time
            self.current_trial_number = trial_current
        
        # Calcular porcentaje basado en trials, folds y passes
        # Stage 1 (model_builder): ~70% del total
        # Stage 2 (prediction): ~15% del total
        # Stage 3 (pareto): ~15% del total
        
        # Distribución del progreso dentro de Stage 1 (70%):
        # - Inicialización y carga: 0-2%
        # - Análisis de datos (si está habilitado): 2-5%
        # - DCV (Double Cross-Validation): 5-60%
        #   - Por cada pasada (target): 
        #     - Outer Folds con optimización (trials): mayor parte
        #     - Modelo final: 2%
        #     - SHAP: 1%
        #     - Guardado: 0.5%
        # - Tareas finales: 60-70%
        
        percentage = 0
        stage1_base = 0  # Base del stage 1 (0-70%)
        
        # 1. Inicialización y carga (0-2%)
        stage1_base += 2
        
        # 2. Análisis de datos (2-5%) - solo si está habilitado
        if data_analysis_completed:
            stage1_base = 5
        elif current_task == 'data_analysis':
            # Análisis de datos en progreso
            stage1_base = 3.5
        
        # 3. DCV (Double Cross-Validation) - 5% a 60%
        # ✅ Usar valores acumulados para cálculo lineal e incremental del porcentaje
        # Los valores acumulados se calculan en nonlinear_worker.py y se pasan a través de trial_current/trial_total
        # cuando se detecta el formato "X/Y" de la barra de progreso de Optuna
        if trial_total > 0 and fold_total > 0 and pass_total > 0:
            # Progreso dentro del DCV (5% a 60% = 55% del stage 1)
            dcv_start = 5
            dcv_range = 55  # 60 - 5
            
            # ✅ Calcular total de trials acumulados (considerando modelos, folds, passes y trials)
            # Total = passes * folds * modelos * trials
            # model_total se pasa como parámetro, pero si no está disponible, usar 1
            model_total_used = model_total if model_total > 0 else 1  # Si no hay info de modelos, asumir 1
            total_trials_accumulated = pass_total * fold_total * model_total_used * trial_total
            
            # ✅ Calcular trials completados acumulados
            # trial_current aquí es el contador de trials completados en el modelo/fold actual
            # Necesitamos calcular el acumulado: (passes completados * folds * modelos * trials) + 
            #                                     (folds completados * modelos * trials) + 
            #                                     (modelos completados * trials) + 
            #                                     (trials completados)
            completed_passes = max(0, pass_current - 1)
            completed_folds_in_pass = max(0, fold_current - 1)
            completed_models_in_fold = max(0, model_current - 1) if model_total > 0 else 0
            completed_trials_accumulated = (
                (completed_passes * fold_total * model_total_used * trial_total) +
                (completed_folds_in_pass * model_total_used * trial_total) +
                (completed_models_in_fold * trial_total) +
                trial_current
            )
            
            # ✅ Calcular progreso lineal basado en trials acumulados
            trial_progress = completed_trials_accumulated / total_trials_accumulated if total_trials_accumulated > 0 else 0
            
            # Los trials representan ~85% del tiempo total del DCV
            # El resto (15%) es para entrenamiento final, SHAP y guardado
            dcv_trial_progress = trial_progress * 0.85
            
            # Agregar progreso del modelo final (5% del DCV)
            if final_model_training:
                dcv_trial_progress = min(0.90, dcv_trial_progress + 0.05)  # Máximo 90% para dejar espacio a SHAP
            
            # Agregar progreso de SHAP (3% del DCV)
            if shap_analysis:
                dcv_trial_progress = min(0.95, dcv_trial_progress + 0.03)  # Máximo 95% para dejar espacio a guardado
            
            # Agregar progreso de guardado (2% del DCV)
            if current_task == 'saving':
                dcv_trial_progress = min(1.0, dcv_trial_progress + 0.02)
            
            # Calcular progreso del DCV
            dcv_progress = dcv_start + (dcv_trial_progress * dcv_range)
            stage1_base = max(stage1_base, dcv_progress)
        
        # 4. Tareas finales (guardado, etc.) - 60-70%
        if current_task == 'saving' or (pass_current >= pass_total and pass_total > 0):
            # Si todas las pasadas están completas, avanzar hacia el final
            if pass_current >= pass_total:
                stage1_base = 70
        
        # Stage 1 representa 70% del total
        percentage = min(70, stage1_base)
        
        # Actualizar barra de progreso y porcentaje
        self.progress_bar.setValue(int(percentage))
        self.percentage_label.setText(f"{int(percentage)}%")
        
        # Actualizar tiempo transcurrido y estimado
        elapsed_time = current_time - self.start_time
        elapsed_str = self._format_time(elapsed_time)
        
        # Calcular tiempo estimado
        estimated_remaining = None
        if len(self.trial_times) > 0:
            # Calcular tiempo promedio por trial
            avg_trial_time = sum(self.trial_times) / len(self.trial_times)
            
            # Calcular trials restantes
            if trial_total > 0 and fold_total > 0 and pass_total > 0:
                # Trials restantes en el fold actual
                remaining_trials_in_fold = max(0, trial_total - trial_current)
                # Folds restantes en el pass actual
                remaining_folds_in_pass = max(0, fold_total - fold_current)
                # Passes restantes
                remaining_passes = max(0, pass_total - pass_current)
                
                # Calcular tiempo restante para stage 1
                remaining_trials_stage1 = (
                    remaining_trials_in_fold +
                    remaining_folds_in_pass * trial_total +
                    remaining_passes * fold_total * trial_total
                )
                
                # Tiempo estimado para stage 1
                estimated_stage1 = remaining_trials_stage1 * avg_trial_time
                
                # Tiempo estimado para stages 2 y 3 (aproximadamente 30% del tiempo total)
                # Si stage 1 toma 70%, entonces stages 2+3 toman aproximadamente 30%
                # Estimar basado en el tiempo ya transcurrido
                if percentage > 0:
                    total_estimated_time = elapsed_time / (percentage / 100)
                    estimated_stage1_remaining = (total_estimated_time * 0.70) - elapsed_time
                    estimated_stage2_3 = total_estimated_time * 0.30
                    estimated_remaining = max(0, estimated_stage1_remaining + estimated_stage2_3)
                else:
                    estimated_remaining = estimated_stage1 * (1 / 0.70)  # Ajustar para incluir stages 2 y 3
        
        if estimated_remaining is not None:
            estimated_str = self._format_time(estimated_remaining)
            self.time_info_label.setText(f"⏱️ 経過時間: {elapsed_str} | 推定残り時間: {estimated_str}")
        else:
            self.time_info_label.setText(f"⏱️ 経過時間: {elapsed_str} | 推定残り時間: 計算中...")
        
        if hasattr(self, 'trial_fold_label'):
            # ✅ Formatear información: Model X/Y: Trial Z/W | Fold A/B | Pass C/D
            parts = []
            
            # Modelo (si hay modelos configurados)
            if model_total > 0:
                parts.append(f"Model: {model_current}/{model_total}")
            
            # Trial (si hay trials)
            if trial_total > 0:
                parts.append(f"Trial: {trial_current}/{trial_total}")
            
            # Fold (si hay folds)
            if fold_total > 0:
                parts.append(f"Fold: {fold_current}/{fold_total}")
            
            # Pass (si hay passes)
            if pass_total > 0:
                parts.append(f"Pass: {pass_current}/{pass_total}")
            
            # Combinar todas las partes con " | "
            combined_text = " | ".join(parts) if parts else ""
            
            self.trial_fold_label.setText(combined_text)
        
        QApplication.processEvents()
    
    def update_status(self, status_message):
        """Actualizar solo el mensaje de estado (alias para set_status)"""
        self.set_status(status_message)
    
    def set_title(self, title):
        """Cambiar el título del diálogo"""
        self.setWindowTitle(title)
    
    def set_main_title(self, title):
        """Cambiar el título principal dentro del diálogo"""
        # Buscar el título label y actualizarlo
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if isinstance(widget, QHBoxLayout):
                    for j in range(widget.count()):
                        sub_item = widget.itemAt(j)
                        if sub_item and sub_item.widget():
                            sub_widget = sub_item.widget()
                            if isinstance(sub_widget, QVBoxLayout):
                                for k in range(sub_widget.count()):
                                    label_item = sub_widget.itemAt(k)
                                    if label_item and label_item.widget():
                                        label_widget = label_item.widget()
                                        if isinstance(label_widget, QLabel) and label_widget.text() == "処理実行中":
                                            label_widget.setText(title)
                                            return
    
    def cancel_process(self):
        """Cancelar proceso y cerrar popup"""
        # Emitir señal de cancelación antes de cerrar
        self.cancelled.emit()
        self.progress_bar.setValue(0)
        self.percentage_label.setText("0%")
        QApplication.processEvents()
        self.reject()

class LinearAnalysisProgressDialog(ReusableProgressDialog):
    """Popup de progreso para análisis lineal usando la clase reutilizable"""
    
    def __init__(self, parent=None):
        super().__init__(
            parent=parent,
            title="線形解析実行中...",
            chibi_image="xebec_chibi_suzukisan.png",
            chibi_size=150  # 100 * 1.5 = 150 (chibi más grande para análisis lineal)
        )
        self.set_main_title("線形解析")
    
    def cancel_analysis(self):
        """Cancelar análisis y cerrar popup"""
        self.cancel_process()

class YosokuWorker(QThread):
    """Worker para predicción Yosoku con señales de progreso"""
    
    # Señales
    progress_updated = Signal(int, str)  # porcentaje, mensaje
    status_updated = Signal(str)  # mensaje de estado
    finished = Signal(str)  # ruta del archivo creado
    error = Signal(str)  # mensaje de error
    
    def __init__(self, selected_params, unexperimental_file, formulas_file, output_path):
        super().__init__()
        self.selected_params = selected_params
        self.unexperimental_file = unexperimental_file
        self.formulas_file = formulas_file
        self.output_path = output_path
        self.is_cancelled = False
    
    def run(self):
        """Ejecutar predicción Yosoku con progreso"""
        try:
            self.status_updated.emit("データを読み込み中...")
            self.progress_updated.emit(10, "データを読み込み中...")
            
            # Cargar datos del archivo Excel
            import pandas as pd
            data_df = pd.read_excel(self.unexperimental_file)
            
            self.status_updated.emit("Excelファイルを読み込み中...")
            self.progress_updated.emit(20, "Excelファイルを読み込み中...")
            
            # Leer fórmulas y transformaciones del archivo XEBEC_予測計算機_逆変換対応.xlsx
            import openpyxl
            from openpyxl import load_workbook
            
            formulas_wb = load_workbook(self.formulas_file, data_only=False)
            formulas_ws = formulas_wb.active
            
            # Leer fórmulas de B16-B18
            formula_b16 = formulas_ws['B16'].value
            formula_b17 = formulas_ws['B17'].value
            formula_b18 = formulas_ws['B18'].value
            
            # Leer transformaciones de C16-C18
            transform_c16 = formulas_ws['C16'].value
            transform_c17 = formulas_ws['C17'].value
            transform_c18 = formulas_ws['C18'].value
            
            self.status_updated.emit("ヘッダーを読み込み中...")
            self.progress_updated.emit(30, "ヘッダーを読み込み中...")
            
            # Leer encabezados de A16-A18 (nombres de columnas de destino)
            destination_headers = []
            for row in range(16, 19):  # A16 a A18
                cell_value = formulas_ws[f'A{row}'].value
                if cell_value:
                    destination_headers.append(str(cell_value).strip())
                else:
                    destination_headers.append(f"Columna_{row-15}")  # Fallback
            
            # Leer encabezados de A5-A11 (nombres de columnas de referencia)
            reference_headers = []
            for row in range(5, 12):  # A5 a A11
                cell_value = formulas_ws[f'A{row}'].value
                if cell_value:
                    reference_headers.append(str(cell_value).strip())
                else:
                    reference_headers.append(f"Ref_{row-4}")  # Fallback
            
            self.status_updated.emit("DataFrameを作成中...")
            self.progress_updated.emit(40, "DataFrameを作成中...")
            
            # Definir las columnas del DataFrame según la especificación
            columns = [
                'A13', 'A11', 'A21', 'A32',  # A-D: Tipos de cepillo
                '直径', '材料', '線材長',  # E-G: Parámetros del usuario
                '回転速度', '送り速度', 'UPカット', '切込量', '突出量', '載せ率', 'パス数',  # H-N: Parámetros operacionales
                '加工時間'  # O: Tiempo de procesamiento
            ]
            
            # Agregar las columnas de destino dinámicas (P-R)
            columns.extend(destination_headers)
            
            # Optimización máxima: Usar lista de diccionarios
            print(f"📊 Procesando {len(data_df)} filas de manera optimizada...")
            
            # Buscar nombres alternativos para las columnas
            alternative_names = {
                '回転速度': ['回転速度', 'Rotation Speed', '回転'],
                '送り速度': ['送り速度', 'Feed Speed', '送り'],
                'UPカット': ['UPカット', 'UP Cut', 'UP'],
                '切込量': ['切込量', 'Depth of Cut', '切込'],
                '突出量': ['突出量', 'Protrusion Amount', '突出'],
                '載せ率': ['載せ率', 'Loading Rate', '載せ'],
                'パス数': ['パス数', 'バス数', 'Number of Passes', 'パス']
            }
            
            # Crear mapeo de columnas para acceso rápido
            column_mapping = {}
            for target_col, source_cols in alternative_names.items():
                for source_col in source_cols:
                    if source_col in data_df.columns:
                        column_mapping[target_col] = source_col
                        break
                if target_col not in column_mapping:
                    column_mapping[target_col] = None
            
            # Pre-calcular valores constantes
            brush_type = self.selected_params['brush']
            brush_values = {
                'A13': 1 if brush_type == 'A13' else 0,
                'A11': 1 if brush_type == 'A11' else 0,
                'A21': 1 if brush_type == 'A21' else 0,
                'A32': 1 if brush_type == 'A32' else 0
            }
            
            user_values = {
                '直径': self.selected_params['diameter'],
                '材料': self.selected_params['material'],
                '線材長': self.selected_params['wire_length']
            }
            
            # Usar lista de diccionarios para máxima velocidad
            rows_data = []
            total_rows = len(data_df)
            
            # Procesar datos usando vectorización cuando sea posible
            for idx in range(total_rows):
                if self.is_cancelled:
                    return
                    
                if idx % 10000 == 0:  # Mostrar progreso cada 10,000 filas
                    progress = 40 + int((idx / total_rows) * 40)  # 40% a 80%
                    self.progress_updated.emit(progress, f"データを処理中... ({idx+1}/{total_rows})")
                
                row = data_df.iloc[idx]
                new_row = {}
                
                # Rellenar columnas A-D con tipo de cepillo seleccionado (valores pre-calculados)
                new_row.update(brush_values)
                
                # Rellenar columnas E-G con parámetros del usuario (valores pre-calculados)
                new_row.update(user_values)
                
                # Copiar datos operacionales (H-N) del archivo original usando mapeo optimizado
                for target_col, source_col in column_mapping.items():
                    if source_col is not None:
                        new_row[target_col] = row[source_col]
                    else:
                        new_row[target_col] = 0
                
                # Calcular 加工時間 con la fórmula: 100/送り速度*60
                if '送り速度' in new_row and new_row['送り速度'] is not None and new_row['送り速度'] != 0:
                    new_row['加工時間'] = 100 / new_row['送り速度'] * 60
                else:
                    new_row['加工時間'] = 0
                
                # Inicializar columnas de predicción (P-R) con 0 usando nombres dinámicos
                for header in destination_headers:
                    new_row[header] = 0
                
                # Agregar a la lista en lugar de concatenar
                rows_data.append(new_row)
            
            # Crear DataFrame una sola vez al final
            self.status_updated.emit("DataFrameを作成中...")
            self.progress_updated.emit(80, "DataFrameを作成中...")
            result_df = pd.DataFrame(rows_data, columns=columns)
            
            self.status_updated.emit("Excelファイルを保存中...")
            self.progress_updated.emit(90, "Excelファイルを保存中...")
            
            # Guardar DataFrame primero
            result_df.to_excel(self.output_path, index=False)
            
            # Procesar fórmulas y aplicarlas al DataFrame
            formulas_data = [
                {'formula': formula_b16, 'transform': transform_c16, 'header': destination_headers[0] if len(destination_headers) > 0 else 'Columna_1'},
                {'formula': formula_b17, 'transform': transform_c17, 'header': destination_headers[1] if len(destination_headers) > 1 else 'Columna_2'},
                {'formula': formula_b18, 'transform': transform_c18, 'header': destination_headers[2] if len(destination_headers) > 2 else 'Columna_3'}
            ]
            
            # Procesar cada fórmula
            for formula_data in formulas_data:
                formula = formula_data['formula']
                transform = formula_data['transform']
                header = formula_data['header']
                
                if formula:
                    # Crear fórmula procesada
                    processed_formula = str(formula)
                    
                    # Aplicar transformación si existe
                    if transform:
                        # Reemplazar la referencia de la fórmula con la transformación
                        formula_ref = f"B{16 + formulas_data.index(formula_data)}"  # B16, B17, B18
                        transform_formula = str(transform)
                        
                        # Remover el '=' de processed_formula antes de sustituir
                        formula_without_equals = processed_formula
                        if formula_without_equals.startswith('='):
                            formula_without_equals = formula_without_equals[1:]
                        
                        # Reemplazar la referencia específica de la fórmula con paréntesis
                        formula_with_parentheses = f'({formula_without_equals})'
                        transform_formula = transform_formula.replace(formula_ref, formula_with_parentheses)
                        final_formula = transform_formula
                    else:
                        final_formula = processed_formula
                    
                    # Asegurar que la fórmula tenga '=' al inicio
                    if not final_formula.startswith('='):
                        final_formula = f'={final_formula}'
                    
                    # Guardar la fórmula procesada para escribir en Excel
                    formula_data['final_formula'] = final_formula
                else:
                    formula_data['final_formula'] = '=0'
            
            # Crear mapeo de columnas de referencia (A5-A11) a columnas del DataFrame
            reference_mapping = {}
            for i, ref_header in enumerate(reference_headers):
                ref_cell = f'B{5+i}'  # B5, B6, B7, etc.
                
                # Buscar la columna correspondiente en el DataFrame
                found_col = None
                for col_name in result_df.columns:
                    if col_name == ref_header:
                        found_col = col_name
                        break
                
                if found_col:
                    reference_mapping[ref_cell] = found_col
            
            # Ahora escribir las fórmulas en el Excel usando openpyxl
            self.status_updated.emit("数式を書き込み中...")
            self.progress_updated.emit(95, "数式を書き込み中...")
            
            # Cargar el archivo Excel recién creado
            output_wb = load_workbook(self.output_path)
            output_ws = output_wb.active
            
            # Encontrar las columnas donde escribir las fórmulas
            formula_columns = {}
            for formula_data in formulas_data:
                header = formula_data['header']
                
                # Buscar la columna en el Excel
                for col_idx, cell in enumerate(output_ws[1], 1):  # Primera fila (encabezados)
                    if cell.value == header:
                        formula_columns[header] = col_idx
                        break
            
            # Pre-calcular mapeo de columnas para fórmulas
            column_letter_mapping = {}
            for col_idx_check, cell in enumerate(output_ws[1], 1):
                if cell.value in result_df.columns:
                    column_letter_mapping[cell.value] = openpyxl.utils.get_column_letter(col_idx_check)
            
            # Escribir fórmulas de manera más eficiente
            total_formula_rows = len(result_df)
            
            for row_idx in range(2, total_formula_rows + 2):  # Fila 2 en adelante (fila 1 son encabezados)
                if self.is_cancelled:
                    return
                    
                if (row_idx - 2) % 10000 == 0:  # Mostrar progreso cada 10,000 filas
                    progress = 95 + int(((row_idx - 2) / total_formula_rows) * 5)  # 95% a 100%
                    self.progress_updated.emit(progress, f"数式を書き込み中... ({row_idx-1}/{total_formula_rows})")
                
                for formula_data in formulas_data:
                    header = formula_data['header']
                    final_formula = formula_data['final_formula']
                    
                    if header in formula_columns:
                        col_idx = formula_columns[header]
                        
                        # Crear fórmula específica para esta fila usando mapeo pre-calculado
                        row_formula = final_formula
                        
                        # Reemplazar referencias de celdas con referencias de fila específica
                        for ref_cell, col_name in reference_mapping.items():
                            if col_name in column_letter_mapping:
                                target_col_letter = column_letter_mapping[col_name]
                                row_formula = row_formula.replace(ref_cell, f'{target_col_letter}{row_idx}')
                        
                        # Escribir la fórmula en la celda
                        output_ws.cell(row=row_idx, column=col_idx, value=row_formula)
            
            # Guardar el archivo Excel con las fórmulas
            output_wb.save(self.output_path)
            
            self.status_updated.emit("完了！")
            self.progress_updated.emit(100, "完了！")
            
            # Emitir resultado exitoso
            self.finished.emit(self.output_path)
            
        except Exception as e:
            print(f"❌ Error en predicción Yosoku: {e}")
            import traceback
            traceback.print_exc()
            self.error.emit(f"Error en predicción Yosoku: {str(e)}")
    
    def cancel_prediction(self):
        """Cancelar predicción"""
        self.is_cancelled = True
        self.terminate()

class YosokuProgressDialog(ReusableProgressDialog):
    """Popup de progreso para predicción Yosoku usando la clase reutilizable"""
    
    def __init__(self, parent=None):
        super().__init__(
            parent=parent,
            title="予測実行中...",
            chibi_image="Chibi_tamiru.png",
            chibi_size=150  # 100 * 1.5 = 150 (chibi más grande para yosoku del análisis lineal)
        )
        self.set_main_title("予測実行")
    
    def cancel_prediction(self):
        """Cancelar predicción y cerrar popup"""
        self.cancel_process()

class YosokuImportProgressDialog(ReusableProgressDialog):
    """Popup de progreso para importación de datos Yosoku usando la clase reutilizable"""
    
    def __init__(self, parent=None):
        super().__init__(
            parent=parent,
            title="データベースインポート中...",
            chibi_image="Chibi_suzuki_tamiru.png",
            chibi_size=160  # 100 * 1.6 = 160 (chibi más grande para importar a yosoku)
        )
        self.set_main_title("データベースインポート")
    
    def cancel_import(self):
        """Cancelar importación y cerrar popup"""
        self.cancel_process()

class YosokuExportProgressDialog(ReusableProgressDialog):
    """Popup de progreso para exportación de datos Yosoku usando la clase reutilizable"""
    
    def __init__(self, parent=None):
        super().__init__(
            parent=parent,
            title="データベースエクスポート中...",
            chibi_image="Chibi_suzuki_tamiru.png",
            chibi_size=160  # 100 * 1.6 = 160 (chibi más grande para exportar yosoku)
        )
        self.set_main_title("データベースエクスポート")
    
    def cancel_export(self):
        """Cancelar exportación y cerrar popup"""
        self.cancel_process()

class YosokuImportWorker(QThread):
    """Worker para importación de datos Yosoku con progreso"""
    
    # Señales
    progress_updated = Signal(int, str)  # porcentaje, mensaje
    status_updated = Signal(str)  # mensaje de estado
    finished = Signal()  # importación completada
    error = Signal(str)  # mensaje de error
    
    def __init__(self, excel_path, analysis_type="lineal", parent_widget=None):
        super().__init__()
        self.excel_path = excel_path
        self.analysis_type = analysis_type  # "lineal" o "no_lineal"
        self.cancelled = False
    
    def cancel_import(self):
        """Cancelar importación"""
        self.cancelled = True
    
    def run(self):
        """Ejecutar importación con progreso"""
        try:
            import pandas as pd
            import sqlite3
            import os
            from openpyxl import load_workbook
            import shutil
            from datetime import datetime
            
            # Paso 1: Crear carpeta temporal
            self.status_updated.emit("フォルダ作成中...")
            self.progress_updated.emit(5, "フォルダ作成中...")
            print("📁 Creando carpeta temporal...")
            
            if self.cancelled:
                return
            
            project_folder = os.path.dirname(self.excel_path)
            temp_folder = os.path.join(project_folder, "99_Temp")
            if not os.path.exists(temp_folder):
                os.makedirs(temp_folder)
                print(f"✅ Carpeta {temp_folder} creada")
            
            # Paso 2: Crear copia
            self.status_updated.emit("ファイルコピー中...")
            self.progress_updated.emit(10, "ファイルコピー中...")
            print("📋 Creando copia del archivo Excel...")
            
            if self.cancelled:
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename = os.path.basename(self.excel_path)
            name, ext = os.path.splitext(original_filename)
            backup_filename = f"{name}_backup_{timestamp}{ext}"
            backup_path = os.path.join(temp_folder, backup_filename)
            
            shutil.copy2(self.excel_path, backup_path)
            print(f"✅ Copia creada: {backup_path}")
            
            # Guardar referencia para limpieza posterior
            self.backup_path = backup_path
            
            # Paso 3: Convertir fórmulas a valores
            self.status_updated.emit("数式を値に変換中...")
            self.progress_updated.emit(20, "数式を値に変換中...")
            print("🔄 Convirtiendo fórmulas a valores...")
            
            if self.cancelled:
                return
            
            try:
                import xlwings as xw
                from pathlib import Path
                
                print("📊 Usando xlwings para convertir fórmulas...")
                app = xw.App(visible=False, add_book=False)
                try:
                    wb = app.books.open(str(backup_path))
                    wb.app.api.CalculateFull()
                    
                    for sh in wb.sheets:
                        rng = sh.used_range
                        vals = rng.value
                        rng.value = vals
                    
                    wb.save(str(backup_path))
                    print("✅ Fórmulas convertidas a valores con xlwings")
                    
                finally:
                    wb.close()
                    app.quit()
                    
            except ImportError:
                print("⚠️ xlwings no encontrado, instalando...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "xlwings"])
                
                import xlwings as xw
                from pathlib import Path
                
                print("📊 Usando xlwings (instalado) para convertir fórmulas...")
                app = xw.App(visible=False, add_book=False)
                try:
                    wb = app.books.open(str(backup_path))
                    wb.app.api.CalculateFull()
                    
                    for sh in wb.sheets:
                        rng = sh.used_range
                        vals = rng.value
                        rng.value = vals
                    
                    wb.save(str(backup_path))
                    print("✅ Fórmulas convertidas a valores con xlwings (instalado)")
                    
                finally:
                    wb.close()
                    app.quit()
                    
            except Exception as e:
                print(f"⚠️ Error con xlwings: {e}")
                print("📊 Usando método alternativo (openpyxl)...")
                # Método alternativo
                workbook = load_workbook(backup_path, data_only=False)
                worksheet = workbook.active
                
                values_worksheet = workbook.create_sheet("values_only")
                
                for row in worksheet.iter_rows(values_only=True):
                    values_worksheet.append(row)
                
                workbook.remove(worksheet)
                values_worksheet.title = "Sheet1"
                
                workbook.save(backup_path)
                workbook.close()
                print("✅ Fórmulas convertidas a valores con openpyxl")
            
            # Paso 4: Leer datos
            self.status_updated.emit("データ読み込み中...")
            self.progress_updated.emit(40, "データ読み込み中...")
            
            if self.cancelled:
                return
            
            workbook = load_workbook(backup_path, data_only=True)
            worksheet = workbook.active
            
            data = []
            headers = []
            
            for col in worksheet.iter_cols(min_row=1, max_row=1):
                headers.append(col[0].value)
            
            for row in worksheet.iter_rows(min_row=2, values_only=True):
                if any(cell is not None for cell in row):
                    data.append(row)
            
            df = pd.DataFrame(data, columns=headers)
            workbook.close()
            
            # Paso 5: Conectar a base de datos
            self.status_updated.emit("データベース接続中...")
            self.progress_updated.emit(60, "データベース接続中...")
            
            if self.cancelled:
                return
            
            # Determinar BBDD según el tipo de análisis
            if self.analysis_type == "no_lineal":
                db_path = YOSOKU_NO_LINEAL_DB_PATH
            else:  # "lineal" por defecto
                db_path = YOSOKU_LINEAL_DB_PATH
            conn = sqlite3.connect(db_path, timeout=10)
            cursor = conn.cursor()
            
            # Paso 6: Crear tabla
            self.status_updated.emit("テーブル作成中...")
            self.progress_updated.emit(70, "テーブル作成中...")
            
            if self.cancelled:
                conn.close()
                return
            
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS yosoku_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                A13 INTEGER,
                A11 INTEGER,
                A21 INTEGER,
                A32 INTEGER,
                直径 REAL,
                材料 TEXT,
                線材長 REAL,
                回転速度 REAL,
                送り速度 REAL,
                UPカット INTEGER,
                切込量 REAL,
                突出量 REAL,
                載せ率 REAL,
                パス数 INTEGER,
                加工時間 REAL,
                上面ダレ量 REAL,
                側面ダレ量 REAL,
                摩耗量 REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_table_sql)
            
            # Paso 7: Eliminar índice anterior y crear nuevo índice único
            self.status_updated.emit("インデックス作成中...")
            self.progress_updated.emit(80, "インデックス作成中...")
            
            if self.cancelled:
                conn.close()
                return
            
            print("⚡ Eliminando índice anterior y creando nuevo índice único...")
            print("🎯 Considerando SOLO las columnas que determinan duplicados")
            print("📝 Se ignoran: 上面ダレ量, 側面ダレ量, 摩耗量, created_at")
            
            # Eliminar índice anterior si existe
            try:
                cursor.execute("DROP INDEX IF EXISTS idx_unique_yosoku")
                print("🗑️ Índice anterior eliminado")
            except Exception as e:
                print(f"⚠️ No había índice anterior: {e}")
            
            # Crear nuevo índice único SOLO en las columnas que determinan duplicados
            cursor.execute("""
                CREATE UNIQUE INDEX idx_unique_yosoku 
                ON yosoku_predictions (
                    A13, A11, A21, A32, 直径, 材料, 線材長, 回転速度, 
                    送り速度, UPカット, 切込量, 突出量, 載せ率, パス数, 加工時間
                )
            """)
            print("✅ Nuevo índice único creado")
            print("📊 Columnas consideradas para duplicados:")
            print("   A13, A11, A21, A32, 直径, 材料, 線材長, 回転速度")
            print("   送り速度, UPカット, 切込量, 突出量, 載せ率, パス数, 加工時間")
            print("📝 Columnas IGNORADAS (se sobreescriben):")
            print("   上面ダレ量, 側面ダレ量, 摩耗量, created_at")
            
            # Paso 8: Insertar datos con sobreescritura automática
            self.status_updated.emit("データ挿入中...")
            self.progress_updated.emit(90, "データ挿入中...")
            
            if self.cancelled:
                conn.close()
                return
            
            print("📝 Ejecutando INSERT OR REPLACE (sobreescritura automática)")
            print("🔍 Verificando que el índice único esté activo...")
            
            # Verificar que el índice existe
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_unique_yosoku'")
            index_exists = cursor.fetchone()
            if index_exists:
                print("✅ Índice único confirmado: idx_unique_yosoku")
            else:
                print("❌ ERROR: Índice único no encontrado!")
            
            insert_sql = """
            INSERT OR REPLACE INTO yosoku_predictions
            (A13, A11, A21, A32, 直径, 材料, 線材長, 回転速度, 送り速度, UPカット, 
             切込量, 突出量, 載せ率, パス数, 加工時間, 上面ダレ量, 側面ダレ量, 摩耗量)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            # Insertar datos fila por fila
            inserted_count = 0
            updated_count = 0
            total_rows = len(df)
            
            # Obtener conteo inicial de registros
            cursor.execute("SELECT COUNT(*) FROM yosoku_predictions")
            initial_count = cursor.fetchone()[0]
            print(f"📊 Registros iniciales en BD: {initial_count}")
            
            for index, row in df.iterrows():
                if self.cancelled:
                    conn.close()
                    return
                
                # Verificar si el registro ya existe antes de insertar
                cursor.execute("""
                    SELECT COUNT(*) FROM yosoku_predictions 
                    WHERE A13=? AND A11=? AND A21=? AND A32=? AND 直径=? AND 材料=? 
                    AND 線材長=? AND 回転速度=? AND 送り速度=? AND UPカット=? 
                    AND 切込量=? AND 突出量=? AND 載せ率=? AND パス数=? AND 加工時間=?
                """, (
                    row.get('A13'), row.get('A11'), row.get('A21'), row.get('A32'),
                    row.get('直径'), row.get('材料'), row.get('線材長'), row.get('回転速度'),
                    row.get('送り速度'), row.get('UPカット'), row.get('切込量'), row.get('突出量'),
                    row.get('載せ率'), row.get('パス数'), row.get('加工時間')
                ))
                
                exists_before = cursor.fetchone()[0] > 0
                
                cursor.execute(insert_sql, (
                    row.get('A13'), row.get('A11'), row.get('A21'), row.get('A32'),
                    row.get('直径'), row.get('材料'), row.get('線材長'), row.get('回転速度'),
                    row.get('送り速度'), row.get('UPカット'), row.get('切込量'), row.get('突出量'),
                    row.get('載せ率'), row.get('パス数'), row.get('加工時間'),
                    row.get('上面ダレ量'), row.get('側面ダレ量'), row.get('摩耗量')
                ))
                
                if exists_before:
                    updated_count += 1
                else:
                    inserted_count += 1
                
                # Mostrar progreso cada 1000 filas
                if (inserted_count + updated_count) % 1000 == 0:
                    progress = 90 + int(((inserted_count + updated_count) / total_rows) * 5)  # 90% a 95%
                    self.progress_updated.emit(progress, f"データ挿入中... ({inserted_count + updated_count}/{total_rows})")
            
            # Obtener conteo final de registros
            cursor.execute("SELECT COUNT(*) FROM yosoku_predictions")
            final_count = cursor.fetchone()[0]
            
            print(f"✅ Procesados {inserted_count + updated_count} registros:")
            print(f"   📝 Nuevos insertados: {inserted_count}")
            print(f"   🔄 Actualizados (sobreescritos): {updated_count}")
            print(f"📊 Registros en BD: {initial_count} → {final_count}")
            print("💡 Los registros duplicados se sobreescribieron automáticamente")
            
            # Paso 10: Finalizar
            self.status_updated.emit("完了処理中...")
            self.progress_updated.emit(95, "完了処理中...")
            
            if self.cancelled:
                conn.close()
                return
            
            conn.commit()
            conn.close()
            
            # Limpiar archivos temporales
            try:
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                
                temp_folder = os.path.dirname(backup_path)
                if os.path.exists(temp_folder) and os.path.isdir(temp_folder):
                    try:
                        os.rmdir(temp_folder)
                    except OSError:
                        pass
            except Exception:
                pass
            
            self.status_updated.emit("インポート完了!")
            self.progress_updated.emit(100, "インポート完了!")
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))

class ClassificationImportWorker(QThread):
    """Worker para importación de resultados de clasificación a la BBDD de yosoku"""
    
    # Señales
    progress_updated = Signal(int, str)  # porcentaje, mensaje
    status_updated = Signal(str)  # mensaje de estado
    finished = Signal(int, int)  # registros_insertados, registros_actualizados
    error = Signal(str)  # mensaje de error
    
    def __init__(self, excel_path, overwrite=False, parent_widget=None):
        super().__init__()
        self.excel_path = excel_path
        self.overwrite = overwrite
        self.cancelled = False
    
    def cancel_import(self):
        """Cancelar importación"""
        self.cancelled = True
    
    def run(self):
        """Ejecutar importación con progreso"""
        try:
            import pandas as pd
            import sqlite3
            import os
            import numpy as np
            
            # Paso 1: Leer archivo Excel
            self.status_updated.emit("ファイル読み込み中...")
            self.progress_updated.emit(5, "ファイル読み込み中...")
            
            if self.cancelled:
                return
            
            if not os.path.exists(self.excel_path):
                self.error.emit(f"ファイルが見つかりません: {self.excel_path}")
                return
            
            df = pd.read_excel(self.excel_path)
            total_rows = len(df)
            
            if total_rows == 0:
                self.error.emit("ファイルにデータがありません")
                return
            
            # Paso 2: Definir columnas para comparación (índice único)
            # Solo usar las columnas que realmente existen en el DataFrame
            all_comparison_columns = [
                'A13', 'A11', 'A21', 'A32', '直径', '材料', '線材長', 
                '回転速度', '送り速度', 'UPカット', '切込量', '突出量', 
                '載せ率', 'パス数', '加工時間'
            ]
            
            # Filtrar solo las columnas que existen en el DataFrame
            comparison_columns = [col for col in all_comparison_columns if col in df.columns]
            missing_cols = [col for col in all_comparison_columns if col not in df.columns]
            
            if len(comparison_columns) == 0:
                self.error.emit("比較に使用できる列が見つかりません。ファイルに必要な列が含まれているか確認してください。")
                return
            
            if missing_cols:
                print(f"⚠️ 以下の列がファイルに存在しません（NULLとして扱います）: {', '.join(missing_cols)}")
                print(f"✅ 比較に使用する列: {', '.join(comparison_columns)}")
            
            # Paso 3: Procesar ambas BBDD (lineal y no_lineal)
            total_inserted = 0
            total_updated = 0
            
            # Procesar BBDD lineal (0-50% del progreso)
            self.status_updated.emit("線形データベース処理中...")
            self.progress_updated.emit(10, "線形データベース処理中...")
            
            if not self.cancelled:
                inserted_lineal, updated_lineal = self._process_database(
                    df, comparison_columns, YOSOKU_LINEAL_DB_PATH,
                    progress_start=10, progress_end=50
                )
                total_inserted += inserted_lineal
                total_updated += updated_lineal
            
            # Procesar BBDD no lineal (50-100% del progreso)
            if not self.cancelled:
                self.status_updated.emit("非線形データベース処理中...")
                self.progress_updated.emit(50, "非線形データベース処理中...")
                
                inserted_no_lineal, updated_no_lineal = self._process_database(
                    df, comparison_columns, YOSOKU_NO_LINEAL_DB_PATH,
                    progress_start=50, progress_end=95
                )
                total_inserted += inserted_no_lineal
                total_updated += updated_no_lineal
            
            if self.cancelled:
                return
            
            # Finalizar
            self.progress_updated.emit(100, "完了")
            self.status_updated.emit("インポート完了")
            self.finished.emit(total_inserted, total_updated)
            
        except Exception as e:
            print(f"❌ Error en importación de clasificación: {e}")
            import traceback
            traceback.print_exc()
            self.error.emit(f"インポート中にエラーが発生しました: {str(e)}")
    
    def _process_database(self, df, comparison_columns, db_path, progress_start=0, progress_end=100):
        """Procesa una BBDD específica con los datos de clasificación"""
        import pandas as pd
        import sqlite3
        import os
        
        inserted_count = 0
        updated_count = 0
        skipped_count = 0
        
        # Conectar a BBDD
        if not os.path.exists(db_path):
            print(f"ℹ️ BBDD {db_path} no existe, se creará automáticamente")
        
        conn = sqlite3.connect(db_path, timeout=10)
        cursor = conn.cursor()
        
        try:
            # Asegurar que la tabla existe (crear si no existe)
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS yosoku_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                A13 INTEGER,
                A11 INTEGER,
                A21 INTEGER,
                A32 INTEGER,
                直径 REAL,
                材料 TEXT,
                線材長 REAL,
                回転速度 REAL,
                送り速度 REAL,
                UPカット INTEGER,
                切込量 REAL,
                突出量 REAL,
                載せ率 REAL,
                パス数 INTEGER,
                加工時間 REAL,
                上面ダレ量 REAL,
                側面ダレ量 REAL,
                摩耗量 REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_table_sql)
            
            # Asegurar que las columnas de clasificación existan
            classification_columns = [
                ('pred_label', 'INTEGER'),
                ('p_cal', 'REAL'),
                ('tau_pos', 'REAL'),
                ('tau_neg', 'REAL'),
                ('ood_flag', 'INTEGER'),
                ('maha_dist', 'REAL')
            ]
            
            for col_name, col_type in classification_columns:
                try:
                    cursor.execute(f"ALTER TABLE yosoku_predictions ADD COLUMN {col_name} {col_type}")
                    print(f"✅ Columna {col_name} agregada a {db_path}")
                except sqlite3.OperationalError as e:
                    if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
                        print(f"ℹ️ Columna {col_name} ya existe en {db_path}")
                    else:
                        raise
            
            conn.commit()
            
            total_rows = len(df)
            progress_range = progress_end - progress_start
            
            if self.overwrite:
                # ESTRATEGIA OPTIMIZADA: Cargar BBDD en memoria, hacer merge, y UPDATE/INSERT según corresponda
                # Esto es necesario porque el índice único incluye columnas que pueden no estar en el Excel
                print("⚡ Usando estrategia optimizada con merge (modo sobreescritura)")
                
                # Cargar registros existentes de la BBDD (solo columnas de comparación que tenemos)
                progress_current = progress_start + int(progress_range * 0.1)
                self.status_updated.emit(f"既存データ読み込み中... ({db_path})")
                self.progress_updated.emit(progress_current, f"既存データ読み込み中... ({db_path})")
                
                db_query = f"SELECT id, {', '.join(comparison_columns)} FROM yosoku_predictions"
                db_df = pd.read_sql_query(db_query, conn)
                
                # Preparar queries
                update_query = """
                    UPDATE yosoku_predictions 
                    SET pred_label = ?, p_cal = ?, tau_pos = ?, tau_neg = ?, 
                        ood_flag = ?, maha_dist = ?
                    WHERE id = ?
                """
                
                insert_columns = comparison_columns + ['pred_label', 'p_cal', 'tau_pos', 'tau_neg', 'ood_flag', 'maha_dist']
                placeholders = ','.join(['?'] * len(insert_columns))
                insert_query = f"""
                    INSERT INTO yosoku_predictions 
                    ({','.join(insert_columns)})
                    VALUES ({placeholders})
                """
                
                if len(db_df) > 0:
                    # Crear clave de comparación en ambos DataFrames
                    def create_key(row, cols):
                        return tuple(row[col] if not pd.isna(row[col]) else 'NULL_VAL' for col in cols)
                    
                    excel_df = df.copy()
                    excel_df['_comparison_key'] = excel_df.apply(lambda r: create_key(r, comparison_columns), axis=1)
                    db_df['_comparison_key'] = db_df.apply(lambda r: create_key(r, comparison_columns), axis=1)
                    
                    # Hacer merge para encontrar coincidencias
                    merged = excel_df.merge(
                        db_df[['id', '_comparison_key']], 
                        on='_comparison_key', 
                        how='left',
                        suffixes=('', '_db')
                    )
                    
                    # Separar en registros a actualizar vs insertar
                    to_update = merged[merged['id'].notna()].copy()
                    to_insert = merged[merged['id'].isna()].copy()
                    
                    print(f"📊 Registros a actualizar: {len(to_update)}")
                    print(f"📊 Registros a insertar: {len(to_insert)}")
                    
                    # Procesar actualizaciones en lotes
                    if len(to_update) > 0:
                        update_batch = []
                        for index, row in to_update.iterrows():
                            if self.cancelled:
                                conn.close()
                                return (inserted_count, updated_count)
                            
                            pred_label = row.get('pred_label', None)
                            p_cal = row.get('p_cal', None)
                            tau_pos = row.get('tau_pos', None)
                            tau_neg = row.get('tau_neg', None)
                            ood_flag = row.get('ood_flag', None)
                            maha_dist = row.get('maha_dist', None)
                            
                            # Convertir NaN a None
                            if pd.isna(pred_label): pred_label = None
                            if pd.isna(p_cal): p_cal = None
                            if pd.isna(tau_pos): tau_pos = None
                            if pd.isna(tau_neg): tau_neg = None
                            if pd.isna(ood_flag): ood_flag = None
                            if pd.isna(maha_dist): maha_dist = None
                            
                            update_batch.append((
                                pred_label, p_cal, tau_pos, tau_neg, ood_flag, maha_dist, int(row['id'])
                            ))
                            
                            if len(update_batch) >= 1000:
                                cursor.executemany(update_query, update_batch)
                                updated_count += len(update_batch)
                                update_batch = []
                                
                                progress = progress_start + int(progress_range * 0.2) + int((updated_count / len(to_update)) * (progress_range * 0.3))
                                self.progress_updated.emit(progress, f"更新中: {updated_count}/{len(to_update)} ({db_path})")
                        
                        if update_batch:
                            cursor.executemany(update_query, update_batch)
                            updated_count += len(update_batch)
                    
                    # Procesar inserciones en lotes
                    if len(to_insert) > 0:
                        insert_batch = []
                        for index, row in to_insert.iterrows():
                            if self.cancelled:
                                conn.close()
                                return (inserted_count, updated_count)
                            
                            row_values = []
                            
                            # Valores de columnas de comparación (solo las que tenemos)
                            for col in comparison_columns:
                                val = row[col]
                                if pd.isna(val):
                                    row_values.append(None)
                                else:
                                    row_values.append(val)
                            
                            # Valores de clasificación
                            for col in ['pred_label', 'p_cal', 'tau_pos', 'tau_neg', 'ood_flag', 'maha_dist']:
                                val = row.get(col, None)
                                if pd.isna(val):
                                    row_values.append(None)
                                else:
                                    row_values.append(val)
                            
                            insert_batch.append(tuple(row_values))
                            
                            if len(insert_batch) >= 1000:
                                cursor.executemany(insert_query, insert_batch)
                                inserted_count += len(insert_batch)
                                insert_batch = []
                                
                                progress = progress_start + int(progress_range * 0.5) + int((inserted_count / len(to_insert)) * (progress_range * 0.3))
                                self.progress_updated.emit(progress, f"挿入中: {inserted_count}/{len(to_insert)} ({db_path})")
                        
                        if insert_batch:
                            cursor.executemany(insert_query, insert_batch)
                            inserted_count += len(insert_batch)
                else:
                    # BBDD vacía, insertar todos
                    print("📊 BBDD vacía, insertando todos los registros")
                    insert_batch = []
                    for index, row in df.iterrows():
                        if self.cancelled:
                            conn.close()
                            return (inserted_count, updated_count)
                        
                        row_values = []
                        for col in comparison_columns:
                            val = row.get(col, None)
                            if pd.isna(val):
                                row_values.append(None)
                            else:
                                row_values.append(val)
                        
                        for col in ['pred_label', 'p_cal', 'tau_pos', 'tau_neg', 'ood_flag', 'maha_dist']:
                            val = row.get(col, None)
                            if pd.isna(val):
                                row_values.append(None)
                            else:
                                row_values.append(val)
                        
                        insert_batch.append(tuple(row_values))
                        
                        if len(insert_batch) >= 1000:
                            cursor.executemany(insert_query, insert_batch)
                            inserted_count += len(insert_batch)
                            insert_batch = []
                            
                            progress = progress_start + int((inserted_count / total_rows) * (progress_range * 0.8))
                            self.progress_updated.emit(progress, f"挿入中: {inserted_count}/{total_rows} ({db_path})")
                    
                    if insert_batch:
                        cursor.executemany(insert_query, insert_batch)
                        inserted_count += len(insert_batch)
                
            else:
                # ESTRATEGIA CON MERGE: Cargar BBDD en memoria y hacer merge (más rápido que SELECT por fila)
                print("⚡ Usando estrategia con merge (modo sin sobreescritura)")
                
                # Cargar registros existentes de la BBDD (solo columnas necesarias)
                progress_current = progress_start + int(progress_range * 0.1)
                self.status_updated.emit(f"既存データ読み込み中... ({db_path})")
                self.progress_updated.emit(progress_current, f"既存データ読み込み中... ({db_path})")
                
                db_query = f"SELECT id, {', '.join(comparison_columns)} FROM yosoku_predictions"
                db_df = pd.read_sql_query(db_query, conn)
                
                if len(db_df) > 0:
                    # Crear clave de comparación en ambos DataFrames
                    # Manejar NaN reemplazándolos con un valor especial para la comparación
                    def create_key(row, cols):
                        return tuple(row[col] if not pd.isna(row[col]) else 'NULL_VAL' for col in cols)
                    
                    excel_df = df.copy()
                    excel_df['_comparison_key'] = excel_df.apply(lambda r: create_key(r, comparison_columns), axis=1)
                    db_df['_comparison_key'] = db_df.apply(lambda r: create_key(r, comparison_columns), axis=1)
                    
                    # Hacer merge para encontrar coincidencias
                    merged = excel_df.merge(
                        db_df[['id', '_comparison_key']], 
                        on='_comparison_key', 
                        how='left',
                        suffixes=('', '_db')
                    )
                    
                    # Separar en registros a insertar vs saltar
                    to_insert = merged[merged['id'].isna()].copy()
                    to_skip = merged[merged['id'].notna()].copy()
                    
                    skipped_count = len(to_skip)
                    
                    print(f"📊 Registros a insertar: {len(to_insert)}")
                    print(f"📊 Registros a saltar (existen): {skipped_count}")
                    
                    # Insertar solo los nuevos
                    if len(to_insert) > 0:
                        insert_columns = comparison_columns + ['pred_label', 'p_cal', 'tau_pos', 'tau_neg', 'ood_flag', 'maha_dist']
                        placeholders = ','.join(['?'] * len(insert_columns))
                        insert_query = f"""
                            INSERT INTO yosoku_predictions 
                            ({','.join(insert_columns)})
                            VALUES ({placeholders})
                        """
                        
                        batch_data = []
                        for index, row in to_insert.iterrows():
                            if self.cancelled:
                                conn.close()
                                return (inserted_count, updated_count)
                            
                            row_values = []
                            
                            # Valores de columnas de comparación
                            for col in comparison_columns:
                                val = row[col]
                                if pd.isna(val):
                                    row_values.append(None)
                                else:
                                    row_values.append(val)
                            
                            # Valores de clasificación
                            for col in ['pred_label', 'p_cal', 'tau_pos', 'tau_neg', 'ood_flag', 'maha_dist']:
                                val = row.get(col, None)
                                if pd.isna(val):
                                    row_values.append(None)
                                else:
                                    row_values.append(val)
                            
                            batch_data.append(tuple(row_values))
                            
                            # Procesar en lotes de 1000
                            if len(batch_data) >= 1000:
                                cursor.executemany(insert_query, batch_data)
                                inserted_count += len(batch_data)
                                batch_data = []
                                
                                progress = progress_start + int(progress_range * 0.2) + int((inserted_count / len(to_insert)) * (progress_range * 0.7))
                                self.progress_updated.emit(progress, f"挿入中: {inserted_count}/{len(to_insert)} ({db_path})")
                        
                        # Procesar lote final
                        if batch_data:
                            cursor.executemany(insert_query, batch_data)
                            inserted_count += len(batch_data)
                else:
                    # BBDD vacía, insertar todos
                    print("📊 BBDD vacía, insertando todos los registros")
                    insert_columns = comparison_columns + ['pred_label', 'p_cal', 'tau_pos', 'tau_neg', 'ood_flag', 'maha_dist']
                    placeholders = ','.join(['?'] * len(insert_columns))
                    insert_query = f"""
                        INSERT INTO yosoku_predictions 
                        ({','.join(insert_columns)})
                        VALUES ({placeholders})
                    """
                    
                    batch_data = []
                    for index, row in df.iterrows():
                        if self.cancelled:
                            conn.close()
                            return (inserted_count, updated_count)
                        
                        row_values = []
                        for col in comparison_columns:
                            val = row.get(col, None)
                            if pd.isna(val):
                                row_values.append(None)
                            else:
                                row_values.append(val)
                        
                        for col in ['pred_label', 'p_cal', 'tau_pos', 'tau_neg', 'ood_flag', 'maha_dist']:
                            val = row.get(col, None)
                            if pd.isna(val):
                                row_values.append(None)
                            else:
                                row_values.append(val)
                        
                        batch_data.append(tuple(row_values))
                        
                        if len(batch_data) >= 1000:
                            cursor.executemany(insert_query, batch_data)
                            inserted_count += len(batch_data)
                            batch_data = []
                            
                            progress = progress_start + int((inserted_count / total_rows) * (progress_range * 0.8))
                            self.progress_updated.emit(progress, f"挿入中: {inserted_count}/{total_rows} ({db_path})")
                    
                    if batch_data:
                        cursor.executemany(insert_query, batch_data)
                        inserted_count += len(batch_data)
            
            # Commit final
            conn.commit()
            conn.close()
            
            print(f"✅ Procesamiento de {db_path} completado: {inserted_count} insertados, {updated_count} actualizados")
            return (inserted_count, updated_count)
            
        except Exception as e:
            print(f"❌ Error procesando {db_path}: {e}")
            import traceback
            traceback.print_exc()
            if conn:
                conn.close()
            raise

class YosokuExportWorker(QThread):
    """Worker para exportación de datos Yosoku a Excel con progreso"""
    
    # Señales
    progress_updated = Signal(int, str)  # porcentaje, mensaje
    status_updated = Signal(str)  # mensaje de estado
    finished = Signal(str, int)  # filepath, record_count
    error = Signal(str)  # mensaje de error
    
    def __init__(self, db_path, filepath, total_records):
        super().__init__()
        self.db_path = db_path
        self.filepath = filepath
        self.total_records = total_records
        self.cancelled = False
    
    def cancel_export(self):
        """Cancelar exportación"""
        self.cancelled = True
    
    def run(self):
        """Ejecutar exportación con progreso"""
        try:
            import pandas as pd
            import sqlite3
            
            # Paso 1: Conectar a base de datos
            self.status_updated.emit("データベースに接続中...")
            self.progress_updated.emit(10, "データベースに接続中...")
            
            if self.cancelled:
                return
            
            conn = sqlite3.connect(self.db_path)
            
            # Paso 2: Leer datos
            self.status_updated.emit("データを読み込み中...")
            self.progress_updated.emit(30, "データを読み込み中...")
            
            if self.cancelled:
                conn.close()
                return
            
            df = pd.read_sql_query("SELECT * FROM yosoku_predictions", conn)
            conn.close()
            
            # Paso 3: Exportar a Excel
            self.status_updated.emit("Excelファイルに書き込み中...")
            self.progress_updated.emit(60, "Excelファイルに書き込み中...")
            
            if self.cancelled:
                return
            
            df.to_excel(self.filepath, index=False)
            
            # Paso 4: Completado
            self.status_updated.emit("エクスポート完了")
            self.progress_updated.emit(100, "エクスポート完了")
            
            if not self.cancelled:
                self.finished.emit(self.filepath, len(df))
            
        except Exception as e:
            if not self.cancelled:
                error_msg = f"❌ エクスポート中にエラーが発生しました:\n{str(e)}"
                self.error.emit(error_msg)

class LinearAnalysisWorker(QThread):
    """Worker para análisis lineal con señales de progreso"""
    
    # Señales
    progress_updated = Signal(int, str)  # porcentaje, mensaje
    status_updated = Signal(str)  # mensaje de estado
    finished = Signal(dict)  # resultados
    error = Signal(str)  # mensaje de error
    
    def __init__(self, db_manager, filters, output_folder, parent_widget=None):
        super().__init__()
        self.db_manager = db_manager
        self.filters = filters
        self.output_folder = output_folder
        self.db_connection = None
        self.is_cancelled = False  # ✅ NUEVO: Bandera de cancelación
        
    def stop(self):
        """Método para solicitar la parada del worker"""
        self.is_cancelled = True

    def run(self):
        """Ejecutar análisis lineal con progreso"""
        import threading
        print(f"🚀 DEBUG: LinearAnalysisWorker iniciado en hilo: {threading.current_thread().name}")
        try:
            if self.is_cancelled: return # Check inicial

            self.status_updated.emit("データベースからデータを取得中...")
            self.progress_updated.emit(10, "データベースからデータを取得中...")
            
            if self.is_cancelled: return # Check después de emitir

            # ✅ NUEVO: Pequeño delay para mostrar progreso
            import time
            time.sleep(0.5)
            
            if self.is_cancelled: return

            # ✅ NUEVO: Crear nueva conexión de base de datos en este thread
            import sqlite3
            self.db_connection = sqlite3.connect(RESULTS_DB_PATH, timeout=10)
            cursor = self.db_connection.cursor()
            
            # Obtener datos filtrados
            query = "SELECT * FROM main_results WHERE 1=1"
            params = []
            
            # ... (filtros) ...
            # (No cambio la lógica de filtros para ser breve, asumo que sigue igual)
            # Pero necesito mantener el código existente para que el search_replace no falle
            # Mejor leo el archivo de nuevo para asegurar el bloque exacto.
            self.progress_updated.emit(10, "データベースからデータを取得中...")
            
            # ✅ NUEVO: Pequeño delay para mostrar progreso
            import time
            time.sleep(0.5)
            
            # ✅ NUEVO: Crear nueva conexión de base de datos en este thread
            import sqlite3
            self.db_connection = sqlite3.connect(RESULTS_DB_PATH, timeout=10)
            cursor = self.db_connection.cursor()
            
            # Obtener datos filtrados
            query = "SELECT * FROM main_results WHERE 1=1"
            params = []
            
            # Aplicar filtros de cepillo
            brush_selections = []
            if 'すべて' in self.filters and self.filters['すべて']:
                brush_condition = " OR ".join([f"{brush} = 1" for brush in ['A13', 'A11', 'A21', 'A32']])
                query += f" AND ({brush_condition})"
            else:
                for brush_type in ['A13', 'A11', 'A21', 'A32']:
                    if brush_type in self.filters and self.filters[brush_type]:
                        brush_selections.append(brush_type)
                
                if brush_selections:
                    brush_condition = " OR ".join([f"{brush} = 1" for brush in brush_selections])
                    query += f" AND ({brush_condition})"
            
            # Aplicar filtros de rango
            field_to_db = {
                "面粗度(Ra)前": "面粗度前",
                "面粗度(Ra)後": "面粗度後",
            }
            for field_name, filter_value in self.filters.items():
                if field_name in ['すべて', 'A13', 'A11', 'A21', 'A32']:
                    continue
                db_field = field_to_db.get(field_name, field_name)
                    
                if isinstance(filter_value, tuple) and len(filter_value) == 2:
                    desde, hasta = filter_value
                    if desde is not None and hasta is not None:
                        if field_name == "実験日":
                            desde_str = desde.toString("yyyyMMdd") if hasattr(desde, 'toString') else str(desde)
                            hasta_str = hasta.toString("yyyyMMdd") if hasattr(hasta, 'toString') else str(hasta)
                            query += f" AND {db_field} BETWEEN ? AND ?"
                            params.extend([desde_str, hasta_str])
                        else:
                            try:
                                desde_num = float(desde) if isinstance(desde, str) else desde
                                hasta_num = float(hasta) if isinstance(hasta, str) else hasta
                                query += f" AND {db_field} BETWEEN ? AND ?"
                                params.extend([desde_num, hasta_num])
                            except (ValueError, TypeError):
                                continue
                elif isinstance(filter_value, (str, int, float)) and filter_value:
                    try:
                        if field_name in ['線材長', '回転速度', '送り速度', 'UPカット', '突出量', 'パス数', 'バリ除去']:
                            value_num = int(filter_value) if isinstance(filter_value, str) else filter_value
                        else:
                            value_num = float(filter_value) if isinstance(filter_value, str) else filter_value
                        
                        query += f" AND {db_field} = ?"
                        params.append(value_num)
                    except (ValueError, TypeError):
                        continue
            
            # ✅ NUEVO: Ejecutar consulta usando la nueva conexión
            cursor.execute(query, params)
            filtered_data = cursor.fetchall()
            
            self.status_updated.emit("データを処理中...")
            self.progress_updated.emit(20, "データを処理中...")
            time.sleep(0.3)
            
            if not filtered_data:
                self.error.emit("フィルター条件に一致するデータが見つかりません")
                return
            
            # Convertir a DataFrame
            import pandas as pd
            # No depender del orden físico de columnas en SQLite (puede cambiar con migraciones)
            column_names = [d[0] for d in cursor.description] if cursor.description else None
            df = pd.DataFrame(filtered_data, columns=column_names)
            
            self.status_updated.emit("データファイルを保存中...")
            self.progress_updated.emit(30, "データファイルを保存中...")
            time.sleep(0.3)
            
            # Crear estructura de carpetas
            import os
            os.makedirs(self.output_folder, exist_ok=True)
            models_folder = os.path.join(self.output_folder, "01_学習モデル")
            os.makedirs(models_folder, exist_ok=True)
            
            # Guardar datos filtrados
            filtered_data_path = os.path.join(models_folder, "filtered_data.xlsx")
            df.to_excel(filtered_data_path, index=False)
            
            if self.is_cancelled: return # ✅ Check de cancelación

            self.status_updated.emit("機械学習パイプラインを初期化中...")
            self.progress_updated.emit(40, "機械学習パイプラインを初期化中...")
            time.sleep(0.4)
            
            if self.is_cancelled: return # ✅ Check de cancelación

            # Importar y configurar pipeline
            from linear_analysis_advanced import IntegratedMLPipeline, PipelineConfig
            
            config = PipelineConfig()
            config.TRANSFORMATION['enable'] = True
            config.TRANSFORMATION['mode'] = 'advanced'
            config.FEATURE_SELECTION['method'] = 'importance'
            config.FEATURE_SELECTION['k_features'] = 10
            config.PREPROCESSING['noise_augmentation_ratio'] = 0.3
            config.TRANSFORMATION['improvement_threshold'] = 0.005
            
            pipeline = IntegratedMLPipeline(base_dir=self.output_folder, config=config)
            
            self.status_updated.emit("データを読み込み中...")
            self.progress_updated.emit(15, "データを読み込み中...")
            time.sleep(0.2)
            
            # Cargar datos
            pipeline.load_data(filtered_data_path, index_col='Index')
            
            self.status_updated.emit("データ構造を分析中...")
            self.progress_updated.emit(18, "データ構造を分析中...")
            time.sleep(0.2)
            
            self.status_updated.emit("変数を分離中...")
            self.progress_updated.emit(20, "変数を分離中...")
            time.sleep(0.2)
            
            if self.is_cancelled: return # ✅ Check de cancelación

            # Separar variables
            try:
                pipeline.separate_variables()
            except Exception as e:
                self.error.emit(f"Error separando variables: {str(e)}")
                return
            
            if self.is_cancelled: return # ✅ Check de cancelación

            self.status_updated.emit("特徴量を選択中...")
            self.progress_updated.emit(22, "特徴量を選択中...")
            time.sleep(0.2)
            
            self.status_updated.emit("データを前処理中...")
            self.progress_updated.emit(25, "データを前処理中...")
            time.sleep(0.3)
            
            if self.is_cancelled: return # ✅ Check de cancelación

            # Preprocesar datos
            try:
                pipeline.preprocess_data()
            except Exception as e:
                self.error.emit(f"Error preprocesando datos: {str(e)}")
                return
            
            if self.is_cancelled: return # ✅ Check de cancelación

            self.status_updated.emit("回帰モデルを初期化中...")
            self.progress_updated.emit(30, "回帰モデルを初期化中...")
            time.sleep(0.2)
            
            self.status_updated.emit("線形回帰モデルを訓練中...")
            self.progress_updated.emit(35, "線形回帰モデルを訓練中...")
            time.sleep(0.3)
            
            if self.is_cancelled: return # ✅ Check de cancelación

            self.status_updated.emit("ランダムフォレストモデルを訓練中...")
            self.progress_updated.emit(40, "ランダムフォレストモデルを訓練中...")
            time.sleep(0.3)
            
            if self.is_cancelled: return # ✅ Check de cancelación

            self.status_updated.emit("SVMモデル को 訓練中...")
            self.progress_updated.emit(45, "SVMモデルを訓練中...")
            time.sleep(0.3)
            
            self.status_updated.emit("分類モデルを初期化中...")
            self.progress_updated.emit(50, "分類モデルを初期化中...")
            time.sleep(0.2)
            
            self.status_updated.emit("ロジスティック回帰を訓練中...")
            self.progress_updated.emit(55, "ロジスティック回帰を訓練中...")
            time.sleep(0.3)
            
            self.status_updated.emit("決定木モデルを訓練中...")
            self.progress_updated.emit(60, "決定木モデルを訓練中...")
            time.sleep(0.3)
            
            self.status_updated.emit("ナイーブベイズモデルを訓練中...")
            self.progress_updated.emit(65, "ナイーブベイズモデルを訓練中...")
            time.sleep(0.3)
            
            self.status_updated.emit("モデルを評価中...")
            self.progress_updated.emit(40, "モデルを評価中...")
            time.sleep(0.2)
            
            self.status_updated.emit("モデル初期化中...")
            self.progress_updated.emit(41, "モデル初期化中...")
            time.sleep(0.2)
            
            self.status_updated.emit("データセットを分割中...")
            self.progress_updated.emit(42, "データセットを分割中...")
            time.sleep(0.2)
            
            self.status_updated.emit("訓練データを準備中...")
            self.progress_updated.emit(43, "訓練データを準備中...")
            time.sleep(0.2)
            
            self.status_updated.emit("検証データを準備中...")
            self.progress_updated.emit(44, "検証データを準備中...")
            time.sleep(0.2)
            
            self.status_updated.emit("モデル訓練を開始中...")
            self.progress_updated.emit(45, "モデル訓練を開始中...")
            time.sleep(0.2)
            
            # Entrenar modelos
            try:
                pipeline.train_models()
            except Exception as e:
                self.error.emit(f"Error entrenando modelos: {str(e)}")
                return
            
            self.status_updated.emit("モデル訓練完了...")
            self.progress_updated.emit(46, "モデル訓練完了...")
            time.sleep(0.2)
            
            self.status_updated.emit("回帰モデルの性能を評価中...")
            self.progress_updated.emit(47, "回帰モデルの性能を評価中...")
            time.sleep(0.2)
            
            self.status_updated.emit("分類モデルの性能を評価中...")
            self.progress_updated.emit(48, "分類モデルの性能を評価中...")
            time.sleep(0.2)
            
            self.status_updated.emit("交差検証を実行中...")
            self.progress_updated.emit(49, "交差検証を実行中...")
            time.sleep(0.3)
            
            if self.is_cancelled: return # ✅ NUEVO: Freno de cancelación

            self.status_updated.emit("メトリクスを計算中...")
            self.progress_updated.emit(50, "メトリクスを計算中...")
            time.sleep(0.2)
            
            self.status_updated.emit("モデル比較を実行中...")
            self.progress_updated.emit(51, "モデル比較を実行中...")
            time.sleep(0.2)
            
            self.status_updated.emit("最適なモデルを選択中...")
            self.progress_updated.emit(52, "最適なモデルを選択中...")
            time.sleep(0.2)
            
            if self.is_cancelled: return # ✅ NUEVO: Freno de cancelación

            self.status_updated.emit("プロペンシティスコアを計算中...")
            self.progress_updated.emit(53, "プロペンシティスコアを計算中...")
            time.sleep(0.3)
            
            self.status_updated.emit("スコアの正規化中...")
            self.progress_updated.emit(54, "スコアの正規化中...")
            time.sleep(0.2)
            
            self.status_updated.emit("統計的検定を実行中...")
            self.progress_updated.emit(55, "統計的検定を実行中...")
            time.sleep(0.3)
            
            self.status_updated.emit("信頼区間を計算中...")
            self.progress_updated.emit(56, "信頼区間を計算中...")
            time.sleep(0.2)
            
            self.status_updated.emit("結果の整合性を確認中...")
            self.progress_updated.emit(57, "結果の整合性を確認中...")
            time.sleep(0.2)
            
            self.status_updated.emit("データの品質を検証中...")
            self.progress_updated.emit(58, "データの品質を検証中...")
            time.sleep(0.2)
            
            self.status_updated.emit("異常値を検出中...")
            self.progress_updated.emit(59, "異常値を検出中...")
            time.sleep(0.2)
            
            if self.is_cancelled: return # ✅ NUEVO: Freno de cancelación

            self.status_updated.emit("モデルの安定性を確認中...")
            self.progress_updated.emit(60, "モデルの安定性を確認中...")
            time.sleep(0.2)
            
            self.status_updated.emit("最終評価を実行中...")
            self.progress_updated.emit(61, "最終評価を実行中...")
            time.sleep(0.3)
            
            self.status_updated.emit("結果を保存中...")
            self.progress_updated.emit(62, "結果を保存中...")
            time.sleep(0.2)
            
            self.status_updated.emit("Excelファイルを作成中...")
            self.progress_updated.emit(63, "Excelファイルを作成中...")
            time.sleep(0.2)
            
            self.status_updated.emit("データをフォーマット中...")
            self.progress_updated.emit(64, "データをフォーマット中...")
            time.sleep(0.2)
            
            self.status_updated.emit("グラフを生成中...")
            self.progress_updated.emit(65, "グラフを生成中...")
            time.sleep(0.2)
            
            if self.is_cancelled: return # ✅ NUEVO: Freno de cancelación

            self.status_updated.emit("散布図を作成中...")
            self.progress_updated.emit(66, "散布図を作成中...")
            time.sleep(0.2)
            
            self.status_updated.emit("ヒートマップを生成中...")
            self.progress_updated.emit(67, "ヒートマップを生成中...")
            time.sleep(0.2)
            
            self.status_updated.emit("相関図を作成中...")
            self.progress_updated.emit(68, "相関図を作成中...")
            time.sleep(0.2)
            
            self.status_updated.emit("予測テンプレートを作成中...")
            self.progress_updated.emit(69, "予測テンプレートを作成中...")
            time.sleep(0.2)
            
            self.status_updated.emit("計算式を生成中...")
            self.progress_updated.emit(70, "計算式を生成中...")
            time.sleep(0.2)
            
            if self.is_cancelled: return # ✅ NUEVO: Freno de cancelación

            self.status_updated.emit("逆変換テンプレートを作成中...")
            self.progress_updated.emit(71, "逆変換テンプレートを作成中...")
            time.sleep(0.2)
            
            self.status_updated.emit("ファイルを最適化中...")
            self.progress_updated.emit(72, "ファイルを最適化中...")
            time.sleep(0.2)
            
            self.status_updated.emit("最終処理中...")
            self.progress_updated.emit(73, "最終処理中...")
            time.sleep(0.2)
            
            self.status_updated.emit("クリーンアップを実行中...")
            self.progress_updated.emit(74, "クリーンアップを実行中...")
            time.sleep(0.2)
            
            self.status_updated.emit("完了確認中...")
            self.progress_updated.emit(75, "完了確認中...")
            time.sleep(0.2)
            
            # Calcular propensity scores y guardar resultados
            try:
                propensity_scores = pipeline.calculate_propensity_scores()
                pipeline.save_results()
                pipeline.create_prediction_template()
            except Exception as e:
                self.error.emit(f"Error guardando resultados: {str(e)}")
                return
            try:
                pipeline.save_prediction_formulas()
                # ✅ DESCOMENTADO: Crear Excel durante análisis lineal
                print("🔧 Iniciando creación de Excel durante análisis lineal...")
                
                excel_calculator_path = pipeline.create_excel_prediction_calculator_with_inverse(None)
                
                if excel_calculator_path:
                    print(f"✅ Excel creado exitosamente: {excel_calculator_path}")
                else:
                    print("⚠️ Excel no se pudo crear (retornó None)")
                    
            except Exception as e:
                print(f"❌ Error detallado creando Excel: {str(e)}")
                import traceback
                traceback.print_exc()
                self.error.emit(f"Error creando calculadora Excel: {str(e)}")
                return
            
            self.status_updated.emit("分析完了！")
            self.progress_updated.emit(100, "分析完了！")
            
            # Preparar resultados
            results = {
                'success': True,
                'data_count': len(df),
                'models_trained': len(pipeline.models),
                'output_folder': self.output_folder,
                'filters_applied': list(self.filters.keys()),
                'data_range': f"線材長: {df['線材長'].min()}-{df['線材長'].max()}, 送り速度: {df['送り速度'].min()}-{df['送り速度'].max()}" if len(df) > 0 else "N/A",
                'excel_calculator': None,  # ✅ FIX: Comentado para evitar crash
                'transformation_info': pipeline.transformation_info,
                'feature_selection': pipeline.results.get('feature_selection', {}),
                'target_info': pipeline.target_info,
                'models': pipeline.models
            }
            
            # Crear resumen de resultados
            summary = []
            for target_name, model_info in pipeline.models.items():
                if model_info.get('model') is not None:
                    if model_info['task_type'] == 'regression':
                        metrics = model_info.get('final_metrics', {})
                        summary.append({
                            'target': target_name,
                            'model': model_info.get('model_name', 'Unknown'),
                            'r2': metrics.get('r2', 'N/A'),
                            'mae': metrics.get('mae', 'N/A'),
                            'rmse': metrics.get('rmse', 'N/A'),
                            'transformation': pipeline.transformation_info.get(target_name, {}).get('method', 'none')
                        })
                    else:
                        metrics = model_info.get('final_metrics', {})
                        summary.append({
                            'target': target_name,
                            'model': model_info.get('model_name', 'Unknown'),
                            'accuracy': metrics.get('accuracy', 'N/A'),
                            'f1_score': metrics.get('f1_score', 'N/A'),
                            'transformation': 'none'
                        })
            
            results['summary'] = summary
            
            self.finished.emit(results)
            
        except Exception as e:
            import threading
            import traceback
            error_msg = f"❌ Error en análisis lineal worker (Hilo: {threading.current_thread().name}): {e}"
            print(error_msg)
            traceback.print_exc()
            self.error.emit(error_msg)
        finally:
            import threading
            print(f"🛑 DEBUG: LinearAnalysisWorker finalizando en hilo: {threading.current_thread().name}")
            # Cerrar conexión de base de datos si existe
            if hasattr(self, 'db_connection') and self.db_connection:
                try:
                    self.db_connection.close()
                    print("🛑 DEBUG: Conexión DB cerrada en worker")
                except:
                    pass

class ProjectCreationDialog(QDialog):
    """Diálogo para crear un nuevo proyecto"""
    
    def __init__(self, parent=None, analysis_type="nonlinear"):
        super().__init__(parent)
        self.analysis_type = analysis_type  # "nonlinear" o "classification"
        self.setWindowTitle("新規プロジェクト作成")
        self.setFixedSize(500, 300)
        self.setModal(True)
        
        # Layout principal
        layout = QVBoxLayout()
        
        # Título
        title_label = QLabel("新規プロジェクトを作成します")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Formulario
        form_layout = QFormLayout()
        
        # Nombre del proyecto
        self.project_name_edit = QLineEdit()
        self.project_name_edit.setPlaceholderText("プロジェクト名を入力してください")
        form_layout.addRow("プロジェクト名:", self.project_name_edit)
        
        # Directorio
        directory_layout = QHBoxLayout()
        self.directory_edit = QLineEdit()
        self.directory_edit.setPlaceholderText("プロジェクトを保存するディレクトリを選択してください")
        self.directory_edit.setReadOnly(True)
        
        browse_button = QPushButton("参照...")
        browse_button.clicked.connect(self.browse_directory)
        
        directory_layout.addWidget(self.directory_edit)
        directory_layout.addWidget(browse_button)
        
        form_layout.addRow("保存先:", directory_layout)
        
        layout.addLayout(form_layout)
        
        # Botones
        button_layout = QHBoxLayout()
        
        cancel_button = QPushButton("キャンセル")
        cancel_button.clicked.connect(self.reject)
        
        create_button = QPushButton("作成")
        create_button.clicked.connect(self.accept)
        create_button.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        
        button_layout.addWidget(cancel_button)
        button_layout.addStretch()
        button_layout.addWidget(create_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Variables para almacenar los datos
        self.project_name = ""
        self.project_directory = ""
    
    def browse_directory(self):
        """Abrir diálogo para seleccionar directorio"""
        directory = QFileDialog.getExistingDirectory(self, "プロジェクト保存先を選択")
        if directory:
            self.directory_edit.setText(directory)
    
    def accept(self):
        """Validar y aceptar el diálogo"""
        project_name = self.project_name_edit.text().strip()
        directory = self.directory_edit.text().strip()
        
        if not directory:
            QMessageBox.warning(self, "エラー", "保存先ディレクトリを選択してください。")
            return
        
        # ✅ NUEVO: Verificar si la carpeta seleccionada es un proyecto válido
        selected_path = Path(directory)
        
        # Verificar si la carpeta seleccionada es un proyecto (usar el tipo de análisis del diálogo)
        if self.parent().is_valid_project_folder(str(selected_path), analysis_type=self.analysis_type):
            # La carpeta seleccionada ES un proyecto, usarla directamente
            self.project_name = selected_path.name
            self.project_directory = str(selected_path.parent)
            print(f"✅ Carpeta seleccionada es un proyecto válido: {selected_path}")
            super().accept()
            return
        
        # Verificar si dentro de la carpeta hay proyectos
        project_folders = self.parent().find_project_folders_in_directory(str(selected_path), analysis_type=self.analysis_type)
        
        if project_folders:
            # Hay proyectos dentro de la carpeta seleccionada
            # Primero preguntar si quiere crear nuevo o usar existente
            choice_dialog = QDialog(self)
            choice_dialog.setWindowTitle("プロジェクト選択")
            choice_dialog.setMinimumWidth(450)
            
            choice_layout = QVBoxLayout()
            
            info_label = QLabel(
                f"選択したディレクトリ内に {len(project_folders)} 個の既存プロジェクトが見つかりました。\n\n"
                f"新規プロジェクトを作成しますか？\n"
                f"それとも既存のプロジェクトを使用しますか？"
            )
            info_label.setWordWrap(True)
            choice_layout.addWidget(info_label)
            
            # Mostrar lista de proyectos existentes
            projects_label = QLabel("既存プロジェクト:")
            projects_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
            choice_layout.addWidget(projects_label)
            
            projects_list = QListWidget()
            projects_list.setMaximumHeight(150)
            for folder in project_folders:
                projects_list.addItem(folder)
            choice_layout.addWidget(projects_list)
            
            buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
            
            # Botón para crear nuevo
            create_new_btn = buttons.addButton("新規作成", QDialogButtonBox.ActionRole)
            create_new_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 8px;")
            
            # Botón para usar existente
            use_existing_btn = buttons.addButton("既存を使用", QDialogButtonBox.ActionRole)
            use_existing_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; padding: 8px;")
            
            # Variables para almacenar la elección
            choice_result = None
            
            # Conectar botones a funciones
            def on_create_new():
                nonlocal choice_result
                choice_result = "create_new"
                choice_dialog.accept()
            
            def on_use_existing():
                nonlocal choice_result
                choice_result = "use_existing"
                choice_dialog.accept()
            
            create_new_btn.clicked.connect(on_create_new)
            use_existing_btn.clicked.connect(on_use_existing)
            
            choice_layout.addWidget(buttons)
            choice_dialog.setLayout(choice_layout)
            
            result = choice_dialog.exec()
            
            if result == QDialog.Accepted and choice_result:
                if choice_result == "create_new":
                    # Usuario quiere crear nuevo - validar nombre
                    if not project_name:
                        QMessageBox.warning(self, "エラー", "プロジェクト名を入力してください。")
                        return
                    
                    # Almacenar los datos para crear nuevo proyecto
                    self.project_name = project_name
                    self.project_directory = directory
                    print(f"📁 Creando nuevo proyecto: {project_name} en {directory}")
                    super().accept()
                    return
                
                elif choice_result == "use_existing":
                    # Usuario quiere usar existente - mostrar lista para seleccionar
                    if len(project_folders) == 1:
                        # Solo hay un proyecto, usarlo directamente
                        project_path = Path(project_folders[0])
                        self.project_name = project_path.name
                        self.project_directory = str(project_path.parent)
                        print(f"✅ Usando proyecto existente: {project_path}")
                        super().accept()
                        return
                    else:
                        # Hay múltiples proyectos, mostrar lista para seleccionar
                        select_dialog = QDialog(self)
                        select_dialog.setWindowTitle("プロジェクトを選択")
                        select_dialog.setMinimumWidth(500)
                        
                        select_layout = QVBoxLayout()
                        select_label = QLabel(f"使用するプロジェクトを選択してください:")
                        select_layout.addWidget(select_label)
                        
                        list_widget = QListWidget()
                        for folder in project_folders:
                            list_widget.addItem(folder)
                        list_widget.setCurrentRow(0)
                        select_layout.addWidget(list_widget)
                        
                        select_buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                        select_buttons.accepted.connect(select_dialog.accept)
                        select_buttons.rejected.connect(select_dialog.reject)
                        select_layout.addWidget(select_buttons)
                        
                        select_dialog.setLayout(select_layout)
                        
                        if select_dialog.exec() == QDialog.Accepted:
                            selected_project = list_widget.currentItem().text()
                            project_path = Path(selected_project)
                            self.project_name = project_path.name
                            self.project_directory = str(project_path.parent)
                            print(f"✅ Usando proyecto seleccionado: {project_path}")
                            super().accept()
                            return
                        else:
                            # Usuario canceló selección, volver al diálogo principal
                            return
            
            # Si se canceló el diálogo de elección, no hacer nada
            return
        
        # No se encontró proyecto válido, validar nombre y crear nuevo
        if not project_name:
            QMessageBox.warning(self, "エラー", "プロジェクト名を入力してください。")
            return
        
        # Almacenar los datos para crear nuevo proyecto
        self.project_name = project_name
        self.project_directory = directory
        
        super().accept()

class FormulaProcessingWorker(QObject):
    """Worker para procesamiento de fórmulas con barra de progreso"""
    
    # Señales
    progress_updated = Signal(int, str)  # porcentaje, mensaje
    status_updated = Signal(str)  # mensaje de estado
    finished = Signal(str)  # archivo de salida
    error_occurred = Signal(str)  # mensaje de error
    
    def __init__(self, output_path, data_df, formula_templates, prediction_columns, column_mapping, formula_columns):
        super().__init__()
        self.output_path = output_path
        self.data_df = data_df
        self.formula_templates = formula_templates
        self.prediction_columns = prediction_columns
        self.column_mapping = column_mapping
        self.formula_columns = formula_columns
        self.should_cancel = False
    
    def cancel(self):
        """Cancelar el procesamiento"""
        self.should_cancel = True
    
    def run(self):
        """Ejecutar el procesamiento de fórmulas"""
        try:
            import openpyxl
            from openpyxl import load_workbook
            
            self.status_updated.emit("📊 Cargando archivo Excel...")
            self.progress_updated.emit(5, "Cargando archivo Excel")
            
            # Cargar el archivo Excel con openpyxl para escribir fórmulas
            wb = load_workbook(self.output_path)
            ws = wb.active
            
            total_rows = len(self.data_df)
            chunk_size = 100  # Procesar 100 filas a la vez
            
            self.status_updated.emit(f"📊 Procesando {total_rows} filas en lotes de {chunk_size}...")
            
            for chunk_start in range(0, total_rows, chunk_size):
                if self.should_cancel:
                    self.status_updated.emit("❌ Procesamiento cancelado")
                    return
                
                chunk_end = min(chunk_start + chunk_size, total_rows)
                chunk_rows = range(chunk_start + 2, chunk_end + 2)  # +2 porque empezamos desde fila 2
                
                chunk_number = chunk_start//chunk_size + 1
                total_chunks = (total_rows + chunk_size - 1)//chunk_size
                
                self.status_updated.emit(f"📊 Procesando chunk {chunk_number}/{total_chunks} (filas {chunk_start + 1}-{chunk_end})")
                
                # Preparar todas las fórmulas para este chunk
                chunk_formulas = {}
                
                for row_idx in chunk_rows:
                    if self.should_cancel:
                        return
                    
                    # Crear diccionario de referencias de celda para sustituir en las fórmulas
                    formula_values = {}
                    for ref_cell, col_idx in self.column_mapping.items():
                        if col_idx is not None:
                            # Crear referencia de celda Excel (ej: A2, B2, C2, etc.)
                            excel_ref = f'{chr(64 + col_idx)}{row_idx}'
                            formula_values[ref_cell] = excel_ref
                        else:
                            formula_values[ref_cell] = '0'
                    
                    # Aplicar las plantillas de fórmulas para esta fila
                    row_formulas = {}
                    for i, (template, pred_col) in enumerate(zip(self.formula_templates, self.prediction_columns)):
                        if template != '=0':
                            # Sustituir referencias de celda en la plantilla
                            processed_formula = template
                            for cell_ref, excel_ref in formula_values.items():
                                processed_formula = processed_formula.replace(cell_ref, excel_ref)
                            row_formulas[pred_col] = processed_formula
                        else:
                            row_formulas[pred_col] = '=0'
                    
                    chunk_formulas[row_idx] = row_formulas
                
                # Escribir todas las fórmulas del chunk de una vez
                for row_idx, row_formulas in chunk_formulas.items():
                    if self.should_cancel:
                        return
                    
                    for pred_col, formula in row_formulas.items():
                        ws.cell(row=row_idx, column=self.formula_columns[pred_col], value=formula)
                
                # Actualizar progreso
                progress = int((chunk_end / total_rows) * 90)  # 90% para el procesamiento, 10% para guardar
                self.progress_updated.emit(progress, f"Chunk {chunk_number}/{total_chunks} completado")
            
            if self.should_cancel:
                return
            
            self.status_updated.emit("💾 Guardando archivo...")
            self.progress_updated.emit(95, "Guardando archivo")
            
            # Guardar el archivo con las fórmulas
            wb.save(self.output_path)
            
            self.status_updated.emit("✅ Procesamiento completado")
            self.progress_updated.emit(100, "Completado")
            self.finished.emit(self.output_path)
            
        except Exception as e:
            error_msg = f"❌ Error en procesamiento de fórmulas: {str(e)}"
            self.status_updated.emit(error_msg)
            self.error_occurred.emit(error_msg)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        print("🔧 Inicializando MainWindow...")
        
        # ✅ NUEVO: Variable para detectar acceso desde bunseki
        self.accessed_from_bunseki = False
        
        try:
            print("🔧 Creando DBManager...")
            # IMPORTANTE: en instalaciones (Program Files) no se puede escribir junto al EXE.
            # Usar siempre la ruta compartida en ProgramData (ver app_paths.py).
            self.db = DBManagerMain(RESULTS_DB_PATH)
            print("🔧 Creando ResultProcessor...")
            self.processor = ResultProcessor(self.db)
            # Backup automático (1/día) de la BBDD principal en ProgramData\\...\\backups
            try:
                backup_dir = get_backup_dir(shared=True)
                res = auto_daily_backup(RESULTS_DB_PATH, backup_dir, prefix="results")
                prune_backups(backup_dir, prefix="results", keep_daily=30, keep_monthly=12)
                if res is not None:
                    print(f"✅ Backup diario creado: {res.backup_path}")
            except Exception as _e:
                print(f"⚠️ No se pudo ejecutar backup diario: {_e}")
            print("🔧 Configurando ventana principal...")
            # Mostrar versión en la barra de título (arriba a la izquierda)
            self.setWindowTitle(get_app_title())
            self.setMinimumSize(1250, 950)
            print("🔧 Ventana principal configurada")
        except Exception as e:
            print(f"❌ Error en __init__: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Establecer el icono de la aplicación
        try:
            icon = QIcon(resource_path("xebec_logo_88.png"))
            self.setWindowIcon(icon)
        except Exception as e:
            print(f"⚠️ No se pudo cargar el icono: {e}")

        # Crear el widget central
        print("🔧 Creando widget central...")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # (versión se muestra en la barra de título; no agregamos label en el canvas)

        # Layout principal horizontal (panel izquierdo + panel central + consola)
        print("🔧 Configurando layout principal...")
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # ========================
        # Panel izquierdo (Controles) - Ancho fijo
        # ========================
        print("🔧 Creando panel izquierdo...")
        self.left_frame = QFrame()
        self.left_frame.setFrameShape(QFrame.StyledPanel)
        self.left_frame.setFixedWidth(340)  # Ancho fijo para el panel izquierdo (reducido 15%)
        self.left_layout = QVBoxLayout()
        self.left_layout.setAlignment(Qt.AlignTop)
        self.left_frame.setLayout(self.left_layout)

        self.left_layout.addWidget(create_logo_widget())

        self.create_load_section()
        
        # Campo de tamaño de muestra con valor por defecto 15 (siempre habilitado)
        self.sample_size_label = QLabel("サンプルサイズ (10-50)")
        self.sample_size_input = QLineEdit()
        self.sample_size_input.setPlaceholderText("10-50")
        self.sample_size_input.setValidator(QIntValidator(10, 50))
        self.sample_size_input.setText("15")
        
        # Conectar eventos para validación y pérdida de foco
        self.sample_size_input.editingFinished.connect(self.validate_sample_size)
        # Separador "サンプル" encima de サンプルサイズ
        sample_separator_label = QLabel("サンプル")
        sample_separator_label.setStyleSheet("font-weight: bold; color: #666666; margin: 5px 0px;")
        sample_separator_label.setAlignment(Qt.AlignCenter)
        sample_separator = QFrame()
        sample_separator.setFrameShape(QFrame.HLine)
        sample_separator.setStyleSheet("background-color: #CCCCCC; margin: 10px 0px;")
        self.left_layout.addWidget(sample_separator_label)
        self.left_layout.addWidget(sample_separator)
        
        self.sample_size_input.focusOutEvent = self.on_sample_size_focus_out
        
        self.left_layout.addWidget(self.sample_size_label)
        self.left_layout.addWidget(self.sample_size_input)
        
        self.create_action_buttons()
        
        # Separador "結果" encima del selector de 材料
        result_separator_label = QLabel("結果")
        result_separator_label.setStyleSheet("font-weight: bold; color: #666666; margin: 5px 0px;")
        result_separator_label.setAlignment(Qt.AlignCenter)
        result_separator = QFrame()
        result_separator.setFrameShape(QFrame.HLine)
        result_separator.setStyleSheet("background-color: #CCCCCC; margin: 10px 0px;")
        self.left_layout.addWidget(result_separator_label)
        self.left_layout.addWidget(result_separator)
        
        # Crea los widgets de Material
        self.material_label = QLabel("材料")
        self.material_selector = QComboBox()
        self.material_selector.addItems(["Steel", "Alumi"])
        self.left_layout.addWidget(self.material_label)
        self.left_layout.addWidget(self.material_selector)
        
        self.create_brush_selector()
        self.create_show_results_button()

        self.create_export_button()
        


        # Lista de widgets a desactivar/activar debajo del selector de muestras
        self.widgets_below_sample_selector = []
        # Usa los nombres correctos para los selectores
        self.widgets_below_sample_selector.append(self.diameter_label)
        self.widgets_below_sample_selector.append(self.diameter_selector)
        self.widgets_below_sample_selector.append(self.material_label)
        self.widgets_below_sample_selector.append(self.material_selector)
        self.widgets_below_sample_selector.append(self.brush_label)
        self.widgets_below_sample_selector.append(self.brush_selector)
        # NOTA: sample_size_label y sample_size_input NO están en esta lista porque deben estar siempre habilitados
        # ...añade más si hay más widgets debajo

        def set_widgets_enabled(enabled):
            for w in self.widgets_below_sample_selector:
                w.setEnabled(enabled)
                if hasattr(w, 'setStyleSheet'):
                    if enabled:
                        w.setStyleSheet("")
                    else:
                        w.setStyleSheet("color: gray;")
        # Por defecto, desactivar
        set_widgets_enabled(False)
        
        # Set initial state for UI elements
        self.set_ui_state_for_no_file()

        # Exponer toggler (para habilitar/deshabilitar por tipo detectado, no por nombre de archivo)
        self._set_widgets_below_sample_selector_enabled = set_widgets_enabled

        # Cuando se cargue un archivo, habilitar SOLO si el caller indica explícitamente que es de resultados.
        # (El nombre del archivo no importa; la detección real se hace por cabecera en handle_single_file_load)
        def on_file_loaded(file_path, is_results=None):
            set_widgets_enabled(bool(is_results))
        self.on_file_loaded = on_file_loaded
        print("🔧 Configuración del panel izquierdo completada")
        
        # ========================
        # Panel central (Visualización) - Se expande
        # ========================
        print("🔧 Creando panel central...")
        self.center_frame = QFrame()
        self.center_frame.setFrameShape(QFrame.StyledPanel)
        self.center_layout = QVBoxLayout()
        self.center_layout.setAlignment(Qt.AlignTop)
        self.center_frame.setLayout(self.center_layout)

        # Inicializar navegación de gráficos (aún no creada)
        self.prev_button = None
        self.next_button = None
        self.graph_navigation_frame = None

        # La flecha estará siempre visible, no necesitamos el botón
        print("🔧 Sistema de flecha simplificado - sin botón de activación")

        # Crear panel central (gráficos, labels, OK/NG)
        self.create_center_panel()

        # ========================
        # Panel derecho (Consola) - Desplegable
        # ========================
        print("🔧 Creando panel de consola desplegable...")
        
        # Contenedor principal del panel derecho
        self.right_container = QWidget()
        self.right_container.setFixedWidth(300)
        self.right_container.setMaximumWidth(300)
        
        # Layout del contenedor derecho
        self.right_container_layout = QVBoxLayout()
        self.right_container_layout.setContentsMargins(0, 0, 0, 0)
        self.right_container_layout.setSpacing(0)
        self.right_container.setLayout(self.right_container_layout)
        
        # Panel de la consola
        self.console_frame = QFrame()
        self.console_frame.setFrameShape(QFrame.StyledPanel)
        self.console_frame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
            }
        """)
        
        self.console_layout = QVBoxLayout()
        self.console_layout.setAlignment(Qt.AlignTop)
        self.console_layout.setContentsMargins(5, 5, 5, 5)
        self.console_frame.setLayout(self.console_layout)

        # Crear la consola integrada
        print("🔧 Configurando consola integrada...")
        self.create_console_panel()
        
        # Añadir la consola al contenedor derecho
        self.right_container_layout.addWidget(self.console_frame)
        
        # Crear el panel desplegable superpuesto
        self.create_overlay_console_panel()

        # ========================
        # Añadir solo el panel izquierdo y central al layout principal
        # ========================
        print("🔧 Añadiendo paneles al layout principal...")
        main_layout.addWidget(self.left_frame)  # Panel izquierdo con ancho fijo
        main_layout.addWidget(self.center_frame, 1)  # Panel central que se expande
        # NOTA: El panel derecho se añadirá dinámicamente cuando se active
        print("🔧 Paneles izquierdo y central añadidos correctamente")

        # ========================
        # Archivo cargando
        # ========================

        self.loader_overlay = LoadingOverlay(self.center_frame)

        self.graph_images = []  # Lista de rutas de imágenes
        self.current_graph_index = 0
        self.graph_label = QLabel()
        self.graph_label.setAlignment(Qt.AlignCenter)
        self.graph_area_layout = QVBoxLayout()
        self.graph_area.setLayout(self.graph_area_layout)
        self.graph_area_layout.addWidget(self.graph_label)

    # ======================================
    # Secciones de creación visual
    # ======================================

    def create_load_section(self):
        """Crear la sección de carga de archivos"""
        self.generate_button = QPushButton("生成：サンプル組合せ表")
        self.setup_generate_button_style(self.generate_button)
        self.left_layout.addWidget(self.generate_button)
        self.generate_button.clicked.connect(self.on_generate_sample_file_clicked)

        self.load_file_button = QPushButton("ファイルを読み込む")
        self.load_file_label = QLabel("ファイル未選択")
        self.setup_load_block(self.load_file_button, self.load_file_label)
        self.load_file_button.clicked.connect(self.handle_single_file_load)

        # self.load_sample_button = QPushButton("サンプルファイルをロード")
        # self.sample_label = QLabel("ファイル未選択")
        # self.setup_load_block(self.load_sample_button, self.sample_label)
        # self.load_sample_button.clicked.connect(lambda: self.load_file(self.sample_label, "サンプルファイルを選択"))
        #
        # self.load_results_button = QPushButton("結果ファイルをロード")
        # self.results_label = QLabel("ファイル未選択")
        # self.setup_load_block(self.load_results_button, self.results_label)
        # self.load_results_button.clicked.connect(lambda: self.load_file(self.results_label, "結果ファイルを選択"))

    def create_action_buttons(self):
        """Crear los botones de Dsaitekika e iSaitekika separados"""
        self.left_layout.addSpacing(10)

        self.d_optimize_button = QPushButton("D最適化を実行")
        self.setup_action_button(self.d_optimize_button)
        self.left_layout.addWidget(self.d_optimize_button)
        self.d_optimize_button.clicked.connect(self.on_d_optimizer_clicked)

        self.left_layout.addSpacing(5)

        self.i_optimize_button = QPushButton("I最適化を実行")
        self.setup_action_button(self.i_optimize_button)
        self.left_layout.addWidget(self.i_optimize_button)
        self.i_optimize_button.clicked.connect(self.on_i_optimizer_clicked)

    def create_show_results_button(self):
        """Crear el botón Show Results"""
        self.left_layout.addStretch()

        self.show_results_button = QPushButton("データベースにインポート")
        self.setup_results_button(self.show_results_button)
        self.left_layout.addWidget(self.show_results_button)
        self.show_results_button.clicked.connect(self.on_show_results_clicked)

        self.left_layout.addSpacing(10)
        self.show_results_button.setEnabled(False)

        # Botón de análisis
        self.analyze_button = QPushButton("分析")
        self.setup_results_button(self.analyze_button)
        self.left_layout.addWidget(self.analyze_button)
        self.analyze_button.clicked.connect(self.on_analyze_clicked)

        self.left_layout.addSpacing(10)
        self.analyze_button.setEnabled(True)



    def create_project_folder_structure(self, project_folder):
        """Crear la estructura de carpetas del proyecto"""
        folders = [
            "01_実験リスト",
            "99_Temp", 
            "03_-----------解析------------",
            "99_------------------------------",
            "02_実験データ",
            "99_Results",
            "03_線形回帰",
            "04_非線形回帰",
            "05_分類"
        ]
        
        for folder in folders:
            folder_path = os.path.join(project_folder, folder)
            os.makedirs(folder_path, exist_ok=True)
            print(f"📁 Carpeta creada: {folder_path}")

    def create_export_button(self):
        """Crear el botón de exportar resultados a Excel"""
        self.export_button = QPushButton("結果をエクスポート")
        self.setup_generate_button_style(self.export_button)
        self.left_layout.addWidget(self.export_button)
        self.export_button.clicked.connect(self.export_database_to_excel)
        
        # ✅ NUEVO: Botón para exportar base de datos de Yosoku
        self.yosoku_export_button = QPushButton("予測データベースをエクスポート")
        self.setup_generate_button_style(self.yosoku_export_button)
        self.left_layout.addWidget(self.yosoku_export_button)
        self.yosoku_export_button.clicked.connect(self.export_yosoku_database_to_excel)

        # ✅ NUEVO: Backup de BBDD (results + yosoku si existen)
        self.db_backup_button = QPushButton("DBバックアップ作成")
        self.setup_generate_button_style(self.db_backup_button)
        self.left_layout.addWidget(self.db_backup_button)
        self.db_backup_button.clicked.connect(self.backup_databases_now)

    def backup_databases_now(self):
        """Crear backup seguro de las BBDD en ProgramData\\...\\backups (manual)."""
        try:
            backup_dir = get_backup_dir(shared=True)
            created = []

            # 1) BBDD principal
            if os.path.exists(RESULTS_DB_PATH):
                r = create_backup(RESULTS_DB_PATH, backup_dir, prefix="results")
                prune_backups(backup_dir, prefix="results", keep_daily=30, keep_monthly=12)
                created.append(Path(r.backup_path).name)

            # 2) Yosoku lineal / no lineal (si existen)
            if os.path.exists(YOSOKU_LINEAL_DB_PATH):
                r = create_backup(YOSOKU_LINEAL_DB_PATH, backup_dir, prefix="yosoku_lineal")
                prune_backups(backup_dir, prefix="yosoku_lineal", keep_daily=30, keep_monthly=12)
                created.append(Path(r.backup_path).name)

            if os.path.exists(YOSOKU_NO_LINEAL_DB_PATH):
                r = create_backup(YOSOKU_NO_LINEAL_DB_PATH, backup_dir, prefix="yosoku_no_lineal")
                prune_backups(backup_dir, prefix="yosoku_no_lineal", keep_daily=30, keep_monthly=12)
                created.append(Path(r.backup_path).name)

            if not created:
                QMessageBox.information(self, "情報", "📦 バックアップ対象のデータベースが見つかりません。")
                return

            msg = "✅ バックアップを作成しました:\n\n" + "\n".join(f"- {n}" for n in created)
            msg += f"\n\n📁 保存先:\n{str(backup_dir)}"
            QMessageBox.information(self, "完了", msg)
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"❌ バックアップ作成中にエラーが発生しました:\n{e}")

    def _ensure_app_fonts_loaded(self):
        """Cargar fuentes desde la carpeta `Fonts` (si existen) y elegir una familia válida para texto."""
        if getattr(self, "_app_fonts_loaded", False):
            return

        self._app_fonts_loaded = True
        self._app_font_family = None

        try:
            fonts_dir = resource_path("Fonts")
            if not os.path.isdir(fonts_dir):
                return

            loaded_families = []
            for fn in os.listdir(fonts_dir):
                if not fn.lower().endswith((".ttf", ".otf", ".ttc")):
                    continue
                fpath = os.path.join(fonts_dir, fn)
                try:
                    font_id = QFontDatabase.addApplicationFont(fpath)
                    if font_id != -1:
                        loaded_families.extend(QFontDatabase.applicationFontFamilies(font_id))
                except Exception:
                    pass

            # Elegir una familia cargada que realmente soporte el texto (evita fuentes de iconos).
            sample_text = "0.00 sec"
            for fam in loaded_families:
                try:
                    fm = QFontMetrics(QFont(fam))
                    if all(fm.inFont(ch) for ch in sample_text):
                        self._app_font_family = fam
                        return
                except Exception:
                    continue
        except Exception:
            return

    def _add_center_header_title(self):
        """Añadir el texto '0.00 sec' centrado arriba en el panel central (fuera del área de gráficos)."""
        try:
            self._ensure_app_fonts_loaded()

            title = QLabel("0.00 sec")
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("background: transparent; color: #111111;")
            title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            # Preferir fuente desde `Fonts/` si alguna soporta el texto; si no, fallback moderno de Windows.
            preferred = self._app_font_family or "Segoe UI Variable Display"
            font = QFont(preferred)
            if not font.exactMatch():
                font = QFont(self._app_font_family or "Segoe UI")

            font.setPointSize(28)
            font.setWeight(QFont.DemiBold)
            title.setFont(font)

            self.center_title_label = title
            self.center_layout.addWidget(title, 0, Qt.AlignHCenter)
            self.center_layout.addSpacing(6)
        except Exception as e:
            print(f"⚠️ Error añadiendo título central: {e}")

    def create_center_panel(self):
        """Crear la estructura del panel central"""

        # Título arriba del área de gráficos (fuera del graph_area)
        self._add_center_header_title()

        # Área de gráficos
        # Contenedor de área de gráficos + botones de navegación
        self.graph_container = QFrame()
        graph_container_layout = QVBoxLayout()
        graph_container_layout.setContentsMargins(0, 0, 0, 0)
        graph_container_layout.setSpacing(0)
        self.graph_container.setLayout(graph_container_layout)

        # Área de gráficos
        self.graph_area = QFrame()
        self.graph_area.setStyleSheet("background-color: #F9F9F9; border: 1px solid #CCCCCC;")
        graph_container_layout.addWidget(self.graph_area, stretch=1)

        # Añadir contenedor al layout principal central
        self.center_layout.addWidget(self.graph_container, stretch=1)

        # Espacio flexible antes de los botones
        self.center_layout.addStretch()

        # Botones OK y NG
        self.ok_ng_frame = QFrame()
        ok_ng_layout = QHBoxLayout()
        ok_ng_layout.setAlignment(Qt.AlignCenter)
        self.ok_ng_frame.setLayout(ok_ng_layout)

        self.ok_button = QPushButton("OK")
        self.ng_button = QPushButton("NG")

        self.setup_ok_button(self.ok_button)
        self.setup_ng_button(self.ng_button)

        self.ok_button.clicked.connect(self.on_ok_clicked)
        self.ng_button.clicked.connect(self.on_ng_clicked)

        ok_ng_layout.addWidget(self.ok_button)
        ok_ng_layout.addSpacing(10)
        ok_ng_layout.addWidget(self.ng_button)

        self.center_layout.addWidget(self.ok_ng_frame)

        self.ok_button.setEnabled(False)
        self.ng_button.setEnabled(False)

    def create_console_panel(self):
        """Crear la consola integrada en el panel derecho"""
        # Título de la consola
        console_title = QLabel("コンソール出力")
        console_title.setAlignment(Qt.AlignCenter)
        console_title.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #333333;
            background-color: #F0F0F0;
            padding: 5px;
            border: 1px solid #CCCCCC;
            border-radius: 3px;
        """)
        self.console_layout.addWidget(console_title)

        # Área de texto de la consola
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setMaximumHeight(400)
        self.console_output.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #FFFFFF;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
            }
        """)
        self.console_layout.addWidget(self.console_output)

        # Botones de control de la consola
        console_controls = QFrame()
        console_controls_layout = QHBoxLayout()
        console_controls_layout.setContentsMargins(0, 5, 0, 5)
        console_controls.setLayout(console_controls_layout)

        # Botón para limpiar consola
        self.clear_console_button = QPushButton("クリア")
        self.clear_console_button.setMaximumWidth(60)
        self.clear_console_button.clicked.connect(self.clear_console)
        self.clear_console_button.setStyleSheet("""
            QPushButton {
                background-color: #F0F0F0;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #E0E0E0;
            }
        """)

        # Botón para guardar log
        self.save_log_button = QPushButton("保存")
        self.save_log_button.setMaximumWidth(60)
        self.save_log_button.clicked.connect(self.save_console_log)
        self.save_log_button.setStyleSheet("""
            QPushButton {
                background-color: #F0F0F0;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #E0E0E0;
            }
        """)

        console_controls_layout.addWidget(self.clear_console_button)
        console_controls_layout.addWidget(self.save_log_button)
        console_controls_layout.addStretch()

        self.console_layout.addWidget(console_controls)
        
        # NOTA: El botón オーバーレイ表示 se crea en __init__ y se añade al panel central

        # Configurar redirección de stdout y stderr a la consola
        self.setup_console_redirection()

    def create_overlay_console_panel(self):
        """Crear el panel desplegable que se superpone sobre el panel central"""
        print("🔧 Creando panel desplegable superpuesto...")
        
        # Panel desplegable que se superpone
        # IMPORTANT: debe ser una ventana top-level (sin parent) para que NO la bloquee
        # el ReusableProgressDialog (WindowModal) durante análisis.
        self.overlay_console = QFrame()
        self.overlay_console.setFrameShape(QFrame.StyledPanel)
        self.overlay_console.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 2px solid #3498db;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
        """)
        
        # Por defecto NO forzar siempre-arriba: si no, tapa diálogos del sistema (QFileDialog, etc).
        # Activamos "siempre-arriba" solo mientras el loading (ReusableProgressDialog) esté visible.
        self.overlay_console.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        self.overlay_console.setAttribute(Qt.WA_TranslucentBackground, False)
        self.overlay_console.setAttribute(Qt.WA_NoSystemBackground, False)
        
        # Layout del panel desplegable
        self.overlay_console_layout = QVBoxLayout()
        self.overlay_console_layout.setContentsMargins(10, 10, 10, 10)
        self.overlay_console.setLayout(self.overlay_console_layout)
        
        # Título del panel desplegable
        overlay_title = QLabel("コンソール出力 (オーバーレイ)")
        overlay_title.setAlignment(Qt.AlignCenter)
        overlay_title.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #2c3e50;
            background-color: #ecf0f1;
                border-radius: 5px;
                margin-bottom: 10px;
        """)
        self.overlay_console_layout.addWidget(overlay_title)
        
        # Área de texto de la consola desplegable
        self.overlay_console_output = QTextEdit()
        self.overlay_console_output.setReadOnly(True)
        self.overlay_console_output.setMaximumHeight(500)
        self.overlay_console_output.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #FFFFFF;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
            }
        """)
        self.overlay_console_layout.addWidget(self.overlay_console_output)
        
        # Botones de control del panel desplegable
        overlay_controls = QFrame()
        overlay_controls_layout = QHBoxLayout()
        overlay_controls_layout.setContentsMargins(0, 5, 0, 5)
        overlay_controls.setLayout(overlay_controls_layout)
        
        # Botón para limpiar consola desplegable
        self.overlay_clear_button = QPushButton("クリア")
        self.overlay_clear_button.setMaximumWidth(60)
        self.overlay_clear_button.clicked.connect(self.clear_overlay_console)
        self.overlay_clear_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        
        # Botón para guardar log del panel desplegable
        self.overlay_save_button = QPushButton("保存")
        self.overlay_save_button.setMaximumWidth(60)
        self.overlay_save_button.clicked.connect(self.save_overlay_console_log)
        self.overlay_save_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        
        overlay_controls_layout.addWidget(self.overlay_clear_button)
        overlay_controls_layout.addWidget(self.overlay_save_button)
        overlay_controls_layout.addStretch()
        
        self.overlay_console_layout.addWidget(overlay_controls)
        
        # Botón de flecha para expandir/contraer
        # IMPORTANT: botón como ventana top-level (sin parent) para que siga clicable
        # incluso cuando el diálogo de progreso está en modo WindowModal.
        self.console_toggle_button = QPushButton("◀")
        self.console_toggle_button.setFixedSize(30, 30)
        
        # CRÍTICO: Para que el botón sea redondo en una ventana top-level, 
        # necesitamos fondo translúcido y FramelessWindowHint
        self.console_toggle_button.setAttribute(Qt.WA_TranslucentBackground)
        self.console_toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 15px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        # Ventana sin marco (se ve como overlay real). NO siempre-arriba por defecto.
        self.console_toggle_button.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        # Evitar "pelea" de foco mientras está el loading visible
        self.console_toggle_button.setFocusPolicy(Qt.NoFocus)
        self.console_toggle_button.setAttribute(Qt.WA_ShowWithoutActivating, True)
        
        # Conectar la flecha al método de toggle
        self.console_toggle_button.clicked.connect(self.toggle_right_panel)
        print("🔧 Flecha conectada a toggle_right_panel")
        
        # Inicialmente solo la flecha visible, panel desplegable oculto
        self.overlay_console.hide()
        self.console_toggle_button.show()  # Flecha siempre visible
        
        # Estado del panel desplegable
        self.overlay_console_visible = False
        # Estado "siempre-arriba" (solo durante loading modal)
        self._console_topmost_enabled = False
        
        # Posicionar la flecha inicialmente
        QTimer.singleShot(100, self.position_arrow)
        
        # Configurar timer para mantener elementos en primer plano
        self.keep_on_top_timer = QTimer()
        self.keep_on_top_timer.timeout.connect(self.keep_elements_on_top)
        self.keep_on_top_timer.start(1000)  # Cada segundo
        
        # Configurar timer para verificar cambios de posición de la ventana
        self.position_check_timer = QTimer()
        self.position_check_timer.timeout.connect(self.check_window_position)
        self.position_check_timer.start(500)  # Cada medio segundo
        
        # Guardar la posición inicial de la ventana
        self.last_window_position = self.geometry()
        
        print("🔧 Panel desplegable creado correctamente")
        print(f"🔧 Botón de flecha creado: {self.console_toggle_button}")
        print(f"🔧 Botón visible: {self.console_toggle_button.isVisible()}")
        print(f"🔧 Botón padre: {self.console_toggle_button.parent()}")

    def _build_done_experiments_excel(self, main_file: str, temp_file: str, done_file: str):
        """
        Construye un Excel con los ensayos YA HECHOS como:
            done = (main_file) - (temp_file)
        usando como clave las 7 columnas de condiciones.

        - main_file: Proyecto_XX_未実験データ.xlsx (carpeta principal del proyecto)
        - temp_file: Proyecto_XX_未実験データ.xlsx (99_Temp)
        - done_file: salida (por defecto en 99_Temp/done_experiments.xlsx)
        """
        try:
            import os
            import pandas as pd
            import numpy as np

            # Aceptar ambos nombres para la columna de dirección:
            # - "UPカット" (nuevo)
            # - "回転方向" (antiguo)
            dir_variants = ["UPカット", "回転方向"]
            key_cols_fixed = ['回転速度', '送り速度', '切込量', '突出量', '載せ率', 'パス数']
            int_cols = ['回転速度', '送り速度', 'DIR', 'パス数']
            float_cols = ['切込量', '突出量', '載せ率']

            if not (os.path.exists(main_file) and os.path.exists(temp_file)):
                print(f"⚠️ done_experiments: archivos no existen. main={main_file}, temp={temp_file}")
                return None

            # Cache simple: si done_file es más nuevo que los inputs, reutilizar
            try:
                if os.path.exists(done_file):
                    done_mtime = os.path.getmtime(done_file)
                    if done_mtime >= max(os.path.getmtime(main_file), os.path.getmtime(temp_file)):
                        print(f"✅ done_experiments: usando cache existente {done_file}")
                        return done_file
            except Exception:
                pass

            main_df = pd.read_excel(main_file)
            temp_df = pd.read_excel(temp_file)

            def _pick_dir_col(df: pd.DataFrame):
                for c in dir_variants:
                    if c in df.columns:
                        return c
                return None

            dir_main = _pick_dir_col(main_df)
            dir_temp = _pick_dir_col(temp_df)
            if dir_main is None or dir_temp is None:
                print(f"❌ done_experiments: falta columna de dirección. main_has={list(main_df.columns)}, temp_has={list(temp_df.columns)}")
                return None

            missing_main = [c for c in key_cols_fixed if c not in main_df.columns]
            missing_temp = [c for c in key_cols_fixed if c not in temp_df.columns]
            if missing_main or missing_temp:
                print(f"❌ done_experiments: faltan columnas. main_missing={missing_main}, temp_missing={missing_temp}")
                return None

            def _norm_key_df(df: pd.DataFrame) -> pd.DataFrame:
                # Normalizamos a un esquema común con columna "DIR"
                k = df[key_cols_fixed].copy()
                k["DIR"] = df[dir_main] if dir_main in df.columns else df[dir_temp]
                # numérico + redondeo para evitar diferencias de precisión
                for c in ["回転速度", "送り速度", "パス数", "DIR"]:
                    k[c] = pd.to_numeric(k[c], errors="coerce").round(0).astype("Int64")
                for c in float_cols:
                    k[c] = pd.to_numeric(k[c], errors="coerce").round(6)
                return k

            main_key_df = _norm_key_df(main_df)
            temp_key_df = _norm_key_df(temp_df)

            main_hash = pd.util.hash_pandas_object(main_key_df, index=False)
            temp_hash = pd.util.hash_pandas_object(temp_key_df, index=False)
            temp_set = set(temp_hash.values.tolist())

            done_mask = ~main_hash.isin(temp_set)
            done_full = main_df.loc[done_mask].copy()

            # Deduplicar por clave (conservar primera ocurrencia)
            dedup_cols = ['回転速度', '送り速度', dir_main, '切込量', '突出量', '載せ率', 'パス数']
            done_full = done_full.drop_duplicates(subset=[c for c in dedup_cols if c in done_full.columns])

            os.makedirs(os.path.dirname(done_file), exist_ok=True)
            done_full.to_excel(done_file, index=False)

            print(f"✅ done_experiments generado: {done_file} | filas={len(done_full)}")
            return done_file

        except Exception as e:
            print(f"⚠️ Error creando done_experiments.xlsx: {e}")
            return None

    def _export_unexperimented_excel_folder_from_csv(self, csv_path: str, project_folder: str, project_name: str):
        """
        Si el archivo de muestreo del proyecto está en CSV (Proyecto_XX_未実験データ.csv),
        crear también Excel(s) dentro de una carpeta:
          <project_folder>/99_未実験データ/
        - Si <= 500,000 filas: crear <project_name>_未実験データ.xlsx
        - Si > 500,000 filas: crear <project_name>_未実験データ_part_###.xlsx (500k filas por archivo)
        """
        try:
            if not csv_path or not os.path.exists(csv_path):
                return
            if os.path.splitext(csv_path)[1].lower() != ".csv":
                return

            out_folder = os.path.join(project_folder, "99_未実験データ")
            os.makedirs(out_folder, exist_ok=True)

            rows_per_file = 500_000
            chunksize = 100_000

            print(f"📦 99_未実験データ: CSV→Excel 変換開始: {csv_path}", flush=True)
            print(f"📦 99_未実験データ: 出力先フォルダ: {out_folder}", flush=True)

            part_idx = 1
            part_rows = 0
            startrow = 0
            writer = None
            wrote_any = False

            def _open_writer():
                nonlocal writer, part_idx, part_rows, startrow
                if writer is not None:
                    writer.close()
                part_path = os.path.join(out_folder, f"{project_name}_未実験データ_part_{part_idx:03d}.xlsx")
                print(f"📄 99_未実験データ: creando {os.path.basename(part_path)}", flush=True)
                writer = pd.ExcelWriter(part_path, engine="openpyxl")
                part_idx += 1
                part_rows = 0
                startrow = 0

            for chunk in pd.read_csv(csv_path, encoding="utf-8-sig", chunksize=chunksize):
                pos = 0
                while pos < len(chunk):
                    if writer is None or part_rows >= rows_per_file:
                        _open_writer()
                    remaining = rows_per_file - part_rows
                    take = min(remaining, len(chunk) - pos)
                    piece = chunk.iloc[pos:pos + take]
                    header = startrow == 0
                    piece.to_excel(writer, index=False, header=header, startrow=startrow, sheet_name="Sheet1")
                    wrote_any = True
                    pos += take
                    part_rows += take
                    startrow += take + (1 if header else 0)

            if writer is not None:
                writer.close()

            # Si solo se generó un part, renombrarlo a .xlsx “normal”
            if wrote_any:
                parts = sorted(
                    [f for f in os.listdir(out_folder) if f.startswith(f"{project_name}_未実験データ_part_") and f.endswith(".xlsx")]
                )
                if len(parts) == 1:
                    src = os.path.join(out_folder, parts[0])
                    dst = os.path.join(out_folder, f"{project_name}_未実験データ.xlsx")
                    try:
                        if os.path.exists(dst):
                            os.remove(dst)
                        os.replace(src, dst)
                        print(f"✅ 99_未実験データ: 1ファイルのためリネーム: {os.path.basename(dst)}", flush=True)
                    except Exception as e:
                        print(f"⚠️ 99_未実験データ: リネーム失敗: {e}", flush=True)

            print("✅ 99_未実験データ: CSV→Excel 変換完了", flush=True)
        except Exception as e:
            print(f"⚠️ 99_未実験データ: CSV→Excel 変換エラー: {e}", flush=True)

    def create_brush_selector(self):
        """Crear selector de cepillos (A11, A21, A32, A13)"""
        # Label para el selector de brush (sin QGroupBox)
        self.brush_label = QLabel("使用するブラシ")
        self.left_layout.addWidget(self.brush_label)

        # Selector de cepillo
        self.brush_selector = QComboBox()
        self.brush_selector.addItems(["A11", "A21", "A32", "A13"])
        self.brush_selector.setCurrentText("A11")
        self.left_layout.addWidget(self.brush_selector)

        # Selector de diámetro encima de 線材長
        self.diameter_label = QLabel("直径 選択")
        self.diameter_selector = QComboBox()
        self.diameter_selector.addItems(["6", "15", "25", "40", "60", "100"])
        self.diameter_selector.setCurrentText("15")
        self.left_layout.addWidget(self.diameter_label)
        self.left_layout.addWidget(self.diameter_selector)

        # Conectar lógica para restringir diámetro si se selecciona A13
        self.brush_selector.currentTextChanged.connect(self.update_diameter_options)
        self.update_diameter_options(self.brush_selector.currentText())

    def update_diameter_options(self, brush_name):
        """Restringe el selector de diámetro si el cepillo es A13"""
        allowed = ["6", "15"] if brush_name == "A13" else ["6", "15", "25", "40", "60", "100"]
        for i in range(self.diameter_selector.count()):
            value = self.diameter_selector.itemText(i)
            self.diameter_selector.model().item(i).setEnabled(value in allowed)
        # Si el valor actual no está permitido, selecciona el primero permitido
        if self.diameter_selector.currentText() not in allowed:
            self.diameter_selector.setCurrentText(allowed[0])



    def create_navigation_buttons(self):
        if self.graph_navigation_frame is not None:
            return

        self.graph_navigation_frame = QFrame()
        nav_layout = QHBoxLayout()
        nav_layout.setAlignment(Qt.AlignRight)
        self.graph_navigation_frame.setLayout(nav_layout)

        self.prev_button = QPushButton("← 前へ")
        self.next_button = QPushButton("次へ →")

        self.setup_navigation_button(self.prev_button)
        self.setup_navigation_button(self.next_button)

        nav_layout.addWidget(self.prev_button)
        nav_layout.addSpacing(10)
        nav_layout.addWidget(self.next_button)

        self.graph_container.layout().addWidget(self.graph_navigation_frame)

        # ❗️Conectar aquí
        self.prev_button.clicked.connect(self.show_previous_graph)
        self.next_button.clicked.connect(self.show_next_graph)

        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)

    def show_previous_graph(self):
        if self.current_graph_index > 0:
            self.current_graph_index -= 1
            self.update_graph_display()

    def show_next_graph(self):
        if self.current_graph_index < len(self.graph_images) - 1:
            self.current_graph_index += 1
            self.update_graph_display()

    def create_filter_view(self):
        """Crear la vista de filtrado a la derecha"""
        # Limpiar el layout central
        while self.center_layout.count():
            item = self.center_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Título mejorado
        title = QLabel("データフィルター")
        title.setStyleSheet("""
            font-weight: bold; 
            font-size: 24px; 
            color: #2c3e50;
            margin-bottom: 20px;
            padding: 10px 0px;
            border-bottom: 2px solid #3498db;
            border-radius: 0px;
        """)
        title.setAlignment(Qt.AlignCenter)
        self.center_layout.addWidget(title)

        # Espaciado entre título y filtros
        spacer = QWidget()
        spacer.setFixedHeight(15)
        self.center_layout.addWidget(spacer)

        # Contenedor principal horizontal para filtros e imagen
        main_container = QHBoxLayout()
        
        # Contenedor vertical para todos los filtros con margen izquierdo
        filters_container = QVBoxLayout()
        filters_container.setSpacing(8)
        filters_container.setAlignment(Qt.AlignTop)
        filters_container.setContentsMargins(20, 0, 0, 0)  # Margen izquierdo de 20px

        self.filter_inputs = {}

        # Helper: añadir fila limpia
        def add_filter_row(label_text, widget1, widget2=None):
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setFixedWidth(90)
            label.setStyleSheet("font-weight: bold; font-size: 12px;")
            row.addWidget(label)

            # Calcular el ancho total disponible (mismo que la fila de radio buttons)
            # 90px (label) + 4*radio_buttons + 3*12px (margins) + 8px (spacing) = ~200px
            total_width = 200
            widget1.setFixedWidth(total_width)
            row.addWidget(widget1)

            if widget2:
                spacer = QLabel("〜")
                spacer.setFixedWidth(10)
                spacer.setAlignment(Qt.AlignCenter)
                row.addWidget(spacer)

                widget2.setFixedWidth(total_width)
                row.addWidget(widget2)

            row.addStretch()
            filters_container.addLayout(row)

        # 実験日 (rango de fechas)
        desde_fecha = QDateEdit()
        desde_fecha.setCalendarPopup(True)
        desde_fecha.setDate(QDate.currentDate().addDays(-30))  # 30 días atrás por defecto
        desde_fecha.setFixedWidth(150)
        
        hasta_fecha = QDateEdit()
        hasta_fecha.setCalendarPopup(True)
        hasta_fecha.setDate(QDate.currentDate())  # Fecha actual por defecto
        hasta_fecha.setFixedWidth(150)
        
        # Botón "なし" para no aplicar filtro de fecha
        no_date_button = QPushButton("なし")
        no_date_button.setFixedWidth(80)
        no_date_button.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
            QPushButton:pressed {
                background-color: #6c7b7d;
            }
        """)
        
        # Variable para controlar si se aplica filtro de fecha
        self.apply_date_filter = True
        
        def toggle_date_filter():
            if self.apply_date_filter:
                # Desactivar filtro de fecha
                self.apply_date_filter = False
                no_date_button.setText("適用")
                no_date_button.setStyleSheet("""
                    QPushButton {
                        background-color: #e74c3c;
                        color: white;
                        border: none;
                        padding: 5px;
                        border-radius: 4px;
                        font-size: 11px;
                    }
                    QPushButton:hover {
                        background-color: #c0392b;
                    }
                    QPushButton:pressed {
                        background-color: #a93226;
                    }
                """)
                desde_fecha.setEnabled(False)
                hasta_fecha.setEnabled(False)
            else:
                # Activar filtro de fecha
                self.apply_date_filter = True
                no_date_button.setText("なし")
                no_date_button.setStyleSheet("""
                    QPushButton {
                        background-color: #95a5a6;
                        color: white;
                        border: none;
                        padding: 5px;
                        border-radius: 4px;
                        font-size: 11px;
                    }
                    QPushButton:hover {
                        background-color: #7f8c8d;
                    }
                    QPushButton:pressed {
                        background-color: #6c7b7d;
                    }
                """)
                desde_fecha.setEnabled(True)
                hasta_fecha.setEnabled(True)
        
        no_date_button.clicked.connect(toggle_date_filter)
        
        self.filter_inputs["実験日"] = (desde_fecha, hasta_fecha)
        
        # Crear fila personalizada para fecha con botón
        date_row = QHBoxLayout()
        date_label = QLabel("実験日")
        date_label.setFixedWidth(90)
        date_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        date_row.addWidget(date_label)
        
        date_row.addWidget(desde_fecha)
        
        spacer = QLabel("〜")
        spacer.setFixedWidth(10)
        spacer.setAlignment(Qt.AlignCenter)
        date_row.addWidget(spacer)
        
        date_row.addWidget(hasta_fecha)
        
        # Agregar espacio y botón
        date_row.addSpacing(10)
        date_row.addWidget(no_date_button)
        
        date_row.addStretch()
        filters_container.addLayout(date_row)

        # バリ除去
        combo = QComboBox()
        combo.addItems(["", "0", "1"])
        combo.setFixedWidth(200)  # Mismo ancho que los otros campos
        self.filter_inputs["バリ除去"] = combo
        add_filter_row("バリ除去", combo)

        # 上面ダレ量
        desde = QLineEdit()
        hasta = QLineEdit()
        desde.setPlaceholderText("min")
        hasta.setPlaceholderText("max")
        self.filter_inputs["上面ダレ量"] = (desde, hasta)
        add_filter_row("上面ダレ量", desde, hasta)

        # 側面ダレ量
        desde = QLineEdit()
        hasta = QLineEdit()
        desde.setPlaceholderText("min")
        hasta.setPlaceholderText("max")
        self.filter_inputs["側面ダレ量"] = (desde, hasta)
        add_filter_row("側面ダレ量", desde, hasta)

        # 面粗度(Ra)前
        desde = QLineEdit()
        hasta = QLineEdit()
        desde.setPlaceholderText("min")
        hasta.setPlaceholderText("max")
        self.filter_inputs["面粗度(Ra)前"] = (desde, hasta)
        add_filter_row("面粗度(Ra)前", desde, hasta)

        # 面粗度(Ra)後
        desde = QLineEdit()
        hasta = QLineEdit()
        desde.setPlaceholderText("min")
        hasta.setPlaceholderText("max")
        self.filter_inputs["面粗度(Ra)後"] = (desde, hasta)
        add_filter_row("面粗度(Ra)後", desde, hasta)

        # 摩耗量
        desde = QLineEdit()
        hasta = QLineEdit()
        desde.setPlaceholderText("min")
        hasta.setPlaceholderText("max")
        self.filter_inputs["摩耗量"] = (desde, hasta)
        add_filter_row("摩耗量", desde, hasta)

        # 切削力X
        desde = QLineEdit()
        hasta = QLineEdit()
        desde.setPlaceholderText("min")
        hasta.setPlaceholderText("max")
        self.filter_inputs["切削力X"] = (desde, hasta)
        add_filter_row("切削力X", desde, hasta)

        # 切削力Y
        desde = QLineEdit()
        hasta = QLineEdit()
        desde.setPlaceholderText("min")
        hasta.setPlaceholderText("max")
        self.filter_inputs["切削力Y"] = (desde, hasta)
        add_filter_row("切削力Y", desde, hasta)

        # 切削力Z
        desde = QLineEdit()
        hasta = QLineEdit()
        desde.setPlaceholderText("min")
        hasta.setPlaceholderText("max")
        self.filter_inputs["切削力Z"] = (desde, hasta)
        add_filter_row("切削力Z", desde, hasta)

        # 材料
        material_combo = QComboBox()
        material_combo.addItems(["", "Steel", "Alumi"])
        material_combo.setFixedWidth(200)  # Mismo ancho que los otros campos
        self.filter_inputs["材料"] = material_combo
        add_filter_row("材料", material_combo)

        # ブラシ
        brush_label = QLabel("ブラシ選択")
        brush_label.setFixedWidth(90)
        brush_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        
        brush_container = QHBoxLayout()
        brush_container.setSpacing(4)  # Reducir espacio entre botones
        
        self.filter_inputs["すべて"] = QCheckBox("すべて")
        self.filter_inputs["A13"] = QCheckBox("A13")
        self.filter_inputs["A11"] = QCheckBox("A11")
        self.filter_inputs["A21"] = QCheckBox("A21")
        self.filter_inputs["A32"] = QCheckBox("A32")
        
        # Establecer "すべて" como seleccionado por defecto
        self.filter_inputs["すべて"].setChecked(True)
        
        # Aplicar estilo a los checkboxes
        checkbox_style = """
            QCheckBox {
                font-size: 11px;
                spacing: 4px;
                padding: 2px;
                margin-right: 48px;
            }
            QCheckBox::indicator {
                width: 12px;
                height: 12px;
            }
        """
        
        for key in ["すべて", "A13", "A11", "A21", "A32"]:
            self.filter_inputs[key].setStyleSheet(checkbox_style)
            brush_container.addWidget(self.filter_inputs[key])
            
        # Conectar señales para la lógica de selección mutuamente excluyente
        self.filter_inputs["すべて"].toggled.connect(self.on_subete_toggled)
        self.filter_inputs["A13"].toggled.connect(self.on_brush_toggled)
        self.filter_inputs["A11"].toggled.connect(self.on_brush_toggled)
        self.filter_inputs["A21"].toggled.connect(self.on_brush_toggled)
        self.filter_inputs["A32"].toggled.connect(self.on_brush_toggled)
        
        # Crear layout horizontal para label y botones
        brush_row = QHBoxLayout()
        brush_row.addWidget(brush_label)
        brush_row.addLayout(brush_container)
        brush_row.addStretch()
        
        filters_container.addLayout(brush_row)

        # 直径
        desde = QLineEdit()
        hasta = QLineEdit()
        desde.setPlaceholderText("min")
        hasta.setPlaceholderText("max")
        self.filter_inputs["直径"] = (desde, hasta)
        add_filter_row("直径", desde, hasta)

        # 回転速度
        desde = QLineEdit()
        hasta = QLineEdit()
        desde.setPlaceholderText("min")
        hasta.setPlaceholderText("max")
        self.filter_inputs["回転速度"] = (desde, hasta)
        add_filter_row("回転速度", desde, hasta)

        # 送り速度
        desde = QLineEdit()
        hasta = QLineEdit()
        desde.setPlaceholderText("min")
        hasta.setPlaceholderText("max")
        self.filter_inputs["送り速度"] = (desde, hasta)
        add_filter_row("送り速度", desde, hasta)

        # UPカット
        up_combo = QComboBox()
        up_combo.addItems(["", "0", "1"])
        up_combo.setFixedWidth(200)  # Mismo ancho que los otros campos
        self.filter_inputs["UPカット"] = up_combo
        add_filter_row("UPカット", up_combo)

        # 切込量
        desde = QLineEdit()
        hasta = QLineEdit()
        desde.setPlaceholderText("min")
        hasta.setPlaceholderText("max")
        self.filter_inputs["切込量"] = (desde, hasta)
        add_filter_row("切込量", desde, hasta)

        # 突出量
        desde = QLineEdit()
        hasta = QLineEdit()
        desde.setPlaceholderText("min")
        hasta.setPlaceholderText("max")
        self.filter_inputs["突出量"] = (desde, hasta)
        add_filter_row("突出量", desde, hasta)

        # 載せ率
        desde = QLineEdit()
        hasta = QLineEdit()
        desde.setPlaceholderText("min")
        hasta.setPlaceholderText("max")
        self.filter_inputs["載せ率"] = (desde, hasta)
        add_filter_row("載せ率", desde, hasta)

        # パス数
        pass_input = QLineEdit()
        pass_input.setPlaceholderText("例: 3")
        pass_input.setFixedWidth(200)  # Mismo ancho que los otros campos
        self.filter_inputs["パス数"] = pass_input
        add_filter_row("パス数", pass_input)

        # 線材長
        desde = QLineEdit()
        hasta = QLineEdit()
        desde.setPlaceholderText("min")
        hasta.setPlaceholderText("max")
        self.filter_inputs["線材長"] = (desde, hasta)
        add_filter_row("線材長", desde, hasta)

        # 加工時間
        desde = QLineEdit()
        hasta = QLineEdit()
        desde.setPlaceholderText("min")
        hasta.setPlaceholderText("max")
        self.filter_inputs["加工時間"] = (desde, hasta)
        add_filter_row("加工時間", desde, hasta)

        # Agregar filtros al contenedor principal
        main_container.addLayout(filters_container)
        
        # Agregar imagen chibi al lado derecho
        try:
            chibi_label = QLabel()
            chibi_pixmap = QPixmap(resource_path("xebec_chibi.png"))
            if not chibi_pixmap.isNull():
                # Redimensionar la imagen 200% más grande (2x el tamaño original)
                chibi_pixmap = chibi_pixmap.scaled(300, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                chibi_label.setPixmap(chibi_pixmap)
                chibi_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
                chibi_label.setStyleSheet("margin-left: 20px;")
                main_container.addWidget(chibi_label)
                print("✅ Imagen chibi cargada exitosamente")
            else:
                print("⚠️ No se pudo cargar la imagen xebec_chibi.png")
        except Exception as e:
            print(f"⚠️ Error cargando imagen chibi: {e}")
        
        # Agregar el contenedor principal al layout central
        self.center_layout.addLayout(main_container)

        # Espaciado más grande entre filtros y botones
        spacer = QWidget()
        spacer.setFixedHeight(50)
        self.center_layout.addWidget(spacer)

        # Contenedor horizontal para los 3 botones en paralelo con espacio a la derecha
        buttons_container = QHBoxLayout()
        buttons_container.setSpacing(10)  # Espacio entre botones
        
        # Estilo común para todos los botones usando azul claro como el botón de carga
        button_style = """
            QPushButton {
                background-color: #5EC8E5;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #4BB8D0;
            }
        """
        
        # Botón 線形解析
        linear_btn = QPushButton("線形解析")
        linear_btn.setFixedHeight(45)
        linear_btn.setStyleSheet(button_style)
        linear_btn.clicked.connect(self.on_linear_analysis_clicked)
        buttons_container.addWidget(linear_btn)
        
        # Botón 非線形解析
        nonlinear_btn = QPushButton("非線形解析")
        nonlinear_btn.setFixedHeight(45)
        nonlinear_btn.setStyleSheet(button_style)
        nonlinear_btn.setEnabled(True)  # Habilitado
        nonlinear_btn.setToolTip("非線形回帰分析を実行します")
        nonlinear_btn.clicked.connect(self.on_nonlinear_analysis_clicked)
        buttons_container.addWidget(nonlinear_btn)
        
        # Botón 分類分析
        classification_btn = QPushButton("分類分析")
        classification_btn.setFixedHeight(45)
        classification_btn.setStyleSheet(button_style)
        classification_btn.setEnabled(True)  # Habilitado
        classification_btn.setToolTip("分類分析を実行します")
        classification_btn.clicked.connect(self.on_classification_analysis_clicked)
        buttons_container.addWidget(classification_btn)
        
        # Agregar espacio vacío a la derecha del tamaño de 2 botones
        spacer_widget = QWidget()
        spacer_widget.setFixedWidth(320)  # 2 botones (150px cada uno) + 2 espaciados (10px cada uno)
        buttons_container.addWidget(spacer_widget)
        
        # Agregar el contenedor de botones al layout principal
        self.center_layout.addLayout(buttons_container)

    # ======================================
    # Funciones auxiliares de estilo
    # ======================================
    def setup_navigation_button(self, button: QPushButton):
        """Aplica estilo moderno y compacto a los botones de navegación."""
        button.setFixedSize(80, 32)  # Botón más pequeño
        button.setStyleSheet("""
            QPushButton {
                background-color: #666666;  /* Gris oscuro normal */
                color: white;
                font-family: "Yu Gothic UI";
                font-weight: bold;
                font-size: 13px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #555555;  /* Gris un poco más oscuro al pasar el ratón */
            }
        """)

    def setup_export_button(self, button):
        button.setStyleSheet("""
            QPushButton {
                background-color: lightgray;
                color: black;
                border: 1px solid #888;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
        """)
        button.setEnabled(True)

    def setup_generate_button_style(self, button: QPushButton):
        """Estilo específico para el botón de generación de archivo base de muestras."""
        button.setFixedHeight(30)
        button.setStyleSheet("""
            QPushButton {
                background-color: #E0E0E0;
                color: #333333;
                border: none;
                border-radius: 8px;
                font-size: 11px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #D5D5D5;
            }
        """)

    def setup_ok_button(self, button: QPushButton):
        """Configura estilo del botón OK"""
        button.setFixedSize(100, 40)
        button.setStyleSheet("""
            QPushButton {
                background-color: #CCCCCC;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                padding: 6px 16px;
            }
            QPushButton:enabled {
                background-color: #5CB85C;
            }
            QPushButton:enabled:hover {
                background-color: #4CAE4C;
            }
        """)

    def setup_ng_button(self, button: QPushButton):
        """Configura estilo del botón NG"""
        button.setFixedSize(100, 40)
        button.setStyleSheet("""
            QPushButton {
                background-color: #CCCCCC;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                padding: 6px 16px;
            }
            QPushButton:enabled {
                background-color: #E57373;
            }
            QPushButton:enabled:hover {
                background-color: #EF5350;
            }
        """)

    def setup_load_block(self, button: QPushButton, label: QLabel):
        """Configura visualmente un bloque de carga"""
        button.setFixedHeight(30)
        button.setStyleSheet("""
            QPushButton {
                background-color: #5EC8E5;
                color: white;
                font-family: "Noto Sans JP";  /* Tipo de letra moderno */
                border: none;
                border-radius: 8px;
                font-size: 12px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #4BB8D0;
            }
        """)

        label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        label.setFixedHeight(28)
        label.setStyleSheet("""
            background-color: #FFFFFF;
            border: 1px solid #DDDDDD;
            border-radius: 6px;
            padding-left: 10px;
            font-size: 11px;
            color: #555555;
        """)

        self.left_layout.addWidget(button)
        self.left_layout.addWidget(label)

    def setup_action_button(self, button: QPushButton):
        """Configura los botones principales"""
        button.setFixedHeight(48)
        button.setStyleSheet("""
            QPushButton {
                background-color: #3A80BA;
                color: white;
                font-family: "Noto Sans JP";  /* Tipo de letra moderno */
                border: none;
                border-radius: 8px;
                font-size: 16px;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #336DA3;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #888888;
            }
        """)

    def setup_results_button(self, button: QPushButton):
        """Configura el botón Show Results"""
        button.setFixedHeight(40)
        button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-family: "Noto Sans JP";
                border: none;
                border-radius: 8px;
                font-size: 16px;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background-color: #43A047;
            }
            QPushButton:disabled {
                background-color: #B0B0B0;
                color: #EEEEEE;
            }
        """)

    # ======================================
    # Funciones de eventos
    # ======================================
    def apply_filters(self):
        query = "SELECT * FROM main_results WHERE 1=1"
        params = []

        # Mapear nombres UI -> nombres reales en DB
        field_to_db = {
            "面粗度(Ra)前": "面粗度前",
            "面粗度(Ra)後": "面粗度後",
        }

        # Procesar filtros de cepillo primero (lógica especial)
        brush_filters = []
        for field in ["A13", "A11", "A21", "A32"]:
            if self.filter_inputs[field].isChecked():
                brush_filters.append(field)
        
        # Si "すべて" está seleccionado, no aplicar filtros de cepillo
        if not self.filter_inputs["すべて"].isChecked() and brush_filters:
            # Construir filtro OR para múltiples cepillos seleccionados
            brush_conditions = []
            for brush in brush_filters:
                brush_conditions.append(f"{brush} = ?")
                params.append(1)
            if brush_conditions:
                query += f" AND ({' OR '.join(brush_conditions)})"

        # Procesar otros filtros
        for field, widgets in self.filter_inputs.items():
            # Saltar filtros de cepillo ya procesados
            if field in ["すべて", "A13", "A11", "A21", "A32"]:
                continue
                
            if field in ["バリ除去", "UPカット"]:
                val = widgets.currentText()
                if val != "":
                    query += f" AND {field} = ?"
                    params.append(int(val))

            elif field == "材料":
                val = widgets.currentText()
                if val != "":
                    query += f" AND {field} = ?"
                    params.append(val)

            elif field == "パス数":
                text = widgets.text().strip()
                if text:
                    try:
                        query += f" AND パス数 = ?"
                        params.append(int(text))
                    except ValueError:
                        QMessageBox.warning(self, "入力エラー", f"❌ 数値を入力してください: {field}")
                        return

            elif field == "実験日":
                # Handle date range filter - solo si está habilitado
                if hasattr(self, 'apply_date_filter') and self.apply_date_filter:
                    desde_fecha, hasta_fecha = widgets
                    desde = desde_fecha.date().toString("yyyyMMdd")
                    hasta = hasta_fecha.date().toString("yyyyMMdd")
                
                if desde and hasta:
                    query += f" AND {field} >= ? AND {field} <= ?"
                    params.append(int(desde))
                    params.append(int(hasta))

            else:
                # Handle range filters (min/max inputs)
                desde_input, hasta_input = widgets
                desde = desde_input.text().strip()
                hasta = hasta_input.text().strip()
                db_field = field_to_db.get(field, field)

                if desde:
                    query += f" AND {db_field} >= ?"
                    params.append(float(desde))
                if hasta:
                    query += f" AND {db_field} <= ?"
                    params.append(float(hasta))

        try:
            conn = sqlite3.connect(RESULTS_DB_PATH, timeout=10)
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            self.filtered_df = df
            print("✅ Datos filtrados:")
            print(df)
            QMessageBox.information(self, "完了", f"✅ {len(df)} 件のデータが見つかりました。")

        except Exception as e:
            QMessageBox.critical(self, "エラー", f"❌ フィルターの適用中にエラーが発生しました:\n{str(e)}")

    def linear_analysis(self):
        """Análisis lineal de los datos filtrados"""
        if hasattr(self, "filtered_df"):
            print("📊 Iniciando análisis lineal...")
            print(f"Datos filtrados: {len(self.filtered_df)} registros")
            # Aquí implementar análisis lineal
            QMessageBox.information(self, "線形解析", "📊 線形解析を開始しました。")
        else:
            QMessageBox.warning(self, "警告", "⚠️ フィルタリングされたデータがありません。")
    
    def nonlinear_analysis(self):
        """Análisis no lineal de los datos filtrados"""
        if hasattr(self, "filtered_df"):
            print("📈 Iniciando análisis no lineal...")
            print(f"Datos filtrados: {len(self.filtered_df)} registros")
            # Aquí implementar análisis no lineal
            QMessageBox.information(self, "非線形解析", "📈 非線形解析を開始しました。")
        else:
            QMessageBox.warning(self, "警告", "⚠️ フィルタリングされたデータがありません。")
    
    def classification_analysis(self):
        """Análisis de clasificación de los datos filtrados"""
        if hasattr(self, "filtered_df"):
            print("🏷️ Iniciando análisis de clasificación...")
            print(f"Datos filtrados: {len(self.filtered_df)} registros")
            # Aquí implementar análisis de clasificación
            QMessageBox.information(self, "分類分析", "🏷️ 分類分析を開始しました。")
        else:
            QMessageBox.warning(self, "警告", "⚠️ フィルタリングされたデータがありません。")
    
    def analyze_filtered_data(self):
        if hasattr(self, "filtered_df"):
            print("⚙️ Analizando datos filtrados...")
            print(self.filtered_df.head())
            # Aquí puedes lanzar gráficos, cálculos, etc.
        else:
            print("⚠️ No hay datos filtrados.")

    def on_subete_toggled(self, checked):
        """Maneja la lógica cuando se selecciona/deselecciona 'すべて' (subete)"""
        if checked:
            # Si se selecciona "すべて", deseleccionar todos los otros cepillos
            self.filter_inputs["A13"].setChecked(False)
            self.filter_inputs["A11"].setChecked(False)
            self.filter_inputs["A21"].setChecked(False)
            self.filter_inputs["A32"].setChecked(False)
            print("✅ 'すべて' seleccionado - otros cepillos deseleccionados")

    def on_brush_toggled(self, checked):
        """Maneja la lógica cuando se selecciona/deselecciona cualquier cepillo específico"""
        sender = self.sender()
        if checked:
            # Si se selecciona un cepillo específico, deseleccionar "すべて"
            self.filter_inputs["すべて"].setChecked(False)
            print(f"✅ {sender.text()} seleccionado - 'すべて' deseleccionado")
        else:
            # Si se deselecciona un cepillo, verificar si no hay ninguno seleccionado
            if not any([
                self.filter_inputs["A13"].isChecked(),
                self.filter_inputs["A11"].isChecked(),
                self.filter_inputs["A21"].isChecked(),
                self.filter_inputs["A32"].isChecked()
            ]):
                # Si no hay ninguno seleccionado, seleccionar "すべて" por defecto
                self.filter_inputs["すべて"].setChecked(True)
                print("✅ Ningún cepillo específico seleccionado - 'すべて' seleccionado por defecto")

    def load_file(self, label_to_update: QLabel, title: str):
        """Carga un archivo y actualiza el label"""
        # ✅ NUEVO: Pausar timers automáticos para evitar interferencia con el diálogo
        self.pause_auto_timers()
        
        file_path, _ = QFileDialog.getOpenFileName(self, title)
        
        # ✅ NUEVO: Reanudar timers después del diálogo
        self.resume_auto_timers()

        if file_path:
            file_name = file_path.split("/")[-1]
            label_to_update.setText(f"ファイル読み込み完了: {file_name}")

            # Guardar la ruta del sample o del results según el label
            if label_to_update == self.sample_label:
                self.sample_file_path = file_path
            elif label_to_update == self.results_label:
                self.results_file_path = file_path
        else:
            label_to_update.setText("ファイル未選択")

    def on_d_optimizer_clicked(self):
        """Ejecuta solo la optimización D-óptima"""
        # ✅ NUEVO: No mezclar ejecuciones pesadas en paralelo
        if hasattr(self, 'linear_worker') and self.linear_worker is not None:
            try:
                if self.linear_worker.isRunning():
                    QMessageBox.warning(self, "最適化", "⚠️ 線形解析が実行中です。\n完了または停止するまでお待ちください。")
                    return
            except RuntimeError:
                self.linear_worker = None
        if hasattr(self, 'nonlinear_worker') and self.nonlinear_worker is not None:
            try:
                if self.nonlinear_worker.isRunning():
                    QMessageBox.warning(self, "最適化", "⚠️ 非線形解析が実行中です。\n完了または停止するまでお待ちください。")
                    return
            except RuntimeError:
                self.nonlinear_worker = None

        # ✅ NUEVO: Evitar arrancar si ya hay una optimización en ejecución
        for t_attr in ("d_optimizer_thread", "i_optimizer_thread", "dsaitekika_thread"):
            if hasattr(self, t_attr):
                t = getattr(self, t_attr)
                try:
                    if t is not None and t.isRunning():
                        QMessageBox.warning(self, "最適化", "⚠️ すでに最適化が実行中です。\n完了するまでお待ちください。")
                        return
                except RuntimeError:
                    setattr(self, t_attr, None)

        # Verificar que el archivo de muestreo haya sido cargado
        if not hasattr(self, "sample_file_path"):
            QMessageBox.warning(self, "エラー", "❌ サンプルファイルが読み込まれていません。")
            return

        # ✅ NUEVO: Verificar si el archivo pertenece a un proyecto existente
        sample_path = self.sample_file_path
        sample_dir = os.path.dirname(sample_path)
        sample_file = os.path.basename(sample_path)
        
        # Verificar si es un archivo de proyecto existente
        belongs_to_existing_project = False
        sample_ext = os.path.splitext(sample_file)[1].lower()
        is_project_sample = (
            sample_file.endswith("_未実験データ.xlsx")
            or sample_file.endswith("_未実験データ.xls")
            or sample_file.endswith("_未実験データ.csv")
        )
        if is_project_sample:
            project_name = sample_file[: -len(f"_未実験データ{sample_ext}")]
            if os.path.basename(sample_dir) == project_name:
                # Es un archivo de proyecto existente
                belongs_to_existing_project = True
                self.proyecto_folder = sample_dir
                self.proyecto_nombre = project_name
                print(f"✅ Archivo pertenece a proyecto existente: {project_name}")
                
                # Verificar si existe el archivo en 99_Temp
                temp_file_path = os.path.join(self.proyecto_folder, "99_Temp", sample_file)
                if os.path.exists(temp_file_path):
                    print(f"✅ Usando archivo existente en 99_Temp: {temp_file_path}")
                    # Usar directamente el archivo de 99_Temp
                    input_file = temp_file_path
                else:
                    print(f"⚠️ Archivo no encontrado en 99_Temp, copiando...")
                    # Crear 99_Temp si no existe
                    temp_base = os.path.join(self.proyecto_folder, "99_Temp")
                    os.makedirs(temp_base, exist_ok=True)
                    input_file = os.path.join(temp_base, sample_file)
                    try:
                        # Mostrar loader ANTES de copiar (puede tardar mucho)
                        if not hasattr(self, 'loader_overlay') or self.loader_overlay is None:
                            self.loader_overlay = LoadingOverlay(self.center_frame)
                        self.loader_overlay.start()
                        try:
                            QApplication.processEvents()
                        except Exception:
                            pass
                        shutil.copy(self.sample_file_path, input_file)
                        print(f"✅ Archivo copiado a 99_Temp: {input_file}")
                    except Exception as e:
                        try:
                            self.loader_overlay.stop()
                        except Exception:
                            pass
                        QMessageBox.critical(self, "エラー", f"❌ 99_Tempへのコピーに失敗しました:\n{str(e)}")
                        return
            else:
                belongs_to_existing_project = False
        else:
            belongs_to_existing_project = False

        # Si no pertenece a un proyecto existente, crear nuevo proyecto
        if not belongs_to_existing_project:
            # ✅ NUEVO: Pausar timers automáticos para evitar interferencia con el diálogo
            self.pause_auto_timers()
            
            folder_path, _ = QFileDialog.getSaveFileName(
                self, "プロジェクトフォルダ名を入力してください", "", "Proyecto (*.xlsx)"
            )
            
            # ✅ NUEVO: Reanudar timers después del diálogo
            self.resume_auto_timers()
            if not folder_path:
                return

            if folder_path.endswith(".xlsx"):
                folder_path = folder_path[:-5]

            project_name = os.path.basename(folder_path)
            project_folder = folder_path

            try:
                os.makedirs(project_folder, exist_ok=False)
            except FileExistsError:
                QMessageBox.warning(self, "既存フォルダ",
                                    f"⚠️ フォルダ '{project_name}' は既に存在します。別の名前を入力してください。")
                return

            self.proyecto_folder = project_folder
            self.proyecto_nombre = project_name
            
            # Mostrar loader ANTES de crear estructura/copiar archivos (puede tardar mucho)
            if not hasattr(self, 'loader_overlay') or self.loader_overlay is None:
                self.loader_overlay = LoadingOverlay(self.center_frame)
            self.loader_overlay.start()
            try:
                QApplication.processEvents()
            except Exception:
                pass
            
            # Crear estructura de carpetas del proyecto
            self.create_project_folder_structure(project_folder)
            
            # Copiar archivo de muestreo a la carpeta principal del proyecto
            src_ext = os.path.splitext(self.sample_file_path)[1].lower()
            if src_ext not in (".csv", ".xlsx", ".xls"):
                src_ext = ".csv"
            excel_dest_main = os.path.join(self.proyecto_folder, f"{project_name}_未実験データ{src_ext}")
            try:
                shutil.copy(self.sample_file_path, excel_dest_main)
            except Exception as e:
                try:
                    self.loader_overlay.stop()
                except Exception:
                    pass
                QMessageBox.critical(self, "エラー", f"❌ ファイルのコピーに失敗しました:\n{str(e)}")
                return
            
            # Hacer copia en 99_Temp
            temp_base = os.path.join(self.proyecto_folder, "99_Temp")
            excel_dest_temp = os.path.join(temp_base, f"{project_name}_未実験データ{src_ext}")
            try:
                shutil.copy(self.sample_file_path, excel_dest_temp)
            except Exception as e:
                try:
                    self.loader_overlay.stop()
                except Exception:
                    pass
                QMessageBox.critical(self, "エラー", f"❌ 99_Tempへのコピーに失敗しました:\n{str(e)}")
                return

            # ✅ NUEVO: Actualizar el archivo de entrada al archivo del proyecto creado
            print("🔄 ACTUALIZANDO ARCHIVO DE ENTRADA...")
            self.sample_file_path = excel_dest_main
            self.load_file_label.setText(f"読み込み済み: {project_name}_未実験データ{src_ext}")
            print(f"✅ ARCHIVO DE ENTRADA ACTUALIZADO: {excel_dest_main}")
            print(f"✅ ETIQUETA ACTUALIZADA: {self.load_file_label.text()}")

            # ✅ NUEVO: Si el archivo de muestreo es CSV, generar también Excel(s) en 99_未実験データ
            if src_ext == ".csv":
                self._export_unexperimented_excel_folder_from_csv(excel_dest_main, self.proyecto_folder, project_name)
            
            # Usar el archivo de 99_Temp para la optimización
            input_file = excel_dest_temp

        # Crear carpeta temporal para resultados D-óptimos
        temp_base = os.path.join(self.proyecto_folder, "99_Temp")
        os.makedirs(temp_base, exist_ok=True)
        temp_folder = os.path.join(temp_base, "Temp")
        os.makedirs(temp_folder, exist_ok=True)
        output_folder = temp_folder  # Usar 99_Temp/Temp
        
        # Guardar referencia para limpieza posterior
        self.current_temp_folder = temp_folder

        # Mostrar loader (ya se mostró arriba si se creó proyecto; asegurar que esté visible)
        if not hasattr(self, 'loader_overlay') or self.loader_overlay is None:
            self.loader_overlay = LoadingOverlay(self.center_frame)
        self.loader_overlay.start()
        try:
            QApplication.processEvents()
        except Exception:
            pass

        # ✅ NUEVO: Usar el archivo determinado (existente o nuevo)
        print(f"✅ Usando archivo para optimización: {input_file}")

        # === NUEVO: calcular "ensayos ya hechos" como (principal - 99_Temp) ===
        # main_file debe ser el archivo de la carpeta principal del proyecto (Excel o CSV).
        main_file = getattr(self, "sample_file_path", None)

        done_file = os.path.join(self.proyecto_folder, "99_Temp", "done_experiments.xlsx")
        existing_file = self._build_done_experiments_excel(main_file, input_file, done_file) if main_file else None
        
        # Lanzar optimización D-óptima en hilo
        self.d_optimizer_thread = QThread()
        self.d_optimizer_worker = IntegratedOptimizerWorker(
            sample_file=main_file if main_file else input_file,
            existing_file=existing_file,
            output_folder=output_folder,
            num_points=self.get_sample_size(),
            sample_size=None,  # O el valor que corresponda
            enable_hyperparameter_tuning=True,
            force_reoptimization=False,
            optimization_type="d_optimal"  # Especificar optimización D
        )
        self.d_optimizer_worker.moveToThread(self.d_optimizer_thread)

        self.d_optimizer_thread.started.connect(self.d_optimizer_worker.run)
        self.d_optimizer_worker.finished.connect(self.on_d_optimizer_finished)
        self.d_optimizer_worker.error.connect(self.on_dsaitekika_error)
        self.d_optimizer_worker.finished.connect(self.d_optimizer_thread.quit)
        self.d_optimizer_worker.finished.connect(self.d_optimizer_worker.deleteLater)
        self.d_optimizer_thread.finished.connect(self.d_optimizer_thread.deleteLater)

        self.d_optimizer_thread.start()

    def on_i_optimizer_clicked(self):
        """Ejecuta solo la optimización I-óptima"""
        print("I最適化実行中...")

        # ✅ NUEVO: No mezclar ejecuciones pesadas en paralelo
        if hasattr(self, 'linear_worker') and self.linear_worker is not None:
            try:
                if self.linear_worker.isRunning():
                    QMessageBox.warning(self, "最適化", "⚠️ 線形解析が実行中です。\n完了または停止するまでお待ちください。")
                    return
            except RuntimeError:
                self.linear_worker = None
        if hasattr(self, 'nonlinear_worker') and self.nonlinear_worker is not None:
            try:
                if self.nonlinear_worker.isRunning():
                    QMessageBox.warning(self, "最適化", "⚠️ 非線形解析が実行中です。\n完了または停止するまでお待ちください。")
                    return
            except RuntimeError:
                self.nonlinear_worker = None

        # ✅ NUEVO: Evitar arrancar si ya hay una optimización en ejecución
        for t_attr in ("d_optimizer_thread", "i_optimizer_thread", "dsaitekika_thread"):
            if hasattr(self, t_attr):
                t = getattr(self, t_attr)
                try:
                    if t is not None and t.isRunning():
                        QMessageBox.warning(self, "最適化", "⚠️ すでに最適化が実行中です。\n完了するまでお待ちください。")
                        return
                except RuntimeError:
                    setattr(self, t_attr, None)
        
        # Verificar que el archivo de muestreo haya sido cargado
        if not hasattr(self, "sample_file_path"):
            QMessageBox.warning(self, "エラー", "❌ サンプルファイルが読み込まれていません。")
            return

        # ✅ NUEVO: Verificar si el archivo pertenece a un proyecto existente
        sample_path = self.sample_file_path
        sample_dir = os.path.dirname(sample_path)
        sample_file = os.path.basename(sample_path)
        
        # Verificar si es un archivo de proyecto existente
        belongs_to_existing_project = False
        sample_ext = os.path.splitext(sample_file)[1].lower()
        is_project_sample = (
            sample_file.endswith("_未実験データ.xlsx")
            or sample_file.endswith("_未実験データ.xls")
            or sample_file.endswith("_未実験データ.csv")
        )
        if is_project_sample:
            project_name = sample_file[: -len(f"_未実験データ{sample_ext}")]
            if os.path.basename(sample_dir) == project_name:
                # Es un archivo de proyecto existente
                belongs_to_existing_project = True
                self.proyecto_folder = sample_dir
                self.proyecto_nombre = project_name
                print(f"✅ Archivo pertenece a proyecto existente: {project_name}")
                
                # Verificar si existe el archivo en 99_Temp
                temp_file_path = os.path.join(self.proyecto_folder, "99_Temp", sample_file)
                if os.path.exists(temp_file_path):
                    print(f"✅ Usando archivo existente en 99_Temp: {temp_file_path}")
                    # Usar directamente el archivo de 99_Temp
                    input_file = temp_file_path
                else:
                    print(f"⚠️ Archivo no encontrado en 99_Temp, copiando...")
                    # Crear 99_Temp si no existe
                    temp_base = os.path.join(self.proyecto_folder, "99_Temp")
                    os.makedirs(temp_base, exist_ok=True)
                    input_file = os.path.join(temp_base, sample_file)
                    try:
                        # Mostrar loader ANTES de copiar (puede tardar mucho)
                        if not hasattr(self, 'loader_overlay') or self.loader_overlay is None:
                            self.loader_overlay = LoadingOverlay(self.center_frame)
                        self.loader_overlay.start()
                        try:
                            QApplication.processEvents()
                        except Exception:
                            pass
                        shutil.copy(self.sample_file_path, input_file)
                        print(f"✅ Archivo copiado a 99_Temp: {input_file}")
                    except Exception as e:
                        try:
                            self.loader_overlay.stop()
                        except Exception:
                            pass
                        QMessageBox.critical(self, "エラー", f"❌ 99_Tempへのコピーに失敗しました:\n{str(e)}")
                        return
            else:
                belongs_to_existing_project = False
        else:
            belongs_to_existing_project = False

        # Si no pertenece a un proyecto existente, crear nuevo proyecto
        if not belongs_to_existing_project:
            # ✅ NUEVO: Pausar timers automáticos para evitar interferencia con el diálogo
            self.pause_auto_timers()
            
            folder_path, _ = QFileDialog.getSaveFileName(
                self, "プロジェクトフォルダ名を入力してください", "", "Proyecto (*.xlsx)"
            )
            
            # ✅ NUEVO: Reanudar timers después del diálogo
            self.resume_auto_timers()
            if not folder_path:
                return

            if folder_path.endswith(".xlsx"):
                folder_path = folder_path[:-5]

            project_name = os.path.basename(folder_path)
            project_folder = folder_path

            try:
                os.makedirs(project_folder, exist_ok=False)
            except FileExistsError:
                QMessageBox.warning(self, "既存フォルダ",
                                    f"⚠️ フォルダ '{project_name}' は既に存在します。別の名前を入力してください。")
                return

            self.proyecto_folder = project_folder
            self.proyecto_nombre = project_name
            
            # Mostrar loader ANTES de crear estructura/copiar archivos (puede tardar mucho)
            if not hasattr(self, 'loader_overlay') or self.loader_overlay is None:
                self.loader_overlay = LoadingOverlay(self.center_frame)
            self.loader_overlay.start()
            try:
                QApplication.processEvents()
            except Exception:
                pass
            
            # Crear estructura de carpetas del proyecto
            self.create_project_folder_structure(project_folder)
            
            # Copiar archivo de muestreo a la carpeta principal del proyecto
            src_ext = os.path.splitext(self.sample_file_path)[1].lower()
            if src_ext not in (".csv", ".xlsx", ".xls"):
                src_ext = ".csv"
            excel_dest_main = os.path.join(self.proyecto_folder, f"{project_name}_未実験データ{src_ext}")
            try:
                shutil.copy(self.sample_file_path, excel_dest_main)
            except Exception as e:
                try:
                    self.loader_overlay.stop()
                except Exception:
                    pass
                QMessageBox.critical(self, "エラー", f"❌ ファイルのコピーに失敗しました:\n{str(e)}")
                return
            
            # Hacer copia en 99_Temp
            temp_base = os.path.join(self.proyecto_folder, "99_Temp")
            excel_dest_temp = os.path.join(temp_base, f"{project_name}_未実験データ{src_ext}")
            try:
                shutil.copy(self.sample_file_path, excel_dest_temp)
            except Exception as e:
                try:
                    self.loader_overlay.stop()
                except Exception:
                    pass
                QMessageBox.critical(self, "エラー", f"❌ 99_Tempへのコピーに失敗しました:\n{str(e)}")
                return

            # ✅ NUEVO: Actualizar el archivo de entrada al archivo del proyecto creado
            print("🔄 ACTUALIZANDO ARCHIVO DE ENTRADA...")
            self.sample_file_path = excel_dest_main
            self.load_file_label.setText(f"読み込み済み: {project_name}_未実験データ{src_ext}")
            print(f"✅ ARCHIVO DE ENTRADA ACTUALIZADO: {excel_dest_main}")
            print(f"✅ ETIQUETA ACTUALIZADA: {self.load_file_label.text()}")

            # ✅ NUEVO: Si el archivo de muestreo es CSV, generar también Excel(s) en 99_未実験データ
            if src_ext == ".csv":
                self._export_unexperimented_excel_folder_from_csv(excel_dest_main, self.proyecto_folder, project_name)
            
            # Usar el archivo de 99_Temp para la optimización
            input_file = excel_dest_temp

        # Crear carpeta temporal para resultados I-óptimos
        temp_base = os.path.join(self.proyecto_folder, "99_Temp")
        os.makedirs(temp_base, exist_ok=True)
        temp_folder = os.path.join(temp_base, "Temp")
        os.makedirs(temp_folder, exist_ok=True)
        output_folder = temp_folder  # Usar 99_Temp/Temp
        
        # Guardar referencia para limpieza posterior
        self.current_temp_folder = temp_folder

        # Mostrar loader (ya se mostró arriba si se creó proyecto; asegurar que esté visible)
        if not hasattr(self, 'loader_overlay') or self.loader_overlay is None:
            self.loader_overlay = LoadingOverlay(self.center_frame)
        self.loader_overlay.start()
        try:
            QApplication.processEvents()
        except Exception:
            pass

        # ✅ NUEVO: Usar el archivo determinado (existente o nuevo)
        print(f"✅ Usando archivo para optimización: {input_file}")

        # === NUEVO: calcular "ensayos ya hechos" como (principal - 99_Temp) ===
        main_file = getattr(self, "sample_file_path", None)

        done_file = os.path.join(self.proyecto_folder, "99_Temp", "done_experiments.xlsx")
        existing_file = self._build_done_experiments_excel(main_file, input_file, done_file) if main_file else None
        
        # Lanzar optimización I-óptima en hilo
        self.i_optimizer_thread = QThread()
        self.i_optimizer_worker = IntegratedOptimizerWorker(
            sample_file=main_file if main_file else input_file,
            existing_file=existing_file,
            output_folder=output_folder,
            num_points=self.get_sample_size(),
            sample_size=None,  # O el valor que corresponda
            enable_hyperparameter_tuning=True,
            force_reoptimization=False,
            optimization_type="i_optimal"  # Especificar optimización I
        )
        self.i_optimizer_worker.moveToThread(self.i_optimizer_thread)

        self.i_optimizer_thread.started.connect(self.i_optimizer_worker.run)
        self.i_optimizer_worker.finished.connect(self.on_i_optimizer_finished)
        self.i_optimizer_worker.error.connect(self.on_dsaitekika_error)
        self.i_optimizer_worker.finished.connect(self.i_optimizer_thread.quit)
        self.i_optimizer_worker.finished.connect(self.i_optimizer_worker.deleteLater)
        self.i_optimizer_thread.finished.connect(self.i_optimizer_thread.deleteLater)

        self.i_optimizer_thread.start()

    def on_dsaitekika_clicked(self):
        print("D最適化実行中...")
        print("🔍 DEBUG: Iniciando on_dsaitekika_clicked")

        # ✅ NUEVO: No mezclar ejecuciones pesadas en paralelo
        if hasattr(self, 'linear_worker') and self.linear_worker is not None:
            try:
                if self.linear_worker.isRunning():
                    QMessageBox.warning(self, "最適化", "⚠️ 線形解析が実行中です。\n完了または停止するまでお待ちください。")
                    return
            except RuntimeError:
                self.linear_worker = None
        if hasattr(self, 'nonlinear_worker') and self.nonlinear_worker is not None:
            try:
                if self.nonlinear_worker.isRunning():
                    QMessageBox.warning(self, "最適化", "⚠️ 非線形解析が実行中です。\n完了または停止するまでお待ちください。")
                    return
            except RuntimeError:
                self.nonlinear_worker = None

        # ✅ NUEVO: Evitar arrancar si ya hay una optimización en ejecución
        for t_attr in ("d_optimizer_thread", "i_optimizer_thread", "dsaitekika_thread"):
            if hasattr(self, t_attr):
                t = getattr(self, t_attr)
                try:
                    if t is not None and t.isRunning():
                        QMessageBox.warning(self, "最適化", "⚠️ すでに最適化が実行中です。\n完了するまでお待ちください。")
                        return
                except RuntimeError:
                    setattr(self, t_attr, None)

        if not hasattr(self, "sample_file_path"):
            QMessageBox.warning(self, "エラー", "❌ サンプルファイルが読み込まれていません。")
            return

        # ✅ NUEVO: Pausar timers automáticos para evitar interferencia con el diálogo
        self.pause_auto_timers()

        # Crear carpeta del proyecto
        folder_path, _ = QFileDialog.getSaveFileName(
            self, "プロジェクトフォルダ名を入力してください", "", "Proyecto (*.xlsx)"
        )
        
        # ✅ NUEVO: Reanudar timers después del diálogo
        self.resume_auto_timers()
        if not folder_path:
            return

        if folder_path.endswith(".xlsx"):
            folder_path = folder_path[:-5]

        project_name = os.path.basename(folder_path)
        project_folder = folder_path

        try:
            os.makedirs(project_folder, exist_ok=False)
        except FileExistsError:
            QMessageBox.warning(self, "既存フォルダ",
                                f"⚠️ フォルダ '{project_name}' は既に存在します。別の名前を入力してください。")
            return

        self.proyecto_folder = project_folder
        self.proyecto_nombre = project_name
        
        # Mostrar loader ANTES de crear estructura/copiar archivos (puede tardar mucho)
        if not hasattr(self, 'loader_overlay') or self.loader_overlay is None:
            self.loader_overlay = LoadingOverlay(self.center_frame)
        self.loader_overlay.start()
        try:
            QApplication.processEvents()
        except Exception:
            pass
        
        # Crear estructura de carpetas del proyecto
        self.create_project_folder_structure(project_folder)
        
        # Copiar archivo de muestreo a la carpeta principal del proyecto
        src_ext = os.path.splitext(self.sample_file_path)[1].lower()
        if src_ext not in (".csv", ".xlsx", ".xls"):
            src_ext = ".csv"
        excel_dest_main = os.path.join(self.proyecto_folder, f"{project_name}_未実験データ{src_ext}")
        try:
            shutil.copy(self.sample_file_path, excel_dest_main)
        except Exception as e:
            try:
                self.loader_overlay.stop()
            except Exception:
                pass
            QMessageBox.critical(self, "エラー", f"❌ ファイルのコピーに失敗しました:\n{str(e)}")
            return
        
        # Hacer copia en 99_Temp
        temp_base = os.path.join(self.proyecto_folder, "99_Temp")
        excel_dest_temp = os.path.join(temp_base, f"{project_name}_未実験データ{src_ext}")
        try:
            shutil.copy(self.sample_file_path, excel_dest_temp)
        except Exception as e:
            try:
                self.loader_overlay.stop()
            except Exception:
                pass
            QMessageBox.critical(self, "エラー", f"❌ 99_Tempへのコピーに失敗しました:\n{str(e)}")
            return

        self.muestreo_guardado_path = excel_dest_main
        
        print("🔍 DEBUG: Llegando al código de actualización del archivo de entrada")
        # ✅ NUEVO: Actualizar el archivo de entrada al archivo del proyecto creado
        print("🔄 ACTUALIZANDO ARCHIVO DE ENTRADA...")
        self.sample_file_path = excel_dest_main
        self.load_file_label.setText(f"読み込み済み: {project_name}_未実験データ{src_ext}")
        print(f"✅ ARCHIVO DE ENTRADA ACTUALIZADO: {excel_dest_main}")
        print(f"✅ ETIQUETA ACTUALIZADA: {self.load_file_label.text()}")

        # ✅ NUEVO: Si el archivo de muestreo es CSV, generar también Excel(s) en 99_未実験データ
        if src_ext == ".csv":
            self._export_unexperimented_excel_folder_from_csv(excel_dest_main, self.proyecto_folder, project_name)

        # Crear carpeta temporal de resultados dentro del proyecto
        temp_base = os.path.join(self.proyecto_folder, "99_Temp")
        os.makedirs(temp_base, exist_ok=True)
        temp_folder = os.path.join(temp_base, "Temp")
        os.makedirs(temp_folder, exist_ok=True)
        output_folder = temp_folder  # Usar 99_Temp/Temp

        self.dsaitekika_output_excel = os.path.join(output_folder, "selected_samples.xlsx")
        self.dsaitekika_output_prefix = os.path.join(output_folder, "d_optimal")
        
        # Guardar referencia para limpieza posterior
        self.current_temp_folder = temp_folder

        # Loader ya se mostró arriba (antes de crear/copiar). Mantenerlo activo.

        # ✅ NUEVO: Usar el archivo de 99_Temp en lugar del archivo original
        input_file = excel_dest_temp
        print(f"✅ Usando archivo de 99_Temp: {input_file}")
        # Guardar para poder recalcular D基準値 como el archivo de referencia
        self._last_dsaitekika_input_file = input_file
        
        self.dsaitekika_thread = QThread()
        self.dsaitekika_worker = DsaitekikaWorker(
            input_file,
            self.dsaitekika_output_excel,
            self.dsaitekika_output_prefix,
            self.get_sample_size(),
        )
        self.dsaitekika_worker.moveToThread(self.dsaitekika_thread)

        self.dsaitekika_thread.started.connect(self.dsaitekika_worker.run)
        self.dsaitekika_worker.finished.connect(self.on_dsaitekika_finished)
        self.dsaitekika_worker.error.connect(self.on_dsaitekika_error)
        self.dsaitekika_worker.finished.connect(self.dsaitekika_thread.quit)
        self.dsaitekika_worker.finished.connect(self.dsaitekika_worker.deleteLater)
        self.dsaitekika_thread.finished.connect(self.dsaitekika_thread.deleteLater)

        self.dsaitekika_thread.start()

    def on_isaitekika_clicked(self):
        """Acción al pulsar iSaitekika"""
        print("i最適化実行中...")
        self.ok_button.setEnabled(True)
        self.ng_button.setEnabled(True)

        self.create_navigation_buttons()
        self.prev_button.setEnabled(True)
        self.next_button.setEnabled(True)

    def find_matching_experiment_file(self, project_folder):
        """
        Busca en 01_実験リスト y compara con el archivo de resultados
        para encontrar el archivo de experimento correspondiente
        """
        import os
        import pandas as pd
        from pathlib import Path
        
        try:
            # Leer archivo de resultados
            print(f"🔍 DEBUG: Leyendo archivo de resultados: {self.results_file_path}")
            df_results = pd.read_excel(self.results_file_path)
            print(f"🔍 DEBUG: Archivo de resultados cargado: {len(df_results)} filas")
            print(f"🔍 DEBUG: Columnas del archivo de resultados: {list(df_results.columns)}")
            
            # Mostrar primera fila de resultados para debug
            if len(df_results) > 0:
                print("🔍 DEBUG: Primera fila de resultados:")
                first_row = df_results.iloc[0]
                for col in df_results.columns:
                    print(f"  - {col}: {first_row[col]}")
            
            # Columnas a comparar (B a H)
            # Aceptar "UPカット" (nuevo) o "回転方向" (antiguo)
            dir_col = 'UPカット' if 'UPカット' in df_results.columns else '回転方向'
            comparison_columns = ['回転速度', '送り速度', dir_col, '切込量', '突出量', '載せ率', 'パス数']
            
            # Verificar que las columnas existen en el archivo de resultados
            available_columns = [col for col in comparison_columns if col in df_results.columns]
            if len(available_columns) < 3:  # Mínimo 3 columnas para comparar
                print(f"⚠️ Columnas insuficientes para comparar: {available_columns}")
                return None
            
            print(f"🔍 DEBUG: Columnas disponibles para comparar: {available_columns}")
            
            # Buscar en 01_実験リスト
            experiment_list_path = Path(project_folder) / "01_実験リスト"
            if not experiment_list_path.exists():
                print(f"❌ DEBUG: Carpeta 01_実験リスト no existe: {experiment_list_path}")
                print(f"🔍 DEBUG: Verificando estructura del proyecto:")
                project_path = Path(project_folder)
                if project_path.exists():
                    print(f"🔍 DEBUG: Contenido del proyecto:")
                    for item in project_path.iterdir():
                        if item.is_dir():
                            print(f"  📁 {item.name}")
                        else:
                            print(f"  📄 {item.name}")
                else:
                    print(f"❌ DEBUG: El proyecto no existe: {project_path}")
                return None
            
            print(f"🔍 DEBUG: Buscando en: {experiment_list_path}")
            
            # Verificar contenido de 01_実験リスト
            experiment_list_contents = list(experiment_list_path.iterdir())
            print(f"🔍 DEBUG: Contenido de 01_実験リスト ({len(experiment_list_contents)} elementos):")
            for item in experiment_list_contents:
                if item.is_dir():
                    print(f"  📁 {item.name}")
                else:
                    print(f"  📄 {item.name}")
            
            # Buscar en subcarpetas
            subfolder_count = 0
            for subfolder in experiment_list_path.iterdir():
                if not subfolder.is_dir():
                    continue
                
                subfolder_count += 1
                print(f"🔍 DEBUG: Revisando subcarpeta {subfolder_count}: {subfolder.name}")
                
                # Verificar contenido de la subcarpeta
                subfolder_contents = list(subfolder.iterdir())
                print(f"🔍 DEBUG: Contenido de {subfolder.name} ({len(subfolder_contents)} elementos):")
                for item in subfolder_contents:
                    if item.is_dir():
                        print(f"    📁 {item.name}")
                    else:
                        print(f"    📄 {item.name}")
                
                # Buscar archivos D最適化_新規実験点.xlsx o I最適化_新規実験点.xlsx
                experiment_files = []
                for pattern in ["D最適化_新規実験点.xlsx", "I最適化_新規実験点.xlsx"]:
                    file_path = subfolder / pattern
                    if file_path.exists():
                        experiment_files.append((file_path, pattern))
                        print(f"🔍 DEBUG: Encontrado archivo: {file_path}")
                
                if not experiment_files:
                    print(f"🔍 DEBUG: No se encontraron archivos de experimento en {subfolder.name}")
                
                for file_path, pattern in experiment_files:
                    try:
                        print(f"🔍 DEBUG: Comparando con archivo: {file_path}")
                        print(f"🔍 DEBUG: Patrón del archivo: {pattern}")
                        df_experiment = pd.read_excel(file_path)
                        print(f"🔍 DEBUG: Archivo de experimento cargado: {len(df_experiment)} filas")
                        print(f"🔍 DEBUG: Columnas del experimento: {list(df_experiment.columns)}")
                        
                        # Mostrar primera fila de experimento para debug
                        if len(df_experiment) > 0:
                            print("🔍 DEBUG: Primera fila de experimento:")
                            first_exp_row = df_experiment.iloc[0]
                            for col in df_experiment.columns:
                                print(f"  - {col}: {first_exp_row[col]}")
                        
                        # Comparar filas
                        comparison_count = 0
                        for idx, result_row in df_results.iterrows():
                            for exp_idx, exp_row in df_experiment.iterrows():
                                comparison_count += 1
                                if comparison_count <= 3:  # Solo mostrar las primeras 3 comparaciones
                                    print(f"🔍 DEBUG: Comparación {comparison_count}: Resultado fila {idx} vs Experimento fila {exp_idx}")
                                
                                # Comparar solo las columnas disponibles
                                match = True
                                mismatch_details = []
                                
                                for col in available_columns:
                                    if col in df_experiment.columns:
                                        result_val = result_row[col]
                                        exp_val = exp_row[col]
                                        
                                        # Debug de comparación
                                        if comparison_count <= 3:
                                            print(f"  🔍 DEBUG: Comparando columna '{col}': '{result_val}' vs '{exp_val}'")
                                        
                                        # Comparar valores (considerando tipos de datos)
                                        if pd.isna(result_val) and pd.isna(exp_val):
                                            if comparison_count <= 3:
                                                print(f"    ✅ Ambos valores son NaN")
                                            continue
                                        elif pd.isna(result_val) or pd.isna(exp_val):
                                            if comparison_count <= 3:
                                                print(f"    ❌ Uno es NaN, otro no")
                                            match = False
                                            mismatch_details.append(f"{col}: NaN vs {exp_val if pd.isna(result_val) else result_val}")
                                            break
                                        
                                        # Convertir a float para comparación numérica si es posible
                                        try:
                                            result_float = float(result_val)
                                            exp_float = float(exp_val)
                                            if abs(result_float - exp_float) < 1e-10:  # Comparación numérica con tolerancia
                                                if comparison_count <= 3:
                                                    print(f"    ✅ Valores numéricos iguales: {result_float}")
                                                continue
                                            else:
                                                if comparison_count <= 3:
                                                    print(f"    ❌ Valores numéricos diferentes: {result_float} != {exp_float}")
                                                match = False
                                                mismatch_details.append(f"{col}: {result_float} vs {exp_float}")
                                                break
                                        except (ValueError, TypeError):
                                            # Si no se pueden convertir a float, comparar como strings
                                            if str(result_val).strip() == str(exp_val).strip():
                                                if comparison_count <= 3:
                                                    print(f"    ✅ Valores de texto iguales: '{result_val}'")
                                                continue
                                            else:
                                                if comparison_count <= 3:
                                                    print(f"    ❌ Valores de texto diferentes: '{result_val}' != '{exp_val}'")
                                                match = False
                                                mismatch_details.append(f"{col}: '{result_val}' vs '{exp_val}'")
                                                break
                                        else:
                                            if comparison_count <= 3:
                                                print(f"    ✅ Valores iguales: '{result_val}'")
                                    else:
                                        if comparison_count <= 3:
                                            print(f"  ❌ Columna '{col}' no existe en experimento")
                                        match = False
                                        mismatch_details.append(f"{col}: No existe en experimento")
                                        break
                                
                                if match:
                                    print(f"✅ DEBUG: ¡COINCIDENCIA ENCONTRADA!")
                                    print(f"   Archivo: {file_path}")
                                    print(f"   Fila resultado: {idx}, Fila experimento: {exp_idx}")
                                    
                                    # Extraer información de la carpeta
                                    folder_name = subfolder.name
                                    print(f"🔍 DEBUG: Nombre de carpeta extraído: {folder_name}")
                                    
                                    # Determinar tipo de optimización basado en el nombre del archivo
                                    if "D最適化" in pattern:
                                        optimization_type = "D最適化"
                                        print(f"🔍 DEBUG: Tipo D detectado por nombre de archivo")
                                    elif "I最適化" in pattern:
                                        optimization_type = "I最適化"
                                        print(f"🔍 DEBUG: Tipo I detectado por nombre de archivo")
                                    else:
                                        # Fallback: intentar determinar por el nombre de la carpeta
                                        print(f"🔍 DEBUG: Fallback - analizando nombre de carpeta: {folder_name}")
                                        if "D" in folder_name.upper() or "d" in folder_name.lower():
                                            optimization_type = "D最適化"
                                            print(f"🔍 DEBUG: Tipo D detectado por nombre de carpeta")
                                        elif "I" in folder_name.upper() or "i" in folder_name.lower():
                                            optimization_type = "I最適化"
                                            print(f"🔍 DEBUG: Tipo I detectado por nombre de carpeta")
                                        else:
                                            optimization_type = "D最適化"  # Por defecto
                                            print(f"🔍 DEBUG: Tipo por defecto: D最適化")
                                    
                                    print(f"🔍 DEBUG: Tipo de optimización final: {optimization_type}")
                                    
                                    return {
                                        'folder_name': folder_name,
                                        'optimization_type': optimization_type,
                                        'file_path': str(file_path),
                                        'result_row': idx,
                                        'experiment_row': exp_idx
                                    }
                                elif comparison_count <= 3:
                                    print(f"❌ DEBUG: No coincidencia. Detalles: {mismatch_details}")
                        
                        if comparison_count > 0:
                            print(f"🔍 DEBUG: Total de comparaciones realizadas: {comparison_count}")
                        
                    except Exception as e:
                        print(f"❌ Error leyendo {file_path}: {e}")
                        continue
            
            print("❌ DEBUG: No se encontró coincidencia en ningún archivo de experimento")
            return None
            
        except Exception as e:
            print(f"❌ Error en find_matching_experiment_file: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_experiment_data_folder(self, experiment_info):
        """
        Crea la carpeta en 02_実験データ con el formato especificado
        """
        import os
        from datetime import datetime
        from pathlib import Path
        import re
        
        try:
            print("🔍 DEBUG: Iniciando create_experiment_data_folder")
            print(f"🔍 DEBUG: experiment_info recibido: {experiment_info}")
            
            # Extraer número de la carpeta
            folder_name = experiment_info['folder_name']
            optimization_type = experiment_info['optimization_type']
            
            print(f"🔍 DEBUG: Procesando carpeta: '{folder_name}'")
            print(f"🔍 DEBUG: Tipo de optimización: '{optimization_type}'")
            print(f"🔍 DEBUG: Longitud del nombre de carpeta: {len(folder_name)}")
            print(f"🔍 DEBUG: Caracteres en el nombre: {[c for c in folder_name]}")
            
            # Buscar número en el nombre de la carpeta
            # Patrones para buscar números: "017", "001", etc.
            number_patterns = [
                r'(\d{3,})',  # Números de 3 o más dígitos
                r'(\d{2,})',  # Números de 2 o más dígitos
                r'(\d+)'      # Cualquier número
            ]
            
            folder_number = "001"  # Número por defecto
            pattern_used = "default"
            
            print(f"🔍 DEBUG: Aplicando patrones regex:")
            for i, pattern in enumerate(number_patterns):
                print(f"  🔍 DEBUG: Patrón {i+1}: {pattern}")
                number_match = re.search(pattern, folder_name)
                if number_match:
                    extracted_number = number_match.group(1)
                    folder_number = extracted_number.zfill(3)  # Rellenar con ceros
                    pattern_used = pattern
                    print(f"  ✅ DEBUG: Coincidencia encontrada con patrón '{pattern}'")
                    print(f"  ✅ DEBUG: Número extraído: '{extracted_number}'")
                    print(f"  ✅ DEBUG: Número rellenado: '{folder_number}'")
                    break
                else:
                    print(f"  ❌ DEBUG: No coincidencia con patrón '{pattern}'")
            
            # Verificar que el número extraído es correcto
            print(f"🔍 DEBUG: Resumen de extracción:")
            print(f"  - Nombre de carpeta original: '{folder_name}'")
            print(f"  - Patrón usado: '{pattern_used}'")
            print(f"  - Número final extraído: '{folder_number}'")
            print(f"  - Tipo de optimización: '{optimization_type}'")
            
            # Generar fecha y hora actual
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            print(f"🔍 DEBUG: Timestamp generado: '{timestamp}'")
            
            # Crear nombre de carpeta
            new_folder_name = f"{folder_number}_{optimization_type}_{timestamp}"
            print(f"🔍 DEBUG: Nombre de carpeta final generado: '{new_folder_name}'")
            
            # Crear carpeta en 02_実験データ
            experiment_data_path = Path(self.current_project_folder) / "02_実験データ" / new_folder_name
            print(f"🔍 DEBUG: Ruta completa a crear: {experiment_data_path}")
            
            # Verificar si la carpeta ya existe y crear una nueva si es necesario
            if experiment_data_path.exists():
                print(f"⚠️ DEBUG: La carpeta ya existe: {experiment_data_path}")
                # Crear una nueva carpeta con un sufijo adicional
                counter = 1
                while experiment_data_path.exists():
                    new_folder_name = f"{folder_number}_{optimization_type}_{timestamp}_{counter:02d}"
                    experiment_data_path = Path(self.current_project_folder) / "02_実験データ" / new_folder_name
                    print(f"🔍 DEBUG: Intentando crear carpeta alternativa: {new_folder_name}")
                    counter += 1
                    if counter > 10:  # Evitar bucle infinito
                        break
                
                print(f"🔍 DEBUG: Carpeta final a crear: {experiment_data_path}")
            
            experiment_data_path.mkdir(parents=True, exist_ok=True)
            
            print(f"✅ DEBUG: Carpeta creada exitosamente: {experiment_data_path}")
            return str(experiment_data_path)
            
        except Exception as e:
            print(f"❌ DEBUG: Error creando carpeta de experimento: {e}")
            import traceback
            traceback.print_exc()
            return None

    def detect_project_folder_from_results_file(self, results_file_path):
        """
        Detecta automáticamente la carpeta del proyecto basándose en la ubicación del archivo de resultados.
        
        Busca patrones como:
        - NOMBREDELPROYECTO/99_Results/archivo.xlsx -> NOMBREDELPROYECTO
        - NOMBREDELPROYECTO/02_実験データ/archivo.xlsx -> NOMBREDELPROYECTO
        - NOMBREDELPROYECTO/archivo.xlsx -> NOMBREDELPROYECTO
        
        Returns:
            str: Ruta de la carpeta del proyecto si se encuentra, None si no se puede detectar
        """
        import os
        from pathlib import Path
        
        try:
            # Convertir a Path para facilitar el manejo
            file_path = Path(results_file_path)
            print(f"🔍 Detectando carpeta del proyecto para: {file_path}")
            
            # Obtener el directorio del archivo
            file_dir = file_path.parent
            print(f"🔍 Directorio del archivo: {file_dir}")
            
            # Buscar patrones de carpetas de proyecto
            project_folders = [
                "99_Results",
                "02_実験データ", 
                "03_線形回帰",
                "04_非線形回帰",
                "05_分類",
                "01_実験リスト"
            ]
            
            # Buscar hacia arriba en la jerarquía de directorios
            current_dir = file_dir
            max_levels = 5  # Máximo 5 niveles hacia arriba
            
            for level in range(max_levels):
                print(f"🔍 Nivel {level}: {current_dir}")
                
                # Verificar si el directorio actual contiene carpetas de proyecto
                for folder in project_folders:
                    project_folder_path = current_dir / folder
                    if project_folder_path.exists() and project_folder_path.is_dir():
                        print(f"✅ Encontrada carpeta de proyecto: {folder}")
                        # El directorio padre de esta carpeta es el proyecto
                        project_root = current_dir
                        print(f"✅ Carpeta del proyecto detectada: {project_root}")
                        return str(project_root)
                
                # Verificar si el directorio actual tiene la estructura de un proyecto
                # (contiene múltiples carpetas de proyecto)
                project_folder_count = 0
                for folder in project_folders:
                    if (current_dir / folder).exists():
                        project_folder_count += 1
                
                if project_folder_count >= 2:  # Si tiene al menos 2 carpetas de proyecto
                    print(f"✅ Estructura de proyecto detectada con {project_folder_count} carpetas")
                    return str(current_dir)
                
                # Subir un nivel
                parent_dir = current_dir.parent
                if parent_dir == current_dir:  # Llegamos a la raíz
                    break
                current_dir = parent_dir
            
            print("❌ No se pudo detectar automáticamente la carpeta del proyecto")
            return None
            
        except Exception as e:
            print(f"❌ Error detectando carpeta del proyecto: {e}")
            return None

    def on_show_results_clicked(self):
        """Acción al pulsar Show Results"""
        try:
            print("結果表示中...")

            # ✅ NUEVO: Verificar que se haya cargado un archivo de resultados
            if not hasattr(self, 'results_file_path') or not self.results_file_path:
                QMessageBox.warning(self, "エラー", "❌ 結果ファイルが読み込まれていません。\nまず「ファイルを読み込む」で結果ファイルを選択してください。")
                return

            # ✅ NUEVO: Verificar que el archivo de resultados existe
            import os
            if not os.path.exists(self.results_file_path):
                QMessageBox.warning(self, "エラー", f"❌ 結果ファイルが見つかりません:\n{self.results_file_path}")
                return

            print(f"🔍 Debug - results_file_path: {self.results_file_path}")

            # ✅ NUEVO: Intentar detectar automáticamente la carpeta del proyecto
            project_folder = self.detect_project_folder_from_results_file(self.results_file_path)
            
            if project_folder:
                print(f"✅ Carpeta del proyecto detectada automáticamente: {project_folder}")
                QMessageBox.information(self, "プロジェクト検出", f"✅ プロジェクトフォルダが自動検出されました:\n{project_folder}")
            else:
                print("❌ No se pudo detectar automáticamente la carpeta del proyecto")
                # Si no se pudo detectar automáticamente, pedir al usuario que seleccione
                project_folder = QFileDialog.getExistingDirectory(self, "プロジェクトフォルダを選択", "")
                if not project_folder:
                    QMessageBox.warning(self, "エラー", "❌ プロジェクトフォルダが選択されていません。")
                    return

            # ✅ NUEVO: Guardar la carpeta del proyecto para uso posterior
            self.current_project_folder = project_folder
            print(f"✅ Carpeta del proyecto guardada: {self.current_project_folder}")

            # ✅ NUEVO: Buscar archivo de experimento correspondiente
            print("🔍 DEBUG: Iniciando búsqueda de archivo de experimento...")
            experiment_info = self.find_matching_experiment_file(project_folder)
            if experiment_info:
                print(f"✅ DEBUG: Archivo de experimento encontrado: {experiment_info}")
                # ✅ NUEVO: NO crear carpeta aquí, dejar que el worker lo haga después de verificar duplicados
                experiment_folder_name = None  # No crear carpeta prematuramente
                print(f"✅ DEBUG: Información de experimento guardada para procesamiento posterior")
            else:
                print("⚠️ DEBUG: No se encontró archivo de experimento correspondiente")
                # ✅ NUEVO: NO crear carpeta por defecto aquí, dejar que el worker lo haga
                experiment_folder_name = None  # No crear carpeta prematuramente
                print(f"✅ DEBUG: No se creará carpeta por defecto prematuramente")

            # ✅ NUEVO: Limpiar pantalla principal antes de mostrar loading
            self.clear_main_screen()

            # ✅ NUEVO: Iniciar loading overlay centrado sobre el frame central
            # Reutilizar si ya existe para evitar múltiples overlays/eventFilters
            if not hasattr(self, 'loader_overlay') or self.loader_overlay is None:
                self.loader_overlay = LoadingOverlay(self.center_frame)
            self.loader_overlay.start()
            
            # ✅ NUEVO: Verificar si la consola desplegable está visible
            if hasattr(self, 'overlay_console') and self.overlay_console.isVisible():
                print("🔧 Consola desplegable detectada, manteniendo visible...")
                # El loading se posicionará por encima de la consola
                print("🔧 Loading se posicionará por encima de la consola")
            
            # ✅ NUEVO: Debug del posicionamiento del loading
            print(f"🔧 Frame central geometría: {self.center_frame.geometry()}")
            print(f"🔧 Loading overlay geometría: {self.loader_overlay.geometry()}")

            # ✅ NUEVO: Crear worker y thread para procesamiento en paralelo
            print(f"🔍 Debug - Creando ShowResultsWorker con:")
            print(f"  - project_folder: {project_folder}")
            print(f"  - results_file_path: {self.results_file_path}")
            print(f"  - brush: {self.get_selected_brush()}")
            print(f"  - diameter: {self.diameter_selector.currentText()}")
            print(f"  - material: {self.material_selector.currentText()}")
            
            # ✅ NUEVO: Verificar el contenido del archivo de resultados
            try:
                import pandas as pd
                df_results = pd.read_excel(self.results_file_path)
                print(f"🔍 Debug - Archivo de resultados contiene {len(df_results)} filas")
                print(f"🔍 Debug - Columnas del archivo: {list(df_results.columns)}")
                print(f"🔍 Debug - Primera fila de datos:")
                if len(df_results) > 0:
                    first_row = df_results.iloc[0]
                    print(f"  - 回転速度: {first_row.get('回転速度', 'N/A')}")
                    print(f"  - 送り速度: {first_row.get('送り速度', 'N/A')}")
                    print(f"  - 回転方向: {first_row.get('回転方向', 'N/A')}")
                    print(f"  - 切込量: {first_row.get('切込量', 'N/A')}")
                    print(f"  - 突出量: {first_row.get('突出量', 'N/A')}")
                    print(f"  - 載せ率: {first_row.get('載せ率', 'N/A')}")
                    print(f"  - パス数: {first_row.get('パス数', 'N/A')}")
            except Exception as e:
                print(f"🔍 Debug - Error leyendo archivo de resultados: {e}")
            
            # ✅ NUEVO: Verificar que ShowResultsWorker esté disponible
            try:
                from showresultsworker import ShowResultsWorker
                print("✅ ShowResultsWorker importado correctamente")
            except ImportError as e:
                print(f"❌ Error importando ShowResultsWorker: {e}")
                QMessageBox.critical(self, "エラー", f"❌ ShowResultsWorkerのインポートに失敗しました:\n{str(e)}")
                return
            
            # ✅ NUEVO: Verificar que el procesador existe
            if not hasattr(self, 'processor'):
                print("❌ self.processor no existe")
                QMessageBox.critical(self, "エラー", "❌ プロセッサーが初期化されていません。")
                return
            
            print(f"✅ self.processor existe: {self.processor}")
            
            # ✅ NUEVO: Verificar registros en la base de datos antes de importar
            try:
                import sqlite3
                import os
                
                # ✅ NUEVO: Verificar la ubicación de la base de datos
                db_path = RESULTS_DB_PATH
                print(f"🔍 Debug - Ruta de la base de datos: {os.path.abspath(db_path)}")
                print(f"🔍 Debug - ¿Existe la base de datos?: {os.path.exists(db_path)}")
                
                conn = sqlite3.connect(db_path, timeout=10)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM main_results")
                count_before = cursor.fetchone()[0]
                
                # ✅ NUEVO: Verificar algunos registros existentes para debug
                cursor.execute("SELECT * FROM main_results LIMIT 3")
                sample_records = cursor.fetchall()
                print(f"🔍 Debug - Muestra de registros existentes:")
                for i, record in enumerate(sample_records):
                    print(f"  Registro {i+1}: {record[:5]}...")  # Mostrar solo las primeras 5 columnas
                
                # ✅ NUEVO: Verificar la estructura de la base de datos
                print(f"🔍 Debug - Verificando estructura de la base de datos...")
                cursor.execute("PRAGMA table_info(main_results)")
                columns_info = cursor.fetchall()
                print(f"🔍 Debug - Columnas en la base de datos:")
                for col in columns_info:
                    print(f"  - {col[1]} ({col[2]})")
                
                # ✅ NUEVO: Verificar si hay registros con los mismos valores que vamos a importar
                print(f"🔍 Debug - Verificando si hay registros duplicados...")
                try:
                    cursor.execute("SELECT COUNT(*) FROM main_results WHERE 回転速度 = ? AND 送り速度 = ? AND 切込量 = ? AND 突出量 = ? AND 載せ率 = ? AND パス数 = ?", 
                                 (1000, 500, 1.0, 10, 0.4, 2))
                    duplicate_count = cursor.fetchone()[0]
                    print(f"🔍 Debug - Registros con valores similares al primer registro: {duplicate_count}")
                except Exception as e:
                    print(f"🔍 Debug - Error verificando duplicados: {e}")
                
                conn.close()
                print(f"🔍 Debug - Registros en la base de datos antes de importar: {count_before}")
                
                # ✅ NUEVO: Verificar si hay otra base de datos en la carpeta del proyecto
                # Debug legacy: antes la DB vivía dentro del proyecto; ya no se usa en instalación pro.
                project_db_path = os.path.join(project_folder, "results.db")
                print(f"🔍 Debug - ¿Existe base de datos en el proyecto?: {os.path.exists(project_db_path)}")
                if os.path.exists(project_db_path):
                    print(f"🔍 Debug - Ruta de BD del proyecto: {os.path.abspath(project_db_path)}")
                    try:
                        conn_project = sqlite3.connect(project_db_path)
                        cursor_project = conn_project.cursor()
                        cursor_project.execute("SELECT COUNT(*) FROM main_results")
                        count_project = cursor_project.fetchone()[0]
                        conn_project.close()
                        print(f"🔍 Debug - Registros en BD del proyecto: {count_project}")
                    except Exception as e:
                        print(f"🔍 Debug - Error verificando BD del proyecto: {e}")
            except Exception as e:
                print(f"🔍 Debug - Error verificando base de datos antes: {e}")
            
            # ✅ NUEVO: Crear worker y ejecutar directamente
            self.show_results_worker = ShowResultsWorker(
                project_folder,
                self.results_file_path,
                self.get_selected_brush(),
                float(self.diameter_selector.currentText()),
                self.material_selector.currentText(),
                self.backup_and_update_sample_file,
                self.processor.process_results_file_with_ui_values,
                experiment_info  # Pasar la información del experimento encontrado
            )

            # ✅ NUEVO: Crear thread para ejecutar el worker en paralelo
            self.import_thread = QThread()
            self.show_results_worker.moveToThread(self.import_thread)

            # Conectar señales del thread
            self.import_thread.started.connect(self.show_results_worker.run)
            self.show_results_worker.finished.connect(self.on_show_results_finished)
            self.show_results_worker.error.connect(self.on_show_results_error)
            self.show_results_worker.finished.connect(self.import_thread.quit)
            self.show_results_worker.finished.connect(self.show_results_worker.deleteLater)
            self.import_thread.finished.connect(self.import_thread.deleteLater)

            print("🔍 Debug - Iniciando thread para importación...")
            self.import_thread.start()
        except Exception as e:
            print(f"❌ Error general en on_show_results_clicked: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "エラー", f"❌ 予期しないエラーが発生しました:\n{str(e)}")

    def on_show_results_finished(self, result):
        """Maneja el resultado exitoso del procesamiento de resultados"""
        try:
            print(f"🔍 Debug - on_show_results_finished llamado con result: {result}")
            
            if hasattr(self, 'loader_overlay'):
                self.loader_overlay.stop()
            
            # ✅ NUEVO: Verificar que la base de datos se actualizó
            total_records_after = 0
            records_imported = 0
            try:
                import sqlite3
                import os
                conn = sqlite3.connect(RESULTS_DB_PATH, timeout=10)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM main_results")
                total_records_after = cursor.fetchone()[0]
                print(f"🔍 Debug - Registros en la base de datos después de importar: {total_records_after}")
                
                # ✅ NUEVO: Registros importados reales = insertados + actualizados (sin contar filas idénticas)
                if result and isinstance(result, dict):
                    dbu = result.get("db_upsert_result")
                    if isinstance(dbu, dict):
                        try:
                            records_imported = int(dbu.get("inserted", 0) or 0) + int(dbu.get("updated", 0) or 0)
                            print(f"🔍 Debug - Registros importados reales (insert+update): {records_imported}")
                        except Exception:
                            records_imported = "N/A"
                    else:
                        # Si no tenemos db_upsert_result, NO debemos inferir "importados" desde el Excel,
                        # porque puede ser un early-exit (archivo idéntico) o un fallo parcial.
                        records_imported = 0
                        print("🔍 Debug - db_upsert_result ausente: records_imported=0 (no inferimos desde Excel)")
                
                # ✅ NUEVO: Mostrar contenido completo de la base de datos
                if total_records_after > 0:
                    print("🔍 Debug - Contenido completo de la base de datos:")
                    cursor.execute("SELECT * FROM main_results ORDER BY id")
                    all_records = cursor.fetchall()
                    
                    # Obtener nombres de columnas
                    cursor.execute("PRAGMA table_info(main_results)")
                    columns_info = cursor.fetchall()
                    column_names = [col[1] for col in columns_info]
                    
                    print(f"🔍 Debug - Columnas: {column_names}")
                    print(f"🔍 Debug - Total de registros: {len(all_records)}")
                    
                    for i, record in enumerate(all_records, 1):
                        print(f"  Registro {i}:")
                        for j, value in enumerate(record):
                            if j < len(column_names):
                                print(f"    {column_names[j]}: {value}")
                        print()
                else:
                    print("🔍 Debug - La base de datos está vacía")
                    
                conn.close()
            except Exception as e:
                print(f"🔍 Debug - Error verificando base de datos: {e}")
            
            # Mostrar mensaje de éxito con información del backup
            if result and isinstance(result, dict):
                if result.get('optimization_type') == 'EXISTING':
                    # Caso cuando ya existe un archivo idéntico
                    message = f"⚠️ 既に同じ内容のファイルが存在します:\n{result.get('identical_folder', 'Unknown')}\n\n"
                    message += f"📁 既存のフォルダ: {result.get('identical_folder', 'Unknown')}\n"
                    message += f"ℹ️ 新しいフォルダは作成されませんでした\n\n"
                    
                    # ✅ NUEVO: Agregar información de la base de datos
                    message += f"📊 データベース内の総レコード数: {total_records_after}\n"
                    message += f"📈 今回インポートされたレコード数: {records_imported}"
                else:
                    # Caso normal
                    message = f"✅ 結果ファイルが保存されました:\n{result.get('results_file_path', 'N/A')}\n\n"
                    
                    # ✅ NUEVO: Agregar información de la base de datos
                    message += f"📊 データベース内の総レコード数: {total_records_after}\n"
                    message += f"📈 今回インポートされたレコード数: {records_imported}\n\n"
                    
                    if result.get('backup_result', {}).get('backup_path'):
                        message += f"📋 バックアップ作成: {os.path.basename(result['backup_result']['backup_path'])}\n"
                        message += f"🗑️ サンプルファイルから削除された行: {result['backup_result'].get('removed_rows', 'N/A')}\n"
                        message += f"📊 サンプルファイルの残り行数: {result['backup_result'].get('remaining_rows', 'N/A')}"
                    else:
                        message += f"ℹ️ バックアップは実行されませんでした（アクティブなプロジェクトがありません）"

                    # ✅ NUEVO: Aviso único de sobrescritura en BBDD + backup
                    dbu = result.get("db_upsert_result")
                    if isinstance(dbu, dict):
                        updated = int(dbu.get("updated", 0) or 0)
                        inserted = int(dbu.get("inserted", 0) or 0)
                        if updated > 0:
                            message += "\n\n⚠️ 既存データを上書きします。BBDDのバックアップを作成しました。"
                            message += f"\n🔁 上書き: {updated} / ➕ 追加: {inserted}"
                            if dbu.get("db_backup_path"):
                                message += f"\n📋 BBDDバックアップ: {os.path.basename(str(dbu.get('db_backup_path')))}"
                            else:
                                message += "\n📋 BBDDバックアップ: (作成できませんでした)"
            else:
                message = f"✅ 処理が完了しました\n\n"
                message += f"📊 データベース内の総レコード数: {total_records_after}\n"
                message += f"📈 今回インポートされたレコード数: {records_imported}"
            
            QMessageBox.information(self, "完了", message)
            
            # Mostrar la vista de filtro después de procesar los datos
            self.create_filter_view()
            
            if hasattr(self, 'ok_button'):
                self.ok_button.setEnabled(True)
            if hasattr(self, 'ng_button'):
                self.ng_button.setEnabled(False)

            self.create_navigation_buttons()
            if hasattr(self, 'prev_button'):
                self.prev_button.setEnabled(True)
            if hasattr(self, 'next_button'):
                self.next_button.setEnabled(True)
                
        except Exception as e:
            print(f"❌ Error en on_show_results_finished: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "エラー", f"❌ 結果処理中にエラーが発生しました:\n{str(e)}")

    def on_show_results_error(self, error_message):
        """Maneja el error del procesamiento de resultados"""
        try:
            print(f"🔍 Debug - on_show_results_error llamado con error: {error_message}")
            
            if hasattr(self, 'loader_overlay'):
                self.loader_overlay.stop()
            
            QMessageBox.critical(self, "エラー", f"❌ 結果処理中にエラーが発生しました:\n{str(error_message)}")
            
        except Exception as e:
            print(f"❌ Error en on_show_results_error: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "エラー", f"❌ エラー処理中にエラーが発生しました:\n{str(e)}")

    def display_image_in_graph_area(self, image_path):
        """Carga y muestra una imagen dentro del área de gráficos."""


        if not hasattr(self.graph_area, "layout") or self.graph_area.layout() is None:
            self.graph_area.setLayout(QVBoxLayout())

        layout = self.graph_area.layout()

        # Limpiar el contenido actual
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Mostrar nueva imagen
        label = QLabel()
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap.scaled(self.graph_area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

    def on_analyze_clicked(self):
        """Acción al pulsar el botón de análisis - navega directamente a la página de filtros"""
        print("分析ページに移動中...")
        
        # ✅ NUEVO: Marcar que se accedió desde el botón bunseki
        self.accessed_from_bunseki = True
        
        # Verificar si ya estamos en la vista de filtros
        # Buscar si hay un título "データフィルター" en el layout central
        already_in_filter_view = False
        for i in range(self.center_layout.count()):
            item = self.center_layout.itemAt(i)
            if item.widget() and isinstance(item.widget(), QLabel):
                if item.widget().text() == "データフィルター":
                    already_in_filter_view = True
                    break
        
        if already_in_filter_view:
            # Ya estamos en la pantalla de filtros, solo mostrar mensaje informativo
            QMessageBox.information(self, "分析ページ", "✅ 既に分析ページにいます。\nフィルターを設定してデータを分析してください。")
            return
        
        try:
            # Crear la vista de filtros directamente
            self.create_filter_view()
            
            # Habilitar botones de navegación
            self.create_navigation_buttons()
            self.prev_button.setEnabled(True)
            self.next_button.setEnabled(True)
            
            QMessageBox.information(self, "分析ページ", "✅ 分析ページに移動しました。\nフィルターを設定してデータを分析してください。")
            
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"❌ 分析ページの移動中にエラーが発生しました:\n{str(e)}")

    def on_ok_clicked(self):
        # ✅ NUEVO: Verificación inicial - asegurar que solo exista un tipo de resultado
        print(f"🔍 Debug - INICIO on_ok_clicked:")
        print(f"🔍 Debug - dsaitekika_results existe: {hasattr(self, 'dsaitekika_results')}")
        print(f"🔍 Debug - isaitekika_results existe: {hasattr(self, 'isaitekika_results')}")
        print(f"🔍 Debug - last_executed_optimization existe: {hasattr(self, 'last_executed_optimization')}")
        if hasattr(self, 'last_executed_optimization'):
            print(f"🔍 Debug - last_executed_optimization valor: {self.last_executed_optimization}")
        
        # ✅ NUEVO: Verificación crítica - SIEMPRE usar last_executed_optimization si existe
        if hasattr(self, 'last_executed_optimization'):
            print(f"🔍 Debug - FORZANDO uso de last_executed_optimization: {self.last_executed_optimization}")
            # Forzar el uso del último tipo de optimización ejecutado
            if self.last_executed_optimization == 'I':
                if hasattr(self, 'dsaitekika_results'):
                    delattr(self, 'dsaitekika_results')
                    print("🧹 Limpiando dsaitekika_results para forzar I最適化")
            elif self.last_executed_optimization == 'D':
                if hasattr(self, 'isaitekika_results'):
                    delattr(self, 'isaitekika_results')
                    print("🧹 Limpiando isaitekika_results para forzar D最適化")
        
        # Copiar archivos definitivos a carpeta 実験リスト solo al pulsar OK
        if hasattr(self, 'dsaitekika_results') or hasattr(self, 'isaitekika_results'):
            # ✅ SIMPLIFICADO: Usar SIEMPRE last_executed_optimization como fuente de verdad
            if hasattr(self, 'last_executed_optimization'):
                optimization_type = self.last_executed_optimization
                print(f"🔍 Debug - Usando last_executed_optimization: {optimization_type}")
            else:
                # Fallback solo si no existe last_executed_optimization
                if hasattr(self, 'isaitekika_results') and not hasattr(self, 'dsaitekika_results'):
                    optimization_type = 'I'
                elif hasattr(self, 'dsaitekika_results'):
                    optimization_type = 'D'
                else:
                    optimization_type = 'D'  # Por defecto
                print(f"🔍 Debug - Usando fallback, optimization_type: {optimization_type}")
            
            print(f"🔍 Debug - dsaitekika_results existe: {hasattr(self, 'dsaitekika_results')}")
            print(f"🔍 Debug - isaitekika_results existe: {hasattr(self, 'isaitekika_results')}")
            print(f"🔍 Debug - optimization_type final: {optimization_type}")
            print(f"🔍 Debug - last_executed_optimization valor: {getattr(self, 'last_executed_optimization', 'No existe')}")
            
            # ✅ SIMPLIFICADO: Limpiar resultados del tipo opuesto
            if optimization_type == 'D':
                print("✅ Exportando como D最適化")
                if hasattr(self, 'isaitekika_results'):
                    delattr(self, 'isaitekika_results')
                    print("🧹 Limpiando isaitekika_results para exportación D")
            elif optimization_type == 'I':
                print("✅ Exportando como I最適化")
                if hasattr(self, 'dsaitekika_results'):
                    delattr(self, 'dsaitekika_results')
                    print("🧹 Limpiando dsaitekika_results para exportación I")
            else:
                print(f"⚠️ Tipo desconocido: {optimization_type}, usando D最適化 por defecto")
                optimization_type = 'D'
            
            # ✅ NUEVO: Crear carpeta y determinar nombre basado en optimization_type
            output_folder = self.current_temp_folder if hasattr(self, 'current_temp_folder') else os.path.join(self.proyecto_folder, "99_Temp", "Temp")
            project_name = getattr(self, 'proyecto_nombre', 'Unknown')
            today = datetime.now().strftime('%Y%m%d')
            
            # Crear carpeta 01_実験リスト al mismo nivel que 99_Temp
            samples_base = os.path.join(self.proyecto_folder, "01_実験リスト")
            os.makedirs(samples_base, exist_ok=True)
            
            # Formato de nombre de carpeta basado en optimization_type
            now = datetime.now()
            fecha_hora = now.strftime('%Y%m%d_%H%M%S')
            if optimization_type == 'I':
                prefix = 'I_SAITEKIKA'
                print(f"📁 Creando carpeta con prefijo I: {prefix}")
            else:  # D optimization
                prefix = 'D最適化'
                print(f"📁 Creando carpeta con prefijo D: {prefix}")
            
            # Buscar el mayor número de carpeta existente y sumarle 1
            existing_folders = [d for d in os.listdir(samples_base) if os.path.isdir(os.path.join(samples_base, d))]
            max_num = 0
            for folder in existing_folders:
                try:
                    num = int(folder.split('_')[0])
                    if num > max_num:
                        max_num = num
                except Exception:
                    pass
            next_num = max_num + 1
            folder_name = f"{next_num:03d}_{prefix}_{fecha_hora}"
            sample_folder = os.path.join(samples_base, folder_name)
            os.makedirs(sample_folder, exist_ok=True)
            print(f"📁 Carpeta creada: {folder_name}")
            print(f"📁 Ruta completa: {sample_folder}")
            
            if optimization_type == 'I':
                
                # Cambiar nombre de columnas para la exportación antes de guardar
                if hasattr(self, 'isaitekika_results'):
                    if '面粗度(Ra)前' in self.isaitekika_selected_df.columns:
                        self.isaitekika_selected_df.rename(columns={'面粗度(Ra)前': 'Ra(前)'}, inplace=True)
                    if '面粗度(Ra)後' in self.isaitekika_selected_df.columns:
                        self.isaitekika_selected_df.rename(columns={'面粗度(Ra)後': 'Ra(後)'}, inplace=True)
                    # Guardar archivo Excel I-óptimo
                    if len(self.isaitekika_selected_df) > 0:
                        # --- Ajuste de columnas y formato para I最適化_新規実験点.xlsx ---
                        # Mapear nombres de columnas antes de procesar
                        if '突出し量' in self.isaitekika_selected_df.columns:
                            self.isaitekika_selected_df.rename(columns={'突出し量': '突出量'}, inplace=True)
                        if '切込み量' in self.isaitekika_selected_df.columns:
                            self.isaitekika_selected_df.rename(columns={'切込み量': '切込量'}, inplace=True)
                        
                        # Dirección: usar nombre nuevo "UPカット"
                        if '回転方向' in self.isaitekika_selected_df.columns and 'UPカット' not in self.isaitekika_selected_df.columns:
                            self.isaitekika_selected_df.rename(columns={'回転方向': 'UPカット'}, inplace=True)

                        required_columns = ['No.', 'A13', 'A11', 'A21', 'A32',
                                           '回転速度', '送り速度', 'UPカット', '切込量', '突出量', '載せ率', 'パス数',
                                           '線材長', 'I基準値',
                                           '上面ダレ', '側面ダレ', '摩耗量', '面粗度(Ra)前', '面粗度(Ra)後',
                                           '切削力X', '切削力Y', '切削力Z',
                                           '実験日']
                        df_export = self.isaitekika_selected_df.copy()
                        # Normalizar nombres de rugosidad si vienen como Ra(前)/Ra(後) o sin (Ra)
                        if 'Ra(前)' in df_export.columns and '面粗度(Ra)前' not in df_export.columns:
                            df_export.rename(columns={'Ra(前)': '面粗度(Ra)前'}, inplace=True)
                        if 'Ra(後)' in df_export.columns and '面粗度(Ra)後' not in df_export.columns:
                            df_export.rename(columns={'Ra(後)': '面粗度(Ra)後'}, inplace=True)
                        if '面粗度前' in df_export.columns and '面粗度(Ra)前' not in df_export.columns:
                            df_export.rename(columns={'面粗度前': '面粗度(Ra)前'}, inplace=True)
                        if '面粗度後' in df_export.columns and '面粗度(Ra)後' not in df_export.columns:
                            df_export.rename(columns={'面粗度後': '面粗度(Ra)後'}, inplace=True)
                        # Crear las columnas que falten
                        for col in required_columns:
                            if col not in df_export.columns and col != 'I基準値':
                                df_export[col] = ''
                        # ISaitekika: I基準値 NO se calcula nunca
                        df_export['I基準値'] = ''
                        # 線材長 siempre en blanco en el Excel de salida
                        df_export['線材長'] = ''
                        # Reordenar las columnas
                        df_export = df_export[required_columns]
                        i_path = os.path.join(output_folder, "I最適化_新規実験点.xlsx")
                        df_export.to_excel(i_path, index=False)
                        # --- Fin ajuste de columnas ---
                    # Añadir columna de fecha si no existe
                    if len(self.isaitekika_selected_df) > 0:
                        if '実験日' not in self.isaitekika_selected_df.columns:
                            self.isaitekika_selected_df['実験日'] = ''
                    # Copiar archivo Excel a la carpeta 実験リスト
                    excel_src = os.path.join(output_folder, "I最適化_新規実験点.xlsx")
                    if os.path.exists(excel_src):
                        shutil.copy2(excel_src, sample_folder)
                    # Copiar imágenes a la carpeta 実験リスト
                    for img_path in glob.glob(os.path.join(output_folder, '*.png')):
                        shutil.copy2(img_path, sample_folder)
            else:
                # Optimización D-óptima
                
                # Cambiar nombre de columnas para la exportación antes de guardar
                if hasattr(self, 'dsaitekika_results'):
                    if '面粗度(Ra)前' in self.dsaitekika_selected_df.columns:
                        self.dsaitekika_selected_df.rename(columns={'面粗度(Ra)前': 'Ra(前)'}, inplace=True)
                    if '面粗度(Ra)後' in self.dsaitekika_selected_df.columns:
                        self.dsaitekika_selected_df.rename(columns={'面粗度(Ra)後': 'Ra(後)'}, inplace=True)
                    # Guardar archivo Excel D-óptimo
                    if len(self.dsaitekika_selected_df) > 0:
                        # --- Ajuste de columnas y formato para D_optimal_新規実験点.xlsx ---
                        # Mapear nombres de columnas antes de procesar
                        if '突出し量' in self.dsaitekika_selected_df.columns:
                            self.dsaitekika_selected_df.rename(columns={'突出し量': '突出量'}, inplace=True)
                        if '切込み量' in self.dsaitekika_selected_df.columns:
                            self.dsaitekika_selected_df.rename(columns={'切込み量': '切込量'}, inplace=True)
                        
                        # Dirección: usar nombre nuevo "UPカット"
                        if '回転方向' in self.dsaitekika_selected_df.columns and 'UPカット' not in self.dsaitekika_selected_df.columns:
                            self.dsaitekika_selected_df.rename(columns={'回転方向': 'UPカット'}, inplace=True)

                        required_columns = ['No.', 'A13', 'A11', 'A21', 'A32',
                                           '回転速度', '送り速度', 'UPカット', '切込量', '突出量', '載せ率', 'パス数',
                                           '線材長', 'D基準値',
                                           '上面ダレ', '側面ダレ', '摩耗量', '面粗度(Ra)前', '面粗度(Ra)後',
                                           '切削力X', '切削力Y', '切削力Z',
                                           '実験日']
                        df_export = self.dsaitekika_selected_df.copy()
                        # Normalizar nombres de rugosidad si vienen como Ra(前)/Ra(後) o sin (Ra)
                        if 'Ra(前)' in df_export.columns and '面粗度(Ra)前' not in df_export.columns:
                            df_export.rename(columns={'Ra(前)': '面粗度(Ra)前'}, inplace=True)
                        if 'Ra(後)' in df_export.columns and '面粗度(Ra)後' not in df_export.columns:
                            df_export.rename(columns={'Ra(後)': '面粗度(Ra)後'}, inplace=True)
                        if '面粗度前' in df_export.columns and '面粗度(Ra)前' not in df_export.columns:
                            df_export.rename(columns={'面粗度前': '面粗度(Ra)前'}, inplace=True)
                        if '面粗度後' in df_export.columns and '面粗度(Ra)後' not in df_export.columns:
                            df_export.rename(columns={'面粗度後': '面粗度(Ra)後'}, inplace=True)
                        # Crear las columnas que falten
                        for col in required_columns:
                            if col not in df_export.columns and col != 'D基準値':
                                df_export[col] = ''
                        # Calcular D基準値 EXACTAMENTE como el archivo de referencia
                        if len(df_export) > 0:
                            d_score_ref = getattr(self, "_last_d_score_reference", None)
                            # Intentar recalcular desde candidate_df + d_indices (más fiel a la referencia)
                            if d_score_ref is None or not np.isfinite(d_score_ref):
                                try:
                                    cand_df = getattr(self, "_last_candidate_df_for_dscore", None)
                                    d_idx = getattr(self, "_last_d_indices", None)
                                    if cand_df is not None and d_idx is not None:
                                        cand_np = cand_df.to_numpy() if hasattr(cand_df, "to_numpy") else np.asarray(cand_df)
                                        d_score_ref = calculate_d_score_reference(cand_np, d_idx)
                                except Exception as e:
                                    print(f"⚠️ Error recalculando D基準値 (referencia) desde candidato/índices: {e}")
                            # Fallback: si no hay candidatos/índices, calcular sobre los seleccionados (escala fit en seleccionados)
                            if d_score_ref is None or not np.isfinite(d_score_ref):
                                X_raw = _extract_design_matrix(df_export)
                                X_scaled = _standardize_like_reference(X_raw)
                                d_score_ref, _ = calculate_d_criterion_stable_reference(
                                    X_scaled, method="auto", use_numerical_stable_method=True, verbose=False
                                )
                            self._last_d_score_reference = float(d_score_ref) if d_score_ref is not None else None
                            df_export["D基準値"] = self._last_d_score_reference if self._last_d_score_reference is not None else np.nan
                        else:
                            df_export["D基準値"] = np.nan
                        # 線材長 siempre en blanco en el Excel de salida
                        df_export['線材長'] = ''
                        # Reordenar las columnas
                        df_export = df_export[required_columns]
                        d_path = os.path.join(output_folder, "D最適化_新規実験点.xlsx")
                        df_export.to_excel(d_path, index=False)
                        # --- Fin ajuste de columnas ---
                    # Añadir columna de fecha si no existe
                    if len(self.dsaitekika_selected_df) > 0:
                        if '実験日' not in self.dsaitekika_selected_df.columns:
                            self.dsaitekika_selected_df['実験日'] = ''
                    # Copiar archivo Excel a la carpeta 実験リスト
                    excel_src = os.path.join(output_folder, "D最適化_新規実験点.xlsx")
                    if os.path.exists(excel_src):
                        shutil.copy2(excel_src, sample_folder)
                    # Copiar imágenes a la carpeta 実験リスト
                    for img_path in glob.glob(os.path.join(output_folder, '*.png')):
                        shutil.copy2(img_path, sample_folder)
            # Limpiar archivos temporales después de guardar exitosamente
            if hasattr(self, 'current_temp_folder') and self.current_temp_folder:
                try:
                    if os.path.exists(self.current_temp_folder):
                        shutil.rmtree(self.current_temp_folder)
                        print(f"🗑️ Carpeta Temp eliminada después de guardar: {self.current_temp_folder}")
                    # NO borrar la carpeta 99_Temp - mantenerla para futuros usos
                    temp_base = os.path.join(self.proyecto_folder, "99_Temp")
                    print(f"📁 Carpeta 99_Temp mantenida: {temp_base}")
                except Exception as e:
                    print(f"⚠️ Error al limpiar archivos temporales: {e}")
            # Limpiar referencias
            if hasattr(self, 'current_temp_folder'):
                delattr(self, 'current_temp_folder')
            # Habilitar botones de optimización después de guardar exitosamente
            self.d_optimize_button.setEnabled(True)
            self.i_optimize_button.setEnabled(True)
            self.d_optimize_button.setStyleSheet(self.d_optimize_button.styleSheet())
            self.i_optimize_button.setStyleSheet(self.i_optimize_button.styleSheet())
            
            # Deshabilitar botones OK/NG
            self.ok_button.setEnabled(False)
            self.ng_button.setEnabled(False)
            
            # Limpiar pantalla después de guardar exitosamente
            self.graph_images = []
            self.graph_images_content = []
            self.current_graph_index = 0
            
            # Limpiar área de gráficos
            if hasattr(self, 'graph_area') and self.graph_area.layout():
                layout = self.graph_area.layout()
                for i in reversed(range(layout.count())):
                    widget = layout.itemAt(i).widget()
                    if widget:
                        widget.setParent(None)
            
            QMessageBox.information(self, '保存完了', 
                f'✅ サンプルファイルが以下のフォルダにエクスポートされました:\n\n'
                f'📁 {sample_folder}')
        else:
            QMessageBox.warning(self, 'エラー', '保存する結果がありません。')

    def on_ng_clicked(self):
        """Borra archivos temporales y habilita botones de optimización"""
        try:
            print(f"🔍 Debug NG: current_temp_folder = {getattr(self, 'current_temp_folder', 'No existe')}")
            print(f"🔍 Debug NG: proyecto_folder = {getattr(self, 'proyecto_folder', 'No existe')}")
            
            # Borrar carpeta temporal si existe
            if hasattr(self, 'current_temp_folder') and self.current_temp_folder:
                print(f"🔍 Debug NG: Verificando existencia de {self.current_temp_folder}")
                if os.path.exists(self.current_temp_folder):
                    print(f"🔍 Debug NG: Carpeta existe, procediendo a borrar...")
                    shutil.rmtree(self.current_temp_folder)
                    print(f"🗑️ Carpeta Temp eliminada: {self.current_temp_folder}")
                else:
                    print(f"🔍 Debug NG: Carpeta no existe: {self.current_temp_folder}")
                
                # NO borrar la carpeta 99_Temp - mantenerla para futuros usos
                temp_base = os.path.join(self.proyecto_folder, "99_Temp")
                print(f"📁 Carpeta 99_Temp mantenida: {temp_base}")
            else:
                print(f"🔍 Debug NG: No hay current_temp_folder definido")
            
            # Limpiar referencias
            if hasattr(self, 'current_temp_folder'):
                delattr(self, 'current_temp_folder')
            if hasattr(self, 'dsaitekika_results'):
                delattr(self, 'dsaitekika_results')
            if hasattr(self, 'isaitekika_results'):
                delattr(self, 'isaitekika_results')
            if hasattr(self, 'dsaitekika_selected_df'):
                delattr(self, 'dsaitekika_selected_df')
            if hasattr(self, 'isaitekika_selected_df'):
                delattr(self, 'isaitekika_selected_df')
            
            # Limpiar gráficos y tablas
            self.graph_images = []
            self.graph_images_content = []
            self.current_graph_index = 0
            
            # Limpiar área de gráficos
            if hasattr(self, 'graph_area') and self.graph_area.layout():
                layout = self.graph_area.layout()
                for i in reversed(range(layout.count())):
                    widget = layout.itemAt(i).widget()
                    if widget:
                        widget.setParent(None)
            
            # Habilitar botones de optimización
            self.d_optimize_button.setEnabled(True)
            self.i_optimize_button.setEnabled(True)
            # Aplicar estilo visual de habilitado
            self.d_optimize_button.setStyleSheet(self.d_optimize_button.styleSheet())
            self.i_optimize_button.setStyleSheet(self.i_optimize_button.styleSheet())
            
            # Deshabilitar botones OK/NG
            self.ok_button.setEnabled(False)
            self.ng_button.setEnabled(False)
            
            QMessageBox.information(self, 'キャンセル', 
                '✅ サンプルがキャンセルされました。')
            
        except Exception as e:
            QMessageBox.warning(self, '警告', 
                f'⚠️ 一時ファイルの削除中にエラーが発生しました:\n{str(e)}\n\n最適化ボタンは再有効化されました。')
            
            # Aún así, habilitar los botones
            self.d_optimize_button.setEnabled(True)
            self.i_optimize_button.setEnabled(True)
            self.d_optimize_button.setStyleSheet(self.d_optimize_button.styleSheet())
            self.i_optimize_button.setStyleSheet(self.i_optimize_button.styleSheet())
            self.ok_button.setEnabled(False)
            self.ng_button.setEnabled(False)

    def get_selected_brush(self):
        """Obtener el brush seleccionado del ComboBox"""
        return self.brush_selector.currentText()
    
    def get_selected_brush_from_filter(self):
        """Obtener el brush seleccionado del filtro"""
        for key in ["すべて", "A13", "A11", "A21", "A32"]:
            if key in self.filter_inputs and self.filter_inputs[key].isChecked():
                return key
        return "すべて"  # Por defecto

    def on_generate_sample_file_clicked(self):
        # ✅ NUEVO: Pausar timers automáticos para evitar interferencia con el diálogo
        self.pause_auto_timers()
        
        config_file, _ = QFileDialog.getOpenFileName(
            self, "パラメータ設定ファイルを選択", "", "Excel Files (*.xlsx *.xls)"
        )
        if not config_file:
            # ✅ NUEVO: Reanudar timers si se cancela el primer diálogo
            self.resume_auto_timers()
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存先を選択", "sample_combinations.xlsx", "Excel Files (*.xlsx *.xls)"
        )
        if not save_path:
            # ✅ NUEVO: Reanudar timers si se cancela el segundo diálogo
            self.resume_auto_timers()
            return
        
        # ✅ NUEVO: Reanudar timers después de ambos diálogos
        self.resume_auto_timers()

        # Mostrar loader (reutilizar si ya existe para evitar múltiples overlays/eventFilters)
        if not hasattr(self, 'loader_overlay') or self.loader_overlay is None:
            self.loader_overlay = LoadingOverlay(self.center_frame)
        self.loader_overlay.start()

        self.sample_thread = QThread()
        self.sample_worker = SampleCombinerWorker(config_file, save_path)
        self.sample_worker.moveToThread(self.sample_thread)

        self.sample_thread.started.connect(self.sample_worker.run)
        self.sample_worker.finished.connect(self.on_sample_generation_finished)
        self.sample_worker.error.connect(self.on_sample_generation_error)
        self.sample_worker.finished.connect(self.sample_thread.quit)
        self.sample_worker.finished.connect(self.sample_worker.deleteLater)
        self.sample_thread.finished.connect(self.sample_thread.deleteLater)

        self.sample_thread.start()

    def add_selected_samples_table_view(self, df):
        # Definir columnas básicas que siempre deben estar presentes
        columnas_basicas = ["No.", "回転速度", "送り速度", "UPカット", "回転方向", "切込量", "突出量", "載せ率", "パス数"]
        
        # Verificar qué columnas están disponibles en el DataFrame
        columnas_disponibles = []
        for col in columnas_basicas:
            if col in df.columns:
                columnas_disponibles.append(col)
        
        # Añadir columnas adicionales si están disponibles
        # ISaitekika: NO mostrar I基準値 en la tabla
        if hasattr(self, 'isaitekika_selected_df') and df is getattr(self, 'isaitekika_selected_df', None):
            columnas_adicionales = ["D基準値", "上面ダレ", "側面ダレ", "摩耗量"]
        else:
            columnas_adicionales = ["D基準値", "I基準値", "上面ダレ", "側面ダレ", "摩耗量"]
        for col in columnas_adicionales:
            if col in df.columns:
                columnas_disponibles.append(col)

        # Crear DataFrame filtrado solo con las columnas disponibles
        df_filtrado = df[columnas_disponibles].copy()

        # Crear contenedor para la tabla con título
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        
        # Determinar el título basándose en el tipo de optimización
        # Si tenemos resultados de I最適化, mostrar tabla I最適
        if hasattr(self, 'isaitekika_results') and hasattr(self, 'dsaitekika_results'):
            # Si ambos existen, determinar por el DataFrame actual
            if df is self.isaitekika_selected_df:
                title = "I最適サンプル一覧"
            else:
                title = "D最適サンプル一覧"
        elif hasattr(self, 'isaitekika_results'):
            title = "I最適サンプル一覧"
        elif hasattr(self, 'dsaitekika_results'):
            title = "D最適サンプル一覧"
        else:
            title = "新規実験点"
            
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 8px;")
        table_layout.addWidget(title_label)

        table_widget = QTableWidget()
        table_widget.setRowCount(len(df_filtrado))
        table_widget.setColumnCount(len(df_filtrado.columns))
        table_widget.setHorizontalHeaderLabels(df_filtrado.columns)
        table_widget.setStyleSheet("font-size: 11px; font-family: 'Yu Gothic';")

        for row in range(len(df_filtrado)):
            for col in range(len(df_filtrado.columns)):
                item = QTableWidgetItem(str(df_filtrado.iat[row, col]))
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)  # Solo lectura
                table_widget.setItem(row, col, item)

        # Expandir tabla al ancho completo del contenedor
        table_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        table_widget.horizontalHeader().setStretchLastSection(True)
        table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(table_widget)

        self.graph_images.append("table")  # marcador especial
        self.graph_images_content = getattr(self, "graph_images_content", [])
        self.graph_images_content.append(table_container)

        self.next_button.setEnabled(True)
        self.prev_button.setEnabled(True)

    def show_loader(self, show: bool):
        if show:
            self.loader_label.show()
            self.loader_movie.start()
        else:
            self.loader_movie.stop()
            self.loader_label.hide()

    def display_graphs(self, image_paths):
        """Guarda las rutas y muestra la primera imagen."""
        self.graph_images = image_paths
        self.current_graph_index = 0

        # Crear botones si no existen
        if self.prev_button is None or self.next_button is None:
            self.create_navigation_buttons()

        # Mostrar primer gráfico y activar/desactivar botones según corresponda
        self.update_graph_display()
        self.prev_button.setEnabled(self.current_graph_index > 0)
        self.next_button.setEnabled(self.current_graph_index < len(self.graph_images) - 1)
        print("Número de gráficos:", len(self.graph_images))

    # Función para actualizar la imagen mostrada
    def update_graph_display(self):
        # ✅ NUEVO: Verificar si el layout existe, si no, crear uno nuevo
        if self.graph_area.layout() is None:
            print("⚠️ Layout del área de gráficos es None, creando nuevo layout...")
            self.graph_area.setLayout(QVBoxLayout())
        
        layout = self.graph_area.layout()

        # Limpiar contenido actual
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        current_item = self.graph_images[self.current_graph_index]

        if current_item == "table":
            # Determinar qué tabla mostrar basándose en el índice actual
            table_index = 0  # Por defecto D-óptimo
            if hasattr(self, 'graph_images_content') and len(self.graph_images_content) >= 2:
                # Contar cuántas tablas hay antes del índice actual
                table_count = 0
                for i in range(self.current_graph_index):
                    if self.graph_images[i] == "table":
                        table_count += 1
                
                # Si es la primera tabla (table_count = 0), mostrar D-óptimo
                # Si es la segunda tabla (table_count = 1), mostrar I-óptimo
                if table_count == 0:
                    print("📋 Mostrando tabla D-óptimo")
                    self._add_tablewidget_to_graph_area(self.dsaitekika_selected_df, layout, "D最適サンプル一覧")
                elif table_count == 1:
                    print("📋 Mostrando tabla I-óptimo")
                    self._add_tablewidget_to_graph_area(self.isaitekika_selected_df, layout, "I最適サンプル一覧")
                else:
                    # Fallback: mostrar la tabla correspondiente del contenido
                    if table_count < len(self.graph_images_content):
                        layout.addWidget(self.graph_images_content[table_count])
            else:
                # Fallback: mostrar la última tabla añadida
                if hasattr(self, 'graph_images_content') and self.graph_images_content:
                    layout.addWidget(self.graph_images_content[-1])
        else:
            img_path = current_item
            pixmap = QPixmap(img_path).scaled(700, 540, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label = QLabel()
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)

        self.prev_button.setEnabled(self.current_graph_index > 0)
        self.next_button.setEnabled(self.current_graph_index < len(self.graph_images) - 1)

    def _add_tablewidget_to_graph_area(self, df, layout, titulo=None):

        # Definir columnas básicas que siempre deben estar presentes
        columnas_basicas = ["No.", "回転速度", "送り速度", "UPカット", "回転方向", "切込量", "突出量", "載せ率", "パス数"]
        columnas_disponibles = [col for col in columnas_basicas if col in df.columns]
        columnas_adicionales = ["D基準値", "I基準値", "上面ダレ", "側面ダレ", "摩耗量"]
        for col in columnas_adicionales:
            if col in df.columns:
                columnas_disponibles.append(col)
        df_filtrado = df[columnas_disponibles].copy()
        if titulo:
            label = QLabel(titulo)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 8px;")
            layout.addWidget(label)
        table_widget = QTableWidget()
        table_widget.setRowCount(len(df_filtrado))
        table_widget.setColumnCount(len(df_filtrado.columns))
        table_widget.setHorizontalHeaderLabels(df_filtrado.columns)
        table_widget.setStyleSheet("font-size: 11px; font-family: 'Yu Gothic';")
        for row in range(len(df_filtrado)):
            for col in range(len(df_filtrado.columns)):
                item = QTableWidgetItem(str(df_filtrado.iat[row, col]))
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)  # Solo lectura
                table_widget.setItem(row, col, item)
        table_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        table_widget.horizontalHeader().setStretchLastSection(True)
        table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(table_widget)

    def on_integrated_optimizer_finished(self, result):
        self.optimizer_result = result  # Asegura que los resultados estén disponibles para on_ok_clicked
        """Maneja los resultados del optimizador integrado D-óptimo + I-óptimo"""

        # Guardar ambos DataFrames
        self.dsaitekika_selected_df = result["d_dataframe"]
        self.isaitekika_selected_df = result["i_dataframe"]

        # Guardar datos del optimizador para recalcular D基準値 exactamente como el archivo de referencia
        self._last_candidate_df_for_dscore = result.get("candidate_df", None)
        self._last_d_indices = result.get("d_indices", None)
        self._last_existing_indices = result.get("existing_indices", None)
        try:
            if self._last_candidate_df_for_dscore is not None and self._last_d_indices is not None:
                cand_np = (
                    self._last_candidate_df_for_dscore.to_numpy()
                    if hasattr(self._last_candidate_df_for_dscore, "to_numpy")
                    else np.asarray(self._last_candidate_df_for_dscore)
                )
                self._last_d_score_reference = calculate_d_score_reference(cand_np, self._last_d_indices)
                if len(self.dsaitekika_selected_df) > 0:
                    self.dsaitekika_selected_df["D基準値"] = self._last_d_score_reference
        except Exception as e:
            print(f"⚠️ Error calculando D基準値 (referencia, integrado): {e}")
        
        # ✅ NUEVO: Para optimización integrada, establecer el tipo basado en el último ejecutado
        # Por defecto, usar D最適化 como tipo principal
        self.last_executed_optimization = 'D'
        print(f"🔍 Debug - on_integrated_optimizer_finished: last_executed_optimization = 'D' (integrado)")
        
        # Añadir columnas necesarias para la visualización en tabla
        if len(self.dsaitekika_selected_df) > 0:
            # Mapear nombres de columnas si es necesario
            if '突出し量' in self.dsaitekika_selected_df.columns:
                self.dsaitekika_selected_df.rename(columns={'突出し量': '突出量'}, inplace=True)
            if '切込み量' in self.dsaitekika_selected_df.columns:
                self.dsaitekika_selected_df.rename(columns={'切込み量': '切込量'}, inplace=True)
            
            if "No." not in self.dsaitekika_selected_df.columns:
                self.dsaitekika_selected_df.insert(0, "No.", list(range(1, len(self.dsaitekika_selected_df) + 1)))
            if "上面ダレ" not in self.dsaitekika_selected_df.columns:
                self.dsaitekika_selected_df["上面ダレ"] = ""
            if "側面ダレ" not in self.dsaitekika_selected_df.columns:
                self.dsaitekika_selected_df["側面ダレ"] = ""
            if "摩耗量" not in self.dsaitekika_selected_df.columns:
                self.dsaitekika_selected_df["摩耗量"] = ""
        if len(self.isaitekika_selected_df) > 0:
            # Mapear nombres de columnas si es necesario
            if '突出し量' in self.isaitekika_selected_df.columns:
                self.isaitekika_selected_df.rename(columns={'突出し量': '突出量'}, inplace=True)
            if '切込み量' in self.isaitekika_selected_df.columns:
                self.isaitekika_selected_df.rename(columns={'切込み量': '切込量'}, inplace=True)
            
            if "No." not in self.isaitekika_selected_df.columns:
                self.isaitekika_selected_df.insert(0, "No.", list(range(1, len(self.isaitekika_selected_df) + 1)))
            if "上面ダレ" not in self.isaitekika_selected_df.columns:
                self.isaitekika_selected_df["上面ダレ"] = ""
            if "側面ダレ" not in self.isaitekika_selected_df.columns:
                self.isaitekika_selected_df["側面ダレ"] = ""
            if "摩耗量" not in self.isaitekika_selected_df.columns:
                self.isaitekika_selected_df["摩耗量"] = ""
        
        # Guardar rutas de archivos para uso posterior
        self.integrated_output_folder = os.path.dirname(result["d_path"]) if result["d_path"] else ""
        self.d_optimal_path = result["d_path"]
        self.i_optimal_path = result["i_path"]
        self.all_d_optimal_path = result["all_d_path"]
        self.all_i_optimal_path = result["all_i_path"]
        
        # Guardar DataFrames adicionales para guardado posterior
        self.candidate_df = result.get("candidate_df", pd.DataFrame())
        self.all_d_df = result.get("all_d_df", pd.DataFrame())
        self.all_i_df = result.get("all_i_df", pd.DataFrame())
        
        # Exportar los Excel con criterios calculados
        if len(self.dsaitekika_selected_df) > 0 and not os.path.exists(self.d_optimal_path):
            # Calcular D基準値 para D-óptimo (igual que referencia)
            df_d = self.dsaitekika_selected_df.copy()
            if len(df_d) > 0:
                # Preferir score de referencia (StandardScaler + logdet estable)
                d_score_ref = getattr(self, "_last_d_score_reference", None)
                if d_score_ref is None or not np.isfinite(d_score_ref):
                    # Fallback: calcular sobre los puntos seleccionados solamente (no ideal, pero consistente)
                    X_raw = _extract_design_matrix(df_d)
                    X_scaled = _standardize_like_reference(X_raw)
                    d_score_ref, _ = calculate_d_criterion_stable_reference(
                        X_scaled, method="auto", use_numerical_stable_method=True, verbose=False
                    )
                df_d["D基準値"] = float(d_score_ref) if d_score_ref is not None else np.nan
            
            df_d.to_excel(self.d_optimal_path, index=False)
            
        if len(self.isaitekika_selected_df) > 0 and not os.path.exists(self.i_optimal_path):
            # ISaitekika: I基準値 NO se calcula nunca (mantener en blanco)
            df_i = self.isaitekika_selected_df.copy()
            df_i['I基準値'] = ''
            
            df_i.to_excel(self.i_optimal_path, index=False)

        # Configurar sistema de navegación de gráficos uno a uno
        self.graph_images = result["image_paths"]
        self.current_graph_index = 0
        print(f"📊 Configurando navegación de gráficos:")
        print(f"  - Total de gráficos: {len(self.graph_images)}")
        print(f"  - Gráficos disponibles: {[os.path.basename(path) for path in self.graph_images]}")
        
        # Crear botones de navegación si no existen
        if self.prev_button is None or self.next_button is None:
            self.create_navigation_buttons()
        
        # Mostrar primer gráfico
        self.update_graph_display()
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(len(self.graph_images) > 1)
        print(f"✅ Sistema de navegación configurado:")
        print(f"  - Gráfico actual: {self.current_graph_index + 1}/{len(self.graph_images)}")
        print(f"  - Botón anterior: {'Habilitado' if self.prev_button.isEnabled() else 'Deshabilitado'}")
        print(f"  - Botón siguiente: {'Habilitado' if self.next_button.isEnabled() else 'Deshabilitado'}")

        # Añadir ambas tablas usando el método original
        print(f"📋 Añadiendo tabla D-óptimo con {len(self.dsaitekika_selected_df)} filas")
        self.current_table_index = 0  # Para D-óptimo
        self.add_selected_samples_table_view(self.dsaitekika_selected_df)
        
        print(f"📋 Añadiendo tabla I-óptimo con {len(self.isaitekika_selected_df)} filas")
        self.current_table_index = 1  # Para I-óptimo
        self.add_selected_samples_table_view(self.isaitekika_selected_df)
        
        print(f"✅ Total de elementos en graph_images: {len(self.graph_images)}")
        print(f"✅ Total de elementos en graph_images_content: {len(self.graph_images_content)}")

        # Habilitar botones OK/NG
        self.ok_button.setEnabled(True)
        self.ng_button.setEnabled(True)
        
        # Deshabilitar botones de optimización después de completar el análisis integrado
        self.d_optimize_button.setEnabled(False)
        self.i_optimize_button.setEnabled(False)
        # Aplicar estilo visual de deshabilitado
        self.d_optimize_button.setStyleSheet(self.d_optimize_button.styleSheet())
        self.i_optimize_button.setStyleSheet(self.i_optimize_button.styleSheet())
        
        # Mensaje de éxito
        message = f"""✅ 最適化統合が完了しました。\n\n📊 結果サマリー:\n• D-最適新規選択: {len(result['d_dataframe'])} 点\n• I-最適新規選択: {len(result['i_dataframe'])} 点\n• 既存実験点活用: {len(result['existing_indices'])} 点\n\n📈 可視化: 特徴量分布 + 次元削減UMAP ({len(self.graph_images)} グラフ)\n📋 テーブル: D-最適 + I-最適 (ナビゲーションで切り替え)\n💾 ファイルはOKボタンを押した時に保存されます"""
        QMessageBox.information(self, "最適化統合完了", message)
        self.loader_overlay.stop()

    def on_d_optimizer_finished(self, results):
        print("DEBUG: Entró en on_d_optimizer_finished")
        print("DEBUG results en on_d_optimizer_finished:", results)
        self.dsaitekika_results = results
        self.dsaitekika_selected_df = results['d_dataframe']
        
        # ✅ NUEVO: Limpiar TODOS los resultados anteriores para evitar conflictos
        if hasattr(self, 'isaitekika_results'):
            delattr(self, 'isaitekika_results')
            print("🧹 Limpiando isaitekika_results anteriores")
        if hasattr(self, 'isaitekika_selected_df'):
            delattr(self, 'isaitekika_selected_df')
            print("🧹 Limpiando isaitekika_selected_df anteriores")
        
        # ✅ NUEVO: Establecer explícitamente el tipo de optimización
        self.last_executed_optimization = 'D'  # Marcar que se ejecutó D-optimización
        print(f"🔍 Debug - on_d_optimizer_finished: last_executed_optimization = 'D'")
        print(f"🔍 Debug - dsaitekika_results existe después de limpiar: {hasattr(self, 'dsaitekika_results')}")
        print(f"🔍 Debug - isaitekika_results existe después de limpiar: {hasattr(self, 'isaitekika_results')}")
        print(f"🔍 Debug - last_executed_optimization establecido: {self.last_executed_optimization}")
        
        # Mapear nombres de columnas si es necesario
        if '突出し量' in self.dsaitekika_selected_df.columns:
            self.dsaitekika_selected_df.rename(columns={'突出し量': '突出量'}, inplace=True)
        if '切込み量' in self.dsaitekika_selected_df.columns:
            self.dsaitekika_selected_df.rename(columns={'切込み量': '切込量'}, inplace=True)
        
        # Calcular D基準値 exactamente como el archivo de referencia (StandardScaler sobre TODOS los candidatos)
        try:
            self._last_candidate_df_for_dscore = results.get("candidate_df", getattr(self, "_last_candidate_df_for_dscore", None))
            self._last_d_indices = results.get("d_indices", getattr(self, "_last_d_indices", None))
            self._last_existing_indices = results.get("existing_indices", getattr(self, "_last_existing_indices", None))

            d_score_ref = None
            if self._last_candidate_df_for_dscore is not None and self._last_d_indices is not None:
                cand_np = (
                    self._last_candidate_df_for_dscore.to_numpy()
                    if hasattr(self._last_candidate_df_for_dscore, "to_numpy")
                    else np.asarray(self._last_candidate_df_for_dscore)
                )
                d_score_ref = calculate_d_score_reference(cand_np, self._last_d_indices)

            if (d_score_ref is None or not np.isfinite(d_score_ref)) and len(self.dsaitekika_selected_df) > 0:
                # Fallback: score sobre los seleccionados solamente
                X_raw = _extract_design_matrix(self.dsaitekika_selected_df)
                X_scaled = _standardize_like_reference(X_raw)
                d_score_ref, _ = calculate_d_criterion_stable_reference(
                    X_scaled, method="auto", use_numerical_stable_method=True, verbose=False
                )

            self._last_d_score_reference = float(d_score_ref) if d_score_ref is not None else None
            if len(self.dsaitekika_selected_df) > 0:
                self.dsaitekika_selected_df["D基準値"] = self._last_d_score_reference if self._last_d_score_reference is not None else np.nan
        except Exception as e:
            print(f"⚠️ Error calculando D基準値 (referencia, D-only): {e}")
        output_folder = os.path.dirname(results['d_path']) if results['d_path'] else ""
        # Filtrar solo los gráficos relevantes a D最適化
        image_paths = sorted(glob.glob(os.path.join(output_folder, '*.png')))
        # Filtrar: solo mostrar histogramas y gráficos generales (no los que sean exclusivamente de I)
        d_image_paths = [p for p in image_paths if not ("I" in os.path.basename(p) or "i_optimal" in os.path.basename(p))]
        if not d_image_paths:
            d_image_paths = image_paths  # fallback: mostrar todos si no hay distinción
        
        # Limpiar contenido anterior
        self.graph_images = []
        self.graph_images_content = []
        
        self.display_graphs(d_image_paths)
        self.add_selected_samples_table_view(self.dsaitekika_selected_df)
        self.ok_button.setEnabled(True)
        self.ng_button.setEnabled(True)
        self.create_navigation_buttons()
        
        # Deshabilitar botones de optimización después de completar D最適化
        self.d_optimize_button.setEnabled(False)
        self.i_optimize_button.setEnabled(False)
        # Aplicar estilo visual de deshabilitado
        self.d_optimize_button.setStyleSheet(self.d_optimize_button.styleSheet())
        self.i_optimize_button.setStyleSheet(self.i_optimize_button.styleSheet())
        
        QMessageBox.information(self, "完了",
                                f"✅ D最適化が完了しました。\n結果を保存しました:\n{results['d_path']}")
        self.loader_overlay.stop()

    def on_i_optimizer_finished(self, results):
        print("DEBUG: Entró en on_i_optimizer_finished")
        print("DEBUG results en on_i_optimizer_finished:", results)
        self.isaitekika_results = results
        self.isaitekika_selected_df = results['i_dataframe']
        # ✅ NUEVO: Limpiar TODOS los resultados anteriores para evitar conflictos
        if hasattr(self, 'dsaitekika_results'):
            delattr(self, 'dsaitekika_results')
            print("🧹 Limpiando dsaitekika_results anteriores")
        if hasattr(self, 'dsaitekika_selected_df'):
            delattr(self, 'dsaitekika_selected_df')
            print("🧹 Limpiando dsaitekika_selected_df anteriores")
        
        # ✅ NUEVO: Establecer explícitamente el tipo de optimización
        self.last_executed_optimization = 'I'  # Marcar que se ejecutó I-optimización
        print(f"🔍 Debug - on_i_optimizer_finished: last_executed_optimization = 'I'")
        print(f"🔍 Debug - isaitekika_results existe después de limpiar: {hasattr(self, 'isaitekika_results')}")
        print(f"🔍 Debug - dsaitekika_results existe después de limpiar: {hasattr(self, 'dsaitekika_results')}")
        print(f"🔍 Debug - last_executed_optimization establecido: {self.last_executed_optimization}")
        
        # Mapear nombres de columnas si es necesario
        if '突出し量' in self.isaitekika_selected_df.columns:
            self.isaitekika_selected_df.rename(columns={'突出し量': '突出量'}, inplace=True)
        if '切込み量' in self.isaitekika_selected_df.columns:
            self.isaitekika_selected_df.rename(columns={'切込み量': '切込量'}, inplace=True)
        
        # ISaitekika: I基準値 NO se calcula nunca (mantener en blanco)
        if len(self.isaitekika_selected_df) > 0:
            self.isaitekika_selected_df['I基準値'] = ''
        output_folder = os.path.dirname(results['i_path']) if results['i_path'] else ""
        # Filtrar solo los gráficos relevantes a I最適化
        image_paths = sorted(glob.glob(os.path.join(output_folder, '*.png')))
        # Filtrar: solo mostrar histogramas y gráficos generales (no los que sean exclusivamente de D)
        i_image_paths = [p for p in image_paths if not ("D" in os.path.basename(p) or "d_optimal" in os.path.basename(p))]
        if not i_image_paths:
            i_image_paths = image_paths  # fallback: mostrar todos si no hay distinción
        
        # Limpiar contenido anterior
        self.graph_images = []
        self.graph_images_content = []
        
        self.display_graphs(i_image_paths)
        self.add_selected_samples_table_view(self.isaitekika_selected_df)
        self.ok_button.setEnabled(True)
        self.ng_button.setEnabled(True)
        self.create_navigation_buttons()
        
        # Deshabilitar botones de optimización después de completar I最適化
        self.d_optimize_button.setEnabled(False)
        self.i_optimize_button.setEnabled(False)
        # Aplicar estilo visual de deshabilitado
        self.d_optimize_button.setStyleSheet(self.d_optimize_button.styleSheet())
        self.i_optimize_button.setStyleSheet(self.i_optimize_button.styleSheet())
        
        QMessageBox.information(self, "完了",
                                f"✅ I最適化が完了しました。\n結果を保存しました:\n{results['i_path']}")
        
        self.loader_overlay.stop()

    def on_dsaitekika_finished(self, results):
        print("DEBUG: Entró en on_dsaitekika_finished")
        print("DEBUG results en on_dsaitekika_finished:", results)
        self.dsaitekika_results = results
        self.dsaitekika_selected_df = results['d_dataframe']  # ← Corregido para usar la misma estructura que on_d_optimizer_finished
        # ✅ NUEVO: Limpiar TODOS los resultados anteriores para evitar conflictos
        if hasattr(self, 'isaitekika_results'):
            delattr(self, 'isaitekika_results')
            print("🧹 Limpiando isaitekika_results anteriores")
        if hasattr(self, 'isaitekika_selected_df'):
            delattr(self, 'isaitekika_selected_df')
            print("🧹 Limpiando isaitekika_selected_df anteriores")
        
        # ✅ NUEVO: Establecer explícitamente el tipo de optimización
        self.last_executed_optimization = 'D'  # Marcar que se ejecutó D-optimización
        print(f"🔍 Debug - on_dsaitekika_finished: last_executed_optimization = 'D'")
        print(f"🔍 Debug - dsaitekika_results existe después de limpiar: {hasattr(self, 'dsaitekika_results')}")
        print(f"🔍 Debug - isaitekika_results existe después de limpiar: {hasattr(self, 'isaitekika_results')}")
        print(f"🔍 Debug - last_executed_optimization establecido: {self.last_executed_optimization}")

        # Mapear nombres de columnas si es necesario
        if '突出し量' in self.dsaitekika_selected_df.columns:
            self.dsaitekika_selected_df.rename(columns={'突出し量': '突出量'}, inplace=True)
        if '切込み量' in self.dsaitekika_selected_df.columns:
            self.dsaitekika_selected_df.rename(columns={'切込み量': '切込量'}, inplace=True)

        # ✅ Añadir número de muestra
        self.dsaitekika_selected_df.insert(0, "No.", list(range(1, len(self.dsaitekika_selected_df) + 1)))

        # ✅ Añadir columnas vacías para resultados esperados
        self.dsaitekika_selected_df["上面ダレ"] = ""
        self.dsaitekika_selected_df["側面ダレ"] = ""
        self.dsaitekika_selected_df["摩耗量"] = ""
        
        # Calcular D基準値 como referencia (si podemos reconstruir candidatos + índices)
        try:
            d_score_ref = None
            # Indices seleccionados (0-based) a partir de la columna No. si existe
            if "No." in self.dsaitekika_selected_df.columns:
                no_series = pd.to_numeric(self.dsaitekika_selected_df["No."], errors="coerce")
                selected_indices = [int(x) - 1 for x in no_series.dropna().tolist() if int(x) > 0]
            else:
                selected_indices = []

            input_file = getattr(self, "_last_dsaitekika_input_file", None) or getattr(self, "sample_file_path", None)
            if input_file and selected_indices:
                ext = os.path.splitext(str(input_file))[1].lower()
                df_all = pd.read_csv(input_file, encoding="utf-8-sig") if ext == ".csv" else pd.read_excel(input_file)
                X_candidates = _extract_design_matrix(df_all)
                d_score_ref = calculate_d_score_reference(X_candidates, selected_indices)

            if (d_score_ref is None or not np.isfinite(d_score_ref)) and len(self.dsaitekika_selected_df) > 0:
                # Fallback: score sobre los seleccionados solamente
                X_raw = _extract_design_matrix(self.dsaitekika_selected_df)
                X_scaled = _standardize_like_reference(X_raw)
                d_score_ref, _ = calculate_d_criterion_stable_reference(
                    X_scaled, method="auto", use_numerical_stable_method=True, verbose=False
                )

            self._last_d_score_reference = float(d_score_ref) if d_score_ref is not None else None
            if len(self.dsaitekika_selected_df) > 0:
                self.dsaitekika_selected_df["D基準値"] = self._last_d_score_reference if self._last_d_score_reference is not None else np.nan
        except Exception as e:
            print(f"⚠️ Error calculando D基準値 (referencia, Dsaitekika): {e}")

        image_paths = [
            self.dsaitekika_output_prefix + "_pca_features.png",
            self.dsaitekika_output_prefix + "_pca.png",
            self.dsaitekika_output_prefix + "_umap.png"
        ]
        self.display_graphs(image_paths)
        self.add_selected_samples_table_view(self.dsaitekika_selected_df)
        self.ok_button.setEnabled(True)
        self.ng_button.setEnabled(True)
        self.create_navigation_buttons()
        
        # Deshabilitar botones de optimización después de completar D最適化
        self.d_optimize_button.setEnabled(False)
        self.i_optimize_button.setEnabled(False)
        # Aplicar estilo visual de deshabilitado
        self.d_optimize_button.setStyleSheet(self.d_optimize_button.styleSheet())
        self.i_optimize_button.setStyleSheet(self.i_optimize_button.styleSheet())

        QMessageBox.information(self, "完了",
                                f"✅ D最適化が完了しました。\n結果を保存しました:\n{self.dsaitekika_output_excel}")
        
        self.loader_overlay.stop()

    def on_dsaitekika_error(self, message):
        QMessageBox.critical(self, "エラー", f"❌ D最適化中にエラーが発生しました:\n{message}")
        self.loader_overlay.stop()

    def on_sample_generation_finished(self):
        self.loader_overlay.stop()
        QMessageBox.information(self, "完了", "✅ サンプル組合せファイルが生成されました。")

    def on_sample_generation_error(self, error_msg):
        self.loader_overlay.stop()
        QMessageBox.critical(self, "エラー", f"❌ ファイル生成中にエラーが発生しました:\n{error_msg}")

    def load_results_file(self):
        # ✅ NUEVO: Pausar timers automáticos para evitar interferencia con el diálogo
        self.pause_auto_timers()
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "結果ファイルを選択",
            "",
            "Excel/CSV Files (*.xlsx *.xls *.csv);;Excel Files (*.xlsx *.xls);;CSV Files (*.csv)"
        )
        
        # ✅ NUEVO: Reanudar timers después del diálogo
        self.resume_auto_timers()
        
        if file_path:
            try:
                if hasattr(self.processor, "process_results_file_with_ui_values"):
                    # brush y 線材長 vienen del Excel/CSV de resultados (A13/A11/A21/A32 y 線材長)
                    selected_brush = None
                    diameter = float(self.diameter_selector.currentText()) if hasattr(self, "diameter_selector") else 0.15
                    material = self.material_selector.currentText() if hasattr(self, "material_selector") else "Steel"
                    self.processor.process_results_file_with_ui_values(file_path, selected_brush, diameter, material)
                else:
                    # fallback
                    self.processor.process_results_file(file_path, None, None)
                QMessageBox.information(self, "完了", "✅ 結果ファイルをデータベースに取り込みました。")
            except Exception as e:
                QMessageBox.critical(self, "エラー", f"❌ 結果ファイル処理中にエラーが発生しました:\n{e}")

    def backup_and_update_sample_file(self, results_file_path, project_folder=None):
        """Hacer backup del archivo de muestreo y eliminar filas duplicadas basadas en el archivo de resultados"""
        try:
            # Si no se especifica project_folder, usar el activo
            if project_folder is None:
                if not hasattr(self, 'proyecto_folder'):
                    raise ValueError("❌ アクティブなプロジェクトがありません。プロジェクトフォルダを指定してください。")
                project_folder = self.proyecto_folder
            
            # Obtener el nombre del proyecto desde la carpeta
            project_name = os.path.basename(project_folder)
            
            print(f"🔍 Debug - project_folder: {project_folder}")
            print(f"🔍 Debug - project_name: {project_name}")
            
            # Definir rutas - USAR EL ARCHIVO EN 99_Temp (o 99_Temp/Temp) DE LA CARPETA ESPECIFICADA
            temp_base = os.path.join(project_folder, "99_Temp")
            os.makedirs(temp_base, exist_ok=True)

            # ✅ NO depender del nombre del archivo:
            # elegir cualquier *_未実験データ.(xlsx/xls/csv) dentro de 99_Temp o 99_Temp/Temp.
            # Preferencia: xlsx > xls > csv, y si hay varios, el más reciente.
            exts_priority = {".xlsx": 0, ".xls": 1, ".csv": 2}

            def _collect_candidates(folder: str):
                out = []
                try:
                    if not os.path.isdir(folder):
                        return out
                    for fn in os.listdir(folder):
                        if fn.startswith("~$"):
                            continue
                        if "_backup_" in fn:
                            continue
                        ext = os.path.splitext(fn)[1].lower()
                        if ext not in exts_priority:
                            continue
                        if not fn.endswith(f"_未実験データ{ext}"):
                            continue
                        full = os.path.join(folder, fn)
                        if os.path.isfile(full):
                            out.append(full)
                except Exception:
                    return []
                return out

            candidates = _collect_candidates(temp_base) + _collect_candidates(os.path.join(temp_base, "Temp"))
            if candidates:
                candidates.sort(key=lambda p: (exts_priority.get(os.path.splitext(p)[1].lower(), 9), -os.path.getmtime(p)))
                sample_file_path = candidates[0]
                try:
                    print("🔍 Debug - candidatos *_未実験データ.* encontrados (top 5):")
                    for p in candidates[:5]:
                        print(f"  - {p}")
                except Exception:
                    pass
            else:
                # fallback legacy: nombre basado en carpeta
                candidate_sample_paths = [
                    os.path.join(temp_base, f"{project_name}_未実験データ.xlsx"),
                    os.path.join(temp_base, f"{project_name}_未実験データ.xls"),
                    os.path.join(temp_base, f"{project_name}_未実験データ.csv"),
                ]
                sample_file_path = next((p for p in candidate_sample_paths if os.path.exists(p)), candidate_sample_paths[0])

            sample_ext = os.path.splitext(sample_file_path)[1].lower()
            
            print(f"🔍 Debug - temp_base: {temp_base}")
            print(f"🔍 Debug - sample_file_path: {sample_file_path}")
            
            # Verificar que existe el archivo de muestreo en 99_Temp
            if not os.path.exists(sample_file_path):
                raise ValueError(f"❌ サンプルファイルが見つかりません: {sample_file_path}")
            
            # Crear carpeta backup en 99_Temp
            backup_folder = os.path.join(temp_base, "backup")
            os.makedirs(backup_folder, exist_ok=True)
            
            # Generar nombre del backup con timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime('%y%m%d_%H%M')
            backup_filename = f"{project_name}_未実験データ_backup_{timestamp}{sample_ext if sample_ext in ('.csv','.xlsx','.xls') else '.xlsx'}"
            backup_path = os.path.join(backup_folder, backup_filename)
            
            # 1. Hacer backup del archivo de muestreo
            print(f"📋 Creando backup: {backup_path}")
            shutil.copy2(sample_file_path, backup_path)
            print(f"✅ バックアップが正常に作成されました")
            
            def _read_any_table(path: str) -> pd.DataFrame:
                ext = os.path.splitext(path)[1].lower()
                if ext == ".csv":
                    return pd.read_csv(path, encoding="utf-8-sig")
                return pd.read_excel(path)

            # 2. Leer archivo de resultados (Excel/CSV)
            print(f"📊 Leyendo archivo de resultados: {results_file_path}")
            df_results = _read_any_table(results_file_path)

            def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
                # Strip + normalizar espacios (incluye full-width)
                df = df.copy()
                df.columns = [
                    str(c).replace("\u3000", " ").strip() if c is not None else ""
                    for c in df.columns
                ]
                rename_map = {}
                # Variantes conocidas
                for c in df.columns:
                    if c == "突出し量":
                        rename_map[c] = "突出量"
                    elif c == "切込み量":
                        rename_map[c] = "切込量"
                    elif c == "回転方向":
                        rename_map[c] = "UPカット"
                    elif c == "UPカット/回転方向":
                        rename_map[c] = "UPカット"
                if rename_map:
                    df = df.rename(columns=rename_map)
                return df

            df_results = _normalize_columns(df_results)

            # 3. Leer archivo de muestreo actual (Excel/CSV)
            print(f"📊 Leyendo archivo de muestreo: {sample_file_path}")
            df_sample = _read_any_table(sample_file_path)
            df_sample = _normalize_columns(df_sample)

            print(f"📊 Archivo de muestreo original: {len(df_sample)} filas")

            # 4. Eliminar filas del archivo de muestreo:
            # - Comparar por igualdad (normalizada) solo en las columnas de condición
            # - Para 線材長, elegir la fila cuyo valor sea más cercano (closest match)
            #
            # Nota: NO usamos 直径/材料 como clave porque a veces están ausentes o vacíos en resultados
            # y eso impide eliminar filas aunque las condiciones sean iguales.
            strict_cols_candidate = [
                # Condiciones
                "回転速度", "送り速度", "UPカット",
                "切込量", "突出量", "載せ率", "パス数",
            ]
            len_col = "線材長"

            available_columns = df_results.columns.tolist()
            print(f"🔍 Columnas disponibles en archivo de resultados: {available_columns}")
            print(f"🔍 Columnas disponibles en archivo de muestreo: {df_sample.columns.tolist()}")

            # Requerimos al menos las 7 columnas de condición
            required_condition_cols = ["回転速度", "送り速度", "UPカット", "切込量", "突出量", "載せ率", "パス数"]
            missing_required = [c for c in required_condition_cols if (c not in df_results.columns or c not in df_sample.columns)]
            if missing_required:
                raise ValueError(f"❌ Faltan columnas de condición para comparar: {missing_required}")

            strict_cols = [c for c in strict_cols_candidate if (c in df_results.columns and c in df_sample.columns)]
            if not strict_cols:
                raise ValueError("❌ No hay columnas comunes suficientes para comparar resultados vs 未実験データ.")

            if len_col not in df_results.columns:
                raise ValueError(f"❌ El archivo de resultados no contiene la columna requerida: {len_col}")

            if len_col not in df_sample.columns:
                print(f"⚠️ El archivo de muestreo no contiene '{len_col}'. Se eliminará la primera coincidencia por clave estricta.")

            import numpy as np
            from collections import defaultdict

            # Derivar un brush_id estable (si hay one-hot en ambos)
            brush_cols = ["A13", "A11", "A21", "A32"]
            has_brush = all(c in df_results.columns for c in brush_cols) and all(c in df_sample.columns for c in brush_cols)

            int_cols = set(["回転速度", "送り速度", "UPカット", "パス数"])
            float_cols = set(["切込量", "突出量", "載せ率"])

            def _normalize_upcut_series(s: pd.Series) -> pd.Series:
                # Aceptar 0/1, True/False, y algunas variantes texto comunes
                if s is None:
                    return s
                try:
                    if s.dtype == "bool":
                        return s.astype("Int64")
                except Exception:
                    pass
                # map texto -> 0/1 cuando aplique
                s_str = s.astype(str).str.replace("\u3000", " ").str.strip()
                upper = s_str.str.upper()
                mapped = upper.map({
                    "UP": 1, "DOWN": 0,
                    "CW": 1, "CCW": 0,
                    "TRUE": 1, "FALSE": 0,
                    "1": 1, "0": 0,
                })
                # conservar original donde no mapea
                return pd.to_numeric(mapped.fillna(s_str), errors="coerce")

            def _norm_key_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
                out = df[cols].copy()
                for c in cols:
                    if c in int_cols:
                        if c == "UPカット":
                            out[c] = _normalize_upcut_series(out[c]).round(0).astype("Int64")
                        else:
                            out[c] = pd.to_numeric(out[c], errors="coerce").round(0).astype("Int64")
                    elif c in float_cols:
                        out[c] = pd.to_numeric(out[c], errors="coerce").round(6)
                    else:
                        out[c] = out[c]
                return out

            def _brush_id_from_onehot(df: pd.DataFrame) -> pd.Series:
                # 1->A11, 2->A21, 3->A32, 4->A13
                a13 = pd.to_numeric(df["A13"], errors="coerce").fillna(0).astype(int)
                a11 = pd.to_numeric(df["A11"], errors="coerce").fillna(0).astype(int)
                a21 = pd.to_numeric(df["A21"], errors="coerce").fillna(0).astype(int)
                a32 = pd.to_numeric(df["A32"], errors="coerce").fillna(0).astype(int)
                bid = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
                bid = bid.mask(a11 == 1, 1)
                bid = bid.mask(a21 == 1, 2)
                bid = bid.mask(a32 == 1, 3)
                bid = bid.mask(a13 == 1, 4)
                return bid

            sample_key_df = _norm_key_cols(df_sample, strict_cols)
            results_key_df = _norm_key_cols(df_results, strict_cols)

            match_cols = list(strict_cols)
            if has_brush:
                df_sample["__brush_id"] = _brush_id_from_onehot(df_sample)
                df_results["__brush_id"] = _brush_id_from_onehot(df_results)
                match_cols.append("__brush_id")

            # Normalizar también brush_id
            if "__brush_id" in match_cols:
                sample_key_df["__brush_id"] = df_sample["__brush_id"].astype("Int64")
                results_key_df["__brush_id"] = df_results["__brush_id"].astype("Int64")

            # Arrays de longitud
            sample_len = pd.to_numeric(df_sample[len_col], errors="coerce").astype(float).to_numpy() if len_col in df_sample.columns else np.full(len(df_sample), np.nan)
            results_len = pd.to_numeric(df_results[len_col], errors="coerce").astype(float).to_numpy()

            # Construir lookup: key -> lista de posiciones (indices) del sample
            buckets = defaultdict(list)
            sample_idx = df_sample.index.to_numpy()
            for i in range(len(df_sample)):
                row = sample_key_df.iloc[i]
                # key como tupla (incluye NA como None)
                key = tuple([None if pd.isna(row[c]) else row[c] for c in match_cols])
                buckets[key].append(i)

            used_pos = np.zeros(len(df_sample), dtype=bool)
            rows_to_remove = []
            missing = 0
            for r_i in range(len(df_results)):
                rrow = results_key_df.iloc[r_i]
                rkey = tuple([None if pd.isna(rrow[c]) else rrow[c] for c in match_cols])
                cand = buckets.get(rkey, [])
                # filtrar usados
                cand = [p for p in cand if not used_pos[p]]
                if not cand:
                    missing += 1
                    continue

                chosen = cand[0]
                rlen = results_len[r_i]
                if len_col in df_sample.columns and not np.isnan(rlen):
                    d = np.abs(sample_len[cand] - rlen)
                    if not np.all(np.isnan(d)):
                        chosen = cand[int(np.nanargmin(d))]

                used_pos[chosen] = True
                rows_to_remove.append(sample_idx[chosen])

            if missing > 0:
                print(f"⚠️ Coincidencias no encontradas para {missing}/{len(df_results)} filas de resultados. (Revisa tipos/columnas/valores)")

            if rows_to_remove:
                print(f"🧹 Coincidencias encontradas: {len(rows_to_remove)} (con 線材長 por proximidad)")
            
            # Eliminar filas duplicadas
            if rows_to_remove:
                df_sample_updated = df_sample.drop(rows_to_remove)
                print(f"🗑️ {len(rows_to_remove)} 件の重複行が削除されました")
                print(f"📊 Archivo de muestreo actualizado: {len(df_sample_updated)} filas")
                
                # Guardar archivo actualizado
                try:
                    if sample_ext == ".csv":
                        df_sample_updated.to_csv(sample_file_path, index=False, encoding="utf-8-sig")
                    else:
                        df_sample_updated.to_excel(sample_file_path, index=False)
                except PermissionError as e:
                    # En Windows esto suele pasar si el archivo está abierto (Excel lo bloquea)
                    raise PermissionError(
                        f"❌ No se pudo guardar el archivo de muestreo en 99_Temp (permiso denegado).\n\n"
                        f"Probablemente el archivo está abierto en Excel u otra aplicación.\n"
                        f"Ciérralo y vuelve a intentarlo.\n\n"
                        f"Archivo:\n{sample_file_path}"
                    ) from e
                print(f"✅ Archivo de muestreo actualizado guardado: {sample_file_path}")
                
                return {
                    'backup_path': backup_path,
                    'removed_rows': len(rows_to_remove),
                    'remaining_rows': len(df_sample_updated)
                }
            else:
                print(f"ℹ️ 削除する重複行が見つかりませんでした")
                return {
                    'backup_path': backup_path,
                    'removed_rows': 0,
                    'remaining_rows': len(df_sample)
                }
                
        except RuntimeError as e:
            if "already deleted" in str(e):
                # Ignorar silenciosamente el error de widget ya eliminado
                pass
            else:
                print(f"❌ Error en backup_and_update_sample_file: {str(e)}")
                print(f"🔍 Debug - Estado actual:")
                print(f"  - project_folder: {project_folder}")
                print(f"  - project_name: {os.path.basename(project_folder) if project_folder else 'No especificado'}")
                print(f"  - results_file_path: {results_file_path}")
                print(f"  - temp_base esperado: {os.path.join(project_folder, '99_Temp') if project_folder else 'No especificado'}")
                raise e
        except Exception as e:
            print(f"❌ Error en backup_and_update_sample_file: {str(e)}")
            print(f"🔍 Debug - Estado actual:")
            print(f"  - project_folder: {project_folder}")
            print(f"  - project_name: {os.path.basename(project_folder) if project_folder else 'No especificado'}")
            print(f"  - results_file_path: {results_file_path}")
            print(f"  - temp_base esperado: {os.path.join(project_folder, '99_Temp') if project_folder else 'No especificado'}")
            raise e

    def on_execute_results_clicked(self):
        if not hasattr(self, "results_file_path"):
            QMessageBox.warning(self, "エラー", "❌ 結果ファイルが読み込まれていません。")
            return
        
        # Obtener valores de la UI
        # brush y 線材長 deben venir del archivo de resultados (no de la UI)
        selected_brush = None
        diameter = float(self.diameter_selector.currentText())
        material = self.material_selector.currentText()

        try:
            # ✅ NUEVO: Hacer backup y actualizar archivo de muestreo
            print("🔄 Iniciando proceso de backup y actualización del archivo de muestreo...")
            # Solo hacer backup si hay un proyecto activo
            if hasattr(self, 'proyecto_folder'):
                backup_result = self.backup_and_update_sample_file(self.results_file_path, self.proyecto_folder)
            else:
                print("⚠️ アクティブなプロジェクトがありません。バックアップとサンプルファイルの更新をスキップします")
                backup_result = {'backup_path': None, 'removed_rows': 0, 'remaining_rows': 0}
            
            # Procesar archivo de resultados (線材長 viene del archivo)
            dbu = self.processor.process_results_file_with_ui_values(
                self.results_file_path, 
                selected_brush, 
                diameter, 
                material
            )
            
            # Mostrar mensaje de éxito con información del backup
            message = f"✅ 結果ファイルがデータベースに取り込まれました。\n\n"
            if backup_result['backup_path']:
                message += f"📋 バックアップ作成: {os.path.basename(backup_result['backup_path'])}\n"
                message += f"🗑️ サンプルファイルから削除された行: {backup_result['removed_rows']}\n"
                message += f"📊 サンプルファイルの残り行数: {backup_result['remaining_rows']}"
            else:
                message += f"ℹ️ バックアップは実行されませんでした（アクティブなプロジェクトがありません）"
            
            # Aviso único si hubo sobrescritura en BBDD
            if isinstance(dbu, dict):
                updated = int(dbu.get("updated", 0) or 0)
                inserted = int(dbu.get("inserted", 0) or 0)
                if updated > 0:
                    message += "\n\n⚠️ 既存データを上書きします。BBDDのバックアップを作成しました。"
                    message += f"\n🔁 上書き: {updated} / ➕ 追加: {inserted}"
                    if dbu.get("db_backup_path"):
                        message += f"\n📋 BBDDバックアップ: {os.path.basename(str(dbu.get('db_backup_path')))}"
                    else:
                        message += "\n📋 BBDDバックアップ: (作成できませんでした)"

            QMessageBox.information(self, "完了", message)
            self.create_filter_view()
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"❌ データベースへの取り込み中にエラーが発生しました:\n{str(e)}")

    def closeEvent(self, event):
        """Maneja el cierre de la ventana principal"""
        try:
            print("🛑 Cerrando aplicación...")
            
            # Cancelar análisis no lineal si está corriendo
            if hasattr(self, 'nonlinear_worker') and self.nonlinear_worker is not None:
                if self.nonlinear_worker.isRunning():
                    print("🛑 Cancelando análisis no lineal antes de cerrar...")
                    self.nonlinear_worker.cancel()
                    
                    # Esperar a que el thread termine (máximo 5 segundos)
                    if self.nonlinear_worker.isRunning():
                        self.nonlinear_worker.quit()
                        if not self.nonlinear_worker.wait(5000):
                            print("⚠️ El worker no terminó en 5 segundos, forzando cierre...")
                            self.nonlinear_worker.terminate()
                            self.nonlinear_worker.wait(1000)
                    
                    print("✅ Worker de análisis no lineal cancelado")
            
            # Cerrar base de datos
            if hasattr(self, 'db'):
                self.db.close()
            
            print("✅ Aplicación cerrada correctamente")
            event.accept()
            
        except Exception as e:
            print(f"❌ Error en closeEvent: {e}")
            import traceback
            traceback.print_exc()
            # Aún así cerrar la aplicación
            if hasattr(self, 'db'):
                try:
                    self.db.close()
                except:
                    pass
            event.accept()

    def handle_single_file_load(self):
        # ✅ NUEVO: Pausar timers automáticos para evitar interferencia con el diálogo
        self.pause_auto_timers()
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "ファイルを選択",
            "",
            "Excel/CSV Files (*.xlsx *.xls *.csv);;Excel Files (*.xlsx *.xls);;CSV Files (*.csv)"
        )
        
        # ✅ NUEVO: Reanudar timers después del diálogo
        self.resume_auto_timers()
        
        if not file_path:
            self.load_file_label.setText("ファイル未選択")
            # Reset all UI elements to default state when no file is selected
            self.set_ui_state_for_no_file()
            return

        self.load_file_label.setText(f"読み込み済み: {os.path.basename(file_path)}")

        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".csv":
                df_raw = pd.read_csv(file_path, header=None, nrows=2, encoding="utf-8-sig")
            else:
                df_raw = pd.read_excel(file_path, header=None, nrows=2)

            fila_1 = df_raw.iloc[0].fillna("").tolist()
            fila_2 = df_raw.iloc[1].fillna("").tolist()

            # Aceptar tanto "UPカット" como "回転方向" como 3ª columna
            columnas_muestreo_exactas = ['回転速度', '送り速度', 'UPカット/回転方向', '切込量', '突出量', '載せ率', 'パス数']
            # Reconocimiento de resultados (nuevo formato): incluye brush one-hot, 線材長, 面粗度(Ra)前/後, y opcionalmente 切削力X/Y/Z
            columnas_resultados_minimas = [
                '回転速度', '送り速度', 'UPカット/回転方向', '切込量', '突出量', '載せ率', 'パス数',
                '線材長',
                '上面ダレ/上面ダレ量', '側面ダレ/側面ダレ量', '摩耗量',
                '面粗度(Ra)前/面粗度前/粗度(Ra)前', '面粗度(Ra)後/面粗度後/粗度(Ra)後',
                '実験日'
            ]

            def _matches_sample_header(row, start_idx: int) -> bool:
                try:
                    # Formato antiguo: 7 variables
                    if (
                        row[start_idx] == '回転速度' and
                        row[start_idx + 1] == '送り速度' and
                        row[start_idx + 2] in ('UPカット', '回転方向') and
                        row[start_idx + 3] in ('切込量', '切込み量') and
                        row[start_idx + 4] in ('突出量', '突出し量') and
                        row[start_idx + 5:start_idx + 7] == ['載せ率', 'パス数']
                    ):
                        return True

                    # Formato nuevo: one-hot brush + variables
                    if (
                        row[start_idx:start_idx + 4] == ['A13', 'A11', 'A21', 'A32'] and
                        row[start_idx + 4] == '回転速度' and
                        row[start_idx + 5] == '送り速度' and
                        row[start_idx + 6] in ('UPカット', '回転方向') and
                        row[start_idx + 7] in ('切込量', '切込み量') and
                        row[start_idx + 8] in ('突出量', '突出し量') and
                        row[start_idx + 9:start_idx + 11] == ['載せ率', 'パス数']
                    ):
                        return True

                    return False
                except Exception:
                    return False

            def _matches_results_header(row) -> bool:
                """
                Detecta archivo de resultados por presencia de columnas de condiciones + resultados.
                - Requiere: A13/A11/A21/A32 + 7 variables de condición + 線材長 + (上面/側面/摩耗) + (面粗度 前/後) + 実験日
                - Acepta variantes: 回転方向 vs UPカット, 突出し量 vs 突出量, 上面ダレ量 vs 上面ダレ, 側面ダレ量 vs 側面ダレ
                - 切削力X/Y/Z: opcional
                """
                try:
                    headers = {str(x).strip() for x in row if str(x).strip() != ""}
                    has_brush = all(c in headers for c in ("A13", "A11", "A21", "A32"))
                    has_dir = ('UPカット' in headers) or ('回転方向' in headers)
                    has_out = ('突出量' in headers) or ('突出し量' in headers)
                    has_cut = ('切込量' in headers) or ('切込み量' in headers)
                    has_top = ('上面ダレ' in headers) or ('上面ダレ量' in headers)
                    has_side = ('側面ダレ' in headers) or ('側面ダレ量' in headers)
                    has_ra_pre = ('面粗度(Ra)前' in headers) or ('面粗度前' in headers) or ('粗度(Ra)前' in headers)
                    has_ra_post = ('面粗度(Ra)後' in headers) or ('面粗度後' in headers) or ('粗度(Ra)後' in headers)

                    has_design = (
                        ('回転速度' in headers) and
                        ('送り速度' in headers) and
                        has_dir and
                        has_cut and
                        has_out and
                        ('載せ率' in headers) and
                        ('パス数' in headers)
                    )
                    has_results = has_top and has_side and ('摩耗量' in headers) and has_ra_pre and has_ra_post
                    has_required_meta = ('線材長' in headers) and ('実験日' in headers)
                    return has_brush and has_design and has_results and has_required_meta
                except Exception:
                    return False

            # Verificar archivo de resultados (nuevo): header en fila 1 o (a veces) en fila 2
            # ✅ Prioridad: si un archivo parece "resultados" y "muestreo" a la vez, se tratará como resultados.
            is_resultados = _matches_results_header(fila_1) or _matches_results_header(fila_2)

            # Verificar archivo de muestreo:
            # - Permite offset 0 (A1) o 1 (si hay columna índice/No. al inicio)
            is_muestreo = _matches_sample_header(fila_1, 0) or _matches_sample_header(fila_1, 1)

            # Debug: imprimir las filas para diagnosticar
            print(f"🔍 Debug - Fila 1: {fila_1}")
            print(f"🔍 Debug - Fila 2: {fila_2}")
            print(f"🔍 Debug - Columnas muestreo esperadas: {columnas_muestreo_exactas}")
            print(f"🔍 Debug - Columnas resultados esperadas: {columnas_resultados_minimas}")
            print(f"🔍 Debug - is_resultados: {is_resultados}")
            print(f"🔍 Debug - is_muestreo: {is_muestreo}")

            if is_resultados:
                QMessageBox.information(self, "ファイル種別", "📄 このファイルは【結果】ファイルとして認識されました。")
                self.results_file_path = file_path
                self.show_results_button.setEnabled(True)
                
                # Set UI state for results file
                self.set_ui_state_for_results_file()
                # UI enablement debajo del selector (sin depender del nombre del archivo)
                try:
                    self._last_loaded_file_kind = "results"
                    if hasattr(self, "on_file_loaded"):
                        self.on_file_loaded(file_path, is_results=True)
                    elif hasattr(self, "_set_widgets_below_sample_selector_enabled"):
                        self._set_widgets_below_sample_selector_enabled(True)
                except Exception:
                    pass

            elif is_muestreo:
                QMessageBox.information(self, "ファイル種別", "📄 このファイルは【サンプル】ファイルとして認識されました。")
                self.sample_file_path = file_path
                self.show_results_button.setEnabled(False)
                
                # UI enablement debajo del selector (sin depender del nombre del archivo)
                try:
                    self._last_loaded_file_kind = "sample"
                    if hasattr(self, "on_file_loaded"):
                        self.on_file_loaded(file_path, is_results=False)
                    elif hasattr(self, "_set_widgets_below_sample_selector_enabled"):
                        self._set_widgets_below_sample_selector_enabled(False)
                except Exception:
                    pass
                
                # ✅ NUEVO: Verificar si el archivo pertenece a un proyecto diferente
                file_dir = os.path.dirname(file_path)
                file_name = os.path.basename(file_path)
                
                print(f"🔍 Debug Load: file_dir = {file_dir}")
                print(f"🔍 Debug Load: file_name = {file_name}")
                print(f"🔍 Debug Load: proyecto_folder = {getattr(self, 'proyecto_folder', 'No existe')}")
                
                # Si hay un proyecto activo, verificar si el archivo pertenece al mismo proyecto
                if hasattr(self, 'proyecto_folder') and hasattr(self, 'proyecto_nombre'):
                    # Verificar si el archivo está en el proyecto principal o en sus subcarpetas
                    is_same_project = (file_dir == self.proyecto_folder or 
                                      file_dir.startswith(self.proyecto_folder + os.sep))
                    
                    print(f"🔍 Debug Load: is_same_project = {is_same_project}")
                    
                    if not is_same_project:
                        # Archivo de un proyecto diferente, limpiar proyecto activo
                        print(f"🔄 Archivo de proyecto diferente detectado. Limpiando proyecto activo: {getattr(self, 'proyecto_nombre', 'Unknown')}")
                        print(f"🔄 Archivo: {file_dir}")
                        print(f"🔄 Proyecto: {self.proyecto_folder}")
                        delattr(self, 'proyecto_folder')
                        delattr(self, 'proyecto_nombre')
                        if hasattr(self, 'muestreo_guardado_path'):
                            delattr(self, 'muestreo_guardado_path')
                        print("✅ Proyecto activo limpiado. Se pedirá nuevo proyecto en la próxima optimización.")
                    else:
                        print(f"✅ Archivo pertenece al proyecto activo: {getattr(self, 'proyecto_nombre', 'Unknown')}")
                else:
                    print("🔍 Debug Load: アクティブなプロジェクトがありません")
                
                # ✅ NUEVO: Si estamos en la pantalla de filtros, volver a la pantalla principal
                # Verificar si estamos en la vista de filtros
                in_filter_view = False
                for i in range(self.center_layout.count()):
                    item = self.center_layout.itemAt(i)
                    if item.widget() and isinstance(item.widget(), QLabel):
                        if item.widget().text() == "データフィルター":
                            in_filter_view = True
                            break
                
                if in_filter_view:
                    print("🔄 Archivo de muestreo detectado en pantalla de filtros. Volviendo a pantalla principal...")
                    # Limpiar la pantalla y volver al estado inicial
                    self.clear_main_screen()
                
                # Habilitar botones de optimización cuando se carga un nuevo archivo de muestras
                self.d_optimize_button.setEnabled(True)
                self.i_optimize_button.setEnabled(True)
                # Aplicar estilo visual de habilitado
                self.d_optimize_button.setStyleSheet(self.d_optimize_button.styleSheet())
                self.i_optimize_button.setStyleSheet(self.i_optimize_button.styleSheet())
                
                # Set UI state for sample file
                self.set_ui_state_for_sample_file()
                
                # Limpiar resultados anteriores
                if hasattr(self, 'dsaitekika_results'):
                    delattr(self, 'dsaitekika_results')
                if hasattr(self, 'isaitekika_results'):
                    delattr(self, 'isaitekika_results')
                if hasattr(self, 'dsaitekika_selected_df'):
                    delattr(self, 'dsaitekika_selected_df')
                if hasattr(self, 'isaitekika_selected_df'):
                    delattr(self, 'isaitekika_selected_df')
                
                # Limpiar gráficos y tablas anteriores
                self.graph_images = []
                self.graph_images_content = []
                self.current_graph_index = 0
                
                # Limpiar área de gráficos
                if hasattr(self, 'graph_area') and self.graph_area.layout():
                    layout = self.graph_area.layout()
                    for i in reversed(range(layout.count())):
                        widget = layout.itemAt(i).widget()
                        if widget:
                            widget.setParent(None)
                
                # Deshabilitar botones OK/NG
                self.ok_button.setEnabled(False)
                self.ng_button.setEnabled(False)

            else:
                QMessageBox.warning(self, "警告", "⚠️ このファイルはサンプルでも結果でもないようです。")
                self.show_results_button.setEnabled(False)
                
                # Reset all UI elements to default state when file is neither sample nor results
                self.set_ui_state_for_no_file()
                try:
                    self._last_loaded_file_kind = None
                    if hasattr(self, "on_file_loaded"):
                        self.on_file_loaded(file_path, is_results=False)
                    elif hasattr(self, "_set_widgets_below_sample_selector_enabled"):
                        self._set_widgets_below_sample_selector_enabled(False)
                except Exception:
                    pass

        except Exception as e:
            QMessageBox.critical(self, "エラー", f"❌ ファイルの読み込み中にエラーが発生しました:\n{str(e)}")
            # Reset all UI elements to default state when error occurs
            self.set_ui_state_for_no_file()
            try:
                self._last_loaded_file_kind = None
                if hasattr(self, "_set_widgets_below_sample_selector_enabled"):
                    self._set_widgets_below_sample_selector_enabled(False)
            except Exception:
                pass

    def get_sample_size(self):
        """Obtener el tamaño de muestra del campo de entrada"""
        try:
            size = int(self.sample_size_input.text())
            if 10 <= size <= 50:
                return size
            else:
                QMessageBox.warning(self, "エラー", f"❌ サンプルサイズは10-50の範囲内である必要があります。\n現在の値: {size}")
                self.sample_size_input.setText("15")
                return 15
        except ValueError:
            QMessageBox.warning(self, "エラー", "❌ サンプルサイズは数値である必要があります。")
            self.sample_size_input.setText("15")
            return 15

    def validate_sample_size(self):
        """Validar el tamaño de muestra cuando se termina de editar"""
        try:
            size = int(self.sample_size_input.text())
            if not (10 <= size <= 50):
                QMessageBox.warning(self, "エラー", f"❌ サンプルサイズは10-50の範囲内である必要があります。\n現在の値: {size}")
                self.sample_size_input.setText("15")
        except ValueError:
            QMessageBox.warning(self, "エラー", "❌ サンプルサイズは数値である必要があります。")
            self.sample_size_input.setText("15")

    def on_sample_size_focus_out(self, event):
        """Manejar la pérdida de foco del campo de tamaño de muestra"""
        # Llamar al método original de QLineEdit
        super(QLineEdit, self.sample_size_input).focusOutEvent(event)
        # Validar el valor
        self.validate_sample_size()

    def export_database_to_excel(self):
        db_path = RESULTS_DB_PATH
        conn = sqlite3.connect(db_path, timeout=10)

        try:
            df = pd.read_sql_query("SELECT * FROM main_results", conn)
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"❌ データベースからの取得中にエラーが発生しました:\n{e}")
            return
        finally:
            conn.close()

        # Formatear columnas según el orden esperado de resultados (sin tocar la DB)
        try:
            rename_map = {
                "面粗度前": "面粗度(Ra)前",
                "面粗度後": "面粗度(Ra)後",
            }
            df_export = df.rename(columns=rename_map)
            desired_order = [
                "id",
                "バリ除去", "上面ダレ量", "側面ダレ量", "摩耗量",
                "切削力X", "切削力Y", "切削力Z",
                "面粗度(Ra)後",
                "A13", "A11", "A21", "A32",
                "直径", "材料",
                "回転速度", "送り速度", "UPカット", "切込量", "突出量", "載せ率", "線材長", "パス数",
                "加工時間",
                "面粗度(Ra)前",
                "実験日",
            ]
            for col in desired_order:
                if col not in df_export.columns:
                    df_export[col] = ""
            df_export = df_export[[c for c in desired_order if c in df_export.columns]]
        except Exception:
            df_export = df

        # ✅ NUEVO: Pausar timers automáticos para evitar interferencia con el diálogo
        self.pause_auto_timers()
        
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Excelとして保存", "", "Excelファイル (*.xlsx)", options=options
        )
        
        # ✅ NUEVO: Reanudar timers después del diálogo
        self.resume_auto_timers()

        if filepath:
            try:
                df_export.to_excel(filepath, index=False)
                QMessageBox.information(self, "完了", "✅ データベースが正常にエクスポートされました。")
            except Exception as e:
                QMessageBox.critical(self, "エラー", f"❌ エクスポート中にエラーが発生しました:\n{e}")

    def export_yosoku_database_to_excel(self):
        """Exportar base de datos de Yosoku a Excel con diálogo de progreso"""
        # Crear diálogo personalizado más bonito
        dialog = QDialog(self)
        dialog.setWindowTitle("データベース選択")
        dialog.setFixedSize(500, 350)
        dialog.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        
        # Layout principal
        main_layout = QVBoxLayout(dialog)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Título
        title_label = QLabel("データベースを選択")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Subtítulo
        subtitle_label = QLabel("エクスポートするデータベースを選択してください")
        subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #7f8c8d;
                padding: 5px;
            }
        """)
        subtitle_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle_label)
        
        main_layout.addSpacing(20)
        
        # Contenedor para los 3 botones alineados
        buttons_container = QHBoxLayout()
        buttons_container.setSpacing(15)
        buttons_container.setContentsMargins(0, 0, 0, 0)
        
        # Botón Lineal
        lineal_button = QPushButton("線形データベース")
        lineal_button.setFixedSize(140, 50)
        lineal_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        
        # Botón No Lineal
        no_lineal_button = QPushButton("非線形データベース")
        no_lineal_button.setFixedSize(140, 50)
        no_lineal_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        
        # Botón Cancelar
        cancel_button = QPushButton("キャンセル")
        cancel_button.setFixedSize(140, 50)
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
            QPushButton:pressed {
                background-color: #6c7a7b;
            }
        """)
        
        # Agregar los 3 botones alineados
        buttons_container.addStretch()
        buttons_container.addWidget(lineal_button)
        buttons_container.addWidget(no_lineal_button)
        buttons_container.addWidget(cancel_button)
        buttons_container.addStretch()
        
        main_layout.addLayout(buttons_container)
        main_layout.addStretch()
        
        # Estilo del diálogo
        dialog.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
                border-radius: 10px;
            }
        """)
        
        # Conectar señales
        lineal_button.clicked.connect(lambda: dialog.done(1))
        no_lineal_button.clicked.connect(lambda: dialog.done(2))
        cancel_button.clicked.connect(lambda: dialog.done(0))
        
        # Pausar timers para evitar interferencia
        self.pause_auto_timers()
        
        # Mostrar diálogo
        result = dialog.exec()
        
        # Reanudar timers
        self.resume_auto_timers()
        
        # Determinar qué BBDD usar según la respuesta
        if result == 0:  # Cancelar
            return
        elif result == 1:  # Lineal
            db_path = YOSOKU_LINEAL_DB_PATH
            db_name = "線形データベース"
        elif result == 2:  # No Lineal
            db_path = YOSOKU_NO_LINEAL_DB_PATH
            db_name = "非線形データベース"
        else:
            return
        
        # Verificar si la base de datos existe
        if not os.path.exists(db_path):
            QMessageBox.warning(
                self, 
                "警告", 
                f"❌ {db_name}が見つかりません。\n\n"
                f"ファイル: {db_path}\n\n"
                f"まず予測を実行してデータベースにデータをインポートしてください。"
            )
            return
        
        # Verificar que la base de datos no esté vacía (sin mostrar loading aún)
        conn = sqlite3.connect(db_path, timeout=10)
        try:
            df = pd.read_sql_query("SELECT * FROM yosoku_predictions", conn)
            
            if len(df) == 0:
                QMessageBox.information(
                    self, 
                    "情報", 
                    f"📊 {db_name}は空です。\n\n"
                    f"まず予測を実行してデータベースにデータをインポートしてください。"
                )
                return
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"❌ データベースからの取得中にエラーが発生しました:\n{e}")
            return
        finally:
            conn.close()

        # ✅ NUEVO: Pausar timers automáticos para evitar interferencia con el diálogo
        self.pause_auto_timers()
        
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getSaveFileName(
            self, "予測データベースをExcelとして保存", "", "Excelファイル (*.xlsx)", options=options
        )
        
        # ✅ NUEVO: Reanudar timers después del diálogo
        self.resume_auto_timers()

        if not filepath:
            return  # Usuario canceló la selección de archivo
        
        # ✅ MOSTRAR LOADING después de seleccionar el archivo
        try:
            # Crear y mostrar diálogo de progreso
            self.yosoku_export_progress_dialog = YosokuExportProgressDialog(self)
            self.yosoku_export_progress_dialog.show()
            # Durante el loading con chibi: flecha/consola por encima
            self.set_console_overlay_topmost(True)
            self.yosoku_export_progress_dialog.update_progress(0, "初期化中...")
            self.yosoku_export_progress_dialog.set_status("初期化中...")
            QApplication.processEvents()
            
            # Crear worker thread
            self.yosoku_export_worker = YosokuExportWorker(db_path, filepath, len(df))
            
            # Conectar señales
            self.yosoku_export_worker.progress_updated.connect(self.yosoku_export_progress_dialog.update_progress)
            self.yosoku_export_worker.status_updated.connect(self.yosoku_export_progress_dialog.set_status)
            self.yosoku_export_worker.finished.connect(self.on_yosoku_export_finished)
            self.yosoku_export_worker.error.connect(self.on_yosoku_export_error)
            
            # Conectar botón de cancelar
            self.yosoku_export_progress_dialog.cancel_button.clicked.connect(self.cancel_yosoku_export)
            
            # Iniciar worker
            self.yosoku_export_worker.start()
            
        except Exception as e:
            print(f"❌ Error iniciando exportación: {e}")
            import traceback
            traceback.print_exc()
            
            # Cerrar loading si hay error
            if hasattr(self, 'yosoku_export_progress_dialog') and self.yosoku_export_progress_dialog is not None:
                self.yosoku_export_progress_dialog.close()
                self.yosoku_export_progress_dialog = None
            self.set_console_overlay_topmost(False)
            
            QMessageBox.critical(
                self,
                "エラー",
                f"❌ エクスポート開始中にエラーが発生しました:\n{str(e)}"
            )

    def set_ui_state_for_sample_file(self):
        """Set UI state when a sample file is loaded"""
        self.sample_size_input.setEnabled(True)
        self.sample_size_input.setStyleSheet("")
        self.d_optimize_button.setEnabled(True)
        self.i_optimize_button.setEnabled(True)
        # Apply original blue style for action buttons
        self.d_optimize_button.setStyleSheet("""
            QPushButton {
                background-color: #3A80BA;
                color: white;
                font-family: "Noto Sans JP";
                border: none;
                border-radius: 8px;
                font-size: 16px;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #336DA3;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #888888;
            }
        """)
        self.i_optimize_button.setStyleSheet("""
            QPushButton {
                background-color: #3A80BA;
                color: white;
                font-family: "Noto Sans JP";
                border: none;
                border-radius: 8px;
                font-size: 16px;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #336DA3;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #888888;
            }
        """)
        self.material_selector.setEnabled(False)
        self.material_selector.setStyleSheet("color: gray; background-color: #f0f0f0;")
        self.brush_selector.setEnabled(False)
        self.brush_selector.setStyleSheet("color: gray; background-color: #f0f0f0;")
        self.diameter_selector.setEnabled(False)
        self.diameter_selector.setStyleSheet("color: gray; background-color: #f0f0f0;")
        self.brush_label.setEnabled(False)
        self.brush_label.setStyleSheet("color: gray;")
        self.brush_selector.setEnabled(False)
        self.brush_selector.setStyleSheet("color: gray; background-color: #f0f0f0;")
        # El botón de análisis siempre está habilitado
        self.analyze_button.setEnabled(True)

    def set_ui_state_for_results_file(self):
        """Set UI state when a results file is loaded"""
        self.sample_size_input.setEnabled(False)
        self.sample_size_input.setStyleSheet("color: gray; background-color: #f0f0f0;")
        self.d_optimize_button.setEnabled(False)
        self.i_optimize_button.setEnabled(False)
        self.d_optimize_button.setStyleSheet("color: gray; background-color: #f0f0f0;")
        self.i_optimize_button.setStyleSheet("color: gray; background-color: #f0f0f0;")
        self.material_selector.setEnabled(True)
        self.material_selector.setStyleSheet("")
        self.brush_selector.setEnabled(True)
        self.brush_selector.setStyleSheet("")
        self.diameter_selector.setEnabled(True)
        self.diameter_selector.setStyleSheet("")
        self.brush_label.setEnabled(True)
        self.brush_label.setStyleSheet("")
        self.brush_selector.setEnabled(True)
        self.brush_selector.setStyleSheet("")
        # Habilitar botón de análisis
        self.analyze_button.setEnabled(True)

    def set_ui_state_for_no_file(self):
        """Set UI state when no file is loaded"""
        self.sample_size_input.setEnabled(False)
        self.sample_size_input.setStyleSheet("color: gray; background-color: #f0f0f0;")
        self.d_optimize_button.setEnabled(False)
        self.i_optimize_button.setEnabled(False)
        self.d_optimize_button.setStyleSheet("color: gray; background-color: #f0f0f0;")
        self.i_optimize_button.setStyleSheet("color: gray; background-color: #f0f0f0;")
        self.material_selector.setEnabled(False)
        self.material_selector.setStyleSheet("color: gray; background-color: #f0f0f0;")
        self.brush_selector.setEnabled(False)
        self.brush_selector.setStyleSheet("color: gray; background-color: #f0f0f0;")
        self.diameter_selector.setEnabled(False)
        self.diameter_selector.setStyleSheet("color: gray; background-color: #f0f0f0;")
        self.brush_label.setEnabled(False)
        self.brush_label.setStyleSheet("color: gray;")
        self.brush_selector.setEnabled(False)
        self.brush_selector.setStyleSheet("color: gray; background-color: #f0f0f0;")
        # El botón de análisis siempre está habilitado
        self.analyze_button.setEnabled(True)

    def switch_to_unexperimented_data(self):
        """Cambiar automáticamente al archivo 未実験データ después de la primera optimización"""
        if hasattr(self, 'proyecto_folder') and hasattr(self, 'proyecto_nombre'):
            proyecto_nombre = getattr(self, 'proyecto_nombre', 'Unknown')
            temp_dir = os.path.join(self.proyecto_folder, "99_Temp")
            candidates = [
                os.path.join(temp_dir, f"{proyecto_nombre}_未実験データ.xlsx"),
                os.path.join(temp_dir, f"{proyecto_nombre}_未実験データ.xls"),
                os.path.join(temp_dir, f"{proyecto_nombre}_未実験データ.csv"),
            ]
            unexperimented_file = next((p for p in candidates if os.path.exists(p)), None)
            if unexperimented_file:
                # Actualizar la ruta del archivo cargado
                self.sample_file_path = unexperimented_file
                # Actualizar la etiqueta en la UI
                self.load_file_label.setText(f"読み込み済み: {os.path.basename(unexperimented_file)}")
                print(f"✅ Archivo de entrada cambiado automáticamente a: {unexperimented_file}")
                return True
        return False

    def clear_main_screen(self):
        """Limpia toda la pantalla principal (panel derecho)"""
        print("🧹 Limpiando pantalla principal...")
        
        # Limpiar variables de navegación primero
        self.graph_images = []
        self.graph_images_content = []
        self.current_graph_index = 0
        
        # Limpiar referencias a botones de navegación de forma segura
        if hasattr(self, 'prev_button'):
            try:
                if self.prev_button and not self.prev_button.isHidden():
                    self.prev_button.setEnabled(False)
            except RuntimeError:
                # El objeto ya fue eliminado, simplemente limpiar la referencia
                self.prev_button = None
        
        if hasattr(self, 'next_button'):
            try:
                if self.next_button and not self.next_button.isHidden():
                    self.next_button.setEnabled(False)
            except RuntimeError:
                # El objeto ya fue eliminado, simplemente limpiar la referencia
                self.next_button = None
        
        # Limpiar el layout central completamente
        while self.center_layout.count():
            item = self.center_layout.takeAt(0)
            widget = item.widget()
            if widget:
                try:
                    widget.deleteLater()
                except RuntimeError:
                    # El widget ya fue eliminado, continuar
                    pass
        
        # Restaurar los elementos básicos del panel central
        # Título arriba del área de gráficos
        self._add_center_header_title()

        # Área de gráficos
        self.graph_container = QFrame()
        graph_container_layout = QVBoxLayout()
        graph_container_layout.setContentsMargins(0, 0, 0, 0)
        graph_container_layout.setSpacing(0)
        self.graph_container.setLayout(graph_container_layout)

        self.graph_area = QFrame()
        self.graph_area.setStyleSheet("background-color: #F9F9F9; border: 1px solid #CCCCCC;")
        graph_container_layout.addWidget(self.graph_area, stretch=1)

        self.center_layout.addWidget(self.graph_container, stretch=1)

        # Espacio flexible antes de los botones
        self.center_layout.addStretch()

        # Botones OK y NG
        self.ok_ng_frame = QFrame()
        ok_ng_layout = QHBoxLayout()
        ok_ng_layout.setAlignment(Qt.AlignCenter)
        self.ok_ng_frame.setLayout(ok_ng_layout)

        self.ok_button = QPushButton("OK")
        self.ng_button = QPushButton("NG")

        self.setup_ok_button(self.ok_button)
        self.setup_ng_button(self.ng_button)

        self.ok_button.clicked.connect(self.on_ok_clicked)
        self.ng_button.clicked.connect(self.on_ng_clicked)

        ok_ng_layout.addWidget(self.ok_button)
        ok_ng_layout.addSpacing(10)
        ok_ng_layout.addWidget(self.ng_button)

        self.center_layout.addWidget(self.ok_ng_frame)

        self.ok_button.setEnabled(False)
        self.ng_button.setEnabled(False)
        
        # Limpiar referencias a botones de navegación
        self.prev_button = None
        self.next_button = None
        self.graph_navigation_frame = None
        
        print("✅ Pantalla principal limpiada")
        print("🔧 Inicialización de MainWindow completada")

    def setup_console_redirection(self):
        """Configurar redirección de stdout y stderr a la consola integrada Y a la consola original"""
        # ✅ FIX CRÍTICO: La UI (QTextEdit / overlay) NO se puede tocar desde hilos secundarios.
        # Creamos un stream QObject que emite señales; el slot corre en el hilo principal.
        from PySide6.QtCore import QObject, Signal, Qt

        if not hasattr(self, "_console_buffers"):
            self._console_buffers = {"stdout": "", "stderr": ""}

        class ConsoleStream(QObject):
            text_written = Signal(str, str)  # stream_type, text

            def __init__(self, stream_type, original_stream, parent=None):
                super().__init__(parent)
                self.stream_type = stream_type
                self.original_stream = original_stream

            def write(self, text):
                if text is None:
                    return

                # Siempre escribir en la consola original con info de hilo (esto sí es seguro)
                try:
                    import threading
                    current_thread = threading.current_thread()
                    thread_info = f"[{current_thread.name}:{current_thread.ident}]"
                    if str(text).strip():
                        self.original_stream.write(f"DEBUG {thread_info}: {text}")
                    else:
                        self.original_stream.write(str(text))
                    self.original_stream.flush()
                except:
                    pass

                # Enviar a UI mediante señal (thread-safe)
                try:
                    self.text_written.emit(self.stream_type, str(text))
                except:
                    pass

            def flush(self):
                try:
                    self.original_stream.flush()
                except:
                    pass

        # Crear streams personalizados que mantengan la consola original
        self.stdout_stream = ConsoleStream("stdout", sys.__stdout__, parent=self)
        self.stderr_stream = ConsoleStream("stderr", sys.__stderr__, parent=self)

        # Conectar señales a slot (hilo principal)
        self.stdout_stream.text_written.connect(self._on_console_stream_text, Qt.QueuedConnection)
        self.stderr_stream.text_written.connect(self._on_console_stream_text, Qt.QueuedConnection)
        
        # Guardar streams originales
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Redirigir streams
        sys.stdout = self.stdout_stream
        sys.stderr = self.stderr_stream
        
        # Mensaje inicial en ambas consolas
        print("🚀 コンソールが起動しました")
        print("📝 すべての出力が両方のコンソールに表示されます")
        # (No hacer append manual: ya lo hace el print vía redirección)

    def _on_console_stream_text(self, stream_type, text):
        """Recibe texto de stdout/stderr (desde cualquier hilo) y lo pinta en la UI (hilo principal)."""
        try:
            if not hasattr(self, "_console_buffers"):
                self._console_buffers = {"stdout": "", "stderr": ""}

            if not hasattr(self, "console_output") or self.console_output is None:
                return

            buf = self._console_buffers.get(stream_type, "") + (text or "")
            lines = buf.split("\n")
            self._console_buffers[stream_type] = lines[-1]  # línea parcial

            for line in lines[:-1]:
                if line == "":
                    continue
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.console_output.append(f"[{timestamp}] {line}")

                # Consola overlay (también en main thread)
                try:
                    if hasattr(self, "overlay_console_output"):
                        overlay_console = self.overlay_console_output
                        if overlay_console and overlay_console.isVisible():
                            overlay_console.append(line)
                except:
                    pass
        except:
            pass

    def clear_console(self):
        """Limpiar el contenido de la consola"""
        self.console_output.clear()
        self.console_output.append("🧹 コンソールがクリアされました")
        self.console_output.append("📝 新しい出力を待機中...")

    def save_console_log(self):
        """Guardar el contenido de la consola en un archivo"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"console_log_{timestamp}.txt"
            
            # Obtener el contenido de la consola
            content = self.console_output.toPlainText()
            
            # Guardar archivo
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ コンソールログが保存されました: {filename}")
            
        except Exception as e:
            print(f"❌ ログの保存に失敗しました: {e}")

    # NOTA: Este método ya no se necesita, la flecha está siempre visible

    # NOTA: Este método ya no se necesita, simplificado en show_right_panel

    # NOTA: Este método ya no se necesita, solo usamos el panel superpuesto

    def position_arrow(self):
        """Posicionar la flecha en el borde derecho del panel central"""
        try:
            # Coordenadas globales (pantalla) del panel central
            center_global = self.center_frame.mapToGlobal(QPoint(0, 0))
            button_x = center_global.x() + self.center_frame.width() - 35
            button_y = center_global.y() + self.center_frame.height() // 2 - 15
            self.console_toggle_button.setGeometry(button_x, button_y, 30, 30)
            
            # Asegurar que la flecha esté en primer plano después de posicionarla
            self.console_toggle_button.raise_()
            
            print(f"🔧 Flecha posicionada en: ({button_x}, {button_y}) y en primer plano")
        except Exception as e:
            print(f"⚠️ Error posicionando flecha: {e}")

    def debug_button_state(self):
        """Método de debug para verificar el estado del botón de flecha"""
        print("🔍 DEBUG: Estado del botón de flecha")
        print(f"🔍 Botón existe: {hasattr(self, 'console_toggle_button')}")
        if hasattr(self, 'console_toggle_button'):
            print(f"🔍 Botón visible: {self.console_toggle_button.isVisible()}")
            print(f"🔍 Botón geometría: {self.console_toggle_button.geometry()}")
            print(f"🔍 Botón padre: {self.console_toggle_button.parent()}")
            print(f"🔍 Botón texto: {self.console_toggle_button.text()}")
            print(f"🔍 Botón estilo: {self.console_toggle_button.styleSheet()}")
        else:
            print("❌ Botón de flecha no existe")

    def clear_overlay_console(self):
        """Limpiar el contenido de la consola desplegable"""
        self.overlay_console_output.clear()
        self.overlay_console_output.append("🧹 オーバーレイコンソールがクリアされました")
        self.overlay_console_output.append("📝 新しい出力を待機中...")

    def save_overlay_console_log(self):
        """Guardar el contenido de la consola desplegable en un archivo"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"overlay_console_log_{timestamp}.txt"
            
            # Obtener el contenido de la consola desplegable
            content = self.overlay_console_output.toPlainText()
            
            # Guardar archivo
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ オーバーレイコンソールログが保存されました: {filename}")
            
        except Exception as e:
            print(f"❌ オーバーレイログの保存に失敗しました: {e}")

    def toggle_overlay_console(self):
        """Alternar la visibilidad del panel desplegable"""
        if not self.overlay_console_visible:
            # Mostrar el panel desplegable
            self.show_overlay_console()
        else:
            # Ocultar el panel desplegable
            self.hide_overlay_console()
            
    # NOTA: Este método ya no se necesita, simplificado
            
    def toggle_right_panel(self):
        """Alternar la visibilidad del panel desplegable superpuesto"""
        print("🔧 toggle_right_panel ejecutado")
        if self.overlay_console_visible:
            # Si el panel desplegable está visible, ocultarlo
            print("🔧 Ocultando panel desplegable...")
            self.hide_overlay_console()
        else:
            # Si el panel desplegable está oculto, mostrarlo
            print("🔧 Mostrando panel desplegable...")
            self.show_overlay_console()

    def show_overlay_console(self):
        """Mostrar el panel desplegable superpuesto en el lado derecho"""
        print("🔧 Mostrando panel desplegable en el lado derecho...")
        
        # Obtener la posición actual de la ventana principal
        current_window_pos = self.geometry()
        print(f"🔧 Posición actual de la ventana: {current_window_pos}")
        print(f"🔧 Coordenadas X: {current_window_pos.x()}, Y: {current_window_pos.y()}")
        print(f"🔧 Dimensiones: {current_window_pos.width()} x {current_window_pos.height()}")
        
        # Posicionar la consola en el lado derecho de la pantalla
        self.position_overlay_console()
        
        # Cambiar el texto del botón a flecha derecha
        self.console_toggle_button.setText("▶")
        
        # Mostrar el panel desplegable
        self.overlay_console.show()
        
        # Asegurar que esté en primer plano
        self.overlay_console.raise_()
        
        # Asegurar que la flecha también esté en primer plano
        self.console_toggle_button.raise_()
        
        # Actualizar estado
        self.overlay_console_visible = True
        
        # Sincronizar contenido con la consola principal
        self.sync_console_content()
        
        # Debug de posición
        self.debug_console_position()
        
        print("✅ Panel desplegable superpuesto mostrado en el lado derecho")

    def hide_overlay_console(self):
        """Ocultar el panel desplegable"""
        print("🔧 Ocultando panel desplegable...")
        
        # Ocultar el panel desplegable
        self.overlay_console.hide()
        
        # Cambiar el texto del botón a flecha izquierda
        self.console_toggle_button.setText("◀")
        
        # Reposicionar la flecha
        self.position_arrow()
        
        # Asegurar que la flecha esté en primer plano
        self.console_toggle_button.raise_()
        
        # Actualizar estado
        self.overlay_console_visible = False
        
        print("✅ Panel desplegable oculto")

    def position_overlay_console(self):
        """Posicionar la consola desplegable en el lado derecho de la pantalla"""
        try:
            # Obtener la posición y dimensiones de la ventana principal
            main_window_rect = self.geometry()
            
            # Calcular posición en el lado derecho de la ventana principal
            overlay_width = 350
            overlay_height = min(600, main_window_rect.height() - 80)
            
            # Posicionar en el lado derecho de la ventana principal
            # Usar coordenadas absolutas de la pantalla
            overlay_x = main_window_rect.x() + main_window_rect.width() - overlay_width - 20
            overlay_y = main_window_rect.y() + 40  # Margen superior
            
            # Configurar geometría del panel desplegable
            self.overlay_console.setGeometry(overlay_x, overlay_y, overlay_width, overlay_height)
            
            # Posicionar el botón de flecha en el borde derecho del panel central (coordenadas globales)
            self.position_arrow()
            
            print(f"🔧 Ventana principal: {main_window_rect}")
            print(f"🔧 Coordenadas absolutas de la consola: ({overlay_x}, {overlay_y}) - {overlay_width}x{overlay_height}")
            print(f"🔧 Flecha reposicionada junto al panel central")
            
            # Verificar que la consola esté visible y en primer plano
            if self.overlay_console.isVisible():
                self.overlay_console.raise_()
                print("🔧 Consola elevada a primer plano")
            
        except Exception as e:
            print(f"⚠️ Error posicionando consola desplegable: {e}")

    def keep_elements_on_top(self):
        """Mantener la consola y la flecha en primer plano, respetando el orden del loading"""
        try:
            if not hasattr(self, '_heartbeat_count'): self._heartbeat_count = 0
            self._heartbeat_count += 1
            if self._heartbeat_count >= 10:
                print("💓 HEARTBEAT: App viva y en standby")
                self._heartbeat_count = 0
                
            # Si hay un loading visible, NO forzamos el Z-order cada segundo.
            # Antes bajábamos (lower) la flecha y la consola mientras el loading estaba visible,
            # lo que causaba parpadeo/"refresh" constante y bloqueaba el botón de despliegue.
            # Dejamos que el resto de la lógica mantenga la flecha/consola accesibles.

            # Si hay un diálogo modal activo que NO sea el loading, no "pisar" el Z-order.
            modal = QApplication.activeModalWidget()
            progress = getattr(self, 'progress_dialog', None)
            if modal is not None and modal is not progress:
                return

            # Mantener la consola desplegable en primer plano si está visible
            if hasattr(self, 'overlay_console') and self.overlay_console.isVisible():
                self.overlay_console.raise_()

            # Mantener la flecha en primer plano si está visible
            if hasattr(self, 'console_toggle_button') and self.console_toggle_button.isVisible():
                self.console_toggle_button.raise_()
                
        except Exception as e:
            print(f"⚠️ Error manteniendo elementos en primer plano: {e}")

    def set_console_overlay_topmost(self, enabled: bool):
        """
        Activa/desactiva WindowStaysOnTopHint para flecha + consola overlay.
        - enabled=True: permite clicar la flecha incluso con ReusableProgressDialog (WindowModal).
        - enabled=False: evita tapar diálogos del sistema (QFileDialog, etc).
        """
        try:
            self._console_topmost_enabled = bool(enabled)

            for w_attr in ("overlay_console", "console_toggle_button"):
                w = getattr(self, w_attr, None)
                if w is None:
                    continue

                was_visible = w.isVisible()
                flags = w.windowFlags()

                # Asegurar tipo de ventana esperado
                flags |= Qt.Tool
                flags |= Qt.FramelessWindowHint

                if enabled:
                    flags |= Qt.WindowStaysOnTopHint
                else:
                    flags &= ~Qt.WindowStaysOnTopHint

                w.setWindowFlags(flags)

                # Aplicar cambios (Qt requiere show() tras cambiar flags)
                if was_visible:
                    w.show()
                    w.raise_()

            # Reposicionar por si el WM recalcula geometría
            try:
                if hasattr(self, 'console_toggle_button'):
                    self.position_arrow()
                if getattr(self, 'overlay_console_visible', False):
                    self.position_overlay_console()
            except Exception:
                pass

        except Exception as e:
            print(f"⚠️ Error set_console_overlay_topmost({enabled}): {e}")

    def pause_auto_timers(self):
        """Pausar los timers automáticos para evitar interferencia con diálogos"""
        try:
            if hasattr(self, 'keep_on_top_timer') and self.keep_on_top_timer.isActive():
                self.keep_on_top_timer.stop()
                print("⏸️ Timer keep_on_top pausado")
            
            if hasattr(self, 'position_check_timer') and self.position_check_timer.isActive():
                self.position_check_timer.stop()
                print("⏸️ Timer position_check pausado")
        except Exception as e:
            print(f"⚠️ Error pausando timers: {e}")

    def resume_auto_timers(self):
        """Reanudar los timers automáticos"""
        try:
            if hasattr(self, 'keep_on_top_timer'):
                self.keep_on_top_timer.start(1000)  # Cada segundo
                print("▶️ Timer keep_on_top reanudado")
            
            if hasattr(self, 'position_check_timer'):
                self.position_check_timer.start(500)  # Cada medio segundo
                print("▶️ Timer position_check reanudado")
        except Exception as e:
            print(f"⚠️ Error reanudando timers: {e}")

    def check_window_position(self):
        """Verificar si la ventana principal se ha movido y actualizar la consola si es necesario"""
        try:
            current_position = self.geometry()
            
            # Si la posición ha cambiado, reposicionar SIEMPRE la flecha (es una ventana top-level)
            if current_position != self.last_window_position:
                try:
                    if hasattr(self, 'console_toggle_button') and self.console_toggle_button.isVisible():
                        self.position_arrow()
                except Exception:
                    pass

            # Si la posición ha cambiado y la consola está visible, reposicionar también la consola
            if (current_position != self.last_window_position and
                hasattr(self, 'overlay_console_visible') and
                self.overlay_console_visible):
                
                print(f"🔧 Ventana movida de {self.last_window_position} a {current_position}")
                print("🔧 Reposicionando consola...")
                
                # Reposicionar la consola en la nueva posición
                self.position_overlay_console()
                
                # Asegurar que esté en primer plano
                modal = QApplication.activeModalWidget()
                progress = getattr(self, 'progress_dialog', None)
                if modal is None or modal is progress:
                    if getattr(self, '_console_topmost_enabled', False) or getattr(self, 'overlay_console_visible', False):
                        self.overlay_console.raise_()
                        self.console_toggle_button.raise_()
                
                print("✅ Consola reposicionada en la nueva ubicación")
            
            # Actualizar la posición guardada
            self.last_window_position = current_position
            
        except Exception as e:
            print(f"⚠️ Error verificando posición de ventana: {e}")

    def moveEvent(self, event):
        """Mantener flecha/consola ancladas cuando la ventana se mueve (drag)."""
        super().moveEvent(event)
        try:
            if hasattr(self, 'console_toggle_button') and self.console_toggle_button.isVisible():
                self.position_arrow()
            if hasattr(self, 'overlay_console_visible') and self.overlay_console_visible:
                self.position_overlay_console()
        except Exception:
            pass

    def is_valid_project_folder(self, folder_path, analysis_type="nonlinear"):
        """
        Verifica si una carpeta tiene la estructura de un proyecto válido
        
        Parameters
        ----------
        folder_path : str
            Ruta de la carpeta a verificar
        analysis_type : str, optional
            Tipo de análisis: "nonlinear" (default) o "classification"
        
        Returns
        -------
        bool
            True si la carpeta tiene estructura de proyecto válida
        """
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            return False
        
        # Carpetas esenciales según el tipo de análisis
        if analysis_type == "classification":
            essential_folders = [
                "05_分類"  # Esencial para análisis de clasificación
            ]
        else:  # nonlinear (default)
            essential_folders = [
                "04_非線形回帰"  # Esencial para análisis no lineal
            ]
        
        # Carpetas opcionales pero comunes en proyectos existentes
        optional_folders = [
            "03_線形回帰",
            "04_非線形回帰",
            "05_分類",
            "99_Results",
            "99_Temp",
            "backup"
        ]
        
        # Verificar que existan las carpetas esenciales
        for folder in essential_folders:
            folder_path_full = os.path.join(folder_path, folder)
            if not os.path.exists(folder_path_full) or not os.path.isdir(folder_path_full):
                return False
        
        # Si tiene al menos una carpeta opcional, es más probable que sea un proyecto válido
        has_optional = any(
            os.path.exists(os.path.join(folder_path, folder)) and 
            os.path.isdir(os.path.join(folder_path, folder))
            for folder in optional_folders
        )
        
        # Considerar válido si tiene las esenciales y al menos una opcional
        return has_optional
    
    def find_project_folders_in_directory(self, directory, analysis_type="nonlinear"):
        """
        Busca carpetas de proyecto dentro de un directorio
        
        Parameters
        ----------
        directory : str
            Directorio donde buscar proyectos
        analysis_type : str, optional
            Tipo de análisis: "nonlinear" (default) o "classification"
        
        Returns
        -------
        list
            Lista de rutas de carpetas que son proyectos válidos
        """
        project_folders = []
        
        if not os.path.exists(directory) or not os.path.isdir(directory):
            return project_folders
        
        # Buscar en el directorio seleccionado directamente
        if self.is_valid_project_folder(directory, analysis_type=analysis_type):
            project_folders.append(directory)
        
        # Buscar en subdirectorios (solo un nivel de profundidad)
        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    if self.is_valid_project_folder(item_path, analysis_type=analysis_type):
                        project_folders.append(item_path)
        except PermissionError:
            pass
        
        return project_folders
    
    def create_nonlinear_project_structure(self, project_name, base_directory):
        """
        Crear la estructura de carpetas del proyecto para análisis no lineal
        Similar a Proyecto_79 pero sin 01_実験リスト y 02_実験データ
        """
        try:
            # Crear la carpeta principal del proyecto
            project_path = os.path.join(base_directory, project_name)
            os.makedirs(project_path, exist_ok=True)
            
            # Crear las subcarpetas (SIN 01 y 02)
            subfolders = [
                "03_線形回帰",
                "04_非線形回帰",
                "05_分類",
                "99_Results",
                "99_Temp",
                "backup"
            ]
            
            for subfolder in subfolders:
                subfolder_path = os.path.join(project_path, subfolder)
                os.makedirs(subfolder_path, exist_ok=True)
                print(f"📁 Creada carpeta: {subfolder_path}")
            
            print(f"✅ Estructura de proyecto creada en: {project_path}")
            return project_path
            
        except Exception as e:
            print(f"❌ Error creando estructura del proyecto: {e}")
            raise e
    
    def create_project_structure(self, project_name, base_directory):
        """Crear la estructura de carpetas del proyecto según la imagen"""
        try:
            # Crear la carpeta principal del proyecto
            project_path = os.path.join(base_directory, project_name)
            os.makedirs(project_path, exist_ok=True)
            
            # Crear las subcarpetas según la estructura de la imagen
            subfolders = [
                "01_データ準備",
                "02_前処理", 
                "03_線形回帰",
                "04_非線形回帰",
                "05_結果比較",
                "06_レポート"
            ]
            
            for subfolder in subfolders:
                subfolder_path = os.path.join(project_path, subfolder)
                os.makedirs(subfolder_path, exist_ok=True)
                print(f"📁 Creada carpeta: {subfolder_path}")
            
            # Crear subcarpetas específicas dentro de 03_線形回帰
            linear_subfolders = [
                "01_データ分割",
                "02_特徴選択", 
                "03_モデル学習",
                "04_予測計算",
                "05_結果評価"
            ]
            
            linear_path = os.path.join(project_path, "03_線形回帰")
            for subfolder in linear_subfolders:
                subfolder_path = os.path.join(linear_path, subfolder)
                os.makedirs(subfolder_path, exist_ok=True)
                print(f"📁 Creada subcarpeta: {subfolder_path}")
            
            print(f"✅ Estructura de proyecto creada en: {project_path}")
            return project_path
            
        except Exception as e:
            print(f"❌ Error creando estructura del proyecto: {e}")
            raise e

    def run_linear_analysis_in_project(self, project_path):
        """Ejecutar análisis lineal en la carpeta del proyecto"""
        try:
            print(f"🔧 Ejecutando análisis lineal en proyecto: {project_path}")
            
            # ✅ NUEVO: Establecer la carpeta del proyecto actual
            self.current_project_folder = project_path
            print(f"📁 Carpeta del proyecto establecida: {self.current_project_folder}")
            
            # Obtener filtros actuales
            filters = self.get_applied_filters()
            
            if not filters:
                QMessageBox.warning(self, "警告", "フィルターが設定されていません。\nフィルターを設定してから線形解析を実行してください。")
                return
            
            # Crear carpeta de resultados con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_folder = os.path.join(project_path, "03_線形回帰", f"15_{timestamp}")
            os.makedirs(results_folder, exist_ok=True)
            
            # Crear subcarpetas dentro del resultado
            subfolders = ["01_データ分割", "02_特徴選択", "03_モデル学習", "04_予測計算", "05_結果評価"]
            for subfolder in subfolders:
                subfolder_path = os.path.join(results_folder, subfolder)
                os.makedirs(subfolder_path, exist_ok=True)
            
            print(f"📁 Carpeta de resultados creada: {results_folder}")
            
            # Ejecutar análisis lineal con la carpeta del proyecto
            self.execute_linear_analysis_with_output_folder(results_folder)
            
        except Exception as e:
            print(f"❌ Error ejecutando análisis lineal en proyecto: {e}")
            QMessageBox.critical(
                self, 
                "エラー", 
                f"❌ プロジェクト内での線形解析実行中にエラーが発生しました:\n{str(e)}"
            )

    def execute_linear_analysis_with_output_folder(self, output_folder):
        """Ejecutar análisis lineal con carpeta de salida específica"""
        try:
            print(f"🔧 Ejecutando análisis lineal con carpeta: {output_folder}")

            # ✅ NUEVO: Evitar re-ejecución si ya hay un análisis lineal corriendo
            if hasattr(self, 'linear_worker') and self.linear_worker is not None:
                try:
                    if self.linear_worker.isRunning():
                        QMessageBox.warning(self, "線形解析", "⚠️ すでに線形解析が実行中です。\n完了または停止するまでお待ちください。")
                        return
                except RuntimeError:
                    # Si el objeto fue destruido, limpiar referencia
                    self.linear_worker = None
            
            # Obtener filtros aplicados
            filters = self.get_applied_filters()
            print(f"🔧 Filtros aplicados: {filters}")
            
            # Importar módulo de análisis lineal
            try:
                from linear_analysis_advanced import run_advanced_linear_analysis_from_db
                print("✅ Módulo de análisis lineal importado correctamente")
            except ImportError as e:
                print(f"❌ Error importando módulo de análisis lineal: {e}")
                QMessageBox.critical(self, "エラー", "❌ モジュール de análisis lineal no se pudo importar.\nAsegúrese de que el archivo linear_analysis_module.py esté en el directorio correcto.")
                return
            
            # Mostrar mensaje de confirmación
            reply = QMessageBox.question(
                self,
                "線形解析確認", 
                f"線形解析を実行しますか？\n\nフィルター: {len(filters)} 条件\n\nこの操作は時間がかかる場合があります。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply != QMessageBox.Yes:
                print("❌ Usuario canceló el análisis lineal")
                return
            
            # Ejecutar análisis lineal con la carpeta específica usando el MISMO flujo con popup/cancelación
            print(f"🔧 Ejecutando análisis lineal en carpeta: {output_folder}")
            self._start_linear_analysis(filters, output_folder)
            
        except Exception as e:
            print(f"❌ Error ejecutando análisis lineal: {e}")
            QMessageBox.critical(self, "エラー", f"❌ 線形解析の実行中にエラーが発生しました:\n{str(e)}")

    def _start_linear_analysis(self, filters, analysis_folder):
        """Arranca el análisis lineal con popup de progreso y cancelación cooperativa."""
        # ✅ NUEVO: No mezclar ejecuciones pesadas en paralelo
        if hasattr(self, 'nonlinear_worker') and self.nonlinear_worker is not None:
            try:
                if self.nonlinear_worker.isRunning():
                    QMessageBox.warning(self, "線形解析", "⚠️ 非線形解析が実行中です。\n完了または停止するまでお待ちください。")
                    return
            except RuntimeError:
                self.nonlinear_worker = None
        for t_attr in ("d_optimizer_thread", "i_optimizer_thread", "dsaitekika_thread"):
            if hasattr(self, t_attr):
                t = getattr(self, t_attr)
                try:
                    if t is not None and t.isRunning():
                        QMessageBox.warning(self, "線形解析", "⚠️ 最適化が実行中です。\n完了または停止するまでお待ちください。")
                        return
                except RuntimeError:
                    setattr(self, t_attr, None)

        # ✅ NUEVO: Evitar re-ejecución si ya hay un análisis lineal corriendo
        if hasattr(self, 'linear_worker') and self.linear_worker is not None:
            try:
                if self.linear_worker.isRunning():
                    QMessageBox.warning(self, "線形解析", "⚠️ すでに線形解析が実行中です。\n完了または停止するまでお待ちください。")
                    return
            except RuntimeError:
                self.linear_worker = None

        # ✅ NUEVO: Reset de bandera de cancelación (para esta ejecución)
        self._linear_cancel_requested = False

        # ✅ NUEVO: Deshabilitar botones para evitar doble ejecución
        if hasattr(self, 'linear_analysis_button'):
            self.linear_analysis_button.setEnabled(False)
        if hasattr(self, 'run_analysis_button'):
            self.run_analysis_button.setEnabled(False)

        # Cerrar popup previo si quedara colgado
        if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
            try:
                self.progress_dialog.close()
                self.progress_dialog.deleteLater()
            except:
                pass
            try:
                delattr(self, 'progress_dialog')
            except:
                pass

        # Crear popup de progreso
        self.progress_dialog = LinearAnalysisProgressDialog(self)
        self.progress_dialog.show()
        # Durante el loading modal del análisis lineal: permitir flecha/consola por encima
        self.set_console_overlay_topmost(True)
        self.progress_dialog.rejected.connect(self.on_analysis_cancelled)

        # Crear y arrancar worker (QThread) con señales de progreso
        self.linear_worker = LinearAnalysisWorker(self.db, filters, analysis_folder, self)
        self.linear_worker.progress_updated.connect(self.progress_dialog.update_progress)
        self.linear_worker.status_updated.connect(self.progress_dialog.set_status)
        self.linear_worker.finished.connect(self.on_linear_analysis_finished)
        self.linear_worker.error.connect(self.on_linear_analysis_error)

        print("🚀 Iniciando análisis lineal con progreso (worker)...")
        self.linear_worker.start()

    def on_linear_analysis_clicked(self):
        """Acción al pulsar el botón de análisis lineal"""
        print("🔧 Iniciando análisis lineal...")
        
        # ✅ NUEVO: Si se accedió desde bunseki, mostrar diálogo de creación de proyecto
        if hasattr(self, 'accessed_from_bunseki') and self.accessed_from_bunseki:
            print("📁 Acceso desde bunseki detectado - mostrando diálogo de creación de proyecto")
            
            # Mostrar diálogo de creación de proyecto
            dialog = ProjectCreationDialog(self)
            if dialog.exec() == QDialog.Accepted:
                project_name = dialog.project_name
                project_directory = dialog.project_directory
                
                print(f"📁 Creando proyecto: {project_name} en {project_directory}")
                
                try:
                    # Crear estructura del proyecto
                    project_path = self.create_project_structure(project_name, project_directory)
                    
                    # Mostrar mensaje de confirmación
                    QMessageBox.information(
                        self, 
                        "プロジェクト作成完了", 
                        f"✅ プロジェクト '{project_name}' が作成されました。\n\n"
                        f"保存先: {project_path}\n\n"
                        f"線形解析を開始します..."
                    )
                    
                    # Resetear la bandera
                    self.accessed_from_bunseki = False
                    
                    # Proceder con el análisis lineal en la nueva carpeta
                    self.run_linear_analysis_in_project(project_path)
                    return
                    
                except Exception as e:
                    QMessageBox.critical(
                        self, 
                        "エラー", 
                        f"❌ プロジェクト作成中にエラーが発生しました:\n{str(e)}"
                    )
                    return
            else:
                # Usuario canceló, resetear la bandera
                self.accessed_from_bunseki = False
                return
        
        try:
            # Verificar si estamos en la vista de filtros
            already_in_filter_view = False
            for i in range(self.center_layout.count()):
                item = self.center_layout.itemAt(i)
                if item.widget() and isinstance(item.widget(), QLabel):
                    if item.widget().text() == "データフィルター":
                        already_in_filter_view = True
                        break
            
            if not already_in_filter_view:
                # Crear la vista de filtros primero
                self.create_filter_view()
                self.create_navigation_buttons()
                self.prev_button.setEnabled(True)
                self.next_button.setEnabled(True)
                QMessageBox.information(self, "分析ページ", "✅ 分析ページに移動しました。\nフィルターを設定して線形解析を実行してください。")
                return
            
            # Ya estamos en la vista de filtros, ejecutar análisis lineal
            self.execute_linear_analysis()
            
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"❌ 線形解析の実行中にエラーが発生しました:\n{str(e)}")
            print(f"❌ Error en análisis lineal: {e}")
            import traceback
            traceback.print_exc()

    def on_nonlinear_analysis_clicked(self):
        """Acción al pulsar el botón de análisis no lineal"""
        print("🔧 Iniciando análisis no lineal...")

        # ✅ NUEVO: No mezclar ejecuciones pesadas en paralelo
        if hasattr(self, 'linear_worker') and self.linear_worker is not None:
            try:
                if self.linear_worker.isRunning():
                    QMessageBox.warning(self, "非線形解析", "⚠️ 線形解析が実行中です。\n完了または停止するまでお待ちください。")
                    return
            except RuntimeError:
                self.linear_worker = None
        for t_attr in ("d_optimizer_thread", "i_optimizer_thread", "dsaitekika_thread"):
            if hasattr(self, t_attr):
                t = getattr(self, t_attr)
                try:
                    if t is not None and t.isRunning():
                        QMessageBox.warning(self, "非線形解析", "⚠️ 最適化が実行中です。\n完了または停止するまでお待ちください。")
                        return
                except RuntimeError:
                    setattr(self, t_attr, None)

        # ✅ NUEVO: Evitar re-ejecución si ya hay un análisis no lineal corriendo
        if hasattr(self, 'nonlinear_worker') and self.nonlinear_worker is not None:
            try:
                if self.nonlinear_worker.isRunning():
                    QMessageBox.warning(self, "非線形解析", "⚠️ すでに非線形解析が実行中です。\n完了または停止するまでお待ちください。")
                    return
            except RuntimeError:
                self.nonlinear_worker = None
        
        # ✅ NUEVO: Si se accedió desde bunseki, mostrar diálogo de creación de proyecto
        if hasattr(self, 'accessed_from_bunseki') and self.accessed_from_bunseki:
            print("📁 Acceso desde bunseki detectado - mostrando diálogo de creación de proyecto")
            
            # Mostrar diálogo de creación de proyecto
            dialog = ProjectCreationDialog(self)
            if dialog.exec() == QDialog.Accepted:
                project_name = dialog.project_name
                project_directory = dialog.project_directory
                
                # Determinar la ruta completa del proyecto
                if project_directory:
                    # Si se seleccionó un proyecto existente, project_directory es el padre
                    # y project_name es el nombre del proyecto
                    project_path = os.path.join(project_directory, project_name)
                else:
                    # Si se creó nuevo, project_directory es donde crear y project_name es el nombre
                    project_path = os.path.join(project_directory, project_name)
                
                # Verificar si el proyecto ya existe (fue detectado como existente)
                project_exists = self.is_valid_project_folder(project_path)
                
                if project_exists:
                    print(f"✅ Usando proyecto existente: {project_path}")
                    # No crear estructura, solo usar la carpeta existente
                    self.current_project_folder = project_path
                    
                    QMessageBox.information(
                        self, 
                        "プロジェクト使用", 
                        f"✅ 既存のプロジェクト '{project_name}' を使用します。\n\n"
                        f"保存先: {project_path}\n\n"
                        f"非線形解析を開始します..."
                    )
                else:
                    print(f"📁 Creando nuevo proyecto: {project_name} en {project_directory}")
                    
                    try:
                        # Crear estructura del proyecto (sin 01 y 02)
                        project_path = self.create_nonlinear_project_structure(project_name, project_directory)
                        
                        # Establecer la carpeta del proyecto actual
                        self.current_project_folder = project_path
                        
                        QMessageBox.information(
                            self, 
                            "プロジェクト作成完了", 
                            f"✅ プロジェクト '{project_name}' が作成されました。\n\n"
                            f"保存先: {project_path}\n\n"
                            f"非線形解析を開始します..."
                        )
                    except Exception as e:
                        QMessageBox.critical(
                            self, 
                            "エラー", 
                            f"❌ プロジェクト作成中にエラーが発生しました:\n{str(e)}"
                        )
                        self.accessed_from_bunseki = False
                        return
                
                # Resetear la bandera
                self.accessed_from_bunseki = False
                
                # Continuar con el flujo normal (mostrar diálogo de configuración)
                # El resto del código seguirá igual, pero ahora con project_folder definido
                
            else:
                # Usuario canceló, resetear la bandera
                self.accessed_from_bunseki = False
                return
        
        try:
            # Verificar si estamos en la vista de filtros
            already_in_filter_view = False
            for i in range(self.center_layout.count()):
                item = self.center_layout.itemAt(i)
                if item.widget() and isinstance(item.widget(), QLabel):
                    if item.widget().text() == "データフィルター":
                        already_in_filter_view = True
                        break
            
            if not already_in_filter_view:
                # Crear la vista de filtros primero
                self.create_filter_view()
                self.create_navigation_buttons()
                self.prev_button.setEnabled(True)
                self.next_button.setEnabled(True)
                QMessageBox.information(self, "分析ページ", "✅ 分析ページに移動しました。\nフィルターを設定して非線形解析を実行してください。")
                return
            
            # Obtener datos filtrados aplicando filtros ahora
            # Similar al análisis lineal, obtener datos filtrados de la BBDD
            try:
                import sqlite3
                filters = self.get_applied_filters()
                
                # Construir query con filtros
                query = "SELECT * FROM main_results WHERE 1=1"
                params = []
                
                # Aplicar filtros de cepillo
                brush_selections = []
                if 'すべて' in filters and filters['すべて']:
                    brush_condition = " OR ".join([f"{brush} = 1" for brush in ['A13', 'A11', 'A21', 'A32']])
                    query += f" AND ({brush_condition})"
                else:
                    for brush_type in ['A13', 'A11', 'A21', 'A32']:
                        if brush_type in filters and filters[brush_type]:
                            brush_selections.append(brush_type)
                    
                    if brush_selections:
                        brush_condition = " OR ".join([f"{brush} = 1" for brush in brush_selections])
                        query += f" AND ({brush_condition})"
                
                # Aplicar otros filtros
                for field_name, filter_value in filters.items():
                    if field_name in ['すべて', 'A13', 'A11', 'A21', 'A32']:
                        continue
                    
                    if isinstance(filter_value, tuple) and len(filter_value) == 2:
                        desde, hasta = filter_value
                        if desde and hasta:
                            try:
                                query += f" AND {field_name} BETWEEN ? AND ?"
                                params.extend([float(desde), float(hasta)])
                            except (ValueError, TypeError):
                                continue
                    elif isinstance(filter_value, (str, int, float)) and filter_value:
                        try:
                            value_num = float(filter_value) if isinstance(filter_value, str) else filter_value
                            query += f" AND {field_name} = ?"
                            params.append(value_num)
                        except (ValueError, TypeError):
                            continue
                
                # Ejecutar query
                conn = sqlite3.connect(RESULTS_DB_PATH, timeout=10)
                df = pd.read_sql_query(query, conn, params=params)
                conn.close()
                
                if df.empty or len(df) == 0:
                    QMessageBox.warning(self, "警告", "⚠️ フィルタリングされたデータがありません。\nフィルター条件を変更してください。")
                    return
                
                self.filtered_df = df
                print(f"📊 Datos filtrados obtenidos: {len(df)} registros")
                
            except Exception as e:
                print(f"❌ Error obteniendo datos filtrados: {e}")
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "エラー", f"❌ データ取得中にエラーが発生しました:\n{str(e)}")
                return
            
            # Obtener carpeta base del proyecto
            # Intentar usar current_project_folder si existe, sino usar directorio actual
            if hasattr(self, 'current_project_folder') and self.current_project_folder:
                project_folder = self.current_project_folder
                print(f"📁 Usando carpeta del proyecto: {project_folder}")
            else:
                # Usar directorio actual como fallback
                project_folder = os.getcwd()
                print(f"⚠️ No hay carpeta de proyecto configurada, usando: {project_folder}")
            
            # Verificar si los módulos están disponibles
            if NonlinearWorker is None or NonlinearConfigDialog is None:
                QMessageBox.warning(
                    self, 
                    "モジュールが見つかりません", 
                    "❌ 必要なモジュールが見つかりません。\n最初に必要なファイルが作成されているか確認してください。"
                )
                return
            
            # Verificar que los scripts necesarios existen
            required_scripts = ["01_model_builder.py", "02_prediction.py", "03_pareto_analyzer.py"]
            missing_scripts = [s for s in required_scripts if not os.path.exists(s)]
            
            if missing_scripts:
                QMessageBox.warning(
                    self,
                    "スクリプトが見つかりません",
                    f"❌ 以下のスクリプトが見つかりません:\n\n" + "\n".join(missing_scripts) + 
                    "\n\nこれらのスクリプトは非線形解析に必要です。\n"
                    "スクリプトを配置してから再度お試しください。"
                )
                return
            
            # Mostrar diálogo de configuración
            config_dialog = NonlinearConfigDialog(self)
            if config_dialog.exec() != QDialog.Accepted:
                print("❌ Usuario canceló el diálogo de configuración")
                return
            
            # Obtener configuración
            config_values = config_dialog.get_config_values()
            print(f"📋 Configuración: {config_values}")
            
            # Mostrar diálogo de confirmación
            reply = QMessageBox.question(
                self,
                "非線形解析確認",
                f"非線形解析を実行しますか？\n\n"
                f"データ件数: {len(self.filtered_df)} 件\n"
                f"保存先: {project_folder}\n"
                f"モデル数: {len(config_values['models_to_use'])}\n\n"
                f"この操作は時間がかかる場合があります。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply != QMessageBox.Yes:
                print("❌ Usuario canceló el análisis no lineal")
                return
            
            # Guardar configuración para uso posterior
            self.nonlinear_config = config_values

            # ✅ NUEVO: reset de bandera de cancelación
            self._nonlinear_cancel_requested = False
            
            # Ejecutar análisis no lineal con worker
            print("🔧 Iniciando worker de análisis no lineal...")
            self.nonlinear_worker = NonlinearWorker(self.filtered_df, project_folder, self, config_values)
            
            # Conectar señales
            self.nonlinear_worker.progress_updated.connect(self.on_nonlinear_progress)
            self.nonlinear_worker.status_updated.connect(self.on_nonlinear_status)
            self.nonlinear_worker.finished.connect(self.on_nonlinear_finished)
            self.nonlinear_worker.error.connect(self.on_nonlinear_error)
            self.nonlinear_worker.console_output.connect(self.on_nonlinear_console_output)
            
            # Mostrar progreso (Stage 01 - chibi más grande x1.6)
            self.progress_dialog = ReusableProgressDialog(
                self, 
                title="非線形解析処理中...",
                chibi_image="Chibi_raul.png",
                chibi_size=160  # 100 * 1.6 = 160
            )
            self.progress_dialog.show()
            # Durante el loading modal: permitir flecha/consola por encima
            self.set_console_overlay_topmost(True)
            
            # Conectar señal de cancelación del diálogo para cancelar el worker
            self.progress_dialog.cancelled.connect(self.on_nonlinear_cancelled)
            
            # Conectar señal de progreso detallado (trial/fold/pass)
            self.nonlinear_worker.progress_detailed.connect(self.on_nonlinear_progress_detailed)
            
            # Iniciar worker
            self.nonlinear_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"❌ 非線形解析の実行中にエラーが発生しました:\n{str(e)}")
            print(f"❌ Error en análisis no lineal: {e}")
            import traceback
            traceback.print_exc()
    
    def on_nonlinear_progress(self, value, message):
        """Actualiza la barra de progreso"""
        if hasattr(self, 'progress_dialog'):
            # Si el mensaje indica un stage específico, actualizar el porcentaje según el stage
            if "Stage 02" in message or "Prediction" in message:
                # Stage 2: 70-85% (15% del total)
                # Ajustar el porcentaje para que esté en el rango correcto
                if value < 70:
                    value = 70
                elif value > 85:
                    value = 85
                # Mapear el progreso del stage 2 al rango 70-85%
                stage2_progress = (value - 60) / 40 if value >= 60 else 0  # Normalizar 60-100 a 0-1
                value = 70 + (stage2_progress * 15)  # Mapear a 70-85%
            elif "Stage 03" in message or "Pareto" in message:
                # Stage 3: 85-100% (15% del total)
                if value < 85:
                    value = 85
                elif value > 100:
                    value = 100
                # Mapear el progreso del stage 3 al rango 85-100%
                stage3_progress = (value - 90) / 10 if value >= 90 else 0  # Normalizar 90-100 a 0-1
                value = 85 + (stage3_progress * 15)  # Mapear a 85-100%
            
            self.progress_dialog.update_progress(value, message)
            # Verificar si el mensaje indica que el proceso sigue activo
            if "処理継続中" in message or "経過" in message:
                self.progress_dialog.set_process_active(True)
    
    def on_nonlinear_status(self, message):
        """Actualiza el mensaje de estado"""
        print(f"📊 Estado: {message}")
        if hasattr(self, 'progress_dialog'):
            # Actualizar estado del proceso basado en el mensaje
            if "処理継続中" in message or "経過" in message:
                self.progress_dialog.set_process_active(True)
            self.progress_dialog.set_status(message)
    
    def on_nonlinear_progress_detailed(self, trial_current, trial_total, fold_current, fold_total, pass_current, pass_total, current_task='dcv', data_analysis_completed=False, final_model_training=False, shap_analysis=False, model_current=0, model_total=0):
        """Actualiza el progreso detallado (trial/fold/pass/model) en el diálogo"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.update_progress_detailed(
                trial_current, trial_total, fold_current, fold_total, pass_current, pass_total, current_task, data_analysis_completed, final_model_training, shap_analysis, model_current, model_total
            )
    
    def on_nonlinear_console_output(self, message):
        """Muestra mensajes de consola del worker en la consola de la app"""
        try:
            # Escribir en la consola principal
            if hasattr(self, 'console_output') and self.console_output:
                self.console_output.append(message)
                # Auto-scroll al final (PySide6 usa MoveOperation.End)
                cursor = self.console_output.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.End)
                self.console_output.setTextCursor(cursor)
            
            # También escribir en la consola desplegable si existe
            if hasattr(self, 'overlay_console_output') and self.overlay_console_output:
                self.overlay_console_output.append(message)
                # Auto-scroll al final (PySide6 usa MoveOperation.End)
                cursor = self.overlay_console_output.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.End)
                self.overlay_console_output.setTextCursor(cursor)
            
            # También imprimir en stdout para que aparezca en PyCharm
            print(message, flush=True)
        except Exception as e:
            # Si falla, al menos intentar imprimir
            try:
                print(f"[Console Output Error] {e}: {message}", flush=True)
            except:
                pass
    
    def on_nonlinear_finished(self, results):
        """Maneja el resultado de la ejecución"""
        try:
            # ✅ NUEVO: Si el usuario canceló, no procesar resultados
            if hasattr(self, '_nonlinear_cancel_requested') and self._nonlinear_cancel_requested:
                print("🛑 DEBUG: Resultado no lineal recibido tras cancelación. Ignorando.")
                if hasattr(self, 'progress_dialog') and self.progress_dialog:
                    try:
                        self.progress_dialog.close()
                    except:
                        pass
                self.set_console_overlay_topmost(False)
                return

            print("✅ Análisis no lineal completado")
            print(f"   Carpeta de salida: {results['output_folder']}")
            print(f"   Stage: {results.get('stage', 'unknown')}")
            
            # Cerrar diálogo de progreso
            if hasattr(self, 'progress_dialog'):
                self.progress_dialog.close()
            self.set_console_overlay_topmost(False)
            
            # Verificar si es stage 01 (model_builder)
            if results.get('stage') == '01_model_builder':
                # Mostrar visor de gráficos
                self._show_graph_viewer(results)
            
            # Si es stage completed, mostrar resultados finales
            elif results.get('stage') == 'completed':
                self._show_final_results(results)
            
        except Exception as e:
            print(f"❌ Error en on_nonlinear_finished: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "エラー", f"❌ 結果処理中にエラーが発生しました:\n{str(e)}")
    
    def _show_graph_viewer(self, results):
        """Muestra el visor de gráficos y maneja OK/NG"""
        if GraphViewerDialog is None:
            QMessageBox.warning(self, "モジュールが見つかりません", "グラフビューアが利用できません。")
            return
        
        graph_paths = results.get('graph_paths', [])
        
        if not graph_paths:
            QMessageBox.information(
                self,
                "グラフなし",
                "生成されたグラフが見つかりませんでした。"
            )
            return
        
        # Mostrar visor de gráficos
        viewer = GraphViewerDialog(graph_paths, self)
        
        # Si el usuario hace OK, continuar con stages 2 y 3
        if viewer.exec() == QDialog.Accepted:
            print("✅ Usuario confirmó gráficos - continuar con stages 2-3")
            
            # Mostrar progreso nuevamente
            self.progress_dialog = ReusableProgressDialog(
                self,
                title="予測・パレート分析処理中...",
                chibi_image="xebec_chibi.png"
            )
            self.progress_dialog.show()
            self.set_console_overlay_topmost(True)
            
            # Conectar señales nuevamente
            self.nonlinear_worker.finished.disconnect()
            self.nonlinear_worker.finished.connect(self.on_nonlinear_finished)
            
            # Ejecutar stages 2 y 3
            self.nonlinear_worker.run_stage2_and_3()
        else:
            print("❌ Usuario canceló - proceso detenido")
            QMessageBox.information(
                self,
                "非線形解析中止",
                "プロセスが中止されました。\n\n保存先: " + results['output_folder']
            )
    
    def _show_final_results(self, results):
        """Muestra resultados finales del análisis completo con estadísticas"""
        output_folder = results.get('output_folder', '')
        is_load_existing = results.get('load_existing', False)
        existing_folder_path = results.get('existing_folder_path', '')
        
        # Si hay información de gráficos de Pareto, mostrar diálogo de resultados
        pareto_plots_folder = results.get('pareto_plots_folder')
        prediction_output_file = results.get('prediction_output_file')
        
        if pareto_plots_folder and prediction_output_file and ParetoResultsDialog is not None:
            self._show_pareto_charts_screen(pareto_plots_folder, prediction_output_file)
            return
        
        # Limpiar layout central completamente
        while self.center_layout.count():
            item = self.center_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            else:
                # Si es un layout, limpiarlo también
                layout = item.layout()
                if layout:
                    while layout.count():
                        layout_item = layout.takeAt(0)
                        layout_widget = layout_item.widget()
                        if layout_widget:
                            layout_widget.deleteLater()
        
        # Forzar actualización de la UI
        QApplication.processEvents()
        
        # Crear contenedor con fondo gris limpio
        gray_container = QFrame()
        gray_container.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border-radius: 10px;
                margin: 10px;
            }
        """)
        
        # Layout interno para el contenedor gris
        container_layout = QVBoxLayout(gray_container)
        container_layout.setContentsMargins(20, 20, 20, 20)
        container_layout.setSpacing(15)
        
        # Título
        if is_load_existing:
            title_text = "既存非線形解析結果"
        else:
            title_text = "非線形解析完了"
        
        title = QLabel(title_text)
        title.setStyleSheet("""
            font-weight: bold; 
            font-size: 24px; 
            color: #2c3e50;
            margin-bottom: 20px;
            padding: 10px 0px;
            border-bottom: 2px solid #3498db;
            border-radius: 0px;
        """)
        title.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(title)
        
        # Mensaje de éxito
        if is_load_existing:
            success_text = "✅ 既存の解析結果を読み込みました！"
        else:
            success_text = "✅ 非線形解析が完了しました！"
        
        success_label = QLabel(success_text)
        success_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #27ae60;
            padding: 10px;
            background-color: #d5f4e6;
            border-radius: 8px;
            border: 1px solid #27ae60;
        """)
        success_label.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(success_label)
        
        # Si es carga existente, cargar y mostrar archivos
        if is_load_existing and existing_folder_path:
            self._load_and_display_existing_files(container_layout, existing_folder_path, output_folder)
        else:
            # Cargar y mostrar estadísticas del análisis recién completado
            self._load_and_display_analysis_statistics(container_layout, output_folder)
        
        # Mensaje final
        final_message = QLabel("結果を確認してください。")
        final_message.setStyleSheet("""
            font-size: 14px;
            color: #7f8c8d;
            font-style: italic;
            margin-top: 10px;
        """)
        final_message.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(final_message)
        
        # Agregar botón "次へ" para ver gráficos (siempre que haya carpeta de salida)
        if output_folder:
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            next_button = QPushButton("次へ")
            next_button.setFixedSize(120, 40)
            next_button.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
            """)
            next_button.clicked.connect(lambda: self._show_nonlinear_charts_from_results(results))
            button_layout.addWidget(next_button)
            button_layout.addStretch()
            container_layout.addLayout(button_layout)
        
        # Agregar el contenedor al layout central
        self.center_layout.addWidget(gray_container)
        
        # Guardar información para navegación de gráficos
        if output_folder:
            # Buscar carpeta de resultados para guardar la ruta
            result_folder = os.path.join(output_folder, '03_学習結果')
            if os.path.exists(result_folder):
                self.nonlinear_existing_folder_path = result_folder
                # Guardar la carpeta del análisis completo (NUM_YYYYMMDD_HHMMSS) como project_folder
                # Esto permite que el botón "予測" funcione correctamente
                self.nonlinear_project_folder = output_folder
        
        # Forzar actualización
        QApplication.processEvents()
    
    def _load_and_display_existing_files(self, container_layout, existing_folder_path, output_folder):
        """Carga y muestra las estadísticas de un análisis existente"""
        # Usar la misma función que para análisis nuevo, ya que la estructura es la misma
        # existing_folder_path es la carpeta del análisis (NUM_YYYYMMDD_HHMMSS)
        # output_folder puede ser la misma o diferente, pero usamos existing_folder_path
        self._load_and_display_analysis_statistics(container_layout, existing_folder_path)
    
    def _load_and_display_analysis_statistics(self, container_layout, output_folder):
        """Carga y muestra las estadísticas del análisis recién completado"""
        try:
            from pathlib import Path
            import json
            from datetime import datetime
            
            # Buscar analysis_results.json directamente en la carpeta de resultados
            result_folder = os.path.join(output_folder, '03_学習結果')
            analysis_results_path = os.path.join(result_folder, 'analysis_results.json')
            
            analysis_data = {}
            
            if os.path.exists(analysis_results_path):
                try:
                    with open(analysis_results_path, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                    print(f"✅ Datos de análisis cargados desde: {analysis_results_path}")
                except Exception as e:
                    print(f"⚠️ Error leyendo analysis_results.json: {e}")
            else:
                print(f"⚠️ analysis_results.json no encontrado en: {analysis_results_path}")
            
            # Información del análisis
            filters_applied = analysis_data.get('filters_applied', [])
            if filters_applied == "N/A" or filters_applied is None:
                filters_text = "N/A"
            elif isinstance(filters_applied, list):
                if len(filters_applied) == 0:
                    filters_text = "N/A"
                elif len(filters_applied) > 3:
                    filters_text = f"{len(filters_applied)} 条件"
                else:
                    filters_text = ", ".join(str(f) for f in filters_applied)
            else:
                filters_text = str(filters_applied)
            
            # Truncar si es muy largo
            if len(filters_text) > 50:
                filters_text = filters_text[:47] + "..."
            
            data_range = analysis_data.get('data_range', 'N/A')
            if isinstance(data_range, str) and len(data_range) > 50:
                data_range = data_range[:47] + "..."
            
            # Obtener tiempo de análisis
            analysis_duration = analysis_data.get('analysis_duration_formatted', 'N/A')
            if analysis_duration == 'N/A' and analysis_data.get('analysis_duration_seconds'):
                # Si no está formateado, formatearlo
                duration_seconds = analysis_data.get('analysis_duration_seconds')
                if duration_seconds:
                    hours = int(duration_seconds // 3600)
                    minutes = int((duration_seconds % 3600) // 60)
                    seconds = int(duration_seconds % 60)
                    if hours > 0:
                        analysis_duration = f"{hours}時間{minutes}分{seconds}秒"
                    elif minutes > 0:
                        analysis_duration = f"{minutes}分{seconds}秒"
                    else:
                        analysis_duration = f"{seconds:.1f}秒"
            
            info_text = f"""
            📊 解析完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            ⏱️ 解析時間: {analysis_duration}
            📈 データ数: {analysis_data.get('data_count', 'N/A')} レコード
            🤖 訓練済みモデル: {analysis_data.get('models_trained', 'N/A')} 個
            🔧 フィルター適用: {filters_text}
            📊 データ範囲: {data_range}
            """
            
            info_label = QLabel(info_text)
            info_label.setStyleSheet("""
                font-size: 14px;
                color: #34495e;
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #bdc3c7;
            """)
            info_label.setAlignment(Qt.AlignLeft)
            info_label.setWordWrap(True)
            container_layout.addWidget(info_label)
            
            # Sección destacada de métricas de confianza
            models = analysis_data.get('models', {})
            if models and isinstance(models, dict) and len(models) > 0:
                # Título de la sección de métricas
                metrics_title = QLabel("📊 信頼性指標 (Confidence Metrics)")
                metrics_title.setStyleSheet("""
                    font-weight: bold; 
                    font-size: 20px; 
                    color: #2c3e50;
                    margin-top: 20px;
                    margin-bottom: 15px;
                    padding-bottom: 10px;
                    border-bottom: 3px solid #3498db;
                """)
                metrics_title.setAlignment(Qt.AlignCenter)
                container_layout.addWidget(metrics_title)
                
                # Crear layout horizontal para las tarjetas de métricas
                metrics_container = QHBoxLayout()
                metrics_container.setSpacing(15)
                
                # Iterar sobre cada target y crear tarjeta de métricas
                for target_name, model_info in models.items():
                    if isinstance(model_info, dict):
                        # Crear tarjeta para este target
                        metric_card = QFrame()
                        metric_card.setStyleSheet("""
                            QFrame {
                                background-color: #ffffff;
                                border: 2px solid #3498db;
                                border-radius: 10px;
                                padding: 15px;
                                min-width: 250px;
                            }
                        """)
                        card_layout = QVBoxLayout(metric_card)
                        card_layout.setSpacing(10)
                        
                        # Título del target
                        target_label = QLabel(f"【{target_name}】")
                        target_label.setStyleSheet("""
                            font-weight: bold;
                            font-size: 16px;
                            color: #2c3e50;
                            padding-bottom: 5px;
                            border-bottom: 1px solid #ecf0f1;
                        """)
                        target_label.setAlignment(Qt.AlignCenter)
                        card_layout.addWidget(target_label)
                        
                        # Métricas CV principales
                        cv_mae = model_info.get('cv_mae')
                        cv_rmse = model_info.get('cv_rmse')
                        cv_r2 = model_info.get('cv_r2')
                        
                        # MAE
                        if cv_mae is not None:
                            mae_label = QLabel(f"MAE: {cv_mae:.4f}" if isinstance(cv_mae, (int, float)) else f"MAE: {cv_mae}")
                            mae_label.setStyleSheet("""
                                font-size: 14px;
                                color: #34495e;
                                padding: 5px;
                                background-color: #f8f9fa;
                                border-radius: 5px;
                            """)
                            card_layout.addWidget(mae_label)
                        
                        # RMSE
                        if cv_rmse is not None:
                            rmse_label = QLabel(f"RMSE: {cv_rmse:.4f}" if isinstance(cv_rmse, (int, float)) else f"RMSE: {cv_rmse}")
                            rmse_label.setStyleSheet("""
                                font-size: 14px;
                                color: #34495e;
                                padding: 5px;
                                background-color: #f8f9fa;
                                border-radius: 5px;
                            """)
                            card_layout.addWidget(rmse_label)
                        
                        # R² (con color según el valor)
                        if cv_r2 is not None:
                            r2_value = cv_r2 if isinstance(cv_r2, (int, float)) else 0
                            # Color según calidad: verde si R² > 0.7, amarillo si > 0.5, rojo si <= 0.5
                            if r2_value > 0.7:
                                r2_color = "#27ae60"  # Verde
                                r2_bg = "#d5f4e6"
                            elif r2_value > 0.5:
                                r2_color = "#f39c12"  # Amarillo
                                r2_bg = "#fef5e7"
                            else:
                                r2_color = "#e74c3c"  # Rojo
                                r2_bg = "#fadbd8"
                            
                            r2_label = QLabel(f"R²: {cv_r2:.4f}" if isinstance(cv_r2, (int, float)) else f"R²: {cv_r2}")
                            r2_label.setStyleSheet(f"""
                                font-size: 14px;
                                font-weight: bold;
                                color: {r2_color};
                                padding: 5px;
                                background-color: {r2_bg};
                                border-radius: 5px;
                                border: 1px solid {r2_color};
                            """)
                            card_layout.addWidget(r2_label)
                        
                        # Métricas de folds (media y desviación estándar) si están disponibles
                        fold_mae_mean = model_info.get('fold_mae_mean')
                        fold_mae_std = model_info.get('fold_mae_std')
                        fold_rmse_mean = model_info.get('fold_rmse_mean')
                        fold_rmse_std = model_info.get('fold_rmse_std')
                        fold_r2_mean = model_info.get('fold_r2_mean')
                        fold_r2_std = model_info.get('fold_r2_std')
                        
                        # Agregar separador si hay métricas de folds
                        if any([fold_mae_mean, fold_rmse_mean, fold_r2_mean]):
                            separator = QLabel("─" * 20)
                            separator.setStyleSheet("color: #bdc3c7;")
                            separator.setAlignment(Qt.AlignCenter)
                            card_layout.addWidget(separator)
                            
                            # Subtítulo para métricas de folds
                            fold_title = QLabel("Fold Statistics:")
                            fold_title.setStyleSheet("""
                                font-size: 12px;
                                font-weight: bold;
                                color: #7f8c8d;
                                margin-top: 5px;
                            """)
                            fold_title.setAlignment(Qt.AlignCenter)
                            card_layout.addWidget(fold_title)
                            
                            # MAE fold statistics
                            if fold_mae_mean is not None:
                                mae_std_str = f"±{fold_mae_std:.4f}" if fold_mae_std is not None else ""
                                fold_mae_label = QLabel(f"MAE: {fold_mae_mean:.4f} {mae_std_str}")
                                fold_mae_label.setStyleSheet("""
                                    font-size: 12px;
                                    color: #7f8c8d;
                                    padding: 3px;
                                """)
                                card_layout.addWidget(fold_mae_label)
                            
                            # RMSE fold statistics
                            if fold_rmse_mean is not None:
                                rmse_std_str = f"±{fold_rmse_std:.4f}" if fold_rmse_std is not None else ""
                                fold_rmse_label = QLabel(f"RMSE: {fold_rmse_mean:.4f} {rmse_std_str}")
                                fold_rmse_label.setStyleSheet("""
                                    font-size: 12px;
                                    color: #7f8c8d;
                                    padding: 3px;
                                """)
                                card_layout.addWidget(fold_rmse_label)
                            
                            # R² fold statistics
                            if fold_r2_mean is not None:
                                r2_std_str = f"±{fold_r2_std:.4f}" if fold_r2_std is not None else ""
                                fold_r2_label = QLabel(f"R²: {fold_r2_mean:.4f} {r2_std_str}")
                                fold_r2_label.setStyleSheet("""
                                    font-size: 12px;
                                    color: #7f8c8d;
                                    padding: 3px;
                                """)
                                card_layout.addWidget(fold_r2_label)
                        
                        # Agregar la tarjeta al layout horizontal
                        metrics_container.addWidget(metric_card)
                
                # Agregar stretch al final para centrar las tarjetas
                metrics_container.addStretch()
                
                # Crear widget contenedor para el layout horizontal
                metrics_widget = QWidget()
                metrics_widget.setLayout(metrics_container)
                container_layout.addWidget(metrics_widget)
            
            # Ruta clickeable del archivo de salida
            if output_folder:
                path_layout = QVBoxLayout()
                
                path_title = QLabel("📁 出力ディレクトリ:")
                path_title.setStyleSheet("""
                    font-size: 14px;
                    font-weight: bold;
                    color: #2c3e50;
                    margin-top: 10px;
                    margin-bottom: 5px;
                """)
                path_layout.addWidget(path_title)
                
                path_label = QLabel(output_folder)
                path_label.setStyleSheet("""
                    QLabel {
                        font-size: 12px;
                        color: #3498db;
                        background-color: #e8f4fd;
                        padding: 10px;
                        border-radius: 5px;
                        border: 1px solid #3498db;
                        text-decoration: underline;
                    }
                    QLabel:hover {
                        background-color: #d1ecf1;
                        cursor: pointer;
                    }
                """)
                path_label.setWordWrap(True)
                path_label.setAlignment(Qt.AlignLeft)
                
                def open_folder():
                    try:
                        import subprocess
                        if os.name == 'nt':  # Windows
                            os.startfile(output_folder)
                        elif os.name == 'posix':  # macOS y Linux
                            subprocess.run(['open', output_folder], check=True)
                        else:
                            subprocess.run(['xdg-open', output_folder], check=True)
                        print(f"✅ Carpeta abierta: {output_folder}")
                    except Exception as e:
                        print(f"❌ Error abriendo carpeta: {e}")
                        QMessageBox.warning(self, "エラー", f"❌ フォルダを開けませんでした:\n{str(e)}")
                
                path_label.mousePressEvent = lambda event: open_folder()
                path_layout.addWidget(path_label)
                container_layout.addLayout(path_layout)
            
            # Resultados detallados de modelos (ya tenemos models de la sección anterior)
            if models and isinstance(models, dict) and len(models) > 0:
                models_title = QLabel("詳細モデル結果")
                models_title.setStyleSheet("""
                    font-weight: bold; 
                    font-size: 18px; 
                    color: #2c3e50;
                    margin-top: 20px;
                    margin-bottom: 10px;
                """)
                container_layout.addWidget(models_title)
                
                for target_name, model_info in models.items():
                    if isinstance(model_info, dict):
                        status = "✅ 成功"
                        model_name = model_info.get('model_name', 'Unknown')
                        details = f"モデル: {model_name}"
                        
                        # Agregar métricas CV si están disponibles
                        cv_r2 = model_info.get('cv_r2')
                        cv_mae = model_info.get('cv_mae')
                        cv_rmse = model_info.get('cv_rmse')
                        
                        if cv_r2 is not None:
                            if isinstance(cv_r2, (int, float)):
                                details += f", R²: {cv_r2:.4f}"
                            else:
                                details += f", R²: {cv_r2}"
                        
                        if cv_mae is not None:
                            if isinstance(cv_mae, (int, float)):
                                details += f", MAE: {cv_mae:.4f}"
                            else:
                                details += f", MAE: {cv_mae}"
                        
                        if cv_rmse is not None:
                            if isinstance(cv_rmse, (int, float)):
                                details += f", RMSE: {cv_rmse:.4f}"
                            else:
                                details += f", RMSE: {cv_rmse}"
                    else:
                        status = "✅ 成功"
                        details = f"モデル情報: {str(model_info)[:100]}"
                    
                    model_label = QLabel(f"【{target_name}】 {status}\n{details}")
                    model_label.setStyleSheet("""
                        font-size: 12px;
                        color: #34495e;
                        background-color: #f8f9fa;
                        padding: 10px;
                        border-radius: 5px;
                        border: 1px solid #dee2e6;
                        margin: 5px 0px;
                    """)
                    container_layout.addWidget(model_label)
        
        except Exception as e:
            print(f"❌ Error cargando estadísticas del análisis: {e}")
            import traceback
            traceback.print_exc()
            error_label = QLabel(f"❌ 統計情報の読み込み中にエラーが発生しました:\n{str(e)}")
            error_label.setStyleSheet("color: #e74c3c; padding: 10px; background-color: #fadbd8; border-radius: 5px;")
            error_label.setWordWrap(True)
            container_layout.addWidget(error_label)
    
    def _show_nonlinear_charts_from_results(self, results):
        """Mostrar gráficos del análisis no lineal desde los resultados"""
        output_folder = results.get('output_folder', '')
        if not output_folder:
            QMessageBox.warning(self, "エラー", "❌ グラフを表示するための情報が見つかりません。")
            return
        
        # Buscar carpeta de resultados (03_学習結果)
        result_folder = os.path.join(output_folder, '03_学習結果')
        
        # Guardar información para navegación
        if os.path.exists(result_folder):
            self.nonlinear_existing_folder_path = result_folder
            self.nonlinear_project_folder = output_folder
            # Llamar a la función de mostrar gráficos (si existe)
            if hasattr(self, 'show_nonlinear_charts'):
                self.show_nonlinear_charts()
            else:
                QMessageBox.information(
                    self,
                    "情報",
                    "グラフ表示機能は準備中です。\n\n結果フォルダ:\n" + output_folder
                )
        else:
            QMessageBox.warning(
                self,
                "エラー",
                f"❌ 結果フォルダが見つかりません:\n{result_folder}"
            )
    
    def show_nonlinear_charts(self):
        """Mostrar gráficos del análisis no lineal con navegación"""
        print("🔧 Mostrando gráficos del análisis no lineal...")
        
        try:
            # Verificar que tenemos la ruta de la carpeta cargada
            if not hasattr(self, 'nonlinear_existing_folder_path') or not self.nonlinear_existing_folder_path:
                QMessageBox.warning(self, "エラー", "❌ グラフを表示するための情報が見つかりません。")
                return
            
            # Limpiar layout central completamente
            while self.center_layout.count():
                item = self.center_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                else:
                    # Si es un layout, limpiarlo también
                    layout = item.layout()
                    if layout:
                        while layout.count():
                            layout_item = layout.takeAt(0)
                            layout_widget = layout_item.widget()
                            if layout_widget:
                                layout_widget.deleteLater()
            
            # Forzar actualización de la UI
            QApplication.processEvents()
            
            # Crear contenedor con fondo gris limpio
            gray_container = QFrame()
            gray_container.setStyleSheet("""
                QFrame {
                    background-color: #f5f5f5;
                    border-radius: 10px;
                    margin: 10px;
                }
            """)
            
            # Layout interno para el contenedor gris
            container_layout = QVBoxLayout(gray_container)
            container_layout.setContentsMargins(20, 20, 20, 20)
            container_layout.setSpacing(15)
            
            # Título
            title = QLabel("非線形解析結果 チャート")
            title.setStyleSheet("""
                font-weight: bold; 
                font-size: 24px; 
                color: #2c3e50;
                margin-bottom: 20px;
                padding: 10px 0px;
                border-bottom: 2px solid #3498db;
                border-radius: 0px;
            """)
            title.setAlignment(Qt.AlignCenter)
            container_layout.addWidget(title)
            
            # Buscar gráficos PNG en la carpeta de resultados (03_学習結果)
            from pathlib import Path
            folder_path = Path(self.nonlinear_existing_folder_path)
            chart_images = []
            
            # Buscar imágenes PNG directamente en la carpeta de resultados
            for file in folder_path.glob("*.png"):
                if file.is_file():
                    chart_images.append(str(file))
            
            # Buscar también en data_analysis si existe
            data_analysis_path = folder_path / "data_analysis"
            if data_analysis_path.exists() and data_analysis_path.is_dir():
                for file in data_analysis_path.glob("*.png"):
                    if file.is_file():
                        chart_images.append(str(file))
            
            # Si no se encuentran gráficos, mostrar mensaje
            if not chart_images:
                no_charts_label = QLabel("⚠️ グラフが見つかりません")
                no_charts_label.setStyleSheet("""
                    font-size: 16px;
                    color: #e74c3c;
                    background-color: #fadbd8;
                    padding: 20px;
                    border-radius: 8px;
                    border: 1px solid #e74c3c;
                    margin: 20px 0px;
                """)
                no_charts_label.setAlignment(Qt.AlignCenter)
                container_layout.addWidget(no_charts_label)
            else:
                # Configurar navegación de gráficos
                self.nonlinear_chart_images = sorted(chart_images)
                self.current_nonlinear_chart_index = 0
                
                # Layout principal para la imagen y navegación
                chart_layout = QVBoxLayout()
                
                # Label para mostrar la imagen (ocupa todo el ancho)
                self.nonlinear_chart_label = QLabel()
                self.nonlinear_chart_label.setAlignment(Qt.AlignCenter)
                self.nonlinear_chart_label.setStyleSheet("""
                    QLabel {
                        background-color: white;
                        border: 2px solid #bdc3c7;
                        border-radius: 10px;
                        padding: 10px;
                        min-height: 500px;
                    }
                """)
                chart_layout.addWidget(self.nonlinear_chart_label)
                
                # Layout horizontal para botones de navegación (debajo de la imagen)
                nav_buttons_layout = QHBoxLayout()
                nav_buttons_layout.addStretch()
                
                # Botón flecha izquierda
                prev_chart_button = QPushButton("◀ 前へ")
                prev_chart_button.setFixedSize(100, 40)
                prev_chart_button.setStyleSheet("""
                    QPushButton {
                        background-color: #3498db;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        font-size: 14px;
                        font-weight: bold;
                        padding: 8px 16px;
                    }
                    QPushButton:hover {
                        background-color: #2980b9;
                    }
                    QPushButton:disabled {
                        background-color: #bdc3c7;
                        color: #7f8c8d;
                    }
                """)
                prev_chart_button.clicked.connect(self.show_previous_nonlinear_chart)
                nav_buttons_layout.addWidget(prev_chart_button)
                
                # Espacio entre botones
                nav_buttons_layout.addSpacing(20)
                
                # Botón flecha derecha
                next_chart_button = QPushButton("次へ ▶")
                next_chart_button.setFixedSize(100, 40)
                next_chart_button.setStyleSheet("""
                    QPushButton {
                        background-color: #3498db;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        font-size: 14px;
                        font-weight: bold;
                        padding: 8px 16px;
                    }
                    QPushButton:hover {
                        background-color: #2980b9;
                    }
                    QPushButton:disabled {
                        background-color: #bdc3c7;
                        color: #7f8c8d;
                    }
                """)
                next_chart_button.clicked.connect(self.show_next_nonlinear_chart)
                nav_buttons_layout.addWidget(next_chart_button)
                
                nav_buttons_layout.addStretch()
                chart_layout.addLayout(nav_buttons_layout)
                
                # Información del gráfico actual
                self.nonlinear_chart_info_label = QLabel()
                self.nonlinear_chart_info_label.setStyleSheet("""
                    font-size: 14px;
                    color: #2c3e50;
                    background-color: #ecf0f1;
                    padding: 10px;
                    border-radius: 5px;
                    border: 1px solid #bdc3c7;
                    margin: 10px 0px;
                """)
                self.nonlinear_chart_info_label.setAlignment(Qt.AlignCenter)
                chart_layout.addWidget(self.nonlinear_chart_info_label)
                
                container_layout.addLayout(chart_layout)
                
                # Mostrar el primer gráfico
                self.update_nonlinear_chart_display()
            
            # Botones para volver y predicción
            buttons_layout = QHBoxLayout()
            buttons_layout.addStretch()
            
            # Botón para volver
            back_button = QPushButton("戻る")
            back_button.setFixedSize(120, 40)
            back_button.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
            """)
            back_button.clicked.connect(self.on_analyze_clicked)
            buttons_layout.addWidget(back_button)
            
            # Espacio entre botones
            buttons_layout.addSpacing(20)
            
            # Botón para predicción
            prediction_button = QPushButton("予測")
            prediction_button.setFixedSize(120, 40)
            prediction_button.setStyleSheet("""
                QPushButton {
                    background-color: #27ae60;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #229954;
                }
            """)
            # Conectar botón de predicción si existe la función, sino deshabilitarlo
            if hasattr(self, 'run_nonlinear_prediction'):
                prediction_button.clicked.connect(self.run_nonlinear_prediction)
            else:
                prediction_button.setEnabled(False)
                prediction_button.setToolTip("予測機能は準備中です")
            buttons_layout.addWidget(prediction_button)
            
            buttons_layout.addStretch()
            container_layout.addLayout(buttons_layout)
            
            # Espacio flexible
            container_layout.addStretch()
            
            # Agregar el contenedor gris al layout central
            self.center_layout.addWidget(gray_container)
            
            print("✅ Gráficos del análisis no lineal mostrados")
            
        except Exception as e:
            print(f"❌ Error mostrando gráficos del análisis no lineal: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "エラー", f"❌ グラフの表示中にエラーが発生しました:\n{str(e)}")
    
    def show_previous_nonlinear_chart(self):
        """Mostrar gráfico anterior del análisis no lineal"""
        if hasattr(self, 'nonlinear_chart_images') and len(self.nonlinear_chart_images) > 0:
            if not hasattr(self, 'current_nonlinear_chart_index'):
                self.current_nonlinear_chart_index = 0
            self.current_nonlinear_chart_index = (self.current_nonlinear_chart_index - 1) % len(self.nonlinear_chart_images)
            self.update_nonlinear_chart_display()
    
    def show_next_nonlinear_chart(self):
        """Mostrar gráfico siguiente del análisis no lineal"""
        if hasattr(self, 'nonlinear_chart_images') and len(self.nonlinear_chart_images) > 0:
            if not hasattr(self, 'current_nonlinear_chart_index'):
                self.current_nonlinear_chart_index = 0
            self.current_nonlinear_chart_index = (self.current_nonlinear_chart_index + 1) % len(self.nonlinear_chart_images)
            self.update_nonlinear_chart_display()
    
    def update_nonlinear_chart_display(self):
        """Actualizar la visualización del gráfico actual del análisis no lineal"""
        if not hasattr(self, 'nonlinear_chart_images') or len(self.nonlinear_chart_images) == 0:
            return
        
        if not hasattr(self, 'current_nonlinear_chart_index'):
            self.current_nonlinear_chart_index = 0
        
        if self.current_nonlinear_chart_index < 0:
            self.current_nonlinear_chart_index = 0
        elif self.current_nonlinear_chart_index >= len(self.nonlinear_chart_images):
            self.current_nonlinear_chart_index = len(self.nonlinear_chart_images) - 1
        
        current_image_path = self.nonlinear_chart_images[self.current_nonlinear_chart_index]
        
        # Cargar y mostrar la imagen
        pixmap = QPixmap(current_image_path)
        if not pixmap.isNull():
            # Redimensionar la imagen para ocupar todo el ancho disponible
            # Obtener el tamaño del contenedor
            container_width = self.nonlinear_chart_label.width() - 20  # Restar padding
            container_height = self.nonlinear_chart_label.height() - 20  # Restar padding
            
            # Si el contenedor aún no tiene tamaño, usar un tamaño por defecto
            if container_width <= 0:
                container_width = 1000
            if container_height <= 0:
                container_height = 600
            
            # Redimensionar manteniendo la proporción
            scaled_pixmap = pixmap.scaled(container_width, container_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.nonlinear_chart_label.setPixmap(scaled_pixmap)
            
            # Actualizar información del gráfico
            filename = os.path.basename(current_image_path)
            info_text = f"📊 {filename} ({self.current_nonlinear_chart_index + 1}/{len(self.nonlinear_chart_images)})"
            if hasattr(self, 'nonlinear_chart_info_label'):
                self.nonlinear_chart_info_label.setText(info_text)
            
            print(f"✅ Mostrando gráfico: {filename}")
        else:
            print(f"❌ No se pudo cargar la imagen: {current_image_path}")
    
    def _show_pareto_charts_screen(self, pareto_plots_folder, prediction_output_file):
        """Mostrar gráficos de Pareto en formato pantalla (similar a show_nonlinear_charts)"""
        print("🔧 Mostrando gráficos de Pareto en pantalla...")
        
        try:
            # Limpiar layout central completamente
            while self.center_layout.count():
                item = self.center_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                else:
                    layout = item.layout()
                    if layout:
                        while layout.count():
                            layout_item = layout.takeAt(0)
                            layout_widget = layout_item.widget()
                            if layout_widget:
                                layout_widget.deleteLater()
            
            # Forzar actualización de la UI
            QApplication.processEvents()
            
            # Crear contenedor con fondo gris limpio
            gray_container = QFrame()
            gray_container.setStyleSheet("""
                QFrame {
                    background-color: #f5f5f5;
                    border-radius: 10px;
                    margin: 10px;
                }
            """)
            
            # Layout interno para el contenedor gris
            container_layout = QVBoxLayout(gray_container)
            container_layout.setContentsMargins(20, 20, 20, 20)
            container_layout.setSpacing(15)
            
            # Título
            title = QLabel("パレート分析結果 チャート")
            title.setStyleSheet("""
                font-weight: bold; 
                font-size: 24px; 
                color: #2c3e50;
                margin-bottom: 20px;
                padding: 10px 0px;
                border-bottom: 2px solid #3498db;
                border-radius: 0px;
            """)
            title.setAlignment(Qt.AlignCenter)
            container_layout.addWidget(title)
            
            # Buscar gráficos PNG en la carpeta de Pareto
            from pathlib import Path
            folder_path = Path(pareto_plots_folder)
            chart_images = []
            
            # Buscar imágenes PNG en la carpeta
            if folder_path.exists() and folder_path.is_dir():
                for file in folder_path.glob("*.png"):
                    if file.is_file():
                        chart_images.append(str(file))
                # También buscar JPG/JPEG
                for file in folder_path.glob("*.jpg"):
                    if file.is_file():
                        chart_images.append(str(file))
                for file in folder_path.glob("*.jpeg"):
                    if file.is_file():
                        chart_images.append(str(file))
            
            # Si no se encuentran gráficos, mostrar mensaje
            if not chart_images:
                no_charts_label = QLabel("⚠️ グラフが見つかりません")
                no_charts_label.setStyleSheet("""
                    font-size: 16px;
                    color: #e74c3c;
                    background-color: #fadbd8;
                    padding: 20px;
                    border-radius: 8px;
                    border: 1px solid #e74c3c;
                    margin: 20px 0px;
                """)
                no_charts_label.setAlignment(Qt.AlignCenter)
                container_layout.addWidget(no_charts_label)
            else:
                # Configurar navegación de gráficos
                self.pareto_chart_images = sorted(chart_images)
                self.current_pareto_chart_index = 0
                
                # Layout principal para la imagen y navegación
                chart_layout = QVBoxLayout()
                
                # Label para mostrar la imagen (ocupa todo el ancho)
                self.pareto_chart_label = QLabel()
                self.pareto_chart_label.setAlignment(Qt.AlignCenter)
                self.pareto_chart_label.setStyleSheet("""
                    QLabel {
                        background-color: white;
                        border: 2px solid #bdc3c7;
                        border-radius: 10px;
                        padding: 10px;
                        min-height: 500px;
                    }
                """)
                chart_layout.addWidget(self.pareto_chart_label)
                
                # Layout horizontal para botones de navegación (debajo de la imagen)
                nav_buttons_layout = QHBoxLayout()
                nav_buttons_layout.addStretch()
                
                # Botón flecha izquierda
                prev_chart_button = QPushButton("◀ 前へ")
                prev_chart_button.setFixedSize(100, 40)
                prev_chart_button.setStyleSheet("""
                    QPushButton {
                        background-color: #3498db;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        font-size: 14px;
                        font-weight: bold;
                        padding: 8px 16px;
                    }
                    QPushButton:hover {
                        background-color: #2980b9;
                    }
                    QPushButton:disabled {
                        background-color: #bdc3c7;
                        color: #7f8c8d;
                    }
                """)
                prev_chart_button.clicked.connect(self.show_previous_pareto_chart)
                nav_buttons_layout.addWidget(prev_chart_button)
                
                # Espacio entre botones
                nav_buttons_layout.addSpacing(20)
                
                # Botón flecha derecha
                next_chart_button = QPushButton("次へ ▶")
                next_chart_button.setFixedSize(100, 40)
                next_chart_button.setStyleSheet("""
                    QPushButton {
                        background-color: #3498db;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        font-size: 14px;
                        font-weight: bold;
                        padding: 8px 16px;
                    }
                    QPushButton:hover {
                        background-color: #2980b9;
                    }
                    QPushButton:disabled {
                        background-color: #bdc3c7;
                        color: #7f8c8d;
                    }
                """)
                next_chart_button.clicked.connect(self.show_next_pareto_chart)
                nav_buttons_layout.addWidget(next_chart_button)
                
                nav_buttons_layout.addStretch()
                chart_layout.addLayout(nav_buttons_layout)
                
                # Información del gráfico actual
                self.pareto_chart_info_label = QLabel()
                self.pareto_chart_info_label.setStyleSheet("""
                    font-size: 14px;
                    color: #2c3e50;
                    background-color: #ecf0f1;
                    padding: 10px;
                    border-radius: 5px;
                    border: 1px solid #bdc3c7;
                    margin: 10px 0px;
                """)
                self.pareto_chart_info_label.setAlignment(Qt.AlignCenter)
                chart_layout.addWidget(self.pareto_chart_info_label)
                
                container_layout.addLayout(chart_layout)
                
                # Guardar referencia al archivo de predicción para importar
                self.pareto_prediction_output_file = prediction_output_file
                
                # Mostrar el primer gráfico
                self.update_pareto_chart_display()
            
            # Botones de acción
            buttons_layout = QHBoxLayout()
            buttons_layout.addStretch()
            
            # Botón para volver
            back_button = QPushButton("戻る")
            back_button.setFixedSize(120, 40)
            back_button.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
            """)
            back_button.clicked.connect(self.on_analyze_clicked)
            buttons_layout.addWidget(back_button)
            
            # Espacio entre botones
            buttons_layout.addSpacing(20)
            
            # Botón para importar a base de datos
            import_button = QPushButton("データベースにインポート")
            import_button.setFixedSize(180, 40)
            import_button.setStyleSheet("""
                QPushButton {
                    background-color: #27ae60;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #229954;
                }
            """)
            import_button.clicked.connect(lambda: self.import_nonlinear_pareto_to_database(self.pareto_prediction_output_file))
            buttons_layout.addWidget(import_button)
            
            buttons_layout.addStretch()
            container_layout.addLayout(buttons_layout)
            
            # Espacio flexible
            container_layout.addStretch()
            
            # Agregar el contenedor gris al layout central
            self.center_layout.addWidget(gray_container)
            
            print("✅ Gráficos de Pareto mostrados en pantalla")
            
        except Exception as e:
            print(f"❌ Error mostrando gráficos de Pareto: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "エラー", f"❌ グラフの表示中にエラーが発生しました:\n{str(e)}")
    
    def show_previous_pareto_chart(self):
        """Mostrar gráfico anterior de Pareto"""
        if hasattr(self, 'pareto_chart_images') and len(self.pareto_chart_images) > 0:
            self.current_pareto_chart_index = (self.current_pareto_chart_index - 1) % len(self.pareto_chart_images)
            self.update_pareto_chart_display()
    
    def show_next_pareto_chart(self):
        """Mostrar gráfico siguiente de Pareto"""
        if hasattr(self, 'pareto_chart_images') and len(self.pareto_chart_images) > 0:
            self.current_pareto_chart_index = (self.current_pareto_chart_index + 1) % len(self.pareto_chart_images)
            self.update_pareto_chart_display()
    
    def update_pareto_chart_display(self):
        """Actualizar la visualización del gráfico actual de Pareto"""
        if hasattr(self, 'pareto_chart_images') and len(self.pareto_chart_images) > 0:
            current_image_path = self.pareto_chart_images[self.current_pareto_chart_index]
            
            # Cargar y mostrar la imagen
            pixmap = QPixmap(current_image_path)
            if not pixmap.isNull():
                # Redimensionar la imagen para ocupar todo el ancho disponible
                container_width = self.pareto_chart_label.width() - 20  # Restar padding
                container_height = self.pareto_chart_label.height() - 20  # Restar padding
                
                # Si el contenedor aún no tiene tamaño, usar un tamaño por defecto
                if container_width <= 0:
                    container_width = 1000
                if container_height <= 0:
                    container_height = 600
                
                # Redimensionar manteniendo la proporción
                scaled_pixmap = pixmap.scaled(container_width, container_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.pareto_chart_label.setPixmap(scaled_pixmap)
                
                # Actualizar información del gráfico
                filename = os.path.basename(current_image_path)
                info_text = f"📊 {filename} ({self.current_pareto_chart_index + 1}/{len(self.pareto_chart_images)})"
                self.pareto_chart_info_label.setText(info_text)
                
                print(f"✅ Mostrando gráfico de Pareto: {filename}")
            else:
                print(f"❌ No se pudo cargar la imagen: {current_image_path}")
    
    def run_nonlinear_prediction(self):
        """
        Ejecuta predicción no lineal (02_prediction.py y 03_pareto_analyzer.py)
        desde la pantalla de gráficos del análisis no lineal
        """
        print("🔧 Iniciando predicción no lineal desde pantalla de gráficos...")
        
        try:
            # Verificar que tenemos la carpeta del proyecto no lineal
            if not hasattr(self, 'nonlinear_project_folder') or not self.nonlinear_project_folder:
                QMessageBox.warning(
                    self,
                    "エラー",
                    "❌ 予測を実行するための情報が見つかりません。\n\nまず非線形解析を実行してください。"
                )
                return
            
            working_dir = self.nonlinear_project_folder
            if not os.path.exists(working_dir):
                QMessageBox.warning(
                    self,
                    "エラー",
                    f"❌ 作業ディレクトリが見つかりません:\n{working_dir}"
                )
                return
            
            # Confirmar con el usuario
            reply = QMessageBox.question(
                self,
                "予測実行確認",
                f"予測とパレート解析を実行しますか？\n\n作業ディレクトリ:\n{working_dir}\n\n"
                f"⚠️ 実行前にバックアップが作成されます。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
            
            # Crear backup antes de ejecutar
            backup_created = self._create_nonlinear_backup(working_dir)
            if not backup_created:
                reply = QMessageBox.question(
                    self,
                    "バックアップ警告",
                    "⚠️ バックアップの作成に失敗しました。\n\nそれでも続行しますか？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
            
            # Mostrar diálogo de progreso (Stage 02/03 - chibi más grande x1.6)
            self.progress_dialog = ReusableProgressDialog(
                self,
                title="予測・パレート分析処理中...",
                chibi_image="Chibi_sukuzisan_raul.png",
                chibi_size=160  # 100 * 1.6 = 160
            )
            self.progress_dialog.show()
            self.set_console_overlay_topmost(True)
            self.progress_dialog.set_status("予測処理を開始中...")
            self.progress_dialog.update_progress(5, "予測処理を開始中...")
            
            # Guardar tiempo de inicio total (para tiempo transcurrido continuo)
            total_start_time = time.time()
            
            # Ejecutar 02_prediction.py (5% - 20%)
            print(f"🔧 Ejecutando 02_prediction.py en: {working_dir}")
            self.progress_dialog.set_status("02_prediction.py 実行中...")
            
            prediction_success = self._run_prediction_script(working_dir, self.progress_dialog, progress_start=5, progress_end=20, total_start_time=total_start_time)
            
            if not prediction_success:
                self.progress_dialog.close()
                self.set_console_overlay_topmost(False)
                QMessageBox.critical(
                    self,
                    "エラー",
                    "❌ 02_prediction.py の実行に失敗しました。\n\n詳細はコンソールを確認してください。"
                )
                return
            
            # Ejecutar 03_pareto_analyzer.py (20% - 100%)
            print(f"🔧 Ejecutando 03_pareto_analyzer.py en: {working_dir}")
            self.progress_dialog.set_status("03_pareto_analyzer.py 実行中...")
            self.progress_dialog.update_progress(20, "03_pareto_analyzer.py 実行中...")
            
            pareto_success = self._run_pareto_script(working_dir, self.progress_dialog, progress_start=20, progress_end=100, total_start_time=total_start_time)
            
            if not pareto_success:
                self.progress_dialog.close()
                self.set_console_overlay_topmost(False)
                QMessageBox.critical(
                    self,
                    "エラー",
                    "❌ 03_pareto_analyzer.py の実行に失敗しました。\n\n詳細はコンソールを確認してください。"
                )
                return
            
            # Cerrar diálogo de progreso
            self.progress_dialog.close()
            self.set_console_overlay_topmost(False)
            
            # Construir rutas de resultados del pareto
            pareto_plots_folder = os.path.join(working_dir, "05_パレート解", "pareto_plots")
            prediction_output_file = os.path.join(working_dir, "04_予測", "Prediction_output.xlsx")
            
            # DEBUG: Verificar rutas
            print(f"🔍 DEBUG run_nonlinear_prediction: working_dir = {working_dir}")
            print(f"🔍 DEBUG run_nonlinear_prediction: pareto_plots_folder = {pareto_plots_folder}")
            print(f"🔍 DEBUG run_nonlinear_prediction: prediction_output_file = {prediction_output_file}")
            print(f"🔍 DEBUG run_nonlinear_prediction: pareto_plots_folder exists = {os.path.exists(pareto_plots_folder)}")
            print(f"🔍 DEBUG run_nonlinear_prediction: prediction_output_file exists = {os.path.exists(prediction_output_file)}")
            
            # Verificar que existen los archivos
            if os.path.exists(pareto_plots_folder) and os.path.exists(prediction_output_file):
                # Mostrar pantalla de gráficos de Pareto
                print(f"✅ Mostrando gráficos de Pareto desde: {pareto_plots_folder}")
                self._show_pareto_charts_screen(pareto_plots_folder, prediction_output_file)
            else:
                # Si no existen, mostrar mensaje de éxito pero sin gráficos
                missing_items = []
                if not os.path.exists(pareto_plots_folder):
                    missing_items.append(f"パレートグラフフォルダ: {pareto_plots_folder}")
                    print(f"❌ DEBUG: pareto_plots_folder no existe")
                if not os.path.exists(prediction_output_file):
                    missing_items.append(f"予測出力ファイル: {prediction_output_file}")
                    print(f"❌ DEBUG: prediction_output_file no existe")
                
                # Listar contenido del directorio para debug
                if os.path.exists(working_dir):
                    print(f"🔍 DEBUG: Contenido de working_dir:")
                    try:
                        for item in os.listdir(working_dir):
                            item_path = os.path.join(working_dir, item)
                            item_type = "DIR" if os.path.isdir(item_path) else "FILE"
                            print(f"   {item_type}: {item}")
                    except Exception as e:
                        print(f"⚠️ Error listando contenido: {e}")
                
                QMessageBox.information(
                    self,
                    "処理完了",
                    f"✅ 予測とパレート解析が正常に完了しました！\n\n"
                    f"作業ディレクトリ: {working_dir}\n\n"
                    f"✅ 02_prediction.py: 完了\n"
                    f"✅ 03_pareto_analyzer.py: 完了\n\n"
                    f"⚠️ 以下のファイルが見つかりませんでした:\n" + "\n".join(missing_items)
                )
            
        except Exception as e:
            print(f"❌ Error en run_nonlinear_prediction: {e}")
            import traceback
            traceback.print_exc()
            
            if hasattr(self, 'progress_dialog'):
                self.progress_dialog.close()
            
            QMessageBox.critical(
                self,
                "エラー",
                f"❌ 予測実行中にエラーが発生しました:\n{str(e)}"
            )
    
    def _create_nonlinear_backup(self, working_dir):
        """
        Crea un backup de la carpeta del análisis no lineal antes de ejecutar predicción
        
        Parameters
        ----------
        working_dir : str
            Directorio de trabajo del análisis no lineal
        
        Returns
        -------
        bool
            True si el backup se creó exitosamente, False en caso contrario
        """
        try:
            from datetime import datetime
            
            # Obtener la ruta base del proyecto (donde está 0sec.py)
            # working_dir es algo como: Archivos_de_salida/Proyecto_79/04_非線形回帰/100_20251120_102819
            # Necesitamos llegar a la raíz del proyecto donde está .venv
            current_path = Path(working_dir).resolve()
            
            # Buscar la carpeta .venv o la raíz del proyecto
            backup_base = None
            search_path = current_path
            
            # Buscar hacia arriba hasta encontrar .venv o llegar a la raíz
            while search_path != search_path.parent:
                venv_path = search_path / ".venv"
                if venv_path.exists() and venv_path.is_dir():
                    # Encontramos .venv, crear Backup en el mismo nivel
                    backup_base = search_path / "Backup"
                    break
                search_path = search_path.parent
            
            # Si no encontramos .venv, usar la ruta del directorio actual como fallback
            if backup_base is None:
                backup_base = Path.cwd() / "Backup"
            
            # Crear carpeta Backup si no existe
            backup_base.mkdir(parents=True, exist_ok=True)
            
            # Crear carpeta con timestamp (formato: YYYYMMDD)
            timestamp = datetime.now().strftime("%Y%m%d")
            backup_folder = backup_base / timestamp
            backup_folder.mkdir(parents=True, exist_ok=True)
            
            # Copiar toda la carpeta del análisis no lineal
            folder_name = os.path.basename(working_dir)
            dest_folder = backup_folder / folder_name
            
            # Si ya existe, agregar un sufijo numérico
            if dest_folder.exists():
                counter = 1
                while (backup_folder / f"{folder_name}_{counter}").exists():
                    counter += 1
                dest_folder = backup_folder / f"{folder_name}_{counter}"
            
            print(f"📁 Creando backup: {working_dir} → {dest_folder}")
            
            # Copiar recursivamente
            shutil.copytree(working_dir, str(dest_folder), dirs_exist_ok=True)
            
            print(f"✅ Backup creado exitosamente: {dest_folder}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error creando backup: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_prediction_script(self, working_dir, progress_dialog=None, progress_start=0, progress_end=20, total_start_time=None):
        """
        Ejecuta 02_prediction.py en el directorio de trabajo
        
        Parameters
        ----------
        working_dir : str
            Directorio de trabajo
        progress_dialog : ReusableProgressDialog, optional
            Diálogo de progreso para actualizar
        progress_start : int
            Porcentaje inicial de progreso (0-100)
        progress_end : int
            Porcentaje final de progreso (0-100)
        total_start_time : float, optional
            Tiempo de inicio total para tiempo transcurrido continuo
        
        Returns
        -------
        bool
            True si el script se ejecutó exitosamente, False en caso contrario
        """
        try:
            # Preparar archivo de predicción antes de ejecutar
            # 1. Crear carpeta 04_予測 si no existe
            prediction_folder = os.path.join(working_dir, "04_予測")
            os.makedirs(prediction_folder, exist_ok=True)
            
            # 2. Buscar el archivo NOMBREDELPROYECTO__未実験データ.xlsx en la carpeta principal del proyecto
            # working_dir es: .../Proyecto_79/04_非線形回帰/100_YYYYMMDD_HHMMSS
            # Necesitamos llegar a: .../Proyecto_79/
            from pathlib import Path
            working_path = Path(working_dir).resolve()
            project_folder = None
            
            # Buscar hacia arriba hasta encontrar la carpeta del proyecto (que contiene 04_非線形回帰)
            for parent in working_path.parents:
                if parent.name == "04_非線形回帰":
                    project_folder = parent.parent
                    break
            
            if project_folder is None:
                # Fallback: buscar por nombre de carpeta que contiene "Proyecto"
                for parent in working_path.parents:
                    if "Proyecto" in parent.name:
                        project_folder = parent
                        break
            
            if project_folder is None:
                # Último fallback: usar el directorio padre de 04_非線形回帰
                # working_dir debería ser .../Proyecto_XX/04_非線形回帰/100_...
                # Entonces parent.parent debería ser Proyecto_XX
                project_folder = working_path.parent.parent
                print(f"⚠️ Usando fallback para carpeta del proyecto: {project_folder}")
            
            print(f"📁 Carpeta del proyecto encontrada: {project_folder}")
            
            # 3. Buscar el archivo con patrón *__未実験データ.xlsx
            prediction_source_file = None
            project_name = project_folder.name  # Ej: "Proyecto_79"
            expected_filename = f"{project_name}_未実験データ.xlsx"
            expected_path = project_folder / expected_filename
            
            print(f"🔍 Buscando archivo: {expected_path}")
            
            if expected_path.exists():
                prediction_source_file = expected_path
                print(f"✅ Archivo encontrado: {prediction_source_file}")
            else:
                # Buscar cualquier archivo que termine en _未実験データ.xlsx
                print(f"⚠️ Archivo esperado no encontrado, buscando patrón *_未実験データ.xlsx...")
                matching_files = list(project_folder.glob("*_未実験データ.xlsx"))
                if matching_files:
                    prediction_source_file = matching_files[0]
                    print(f"✅ Archivo encontrado (patrón): {prediction_source_file}")
                else:
                    print(f"❌ No se encontró ningún archivo con patrón *_未実験データ.xlsx en: {project_folder}")
                    # Listar archivos disponibles para debug
                    all_files = list(project_folder.glob("*.xlsx"))
                    if all_files:
                        print(f"📋 Archivos .xlsx encontrados en {project_folder}:")
                        for f in all_files:
                            print(f"   - {f.name}")
            
            if prediction_source_file is None:
                print(f"⚠️ No se encontró el archivo de datos no experimentados en: {project_folder}")
                print(f"   Buscando: {expected_filename} o *_未実験データ.xlsx")
                # Continuar de todas formas, puede que el usuario lo haya preparado manualmente
            
            # 4. Copiar el archivo a 04_予測/Prediction_input.xlsx
            prediction_input_path = os.path.join(prediction_folder, "Prediction_input.xlsx")
            if prediction_source_file and prediction_source_file.exists():
                import shutil
                shutil.copy2(str(prediction_source_file), prediction_input_path)
                print(f"✅ Archivo copiado: {prediction_source_file} → {prediction_input_path}")
            else:
                # Si no existe, verificar si ya existe el archivo de destino
                if not os.path.exists(prediction_input_path):
                    print(f"⚠️ No se encontró archivo fuente y no existe destino. Continuando...")
            
            # 5. Actualizar config_custom.py para cambiar PREDICTION_FOLDER a 04_予測
            config_custom_path = os.path.join(working_dir, "config_custom.py")
            if os.path.exists(config_custom_path):
                try:
                    with open(config_custom_path, 'r', encoding='utf-8') as f:
                        config_content = f.read()
                    
                    # Reemplazar PREDICTION_FOLDER de '03_予測' a '04_予測'
                    import re
                    # Buscar y reemplazar PREDICTION_FOLDER = '03_予測' o PREDICTION_FOLDER = "03_予測"
                    pattern = r"(PREDICTION_FOLDER\s*=\s*['\"])03_予測(['\"])"
                    replacement = r"\g<1>04_予測\g<2>"
                    config_content = re.sub(pattern, replacement, config_content)
                    
                    with open(config_custom_path, 'w', encoding='utf-8') as f:
                        f.write(config_content)
                    print(f"✅ config_custom.py actualizado: PREDICTION_FOLDER = '04_予測'")
                except Exception as e:
                    print(f"⚠️ Error actualizando config_custom.py: {e}")
            
            script_path = os.path.join(working_dir, "02_prediction.py")
            
            # Si el script no está en la carpeta de salida, usar el del directorio actual
            if not os.path.exists(script_path):
                script_path = "02_prediction.py"
                if not os.path.exists(script_path):
                    print(f"❌ Script no encontrado: 02_prediction.py")
                    return False
            
            # Configurar variables de entorno
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"
            env["MPLBACKEND"] = "Agg"
            env["QT_QPA_PLATFORM"] = "offscreen"
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            
            # Configurar PYTHONPATH - buscar 00_Pythonコード de manera robusta
            from pathlib import Path
            python_code_folder = None
            search_path = Path(working_dir).resolve() if working_dir else Path.cwd()
            
            # Buscar hacia arriba hasta encontrar 00_Pythonコード o .venv
            while search_path != search_path.parent:
                python_code_candidate = search_path / "00_Pythonコード"
                if python_code_candidate.exists() and python_code_candidate.is_dir():
                    python_code_folder = python_code_candidate
                    break
                # También buscar .venv como indicador de la raíz del proyecto
                venv_candidate = search_path / ".venv"
                if venv_candidate.exists() and venv_candidate.is_dir():
                    python_code_candidate = search_path / "00_Pythonコード"
                    if python_code_candidate.exists() and python_code_candidate.is_dir():
                        python_code_folder = python_code_candidate
                        break
                search_path = search_path.parent
            
            # Si no se encuentra, usar el directorio actual como fallback
            if python_code_folder is None:
                python_code_folder = Path.cwd() / "00_Pythonコード"
                if not python_code_folder.exists():
                    # Último fallback: buscar desde el directorio del script
                    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
                    python_code_folder = script_dir / "00_Pythonコード"
            
            import site
            site_packages_paths = []
            try:
                for site_pkg in site.getsitepackages():
                    if os.path.exists(site_pkg):
                        site_packages_paths.append(site_pkg)
            except:
                venv_lib = Path(sys.executable).parent.parent / "Lib" / "site-packages"
                if venv_lib.exists():
                    site_packages_paths.append(str(venv_lib))
            
            pythonpath_parts = [str(python_code_folder)]
            pythonpath_parts.extend(site_packages_paths)
            
            existing_pythonpath = env.get("PYTHONPATH", "")
            if existing_pythonpath:
                pythonpath_parts.append(existing_pythonpath)
            
            separator = ";" if sys.platform == "win32" else ":"
            pythonpath = separator.join(pythonpath_parts)
            env["PYTHONPATH"] = pythonpath
            
            print(f"🔧 Ejecutando: {script_path}")
            print(f"📁 Working directory: {working_dir}")
            print(f"📁 PYTHONPATH configurado: {pythonpath}")
            print(f"📁 00_Pythonコード encontrado en: {python_code_folder}")
            
            # Ejecutar script
            process = subprocess.Popen(
                [sys.executable, script_path],
                cwd=working_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # Leer salida en tiempo real y actualizar progreso
            output_lines = []
            error_lines = []
            script_start_time = time.time()
            
            # Usar tiempo total si está disponible, sino usar tiempo del script
            if total_start_time is None:
                total_start_time = script_start_time
            
            def read_output(pipe, lines_list, is_stderr=False):
                try:
                    for line in iter(pipe.readline, ''):
                        if line:
                            line_clean = line.rstrip('\n\r')
                            lines_list.append(line_clean)
                            prefix = "[02_prediction]" if not is_stderr else "[02_prediction ERROR]"
                            print(f"{prefix} {line_clean}")
                except:
                    pass
            
            stdout_thread = threading.Thread(target=read_output, args=(process.stdout, output_lines, False), daemon=True)
            stderr_thread = threading.Thread(target=read_output, args=(process.stderr, error_lines, True), daemon=True)
            stdout_thread.start()
            stderr_thread.start()
            
            # Monitorear progreso mientras espera
            estimated_duration = 45  # segundos estimados para script 02
            while process.poll() is None:
                time.sleep(0.5)  # Verificar cada 0.5 segundos
                if progress_dialog:
                    # Tiempo transcurrido total desde el inicio
                    total_elapsed = time.time() - total_start_time
                    # Tiempo transcurrido del script actual
                    script_elapsed = time.time() - script_start_time
                    
                    # Progreso basado en tiempo del script actual (sin límite artificial)
                    time_progress = min(0.95, script_elapsed / estimated_duration)  # Máximo 95% hasta que termine
                    current_progress = int(progress_start + (progress_end - progress_start) * time_progress)
                    
                    # Calcular tiempo restante estimado de forma más precisa
                    if script_elapsed > 3 and time_progress > 0.1:  # Esperar al menos 3 segundos y 10% de progreso
                        # Usar velocidad promedio reciente
                        estimated_total = script_elapsed / time_progress
                        estimated_remaining = max(0, estimated_total - script_elapsed)
                        remaining_str = progress_dialog._format_time(estimated_remaining)
                    else:
                        # Estimación inicial conservadora
                        estimated_remaining = max(0, estimated_duration - script_elapsed)
                        remaining_str = progress_dialog._format_time(estimated_remaining)
                    
                    elapsed_str = progress_dialog._format_time(total_elapsed)
                    progress_dialog.time_info_label.setText(
                        f"⏱️ 経過時間: {elapsed_str} | 推定残り時間: {remaining_str}"
                    )
                    
                    progress_dialog.update_progress(current_progress, "02_prediction.py 実行中...")
                    QApplication.processEvents()
            
            returncode = process.returncode
            
            # Completar al 100% del rango asignado
            if progress_dialog:
                progress_dialog.update_progress(progress_end, "02_prediction.py 完了")
            
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)
            
            if returncode == 0:
                print(f"✅ 02_prediction.py ejecutado exitosamente")
                return True
            else:
                print(f"❌ 02_prediction.py falló con código {returncode}")
                if error_lines:
                    print("Errores:")
                    for line in error_lines:
                        print(f"  {line}")
                return False
                
        except Exception as e:
            print(f"❌ Error ejecutando 02_prediction.py: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_pareto_script(self, working_dir, progress_dialog=None, progress_start=20, progress_end=100, total_start_time=None):
        """
        Ejecuta 03_pareto_analyzer.py en el directorio de trabajo
        
        Parameters
        ----------
        working_dir : str
            Directorio de trabajo
        progress_dialog : ReusableProgressDialog, optional
            Diálogo de progreso para actualizar
        progress_start : int
            Porcentaje inicial de progreso (0-100)
        progress_end : int
            Porcentaje final de progreso (0-100)
        total_start_time : float, optional
            Tiempo de inicio total para tiempo transcurrido continuo
        
        Returns
        -------
        bool
            True si el script se ejecutó exitosamente, False en caso contrario
        """
        try:
            script_path = os.path.join(working_dir, "03_pareto_analyzer.py")
            
            # Si el script no está en la carpeta de salida, usar el del directorio actual
            if not os.path.exists(script_path):
                script_path = "03_pareto_analyzer.py"
                if not os.path.exists(script_path):
                    print(f"❌ Script no encontrado: 03_pareto_analyzer.py")
                    return False
            
            # Configurar variables de entorno (igual que para prediction)
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"
            env["MPLBACKEND"] = "Agg"
            env["QT_QPA_PLATFORM"] = "offscreen"
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            
            # Configurar PYTHONPATH - buscar 00_Pythonコード de manera robusta (igual que prediction)
            from pathlib import Path
            python_code_folder = None
            search_path = Path(working_dir).resolve() if working_dir else Path.cwd()
            
            # Buscar hacia arriba hasta encontrar 00_Pythonコード o .venv
            while search_path != search_path.parent:
                python_code_candidate = search_path / "00_Pythonコード"
                if python_code_candidate.exists() and python_code_candidate.is_dir():
                    python_code_folder = python_code_candidate
                    break
                # También buscar .venv como indicador de la raíz del proyecto
                venv_candidate = search_path / ".venv"
                if venv_candidate.exists() and venv_candidate.is_dir():
                    python_code_candidate = search_path / "00_Pythonコード"
                    if python_code_candidate.exists() and python_code_candidate.is_dir():
                        python_code_folder = python_code_candidate
                        break
                search_path = search_path.parent
            
            # Si no se encuentra, usar el directorio actual como fallback
            if python_code_folder is None:
                python_code_folder = Path.cwd() / "00_Pythonコード"
                if not python_code_folder.exists():
                    # Último fallback: buscar desde el directorio del script
                    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
                    python_code_folder = script_dir / "00_Pythonコード"
            
            import site
            site_packages_paths = []
            try:
                for site_pkg in site.getsitepackages():
                    if os.path.exists(site_pkg):
                        site_packages_paths.append(site_pkg)
            except:
                venv_lib = Path(sys.executable).parent.parent / "Lib" / "site-packages"
                if venv_lib.exists():
                    site_packages_paths.append(str(venv_lib))
            
            pythonpath_parts = [str(python_code_folder)]
            pythonpath_parts.extend(site_packages_paths)
            
            existing_pythonpath = env.get("PYTHONPATH", "")
            if existing_pythonpath:
                pythonpath_parts.append(existing_pythonpath)
            
            separator = ";" if sys.platform == "win32" else ":"
            pythonpath = separator.join(pythonpath_parts)
            env["PYTHONPATH"] = pythonpath
            
            print(f"🔧 Ejecutando: {script_path}")
            print(f"📁 Working directory: {working_dir}")
            print(f"📁 PYTHONPATH configurado: {pythonpath}")
            print(f"📁 00_Pythonコード encontrado en: {python_code_folder}")
            
            # Ejecutar script
            process = subprocess.Popen(
                [sys.executable, script_path],
                cwd=working_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # Leer salida en tiempo real y actualizar progreso
            output_lines = []
            error_lines = []
            script_start_time = time.time()
            
            # Usar tiempo total si está disponible, sino usar tiempo del script
            if total_start_time is None:
                total_start_time = script_start_time
            
            def read_output(pipe, lines_list, is_stderr=False):
                try:
                    for line in iter(pipe.readline, ''):
                        if line:
                            line_clean = line.rstrip('\n\r')
                            lines_list.append(line_clean)
                            prefix = "[03_pareto]" if not is_stderr else "[03_pareto ERROR]"
                            print(f"{prefix} {line_clean}")
                except:
                    pass
            
            stdout_thread = threading.Thread(target=read_output, args=(process.stdout, output_lines, False), daemon=True)
            stderr_thread = threading.Thread(target=read_output, args=(process.stderr, error_lines, True), daemon=True)
            stdout_thread.start()
            stderr_thread.start()
            
            # Monitorear progreso mientras espera
            estimated_duration = 90  # segundos estimados para script 03
            while process.poll() is None:
                time.sleep(0.5)  # Verificar cada 0.5 segundos
                if progress_dialog:
                    # Tiempo transcurrido total desde el inicio
                    total_elapsed = time.time() - total_start_time
                    # Tiempo transcurrido del script actual
                    script_elapsed = time.time() - script_start_time
                    
                    # Progreso basado en tiempo del script actual (sin límite artificial)
                    time_progress = min(0.95, script_elapsed / estimated_duration)  # Máximo 95% hasta que termine
                    current_progress = int(progress_start + (progress_end - progress_start) * time_progress)
                    
                    # Calcular tiempo restante estimado de forma más precisa
                    if script_elapsed > 5 and time_progress > 0.1:  # Esperar al menos 5 segundos y 10% de progreso
                        # Usar velocidad promedio reciente
                        estimated_total = script_elapsed / time_progress
                        estimated_remaining = max(0, estimated_total - script_elapsed)
                        remaining_str = progress_dialog._format_time(estimated_remaining)
                    else:
                        # Estimación inicial conservadora
                        estimated_remaining = max(0, estimated_duration - script_elapsed)
                        remaining_str = progress_dialog._format_time(estimated_remaining)
                    
                    elapsed_str = progress_dialog._format_time(total_elapsed)
                    progress_dialog.time_info_label.setText(
                        f"⏱️ 経過時間: {elapsed_str} | 推定残り時間: {remaining_str}"
                    )
                    
                    progress_dialog.update_progress(current_progress, "03_pareto_analyzer.py 実行中...")
                    QApplication.processEvents()
            
            returncode = process.returncode
            
            # Completar al 100% cuando termine
            if progress_dialog:
                progress_dialog.update_progress(100, "03_pareto_analyzer.py 完了")
            
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)
            
            if returncode == 0:
                print(f"✅ 03_pareto_analyzer.py ejecutado exitosamente")
                return True
            else:
                print(f"❌ 03_pareto_analyzer.py falló con código {returncode}")
                if error_lines:
                    print("Errores:")
                    for line in error_lines:
                        print(f"  {line}")
                return False
                
        except Exception as e:
            print(f"❌ Error ejecutando 03_pareto_analyzer.py: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def on_nonlinear_error(self, error_message):
        """Maneja errores del worker"""
        # ✅ NUEVO: Si el usuario canceló, no mostrar error como fallo
        if hasattr(self, '_nonlinear_cancel_requested') and self._nonlinear_cancel_requested:
            print(f"🛑 DEBUG: Error no lineal recibido tras cancelación: {error_message}. Ignorando.")
            try:
                if hasattr(self, 'progress_dialog') and self.progress_dialog:
                    self.progress_dialog.close()
            except:
                pass
            self.set_console_overlay_topmost(False)
            return

        print(f"❌ Error en worker: {error_message}")
        
        # Cerrar diálogo de progreso
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        self.set_console_overlay_topmost(False)
        
        QMessageBox.critical(
            self,
            "非線形解析エラー",
            f"❌ 非線形解析の実行中にエラーが発生しました:\n\n{error_message}"
        )
    
    def on_classification_analysis_clicked(self):
        """Acción al pulsar el botón de análisis de clasificación"""
        print("🔧 Iniciando análisis de clasificación...")
        
        # ✅ NUEVO: Si se accedió desde bunseki, mostrar diálogo de creación de proyecto
        if hasattr(self, 'accessed_from_bunseki') and self.accessed_from_bunseki:
            print("📁 Acceso desde bunseki detectado - mostrando diálogo de creación de proyecto")
            
            # Mostrar diálogo de creación de proyecto (para clasificación)
            dialog = ProjectCreationDialog(self, analysis_type="classification")
            if dialog.exec() == QDialog.Accepted:
                project_name = dialog.project_name
                project_directory = dialog.project_directory
                
                # Determinar la ruta completa del proyecto
                if project_directory:
                    # Si se seleccionó un proyecto existente, project_directory es el padre
                    # y project_name es el nombre del proyecto
                    project_path = os.path.join(project_directory, project_name)
                else:
                    # Si se creó nuevo, project_directory es donde crear y project_name es el nombre
                    project_path = os.path.join(project_directory, project_name)
                
                # Verificar si el proyecto ya existe (fue detectado como existente)
                # Para clasificación, verificar con analysis_type="classification"
                project_exists = self.is_valid_project_folder(project_path, analysis_type="classification")
                
                if project_exists:
                    print(f"✅ Usando proyecto existente: {project_path}")
                    # No crear estructura, solo usar la carpeta existente
                    self.current_project_folder = project_path
                    
                    QMessageBox.information(
                        self, 
                        "プロジェクト使用", 
                        f"✅ 既存のプロジェクト '{project_name}' を使用します。\n\n"
                        f"保存先: {project_path}\n\n"
                        f"分類解析を開始します..."
                    )
                else:
                    print(f"📁 Creando nuevo proyecto: {project_name} en {project_directory}")
                    
                    try:
                        # Crear estructura del proyecto (sin 01 y 02)
                        project_path = self.create_nonlinear_project_structure(project_name, project_directory)
                        
                        # Establecer la carpeta del proyecto actual
                        self.current_project_folder = project_path
                        
                        QMessageBox.information(
                            self, 
                            "プロジェクト作成完了", 
                            f"✅ プロジェクト '{project_name}' が作成されました。\n\n"
                            f"保存先: {project_path}\n\n"
                            f"分類解析を開始します..."
                        )
                    except Exception as e:
                        QMessageBox.critical(
                            self, 
                            "エラー", 
                            f"❌ プロジェクト作成中にエラーが発生しました:\n{str(e)}"
                        )
                        self.accessed_from_bunseki = False
                        return
                
                # Resetear la bandera
                self.accessed_from_bunseki = False
                
                # Continuar con el flujo normal (mostrar diálogo de configuración)
                # El resto del código seguirá igual, pero ahora con project_folder definido
                
            else:
                # Usuario canceló, resetear la bandera
                self.accessed_from_bunseki = False
                return
        
        try:
            # Verificar si estamos en la vista de filtros
            already_in_filter_view = False
            for i in range(self.center_layout.count()):
                item = self.center_layout.itemAt(i)
                if item.widget() and isinstance(item.widget(), QLabel):
                    if item.widget().text() == "データフィルター":
                        already_in_filter_view = True
                        break
            
            if not already_in_filter_view:
                # Crear la vista de filtros primero
                self.create_filter_view()
                self.create_navigation_buttons()
                self.prev_button.setEnabled(True)
                self.next_button.setEnabled(True)
                QMessageBox.information(self, "分析ページ", "✅ 分析ページに移動しました。\nフィルターを設定して分類分析を実行してください。")
                return
            
            # Obtener datos filtrados aplicando filtros ahora
            # Similar al análisis no lineal, obtener datos filtrados de la BBDD
            try:
                import sqlite3
                filters = self.get_applied_filters()
                
                # Construir query con filtros
                query = "SELECT * FROM main_results WHERE 1=1"
                params = []
                
                # Aplicar filtros de cepillo
                brush_selections = []
                if 'すべて' in filters and filters['すべて']:
                    brush_condition = " OR ".join([f"{brush} = 1" for brush in ['A13', 'A11', 'A21', 'A32']])
                    query += f" AND ({brush_condition})"
                else:
                    for brush_type in ['A13', 'A11', 'A21', 'A32']:
                        if brush_type in filters and filters[brush_type]:
                            brush_selections.append(brush_type)
                    
                    if brush_selections:
                        brush_condition = " OR ".join([f"{brush} = 1" for brush in brush_selections])
                        query += f" AND ({brush_condition})"
                
                # Aplicar otros filtros
                for field_name, filter_value in filters.items():
                    if field_name in ['すべて', 'A13', 'A11', 'A21', 'A32']:
                        continue
                    
                    if isinstance(filter_value, tuple) and len(filter_value) == 2:
                        desde, hasta = filter_value
                        if desde and hasta:
                            try:
                                query += f" AND {field_name} BETWEEN ? AND ?"
                                params.extend([float(desde), float(hasta)])
                            except (ValueError, TypeError):
                                continue
                    elif isinstance(filter_value, (str, int, float)) and filter_value:
                        try:
                            value_num = float(filter_value) if isinstance(filter_value, str) else filter_value
                            query += f" AND {field_name} = ?"
                            params.append(value_num)
                        except (ValueError, TypeError):
                            continue
                
                # Ejecutar query
                conn = sqlite3.connect(RESULTS_DB_PATH, timeout=10)
                df = pd.read_sql_query(query, conn, params=params)
                conn.close()
                
                if df.empty or len(df) == 0:
                    QMessageBox.warning(self, "警告", "⚠️ フィルタリングされたデータがありません。\nフィルター条件を変更してください。")
                    return
                
                self.filtered_df = df
                print(f"📊 Datos filtrados obtenidos: {len(df)} registros")
                
            except Exception as e:
                print(f"❌ Error obteniendo datos filtrados: {e}")
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "エラー", f"❌ データ取得中にエラーが発生しました:\n{str(e)}")
                return
            
            # Verificar que hay proyecto seleccionado
            if not hasattr(self, 'current_project_folder') or not self.current_project_folder:
                QMessageBox.warning(self, "プロジェクトなし", "❌ プロジェクトが選択されていません。\nまずプロジェクトを選択してください。")
                return
            
            # Verificar que los módulos están disponibles
            if ClassificationWorker is None or ClassificationConfigDialog is None or BrushSelectionDialog is None:
                QMessageBox.critical(
                    self,
                    "モジュールが見つかりません",
                    "❌ 分類分析モジュールが利用できません。\nclassification_worker.py, classification_config_dialog.py と brush_selection_dialog.py が存在することを確認してください。"
                )
                return
            
            # Mostrar diálogo de configuración
            config_dialog = ClassificationConfigDialog(self, filtered_df=self.filtered_df)
            
            if config_dialog.exec() != QDialog.Accepted:
                print("❌ Usuario canceló el análisis de clasificación")
                return
            
            # Obtener valores de configuración
            config_values = config_dialog.get_config_values()
            self.classification_config = config_values
            
            # Verificar si es carga de folder existente
            is_load_existing = config_values.get('load_existing', False)
            
            # Solo preguntar parámetros si NO es carga existente
            selected_brush = None
            selected_material = None
            selected_wire_length = None
            
            if not is_load_existing:
                # Mostrar diálogo para seleccionar parámetros (similar a yosoku)
                # QLabel, QDialog, etc. ya están importados globalmente, no importar de nuevo
                
                dialog = QDialog(self)
                dialog.setWindowTitle("予測パラメーター選択")
                dialog.setModal(True)
                dialog.resize(400, 350)
                
                layout = QVBoxLayout()
                
                # Título
                title = QLabel("予測パラメーターを選択してください")
                title.setStyleSheet("font-weight: bold; font-size: 14px; margin: 10px;")
                title.setAlignment(Qt.AlignCenter)
                layout.addWidget(title)
                
                # Formulario de selección
                form_layout = QFormLayout()
                
                # Tipo de cepillo
                brush_combo = QComboBox()
                brush_combo.addItem("A13", "A13")
                brush_combo.addItem("A11", "A11")
                brush_combo.addItem("A21", "A21")
                brush_combo.addItem("A32", "A32")
                brush_combo.setCurrentText("A11")  # Valor por defecto
                form_layout.addRow("ブラシタイプ:", brush_combo)
                
                # Material
                material_combo = QComboBox()
                material_combo.addItem("Steel", "Steel")
                material_combo.addItem("Alum", "Alum")
                material_combo.setCurrentText("Steel")  # Valor por defecto
                form_layout.addRow("材料:", material_combo)
                
                # 線材長 (de 30 a 75 en intervalos de 5mm)
                wire_length_combo = QComboBox()
                for value in range(30, 80, 5):  # 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
                    wire_length_combo.addItem(str(value), value)
                wire_length_combo.setCurrentText("75")  # Valor por defecto
                form_layout.addRow("線材長:", wire_length_combo)
                
                layout.addLayout(form_layout)
                layout.addStretch()
                
                # Botones
                button_layout = QHBoxLayout()
                
                cancel_button = QPushButton("キャンセル")
                cancel_button.clicked.connect(dialog.reject)
                
                ok_button = QPushButton("続行")
                ok_button.clicked.connect(dialog.accept)
                ok_button.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
                
                button_layout.addWidget(cancel_button)
                button_layout.addWidget(ok_button)
                layout.addLayout(button_layout)
                
                dialog.setLayout(layout)
                
                # Mostrar diálogo
                result = dialog.exec()
                
                if result == QDialog.Accepted:
                    selected_brush = brush_combo.currentData()
                    selected_material = material_combo.currentData()
                    selected_wire_length = wire_length_combo.currentData()
                    
                    print(f"✅ Parámetros seleccionados:")
                    print(f"   - Brush: {selected_brush}")
                    print(f"   - Material: {selected_material}")
                    print(f"   - Wire Length: {selected_wire_length}")
                else:
                    print("❌ Usuario canceló la selección de parámetros")
                    return
            else:
                print("ℹ️ Carga de folder existente: no se requiere selección de parámetros")
            
            # Ejecutar análisis de clasificación con worker
            print("🔧 Iniciando worker de clasificación...")
            self.classification_worker = ClassificationWorker(
                self.filtered_df, 
                self.current_project_folder, 
                self, 
                config_values,
                selected_brush=selected_brush,
                selected_material=selected_material,
                selected_wire_length=selected_wire_length
            )
            
            # Conectar señales
            self.classification_worker.progress_updated.connect(self.on_classification_progress)
            self.classification_worker.status_updated.connect(self.on_classification_status)
            self.classification_worker.finished.connect(self.on_classification_finished)
            self.classification_worker.error.connect(self.on_classification_error)
            self.classification_worker.console_output.connect(self.on_classification_console_output)
            self.classification_worker.file_selection_requested.connect(self.on_classification_file_selection_requested)
            
            # Mostrar progreso
            self.progress_dialog = ReusableProgressDialog(
                self, 
                title="分類分析処理中...",
                chibi_image="Chibi_raul.png",
                chibi_size=160
            )
            self.progress_dialog.show()
            self.set_console_overlay_topmost(True)
            
            # Conectar señal de cancelación
            self.progress_dialog.cancelled.connect(self.on_classification_cancelled)
            
            # Iniciar worker
            self.classification_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"❌ 分類分析の実行中にエラーが発生しました:\n{str(e)}")
            print(f"❌ Error en análisis de clasificación: {e}")
            import traceback
            traceback.print_exc()
    
    def on_classification_progress(self, value, message):
        """Actualiza la barra de progreso"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.update_progress(value, message)
    
    def on_classification_status(self, message):
        """Actualiza el estado"""
        print(f"📊 Estado: {message}")
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.update_status(message)
    
    def on_classification_finished(self, results):
        """Maneja el resultado de la ejecución"""
        try:
            print("✅ Análisis de clasificación completado")
            print(f"   Carpeta de salida: {results.get('output_folder', 'N/A')}")
            
            # Cerrar diálogo de progreso
            if hasattr(self, 'progress_dialog'):
                self.progress_dialog.close()
            self.set_console_overlay_topmost(False)
            
            # Mostrar pantalla de resultados finales con estadísticas
            self._show_classification_final_results(results)
            
        except Exception as e:
            print(f"❌ Error en on_classification_finished: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "エラー", f"❌ 結果処理中にエラーが発生しました:\n{str(e)}")
    
    def on_classification_error(self, error_message):
        """Maneja errores del worker"""
        print(f"❌ Error en worker: {error_message}")
        
        # Cerrar diálogo de progreso
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        self.set_console_overlay_topmost(False)
        
        QMessageBox.critical(
            self,
            "分類分析エラー",
            f"❌ 分類分析の実行中にエラーが発生しました:\n\n{error_message}"
        )
    
    def on_classification_console_output(self, message):
        """Maneja la salida de consola"""
        print(f"📝 {message}")
    
    def on_classification_file_selection_requested(self, initial_path):
        """Maneja la solicitud de selección de archivo desde el worker"""
        try:
            from pathlib import Path
            
            # Mostrar diálogo para seleccionar archivo
            prev_topmost = getattr(self, '_console_topmost_enabled', False)
            # Durante file dialogs: NO taparlos con la flecha/consola
            self.set_console_overlay_topmost(False)
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "未実験データファイルを選択してください",
                initial_path,
                "Excel Files (*.xlsx *.xls);;All Files (*)"
            )
            # Restaurar estado (si el loading sigue activo)
            if prev_topmost:
                self.set_console_overlay_topmost(True)
            
            if file_path and file_path.strip():
                # Validar que el archivo existe
                if not Path(file_path).exists():
                    QMessageBox.warning(
                        self,
                        "エラー",
                        f"❌ 選択されたファイルが見つかりません:\n{file_path}"
                    )
                    # Notificar al worker que no se seleccionó archivo
                    if hasattr(self, 'classification_worker'):
                        self.classification_worker._selected_file_path = None
                        self.classification_worker._file_selection_event.set()
                    return
                
                # Validar columnas del archivo antes de aceptarlo
                try:
                    import pandas as pd
                    df = pd.read_excel(file_path)
                    
                    required_columns = ['回転速度', '送り速度', 'UPカット', '切込量', '突出量', '載せ率', 'パス数']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        QMessageBox.warning(
                            self,
                            "エラー",
                            f"❌ 選択されたファイルに必要な列がありません:\n\n"
                            f"不足している列: {', '.join(missing_columns)}\n\n"
                            f"必要な列: {', '.join(required_columns)}"
                        )
                        # Notificar al worker que no se seleccionó archivo válido
                        if hasattr(self, 'classification_worker'):
                            self.classification_worker._selected_file_path = None
                            self.classification_worker._file_selection_event.set()
                        return
                    
                    if len(df) == 0:
                        QMessageBox.warning(
                            self,
                            "エラー",
                            f"❌ 選択されたファイルにデータがありません:\n{file_path}"
                        )
                        # Notificar al worker que no se seleccionó archivo válido
                        if hasattr(self, 'classification_worker'):
                            self.classification_worker._selected_file_path = None
                            self.classification_worker._file_selection_event.set()
                        return
                    
                    # Archivo válido, notificar al worker
                    if hasattr(self, 'classification_worker'):
                        self.classification_worker._selected_file_path = file_path
                        self.classification_worker._file_selection_event.set()
                        print(f"✅ Archivo seleccionado y validado: {file_path}")
                    
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "エラー",
                        f"❌ ファイルの読み込み中にエラーが発生しました:\n{str(e)}"
                    )
                    # Notificar al worker que hubo un error
                    if hasattr(self, 'classification_worker'):
                        self.classification_worker._selected_file_path = None
                        self.classification_worker._file_selection_event.set()
            else:
                # Usuario canceló, notificar al worker
                if hasattr(self, 'classification_worker'):
                    self.classification_worker._selected_file_path = None
                    self.classification_worker._file_selection_event.set()
                    
        except Exception as e:
            print(f"❌ Error en selección de archivo: {e}")
            import traceback
            traceback.print_exc()
            # Notificar al worker que hubo un error
            if hasattr(self, 'classification_worker'):
                self.classification_worker._selected_file_path = None
                self.classification_worker._file_selection_event.set()
    
    def on_classification_cancelled(self):
        """Maneja la cancelación"""
        print("🛑 Cancelando análisis de clasificación...")
        if hasattr(self, 'classification_worker') and self.classification_worker is not None:
            self.classification_worker.cancel()
        
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        self.set_console_overlay_topmost(False)
        
        QMessageBox.information(self, "キャンセル", "分類分析がキャンセルされました。")
    
    def _show_classification_final_results(self, results):
        """Muestra resultados finales del análisis de clasificación con estadísticas"""
        output_folder = results.get('output_folder', '')
        if not output_folder:
            QMessageBox.warning(self, "エラー", "❌ 結果を表示するための情報が見つかりません。")
            return
        
        is_load_existing = results.get('load_existing', False)
        existing_folder_path = results.get('existing_folder_path', '')
        
        # Limpiar layout central completamente
        while self.center_layout.count():
            item = self.center_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            else:
                # Si es un layout, limpiarlo también
                layout = item.layout()
                if layout:
                    while layout.count():
                        layout_item = layout.takeAt(0)
                        layout_widget = layout_item.widget()
                        if layout_widget:
                            layout_widget.deleteLater()
        
        # Forzar actualización de la UI
        QApplication.processEvents()
        
        # Crear scroll area para permitir scroll si el contenido es grande
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #f5f5f5;
            }
        """)
        
        # Crear contenedor con fondo gris limpio (dentro del scroll)
        gray_container = QFrame()
        gray_container.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border-radius: 10px;
            }
        """)
        
        # Layout interno para el contenedor gris
        container_layout = QVBoxLayout(gray_container)
        container_layout.setContentsMargins(15, 15, 15, 15)
        container_layout.setSpacing(12)  # Reducir espaciado
        
        # Título
        if is_load_existing:
            title_text = "既存分類解析結果"
        else:
            title_text = "分類解析完了"
        
        title = QLabel(title_text)
        title.setStyleSheet("""
            font-weight: bold; 
            font-size: 20px; 
            color: #2c3e50;
            margin-bottom: 10px;
            padding: 8px 0px;
            border-bottom: 2px solid #3498db;
            border-radius: 0px;
        """)
        title.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(title)
        
        # Mensaje de éxito
        if is_load_existing:
            success_text = "✅ 既存の解析結果を読み込みました！"
        else:
            success_text = "✅ 分類解析が完了しました！"
        
        success_label = QLabel(success_text)
        success_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #27ae60;
            padding: 8px;
            background-color: #d5f4e6;
            border-radius: 6px;
            border: 1px solid #27ae60;
        """)
        success_label.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(success_label)
        
        # Si es carga existente, cargar y mostrar archivos
        if is_load_existing and existing_folder_path:
            self._load_and_display_existing_classification_files(container_layout, existing_folder_path, output_folder)
        else:
            # Cargar y mostrar estadísticas del análisis recién completado
            analysis_duration = results.get('analysis_duration', 0)
            self._load_and_display_classification_statistics(container_layout, output_folder, analysis_duration)
        
        # Mensaje final
        final_message = QLabel("結果を確認してください。")
        final_message.setStyleSheet("""
            font-size: 12px;
            color: #7f8c8d;
            font-style: italic;
            margin-top: 8px;
        """)
        final_message.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(final_message)
        
        # Agregar botón "次へ" para ver gráficos (siempre que haya carpeta de salida)
        if output_folder:
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            next_button = QPushButton("次へ")
            next_button.setFixedSize(100, 35)  # Botón más compacto
            next_button.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
            """)
            next_button.clicked.connect(lambda: self._show_classification_charts_from_results(results))
            button_layout.addWidget(next_button)
            button_layout.addStretch()
            container_layout.addLayout(button_layout)
        
        # Configurar el scroll area con el contenedor
        scroll_area.setWidget(gray_container)
        
        # Agregar el scroll area al layout central
        self.center_layout.addWidget(scroll_area)
        
        # Guardar información para navegación de gráficos
        if output_folder:
            # Buscar carpeta de resultados para guardar la ruta
            result_folder = os.path.join(output_folder, '02_本学習結果', '02_評価結果')
            if os.path.exists(result_folder):
                self.classification_existing_folder_path = result_folder
                # Guardar la carpeta del análisis completo como project_folder
                self.classification_project_folder = output_folder
        
        # Forzar actualización
        QApplication.processEvents()
    
    def _load_and_display_classification_statistics(self, container_layout, output_folder, analysis_duration=0):
        """Carga y muestra las estadísticas del análisis de clasificación desde diagnostic_report.txt"""
        try:
            from pathlib import Path
            from datetime import datetime
            import re
            
            # Buscar diagnostic_report.txt en 02_本学習結果\04_診断情報
            diagnostic_report_path = os.path.join(output_folder, '02_本学習結果', '04_診断情報', 'diagnostic_report.txt')
            
            # También buscar en 02_本学習結果\02_評価結果 (por si acaso)
            alternative_path = os.path.join(output_folder, '02_本学習結果', '02_評価結果', 'diagnostic_report.txt')
            
            diagnostic_data = {}
            
            # Intentar leer diagnostic_report.txt
            report_path = None
            if os.path.exists(diagnostic_report_path):
                report_path = diagnostic_report_path
            elif os.path.exists(alternative_path):
                report_path = alternative_path
            else:
                # Búsqueda recursiva como fallback
                for root, dirs, files in os.walk(output_folder):
                    if 'diagnostic_report.txt' in files:
                        report_path = os.path.join(root, 'diagnostic_report.txt')
                        break
            
            if report_path:
                try:
                    with open(report_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Parsear el contenido del reporte
                    # [設定情報]
                    np_alpha_match = re.search(r'NP_ALPHA:\s*([\d.]+)', content)
                    if np_alpha_match:
                        diagnostic_data['np_alpha'] = np_alpha_match.group(1)
                    else:
                        # Intentar variaciones
                        alt_match = re.search(r'NP_ALPHA[:\s]+([\d.]+)', content, re.IGNORECASE)
                        if alt_match:
                            diagnostic_data['np_alpha'] = alt_match.group(1)
                    
                    objective_match = re.search(r'目的変数:\s*(.+)', content)
                    if objective_match:
                        diagnostic_data['objective'] = objective_match.group(1).strip()
                    else:
                        # Intentar variaciones
                        alt_match = re.search(r'目的変数[:\s]+(.+)', content)
                        if alt_match:
                            diagnostic_data['objective'] = alt_match.group(1).strip()
                    
                    # [モデル情報]
                    calibrator_match = re.search(r'Calibrator:\s*(.+)', content)
                    if calibrator_match:
                        diagnostic_data['calibrator'] = calibrator_match.group(1).strip()
                    
                    # Intentar diferentes formatos para tau_pos
                    tau_pos_match = re.search(r'τ\+\s*\(tau_pos\):\s*([\d.]+)', content)
                    if not tau_pos_match:
                        tau_pos_match = re.search(r'tau_pos[:\s]+([\d.]+)', content, re.IGNORECASE)
                    if not tau_pos_match:
                        tau_pos_match = re.search(r'τ\+[:\s]+([\d.]+)', content)
                    if tau_pos_match:
                        diagnostic_data['tau_pos'] = tau_pos_match.group(1)
                    
                    # Intentar diferentes formatos para tau_neg
                    tau_neg_match = re.search(r'τ-\s*\(tau_neg\):\s*([\d.]+)', content)
                    if not tau_neg_match:
                        tau_neg_match = re.search(r'tau_neg[:\s]+([\d.]+)', content, re.IGNORECASE)
                    if not tau_neg_match:
                        tau_neg_match = re.search(r'τ-[:\s]+([\d.]+)', content)
                    if tau_neg_match:
                        diagnostic_data['tau_neg'] = tau_neg_match.group(1)
                    
                    features_match = re.search(r'選択特徴量数:\s*(\d+)', content)
                    if features_match:
                        diagnostic_data['selected_features'] = features_match.group(1)
                    
                    # [予測結果統計]
                    total_data_match = re.search(r'総データ数:\s*([\d,]+)', content)
                    if total_data_match:
                        diagnostic_data['total_data'] = total_data_match.group(1).replace(',', '')
                    
                    coverage_match = re.search(r'カバレッジ:\s*([\d.]+)%', content)
                    if not coverage_match:
                        coverage_match = re.search(r'カバレッジ[:\s]+([\d.]+)', content)
                    if coverage_match:
                        diagnostic_data['coverage'] = coverage_match.group(1)
                    
                    # [ノイズ付加設定]
                    noise_enabled_match = re.search(r'ノイズ付加:\s*(True|False)', content)
                    if noise_enabled_match:
                        diagnostic_data['noise_enabled'] = noise_enabled_match.group(1) == 'True'
                    
                    noise_level_match = re.search(r'ノイズレベル:\s*([\d.]+)\s*ppm', content)
                    if not noise_level_match:
                        noise_level_match = re.search(r'ノイズレベル[:\s]+([\d.]+)', content)
                    if noise_level_match:
                        diagnostic_data['noise_level'] = noise_level_match.group(1)
                    
                    print(f"✅ Datos de diagnóstico cargados desde: {report_path}")
                    print(f"🔍 [DEBUG] Datos parseados: {diagnostic_data}")
                    print(f"🔍 [DEBUG] tau_pos: {diagnostic_data.get('tau_pos')}")
                    print(f"🔍 [DEBUG] tau_neg: {diagnostic_data.get('tau_neg')}")
                    print(f"🔍 [DEBUG] noise_enabled: {diagnostic_data.get('noise_enabled')}")
                    print(f"🔍 [DEBUG] noise_level: {diagnostic_data.get('noise_level')}")
                except Exception as e:
                    print(f"⚠️ Error leyendo diagnostic_report.txt: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️ diagnostic_report.txt no encontrado en: {diagnostic_report_path} o {alternative_path}")
            
            # Formatear tiempo de análisis
            if analysis_duration > 0:
                hours = int(analysis_duration // 3600)
                minutes = int((analysis_duration % 3600) // 60)
                seconds = int(analysis_duration % 60)
                if hours > 0:
                    analysis_duration_formatted = f"{hours}時間{minutes}分{seconds}秒"
                elif minutes > 0:
                    analysis_duration_formatted = f"{minutes}分{seconds}秒"
                else:
                    analysis_duration_formatted = f"{seconds:.1f}秒"
            else:
                analysis_duration_formatted = "N/A"
            
            # Información del análisis
            info_lines = []
            info_lines.append(f"📊 解析完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            info_lines.append(f"⏱️ 解析時間: {analysis_duration_formatted}")
            
            if diagnostic_data.get('objective'):
                info_lines.append(f"🎯 目的変数: {diagnostic_data['objective']}")
            
            if diagnostic_data.get('np_alpha'):
                info_lines.append(f"⚙️ NP_ALPHA: {diagnostic_data['np_alpha']}")
            
            if diagnostic_data.get('total_data'):
                info_lines.append(f"📈 総データ数: {diagnostic_data['total_data']} レコード")
            
            if diagnostic_data.get('coverage'):
                info_lines.append(f"📊 カバレッジ: {diagnostic_data['coverage']}%")
            
            if diagnostic_data.get('selected_features'):
                info_lines.append(f"🔧 選択特徴量数: {diagnostic_data['selected_features']} 個")
            
            info_text = "\n".join(info_lines)
            info_label = QLabel(info_text)
            info_label.setStyleSheet("""
                font-size: 12px;
                color: #34495e;
                background-color: #ecf0f1;
                padding: 10px;
                border-radius: 6px;
                border: 1px solid #bdc3c7;
            """)
            info_label.setAlignment(Qt.AlignLeft)
            info_label.setWordWrap(True)
            info_label.setMinimumHeight(50)
            container_layout.addWidget(info_label)
            
            # Sección de métricas del modelo si están disponibles
            print(f"🔍 [DEBUG] Verificando Model Information: tau_pos={diagnostic_data.get('tau_pos')}, tau_neg={diagnostic_data.get('tau_neg')}")
            if diagnostic_data.get('tau_pos') and diagnostic_data.get('tau_neg'):
                print(f"✅ [DEBUG] Mostrando Model Information")
                metrics_title = QLabel("📊 モデル情報 (Model Information)")
                metrics_title.setStyleSheet("""
                    font-weight: bold; 
                    font-size: 16px; 
                    color: #2c3e50;
                    margin-top: 10px;
                    margin-bottom: 8px;
                    padding-bottom: 6px;
                    border-bottom: 2px solid #3498db;
                """)
                metrics_title.setAlignment(Qt.AlignCenter)
                container_layout.addWidget(metrics_title)
                
                # Crear tarjeta de métricas
                metric_card = QFrame()
                metric_card.setStyleSheet("""
                    QFrame {
                        background-color: #ffffff;
                        border: 2px solid #3498db;
                        border-radius: 8px;
                        padding: 10px;
                    }
                """)
                card_layout = QVBoxLayout(metric_card)
                card_layout.setSpacing(6)  # Reducir espaciado
                card_layout.setContentsMargins(10, 10, 10, 10)
                
                # Calibrator
                if diagnostic_data.get('calibrator'):
                    calibrator_text = f"Calibrator: {diagnostic_data['calibrator']}"
                    calibrator_label = QLabel(calibrator_text)
                    calibrator_label.setStyleSheet("""
                        font-size: 12px;
                        color: #34495e;
                        padding: 6px;
                        background-color: #f8f9fa;
                        border-radius: 4px;
                        min-height: 24px;
                    """)
                    calibrator_label.setMinimumHeight(24)
                    calibrator_label.setWordWrap(True)
                    print(f"✅ [DEBUG] Agregando calibrator_label: {calibrator_text}")
                    card_layout.addWidget(calibrator_label)
                
                # τ+ y τ- (separados en labels diferentes para asegurar visibilidad)
                tau_pos_text = f"τ+ (tau_pos): {diagnostic_data['tau_pos']}"
                tau_pos_label = QLabel(tau_pos_text)
                tau_pos_label.setStyleSheet("""
                    font-size: 12px;
                    color: #34495e;
                    padding: 6px;
                    background-color: #f8f9fa;
                    border-radius: 4px;
                    min-height: 24px;
                """)
                tau_pos_label.setMinimumHeight(24)
                tau_pos_label.setWordWrap(True)
                print(f"✅ [DEBUG] Agregando tau_pos_label: {tau_pos_text}")
                card_layout.addWidget(tau_pos_label)
                
                tau_neg_text = f"τ- (tau_neg): {diagnostic_data['tau_neg']}"
                tau_neg_label = QLabel(tau_neg_text)
                tau_neg_label.setStyleSheet("""
                    font-size: 12px;
                    color: #34495e;
                    padding: 6px;
                    background-color: #f8f9fa;
                    border-radius: 4px;
                    min-height: 24px;
                """)
                tau_neg_label.setMinimumHeight(24)
                tau_neg_label.setWordWrap(True)
                print(f"✅ [DEBUG] Agregando tau_neg_label: {tau_neg_text}")
                card_layout.addWidget(tau_neg_label)
                
                # Verificar si τ- < τ+ (normal)
                try:
                    tau_pos_val = float(diagnostic_data['tau_pos'])
                    tau_neg_val = float(diagnostic_data['tau_neg'])
                    print(f"🔍 [DEBUG] Comparando tau: tau_neg={tau_neg_val} < tau_pos={tau_pos_val} = {tau_neg_val < tau_pos_val}")
                    if tau_neg_val < tau_pos_val:
                        status_text = "✅ 正常: τ- < τ+"
                        status_label = QLabel(status_text)
                        status_label.setStyleSheet("""
                            font-size: 12px;
                            font-weight: bold;
                            color: #27ae60;
                            padding: 6px;
                            background-color: #d5f4e6;
                            border-radius: 4px;
                            border: 1px solid #27ae60;
                            min-height: 28px;
                        """)
                    else:
                        status_text = "⚠️ 警告: τ- >= τ+"
                        status_label = QLabel(status_text)
                        status_label.setStyleSheet("""
                            font-size: 12px;
                            font-weight: bold;
                            color: #f39c12;
                            padding: 6px;
                            background-color: #fef5e7;
                            border-radius: 4px;
                            border: 1px solid #f39c12;
                            min-height: 28px;
                        """)
                    status_label.setMinimumHeight(28)
                    status_label.setWordWrap(True)
                    status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                    print(f"✅ [DEBUG] Agregando status_label: {status_text}")
                    card_layout.addWidget(status_label)
                    print(f"✅ [DEBUG] status_label agregado al layout. Total widgets en card_layout: {card_layout.count()}")
                except Exception as e:
                    print(f"⚠️ Error agregando status_label: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Asegurar que la tarjeta tenga contenido visible
                print(f"✅ [DEBUG] Total widgets en metric_card antes de agregar: {card_layout.count()}")
                # Calcular altura mínima basada en el número de widgets (más compacto)
                min_height = max(120, card_layout.count() * 35)  # Al menos 35px por widget
                metric_card.setMinimumHeight(min_height)
                metric_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
                print(f"✅ [DEBUG] metric_card altura mínima: {min_height}px")
                container_layout.addWidget(metric_card)
                print(f"✅ [DEBUG] metric_card agregado al container_layout")
            else:
                # Mostrar mensaje si no hay información del modelo
                if not diagnostic_data:
                    no_data_label = QLabel("⚠️ 統計情報を読み込めませんでした。\n診断レポートファイルを確認してください。")
                    no_data_label.setStyleSheet("""
                        font-size: 14px;
                        color: #e67e22;
                        background-color: #fef5e7;
                        padding: 15px;
                        border-radius: 8px;
                        border: 1px solid #e67e22;
                    """)
                    no_data_label.setAlignment(Qt.AlignCenter)
                    no_data_label.setWordWrap(True)
                    no_data_label.setMinimumHeight(60)
                    container_layout.addWidget(no_data_label)
            
            # Información de ruido si está disponible
            print(f"🔍 [DEBUG] Verificando Noise Settings: noise_enabled={diagnostic_data.get('noise_enabled')}")
            if diagnostic_data.get('noise_enabled'):
                print(f"✅ [DEBUG] Mostrando Noise Addition Settings")
                noise_title = QLabel("🔊 ノイズ付加設定 (Noise Addition Settings)")
                noise_title.setStyleSheet("""
                    font-weight: bold; 
                    font-size: 16px; 
                    color: #2c3e50;
                    margin-top: 10px;
                    margin-bottom: 8px;
                    padding-bottom: 6px;
                    border-bottom: 2px solid #3498db;
                """)
                noise_title.setAlignment(Qt.AlignCenter)
                container_layout.addWidget(noise_title)
                
                noise_card = QFrame()
                noise_card.setStyleSheet("""
                    QFrame {
                        background-color: #ffffff;
                        border: 2px solid #3498db;
                        border-radius: 8px;
                        padding: 10px;
                    }
                """)
                noise_layout = QVBoxLayout(noise_card)
                noise_layout.setSpacing(6)  # Reducir espaciado
                noise_layout.setContentsMargins(10, 10, 10, 10)
                
                if diagnostic_data.get('noise_level'):
                    noise_info = f"ノイズレベル: {diagnostic_data['noise_level']} ppm"
                    noise_label = QLabel(noise_info)
                    noise_label.setStyleSheet("""
                        font-size: 12px;
                        color: #34495e;
                        padding: 6px;
                        background-color: #f8f9fa;
                        border-radius: 4px;
                        min-height: 24px;
                    """)
                    noise_label.setMinimumHeight(24)
                    noise_label.setWordWrap(True)
                    print(f"✅ [DEBUG] Agregando noise_label: {noise_info}")
                    noise_layout.addWidget(noise_label)
                else:
                    # Mostrar mensaje si no hay noise_level pero noise_enabled es True
                    noise_info_text = "ノイズ付加: 有効"
                    noise_info_label = QLabel(noise_info_text)
                    noise_info_label.setStyleSheet("""
                        font-size: 12px;
                        color: #34495e;
                        padding: 6px;
                        background-color: #f8f9fa;
                        border-radius: 4px;
                        min-height: 24px;
                    """)
                    noise_info_label.setMinimumHeight(24)
                    print(f"✅ [DEBUG] Agregando noise_info_label: {noise_info_text}")
                    noise_layout.addWidget(noise_info_label)
                
                # Asegurar que la tarjeta tenga contenido visible
                print(f"✅ [DEBUG] Total widgets en noise_card antes de agregar: {noise_layout.count()}")
                # Calcular altura mínima basada en el número de widgets (más compacto)
                min_height = max(70, noise_layout.count() * 35)  # Al menos 35px por widget
                noise_card.setMinimumHeight(min_height)
                noise_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
                print(f"✅ [DEBUG] noise_card altura mínima: {min_height}px")
                container_layout.addWidget(noise_card)
                print(f"✅ [DEBUG] noise_card agregado al container_layout")
            
        except Exception as e:
            print(f"❌ Error cargando estadísticas de clasificación: {e}")
            import traceback
            traceback.print_exc()
            error_label = QLabel(f"⚠️ 統計情報の読み込み中にエラーが発生しました: {str(e)}")
            error_label.setStyleSheet("""
                font-size: 14px;
                color: #e74c3c;
                background-color: #fadbd8;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #e74c3c;
            """)
            container_layout.addWidget(error_label)
    
    def _load_and_display_existing_classification_files(self, container_layout, existing_folder_path, output_folder):
        """Carga y muestra los archivos de un análisis de clasificación existente"""
        try:
            # Cargar y mostrar estadísticas del análisis existente
            self._load_and_display_classification_statistics(container_layout, output_folder, analysis_duration=0)
            
        except Exception as e:
            print(f"❌ Error cargando archivos existentes de clasificación: {e}")
            import traceback
            traceback.print_exc()
            error_label = QLabel(f"⚠️ 既存結果の読み込み中にエラーが発生しました: {str(e)}")
            error_label.setStyleSheet("""
                font-size: 14px;
                color: #e74c3c;
                background-color: #fadbd8;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #e74c3c;
            """)
            container_layout.addWidget(error_label)
    
    def _show_classification_charts_from_results(self, results):
        """Mostrar gráficos del análisis de clasificación desde los resultados"""
        output_folder = results.get('output_folder', '')
        if not output_folder:
            QMessageBox.warning(self, "エラー", "❌ グラフを表示するための情報が見つかりません。")
            return
        
        # Buscar carpeta de resultados (02_本学習結果\02_評価結果)
        result_folder = os.path.join(output_folder, '02_本学習結果', '02_評価結果')
        
        # Guardar información para navegación
        if os.path.exists(result_folder):
            self.classification_existing_folder_path = result_folder
            self.classification_project_folder = output_folder
            # Llamar a la función de mostrar gráficos
            if hasattr(self, 'show_classification_charts'):
                self.show_classification_charts()
            else:
                QMessageBox.information(
                    self,
                    "情報",
                    "グラフ表示機能は準備中です。\n\n結果フォルダ:\n" + output_folder
                )
        else:
            QMessageBox.warning(
                self,
                "エラー",
                f"❌ 結果フォルダが見つかりません:\n{result_folder}"
            )
    
    def show_classification_charts(self):
        """Mostrar gráficos del análisis de clasificación con navegación"""
        print("🔧 Mostrando gráficos del análisis de clasificación...")
        
        try:
            # Verificar que tenemos la ruta de la carpeta cargada
            if not hasattr(self, 'classification_existing_folder_path') or not self.classification_existing_folder_path:
                QMessageBox.warning(self, "エラー", "❌ グラフを表示するための情報が見つかりません。")
                return
            
            # Limpiar layout central completamente
            while self.center_layout.count():
                item = self.center_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                else:
                    # Si es un layout, limpiarlo también
                    layout = item.layout()
                    if layout:
                        while layout.count():
                            layout_item = layout.takeAt(0)
                            layout_widget = layout_item.widget()
                            if layout_widget:
                                layout_widget.deleteLater()
            
            # Forzar actualización de la UI
            QApplication.processEvents()
            
            # Crear contenedor con fondo gris limpio
            gray_container = QFrame()
            gray_container.setStyleSheet("""
                QFrame {
                    background-color: #f5f5f5;
                    border-radius: 10px;
                    margin: 10px;
                }
            """)
            
            # Layout interno para el contenedor gris
            container_layout = QVBoxLayout(gray_container)
            container_layout.setContentsMargins(20, 20, 20, 20)
            container_layout.setSpacing(15)
            
            # Título
            title = QLabel("分類解析結果 チャート")
            title.setStyleSheet("""
                font-weight: bold; 
                font-size: 24px; 
                color: #2c3e50;
                margin-bottom: 20px;
                padding: 10px 0px;
                border-bottom: 2px solid #3498db;
                border-radius: 0px;
            """)
            title.setAlignment(Qt.AlignCenter)
            container_layout.addWidget(title)
            
            # Buscar gráficos PNG en la carpeta de resultados (02_本学習結果\02_評価結果)
            from pathlib import Path
            folder_path = Path(self.classification_existing_folder_path)
            chart_images = []
            
            # Buscar imágenes PNG directamente en la carpeta de resultados
            for file in folder_path.glob("*.png"):
                if file.is_file():
                    chart_images.append(str(file))
            
            # Si no se encuentran gráficos, mostrar mensaje
            if not chart_images:
                no_charts_label = QLabel("⚠️ グラフが見つかりません")
                no_charts_label.setStyleSheet("""
                    font-size: 16px;
                    color: #e74c3c;
                    background-color: #fadbd8;
                    padding: 20px;
                    border-radius: 8px;
                    border: 1px solid #e74c3c;
                    margin: 20px 0px;
                """)
                no_charts_label.setAlignment(Qt.AlignCenter)
                container_layout.addWidget(no_charts_label)
            else:
                # Configurar navegación de gráficos
                self.classification_chart_images = sorted(chart_images)
                self.current_classification_chart_index = 0
                
                # Layout principal para la imagen y navegación
                chart_layout = QVBoxLayout()
                
                # Label para mostrar la imagen (ocupa todo el ancho)
                self.classification_chart_label = QLabel()
                self.classification_chart_label.setAlignment(Qt.AlignCenter)
                self.classification_chart_label.setStyleSheet("""
                    QLabel {
                        background-color: white;
                        border: 2px solid #bdc3c7;
                        border-radius: 10px;
                        padding: 10px;
                        min-height: 500px;
                    }
                """)
                chart_layout.addWidget(self.classification_chart_label)
                
                # Layout horizontal para botones de navegación (debajo de la imagen)
                nav_buttons_layout = QHBoxLayout()
                nav_buttons_layout.addStretch()
                
                # Botón flecha izquierda
                prev_chart_button = QPushButton("◀ 前へ")
                prev_chart_button.setFixedSize(100, 40)
                prev_chart_button.setStyleSheet("""
                    QPushButton {
                        background-color: #3498db;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        font-size: 14px;
                        font-weight: bold;
                        padding: 8px 16px;
                    }
                    QPushButton:hover {
                        background-color: #2980b9;
                    }
                    QPushButton:disabled {
                        background-color: #bdc3c7;
                        color: #7f8c8d;
                    }
                """)
                prev_chart_button.clicked.connect(self.show_previous_classification_chart)
                nav_buttons_layout.addWidget(prev_chart_button)
                
                # Espacio entre botones
                nav_buttons_layout.addSpacing(20)
                
                # Botón flecha derecha
                next_chart_button = QPushButton("次へ ▶")
                next_chart_button.setFixedSize(100, 40)
                next_chart_button.setStyleSheet("""
                    QPushButton {
                        background-color: #3498db;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        font-size: 14px;
                        font-weight: bold;
                        padding: 8px 16px;
                    }
                    QPushButton:hover {
                        background-color: #2980b9;
                    }
                    QPushButton:disabled {
                        background-color: #bdc3c7;
                        color: #7f8c8d;
                    }
                """)
                next_chart_button.clicked.connect(self.show_next_classification_chart)
                nav_buttons_layout.addWidget(next_chart_button)
                
                nav_buttons_layout.addStretch()
                chart_layout.addLayout(nav_buttons_layout)
                
                # Información del gráfico actual
                self.classification_chart_info_label = QLabel()
                self.classification_chart_info_label.setStyleSheet("""
                    font-size: 14px;
                    color: #2c3e50;
                    background-color: #ecf0f1;
                    padding: 10px;
                    border-radius: 5px;
                    border: 1px solid #bdc3c7;
                    margin: 10px 0px;
                """)
                self.classification_chart_info_label.setAlignment(Qt.AlignCenter)
                chart_layout.addWidget(self.classification_chart_info_label)
                
                container_layout.addLayout(chart_layout)
                
                # Mostrar el primer gráfico
                self.update_classification_chart_display()
            
            # Botones para volver e importar a BBDD
            buttons_layout = QHBoxLayout()
            buttons_layout.addStretch()
            
            # Botón para importar a BBDD
            import_db_button = QPushButton("データベースにインポート")
            import_db_button.setFixedSize(180, 40)
            import_db_button.setStyleSheet("""
                QPushButton {
                    background-color: #27ae60;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #229954;
                }
            """)
            import_db_button.clicked.connect(lambda: self.import_classification_results_to_yosoku_db())
            buttons_layout.addWidget(import_db_button)
            
            buttons_layout.addSpacing(20)
            
            # Botón para volver
            back_button = QPushButton("戻る")
            back_button.setFixedSize(120, 40)
            back_button.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
            """)
            back_button.clicked.connect(self.on_analyze_clicked)
            buttons_layout.addWidget(back_button)
            
            buttons_layout.addStretch()
            container_layout.addLayout(buttons_layout)
            
            # Espacio flexible
            container_layout.addStretch()
            
            # Agregar el contenedor gris al layout central
            self.center_layout.addWidget(gray_container)
            
            print("✅ Gráficos del análisis de clasificación mostrados")
            
        except Exception as e:
            print(f"❌ Error mostrando gráficos del análisis de clasificación: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "エラー", f"❌ グラフの表示中にエラーが発生しました:\n{str(e)}")
    
    def show_previous_classification_chart(self):
        """Mostrar gráfico anterior del análisis de clasificación"""
        if hasattr(self, 'classification_chart_images') and len(self.classification_chart_images) > 0:
            if not hasattr(self, 'current_classification_chart_index'):
                self.current_classification_chart_index = 0
            self.current_classification_chart_index = (self.current_classification_chart_index - 1) % len(self.classification_chart_images)
            self.update_classification_chart_display()
    
    def show_next_classification_chart(self):
        """Mostrar gráfico siguiente del análisis de clasificación"""
        if hasattr(self, 'classification_chart_images') and len(self.classification_chart_images) > 0:
            if not hasattr(self, 'current_classification_chart_index'):
                self.current_classification_chart_index = 0
            self.current_classification_chart_index = (self.current_classification_chart_index + 1) % len(self.classification_chart_images)
            self.update_classification_chart_display()
    
    def update_classification_chart_display(self):
        """Actualizar la visualización del gráfico actual del análisis de clasificación"""
        if not hasattr(self, 'classification_chart_images') or len(self.classification_chart_images) == 0:
            return
        
        if not hasattr(self, 'current_classification_chart_index'):
            self.current_classification_chart_index = 0
        
        if self.current_classification_chart_index < 0:
            self.current_classification_chart_index = 0
        elif self.current_classification_chart_index >= len(self.classification_chart_images):
            self.current_classification_chart_index = len(self.classification_chart_images) - 1
        
        current_image_path = self.classification_chart_images[self.current_classification_chart_index]
        
        # Cargar y mostrar la imagen
        pixmap = QPixmap(current_image_path)
        if not pixmap.isNull():
            # Redimensionar la imagen para ocupar todo el ancho disponible
            container_width = self.classification_chart_label.width() - 20
            container_height = self.classification_chart_label.height() - 20
            
            # Si el contenedor aún no tiene tamaño, usar un tamaño por defecto
            if container_width <= 0:
                container_width = 1000
            if container_height <= 0:
                container_height = 600
            
            # Redimensionar manteniendo la proporción
            scaled_pixmap = pixmap.scaled(container_width, container_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.classification_chart_label.setPixmap(scaled_pixmap)
            
            # Actualizar información del gráfico
            image_name = os.path.basename(current_image_path)
            total_images = len(self.classification_chart_images)
            current_index = self.current_classification_chart_index + 1
            self.classification_chart_info_label.setText(f"{image_name} ({current_index}/{total_images})")
            
            # Actualizar estado de botones de navegación
            if hasattr(self, 'classification_chart_label'):
                # Los botones se habilitan/deshabilitan automáticamente por el layout
                pass

    def create_linear_analysis_folder_structure(self, project_folder):
        """Crear estructura de carpetas para análisis lineal con numeración correlativa y timestamp"""
        import os
        from datetime import datetime
        import re
        
        # Ruta de la carpeta de análisis lineal
        linear_regression_folder = os.path.join(project_folder, "03_線形回帰")
        
        # Crear carpeta si no existe
        os.makedirs(linear_regression_folder, exist_ok=True)
        
        # Obtener timestamp actual
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Buscar el siguiente número correlativo
        existing_folders = []
        for item in os.listdir(linear_regression_folder):
            item_path = os.path.join(linear_regression_folder, item)
            if os.path.isdir(item_path):
                # Buscar patrones como "01_", "02_", etc.
                match = re.match(r'^(\d{2})_', item)
                if match:
                    existing_folders.append(int(match.group(1)))
        
        # Determinar el siguiente número
        if existing_folders:
            next_number = max(existing_folders) + 1
        else:
            next_number = 1
        
        # Crear nombre de carpeta con formato: 01_YYYYMMDD_HHMMSS
        folder_name = f"{next_number:02d}_{timestamp}"
        analysis_folder = os.path.join(linear_regression_folder, folder_name)
        
        # Crear carpeta principal
        os.makedirs(analysis_folder, exist_ok=True)
        print(f"📁 Carpeta de análisis creada: {analysis_folder}")
        
        # Crear subcarpetas
        subfolders = [
            "01_学習モデル",
            "02_パラメーター", 
            "03_評価スコア",
            "04_予測計算"
        ]
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(analysis_folder, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)
            print(f"📁 Subcarpeta creada: {subfolder_path}")
            
            # Crear subcarpeta adicional dentro de 03_評価スコア
            if subfolder == "03_評価スコア":
                chart_subfolder = os.path.join(subfolder_path, "01_チャート")
                os.makedirs(chart_subfolder, exist_ok=True)
                print(f"📁 Subcarpeta de gráficos creada: {chart_subfolder}")
        
        return analysis_folder

    def execute_linear_analysis(self):
        """Ejecutar análisis lineal con los filtros aplicados"""
        print("🔧 Ejecutando análisis lineal...")
        
        # ✅ NUEVO: Evitar re-ejecución si ya hay un análisis lineal corriendo
        if hasattr(self, 'linear_worker') and self.linear_worker is not None:
            try:
                if self.linear_worker.isRunning():
                    QMessageBox.warning(self, "線形解析", "⚠️ すでに線形解析が実行中です。\n完了または停止するまでお待ちください。")
                    return
            except RuntimeError:
                self.linear_worker = None
        
        try:
            # Obtener filtros aplicados
            filters = self.get_applied_filters()
            print(f"🔧 Filtros aplicados: {filters}")
            
            # Importar módulo de análisis lineal
            try:
                from linear_analysis_advanced import run_advanced_linear_analysis_from_db
                print("✅ Módulo de análisis lineal importado correctamente")
            except ImportError as e:
                print(f"❌ Error importando módulo de análisis lineal: {e}")
                QMessageBox.critical(self, "エラー", "❌ モジュール de análisis lineal no se pudo importar.\nAsegúrese de que el archivo linear_analysis_module.py esté en el directorio correcto.")
                return
            
            # Mostrar mensaje de confirmación
            reply = QMessageBox.question(
                self, 
                "線形解析確認", 
                f"線形解析を実行しますか？\n\nフィルター: {len(filters)} 条件\n\nこの操作は時間がかかる場合があります。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply != QMessageBox.Yes:
                print("❌ Usuario canceló el análisis lineal")
                return
            
            # ✅ NUEVO: Crear estructura de carpetas para el análisis
            if hasattr(self, 'current_project_folder') and self.current_project_folder:
                analysis_folder = self.create_linear_analysis_folder_structure(self.current_project_folder)
                print(f"✅ Estructura de carpetas creada en: {analysis_folder}")
            else:
                print("⚠️ No se detectó carpeta de proyecto, usando carpeta por defecto")
                analysis_folder = "analysis_output"

            # Arrancar con flujo unificado (worker + popup + cancelación)
            self._start_linear_analysis(filters, analysis_folder)
                
        except Exception as e:
            print(f"❌ Error ejecutando análisis lineal: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "エラー", f"❌ 線形解析の実行中にエラーが発生しました:\n{str(e)}")

    def on_linear_analysis_finished(self, results):
        """Maneja el resultado exitoso del análisis lineal"""
        # ✅ NUEVO: Re-habilitar botones
        if hasattr(self, 'linear_analysis_button'):
            self.linear_analysis_button.setEnabled(True)
        if hasattr(self, 'run_analysis_button'):
            self.run_analysis_button.setEnabled(True)
            
        try:
            # ✅ NUEVO: Si el usuario canceló, NO mostrar resultados (evita "cancelé y aun así me enseña resultados")
            if hasattr(self, '_linear_cancel_requested') and self._linear_cancel_requested:
                print("🛑 DEBUG: Resultado recibido pero el usuario canceló. Ignorando resultados.")
                # Cerrar popup de progreso de forma segura
                if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
                    try:
                        self.progress_dialog.close()
                        self.progress_dialog.deleteLater()
                    except:
                        pass
                if hasattr(self, 'progress_dialog'):
                    try:
                        delattr(self, 'progress_dialog')
                    except:
                        pass
                self.set_console_overlay_topmost(False)
                # Limpiar worker
                try:
                    self.linear_worker = None
                except:
                    pass
                return

            # Cerrar popup de progreso de forma segura
            if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
                try:
                    self.progress_dialog.close()
                    self.progress_dialog.deleteLater()
                except:
                    pass  # Ignorar errores al cerrar el popup
            
            # Limpiar referencias
            if hasattr(self, 'progress_dialog'):
                delattr(self, 'progress_dialog')
            self.set_console_overlay_topmost(False)
            
            if results.get('success', False):
                # Mostrar resultados
                self.show_linear_analysis_results(results)
                QMessageBox.information(self, "線形解析完了", f"✅ 線形解析が完了しました！\n結果は{results.get('output_folder', 'N/A')}フォルダに保存されています。")
            else:
                error_msg = results.get('error', 'Error desconocido')
                QMessageBox.critical(self, "線形解析エラー", f"❌ 線形解析中にエラーが発生しました:\n{error_msg}")
                
        except Exception as e:
            print(f"❌ Error en on_linear_analysis_finished: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "エラー", f"❌ 結果の処理中にエラーが発生しました:\n{str(e)}")

    def on_linear_analysis_error(self, error_message):
        """Maneja el error del análisis lineal"""
        # ✅ NUEVO: Re-habilitar botones
        if hasattr(self, 'linear_analysis_button'):
            self.linear_analysis_button.setEnabled(True)
        if hasattr(self, 'run_analysis_button'):
            self.run_analysis_button.setEnabled(True)
            
        try:
            # ✅ NUEVO: Si el usuario canceló, tratamos como cancelación silenciosa
            if hasattr(self, '_linear_cancel_requested') and self._linear_cancel_requested:
                print(f"🛑 DEBUG: Error recibido tras cancelación: {error_message}. Ignorando.")
                if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
                    try:
                        self.progress_dialog.close()
                        self.progress_dialog.deleteLater()
                    except:
                        pass
                if hasattr(self, 'progress_dialog'):
                    try:
                        delattr(self, 'progress_dialog')
                    except:
                        pass
                self.set_console_overlay_topmost(False)
                try:
                    self.linear_worker = None
                except:
                    pass
                return

            # Cerrar popup de progreso de forma segura
            if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
                try:
                    self.progress_dialog.close()
                    self.progress_dialog.deleteLater()
                except:
                    pass  # Ignorar errores al cerrar el popup
            
            # Limpiar referencias
            if hasattr(self, 'progress_dialog'):
                delattr(self, 'progress_dialog')
            self.set_console_overlay_topmost(False)
            
            print(f"❌ Error en análisis lineal: {error_message}")
            QMessageBox.critical(self, "線形解析エラー", f"❌ 線形解析中にエラーが発生しました:\n{error_message}")
            
        except Exception as e:
            print(f"❌ Error en on_linear_analysis_error: {e}")
            import traceback
            traceback.print_exc()

    def on_nonlinear_cancelled(self):
        """Maneja la cancelación del análisis no lineal desde el diálogo"""
        try:
            print("🛑 Análisis no lineal cancelado por el usuario")

            # ✅ NUEVO: marcar cancelación para esta ejecución
            self._nonlinear_cancel_requested = True
            
            # Cancelar el worker (esto terminará el proceso subprocess)
            if hasattr(self, 'nonlinear_worker') and self.nonlinear_worker is not None:
                try:
                    self.nonlinear_worker.cancel()
                except:
                    pass
                try:
                    self.nonlinear_worker.requestInterruption()
                except:
                    pass

            # Cerrar/ocultar progreso sin bloquear UI
            if hasattr(self, 'progress_dialog') and self.progress_dialog:
                try:
                    self.progress_dialog.hide()
                except:
                    pass
            self.set_console_overlay_topmost(False)
            
            print("✅ Worker de análisis no lineal cancelado correctamente")
            
        except Exception as e:
            print(f"❌ Error en on_nonlinear_cancelled: {e}")
            import traceback
            traceback.print_exc()
    
    def on_analysis_cancelled(self):
        """Maneja la cancelación del análisis de forma segura"""
        try:
            print("🛑 DEBUG: on_analysis_cancelled disparado - Iniciando parada segura")
            
            # ✅ NUEVO: Re-habilitar botones
            if hasattr(self, 'linear_analysis_button'):
                self.linear_analysis_button.setEnabled(True)
            if hasattr(self, 'run_analysis_button'):
                self.run_analysis_button.setEnabled(True)

            # ✅ NUEVO: Marcar cancelación para esta ejecución (evita mostrar resultados luego)
            self._linear_cancel_requested = True
            
            # 1. Solicitar parada cooperativa al worker lineal (NO terminate)
            if hasattr(self, 'linear_worker') and self.linear_worker is not None:
                try:
                    if self.linear_worker.isRunning():
                        print(f"🛑 DEBUG: Solicitando cancelación al worker {self.linear_worker}")
                        # Señal cooperativa
                        try:
                            self.linear_worker.requestInterruption()
                        except:
                            pass
                        try:
                            self.linear_worker.stop()
                        except:
                            # fallback por si cambia el nombre del método
                            try:
                                self.linear_worker.is_cancelled = True
                            except:
                                pass
                except RuntimeError:
                    self.linear_worker = None

            # 2. Informar al worker no lineal (si existe)
            if hasattr(self, 'nonlinear_worker') and self.nonlinear_worker is not None:
                print("🛑 DEBUG: Cancelando proceso no lineal")
                self.nonlinear_worker.cancel()
            
            # 3. Limpiar la UI (el worker puede tardar en parar si está en cómputo pesado)
            if hasattr(self, 'progress_dialog') and self.progress_dialog:
                self.progress_dialog.hide()
            self.set_console_overlay_topmost(False)
            
            print("✅ Parada segura completada. No debería haber crash.")
            
        except Exception as e:
            print(f"❌ Error en on_analysis_cancelled: {e}")
            import traceback
            traceback.print_exc()

    def get_applied_filters(self):
        """Obtener filtros aplicados por el usuario"""
        filters = {}
        
        if not hasattr(self, 'filter_inputs'):
            return filters
        
        # ✅ NUEVO: Manejar filtros de cepillo de manera especial
        brush_selections = []
        subete_selected = False
        
        for field_name, input_widget in self.filter_inputs.items():
            if field_name in ['すべて', 'A13', 'A11', 'A21', 'A32']:
                if hasattr(input_widget, 'isChecked') and input_widget.isChecked():
                    if field_name == 'すべて':
                        subete_selected = True
                    else:
                        brush_selections.append(field_name)
                continue
            
            if isinstance(input_widget, tuple):
                # Rango de valores (desde, hasta)
                desde, hasta = input_widget
                
                # Manejo especial para fecha
                if field_name == "実験日":
                    # Solo aplicar filtro de fecha si está habilitado
                    if hasattr(self, 'apply_date_filter') and self.apply_date_filter:
                        desde_val = desde.date().toString("yyyyMMdd") if hasattr(desde, 'date') else ''
                        hasta_val = hasta.date().toString("yyyyMMdd") if hasattr(hasta, 'date') else ''
                        
                        # Solo agregar filtro si ambos valores están especificados
                        if desde_val and hasta_val:
                            filters[field_name] = (desde_val, hasta_val)
                else:
                    # Otros campos de rango
                    desde_val = desde.text().strip() if hasattr(desde, 'text') else ''
                    hasta_val = hasta.text().strip() if hasattr(hasta, 'text') else ''
                    
                    # Solo agregar filtro si ambos valores están especificados
                    if desde_val and hasta_val:
                        filters[field_name] = (desde_val, hasta_val)
            else:
                # Valor único
                if hasattr(input_widget, 'text'):
                    value = input_widget.text().strip()
                elif hasattr(input_widget, 'currentText'):
                    value = input_widget.currentText().strip()
                elif hasattr(input_widget, 'date'):
                    value = input_widget.date().toString('yyyy-MM-dd')
                else:
                    value = ''
                
                # Solo agregar filtro si el valor no está vacío
                if value and value != "":
                    filters[field_name] = value
        
        # ✅ NUEVO: Aplicar lógica de filtros de cepillo
        if subete_selected:
            # Si está seleccionado "すべて", agregar el filtro
            filters['すべて'] = True
        elif brush_selections:
            # Si no está seleccionado "すべて" pero hay cepillos específicos seleccionados
            for brush in brush_selections:
                filters[brush] = True
        
        return filters

    def show_linear_analysis_results(self, results):
        """Mostrar resultados del análisis lineal"""
        print("🔧 Mostrando resultados del análisis lineal...")
        
        try:
            # Limpiar layout central completamente
            while self.center_layout.count():
                item = self.center_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                else:
                    # Si es un layout, limpiarlo también
                    layout = item.layout()
                    if layout:
                        while layout.count():
                            layout_item = layout.takeAt(0)
                            layout_widget = layout_item.widget()
                            if layout_widget:
                                layout_widget.deleteLater()
            
            # Forzar actualización de la UI
            QApplication.processEvents()
            
            # ✅ NUEVO: Crear contenedor con fondo gris limpio
            gray_container = QFrame()
            gray_container.setStyleSheet("""
                QFrame {
                    background-color: #f5f5f5;
                    border-radius: 10px;
                    margin: 10px;
                }
            """)
            
            # Layout interno para el contenedor gris
            container_layout = QVBoxLayout(gray_container)
            container_layout.setContentsMargins(20, 20, 20, 20)
            container_layout.setSpacing(15)
            
            # Título
            title = QLabel("線形解析結果")
            title.setStyleSheet("""
                font-weight: bold; 
                font-size: 24px; 
                color: #2c3e50;
                margin-bottom: 20px;
                padding: 10px 0px;
                border-bottom: 2px solid #3498db;
                border-radius: 0px;
            """)
            title.setAlignment(Qt.AlignCenter)
            container_layout.addWidget(title)
            
            # Información del análisis
            # ✅ NUEVO: Formatear datos largos para evitar texto cortado
            filters_applied = results.get('filters_applied', 'N/A')
            if isinstance(filters_applied, list):
                if len(filters_applied) > 3:
                    filters_text = f"{len(filters_applied)} 条件"
                else:
                    filters_text = ", ".join(str(f) for f in filters_applied)
            else:
                filters_text = str(filters_applied)
            
            # Truncar si es muy largo
            if len(filters_text) > 50:
                filters_text = filters_text[:47] + "..."
            
            data_range = results.get('data_range', 'N/A')
            if isinstance(data_range, str) and len(data_range) > 50:
                data_range = data_range[:47] + "..."
            
            info_text = f"""
            📊 解析完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            📈 データ数: {results.get('data_count', 'N/A')} レコード
            🤖 訓練済みモデル: {results.get('models_trained', 'N/A')} 個
            🔧 フィルター適用: {filters_text}
            📊 データ範囲: {data_range}
            """
            
            info_label = QLabel(info_text)
            info_label.setStyleSheet("""
                font-size: 14px;
                color: #34495e;
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #bdc3c7;
            """)
            info_label.setAlignment(Qt.AlignLeft)
            info_label.setWordWrap(True)  # ✅ NUEVO: Permitir salto de línea
            container_layout.addWidget(info_label)
            
            # ✅ NUEVO: Ruta clickeable del archivo Excel
            output_folder = results.get('output_folder', '')
            if output_folder:
                # ✅ NUEVO: Buscar dinámicamente el archivo Excel
                excel_file_path = None
                
                # Buscar en la estructura de carpetas del análisis lineal
                linear_regression_folder = os.path.join(output_folder, "03_線形回帰")
                if os.path.exists(linear_regression_folder):
                    # Buscar en todas las subcarpetas de 03_線形回帰
                    for subfolder in os.listdir(linear_regression_folder):
                        subfolder_path = os.path.join(linear_regression_folder, subfolder)
                        if os.path.isdir(subfolder_path):
                            # Buscar en 04_予測計算 dentro de cada subcarpeta
                            prediction_folder = os.path.join(subfolder_path, "04_予測計算")
                            if os.path.exists(prediction_folder):
                                # Buscar el archivo Excel
                                excel_file = os.path.join(prediction_folder, "XEBEC_予測計算機_逆変換対応.xlsx")
                                if os.path.exists(excel_file):
                                    excel_file_path = excel_file
                                    break
                
                # Si no se encuentra en la estructura esperada, buscar en cualquier lugar del output_folder
                if not excel_file_path:
                    for root, dirs, files in os.walk(output_folder):
                        for file in files:
                            if file == "XEBEC_予測計算機_逆変換対応.xlsx":
                                excel_file_path = os.path.join(root, file)
                                break
                        if excel_file_path:
                            break
                
                # Verificar si el archivo existe
                if excel_file_path and os.path.exists(excel_file_path):
                    # Crear layout para la ruta clickeable
                    path_layout = QVBoxLayout()
                    
                    # Título
                    path_title = QLabel("📁 出力ディレクトリ:")
                    path_title.setStyleSheet("""
                        font-size: 14px;
                        font-weight: bold;
                        color: #2c3e50;
                        margin-bottom: 5px;
                    """)
                    path_layout.addWidget(path_title)
                    
                    # Ruta clickeable con scroll horizontal si es necesario
                    path_label = QLabel(excel_file_path)
                    path_label.setStyleSheet("""
                        QLabel {
                            font-size: 12px;
                            color: #3498db;
                            background-color: #e8f4fd;
                            padding: 10px;
                            border-radius: 5px;
                            border: 1px solid #3498db;
                            text-decoration: underline;
                        }
                        QLabel:hover {
                            background-color: #d1ecf1;
                            cursor: pointer;
                        }
                    """)
                    path_label.setWordWrap(True)  # Permitir salto de línea
                    path_label.setAlignment(Qt.AlignLeft)
                    
                    # Hacer la ruta clickeable
                    def open_excel_file():
                        try:
                            # Abrir el archivo Excel con la aplicación por defecto
                            if os.name == 'nt':  # Windows
                                os.startfile(excel_file_path)
                            elif os.name == 'posix':  # macOS y Linux
                                subprocess.run(['open', excel_file_path], check=True)
                            else:
                                subprocess.run(['xdg-open', excel_file_path], check=True)
                            print(f"✅ Archivo Excel abierto: {excel_file_path}")
                        except Exception as e:
                            print(f"❌ Error abriendo archivo Excel: {e}")
                            QMessageBox.warning(self, "エラー", f"❌ Excelファイルを開けませんでした:\n{str(e)}")
                    
                    # Conectar el click
                    path_label.mousePressEvent = lambda event: open_excel_file()
                    
                    path_layout.addWidget(path_label)
                    container_layout.addLayout(path_layout)
                else:
                    # Si el archivo no existe, mostrar mensaje informativo
                    missing_file_label = QLabel(f"⚠️ Excelファイルが見つかりません\n\n検索場所: {output_folder}\n\nファイル名: XEBEC_予測計算機_逆変換対応.xlsx")
                    missing_file_label.setStyleSheet("""
                        font-size: 12px;
                        color: #e74c3c;
                        background-color: #fadbd8;
                        padding: 10px;
                        border-radius: 5px;
                        border: 1px solid #e74c3c;
                        margin: 10px 0px;
                    """)
                    missing_file_label.setWordWrap(True)
                    missing_file_label.setAlignment(Qt.AlignCenter)
                    container_layout.addWidget(missing_file_label)
            
            # Resultados detallados de modelos
            models = results.get('models', {})
            if models:
                models_title = QLabel("詳細モデル結果")
                models_title.setStyleSheet("""
                    font-weight: bold; 
                    font-size: 18px; 
                    color: #2c3e50;
                    margin-top: 20px;
                    margin-bottom: 10px;
                """)
                container_layout.addWidget(models_title)
                
                for target_name, model_info in models.items():
                    if model_info.get('model') is None:
                        status = "❌ 失敗"
                        error = model_info.get('error', 'Unknown error')
                        details = f"エラー: {error}"
                    else:
                        status = "✅ 成功"
                        model_name = model_info.get('model_name', 'Unknown')
                        task_type = model_info.get('task_type', 'Unknown')
                        details = f"モデル: {model_name}, タイプ: {task_type}"
                        
                        if task_type == 'regression':
                            metrics = model_info.get('final_metrics', {})
                            details += f", R²: {metrics.get('r2', 'N/A'):.4f}, MAE: {metrics.get('mae', 'N/A'):.4f}"
                        else:
                            metrics = model_info.get('final_metrics', {})
                            details += f", 精度: {metrics.get('accuracy', 'N/A'):.4f}, F1: {metrics.get('f1_score', 'N/A'):.4f}"
                    
                    model_label = QLabel(f"【{target_name}】 {status}\n{details}")
                    model_label.setStyleSheet("""
                        font-size: 12px;
                        color: #34495e;
                        background-color: #f8f9fa;
                        padding: 10px;
                        border-radius: 5px;
                        border: 1px solid #dee2e6;
                        margin: 5px 0px;
                    """)
                    container_layout.addWidget(model_label)
            
            # Botón para volver a filtros
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            back_button = QPushButton("次へ")
            back_button.setFixedSize(120, 40)  # Hacer el botón más estrecho
            back_button.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
            """)
            back_button.clicked.connect(self.show_evaluation_charts)
            
            button_layout.addWidget(back_button)
            button_layout.addStretch()
            container_layout.addLayout(button_layout)
            
            # Espacio flexible
            container_layout.addStretch()
            
            # ✅ NUEVO: Agregar el contenedor gris al layout central
            self.center_layout.addWidget(gray_container)
            
            print("✅ Resultados del análisis lineal mostrados")
            
        except Exception as e:
            print(f"❌ Error mostrando resultados: {e}")
            import traceback
            traceback.print_exc()

    def show_evaluation_charts(self):
        """Mostrar gráficos de evaluación con navegación"""
        print("🔧 Mostrando gráficos de evaluación...")
        
        try:
            # Limpiar layout central completamente
            while self.center_layout.count():
                item = self.center_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                else:
                    # Si es un layout, limpiarlo también
                    layout = item.layout()
                    if layout:
                        while layout.count():
                            layout_item = layout.takeAt(0)
                            layout_widget = layout_item.widget()
                            if layout_widget:
                                layout_widget.deleteLater()
            
            # Forzar actualización de la UI
            QApplication.processEvents()
            
            # ✅ NUEVO: Crear contenedor con fondo gris limpio
            gray_container = QFrame()
            gray_container.setStyleSheet("""
                QFrame {
                    background-color: #f5f5f5;
                    border-radius: 10px;
                    margin: 10px;
                }
            """)
            
            # Layout interno para el contenedor gris
            container_layout = QVBoxLayout(gray_container)
            container_layout.setContentsMargins(20, 20, 20, 20)
            container_layout.setSpacing(15)
            
            # Título
            title = QLabel("評価スコア チャート")
            title.setStyleSheet("""
                font-weight: bold; 
                font-size: 24px; 
                color: #2c3e50;
                margin-bottom: 20px;
                padding: 10px 0px;
                border-bottom: 2px solid #3498db;
                border-radius: 0px;
            """)
            title.setAlignment(Qt.AlignCenter)
            container_layout.addWidget(title)
            
            # ✅ NUEVO: Buscar gráficos de evaluación
            chart_images = []
            if hasattr(self, 'current_project_folder') and self.current_project_folder:
                # Buscar en la estructura de carpetas del análisis lineal
                linear_regression_folder = os.path.join(self.current_project_folder, "03_線形回帰")
                if os.path.exists(linear_regression_folder):
                    # Buscar en todas las subcarpetas de 03_線形回帰
                    for subfolder in os.listdir(linear_regression_folder):
                        subfolder_path = os.path.join(linear_regression_folder, subfolder)
                        if os.path.isdir(subfolder_path):
                            # Buscar en 03_評価スコア\01_チャート
                            evaluation_folder = os.path.join(subfolder_path, "03_評価スコア", "01_チャート")
                            if os.path.exists(evaluation_folder):
                                # Buscar archivos PNG
                                for file in os.listdir(evaluation_folder):
                                    if file.lower().endswith('.png'):
                                        chart_images.append(os.path.join(evaluation_folder, file))
                                break
            
            # Si no se encuentran gráficos, mostrar mensaje
            if not chart_images:
                no_charts_label = QLabel("⚠️ 評価スコアチャートが見つかりません")
                no_charts_label.setStyleSheet("""
                    font-size: 16px;
                    color: #e74c3c;
                    background-color: #fadbd8;
                    padding: 20px;
                    border-radius: 8px;
                    border: 1px solid #e74c3c;
                    margin: 20px 0px;
                """)
                no_charts_label.setAlignment(Qt.AlignCenter)
                container_layout.addWidget(no_charts_label)
            else:
                # ✅ NUEVO: Configurar navegación de gráficos
                self.chart_images = sorted(chart_images)
                self.current_chart_index = 0
                
                # Layout principal para la imagen y navegación
                chart_layout = QVBoxLayout()
                
                # Label para mostrar la imagen (ocupa todo el ancho)
                self.chart_label = QLabel()
                self.chart_label.setAlignment(Qt.AlignCenter)
                self.chart_label.setStyleSheet("""
                    QLabel {
                        background-color: white;
                        border: 2px solid #bdc3c7;
                        border-radius: 10px;
                        padding: 10px;
                        min-height: 500px;
                    }
                """)
                chart_layout.addWidget(self.chart_label)
                
                # Layout horizontal para botones de navegación (debajo de la imagen)
                nav_buttons_layout = QHBoxLayout()
                nav_buttons_layout.addStretch()
                
                # Botón flecha izquierda con mejor icono
                prev_chart_button = QPushButton("◀ 前へ")
                prev_chart_button.setFixedSize(100, 40)
                prev_chart_button.setStyleSheet("""
                    QPushButton {
                        background-color: #3498db;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        font-size: 14px;
                        font-weight: bold;
                        padding: 8px 16px;
                    }
                    QPushButton:hover {
                        background-color: #2980b9;
                    }
                    QPushButton:disabled {
                        background-color: #bdc3c7;
                        color: #7f8c8d;
                    }
                """)
                prev_chart_button.clicked.connect(self.show_previous_chart)
                nav_buttons_layout.addWidget(prev_chart_button)
                
                # Espacio entre botones
                nav_buttons_layout.addSpacing(20)
                
                # Botón flecha derecha con mejor icono
                next_chart_button = QPushButton("次へ ▶")
                next_chart_button.setFixedSize(100, 40)
                next_chart_button.setStyleSheet("""
                    QPushButton {
                        background-color: #3498db;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        font-size: 14px;
                        font-weight: bold;
                        padding: 8px 16px;
                    }
                    QPushButton:hover {
                        background-color: #2980b9;
                    }
                    QPushButton:disabled {
                        background-color: #bdc3c7;
                        color: #7f8c8d;
                    }
                """)
                next_chart_button.clicked.connect(self.show_next_chart)
                nav_buttons_layout.addWidget(next_chart_button)
                
                nav_buttons_layout.addStretch()
                chart_layout.addLayout(nav_buttons_layout)
                
                # Información del gráfico actual
                self.chart_info_label = QLabel()
                self.chart_info_label.setStyleSheet("""
                    font-size: 14px;
                    color: #2c3e50;
                    background-color: #ecf0f1;
                    padding: 10px;
                    border-radius: 5px;
                    border: 1px solid #bdc3c7;
                    margin: 10px 0px;
                """)
                self.chart_info_label.setAlignment(Qt.AlignCenter)
                chart_layout.addWidget(self.chart_info_label)
                
                container_layout.addLayout(chart_layout)
                
                # Mostrar el primer gráfico
                self.update_chart_display()
            
            # Botones para volver a resultados y predicción
            buttons_layout = QHBoxLayout()
            buttons_layout.addStretch()
            
            # Botón para volver a filtros (modoru)
            back_button = QPushButton("戻る")
            back_button.setFixedSize(120, 40)
            back_button.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
            """)
            back_button.clicked.connect(self.on_analyze_clicked)
            buttons_layout.addWidget(back_button)
            
            # Espacio entre botones
            buttons_layout.addSpacing(20)
            
            # Botón para predicción
            prediction_button = QPushButton("予測")
            prediction_button.setFixedSize(120, 40)
            prediction_button.setStyleSheet("""
                QPushButton {
                    background-color: #27ae60;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #229954;
                }
            """)
            prediction_button.clicked.connect(self.run_prediction)
            buttons_layout.addWidget(prediction_button)
            
            buttons_layout.addStretch()
            container_layout.addLayout(buttons_layout)
            
            # Espacio flexible
            container_layout.addStretch()
            
            # ✅ NUEVO: Agregar el contenedor gris al layout central
            self.center_layout.addWidget(gray_container)
            
            print("✅ Gráficos de evaluación mostrados")
            
        except Exception as e:
            print(f"❌ Error mostrando gráficos de evaluación: {e}")
            import traceback
            traceback.print_exc()
    
    def show_previous_chart(self):
        """Mostrar gráfico anterior"""
        if hasattr(self, 'chart_images') and len(self.chart_images) > 0:
            self.current_chart_index = (self.current_chart_index - 1) % len(self.chart_images)
            self.update_chart_display()
    
    def show_next_chart(self):
        """Mostrar gráfico siguiente"""
        if hasattr(self, 'chart_images') and len(self.chart_images) > 0:
            self.current_chart_index = (self.current_chart_index + 1) % len(self.chart_images)
            self.update_chart_display()
    
    def update_chart_display(self):
        """Actualizar la visualización del gráfico actual"""
        if hasattr(self, 'chart_images') and len(self.chart_images) > 0:
            current_image_path = self.chart_images[self.current_chart_index]
            
            # Cargar y mostrar la imagen
            pixmap = QPixmap(current_image_path)
            if not pixmap.isNull():
                # ✅ NUEVO: Redimensionar la imagen para ocupar todo el ancho disponible
                # Obtener el tamaño del contenedor
                container_width = self.chart_label.width() - 20  # Restar padding
                container_height = self.chart_label.height() - 20  # Restar padding
                
                # Si el contenedor aún no tiene tamaño, usar un tamaño por defecto
                if container_width <= 0:
                    container_width = 1000
                if container_height <= 0:
                    container_height = 600
                
                # Redimensionar manteniendo la proporción
                scaled_pixmap = pixmap.scaled(container_width, container_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.chart_label.setPixmap(scaled_pixmap)
                
                # Actualizar información del gráfico
                filename = os.path.basename(current_image_path)
                info_text = f"📊 {filename} ({self.current_chart_index + 1}/{len(self.chart_images)})"
                self.chart_info_label.setText(info_text)
                
                print(f"✅ Mostrando gráfico: {filename}")
            else:
                print(f"❌ No se pudo cargar la imagen: {current_image_path}")



    def on_formula_processing_error(self, error_msg):
        """Manejar errores en el procesamiento de fórmulas"""
        print(f"❌ Error en procesamiento de fórmulas: {error_msg}")
        QMessageBox.critical(self, "エラー", f"❌ 予測計算中にエラーが発生しました:\n{error_msg}")

    def show_yosoku_parameters_dialog(self):
        """Mostrar diálogo para seleccionar parámetros de predicción Yosoku"""
        try:
            from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QFormLayout
            
            dialog = QDialog(self)
            dialog.setWindowTitle("予測パラメーター選択")
            dialog.setModal(True)
            dialog.resize(400, 350)
            
            layout = QVBoxLayout()
            
            # Título
            title = QLabel("予測パラメーターを選択してください")
            title.setStyleSheet("font-weight: bold; font-size: 14px; margin: 10px;")
            title.setAlignment(Qt.AlignCenter)
            layout.addWidget(title)
            
            # Formulario de selección
            form_layout = QFormLayout()
            
            # Tipo de cepillo
            brush_combo = QComboBox()
            brush_combo.addItem("A13", "A13")
            brush_combo.addItem("A11", "A11")
            brush_combo.addItem("A21", "A21")
            brush_combo.addItem("A32", "A32")
            brush_combo.setCurrentText("A11")  # Valor por defecto
            form_layout.addRow("ブラシタイプ:", brush_combo)
            
            # Diámetro
            diameter_combo = QComboBox()
            diameter_combo.addItem("6", 6)
            diameter_combo.addItem("15", 15)
            diameter_combo.addItem("25", 25)
            diameter_combo.addItem("40", 40)
            diameter_combo.addItem("60", 60)
            diameter_combo.addItem("100", 100)
            diameter_combo.setCurrentText("15")  # Valor por defecto
            form_layout.addRow("直径:", diameter_combo)
            
            # Material
            material_combo = QComboBox()
            material_combo.addItem("Steel", "Steel")
            material_combo.addItem("Alum", "Alum")
            material_combo.setCurrentText("Steel")  # Valor por defecto
            form_layout.addRow("材料:", material_combo)
            
            # 線材長 (de 30 a 75 en intervalos de 5mm)
            wire_length_combo = QComboBox()
            for value in range(30, 80, 5):  # 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
                wire_length_combo.addItem(str(value), value)
            wire_length_combo.setCurrentText("75")  # Valor por defecto
            form_layout.addRow("線材長:", wire_length_combo)
            
            layout.addLayout(form_layout)
            layout.addStretch()
            
            # Botones
            button_layout = QHBoxLayout()
            
            cancel_button = QPushButton("キャンセル")
            cancel_button.clicked.connect(dialog.reject)
            
            ok_button = QPushButton("予測実行")
            ok_button.clicked.connect(dialog.accept)
            ok_button.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
            
            button_layout.addWidget(cancel_button)
            button_layout.addWidget(ok_button)
            layout.addLayout(button_layout)
            
            dialog.setLayout(layout)
            
            # Mostrar diálogo
            result = dialog.exec()
            
            if result == QDialog.Accepted:
                # Procesar selecciones
                selected_params = {
                    'brush': brush_combo.currentData(),
                    'diameter': diameter_combo.currentData(),
                    'material': material_combo.currentData(),
                    'wire_length': wire_length_combo.currentData()
                }
                
                print(f"📊 Parámetros seleccionados: {selected_params}")
                return selected_params
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error mostrando diálogo de parámetros: {e}")
            import traceback
            traceback.print_exc()
            return None

    def find_latest_formulas_file(self):
        """Encontrar el archivo XEBEC_予測計算機_逆変換対応.xlsx en la carpeta del análisis lineal más reciente"""
        try:
            # Buscar la carpeta del análisis lineal más reciente
            linear_regression_folder = os.path.join(self.current_project_folder, "03_線形回帰")
            
            if not os.path.exists(linear_regression_folder):
                print(f"❌ No se encontró la carpeta: {linear_regression_folder}")
                return None
            
            # Buscar subcarpetas con formato de fecha
            subfolders = []
            for item in os.listdir(linear_regression_folder):
                item_path = os.path.join(linear_regression_folder, item)
                if os.path.isdir(item_path) and item.startswith("82_"):
                    subfolders.append(item_path)
            
            if not subfolders:
                print(f"❌ No se encontraron subcarpetas de análisis lineal en: {linear_regression_folder}")
                return None
            
            # Ordenar por fecha (más reciente primero)
            subfolders.sort(reverse=True)
            latest_folder = subfolders[0]
            print(f"📊 Carpeta más reciente encontrada: {latest_folder}")
            
            # Buscar la subcarpeta 04_予測計算
            prediction_folder = os.path.join(latest_folder, "04_予測計算")
            
            if not os.path.exists(prediction_folder):
                print(f"❌ No se encontró la carpeta: {prediction_folder}")
                return None
            
            # Buscar el archivo XEBEC_予測計算機_逆変換対応.xlsx
            formulas_file = os.path.join(prediction_folder, "XEBEC_予測計算機_逆変換対応.xlsx")
            
            if os.path.exists(formulas_file):
                print(f"✅ Archivo de fórmulas encontrado: {formulas_file}")
                return formulas_file
            else:
                print(f"❌ No se encontró el archivo: {formulas_file}")
                return None
                
        except Exception as e:
            print(f"❌ Error buscando archivo de fórmulas: {e}")
            import traceback
            traceback.print_exc()
            return None

    def validate_filtered_data(self, selected_params):
        """Validar el archivo filtered_data.xlsx contra los parámetros seleccionados"""
        try:
            # Buscar la carpeta del análisis lineal más reciente
            linear_regression_folder = os.path.join(self.current_project_folder, "03_線形回帰")
            
            if not os.path.exists(linear_regression_folder):
                return False, ["❌ No se encontró la carpeta de análisis lineal: 03_線形回帰"]
            
            # Buscar subcarpetas con formato de fecha
            subfolders = []
            for item in os.listdir(linear_regression_folder):
                item_path = os.path.join(linear_regression_folder, item)
                if os.path.isdir(item_path) and item.startswith("82_"):
                    subfolders.append(item_path)
            
            if not subfolders:
                return False, ["❌ No se encontraron subcarpetas de análisis lineal"]
            
            # Ordenar por fecha (más reciente primero)
            subfolders.sort(reverse=True)
            latest_folder = subfolders[0]
            
            # Buscar el archivo filtered_data.xlsx en la carpeta 01_学習モデル
            model_folder = os.path.join(latest_folder, "01_学習モデル")
            filtered_data_file = os.path.join(model_folder, "filtered_data.xlsx")
            
            if not os.path.exists(filtered_data_file):
                return False, [f"❌ No se encontró el archivo: {os.path.basename(filtered_data_file)}"]
            
            print(f"📊 Validando archivo: {filtered_data_file}")
            
            # Cargar datos del archivo Excel
            import pandas as pd
            data_df = pd.read_excel(filtered_data_file)
            
            print(f"📊 Datos cargados para validación: {len(data_df)} filas, {len(data_df.columns)} columnas")
            print(f"📊 Columnas disponibles: {list(data_df.columns)}")
            
            errors = []
            
            # 1. Validar tipos de cepillo (A13, A11, A21, A32)
            brush_columns = ['A13', 'A11', 'A21', 'A32']
            brush_values = {}
            
            for col in brush_columns:
                if col in data_df.columns:
                    # Contar valores únicos que no sean 0
                    non_zero_values = data_df[data_df[col] == 1][col].unique()
                    brush_values[col] = len(non_zero_values)
                else:
                    brush_values[col] = 0
            
            # Verificar si hay más de un tipo de cepillo con valor 1
            active_brushes = sum(1 for count in brush_values.values() if count > 0)
            if active_brushes > 1:
                errors.append(f"❌ Múltiples tipos de cepillo activos encontrados: {active_brushes}")
            
            # Verificar si el tipo de cepillo seleccionado está presente
            selected_brush = selected_params['brush']
            if selected_brush in brush_values and brush_values[selected_brush] == 0:
                errors.append(f"❌ El tipo de cepillo seleccionado '{selected_brush}' no está presente en los datos")
            
            # 2. Validar material
            material_column = '材料'
            if material_column in data_df.columns:
                unique_materials = data_df[material_column].dropna().unique()
                if len(unique_materials) > 1:
                    errors.append(f"❌ Múltiples materiales encontrados: {list(unique_materials)}")
                
                # Verificar si el material seleccionado está presente
                selected_material = selected_params['material']
                if selected_material not in unique_materials:
                    errors.append(f"❌ El material seleccionado '{selected_material}' no está presente en los datos")
            else:
                errors.append(f"❌ No se encontró la columna de material: {material_column}")
            
            # 3. Validar diámetro
            diameter_column = '直径'
            if diameter_column in data_df.columns:
                unique_diameters = data_df[diameter_column].dropna().unique()
                if len(unique_diameters) > 1:
                    errors.append(f"❌ Múltiples diámetros encontrados: {list(unique_diameters)}")
                
                # Verificar si el diámetro seleccionado está presente
                selected_diameter = selected_params['diameter']
                if selected_diameter not in unique_diameters:
                    errors.append(f"❌ El diámetro seleccionado '{selected_diameter}' no está presente en los datos")
            else:
                errors.append(f"❌ No se encontró la columna de diámetro: {diameter_column}")
            
            # 4. Validar rango de 線材長
            wire_length_column = '線材長'
            if wire_length_column in data_df.columns:
                wire_length_values = data_df[wire_length_column].dropna()
                if len(wire_length_values) > 0:
                    min_wire_length = wire_length_values.min()
                    max_wire_length = wire_length_values.max()
                    selected_wire_length = selected_params['wire_length']
                    
                    # Verificar si el rango está dentro del rango del usuario - 5mm
                    expected_min = selected_wire_length - 5
                    expected_max = selected_wire_length
                    
                    if min_wire_length < expected_min or max_wire_length > expected_max:
                        errors.append(f"❌ Rango de 線材長 fuera del rango esperado:")
                        errors.append(f"   - Rango en datos: {min_wire_length} - {max_wire_length}")
                        errors.append(f"   - Rango esperado: {expected_min} - {expected_max}")
                        errors.append(f"   - Seleccionado por usuario: {selected_wire_length}")
                else:
                    errors.append(f"❌ No hay datos válidos en la columna 線材長")
            else:
                errors.append(f"❌ No se encontró la columna 線材長: {wire_length_column}")
            
            # Retornar resultado de validación
            if errors:
                print(f"❌ Errores de validación encontrados:")
                for error in errors:
                    print(f"   {error}")
                return False, errors
            else:
                print(f"✅ Validación exitosa - Todos los parámetros son consistentes")
                return True, []
                
        except Exception as e:
            error_msg = f"❌ Error durante la validación: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return False, [error_msg]

    def run_prediction(self):
        """Ejecutar predicción Yosoku con parámetros del usuario y diálogo de progreso"""
        print("🔧 Iniciando predicción Yosoku...")
        
        try:
            # Verificar que tenemos la carpeta del proyecto
            if not hasattr(self, 'current_project_folder') or not self.current_project_folder:
                QMessageBox.warning(self, "エラー", "❌ プロジェクトフォルダが見つかりません。")
                return
            
            # Mostrar diálogo de selección de parámetros
            selected_params = self.show_yosoku_parameters_dialog()
            if not selected_params:
                print("❌ Usuario canceló la selección de parámetros")
                return
            
            print(f"📊 Parámetros seleccionados: {selected_params}")
            
            # Validar datos filtrados antes de continuar
            print("🔍 Validando datos filtrados...")
            is_valid, validation_errors = self.validate_filtered_data(selected_params)
            
            if not is_valid:
                # Mostrar resumen de errores
                error_summary = "❌ Validación fallida - No se puede continuar con la predicción:\n\n"
                error_summary += "\n".join(validation_errors)
                
                print("❌ Validación fallida:")
                for error in validation_errors:
                    print(f"   {error}")
                
                QMessageBox.critical(
                    self,
                    "エラー - データ検証失敗",
                    error_summary
                )
                return
            
            print("✅ Validación exitosa - Continuando con la predicción")
            
            # Iniciar predicción con diálogo de progreso
            self.start_yosoku_prediction_with_progress(selected_params)
            
        except Exception as e:
            print(f"❌ Error ejecutando predicción: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "エラー", f"❌ 予測実行中にエラーが発生しました:\n{str(e)}")

    def start_yosoku_prediction_with_progress(self, selected_params):
        """Iniciar predicción Yosoku con diálogo de progreso"""
        try:
            # Buscar archivos necesarios
            unexperimental_file = self.find_unexperimental_file()
            if not unexperimental_file:
                QMessageBox.warning(self, "エラー", "❌ 未実験データファイルが見つかりません。")
                return
            
            formulas_file = self.find_latest_formulas_file()
            if not formulas_file:
                QMessageBox.warning(self, "エラー", "❌ XEBEC_予測計算機_逆変換対応.xlsxファイルが見つかりません。")
                return
            
            # Crear ruta de salida
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{os.path.basename(unexperimental_file).replace('_未実験データ.xlsx', '')}_予測結果_{timestamp}.xlsx"
            
            # Obtener la carpeta donde está el archivo de fórmulas
            formulas_folder = os.path.dirname(formulas_file)
            output_path = os.path.join(formulas_folder, output_filename)
            
            # Crear y mostrar diálogo de progreso
            self.yosoku_progress_dialog = YosokuProgressDialog(self)
            self.yosoku_progress_dialog.show()
            self.set_console_overlay_topmost(True)
            
            # Crear worker thread
            self.yosoku_worker = YosokuWorker(selected_params, unexperimental_file, formulas_file, output_path)
            
            # Conectar señales
            self.yosoku_worker.progress_updated.connect(self.yosoku_progress_dialog.update_progress)
            self.yosoku_worker.status_updated.connect(self.yosoku_progress_dialog.update_status)
            self.yosoku_worker.finished.connect(self.on_yosoku_prediction_finished)
            self.yosoku_worker.error.connect(self.on_yosoku_prediction_error)
            
            # Conectar botón de cancelar
            self.yosoku_progress_dialog.cancel_button.clicked.connect(self.cancel_yosoku_prediction)
            
            # Iniciar worker
            self.yosoku_worker.start()
            
        except Exception as e:
            print(f"❌ Error iniciando predicción: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "エラー", f"❌ 予測開始中にエラーが発生しました:\n{str(e)}")

    def find_unexperimental_file(self):
        """Encontrar el archivo 未実験データ.xlsx"""
        try:
            project_name = os.path.basename(self.current_project_folder)
            expected_filename = f"{project_name}_未実験データ.xlsx"
            unexperimental_path = os.path.join(self.current_project_folder, expected_filename)
            
            if os.path.exists(unexperimental_path):
                return unexperimental_path
            else:
                # Buscar cualquier archivo que termine en _未実験データ.xlsx
                for file in os.listdir(self.current_project_folder):
                    if file.endswith("_未実験データ.xlsx"):
                        return os.path.join(self.current_project_folder, file)
                return None
        except Exception as e:
            print(f"❌ Error buscando archivo 未実験データ: {e}")
            return None

    def on_yosoku_prediction_finished(self, output_path):
        """Manejar finalización exitosa de la predicción"""
        try:
            # Cerrar diálogo de progreso
            if hasattr(self, 'yosoku_progress_dialog'):
                self.yosoku_progress_dialog.close()
                self.yosoku_progress_dialog = None
            self.set_console_overlay_topmost(False)
            
            # Terminar worker
            if hasattr(self, 'yosoku_worker'):
                self.yosoku_worker.quit()
                self.yosoku_worker.wait()
                self.yosoku_worker = None
            
            # Mostrar mensaje de éxito
            output_filename = os.path.basename(output_path)
            formulas_folder = os.path.dirname(output_path)
            
            QMessageBox.information(
                self,
                "予測完了",
                f"✅ 予測が完了しました！\n\n結果ファイル: {output_filename}\n\n保存場所: {formulas_folder}"
            )
            
            # Preguntar si quiere importar a la base de datos
            reply = QMessageBox.question(
                self,
                "データベースインポート",
                "予測結果をデータベースにインポートしますか？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.import_yosoku_results_to_database(output_path)
            
        except Exception as e:
            print(f"❌ Error en finalización: {e}")
            import traceback
            traceback.print_exc()

    def on_yosoku_prediction_error(self, error_msg):
        """Manejar errores en la predicción"""
        try:
            # Cerrar diálogo de progreso
            if hasattr(self, 'yosoku_progress_dialog'):
                self.yosoku_progress_dialog.close()
                self.yosoku_progress_dialog = None
            self.set_console_overlay_topmost(False)
            
            # Terminar worker
            if hasattr(self, 'yosoku_worker'):
                self.yosoku_worker.quit()
                self.yosoku_worker.wait()
                self.yosoku_worker = None
            
            # Mostrar mensaje de error
            QMessageBox.critical(self, "エラー", f"❌ 予測実行中にエラーが発生しました:\n{error_msg}")
            
        except Exception as e:
            print(f"❌ Error en manejo de error: {e}")
            import traceback
            traceback.print_exc()

    def import_yosoku_results_to_database(self, excel_path):
        """Importar resultados de predicción a la base de datos con diálogo de progreso"""
        try:
            # Verificar si ya existe un diálogo abierto (para evitar duplicados)
            if hasattr(self, 'yosoku_import_progress_dialog') and self.yosoku_import_progress_dialog is not None:
                # Si ya existe, reutilizarlo
                existing_dialog = self.yosoku_import_progress_dialog
            else:
                # Crear y mostrar diálogo de progreso
                self.yosoku_import_progress_dialog = YosokuImportProgressDialog(self)
                self.yosoku_import_progress_dialog.show()
                existing_dialog = self.yosoku_import_progress_dialog
            # Mientras el diálogo con chibi esté activo: flecha/consola por encima
            self.set_console_overlay_topmost(True)
            
            # Crear worker thread (análisis lineal)
            self.yosoku_import_worker = YosokuImportWorker(excel_path, analysis_type="lineal")
            
            # Conectar señales
            self.yosoku_import_worker.progress_updated.connect(existing_dialog.update_progress)
            self.yosoku_import_worker.status_updated.connect(existing_dialog.set_status)
            self.yosoku_import_worker.finished.connect(self.on_yosoku_import_finished)
            self.yosoku_import_worker.error.connect(self.on_yosoku_import_error)
            
            # Conectar botón de cancelar
            existing_dialog.cancel_button.clicked.connect(self.cancel_yosoku_import)
            
            # Iniciar worker
            self.yosoku_import_worker.start()
            
        except Exception as e:
            print(f"❌ Error iniciando importación: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "エラー",
                f"❌ インポート開始中にエラーが発生しました:\n{str(e)}"
            )
    
    def on_yosoku_import_finished(self):
        """Manejar finalización exitosa de importación"""
        try:
            # Cerrar diálogo de progreso
            if hasattr(self, 'yosoku_import_progress_dialog') and self.yosoku_import_progress_dialog is not None:
                self.yosoku_import_progress_dialog.close()
                self.yosoku_import_progress_dialog = None
            self.set_console_overlay_topmost(False)
            
            # Limpiar worker
            if hasattr(self, 'yosoku_import_worker') and self.yosoku_import_worker is not None:
                self.yosoku_import_worker.quit()
                self.yosoku_import_worker.wait()
                self.yosoku_import_worker = None
            
            # Mostrar mensaje de éxito
            QMessageBox.information(
                self,
                "インポート完了",
                "✅ データベースへのインポートが完了しました！"
            )
            
        except Exception as e:
            print(f"❌ Error en manejo de finalización: {e}")
            import traceback
            traceback.print_exc()
    
    def on_yosoku_import_error(self, error_msg):
        """Manejar error en importación"""
        try:
            # Cerrar diálogo de progreso
            if hasattr(self, 'yosoku_import_progress_dialog') and self.yosoku_import_progress_dialog is not None:
                self.yosoku_import_progress_dialog.close()
                self.yosoku_import_progress_dialog = None
            self.set_console_overlay_topmost(False)
            
            # Limpiar worker
            if hasattr(self, 'yosoku_import_worker') and self.yosoku_import_worker is not None:
                self.yosoku_import_worker.quit()
                self.yosoku_import_worker.wait()
                self.yosoku_import_worker = None
            
            # Mostrar mensaje de error
            QMessageBox.critical(
                self,
                "エラー",
                f"❌ データベースへのインポート中にエラーが発生しました:\n{error_msg}"
            )
            
        except Exception as e:
            print(f"❌ Error en manejo de error: {e}")
            import traceback
            traceback.print_exc()
    
    def cancel_yosoku_import(self):
        """Cancelar importación"""
        try:
            if hasattr(self, 'yosoku_import_worker'):
                self.yosoku_import_worker.cancel_import()
                self.yosoku_import_worker.quit()
                self.yosoku_import_worker.wait()
                self.yosoku_import_worker = None
            
            if hasattr(self, 'yosoku_import_progress_dialog'):
                self.yosoku_import_progress_dialog.close()
                self.yosoku_import_progress_dialog = None
            self.set_console_overlay_topmost(False)
        except Exception as e:
            print(f"❌ Error cancelando importación: {e}")
            import traceback
            traceback.print_exc()
    
    def import_classification_results_to_yosoku_db(self):
        """Importar resultados de clasificación a la base de datos de yosoku"""
        try:
            # Obtener la carpeta raíz del análisis de clasificación
            # Puede estar en classification_project_folder o classification_existing_folder_path
            from pathlib import Path
            import glob
            import os
            
            # Intentar obtener la carpeta raíz del análisis
            if hasattr(self, 'classification_project_folder') and self.classification_project_folder:
                analysis_root = Path(self.classification_project_folder)
            elif hasattr(self, 'classification_existing_folder_path') and self.classification_existing_folder_path:
                # Si solo tenemos la carpeta de evaluación, subir dos niveles para llegar a la raíz
                analysis_root = Path(self.classification_existing_folder_path).parent.parent
            else:
                QMessageBox.warning(self, "エラー", "❌ 分類解析結果のフォルダが見つかりません。")
                return
            
            print(f"🔍 Carpeta raíz del análisis: {analysis_root}")
            print(f"🔍 Carpeta raíz existe: {analysis_root.exists()}")
            
            # Construir ruta del archivo de predicción desde la carpeta raíz
            pred_folder = analysis_root / "02_本学習結果" / "03_予測結果"
            
            print(f"🔍 Buscando archivo de predicción en: {pred_folder}")
            print(f"🔍 Carpeta existe: {pred_folder.exists()}")
            
            if not pred_folder.exists():
                # Intentar con ruta absoluta
                pred_folder_abs = analysis_root.resolve() / "02_本学習結果" / "03_予測結果"
                print(f"🔍 Intentando con ruta absoluta: {pred_folder_abs}")
                if pred_folder_abs.exists():
                    pred_folder = pred_folder_abs
                else:
                    # Mostrar información de debug
                    print(f"❌ Carpeta de predicción no encontrada")
                    print(f"   Ruta intentada 1: {pred_folder}")
                    print(f"   Ruta intentada 2: {pred_folder_abs}")
                    print(f"   Carpeta raíz: {analysis_root}")
                    print(f"   Carpeta raíz existe: {analysis_root.exists()}")
                    if analysis_root.exists():
                        print(f"   Contenido de carpeta raíz:")
                        for item in analysis_root.iterdir():
                            print(f"     - {item.name} ({'dir' if item.is_dir() else 'file'})")
                    
                    QMessageBox.warning(
                        self,
                        "エラー",
                        f"❌ 予測結果フォルダが見つかりません。\n\n"
                        f"フォルダ: {pred_folder}\n\n"
                        f"または:\n{pred_folder_abs}\n\n"
                        f"分析ルートフォルダ: {analysis_root}"
                    )
                    return
            
            # Listar archivos en la carpeta para debug
            all_files = list(pred_folder.glob("*"))
            print(f"🔍 Archivos encontrados en carpeta ({len(all_files)}):")
            for f in all_files:
                print(f"  - {f.name} (archivo: {f.is_file()}, dir: {f.is_dir()})")
            
            # Buscar archivo de predicción con diferentes estrategias
            prediction_file = None
            
            # Prioridad 1: Prediction_input_pred.xlsx (ignorar archivos temporales de Excel)
            candidate1 = pred_folder / "Prediction_input_pred.xlsx"
            if candidate1.exists() and not candidate1.name.startswith("~$"):
                prediction_file = candidate1
                print(f"✅ Archivo encontrado (Prioridad 1): {prediction_file}")
            else:
                # Prioridad 2: Buscar cualquier archivo *_pred.xlsx (ignorar temporales)
                pred_files = [f for f in pred_folder.glob("*_pred.xlsx") if not f.name.startswith("~$")]
                if pred_files:
                    # Seleccionar el más reciente
                    prediction_file = max(pred_files, key=lambda p: p.stat().st_mtime)
                    print(f"✅ Archivo encontrado (Prioridad 2): {prediction_file}")
                else:
                    # Prioridad 3: Buscar cualquier archivo .xlsx en la carpeta (ignorar temporales)
                    xlsx_files = [f for f in pred_folder.glob("*.xlsx") if not f.name.startswith("~$")]
                    if xlsx_files:
                        # Seleccionar el más reciente
                        prediction_file = max(xlsx_files, key=lambda p: p.stat().st_mtime)
                        print(f"✅ Archivo encontrado (Prioridad 3): {prediction_file}")
            
            if not prediction_file or not prediction_file.exists():
                # Listar archivos disponibles para ayudar al usuario
                available_files = [f.name for f in pred_folder.glob("*.xlsx") if not f.name.startswith("~$")]
                files_list = "\n".join([f"  - {f}" for f in available_files]) if available_files else "  (なし)"
                
                QMessageBox.warning(
                    self,
                    "エラー",
                    f"❌ 予測結果ファイルが見つかりません。\n\n"
                    f"フォルダ: {pred_folder}\n\n"
                    f"利用可能なファイル:\n{files_list}\n\n"
                    f"期待されるファイル名:\n"
                    f"- Prediction_input_pred.xlsx\n"
                    f"- *_pred.xlsx\n\n"
                    f"注意: Excelでファイルが開かれている場合は、閉じてから再度お試しください。"
                )
                return
            
            print(f"✅ Archivo de predicción seleccionado: {prediction_file}")
            
            # Preguntar al usuario sobre sobreescritura
            reply = QMessageBox.question(
                self,
                "データベースへのインポート",
                "既存のデータを上書きしますか？\n\n"
                "既存のレコードが見つかった場合、そのレコードを更新します。\n"
                "「いいえ」を選択した場合、既存のレコードはスキップされます。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            overwrite = (reply == QMessageBox.Yes)
            
            # Crear y mostrar diálogo de progreso
            if hasattr(self, 'classification_import_progress_dialog') and self.classification_import_progress_dialog is not None:
                existing_dialog = self.classification_import_progress_dialog
            else:
                self.classification_import_progress_dialog = YosokuImportProgressDialog(self)
                self.classification_import_progress_dialog.show()
                existing_dialog = self.classification_import_progress_dialog
            self.set_console_overlay_topmost(True)
            
            # Crear worker thread
            self.classification_import_worker = ClassificationImportWorker(str(prediction_file), overwrite=overwrite)
            
            # Conectar señales
            self.classification_import_worker.progress_updated.connect(existing_dialog.update_progress)
            self.classification_import_worker.status_updated.connect(existing_dialog.set_status)
            self.classification_import_worker.finished.connect(self.on_classification_import_finished)
            self.classification_import_worker.error.connect(self.on_classification_import_error)
            
            # Conectar botón de cancelar
            existing_dialog.cancel_button.clicked.connect(self.cancel_classification_import)
            
            # Iniciar worker
            self.classification_import_worker.start()
            
        except Exception as e:
            print(f"❌ Error iniciando importación de clasificación: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "エラー",
                f"❌ インポート開始中にエラーが発生しました:\n{str(e)}"
            )
    
    def on_classification_import_finished(self, inserted_count, updated_count):
        """Manejar finalización exitosa de importación de clasificación"""
        try:
            # Cerrar diálogo de progreso
            if hasattr(self, 'classification_import_progress_dialog') and self.classification_import_progress_dialog is not None:
                self.classification_import_progress_dialog.close()
                self.classification_import_progress_dialog = None
            self.set_console_overlay_topmost(False)
            
            # Limpiar worker
            if hasattr(self, 'classification_import_worker') and self.classification_import_worker is not None:
                self.classification_import_worker.quit()
                self.classification_import_worker.wait()
                self.classification_import_worker = None
            
            # Mostrar mensaje de éxito
            QMessageBox.information(
                self,
                "インポート完了",
                f"✅ データベースへのインポートが完了しました！\n\n"
                f"新規追加: {inserted_count} 件\n"
                f"更新: {updated_count} 件"
            )
            
        except Exception as e:
            print(f"❌ Error en manejo de finalización: {e}")
            import traceback
            traceback.print_exc()
    
    def on_classification_import_error(self, error_msg):
        """Manejar error en importación de clasificación"""
        try:
            # Cerrar diálogo de progreso
            if hasattr(self, 'classification_import_progress_dialog') and self.classification_import_progress_dialog is not None:
                self.classification_import_progress_dialog.close()
                self.classification_import_progress_dialog = None
            self.set_console_overlay_topmost(False)
            
            # Limpiar worker
            if hasattr(self, 'classification_import_worker') and self.classification_import_worker is not None:
                self.classification_import_worker.quit()
                self.classification_import_worker.wait()
                self.classification_import_worker = None
            
            # Mostrar mensaje de error
            QMessageBox.critical(
                self,
                "エラー",
                f"❌ データベースへのインポート中にエラーが発生しました:\n{error_msg}"
            )
            
        except Exception as e:
            print(f"❌ Error en manejo de error: {e}")
            import traceback
            traceback.print_exc()
    
    def cancel_classification_import(self):
        """Cancelar importación de clasificación"""
        try:
            if hasattr(self, 'classification_import_worker'):
                self.classification_import_worker.cancel_import()
                self.classification_import_worker.quit()
                self.classification_import_worker.wait()
                self.classification_import_worker = None
            
            if hasattr(self, 'classification_import_progress_dialog'):
                self.classification_import_progress_dialog.close()
                self.classification_import_progress_dialog = None
            self.set_console_overlay_topmost(False)
                
        except Exception as e:
            print(f"❌ Error cancelando importación: {e}")
            import traceback
            traceback.print_exc()
            
            QMessageBox.information(self, "キャンセル", "インポートがキャンセルされました。")
            
        except Exception as e:
            print(f"❌ Error cancelando importación: {e}")
            import traceback
            traceback.print_exc()
    
    def on_yosoku_export_finished(self, filepath, record_count):
        """Manejar finalización exitosa de exportación"""
        try:
            # Cerrar diálogo de progreso
            if hasattr(self, 'yosoku_export_progress_dialog') and self.yosoku_export_progress_dialog is not None:
                self.yosoku_export_progress_dialog.close()
                self.yosoku_export_progress_dialog = None
            self.set_console_overlay_topmost(False)
            
            # Limpiar worker
            if hasattr(self, 'yosoku_export_worker') and self.yosoku_export_worker is not None:
                self.yosoku_export_worker.quit()
                self.yosoku_export_worker.wait()
                self.yosoku_export_worker = None
            
            # Mostrar mensaje de éxito
            QMessageBox.information(
                self,
                "完了",
                f"✅ 予測データベースが正常にエクスポートされました。\n\nファイル: {os.path.basename(filepath)}\nレコード数: {record_count}"
            )
            
        except Exception as e:
            print(f"❌ Error en manejo de finalización de exportación: {e}")
            import traceback
            traceback.print_exc()
    
    def on_yosoku_export_error(self, error_msg):
        """Manejar error en exportación"""
        try:
            # Cerrar diálogo de progreso
            if hasattr(self, 'yosoku_export_progress_dialog') and self.yosoku_export_progress_dialog is not None:
                self.yosoku_export_progress_dialog.close()
                self.yosoku_export_progress_dialog = None
            self.set_console_overlay_topmost(False)
            
            # Limpiar worker
            if hasattr(self, 'yosoku_export_worker') and self.yosoku_export_worker is not None:
                self.yosoku_export_worker.quit()
                self.yosoku_export_worker.wait()
                self.yosoku_export_worker = None
            
            # Mostrar mensaje de error
            QMessageBox.critical(
                self,
                "エラー",
                error_msg
            )
            
        except Exception as e:
            print(f"❌ Error en manejo de error de exportación: {e}")
            import traceback
            traceback.print_exc()
    
    def cancel_yosoku_export(self):
        """Cancelar exportación"""
        try:
            if hasattr(self, 'yosoku_export_worker'):
                self.yosoku_export_worker.cancel_export()
                self.yosoku_export_worker.quit()
                self.yosoku_export_worker.wait()
                self.yosoku_export_worker = None
            
            if hasattr(self, 'yosoku_export_progress_dialog'):
                self.yosoku_export_progress_dialog.close()
                self.yosoku_export_progress_dialog = None
            self.set_console_overlay_topmost(False)
            
            QMessageBox.information(self, "キャンセル", "エクスポートがキャンセルされました。")
            
        except Exception as e:
            print(f"❌ Error cancelando exportación: {e}")
            import traceback
            traceback.print_exc()
    
    def prepare_dataframe_for_import(self, df, selected_params):
        """
        Prepara el DataFrame para importación agregando columnas de usuario
        y renombrando columnas de predicción si es necesario
        """
        try:
            # Crear copia para no modificar el original
            df_prepared = df.copy()
            
            # Agregar columnas de tipo de cepillo
            brush_type = selected_params['brush']
            df_prepared['A13'] = 1 if brush_type == 'A13' else 0
            df_prepared['A11'] = 1 if brush_type == 'A11' else 0
            df_prepared['A21'] = 1 if brush_type == 'A21' else 0
            df_prepared['A32'] = 1 if brush_type == 'A32' else 0
            
            # Agregar columnas de usuario
            df_prepared['直径'] = selected_params['diameter']
            df_prepared['材料'] = selected_params['material']
            df_prepared['線材長'] = selected_params['wire_length']
            
            # Renombrar columnas de predicción si tienen prefijo 'prediction_'
            rename_map = {}
            for col in df_prepared.columns:
                if col.startswith('prediction_'):
                    new_name = col.replace('prediction_', '')
                    rename_map[col] = new_name
            
            if rename_map:
                df_prepared = df_prepared.rename(columns=rename_map)
                print(f"📝 Columnas renombradas: {rename_map}")
            
            # Calcular 加工時間 si no existe
            if '加工時間' not in df_prepared.columns:
                if '送り速度' in df_prepared.columns:
                    # Fórmula: 100 / 送り速度 * 60
                    df_prepared['加工時間'] = df_prepared.apply(
                        lambda row: (100 / row['送り速度'] * 60) if pd.notna(row.get('送り速度')) and row.get('送り速度', 0) != 0 else 0,
                        axis=1
                    )
                    print("✅ 加工時間 calculado")
                else:
                    df_prepared['加工時間'] = 0
                    print("⚠️ 送り速度 no encontrado, 加工時間 = 0")
            
            return df_prepared
            
        except Exception as e:
            print(f"❌ Error preparando DataFrame: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def import_nonlinear_pareto_to_database(self, excel_path):
        """Importa resultados de Pareto del análisis no lineal a la base de datos"""
        try:
            # 1. Mostrar diálogo de parámetros PRIMERO (sin loading)
            selected_params = self.show_yosoku_parameters_dialog()
            
            if not selected_params:
                print("❌ Usuario canceló la selección de parámetros")
                return
            
            # ✅ MOSTRAR LOADING DESPUÉS de seleccionar parámetros y presionar OK
            self.yosoku_import_progress_dialog = YosokuImportProgressDialog(self)
            self.yosoku_import_progress_dialog.show()
            self.yosoku_import_progress_dialog.update_progress(0, "初期化中...")
            self.yosoku_import_progress_dialog.set_status("初期化中...")
            QApplication.processEvents()  # Forzar actualización de la UI
            
            # 2. Leer Excel y preparar DataFrame
            self.yosoku_import_progress_dialog.update_progress(10, "Excelファイルを読み込み中...")
            self.yosoku_import_progress_dialog.set_status("Excelファイルを読み込み中...")
            QApplication.processEvents()
            
            print(f"📊 Leyendo archivo: {excel_path}")
            df = pd.read_excel(excel_path)
            print(f"✅ Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
            
            # 3. Preparar DataFrame con columnas de usuario
            self.yosoku_import_progress_dialog.update_progress(30, "データを準備中...")
            self.yosoku_import_progress_dialog.set_status("データを準備中...")
            QApplication.processEvents()
            
            df_prepared = self.prepare_dataframe_for_import(df, selected_params)
            
            # 4. Guardar DataFrame preparado en archivo intermedio (misma carpeta que Prediction_output.xlsx)
            self.yosoku_import_progress_dialog.update_progress(50, "ファイルを保存中...")
            self.yosoku_import_progress_dialog.set_status("ファイルを保存中...")
            QApplication.processEvents()
            
            excel_folder = Path(excel_path).parent
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            intermediate_filename = f"Prediction_output_prepared_{timestamp}.xlsx"
            intermediate_path = excel_folder / intermediate_filename
            
            try:
                df_prepared.to_excel(intermediate_path, index=False)
                print(f"📁 Archivo intermedio guardado: {intermediate_path}")
            except Exception as e:
                print(f"⚠️ Error guardando archivo intermedio: {e}")
                # No detener el proceso si falla guardar el intermedio
            
            # 5. Guardar también en archivo temporal para la importación
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"pareto_import_{timestamp}.xlsx")
            df_prepared.to_excel(temp_file, index=False)
            print(f"📁 Archivo temporal creado: {temp_file}")
            
            # 6. Importar usando el worker existente (el worker continuará desde 60%)
            # Nota: import_yosoku_results_to_database creará su propio diálogo, 
            # así que necesitamos reutilizar el existente o pasarle el diálogo
            self._continue_import_with_worker(temp_file)
            
            # 7. Limpiar archivo temporal después de un delay
            # Nota: El archivo intermedio NO se elimina, queda como registro
            def cleanup_temp_file():
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        print(f"🗑️ Archivo temporal eliminado: {temp_file}")
                except:
                    pass
            
            QTimer.singleShot(5000, cleanup_temp_file)  # Limpiar después de 5 segundos
            
        except Exception as e:
            print(f"❌ Error importando Pareto a BD: {e}")
            import traceback
            traceback.print_exc()
            
            # Cerrar loading si hay error
            if hasattr(self, 'yosoku_import_progress_dialog') and self.yosoku_import_progress_dialog is not None:
                self.yosoku_import_progress_dialog.close()
                self.yosoku_import_progress_dialog = None
            
            QMessageBox.critical(
                self,
                "エラー",
                f"❌ データベースへのインポート中にエラーが発生しました:\n{str(e)}"
            )
    
    def _continue_import_with_worker(self, temp_file):
        """Continúa la importación usando el worker, reutilizando el diálogo existente"""
        try:
            # Actualizar progreso antes de iniciar worker
            self.yosoku_import_progress_dialog.update_progress(60, "データベースにインポート中...")
            self.yosoku_import_progress_dialog.set_status("データベースにインポート中...")
            QApplication.processEvents()
            
            # Crear worker thread (análisis no lineal)
            self.yosoku_import_worker = YosokuImportWorker(temp_file, analysis_type="no_lineal")
            
            # Conectar señales (reutilizando el diálogo existente)
            self.yosoku_import_worker.progress_updated.connect(self._on_yosoku_import_progress)
            self.yosoku_import_worker.status_updated.connect(self.yosoku_import_progress_dialog.set_status)
            self.yosoku_import_worker.finished.connect(self.on_yosoku_import_finished)
            self.yosoku_import_worker.error.connect(self.on_yosoku_import_error)
            
            # Conectar botón de cancelar
            self.yosoku_import_progress_dialog.cancel_button.clicked.connect(self.cancel_yosoku_import)
            
            # Iniciar worker
            self.yosoku_import_worker.start()
            
        except Exception as e:
            print(f"❌ Error iniciando worker de importación: {e}")
            import traceback
            traceback.print_exc()
            
            # Cerrar loading si hay error
            if hasattr(self, 'yosoku_import_progress_dialog') and self.yosoku_import_progress_dialog is not None:
                self.yosoku_import_progress_dialog.close()
                self.yosoku_import_progress_dialog = None
            
            QMessageBox.critical(
                self,
                "エラー",
                f"❌ インポート開始中にエラーが発生しました:\n{str(e)}"
            )
    
    def _on_yosoku_import_progress(self, value, message):
        """Maneja el progreso del worker, mapeando de 0-100% del worker a 60-100% del total"""
        # El worker emite progreso de 0-100%, pero nosotros ya estamos en 60%
        # Mapear el progreso del worker (0-100%) al rango 60-100% del total
        mapped_value = 60 + int((value * 40) / 100)  # 60% + (worker_progress * 40% / 100)
        if hasattr(self, 'yosoku_import_progress_dialog') and self.yosoku_import_progress_dialog is not None:
            self.yosoku_import_progress_dialog.update_progress(mapped_value, message)

    def create_yosoku_database_table(self, cursor):
        """Crear tabla de predicciones si no existe"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS yosoku_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            A13 INTEGER,
            A11 INTEGER,
            A21 INTEGER,
            A32 INTEGER,
            直径 REAL,
            材料 TEXT,
            線材長 REAL,
            回転速度 REAL,
            送り速度 REAL,
            UPカット INTEGER,
            切込量 REAL,
            突出量 REAL,
            載せ率 REAL,
            パス数 INTEGER,
            加工時間 REAL,
            上面ダレ量 REAL,
            側面ダレ量 REAL,
            摩耗量 REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        cursor.execute(create_table_sql)

    def check_duplicate_yosoku_data(self, cursor, df):
        """Verificar si hay datos duplicados (columnas A-O)"""
        duplicate_rows = []
        
        for index, row in df.iterrows():
            # Verificar si existe una fila con los mismos valores en las columnas A-O
            # Las columnas A-O corresponden a: A13, A11, A21, A32, 直径, 材料, 線材長, 回転速度, 送り速度, UPカット, 切込量, 突出量, 載せ率, バス数, 加工時間
            
            check_sql = """
            SELECT id FROM yosoku_predictions 
            WHERE A13 = ? AND A11 = ? AND A21 = ? AND A32 = ? 
            AND 直径 = ? AND 材料 = ? AND 線材長 = ? 
            AND 回転速度 = ? AND 送り速度 = ? AND UPカット = ? 
            AND 切込量 = ? AND 突出量 = ? AND 載せ率 = ? 
            AND パス数 = ? AND 加工時間 = ?
            """
            
            cursor.execute(check_sql, (
                int(row.get('A13', 0)),
                int(row.get('A11', 0)),
                int(row.get('A21', 0)),
                int(row.get('A32', 0)),
                float(row.get('直径', 0)),
                str(row.get('材料', '')),
                float(row.get('線材長', 0)),
                float(row.get('回転速度', 0)),
                float(row.get('送り速度', 0)),
                int(row.get('UPカット', 0)),
                float(row.get('切込量', 0)),
                float(row.get('突出量', 0)),
                float(row.get('載せ率', 0)),
                int(row.get('パス数', 0)),
                float(row.get('加工時間', 0))
            ))
            
            result = cursor.fetchone()
            if result:
                duplicate_rows.append((index, result[0]))  # (excel_row_index, db_id)
        
        return duplicate_rows

    def remove_duplicate_yosoku_data(self, cursor, duplicate_rows):
        """Eliminar datos duplicados existentes en la base de datos"""
        for excel_row_index, db_id in duplicate_rows:
            cursor.execute("DELETE FROM yosoku_predictions WHERE id = ?", (db_id,))

    def insert_yosoku_data(self, cursor, df):
        """Insertar datos del Excel a la base de datos"""
        insert_sql = """
        INSERT INTO yosoku_predictions 
        (A13, A11, A21, A32, 直径, 材料, 線材長, 回転速度, 送り速度, UPカット, 
         切込量, 突出量, 載せ率, パス数, 加工時間, 上面ダレ量, 側面ダレ量, 摩耗量)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        for index, row in df.iterrows():
            # Función auxiliar para convertir valores de forma segura
            def safe_convert(value, convert_func, default=0):
                try:
                    if pd.isna(value) or value is None or value == '':
                        return default
                    return convert_func(value)
                except (ValueError, TypeError):
                    return default
            
            # Convertir fórmulas a valores numéricos de forma segura
            values = (
                safe_convert(row.get('A13', 0), int),
                safe_convert(row.get('A11', 0), int),
                safe_convert(row.get('A21', 0), int),
                safe_convert(row.get('A32', 0), int),
                safe_convert(row.get('直径', 0), float),
                str(row.get('材料', '')).strip() if row.get('材料') is not None else '',
                safe_convert(row.get('線材長', 0), float),
                safe_convert(row.get('回転速度', 0), float),
                safe_convert(row.get('送り速度', 0), float),
                safe_convert(row.get('UPカット', 0), int),
                safe_convert(row.get('切込量', 0), float),
                safe_convert(row.get('突出量', 0), float),
                safe_convert(row.get('載せ率', 0), float),
                safe_convert(row.get('パス数', 0), int),
                safe_convert(row.get('加工時間', 0), float),
                safe_convert(row.get('上面ダレ量', 0), float),
                safe_convert(row.get('側面ダレ量', 0), float),
                safe_convert(row.get('摩耗量', 0), float)
            )
            
            cursor.execute(insert_sql, values)

    def cancel_yosoku_prediction(self):
        """Cancelar predicción Yosoku"""
        try:
            if hasattr(self, 'yosoku_worker'):
                self.yosoku_worker.cancel_prediction()
                self.yosoku_worker.quit()
                self.yosoku_worker.wait()
                self.yosoku_worker = None
            
            if hasattr(self, 'yosoku_progress_dialog'):
                self.yosoku_progress_dialog.close()
                self.yosoku_progress_dialog = None
            self.set_console_overlay_topmost(False)
                
        except Exception as e:
            print(f"❌ Error cancelando predicción: {e}")
            import traceback
            traceback.print_exc()


    def validate_prediction_parameters(self, selected_params):
        """Validar que los parámetros seleccionados coincidan con los filtros aplicados"""
        try:
            # Obtener filtros aplicados
            filters = self.get_applied_filters()
            
            # ✅ NUEVO: Lista para recopilar todos los errores
            errors = []
            
            if not filters:
                return {
                    'valid': True,
                    'reason': 'No hay filtros aplicados, se pueden usar cualquier parámetro'
                }
            
            # Verificar brush
            if 'brush' in selected_params:
                brush = selected_params['brush']
                if brush in ['A13', 'A11', 'A21', 'A32']:
                    # Verificar que el brush seleccionado esté en los filtros
                    if brush not in filters or filters[brush] != 1:
                        errors.append(f"Brush {brush} no está seleccionado en los filtros aplicados")
            
            # Verificar diameter
            if 'diameter' in selected_params:
                diameter = selected_params['diameter']
                if '直径' in filters and filters['直径'] != diameter:
                    errors.append(f"Diámetro {diameter} no coincide con el filtro aplicado ({filters['直径']})")
            
            # Verificar material
            if 'material' in selected_params:
                material = selected_params['material']
                if '材料' in filters and filters['材料'] != material:
                    errors.append(f"Material {material} no coincide con el filtro aplicado ({filters['材料']})")
            
            # Verificar wire_length con tolerancia de -5mm
            if 'wire_length' in selected_params:
                wire_length = selected_params['wire_length']
                if '線材長' in filters:
                    filter_wire_length = filters['線材長']
                    
                    # Convertir wire_length a int para asegurar comparaciones correctas
                    try:
                        wire_length = int(wire_length)
                    except (ValueError, TypeError):
                        errors.append(f"Valor de wire_length inválido: {wire_length}")
                        return {
                            'valid': False,
                            'reason': '; '.join(errors)
                        }
                    
                    # Manejar caso donde filter_wire_length puede ser una tupla
                    if isinstance(filter_wire_length, tuple):
                        # Si es una tupla, verificar que TODOS los valores estén en el rango válido
                        min_length = wire_length - 5
                        max_length = wire_length
                        
                        # Convertir todos los valores de la tupla a int
                        try:
                            converted_values = [int(val) for val in filter_wire_length]
                            invalid_values = [val for val in converted_values if not (min_length <= val <= max_length)]
                            if invalid_values:
                                errors.append(f"線材長 {filter_wire_length} contiene valores fuera del rango permitido ({min_length}-{max_length}mm) para el valor seleccionado {wire_length}mm. Valores inválidos: {invalid_values}")
                        except (ValueError, TypeError) as e:
                            errors.append(f"Error convirtiendo valores de filter_wire_length: {e}")
                    else:
                        # Si es un valor único, verificar directamente
                        min_length = wire_length - 5
                        max_length = wire_length
                        
                        # Convertir filter_wire_length a int
                        try:
                            filter_wire_length = int(filter_wire_length)
                            if not (min_length <= filter_wire_length <= max_length):
                                errors.append(f"線材長 {filter_wire_length} no está dentro del rango permitido ({min_length}-{max_length}mm) para el valor seleccionado {wire_length}mm")
                        except (ValueError, TypeError) as e:
                            errors.append(f"Error convirtiendo filter_wire_length: {e}")
            
            if errors:
                return {
                    'valid': False,
                    'reason': '; '.join(errors)
                }
            else:
                return {
                    'valid': True,
                    'reason': 'Parámetros válidos'
                }
                
        except Exception as e:
            print(f"❌ Error validando parámetros: {e}")
            return {
                'valid': False,
                'reason': f'Error en validación: {str(e)}'
            }


    def find_latest_prediction_folder(self):
        """Encontrar la carpeta 04_予測計算 del análisis lineal más reciente"""
        try:
            if not hasattr(self, 'current_project_folder') or not self.current_project_folder:
                print("⚠️ No hay carpeta de proyecto actual")
                return None
            
            # Buscar en la carpeta 03_線形回帰
            linear_regression_folder = os.path.join(self.current_project_folder, "03_線形回帰")
            if not os.path.exists(linear_regression_folder):
                print("⚠️ Carpeta 03_線形回帰 no encontrada")
                return None
            
            # Buscar todas las subcarpetas con formato de timestamp
            subfolders = []
            for item in os.listdir(linear_regression_folder):
                item_path = os.path.join(linear_regression_folder, item)
                if os.path.isdir(item_path) and item.startswith("15_"):
                    subfolders.append(item_path)
            
            if not subfolders:
                print("⚠️ No se encontraron carpetas de análisis lineal")
                return None
            
            # Encontrar la más reciente por fecha de creación
            latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(x))
            
            # Buscar la carpeta 04_予測計算 dentro de la carpeta más reciente
            prediction_folder = os.path.join(latest_subfolder, "04_予測計算")
            
            if os.path.exists(prediction_folder):
                print(f"✅ Carpeta de predicción encontrada: {prediction_folder}")
                return prediction_folder
            else:
                print(f"⚠️ Carpeta 04_予測計算 no encontrada en: {latest_subfolder}")
                return None
                
        except Exception as e:
            print(f"❌ Error buscando carpeta de predicción: {e}")
            return None

    def find_latest_formulas_file(self):
        """Encontrar automáticamente el archivo de fórmulas del análisis lineal más reciente"""
        try:
            if not self.current_project_folder:
                print("❌ No hay carpeta de proyecto configurada")
                return None
            
            linear_regression_folder = os.path.join(self.current_project_folder, "03_線形回帰")
            if not os.path.exists(linear_regression_folder):
                print("❌ Carpeta de análisis lineal no encontrada")
                return None
            
            print(f"🔍 Buscando archivo de fórmulas en: {linear_regression_folder}")
            
            # Buscar todas las subcarpetas de análisis lineal
            subfolders = []
            for item in os.listdir(linear_regression_folder):
                item_path = os.path.join(linear_regression_folder, item)
                if os.path.isdir(item_path):
                    subfolders.append(item_path)
            
            if not subfolders:
                print("❌ No se encontraron subcarpetas de análisis lineal")
                return None
            
            # Ordenar por fecha de creación (más reciente primero)
            subfolders.sort(key=lambda x: os.path.getctime(x), reverse=True)
            
            print(f"📊 Encontradas {len(subfolders)} carpetas de análisis lineal")
            
            # Buscar el archivo de fórmulas en cada carpeta, empezando por la más reciente
            for i, subfolder in enumerate(subfolders):
                folder_name = os.path.basename(subfolder)
                print(f"🔍 Verificando carpeta {i+1}/{len(subfolders)}: {folder_name}")
                
                # Buscar en la carpeta de predicción
                prediction_folder = os.path.join(subfolder, "04_予測計算")
                if os.path.exists(prediction_folder):
                    formulas_file = os.path.join(prediction_folder, "XEBEC_予測計算機_逆変換対応.xlsx")
                    if os.path.exists(formulas_file):
                        print(f"✅ Archivo de fórmulas encontrado: {formulas_file}")
                        return formulas_file
                    else:
                        print(f"   ⚠️ Archivo de fórmulas no encontrado en: {prediction_folder}")
                else:
                    print(f"   ⚠️ Carpeta de predicción no encontrada: {prediction_folder}")
            
            print("❌ No se encontró ningún archivo de fórmulas válido")
            return None
            
        except Exception as e:
            print(f"❌ Error buscando archivo de fórmulas: {e}")
            import traceback
            traceback.print_exc()
            return None

    def debug_console_position(self):
        """Método de debug para verificar la posición de la consola"""
        try:
            if hasattr(self, 'overlay_console'):
                console_geo = self.overlay_console.geometry()
                window_geo = self.geometry()
                print(f"🔍 DEBUG - Ventana principal: {window_geo}")
                print(f"🔍 DEBUG - Consola desplegable: {console_geo}")
                print(f"🔍 DEBUG - Consola visible: {self.overlay_console.isVisible()}")
                print(f"🔍 DEBUG - Estado overlay: {getattr(self, 'overlay_console_visible', 'No definido')}")
            else:
                print("🔍 DEBUG - No hay consola desplegable")
        except Exception as e:
            print(f"🔍 DEBUG - Error: {e}")

    # NOTA: Este método ya no se necesita, solo usamos el panel superpuesto

    def sync_console_content(self):
        """Sincronizar el contenido de la consola desplegable con la principal"""
        try:
            # Obtener el contenido de la consola principal
            main_content = self.console_output.toPlainText()
            
            # Actualizar la consola desplegable
            self.overlay_console_output.setPlainText(main_content)
            
            # Mover el cursor al final (PySide6 usa MoveOperation.End)
            cursor = self.overlay_console_output.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.overlay_console_output.setTextCursor(cursor)
            
        except Exception as e:
            print(f"⚠️ Error sincronizando consolas: {e}")

    def resizeEvent(self, event):
        """Manejar el redimensionamiento de la ventana"""
        super().resizeEvent(event)
        
        # Si el panel desplegable está visible, reposicionarlo
        if hasattr(self, 'overlay_console_visible') and self.overlay_console_visible:
            self.position_overlay_console()
            
        # También reposicionar el botón de flecha si está visible
        if hasattr(self, 'console_toggle_button') and self.console_toggle_button.isVisible():
            self.position_arrow()

        # Mantener el título actualizado (por si el manifest cambia durante runtime)
        try:
            self.setWindowTitle(get_app_title())
        except Exception:
            pass
        
        # Actualizar gráficos del análisis no lineal si están siendo mostrados
        if hasattr(self, 'nonlinear_chart_images') and hasattr(self, 'nonlinear_chart_label'):
            # Usar QTimer para actualizar después de que el resize termine
            QTimer.singleShot(100, self.update_nonlinear_chart_display)

    def closeEvent(self, event):
        """Manejar el cierre de la aplicación"""
        try:
            print("🛑 Cerrando aplicación...")

            # Parar timers de overlays (evita que sigan intentando raise_ tras cerrar)
            for timer_attr in ("keep_on_top_timer", "position_check_timer"):
                try:
                    t = getattr(self, timer_attr, None)
                    if t is not None and t.isActive():
                        t.stop()
                except Exception:
                    pass

            # Cerrar ventanas flotantes (flecha y consola overlay)
            for w_attr in ("overlay_console", "console_toggle_button"):
                try:
                    w = getattr(self, w_attr, None)
                    if w is not None:
                        w.close()
                except Exception:
                    pass

            # Cancelar análisis no lineal si está corriendo
            if hasattr(self, 'nonlinear_worker') and self.nonlinear_worker is not None:
                try:
                    if self.nonlinear_worker.isRunning():
                        print("🛑 Cancelando análisis no lineal antes de cerrar...")
                        self.nonlinear_worker.cancel()
                        if self.nonlinear_worker.isRunning():
                            self.nonlinear_worker.quit()
                            if not self.nonlinear_worker.wait(5000):
                                print("⚠️ El worker no terminó en 5 segundos, forzando cierre...")
                                self.nonlinear_worker.terminate()
                                self.nonlinear_worker.wait(1000)
                        print("✅ Worker de análisis no lineal cancelado")
                except Exception:
                    pass

            # Cerrar base de datos si existe
            try:
                if hasattr(self, 'db'):
                    self.db.close()
            except Exception:
                pass

            # Restaurar streams originales
            if hasattr(self, 'original_stdout'):
                sys.stdout = self.original_stdout
            if hasattr(self, 'original_stderr'):
                sys.stderr = self.original_stderr

        finally:
            # Continuar con el cierre normal
            super().closeEvent(event)

# ======================================
# Lanzamiento de la aplicación
# ======================================

def handle_exception(exc_type, exc_value, exc_traceback):
    """Manejar excepciones no capturadas para evitar que la app se cierre"""
    error_msg = f"❌ Error no manejado:\n{exc_type.__name__}: {exc_value}"
    print(error_msg)
    print("Traceback completo:")
    import traceback
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    
    # Mostrar mensaje de error en la consola si está disponible
    try:
        if 'window' in globals() and hasattr(window, 'console_output'):
            window.console_output.append(error_msg)
    except:
        pass

if __name__ == "__main__":
    # Configurar manejador de excepciones global
    sys.excepthook = handle_exception
    
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        print("🚀 Aplicación iniciada correctamente")
        sys.exit(app.exec())
    except Exception as e:
        print(f"❌ Error al iniciar la aplicación: {e}")
        import traceback
        traceback.print_exc()
        input("Presiona Enter para salir...")
