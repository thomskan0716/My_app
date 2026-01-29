"""
ES: Diálogo para mostrar resultados de análisis de Pareto.
EN: Dialog to display Pareto analysis results.
JA: パレート解析結果を表示するダイアログ。

ES: Muestra gráficos y permite importar a base de datos.
EN: Shows plots and allows importing into the database.
JA: グラフ表示とDBインポートが可能。
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
    """ES: Diálogo para mostrar resultados de Pareto con opción de importar a BD
    EN: Dialog to show Pareto results with an option to import into the DB
    JA: パレート結果表示（DBインポート機能付き）ダイアログ
    """
    
    # ES: Señal emitida cuando se solicita importar a BD
    # EN: Signal emitted when an import-to-DB is requested
    # JA: DBインポート要求時に発行されるシグナル
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
        
        print(f"🔍 デバッグ ParetoResultsDialog.__init__: 読み込んだグラフ数 = {len(self.graph_paths)}")
    
    def setup_ui(self):
        """ES: Configura la interfaz
        EN: Build the UI
        JA: UIを構築
        """
        layout = QVBoxLayout()
        
        # ES: Título | EN: Title | JA: タイトル
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
        
        # ES: Información del gráfico | EN: Plot info | JA: グラフ情報
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 12px; color: #7f8c8d; margin: 5px;")
        layout.addWidget(self.info_label)
        
        # ES: Navegación | EN: Navigation | JA: ナビゲーション
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
        
        # ES: Botones de acción | EN: Action buttons | JA: 操作ボタン
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        
        # ES: Botón 戻る (Volver) | EN: Back button | JA: 戻るボタン
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
        
        # ES: Botón データベースにインポート | EN: Import-to-DB button | JA: DBインポートボタン
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
        """ES: Carga los gráficos de Pareto desde la carpeta
        EN: Load Pareto plots from the folder
        JA: フォルダからパレートグラフを読み込み
        """
        print(f"🔍 デバッグ load_graphs: フォルダー = {self.pareto_plots_folder}")
        print(f"🔍 DEBUG load_graphs: existe = {os.path.exists(self.pareto_plots_folder) if self.pareto_plots_folder else False}")
        
        if not os.path.exists(self.pareto_plots_folder):
            print(f"⚠️ グラフフォルダーが見つかりません: {self.pareto_plots_folder}")
            return
        
        # ES: Buscar archivos de imagen en la carpeta
        # EN: Search for image files in the folder
        # JA: フォルダ内の画像ファイルを探索
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.svg']
        for ext in image_extensions:
            pattern = os.path.join(self.pareto_plots_folder, ext)
            found = glob.glob(pattern)
            print(f"🔍 デバッグ load_graphs: 検索 {pattern}, 件数 = {len(found)}")
            self.graph_paths.extend(found)
        
        # ES: Ordenar por nombre | EN: Sort by name | JA: 名前順にソート
        self.graph_paths.sort()
        
        print(f"📊 Paretoグラフを {len(self.graph_paths)} 件検出")
        if self.graph_paths:
            print(f"🔍 デバッグ load_graphs: 先頭のグラフ = {[os.path.basename(p) for p in self.graph_paths[:3]]}")
    
    def update_display(self):
        """ES: Actualiza la visualización del gráfico actual
        EN: Update the current plot display
        JA: 現在のグラフ表示を更新
        """
        if not self.graph_paths:
            self.image_label.setText("グラフが見つかりません")
            self.info_label.setText("")
            self.counter_label.setText("0 / 0")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return
        
        # ES: Actualizar índice | EN: Update index | JA: インデックス更新
        if self.current_index < 0:
            self.current_index = 0
        elif self.current_index >= len(self.graph_paths):
            self.current_index = len(self.graph_paths) - 1
        
        # ES: Cargar imagen | EN: Load image | JA: 画像を読み込み
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
        
        # ES: Actualizar contador | EN: Update counter | JA: カウンター更新
        total = len(self.graph_paths)
        current = self.current_index + 1
        self.counter_label.setText(f"{current} / {total}")
        
        # ES: Actualizar información | EN: Update info | JA: 情報更新
        graph_name = os.path.basename(current_graph)
        self.info_label.setText(f"📊 {graph_name}")
        
        # ES: Actualizar estado de botones | EN: Update button state | JA: ボタン状態更新
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.graph_paths) - 1)
    
    def show_previous(self):
        """ES: Muestra el gráfico anterior
        EN: Show previous graph
        JA: 前のグラフを表示"""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
    
    def show_next(self):
        """ES: Muestra el siguiente gráfico
        EN: Show next graph
        JA: 次のグラフを表示"""
        if self.current_index < len(self.graph_paths) - 1:
            self.current_index += 1
            self.update_display()
    
    def import_to_database(self):
        """ES: Solicita importar a base de datos
        EN: Request importing into the database
        JA: DBへのインポートを要求
        """
        # ES: Verificar que el archivo existe
        # EN: Verify the file exists
        # JA: ファイルの存在確認
        if not os.path.exists(self.prediction_output_file):
            QMessageBox.warning(
                self,
                "エラー",
                f"❌ 予測結果ファイルが見つかりません:\n\n{self.prediction_output_file}"
            )
            return
        
        # ES: Emitir señal para que el padre maneje la importación
        # EN: Emit a signal so the parent can handle the import
        # JA: 親側でインポート処理できるようシグナル送信
        self.import_requested.emit(self.prediction_output_file)
    
    def resizeEvent(self, event):
        """ES: Redimensionar imagen cuando se redimensiona el diálogo
        EN: Resize image when the dialog is resized
        JA: ダイアログのリサイズに合わせて画像をリサイズ
        """
        super().resizeEvent(event)
        self.update_display()

