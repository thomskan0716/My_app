"""
Launcher liviano para mostrar un SplashScreen ANTES de importar 0sec.py (imports pesados).

Recomendación para EXE (PyInstaller):
  - entrypoint: bootstrap.py
  - incluir assets: xebec_logo_88.png (y opcional xebec.ico)
"""

from __future__ import annotations

import sys
import traceback

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPainterPath, QPixmap
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QSplashScreen, QMessageBox, QLabel

from app_paths import resource_path
from app_manifest import get_app_title


def _rounded_pixmap(src: QPixmap, radius: int) -> QPixmap:
    """
    Devuelve un pixmap con esquinas redondeadas usando transparencia real.
    """
    if src.isNull():
        return src

    radius = max(0, min(radius, min(src.width(), src.height()) // 2))

    out = QPixmap(src.size())
    out.fill(Qt.transparent)

    painter = QPainter(out)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

    path = QPainterPath()
    path.addRoundedRect(0, 0, src.width(), src.height(), radius, radius)
    painter.setClipPath(path)
    painter.drawPixmap(0, 0, src)
    painter.end()

    return out


def main() -> int:
    app = QApplication(sys.argv)

    # ES: Splash (antes de imports pesados)
    # EN: Splash (before heavy imports)
    # JA: スプラッシュ（重いimportの前）
    try:
        title = get_app_title()
        splash_path = resource_path("SlpashScreen_General.png")
        splash_pm = QPixmap(splash_path)
        if splash_pm.isNull():
            # ES: Fallback por si falta el asset
            # EN: Fallback in case the asset is missing
            # JA: アセットが無い場合のフォールバック
            splash_pm = QPixmap(resource_path("SplashScreen.scale-100.png"))

        # ES: Reescalar toda la imagen (30% más pequeño)
        # EN: Scale the whole image down (30% smaller)
        # JA: 画像全体を縮小（30%小さく）
        scale = 0.70
        target_w = max(1, int(splash_pm.width() * scale))
        target_h = max(1, int(splash_pm.height() * scale))
        splash_pm = splash_pm.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # ES: Redondear esquinas (ajustable) - proporcional al escalado
        # EN: Round corners (tunable) - proportional to scaling
        # JA: 角丸（調整可）- スケールに比例
        radius = max(8, int(24 * scale))
        splash_pm = _rounded_pixmap(splash_pm, radius=radius)
        splash = QSplashScreen(splash_pm)
        splash.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        splash.setAttribute(Qt.WA_TranslucentBackground, True)

        # ES: Versión (muy pequeño, blanco) centrada abajo del todo
        # EN: Version label (small, white) centered at the bottom
        # JA: バージョン表示（小さく白）を下部中央に配置
        try:
            ver_label = QLabel(title, splash)
            ver_label.setStyleSheet("color: white; background: transparent;")
            f = QFont()
            f.setPointSize(9)
            ver_label.setFont(f)
            ver_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            h = 18
            y = max(0, splash_pm.height() - h - 10)
            ver_label.setGeometry(0, y, splash_pm.width(), h)
            ver_label.show()
        except Exception:
            pass

        # ES: Centrar en pantalla | EN: Center on screen | JA: 画面中央に配置
        screen = app.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            x = geo.x() + (geo.width() - splash_pm.width()) // 2
            y = geo.y() + (geo.height() - splash_pm.height()) // 2
            splash.move(x, y)

        splash.show()
        app.processEvents()  # Force immediate paint
    except Exception:
        splash = None

    try:
        # ES: Import pesado: aquí es donde hoy se pierde tiempo antes de ver UI
        # EN: Heavy import: this is where time is spent before showing the UI
        # JA: 重いimport：UI表示前の待ち時間になりやすい箇所
        import importlib

        app_module = importlib.import_module("0sec")

        # ES: Mantener el mismo handler de excepciones global (si existe)
        # EN: Keep the same global exception handler (if present)
        # JA: グローバル例外ハンドラを維持（存在する場合）
        if hasattr(app_module, "handle_exception"):
            sys.excepthook = app_module.handle_exception

        window = app_module.MainWindow()
        # ES: Para que handle_exception (en el módulo 0sec) pueda acceder a window
        # EN: So handle_exception (in 0sec module) can access window
        # JA: 0sec モジュールの handle_exception から window にアクセスできるようにする
        try:
            setattr(app_module, "window", window)
        except Exception:
            pass

        window.show()
        if splash is not None:
            splash.finish(window)
        return app.exec()

    except Exception as e:
        if splash is not None:
            splash.close()

        # ES: Ya tenemos QApplication, así que podemos mostrar un error visual
        # EN: QApplication is already available, so we can show a visual error
        # JA: QApplication があるので、画面でエラー表示できる
        msg = f"No se pudo iniciar la aplicación:\n\n{type(e).__name__}: {e}"
        detail = traceback.format_exc()
        try:
            box = QMessageBox()
            box.setIcon(QMessageBox.Critical)
            box.setWindowTitle("Error de inicio")
            box.setText(msg)
            box.setDetailedText(detail)
            box.exec()
        except Exception:
            print(msg)
            print(detail)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


