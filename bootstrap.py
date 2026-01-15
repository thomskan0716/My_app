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

    # Splash (antes de imports pesados)
    try:
        title = get_app_title()
        splash_path = resource_path("SlpashScreen_General.png")
        splash_pm = QPixmap(splash_path)
        if splash_pm.isNull():
            # Fallback por si falta el asset
            splash_pm = QPixmap(resource_path("SplashScreen.scale-100.png"))

        # Reescalar toda la imagen (30% más pequeño)
        scale = 0.70
        target_w = max(1, int(splash_pm.width() * scale))
        target_h = max(1, int(splash_pm.height() * scale))
        splash_pm = splash_pm.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Redondear esquinas (ajustable) - proporcional al escalado
        radius = max(8, int(24 * scale))
        splash_pm = _rounded_pixmap(splash_pm, radius=radius)
        splash = QSplashScreen(splash_pm)
        splash.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        splash.setAttribute(Qt.WA_TranslucentBackground, True)

        # Versión (muy pequeño, blanco) centrada abajo del todo
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

        # Centrar en pantalla
        screen = app.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            x = geo.x() + (geo.width() - splash_pm.width()) // 2
            y = geo.y() + (geo.height() - splash_pm.height()) // 2
            splash.move(x, y)

        splash.show()
        app.processEvents()  # fuerza el pintado inmediato
    except Exception:
        splash = None

    try:
        # Import pesado: aquí es donde hoy se pierde tiempo antes de ver UI
        import importlib

        app_module = importlib.import_module("0sec")

        # Mantener el mismo handler de excepciones global (si existe)
        if hasattr(app_module, "handle_exception"):
            sys.excepthook = app_module.handle_exception

        window = app_module.MainWindow()
        # Para que handle_exception (en el módulo 0sec) pueda acceder a window
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

        # Ya tenemos QApplication, así que podemos mostrar un error visual
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


