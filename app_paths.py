"""
Helpers de rutas para ejecutar tanto en dev (PyCharm) como empaquetado con PyInstaller,
y para instalar la app de forma profesional en Windows.

Incluye:
- `resource_path(...)`: recursos empaquetados (imágenes, fuentes) en dev / PyInstaller
- Rutas de datos (SQLite, backups, logs) en ubicaciones estándar:
  - Program Files: binarios (no escribible)
  - ProgramData: datos compartidos (BBDD + backups)
  - LocalAppData: logs/config por usuario

Uso:
    from app_paths import resource_path, get_db_path, get_backup_dir
    icon_path = resource_path("xebec_logo_88.png")
    db_path = get_db_path("results.db")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def resource_path(relative_path: str) -> str:
    """
    Devuelve una ruta absoluta a un recurso.

    - En dev: relativo al directorio del proyecto (donde vive este archivo)
    - En PyInstaller onefile: relativo a sys._MEIPASS
    - En PyInstaller onedir: también funciona con sys._MEIPASS (si existe) o al lado del exe
    """

    # PyInstaller (onefile / onedir) suele exponer sys._MEIPASS
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return str(Path(base) / relative_path)

    # Fallback: carpeta del código
    here = Path(__file__).resolve().parent
    candidate = here / relative_path
    if candidate.exists():
        return str(candidate)

    # Último fallback: CWD (por si se lanza desde otro entrypoint)
    return str(Path(os.getcwd()) / relative_path)


# ==============================
# Rutas de instalación (Windows)
# ==============================

COMPANY_NAME = "XebecTechnology"
APP_NAME = "0.00sec"


def _is_windows() -> bool:
    return os.name == "nt"


def get_program_data_dir(company: str = COMPANY_NAME, app: str = APP_NAME) -> Path:
    """
    Directorio compartido para datos (BBDD, backups) — típico:
      C:\\ProgramData\\Company\\App
    """
    if _is_windows():
        base = os.environ.get("PROGRAMDATA") or r"C:\ProgramData"
        return Path(base) / company / app
    # Fallback no-Windows (por si alguien ejecuta en otro OS)
    return Path.home() / f".{company}" / app


def get_local_appdata_dir(company: str = COMPANY_NAME, app: str = APP_NAME) -> Path:
    """
    Directorio por usuario para logs/config — típico:
      %LOCALAPPDATA%\\Company\\App
    """
    if _is_windows():
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / company / app
    return Path.home() / f".{company}" / app


def ensure_app_dirs(shared: bool = True) -> Path:
    """
    Asegura que existan las carpetas base de la app.

    - shared=True: ProgramData\\... (datos compartidos)
    - shared=False: LocalAppData\\... (por usuario)

    Si no se puede crear en ProgramData (permisos), hace fallback a LocalAppData.
    Devuelve la carpeta base efectiva.
    """
    base = get_program_data_dir() if shared else get_local_appdata_dir()
    try:
        base.mkdir(parents=True, exist_ok=True)
        return base
    except Exception:
        # Fallback seguro
        base = get_local_appdata_dir()
        base.mkdir(parents=True, exist_ok=True)
        return base


def get_data_dir(shared: bool = True) -> Path:
    base = ensure_app_dirs(shared=shared)
    d = base / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_backup_dir(shared: bool = True) -> Path:
    base = ensure_app_dirs(shared=shared)
    d = base / "backups"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_logs_dir() -> Path:
    base = ensure_app_dirs(shared=False)
    d = base / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_db_path(filename: str = "results.db", shared: bool = True) -> str:
    """
    Ruta absoluta para una DB (SQLite) ubicada en la carpeta de datos.
    Devuelve string para pasarlo directo a sqlite3.connect(...).
    """
    return str(get_data_dir(shared=shared) / filename)


def find_legacy_db(filename: str) -> Optional[Path]:
    """
    Busca una DB legacy (cuando antes estaba 'al lado del .py' o 'al lado del exe').
    """
    candidates: list[Path] = []

    # 1) CWD
    try:
        candidates.append(Path.cwd() / filename)
    except Exception:
        pass

    # 2) Junto al ejecutable (PyInstaller)
    try:
        candidates.append(Path(sys.executable).resolve().parent / filename)
    except Exception:
        pass

    # 3) Junto al código (dev)
    try:
        candidates.append(Path(__file__).resolve().parent / filename)
    except Exception:
        pass

    for p in candidates:
        try:
            if p.exists() and p.is_file() and p.stat().st_size > 0:
                return p
        except Exception:
            continue
    return None


def migrate_legacy_db_if_needed(filename: str, shared: bool = True) -> str:
    """
    Si la DB objetivo (ProgramData\\...\\data\\filename) no existe, intenta migrar
    una DB antigua encontrada en CWD/junto al exe/junto al código.

    Devuelve la ruta final (string) a usar.
    """
    target = Path(get_db_path(filename, shared=shared))
    if target.exists() and target.stat().st_size > 0:
        return str(target)

    legacy = find_legacy_db(filename)
    if legacy is None:
        return str(target)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Copia preservando timestamps; no borra el origen
        import shutil
        shutil.copy2(str(legacy), str(target))
        return str(target)
    except Exception:
        # Si falla migración, usar legacy (mejor que romper)
        return str(legacy)


