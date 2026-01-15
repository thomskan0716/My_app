"""
Lee el manifiesto editable de la app (manifest.json) para obtener name/version.

Regla: primero intenta rutas "editables" (CWD / junto al exe),
y luego fallback al bundle (resource_path / carpeta del código).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _candidate_manifest_paths() -> list[Path]:
    paths: list[Path] = []

    # 1) CWD (útil cuando el usuario edita manifest junto al exe y ejecuta desde ahí)
    try:
        paths.append(Path.cwd() / "manifest.json")
    except Exception:
        pass

    # 2) Junto al ejecutable (PyInstaller)
    try:
        paths.append(Path(sys.executable).resolve().parent / "manifest.json")
    except Exception:
        pass

    # 3) Junto al script (dev)
    try:
        paths.append(Path(__file__).resolve().parent / "manifest.json")
    except Exception:
        pass

    # 4) Bundle de PyInstaller (_MEIPASS)
    base = getattr(sys, "_MEIPASS", None)
    if base:
        try:
            paths.append(Path(base) / "manifest.json")
        except Exception:
            pass

    # Dedup manteniendo orden
    out: list[Path] = []
    seen = set()
    for p in paths:
        s = str(p)
        if s not in seen:
            seen.add(s)
            out.append(p)
    return out


def read_manifest() -> Dict[str, Any]:
    for p in _candidate_manifest_paths():
        try:
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            continue
    return {}


def get_app_name(default: str = "0.00sec システム") -> str:
    m = read_manifest()
    v = m.get("name", default)
    return str(v) if v is not None else default


def get_app_version(default: str = "XXX") -> str:
    m = read_manifest()
    v = m.get("version", default)
    return str(v) if v is not None else default


def get_app_title(default_name: str = "0.00sec システム", default_version: str = "XXX") -> str:
    name = get_app_name(default_name)
    ver = get_app_version(default_version)
    return f"{name} ver. {ver}"




