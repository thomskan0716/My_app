"""
ES: Lee el manifiesto editable de la app (manifest.json) para obtener name/version.
EN: Read the app's editable manifest (manifest.json) to get name/version.
JA: 編集可能なマニフェスト（manifest.json）から name/version を取得。

ES: Regla: primero intenta rutas "editables" (CWD / junto al exe),
EN: Rule: first try "editable" paths (CWD / next to the exe),
JA: ルール：まず「編集しやすい」場所（CWD / exe隣）を試す。
ES: y luego fallback al bundle (resource_path / carpeta del código).
EN: then fall back to the bundled resources (resource_path / code folder).
JA: 次にバンドル側へフォールバック（resource_path / コードフォルダ）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _candidate_manifest_paths() -> list[Path]:
    paths: list[Path] = []

    # ES: 1) CWD (útil cuando el usuario edita manifest junto al exe y ejecuta desde ahí)
    # EN: 1) CWD (useful when the user edits the manifest next to the exe and runs from there)
    # JA: 1) CWD（exe隣で編集してそこから実行する場合に有用）
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

    # ES: Dedup manteniendo orden | EN: De-duplicate while preserving order | JA: 順序を保って重複排除
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




