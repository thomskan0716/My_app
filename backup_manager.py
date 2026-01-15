from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


TIMESTAMP_FMT = "%Y-%m-%d_%H%M%S"


@dataclass(frozen=True)
class BackupResult:
    db_path: str
    backup_path: str
    created_at: datetime


def _safe_prefix(prefix: str) -> str:
    # Evita nombres raros en Windows
    prefix = prefix.strip() or "backup"
    return re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", prefix)


def build_backup_filename(prefix: str, when: Optional[datetime] = None, ext: str = ".db") -> str:
    when = when or datetime.now()
    prefix = _safe_prefix(prefix)
    return f"{prefix}_{when.strftime(TIMESTAMP_FMT)}{ext}"


def create_sqlite_backup(src_db_path: str, dest_db_path: str) -> None:
    """
    Backup seguro de SQLite usando la Online Backup API.
    Esto evita corrupción incluso si la DB está en uso (con WAL es aún mejor).
    """
    src = sqlite3.connect(src_db_path, timeout=30)
    try:
        dest = sqlite3.connect(dest_db_path)
        try:
            # Si el origen está en WAL, este backup es consistente.
            src.backup(dest)
            dest.commit()
        finally:
            dest.close()
    finally:
        src.close()


def create_backup(db_path: str, backup_dir: Path, prefix: str = "results") -> BackupResult:
    backup_dir.mkdir(parents=True, exist_ok=True)
    when = datetime.now()
    backup_path = backup_dir / build_backup_filename(prefix=prefix, when=when, ext=".db")
    create_sqlite_backup(db_path, str(backup_path))
    return BackupResult(db_path=db_path, backup_path=str(backup_path), created_at=when)


def _marker_file(backup_dir: Path, prefix: str) -> Path:
    prefix = _safe_prefix(prefix)
    return backup_dir / f".{prefix}_last_daily_backup.txt"


def auto_daily_backup(db_path: str, backup_dir: Path, prefix: str = "results") -> Optional[BackupResult]:
    """
    Crea 1 backup por día (si aún no se hizo hoy).
    Usa un marker file en el directorio de backups.
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    marker = _marker_file(backup_dir, prefix=prefix)
    today = datetime.now().strftime("%Y-%m-%d")

    try:
        if marker.exists():
            last = marker.read_text(encoding="utf-8").strip()
            if last.startswith(today):
                return None
    except Exception:
        # Si el marker está corrupto, seguimos y creamos backup
        pass

    res = create_backup(db_path=db_path, backup_dir=backup_dir, prefix=prefix)
    try:
        marker.write_text(f"{today} {Path(res.backup_path).name}\n", encoding="utf-8")
    except Exception:
        pass
    return res


_BACKUP_RE = re.compile(r"^(?P<prefix>.+)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{6})\.db$", re.IGNORECASE)


def _iter_backups(backup_dir: Path, prefix: str) -> Iterable[tuple[Path, datetime]]:
    prefix = _safe_prefix(prefix)
    for p in backup_dir.glob(f"{prefix}_*.db"):
        m = _BACKUP_RE.match(p.name)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group("ts"), TIMESTAMP_FMT)
        except Exception:
            continue
        yield p, ts


def prune_backups(backup_dir: Path, prefix: str = "results", keep_daily: int = 30, keep_monthly: int = 12) -> None:
    """
    Retención simple:
    - Mantener los últimos `keep_daily` días (1 backup por día, el más reciente)
    - Mantener `keep_monthly` meses (1 backup por mes, el más reciente)
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    items = sorted(_iter_backups(backup_dir, prefix=prefix), key=lambda x: x[1], reverse=True)
    if not items:
        return

    # 1) Mantener 1 por día (los más recientes)
    keep: set[Path] = set()
    day_keys_seen: set[str] = set()
    for p, ts in items:
        day_key = ts.strftime("%Y-%m-%d")
        if day_key in day_keys_seen:
            continue
        day_keys_seen.add(day_key)
        keep.add(p)
        if len(day_keys_seen) >= keep_daily:
            break

    # 2) Mantener 1 por mes (los más recientes)
    month_keys_seen: set[str] = set()
    for p, ts in items:
        month_key = ts.strftime("%Y-%m")
        if month_key in month_keys_seen:
            continue
        month_keys_seen.add(month_key)
        keep.add(p)
        if len(month_keys_seen) >= keep_monthly:
            break

    # 3) Borrar el resto
    for p, _ in items:
        if p in keep:
            continue
        try:
            p.unlink(missing_ok=True)
        except Exception:
            continue




