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
    # ES: Evita nombres raros en Windows | EN: Avoid odd filenames on Windows | JA: Windowsで変なファイル名を避ける
    prefix = prefix.strip() or "backup"
    return re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", prefix)


def build_backup_filename(prefix: str, when: Optional[datetime] = None, ext: str = ".db") -> str:
    when = when or datetime.now()
    prefix = _safe_prefix(prefix)
    return f"{prefix}_{when.strftime(TIMESTAMP_FMT)}{ext}"


def create_sqlite_backup(src_db_path: str, dest_db_path: str) -> None:
    """
    ES: Backup seguro de SQLite usando la Online Backup API.
    EN: Safe SQLite backup using the Online Backup API.
    JA: Online Backup API を使った安全なSQLiteバックアップ。

    ES: Esto evita corrupción incluso si la DB está en uso (con WAL es aún mejor).
    EN: This avoids corruption even if the DB is in use (even better with WAL).
    JA: DB使用中でも破損を防ぐ（WALならさらに良い）。
    """
    src = sqlite3.connect(src_db_path, timeout=30)
    try:
        dest = sqlite3.connect(dest_db_path)
        try:
            # ES: Si el origen está en WAL, este backup es consistente.
            # EN: If the source uses WAL, this backup is consistent.
            # JA: ソースがWALなら、このバックアップは整合性が保たれる。
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
    ES: Crea 1 backup por día (si aún no se hizo hoy).
    EN: Create one backup per day (if not already done today).
    JA: 1日1回バックアップを作成（当日分が未作成の場合）。

    ES: Usa un marker file en el directorio de backups.
    EN: Uses a marker file in the backups directory.
    JA: バックアップディレクトリ内のマーカーファイルを使用。
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
        # ES: Si el marker está corrupto, seguimos y creamos backup
        # EN: If the marker is corrupted, proceed and create a backup
        # JA: マーカーが壊れていても続行してバックアップを作成
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
    ES: Retención simple:
    EN: Simple retention policy:
    JA: シンプルな保持ポリシー：
    ES: - Mantener los últimos `keep_daily` días (1 backup por día, el más reciente)
    EN: - Keep the last `keep_daily` days (1 backup per day, the most recent)
    JA: - 直近 `keep_daily` 日を保持（1日1個、最新）
    ES: - Mantener `keep_monthly` meses (1 backup por mes, el más reciente)
    EN: - Keep `keep_monthly` months (1 backup per month, the most recent)
    JA: - `keep_monthly` ヶ月を保持（1ヶ月1個、最新）
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    items = sorted(_iter_backups(backup_dir, prefix=prefix), key=lambda x: x[1], reverse=True)
    if not items:
        return

    # ES: 1) Mantener 1 por día (los más recientes)
    # EN: 1) Keep 1 per day (most recent)
    # JA: 1) 1日1個保持（最新）
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

    # ES: 2) Mantener 1 por mes (los más recientes)
    # EN: 2) Keep 1 per month (most recent)
    # JA: 2) 1ヶ月1個保持（最新）
    month_keys_seen: set[str] = set()
    for p, ts in items:
        month_key = ts.strftime("%Y-%m")
        if month_key in month_keys_seen:
            continue
        month_keys_seen.add(month_key)
        keep.add(p)
        if len(month_keys_seen) >= keep_monthly:
            break

    # ES: 3) Borrar el resto | EN: 3) Delete the rest | JA: 3) 残りを削除
    for p, _ in items:
        if p in keep:
            continue
        try:
            p.unlink(missing_ok=True)
        except Exception:
            continue




