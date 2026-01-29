"""
ES: Migra results.db para añadir/reordenar la columna 線材本数.
EN: Migrate results.db to add/reorder the 線材本数 column.
JP: results.db に 線材本数 列を追加/並べ替えする移行スクリプト。
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import time
import time as _time
import sys
from typing import Dict, List, Tuple


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _table_exists(cur: sqlite3.Cursor, table: str) -> bool:
    row = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _pragma_table_info(cur: sqlite3.Cursor, table: str) -> List[Tuple]:
    return cur.execute(f"PRAGMA table_info({_quote_ident(table)});").fetchall()


def _ensure_wire_count_after_wire_length(names: List[str]) -> List[str]:
    """
    ES: Devuelve un orden de columnas donde 線材本数 va justo detrás de 線材長.
    EN: Return a column order where 線材本数 is right after 線材長.
    JP: 線材本数 を 線材長 の直後に配置した列順を返す。
    """
    out: List[str] = []
    for n in names:
        if n == "線材本数":
            continue
        out.append(n)
        if n == "線材長" and "線材本数" not in out:
            out.append("線材本数")
    if "線材本数" not in out:
        out.append("線材本数")
    return out


def _build_create_columns(table_info: List[Tuple], ordered_names: List[str]) -> List[str]:
    """
    ES: Construye definiciones de columnas para CREATE TABLE preservando tipos/defaults.
    EN: Build CREATE TABLE column definitions preserving types/defaults.
    JP: 型/デフォルトを保持して CREATE TABLE の列定義を生成。
    """
    info_by_name = {row[1]: row for row in table_info}  # name -> (cid,name,type,notnull,dflt,pk)
    defs: List[str] = []
    for name in ordered_names:
        if name == "線材本数" and name not in info_by_name:
            defs.append(f'{_quote_ident("線材本数")} INTEGER DEFAULT 6')
            continue

        row = info_by_name[name]
        col_name = _quote_ident(row[1])
        col_type = (row[2] or "").strip()
        notnull = " NOT NULL" if row[3] else ""
        dflt = f" DEFAULT {row[4]}" if row[4] is not None else ""
        pk = " PRIMARY KEY" if row[5] else ""

        if row[1] == "id" and row[5]:
            # ES: Forzar patrón típico de SQLite para id autoincrement.
            # EN: Force the typical SQLite autoincrement id pattern.
            # JP: id の一般的な AUTOINCREMENT パターンを強制。
            defs.append(f'{_quote_ident("id")} INTEGER PRIMARY KEY AUTOINCREMENT')
        else:
            tail = (" " + col_type) if col_type else ""
            defs.append(f"{col_name}{tail}{notnull}{dflt}{pk}".strip())
    return defs


def _fix_sqlite_sequence(cur: sqlite3.Cursor, table: str) -> None:
    try:
        max_id = cur.execute(f"SELECT MAX(id) FROM {_quote_ident(table)};").fetchone()[0] or 0
        row = cur.execute("SELECT name FROM sqlite_sequence WHERE name=?", (table,)).fetchone()
        if row is None:
            cur.execute("INSERT INTO sqlite_sequence(name,seq) VALUES(?,?)", (table, max_id))
        else:
            cur.execute("UPDATE sqlite_sequence SET seq=? WHERE name=?", (max_id, table))
    except Exception:
        return


def _recreate_table_with_wire_count(
    con: sqlite3.Connection,
    table: str,
) -> str:
    cur = con.cursor()
    if not _table_exists(cur, table):
        return "NO_TABLE"

    table_info = _pragma_table_info(cur, table)
    names = [row[1] for row in table_info]
    if "線材長" not in names:
        return "SKIP_NO_WIRE_LEN"

    desired = _ensure_wire_count_after_wire_length(names)
    need_add = "線材本数" not in names

    # ES: Si ya está en la posición correcta y no hay que añadir, no tocar.
    # EN: If already in correct position and no need to add, do nothing.
    # JP: 既に正しい位置かつ追加不要なら何もしない。
    if (not need_add) and desired == names:
        return "OK_NO_CHANGE"

    tmp = f"{table}__tmp_reorder"
    con.execute("BEGIN")
    try:
        con.execute(f"DROP TABLE IF EXISTS {_quote_ident(tmp)};")
        defs = _build_create_columns(table_info, desired)
        con.execute(f"CREATE TABLE {_quote_ident(tmp)} ({', '.join(defs)});")

        insert_cols = ", ".join(_quote_ident(n) for n in desired)
        select_exprs: List[str] = []
        for n in desired:
            if n == "線材本数" and need_add:
                select_exprs.append('6 AS "線材本数"')
            else:
                select_exprs.append(_quote_ident(n))
        select_sql = ", ".join(select_exprs)
        con.execute(
            f"INSERT INTO {_quote_ident(tmp)} ({insert_cols}) "
            f"SELECT {select_sql} FROM {_quote_ident(table)};"
        )

        # ES: Swap
        # EN: Swap
        # JP: 入れ替え
        con.execute(f"DROP TABLE {_quote_ident(table)};")
        con.execute(f"ALTER TABLE {_quote_ident(tmp)} RENAME TO {_quote_ident(table)};")

        # ES: Rellenar NULLs en 線材本数 con 6.
        # EN: Backfill NULLs in 線材本数 with 6.
        # JP: 線材本数 のNULLを 6 で埋める。
        con.execute(
            f"UPDATE {_quote_ident(table)} SET {_quote_ident('線材本数')}=6 "
            f"WHERE {_quote_ident('線材本数')} IS NULL;"
        )
        _fix_sqlite_sequence(con.cursor(), table)
        con.execute("COMMIT")
        return "OK_RECREATED"
    except Exception:
        con.execute("ROLLBACK")
        raise


def main() -> int:
    # ES: Permite pasar una ruta explícita como argumento.
    # EN: Allow passing an explicit DB path as an argument.
    # JP: 引数でDBパスを明示指定できる。
    if len(sys.argv) >= 2 and sys.argv[1]:
        db_path = sys.argv[1]
    else:
        try:
            from app_paths import migrate_legacy_db_if_needed

            db_path = migrate_legacy_db_if_needed("results.db", shared=True)
        except Exception:
            db_path = r"C:\ProgramData\XebecTechnology\0.00sec\data\results.db"

    if not os.path.exists(db_path):
        print("DB_EXISTS=0")
        return 2

    backup_path = db_path + ".bak_" + time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(db_path, backup_path)
    print("BACKUP_CREATED=1")

    con = sqlite3.connect(db_path, timeout=60)
    try:
        try:
            con.execute("PRAGMA busy_timeout=60000;")
        except Exception:
            pass
        cur = con.cursor()
        results: Dict[str, str] = {}
        for t in ["main_results", "Results", "TemporaryResults"]:
            try:
                # ES: Reintentos por si el proceso principal tiene un lock temporal.
                # EN: Retry in case another process holds a temporary lock.
                # JP: 一時ロックの場合に備えてリトライ。
                last_exc: Exception | None = None
                for attempt in range(1, 11):
                    try:
                        results[t] = _recreate_table_with_wire_count(con, t)
                        last_exc = None
                        break
                    except sqlite3.OperationalError as e:
                        last_exc = e
                        if "locked" in str(e).lower():
                            _time.sleep(min(5.0, 0.5 * attempt))
                            continue
                        raise
                if last_exc is not None:
                    raise last_exc
            except Exception as e:
                # ES: Reportar error en ASCII para evitar problemas de encoding en consola.
                # EN: Report errors in ASCII to avoid console encoding issues.
                # JP: コンソール文字化け回避のためASCIIでエラーを出力。
                msg = f"{type(e).__name__}:{e}"
                try:
                    msg = msg.encode("ascii", "backslashreplace").decode("ascii")
                except Exception:
                    msg = type(e).__name__
                results[t] = "ERROR"
                print(f"ERR_{t}={msg}")
        # ES: Reporte (solo ASCII).
        # EN: Report (ASCII only).
        # JP: レポート（ASCIIのみ）。
        print("MIGRATION_DONE=1")
        for t, s in results.items():
            print(f"TABLE_{t}={s}")

        # ES: Verificación rápida.
        # EN: Quick verification.
        # JP: 簡易検証。
        for t in ["main_results", "Results", "TemporaryResults"]:
            if not _table_exists(cur, t):
                continue
            cols = [r[1] for r in _pragma_table_info(cur, t)]
            has = 1 if "線材本数" in cols else 0
            pos_len = cols.index("線材長") if "線材長" in cols else -1
            pos_cnt = cols.index("線材本数") if "線材本数" in cols else -1
            print(f"VERIFY_{t}_HAS={has}_POSLEN={pos_len}_POSCNT={pos_cnt}")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

