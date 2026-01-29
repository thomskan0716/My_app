import pandas as pd
import sqlite3
import os
import hashlib
import shutil
from datetime import datetime
from typing import Optional, Dict, Any

class DBManager:
    def __init__(self, db_path="results.db", custom_conn=None):
        if custom_conn is not None:
            self.conn = custom_conn
            # ES: Intentar deducir el path real del archivo DB desde la conexión (para backups)
            # EN: Try to infer the real DB file path from the connection (for backups)
            # JA: 接続からDBファイルの実パスを推定（バックアップ用）
            self.db_path = self._infer_db_path_from_conn(custom_conn) or db_path
        else:
            self.conn = sqlite3.connect(db_path)
            self.db_path = db_path
        self.create_tables()
        self._migrate_db_schema()

    @staticmethod
    def _infer_db_path_from_conn(conn) -> Optional[str]:
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA database_list;")
            rows = cur.fetchall()
            # (seq, name, file)
            for _, name, file_path in rows:
                if name == "main" and file_path:
                    return str(file_path)
        except Exception:
            return None
        return None

    @staticmethod
    def map_column_names(df):
        column_mapping = {
            '上面ダレ': '上面ダレ量',
            '上面ダレ量': '上面ダレ量',
            '側面ダレ': '側面ダレ量',
            '側面ダレ量': '側面ダレ量',
            '回転方向': 'UPカット',
            'UPカット': 'UPカット',
            '切込量': '切込量',
            '切込み量': '切込量',
            '面粗度(Ra)前': '面粗度前',
            '粗度(Ra)前': '面粗度前',
            '面粗度前': '面粗度前',
            '面粗度(Ra)後': '面粗度後',
            '粗度(Ra)後': '面粗度後',
            '面粗度後': '面粗度後',
            '突出量': '突出量',
            '突出し量': '突出量',
            '載せ率': '載せ率',
            '線材長': '線材長',  # Keep original name
            '実験日': '実験日',  # Keep original name
            '摩耗量': '摩耗量',
            '回転速度': '回転速度',
            '送り速度': '送り速度',
            'パス数': 'パス数',
            '切削力X': '切削力X',
            '切削力Y': '切削力Y',
            '切削力Z': '切削力Z',
            # '加工時間': '加工時間(s/100mm)'  # No se importa, se calcula automáticamente
        }
        return df.rename(columns=column_mapping)

    def create_tables(self):
        query = """
        CREATE TABLE IF NOT EXISTS main_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            実験日 TEXT,
            バリ除去 INTEGER,
            上面ダレ量 REAL,
            側面ダレ量 REAL,
            摩耗量 REAL,
            面粗度前 REAL,
            面粗度後 REAL,
            A13 INTEGER,
            A11 INTEGER,
            A21 INTEGER,
            A32 INTEGER,
            直径 REAL,
            材料 TEXT,
            線材長 INTEGER,
            線材本数 INTEGER DEFAULT 6,
            回転速度 INTEGER,
            送り速度 INTEGER,
            UPカット INTEGER,
            切込量 REAL,
            突出量 INTEGER,
            載せ率 REAL,
            パス数 INTEGER,
            加工時間 REAL
        );
        """
        self.conn.execute(query)
        self.conn.commit()

    def _migrate_db_schema(self):
        """ES: Añade columnas nuevas a tablas existentes sin romper BDs antiguas.
        EN: Add new columns to existing tables without breaking old DBs.
        JA: 既存テーブルに新列を追加（古いDBを壊さない）。"""
        try:
            table = "main_results"
            desired_cols = {
                "線材本数": "INTEGER DEFAULT 6",
            }
            cur = self.conn.cursor()
            cur.execute(f"PRAGMA table_info({table});")
            existing = {row[1] for row in cur.fetchall()}  # row[1] = name
            for col, col_type in desired_cols.items():
                if col not in existing:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            try:
                self.conn.execute("UPDATE main_results SET 線材本数 = 6 WHERE 線材本数 IS NULL")
                self.conn.commit()
            except Exception:
                pass
        except Exception:
            # Migración best-effort
            pass

    @staticmethod
    def normalize_for_hash(df, key_cols):
        df_norm = df[key_cols].copy().fillna("")
        for col in key_cols:
            df_norm[col] = df_norm[col].apply(
                lambda x: f"{float(x):.5f}" if str(x).replace('.', '', 1).isdigit() else str(x).strip()
            )
        return df_norm

    def insert_results(self, df):
        if df.empty:
            print("⚠️ 挿入するデータがありません。")
            return

        if "id" in df.columns:
            df = df.drop(columns=["id"])

        # 🔑 Columnas clave para identificar duplicados
        key_cols = [
            "実験日",
            "A13", "A11", "A21", "A32",
            "直径", "材料",
            "線材長",
            "線材本数",
            "回転速度", "送り速度", "UPカット", "切込量", "突出量", "載せ率", "パス数",
            "バリ除去",
            "上面ダレ量", "側面ダレ量", "摩耗量",
            "切削力X", "切削力Y", "切削力Z",
            "面粗度前", "面粗度後",
        ]

        for col in key_cols:
            if col not in df.columns:
                raise ValueError(f"❌ ファイルにキー列がありません: {col}")

        # ES: 💾 Leer registros actuales desde la BBDD
        # EN: 💾 Read current records from the database
        # JP: 💾 現在のレコードをDBから読み込む
        db_df = pd.read_sql_query(f"SELECT {', '.join(key_cols)} FROM main_results", self.conn)

        df_cmp_norm = DBManager.normalize_for_hash(df, key_cols)
        db_cmp_norm = DBManager.normalize_for_hash(db_df, key_cols)

        df_cmp_norm_hashes = df_cmp_norm.apply(lambda row: "||".join(row.values.astype(str)), axis=1)
        db_cmp_norm_hashes = db_cmp_norm.apply(lambda row: "||".join(row.values.astype(str)), axis=1)

        df["__hash"] = df_cmp_norm_hashes
        db_hashes = set(db_cmp_norm_hashes)

        df_to_insert = df[~df["__hash"].isin(db_hashes)].drop(columns=["__hash"])

        # ✅ AHORA VA ESTO:
        if df_to_insert.empty:
            print("⚠️ すべてのレコードはすでにDBに存在します。")
            return

        df_to_insert.to_sql("main_results", self.conn, if_exists="append", index=False)
        print(f"✅ 新規レコードを {len(df_to_insert)} 件挿入しました。")

    def _create_db_backup(self) -> Optional[str]:
        """
        Crear backup del archivo SQLite antes de sobrescribir registros.
        Devuelve el path del backup si se pudo crear.
        """
        try:
            db_path = self.db_path
            if not db_path:
                db_path = self._infer_db_path_from_conn(self.conn)
            if not db_path or not os.path.exists(db_path):
                return None
            db_dir = os.path.dirname(db_path) or "."
            backup_dir = os.path.join(db_dir, "backup")
            os.makedirs(backup_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"results_backup_{ts}.db")
            shutil.copy2(db_path, backup_path)
            return backup_path
        except Exception:
            return None

    def upsert_results(self, df: pd.DataFrame, debug: bool = False) -> Dict[str, Any]:
        """
        Upsert:
        - Si existe una fila con la misma clave (condiciones), se actualiza (sobrescribe) el resto de campos.
        - Si no existe, se inserta.

        Devuelve: {'inserted': int, 'updated': int, 'db_backup_path': str|None}
        """
        if df.empty:
            return {"inserted": 0, "updated": 0, "db_backup_path": None}

        if "id" in df.columns:
            df = df.drop(columns=["id"])

        # Clave de comparación (condiciones + meta necesaria)
        key_cols = [
            "実験日",
            "A13", "A11", "A21", "A32",
            "直径", "材料",
            "線材長",
            "線材本数",
            "回転速度", "送り速度", "UPカット", "切込量", "突出量", "載せ率", "パス数",
        ]

        for col in key_cols:
            if col not in df.columns:
                raise ValueError(f"❌ upsert 用のキー列がありません: {col}")

        # ES: Columnas a actualizar (todas menos la clave)
        # EN: Columns to update (everything except the key)
        # JA: 更新対象列（キー列以外すべて）
        # ES: Importante:
        # EN: Notes:
        # JA: 注意:
        # ES: - Ignorar columnas que no existan en la tabla real
        # EN: - Ignore columns that do not exist in the actual table
        # JA: - 実テーブルに存在しない列は無視
        # ES: - Ignorar columnas totalmente vacías (suelen venir de defaults cuando el archivo no las trae)
        # EN: - Ignore columns that are entirely empty (often defaults when the file doesn't include them)
        # JA: - 全て空の列は無視（ファイル未提供時のデフォルト由来が多い）
        update_cols_raw = [c for c in df.columns if c not in key_cols]
        update_cols_raw = [c for c in update_cols_raw if not df[c].isna().all()]

        # Columnas reales en la tabla (evita comparar/actualizar campos no existentes)
        cur = self.conn.cursor()
        cur.execute("PRAGMA table_info(main_results)")
        existing_cols = {row[1] for row in cur.fetchall()}  # name
        update_cols = [c for c in update_cols_raw if c in existing_cols]

        # ES: Leer ids existentes + clave + columnas a comparar (mínimas)
        # EN: Read existing ids + key + minimal columns to compare
        # JP: 既存ID + キー + 比較用の最小列を読み込む
        db_cols = ["id"] + key_cols + update_cols
        db_df = pd.read_sql_query(f"SELECT {', '.join(db_cols)} FROM main_results", self.conn)

        df_key_norm = DBManager.normalize_for_hash(df, key_cols)
        db_key_norm = DBManager.normalize_for_hash(db_df, key_cols) if not db_df.empty else db_df

        df_keys = df_key_norm.apply(lambda r: "||".join(r.values.astype(str)), axis=1).tolist()
        db_map = {}
        if not db_df.empty:
            db_keys = db_key_norm.apply(lambda r: "||".join(r.values.astype(str)), axis=1).tolist()
            for k, row_id in zip(db_keys, db_df["id"].tolist()):
                # ES: Si hay duplicados previos, nos quedamos con el primero
                # EN: If there are existing duplicates, keep the first one
                # JA: 既存の重複がある場合は最初のものを採用
                if k not in db_map:
                    db_map[k] = row_id

        # Preparar updates/inserts
        to_update = []
        to_insert_rows = []
        updated_count = 0
        inserted_count = 0

        def _norm_val(v: Any) -> str:
            # ES: Normalización robusta para comparar "igualdad" (evita falsos positivos por formato)
            # EN: Robust normalization for equality checks (avoids format-based false positives)
            # JA: 等価比較用の堅牢な正規化（書式差による誤判定を防ぐ）
            try:
                if pd.isna(v):
                    return ""
            except Exception:
                pass
            if v is None:
                return ""
            # ES: Numéricos: fijar precisión
            # EN: Numerics: fix precision
            # JA: 数値：精度を固定
            try:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return f"{float(v):.6f}"
            except Exception:
                pass
            # ES: Strings numéricos: intentar convertir
            # EN: Numeric strings: try to convert
            # JA: 数値文字列：変換を試行
            try:
                s = str(v).strip()
                if s == "":
                    return ""
                n = pd.to_numeric(s, errors="coerce")
                if pd.notna(n):
                    return f"{float(n):.6f}"
                return s
            except Exception:
                return str(v).strip()

        # Para decidir si realmente se sobrescribe algo, comparamos update_cols (si existen en DB)
        db_update_lookup = {}
        if not db_df.empty and update_cols:
            # index por id
            db_update_lookup = db_df.set_index("id")[update_cols].to_dict(orient="index")

        def _key_brief_from_row(row: pd.Series) -> str:
            # Resumen corto para logs
            parts = []
            for c in ["実験日", "回転速度", "送り速度", "UPカット", "切込量", "突出量", "載せ率", "パス数", "線材長"]:
                if c in row.index:
                    parts.append(f"{c}={row.get(c)}")
            # Brush one-hot
            for c in ["A13", "A11", "A21", "A32"]:
                if c in row.index:
                    parts.append(f"{c}={row.get(c)}")
            return ", ".join(parts)

        if debug:
            try:
                print(
                    f"🧾 UPSERT DEBUG: filas_entrada={len(df)} | cols_update={len(update_cols)} | cols_update_list={update_cols}",
                    flush=True,
                )
            except Exception:
                pass

        for i, k in enumerate(df_keys):
            if k in db_map:
                row_id = db_map[k]
                # Determinar si cambia algo
                will_change = False
                diffs = []
                if update_cols:
                    old = db_update_lookup.get(row_id, {})
                    for c in update_cols:
                        new_val = df.iloc[i][c]
                        old_val = old.get(c)
                        # ES: Si el archivo no trae valor (NaN/None/""), no lo usamos para decidir ni para sobrescribir
                        # EN: If the file does not provide a value (NaN/None/\"\"), do not use it to decide or overwrite
                        # JP: ファイル側に値が無い場合（NaN/None/\"\"）、判断にも上書きにも使わない
                        if _norm_val(new_val) == "":
                            continue
                        if _norm_val(new_val) != _norm_val(old_val):
                            will_change = True
                            if debug:
                                diffs.append((c, old_val, new_val, _norm_val(old_val), _norm_val(new_val)))
                            else:
                                # Si no estamos en modo debug, con el primer cambio basta
                                break
                else:
                    will_change = False

                if will_change:
                    updated_count += 1
                    params = [df.iloc[i][c] for c in update_cols] + [row_id]
                    to_update.append(params)
                    if debug:
                        try:
                            brief = _key_brief_from_row(df.iloc[i])
                            if diffs:
                                diff_str = " | ".join(
                                    [
                                        f"{c}: {old_norm} -> {new_norm} (raw {old_raw!r} -> {new_raw!r})"
                                        for (c, old_raw, new_raw, old_norm, new_norm) in diffs
                                    ]
                                )
                                print(f"🟥 UPSERT UPDATE id={row_id} | {brief} | diffs={diff_str}", flush=True)
                            else:
                                print(f"🟥 UPSERT UPDATE id={row_id} | {brief}", flush=True)
                        except Exception:
                            pass
                else:
                    if debug:
                        try:
                            brief = _key_brief_from_row(df.iloc[i])
                            print(f"🟦 UPSERT SKIP（同一） id={row_id} | {brief}", flush=True)
                        except Exception:
                            pass
            else:
                to_insert_rows.append(df.iloc[i])
                inserted_count += 1
                if debug:
                    try:
                        brief = _key_brief_from_row(df.iloc[i])
                        print(f"🟩 UPSERT INSERT | {brief}", flush=True)
                    except Exception:
                        pass

        db_backup_path = None
        if updated_count > 0:
                # ES: ✅ Crear backup UNA SOLA VEZ antes de sobrescribir
                # EN: ✅ Create a backup ONCE before overwriting
                # JP: ✅ 上書き前にバックアップを一度だけ作成する
            db_backup_path = self._create_db_backup()
            if db_backup_path:
                print(f"📋 Backup de BBDD creado: {db_backup_path}", flush=True)
            else:
                print("⚠️ 自動DBバックアップを作成できませんでした（パスが利用できません）。", flush=True)

        # Ejecutar updates
        if to_update and update_cols:
            # COALESCE: si llega NULL, conserva el valor existente (no sobrescribe con vacío)
            set_clause = ", ".join([f"{c} = COALESCE(?, {c})" for c in update_cols])
            sql = f"UPDATE main_results SET {set_clause} WHERE id = ?"
            cur = self.conn.cursor()
            cur.executemany(sql, to_update)
            self.conn.commit()

        # Ejecutar inserts
        if to_insert_rows:
            df_to_insert = pd.DataFrame(to_insert_rows)
            df_to_insert.to_sql("main_results", self.conn, if_exists="append", index=False)

        return {"inserted": inserted_count, "updated": updated_count, "db_backup_path": db_backup_path}

    def fetch_all(self, table):
        """Obtener todos los registros de una tabla"""
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM {table};")
        return cursor.fetchall()
    
    def print_all_results(self):
        """Imprimir todos los registros de la tabla main_results"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM main_results")
        results = cursor.fetchall()
        print(f"📊 DBの総レコード数: {len(results)}")
        if results:
            print("📋 Primeros 5 registros:")
            for i, row in enumerate(results[:5]):
                print(f"  Registro {i+1}: {row}")
        else:
            print("📋 DBにレコードがありません")

        df["__hash"] = df_cmp_norm_hashes
        db_hashes = set(db_cmp_norm_hashes)

        df_to_insert = df[~df["__hash"].isin(db_hashes)].drop(columns=["__hash"])

        # ✅ AHORA VA ESTO:
        if df_to_insert.empty:
            print("⚠️ すべてのレコードはすでにDBに存在します。")
            return

        df_to_insert.to_sql("main_results", self.conn, if_exists="append", index=False)
        print(f"✅ {len(df_to_insert)} registros nuevos insertados.")

    def close(self):
        self.conn.close()

    def print_all_results(self):
        query = "SELECT * FROM main_results"
        df = pd.read_sql_query(query, self.conn)

        if df.empty:
            print("⚠️ データベースが空です。")
        else:
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", None)
            pd.set_option("display.width", None)
            pd.set_option("display.colheader_justify", "left")
            print("📊 DBの全内容:\n")
            print(df)


class ResultProcessor:
    def __init__(self, db_manager):
        self.db = db_manager

    def _read_any_table(self, file_path: str) -> pd.DataFrame:
        ext = os.path.splitext(str(file_path))[1].lower()
        if ext == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8-sig")
        else:
            df = pd.read_excel(file_path, header=0)

        # Normalizar nombres de columnas (evita fallos por espacios invisibles)
        try:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [" ".join([str(x).strip() for x in tup if str(x).strip() != ""]).strip() for tup in df.columns]
            else:
                df.columns = [str(c).strip() for c in df.columns]
        except Exception:
            pass
        return df

    def process_results_file(self, file_path, selected_brush, senzai_length):
        df = self._read_any_table(file_path)
        df = DBManager.map_column_names(df)
        
        # ES: Eliminar 加工時間 si está presente (se calcula automáticamente)
        # EN: Drop 加工時間 if present (it is computed automatically)
        # JA: 加工時間 があれば削除（自動計算するため）
        if '加工時間' in df.columns:
            df = df.drop(columns=['加工時間'])
        if '加工時間(s/100mm)' in df.columns:
            df = df.drop(columns=['加工時間(s/100mm)'])

        columns_required = ['回転速度', '送り速度', 'UPカット', '切込量', '突出量', '載せ率', 'パス数',
                            '線材長', '上面ダレ量', '側面ダレ量', '摩耗量', '面粗度前', '面粗度後', '実験日']

        missing_columns = [col for col in columns_required if col not in df.columns]
        if missing_columns:
            raise ValueError(f"❌ El archivo de resultados no contiene las siguientes columnas necesarias: {', '.join(missing_columns)}")

        df_filtered = df[columns_required].copy()

        # ES: Calcular バリ除去 basado en 上面ダレ量
        # EN: Compute バリ除去 based on 上面ダレ量
        # JA: 上面ダレ量 に基づき バリ除去 を算出
        df_filtered['バリ除去'] = df_filtered['上面ダレ量'].apply(lambda x: 1 if x > 0 else 0)

        # ES: Brush: SIEMPRE desde el archivo (one-hot A13/A11/A21/A32). No usar UI.
        # EN: Brush: ALWAYS from the file (one-hot A13/A11/A21/A32). Do not use UI.
        # JA: ブラシ：必ずファイルから（A13/A11/A21/A32のone-hot）。UIは使わない。
        brush_cols = ["A13", "A11", "A21", "A32"]
        missing_brush = [c for c in brush_cols if c not in df.columns]
        if missing_brush:
            raise ValueError(
                "❌ El archivo de resultados debe incluir columnas de cepillo one-hot: "
                f"{', '.join(brush_cols)} (faltan: {', '.join(missing_brush)})"
            )

        for c in brush_cols:
            df_filtered[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

        # ES: Validación básica: exactamente 1 cepillo activo por fila
        # EN: Basic validation: exactly one active brush per row
        # JA: 基本検証：各行で有効ブラシは1つのみ
        try:
            s = df_filtered[brush_cols].sum(axis=1)
            bad = df_filtered[(s != 1)]
            if not bad.empty:
                raise ValueError(
                    "❌ Formato de cepillo inválido: cada fila debe tener exactamente un 1 "
                    f"en {brush_cols}. Filas inválidas: {bad.index.tolist()[:10]}"
                )
        except ValueError:
            raise
        except Exception:
            # ES: Si falla la validación por algún motivo, no bloquear el import
            # EN: If validation fails for any reason, do not block the import
            # JA: 検証が失敗してもインポートをブロックしない（安全側）
            pass
        
        # ES: Calcular 加工時間 usando la fórmula: 100/送り速度*60
        # EN: Compute 加工時間 using the formula: 100/送り速度*60
        # JA: 加工時間 を計算（式：100/送り速度*60）
        df_filtered['加工時間'] = (100 / df_filtered['送り速度']) * 60
        
        # ES: Asignar valores por defecto para campos que pueden no estar en el archivo
        # EN: Assign default values for fields that may be missing from the file
        # JA: ファイルにない可能性のある項目にデフォルト値を設定
        if '直径' in df.columns:
            df_filtered['直径'] = df['直径']
        else:
            df_filtered['直径'] = 0.15  # Default value
        if '材料' in df.columns:
            df_filtered['材料'] = df['材料']
        else:
            df_filtered['材料'] = 'Steel'  # Default value

        # ES: Cutting forces opcionales:
        # EN: Optional cutting forces:
        # JA: 切削力（任意）
        # ES: Si no vienen en el archivo, NO crear la columna (así no se usa para comparar/actualizar)
        # EN: If they are not present in the file, do NOT create the column (so it won't be used for compare/update)
        # JA: ファイルに無ければ列を作らない（比較/更新に使わないため）
        for c in ["切削力X", "切削力Y", "切削力Z"]:
            if c in df.columns:
                df_filtered[c] = pd.to_numeric(df[c], errors="coerce")

        df_filtered = DBManager.map_column_names(df_filtered)

        # ES: Upsert: sobreescribe si ya existe la misma clave (condiciones)
        # EN: Upsert: overwrite when the same key (conditions) already exists
        # JA: アップサート：同一キー（条件）があれば上書き
        res = self.db.upsert_results(df_filtered, debug=True)
        print(f"✅ Upsert completado. insertados={res['inserted']} actualizados={res['updated']}")
        print("✅ 処理と挿入が完了しました。")
        self.db.print_all_results()

    def process_results_file_with_ui_values(self, file_path, selected_brush, diameter, material, wire_count, custom_conn=None):
        """ES: Procesar archivo de resultados importando columnas específicas y usando valores de UI
        EN: Process a results file importing specific columns and using UI values
        JA: 結果ファイルを処理（特定列を取り込み、UI値を使用）
        """
        # ES: Leer todas las columnas del archivo para asegurar que 実験日 esté incluido
        # EN: Read all columns to ensure 実験日 is included
        # JA: 実験日 を確実に含めるため全列を読み込む
        df = self._read_any_table(file_path)
        
        # ES: Mapear nombres de columnas
        # EN: Map column names
        # JA: 列名をマッピング
        df = DBManager.map_column_names(df)
        
        # ES: Eliminar 加工時間 si está presente (se calcula automáticamente)
        # EN: Drop 加工時間 if present (it is computed automatically)
        # JA: 加工時間 があれば削除（自動計算するため）
        if '加工時間' in df.columns:
            df = df.drop(columns=['加工時間'])
        if '加工時間(s/100mm)' in df.columns:
            df = df.drop(columns=['加工時間(s/100mm)'])
        
        # Columnas requeridas (después del mapeo)
        columns_required = ['回転速度', '送り速度', 'UPカット', '切込量', '突出量', '載せ率', 'パス数',
                            '線材長', '上面ダレ量', '側面ダレ量', '摩耗量', '面粗度前', '面粗度後', '実験日']
        
        # ES: Verificar columnas faltantes
        # EN: Check for missing columns
        # JA: 不足列をチェック
        missing_columns = [col for col in columns_required if col not in df.columns]
        if missing_columns:
            raise ValueError(f"❌ El archivo de resultados no contiene las siguientes columnas necesarias: {', '.join(missing_columns)}")
        
        # ES: Filtrar solo las columnas requeridas
        # EN: Keep only the required columns
        # JA: 必須列のみ抽出
        df_filtered = df[columns_required].copy()
        
        # ES: Calcular バリ除去 basado en 上面ダレ量
        # EN: Compute バリ除去 based on 上面ダレ量
        # JA: 上面ダレ量 に基づき バリ除去 を算出
        df_filtered['バリ除去'] = df_filtered['上面ダレ量'].apply(lambda x: 1 if x > 0 else 0)
        
        # ES: Brush: SIEMPRE desde el archivo (one-hot A13/A11/A21/A32). No usar UI.
        # EN: Brush: ALWAYS from the file (one-hot A13/A11/A21/A32). Do not use UI.
        # JA: ブラシ：必ずファイルから（A13/A11/A21/A32のone-hot）。UIは使わない。
        brush_cols = ["A13", "A11", "A21", "A32"]
        missing_brush = [c for c in brush_cols if c not in df.columns]
        if missing_brush:
            raise ValueError(
                "❌ El archivo de resultados debe incluir columnas de cepillo one-hot: "
                f"{', '.join(brush_cols)} (faltan: {', '.join(missing_brush)})"
            )

        for c in brush_cols:
            df_filtered[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

        # ES: Validación básica: exactamente 1 cepillo activo por fila
        # EN: Basic validation: exactly one active brush per row
        # JA: 基本検証：各行で有効ブラシは1つのみ
        try:
            s = df_filtered[brush_cols].sum(axis=1)
            bad = df_filtered[(s != 1)]
            if not bad.empty:
                raise ValueError(
                    "❌ Formato de cepillo inválido: cada fila debe tener exactamente un 1 "
                    f"en {brush_cols}. Filas inválidas: {bad.index.tolist()[:10]}"
                )
        except ValueError:
            raise
        except Exception:
            # ES: Si falla la validación por algún motivo, no bloquear el import
            # EN: If validation fails for any reason, do not block the import
            # JA: 検証が失敗してもインポートをブロックしない（安全側）
            pass
        
        # ES: 直径/材料: usar archivo si existe, si no UI
        # EN: 直径/材料: use file values if present, otherwise UI
        # JA: 直径/材料：ファイルにあれば使用、なければUI
        df_filtered['直径'] = df['直径'] if '直径' in df.columns else diameter
        df_filtered['材料'] = df['材料'] if '材料' in df.columns else material

        # ES: 線材本数: SIEMPRE desde UI (ignorar archivo si existe)
        # EN: 線材本数: ALWAYS from UI (ignore file even if present)
        # JA: 線材本数：常にUI値（ファイルにあっても無視）
        df_filtered['線材本数'] = int(wire_count)

        # ES: Cutting forces opcionales:
        # EN: Optional cutting forces:
        # JA: 切削力（任意）
        # ES: Si no vienen en el archivo, NO crear la columna (así no se usa para comparar/actualizar)
        # EN: If they are not present in the file, do NOT create the column (so it won't be used for compare/update)
        # JA: ファイルに無ければ列を作らない（比較/更新に使わないため）
        for c in ["切削力X", "切削力Y", "切削力Z"]:
            if c in df.columns:
                df_filtered[c] = pd.to_numeric(df[c], errors="coerce")
        
        # ES: Calcular 加工時間 usando la fórmula: 100/送り速度*60
        # EN: Compute 加工時間 using the formula: 100/送り速度*60
        # JA: 加工時間 を計算（式：100/送り速度*60）
        df_filtered['加工時間'] = (100 / df_filtered['送り速度']) * 60
        
        # ES: Mapear nombres de columnas para la base de datos
        # EN: Map column names for the database
        # JA: DB用に列名をマッピング
        df_filtered = DBManager.map_column_names(df_filtered)
        
        # ES: Usar conexión personalizada si se proporciona, sino usar la del db manager
        # EN: Use custom connection if provided; otherwise use the DB manager connection
        # JA: カスタム接続があれば使用、なければDBマネージャの接続を使用
        if custom_conn is not None:
            # ES: Crear un DBManager temporal con la conexión personalizada
            # EN: Create a temporary DBManager with the custom connection
            # JA: カスタム接続で一時DBManagerを作成
            temp_db = DBManager(custom_conn=custom_conn)
            res = temp_db.upsert_results(df_filtered, debug=True)
            print(f"✅ Upsert completado (conn personalizada). insertados={res['inserted']} actualizados={res['updated']}")
            print("✅ UI値（カスタム接続）での処理と挿入が完了しました。")
            return res
        else:
            res = self.db.upsert_results(df_filtered, debug=True)
            print(f"✅ Upsert completado. insertados={res['inserted']} actualizados={res['updated']}")
            print("✅ Procesamiento completado con valores de UI.")
            self.db.print_all_results()
            return res

    def print_all_results(self):
        """Imprimir todos los registros de la tabla main_results"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM main_results")
        results = cursor.fetchall()
        print(f"📊 DBの総レコード数: {len(results)}")
        if results:
            print("📋 Primeros 5 registros:")
            for i, row in enumerate(results[:5]):
                print(f"  Registro {i+1}: {row}")
        else:
            print("📋 DBにレコードがありません")