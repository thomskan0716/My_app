import sqlite3
import pandas as pd

class DBManager:
    def __init__(self, db_path='results.db', custom_conn=None):
        # custom_conn: permite reutilizar una conexión existente (p.ej. desde threads/workers)
        self.conn = custom_conn if custom_conn is not None else sqlite3.connect(db_path)
        self.create_tables()
        self._migrate_db_schema()

    def create_tables(self):
        with self.conn:
            # Tabla principal (la app usa main_results para consultas/filtros/export)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS main_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    実験日 INTEGER,
                    バリ除去 INTEGER,
                    上面ダレ量 REAL,
                    側面ダレ量 REAL,
                    摩耗量 REAL,
                    切削力X REAL,
                    切削力Y REAL,
                    切削力Z REAL,
                    面粗度前 TEXT,
                    面粗度後 TEXT,
                    A13 INTEGER,
                    A11 INTEGER,
                    A21 INTEGER,
                    A32 INTEGER,
                    直径 REAL,
                    材料 TEXT,
                    線材長 INTEGER,
                    回転速度 INTEGER,
                    送り速度 INTEGER,
                    UPカット INTEGER,
                    切込量 REAL,
                    突出量 INTEGER,
                    載せ率 REAL,
                    パス数 INTEGER,
                    加工時間 REAL
                );
            """)

            # Tabla General de Resultados
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS Results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    実験日 INTEGER,
                    バリ除去 INTEGER,
                    上面ダレ量 REAL,
                    側面ダレ量 REAL,
                    摩耗量 REAL,
                    切削力X REAL,
                    切削力Y REAL,
                    切削力Z REAL,
                    面粗度前 TEXT,
                    面粗度後 TEXT,
                    A13 INTEGER,
                    A11 INTEGER,
                    A21 INTEGER,
                    A32 INTEGER,
                    直径 REAL,
                    材料 TEXT,
                    回転速度 INTEGER,
                    送り速度 INTEGER,
                    UPカット INTEGER,
                    切込量 REAL,
                    突出量 INTEGER,
                    載せ率 REAL,
                    パス数 INTEGER,
                    線材長 INTEGER,
                    加工時間 REAL
                );
            """)

            # Tabla Temporal para Análisis (Solo los que pasaron)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS TemporaryResults (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    実験日 INTEGER,
                    バリ除去 INTEGER,
                    上面ダレ量 REAL,
                    側面ダレ量 REAL,
                    摩耗量 REAL,
                    切削力X REAL,
                    切削力Y REAL,
                    切削力Z REAL,
                    面粗度前 TEXT,
                    面粗度後 TEXT,
                    A13 INTEGER,
                    A11 INTEGER,
                    A21 INTEGER,
                    A32 INTEGER,
                    直径 REAL,
                    材料 TEXT,
                    回転速度 INTEGER,
                    送り速度 INTEGER,
                    UPカット INTEGER,
                    切込量 REAL,
                    突出量 INTEGER,
                    載せ率 REAL,
                    パス数 INTEGER,
                    線材長 INTEGER,
                    加工時間 REAL
                );
            """)

    def _migrate_db_schema(self):
        """Añade columnas nuevas a tablas existentes sin romper BDs antiguas."""
        try:
            targets = ["main_results", "Results", "TemporaryResults"]
            desired_cols = {
                "切削力X": "REAL",
                "切削力Y": "REAL",
                "切削力Z": "REAL",
            }

            for table in targets:
                try:
                    cur = self.conn.cursor()
                    cur.execute(f"PRAGMA table_info({table});")
                    existing = {row[1] for row in cur.fetchall()}  # row[1] = name
                    for col, col_type in desired_cols.items():
                        if col not in existing:
                            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
                except Exception:
                    # Tabla puede no existir en instalaciones antiguas; create_tables() la crea para main_results
                    continue
        except Exception:
            # Migración best-effort
            pass

    def insert_result(self, table, row):
        with self.conn:
            # Calcular 加工時間 usando la fórmula: 100/送り速度*60
            if '送り速度' in row and row['送り速度'] is not None and row['送り速度'] != 0:
                row['加工時間'] = (100 / row['送り速度']) * 60
            else:
                row['加工時間'] = None
            
            placeholders = ', '.join(['?'] * len(row))
            columns = ', '.join(row.keys())
            self.conn.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders});",
                list(row.values())
            )

    def clear_temporary(self):
        with self.conn:
            self.conn.execute("DELETE FROM TemporaryResults;")

    def fetch_all(self, table):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM {table};")
        return cursor.fetchall()
    
    def fetch_filtered(self, table, query, params=None):
        """Ejecutar consulta con filtros"""
        cursor = self.conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor.fetchall()
    
    def insert_results(self, df):
        """Insertar resultados en la tabla main_results"""
        if df.empty:
            print("⚠️ No hay datos para insertar.")
            return

        if "id" in df.columns:
            df = df.drop(columns=["id"])

        # 🔑 Columnas clave para identificar duplicados
        key_cols = [
            "実験日",
            "A13", "A11", "A21", "A32",
            "直径", "材料",
            "線材長",
            "回転速度", "送り速度", "UPカット", "切込量", "突出量", "載せ率", "パス数",
            "バリ除去",
            "上面ダレ量", "側面ダレ量", "摩耗量",
            "切削力X", "切削力Y", "切削力Z",
            "面粗度前", "面粗度後",
        ]

        for col in key_cols:
            if col not in df.columns:
                raise ValueError(f"❌ Falta la columna clave en el archivo: {col}")

        # 💾 Leer registros actuales desde la BBDD
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
            print("⚠️ Todos los registros ya existían en la base de datos.")
            return

        df_to_insert.to_sql("main_results", self.conn, if_exists="append", index=False)
        print(f"✅ {len(df_to_insert)} registros nuevos insertados.")
    
    @staticmethod
    def normalize_for_hash(df, key_cols):
        df_norm = df[key_cols].copy().fillna("")
        for col in key_cols:
            df_norm[col] = df_norm[col].apply(
                lambda x: f"{float(x):.5f}" if str(x).replace('.', '', 1).isdigit() else str(x).strip()
            )
        return df_norm
    
    def print_all_results(self):
        """Imprimir todos los registros de la tabla main_results"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM main_results")
        results = cursor.fetchall()
        print(f"📊 Total de registros en la base de datos: {len(results)}")
        if results:
            print("📋 Primeros 5 registros:")
            for i, row in enumerate(results[:5]):
                print(f"  Registro {i+1}: {row}")
        else:
            print("📋 No hay registros en la base de datos")
    
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
            '線材長': '線材長',  # Mantener el nombre original
            '実験日': '実験日',  # Mantener el nombre original
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
    
    def recreate_tables(self):
        """Recrear las tablas con el nuevo esquema"""
        with self.conn:
            # Eliminar tablas existentes
            self.conn.execute("DROP TABLE IF EXISTS Results;")
            self.conn.execute("DROP TABLE IF EXISTS TemporaryResults;")
            # Crear tablas con nuevo esquema
            self.create_tables()
    
    def get_table_info(self, table):
        """Obtener información de la estructura de la tabla"""
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table});")
        return cursor.fetchall()
