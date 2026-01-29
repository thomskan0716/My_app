import sqlite3

def check_and_update_db():
    """ES: Verificar y actualizar los nombres de columnas en la base de datos
    EN: Check and update column names in the database
    JA: DBの列名を確認・更新
    """
    
    # ES: Conectar a la base de datos | EN: Connect to the database | JA: DBに接続
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()
    
    # ES: Verificar si la tabla main_results existe | EN: Check whether main_results exists | JA: main_results テーブルの存在確認
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='main_results';")
    if not cursor.fetchone():
        print("❌ テーブル 'main_results' が存在しません")
        conn.close()
        return
    
    # ES: Obtener información de las columnas | EN: Get column info | JA: 列情報を取得
    cursor.execute("PRAGMA table_info(main_results);")
    columns = cursor.fetchall()
    
    print("📋 main_results の現在の列:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    # ES: Verificar si existen las columnas antiguas | EN: Check legacy columns | JA: 旧列の存在確認
    old_compression_exists = any(col[1] == '絞せ率' for col in columns)
    new_compression_exists = any(col[1] == '載せ率' for col in columns)
    old_surface_exists = any(col[1] == '上面気し量' for col in columns)
    new_surface_exists = any(col[1] == '上面ダレ量' for col in columns)
    
    needs_update = (old_compression_exists and not new_compression_exists) or (old_surface_exists and not new_surface_exists)
    
    if needs_update:
        print("\n🔄 列名を更新中:")
        if old_compression_exists and not new_compression_exists:
            print("  - 絞せ率 → 載せ率")
        if old_surface_exists and not new_surface_exists:
            print("  - 上面気し量 → 上面ダレ量")
        
        try:
            # ES: Eliminar tabla temporal si existe | EN: Drop temp table if it exists | JA: 一時テーブルがあれば削除
            cursor.execute("DROP TABLE IF EXISTS main_results_new;")
            
            # ES: Crear tabla temporal con el nuevo esquema
            # EN: Create temp table with the new schema
            # JA: 新スキーマで一時テーブルを作成
            cursor.execute("""
                CREATE TABLE main_results_new (
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
            
            # ES: Construir la consulta de inserción dinámicamente
            # EN: Build the INSERT query dynamically
            # JA: INSERTクエリを動的に構築
            select_columns = []
            for col in columns:
                col_name = col[1]
                if col_name == '絞せ率':
                    select_columns.append('絞せ率 AS 載せ率')
                elif col_name == '上面気し量':
                    select_columns.append('上面気し量 AS 上面ダレ量')
                else:
                    select_columns.append(col_name)
            
            select_query = f"SELECT {', '.join(select_columns)} FROM main_results"
            print(f"🔧 移行クエリ: {select_query}")
            
            # ES: Copiar datos de la tabla antigua a la nueva
            # EN: Copy data from the old table to the new one
            # JA: 旧テーブルから新テーブルへデータコピー
            cursor.execute(f"INSERT INTO main_results_new {select_query}")
            
            # ES: Eliminar tabla antigua y renombrar la nueva
            # EN: Drop old table and rename the new one
            # JA: 旧テーブル削除→新テーブルをリネーム
            cursor.execute("DROP TABLE main_results;")
            cursor.execute("ALTER TABLE main_results_new RENAME TO main_results;")
            
            conn.commit()
            print("✅ データベースを更新しました")
            
        except Exception as e:
            print(f"❌ データベース更新中にエラー: {e}")
            conn.rollback()
    
    elif new_compression_exists and new_surface_exists:
        print("✅ 列はすでに最新です")
        print("📋 main_results の現在の列:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
    
    else:
        print("⚠️ 期待する列が見つかりませんでした")
        print("📋 main_results の現在の列:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
    
    conn.close()

if __name__ == "__main__":
    check_and_update_db() 