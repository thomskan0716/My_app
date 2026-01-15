import sqlite3

def check_and_update_db():
    """Verificar y actualizar los nombres de columnas en la base de datos"""
    
    # Conectar a la base de datos
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()
    
    # Verificar si la tabla main_results existe
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='main_results';")
    if not cursor.fetchone():
        print("❌ La tabla 'main_results' no existe")
        conn.close()
        return
    
    # Obtener información de las columnas
    cursor.execute("PRAGMA table_info(main_results);")
    columns = cursor.fetchall()
    
    print("📋 Columnas actuales en main_results:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    # Verificar si existen las columnas antiguas
    old_compression_exists = any(col[1] == '絞せ率' for col in columns)
    new_compression_exists = any(col[1] == '載せ率' for col in columns)
    old_surface_exists = any(col[1] == '上面気し量' for col in columns)
    new_surface_exists = any(col[1] == '上面ダレ量' for col in columns)
    
    needs_update = (old_compression_exists and not new_compression_exists) or (old_surface_exists and not new_surface_exists)
    
    if needs_update:
        print("\n🔄 Actualizando nombres de columnas:")
        if old_compression_exists and not new_compression_exists:
            print("  - 絞せ率 → 載せ率")
        if old_surface_exists and not new_surface_exists:
            print("  - 上面気し量 → 上面ダレ量")
        
        try:
            # Eliminar tabla temporal si existe
            cursor.execute("DROP TABLE IF EXISTS main_results_new;")
            
            # Crear tabla temporal con el nuevo esquema
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
            
            # Construir la consulta de inserción dinámicamente
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
            print(f"🔧 Query de migración: {select_query}")
            
            # Copiar datos de la tabla antigua a la nueva
            cursor.execute(f"INSERT INTO main_results_new {select_query}")
            
            # Eliminar tabla antigua y renombrar la nueva
            cursor.execute("DROP TABLE main_results;")
            cursor.execute("ALTER TABLE main_results_new RENAME TO main_results;")
            
            conn.commit()
            print("✅ Base de datos actualizada correctamente")
            
        except Exception as e:
            print(f"❌ Error actualizando la base de datos: {e}")
            conn.rollback()
    
    elif new_compression_exists and new_surface_exists:
        print("✅ Las columnas ya están actualizadas en la base de datos")
        print("📋 Columnas actuales en main_results:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
    
    else:
        print("⚠️ No se encontraron las columnas esperadas")
        print("📋 Columnas actuales en main_results:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
    
    conn.close()

if __name__ == "__main__":
    check_and_update_db() 