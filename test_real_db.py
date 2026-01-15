#!/usr/bin/env python
# coding: utf-8

"""
Script para probar la función de análisis lineal con la base de datos real
"""

import os
import sys
from db_manager import DBManager
from linear_analysis_advanced import run_advanced_linear_analysis_from_db

def test_real_database():
    """Probar la función con la base de datos real"""
    
    print("🔧 Probando análisis lineal con base de datos real...")
    
    try:
        # Crear instancia de DBManager
        db = DBManager()
        print("✅ DBManager creado correctamente")
        
        # Verificar tablas y datos
        cursor = db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📊 Tablas disponibles: {[t[0] for t in tables]}")
        
        # Verificar datos en cada tabla
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            print(f"📈 {table_name}: {count} registros")
        
        # Crear filtros de prueba
        test_filters = {
            'A32': True,  # Solo cepillo A32
            '送り速度': ('20', '40'),  # Rango de velocidad de alimentación
            '切込量': ('0.8', '1.5')  # Rango de profundidad de corte
        }
        
        print(f"🔧 Filtros de prueba: {test_filters}")
        
        # Crear carpeta de salida
        output_folder = "test_real_analysis"
        os.makedirs(output_folder, exist_ok=True)
        
        # Ejecutar análisis
        print("🚀 Ejecutando análisis lineal...")
        results = run_advanced_linear_analysis_from_db(db, test_filters, output_folder)
        
        # Mostrar resultados
        print(f"✅ Resultados: {results.get('success', False)}")
        
        if results.get('success', False):
            print(f"📊 Datos procesados: {results.get('data_count', 0)}")
            print(f"🤖 Modelos entrenados: {results.get('models_trained', 0)}")
            print(f"📁 Carpeta de salida: {results.get('output_folder', 'N/A')}")
            
            # Mostrar resumen de modelos
            summary = results.get('summary', [])
            if summary:
                print("📋 Resumen de modelos:")
                for item in summary:
                    print(f"  - {item}")
        else:
            error_msg = results.get('error', 'Error desconocido')
            print(f"❌ Error: {error_msg}")
        
        print("✅ Prueba completada")
        
    except Exception as e:
        print(f"❌ Error en la prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_database()
