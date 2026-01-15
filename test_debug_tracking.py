#!/usr/bin/env python
# coding: utf-8

from db_manager import DBManager
from linear_analysis_advanced import run_advanced_linear_analysis_from_db
import pandas as pd
import os

def test_debug_tracking():
    """Testear el tracking de datos con debugs"""
    try:
        print("🔧 Testeando tracking de datos con debugs...")
        print("=" * 70)
        
        # Crear instancia de DBManager
        db_manager = DBManager()
        
        # Filtros que deberían devolver solo registros con 線材長 = 75
        filters = {
            'A11': True,
            '線材長': ('75', '75')
        }
        
        print(f"📊 Filtros aplicados: {filters}")
        print("=" * 70)
        
        # Ejecutar análisis lineal con debugs
        results = run_advanced_linear_analysis_from_db(
            db_manager, 
            filters, 
            "debug_tracking_test"
        )
        
        print("=" * 70)
        print("📋 VERIFICACIÓN FINAL:")
        print("=" * 70)
        
        if results.get('success', False):
            print(f"✅ Análisis exitoso")
            print(f"📊 Datos procesados: {results.get('data_count', 0)}")
            print(f"📊 Rango de datos: {results.get('data_range', 'N/A')}")
            
            # Verificar el archivo Excel final
            excel_path = os.path.join("debug_tracking_test", "01_学習モデル", "filtered_data.xlsx")
            if os.path.exists(excel_path):
                df_final = pd.read_excel(excel_path)
                print(f"\n📊 Verificación del archivo Excel final:")
                print(f"📊 Filas en archivo final: {len(df_final)}")
                
                if '線材長' in df_final.columns:
                    unique_final = df_final['線材長'].unique()
                    print(f"📊 Valores únicos finales en 線材長: {unique_final}")
                    
                    count_74_final = len(df_final[df_final['線材長'] == 74])
                    count_75_final = len(df_final[df_final['線材長'] == 75])
                    print(f"📊 Registros finales con 線材長 = 74: {count_74_final}")
                    print(f"📊 Registros finales con 線材長 = 75: {count_75_final}")
                    
                    if count_74_final > 0:
                        print(f"❌ PROBLEMA: El archivo final contiene {count_74_final} registros con 線材長 = 74")
                    else:
                        print(f"✅ CORRECTO: El archivo final contiene solo registros con 線材長 = 75")
                else:
                    print("❌ Columna 線材長 no encontrada en archivo final")
            else:
                print(f"❌ Archivo Excel final no encontrado: {excel_path}")
        else:
            print(f"❌ Error en análisis: {results.get('error', 'Error desconocido')}")
        
        db_manager.conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_debug_tracking()

