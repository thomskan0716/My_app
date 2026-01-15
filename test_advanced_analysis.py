#!/usr/bin/env python
# coding: utf-8

"""
Script de prueba para el módulo de análisis lineal avanzado
"""

import sys
import os
from pathlib import Path

# Agregar el directorio actual al path
sys.path.append(str(Path(__file__).parent))

try:
    from linear_analysis_advanced import run_advanced_linear_analysis_from_db
    from db_manager import DBManager
    print("✅ Módulos importados correctamente")
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    sys.exit(1)

def test_advanced_analysis():
    """Prueba del análisis lineal avanzado"""
    print("🔧 Iniciando prueba del análisis lineal avanzado...")
    
    try:
        # Crear DBManager
        db_manager = DBManager()
        print("✅ DBManager creado correctamente")
        
        # Verificar conexión
        if not db_manager.conn:
            print("❌ No hay conexión a la base de datos")
            return False
        
        # Ejecutar análisis sin filtros
        print("🔧 Ejecutando análisis sin filtros...")
        results = run_advanced_linear_analysis_from_db(db_manager)
        
        if results.get('success', False):
            print("✅ Análisis ejecutado exitosamente")
            print(f"📁 Directorio de salida: {results.get('output_directory', 'N/A')}")
            print(f"📊 Forma de datos: {results.get('data_shape', 'N/A')}")
            print(f"📈 Calculadora Excel: {results.get('excel_calculator', 'N/A')}")
            
            # Verificar estructura de carpetas
            output_dir = Path(results.get('output_directory', ''))
            if output_dir.exists():
                print("\n📁 Estructura de carpetas generada:")
                for item in output_dir.rglob('*'):
                    if item.is_file():
                        print(f"  📄 {item.relative_to(output_dir)}")
                    elif item.is_dir():
                        print(f"  📁 {item.relative_to(output_dir)}/")
            
            return True
        else:
            print(f"❌ Error en el análisis: {results.get('error', 'Error desconocido')}")
            return False
            
    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Iniciando prueba del análisis lineal avanzado...")
    success = test_advanced_analysis()
    
    if success:
        print("\n✅ PRUEBA EXITOSA: El análisis lineal avanzado funciona correctamente")
    else:
        print("\n❌ PRUEBA FALLIDA: Hay problemas con el análisis lineal avanzado")
    
    print("\nPrueba completada.")
