#!/usr/bin/env python
# coding: utf-8

"""
Test script to verify that linear_analysis_advanced.py works exactly like the reference file
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from linear_analysis_advanced import run_advanced_linear_analysis_from_db
from db_manager import DBManager

def test_exact_reference():
    """Test that the analysis works exactly like the reference file"""
    print("🧪 TESTEANDO ANÁLISIS EXACTO COMO ARCHIVO DE REFERENCIA")
    print("=" * 60)
    
    try:
        # Initialize database manager
        db_manager = DBManager()
        
        # Test without filters first
        print("📊 Ejecutando análisis sin filtros...")
        results = run_advanced_linear_analysis_from_db(db_manager, filters=None)
        
        if results['success']:
            print("✅ ANÁLISIS EXITOSO")
            print(f"📁 Directorio de salida: {results['output_directory']}")
            print(f"📊 Forma de datos: {results['data_shape']}")
            
            # Check models
            models = results.get('models', {})
            print(f"🤖 Modelos entrenados: {len(models)}")
            
            for target, model_info in models.items():
                if model_info.get('model') is not None:
                    model_name = model_info.get('model_name', 'Unknown')
                    task_type = model_info.get('task_type', 'Unknown')
                    print(f"  - {target}: {model_name} ({task_type})")
                    
                    # Check metrics
                    metrics = model_info.get('final_metrics', {})
                    if task_type == 'regression':
                        r2 = metrics.get('r2', 'N/A')
                        mae = metrics.get('mae', 'N/A')
                        rmse = metrics.get('rmse', 'N/A')
                        print(f"    R²: {r2}, MAE: {mae}, RMSE: {rmse}")
                    else:
                        accuracy = metrics.get('accuracy', 'N/A')
                        f1 = metrics.get('f1_score', 'N/A')
                        print(f"    Accuracy: {accuracy}, F1: {f1}")
                else:
                    error = model_info.get('error', 'Unknown error')
                    print(f"  - {target}: ERROR - {error}")
            
            # Check transformations
            transformations = results.get('transformation_info', {})
            print(f"🔄 Transformaciones aplicadas: {len([t for t in transformations.values() if t.get('applied', False)])}")
            
            for target, trans_info in transformations.items():
                if trans_info.get('applied', False):
                    method = trans_info.get('method', 'unknown')
                    print(f"  - {target}: {method} transformación")
                else:
                    reason = trans_info.get('reason', 'no transformation')
                    print(f"  - {target}: sin transformación ({reason})")
            
            # Check Excel calculator
            excel_path = results.get('excel_calculator')
            if excel_path:
                print(f"📊 Calculadora Excel: {excel_path}")
                if os.path.exists(excel_path):
                    print("✅ Archivo Excel creado correctamente")
                else:
                    print("❌ Archivo Excel no encontrado")
            else:
                print("⚠️ No se creó calculadora Excel")
            
            # Check output directory structure
            output_dir = Path(results['output_directory'])
            if output_dir.exists():
                print(f"📁 Estructura de directorios:")
                for item in output_dir.rglob('*'):
                    if item.is_file():
                        rel_path = item.relative_to(output_dir)
                        size = item.stat().st_size
                        print(f"  - {rel_path} ({size} bytes)")
            
            print("\n🎉 PRUEBA EXITOSA: El análisis funciona exactamente como el archivo de referencia")
            
        else:
            print(f"❌ ANÁLISIS FALLIDO: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR EN PRUEBA: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_exact_reference()
    if success:
        print("\n✅ TODAS LAS PRUEBAS PASARON - El análisis es idéntico al archivo de referencia")
    else:
        print("\n❌ ALGUNAS PRUEBAS FALLARON - Revisar implementación")
