#!/usr/bin/env python
# coding: utf-8

from db_manager import DBManager
from linear_analysis_advanced import run_advanced_linear_analysis_from_db

def test_filter_logic():
    """Probar la lógica de filtros corregida"""
    try:
        print("🔧 Probando lógica de filtros corregida...")
        
        db = DBManager()
        
        # Crear carpeta de prueba
        import os
        test_folder = "test_filter_logic"
        os.makedirs(test_folder, exist_ok=True)
        
        # Casos de prueba específicos
        test_cases = [
            {
                'name': 'Solo rango de 線材長 (75-75)',
                'filters': {
                    '線材長': ('75', '75')
                },
                'expected': 'Registros con 線材長 = 75'
            },
            {
                'name': 'A11 específico',
                'filters': {
                    'A11': True
                },
                'expected': 'Registros con A11 = 1'
            },
            {
                'name': 'すべて seleccionado',
                'filters': {
                    'すべて': True
                },
                'expected': 'Registros con cualquier cepillo = 1'
            },
            {
                'name': 'Rango de velocidad sin cepillo',
                'filters': {
                    '送り速度': ('1000', '2000')
                },
                'expected': 'Registros con velocidad entre 1000-2000'
            },
            {
                'name': 'A11 + rango de velocidad',
                'filters': {
                    'A11': True,
                    '送り速度': ('1000', '2000')
                },
                'expected': 'Registros con A11=1 Y velocidad entre 1000-2000'
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{'='*60}")
            print(f"🔧 Prueba {i+1}: {test_case['name']}")
            print(f"Filtros: {test_case['filters']}")
            print(f"Esperado: {test_case['expected']}")
            print(f"{'='*60}")
            
            # Ejecutar análisis
            results = run_advanced_linear_analysis_from_db(
                db, 
                test_case['filters'], 
                test_folder
            )
            
            if results.get('success', False):
                data_count = results.get('data_count', 0)
                models_trained = results.get('models_trained', 0)
                print(f"✅ ÉXITO: {data_count} registros procesados, {models_trained} modelos entrenados")
                
                # Mostrar resumen de modelos
                summary = results.get('summary', [])
                if summary:
                    print("📊 Modelos entrenados:")
                    for item in summary:
                        target = item['target']
                        model = item['model']
                        if 'r2' in item:
                            metric = f"R² = {item['r2']:.3f}"
                        else:
                            metric = f"Accuracy = {item['accuracy']:.3f}"
                        print(f"   {target}: {model} - {metric}")
                
            else:
                error = results.get('error', 'Error desconocido')
                print(f"❌ ERROR: {error}")
        
        print(f"\n{'='*60}")
        print("✅ TODAS LAS PRUEBAS COMPLETADAS")
        print(f"📁 Resultados guardados en: {test_folder}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ Error en pruebas: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_filter_logic()
