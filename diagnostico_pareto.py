"""
Script de diagnóstico para el error de Pareto Analysis
Ejecuta este script para identificar la causa del problema
"""
import sys
import os
from pathlib import Path

print("=" * 80)
print("DIAGNÓSTICO DE ERROR DE PARETO ANALYSIS")
print("=" * 80)

# 1. Verificar xlsxwriter
print("\n[1] Verificando xlsxwriter...")
try:
    import xlsxwriter
    print(f"  ✅ xlsxwriter instalado: versión {xlsxwriter.__version__}")
    print(f"  📍 Ubicación: {xlsxwriter.__file__}")
except ImportError as e:
    print(f"  ❌ xlsxwriter NO está instalado: {e}")
    sys.exit(1)
except Exception as e:
    print(f"  ❌ Error al importar xlsxwriter: {e}")
    sys.exit(1)

# 2. Verificar pandas
print("\n[2] Verificando pandas...")
try:
    import pandas as pd
    print(f"  ✅ pandas instalado: versión {pd.__version__}")
    print(f"  📍 Ubicación: {pd.__file__}")
except ImportError as e:
    print(f"  ❌ pandas NO está instalado: {e}")
    sys.exit(1)

# 3. Probar ExcelWriter
print("\n[3] Probando ExcelWriter con engine='xlsxwriter'...")
try:
    test_file = "test_pareto_diagnostico.xlsx"
    writer = pd.ExcelWriter(test_file, engine='xlsxwriter')
    print("  ✅ ExcelWriter creado exitosamente")
    
    # Crear un DataFrame de prueba
    test_df = pd.DataFrame({'test': [1, 2, 3]})
    test_df.to_excel(writer, sheet_name='test', index=False)
    writer.close()
    print("  ✅ Archivo Excel creado exitosamente")
    
    # Limpiar
    if os.path.exists(test_file):
        os.remove(test_file)
        print("  ✅ Archivo de prueba eliminado")
except Exception as e:
    print(f"  ❌ Error al crear ExcelWriter: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Verificar entorno
print("\n[4] Verificando entorno...")
print(f"  Python ejecutable: {sys.executable}")
print(f"  Python versión: {sys.version}")
print(f"  Directorio actual: {os.getcwd()}")

# 5. Verificar sys.path
print("\n[5] Verificando sys.path...")
venv_paths = [p for p in sys.path if '.venv' in p or 'venv' in p or 'site-packages' in p]
if venv_paths:
    print(f"  ✅ Encontrados {len(venv_paths)} paths del venv:")
    for p in venv_paths[:5]:  # Mostrar primeros 5
        print(f"    - {p}")
else:
    print("  ⚠️ No se encontraron paths del venv en sys.path")

# 6. Verificar variables de entorno relevantes
print("\n[6] Verificando variables de entorno...")
env_vars = [
    'PYTHONPATH',
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'KMP_DUPLICATE_LIB_OK',
    'MPLBACKEND',
    'QT_QPA_PLATFORM'
]
for var in env_vars:
    value = os.environ.get(var, 'No definida')
    print(f"  {var}: {value}")

# 7. Verificar permisos de escritura
print("\n[7] Verificando permisos de escritura...")
try:
    test_write = "test_write_permissions.txt"
    with open(test_write, 'w') as f:
        f.write("test")
    os.remove(test_write)
    print("  ✅ Permisos de escritura OK en el directorio actual")
except Exception as e:
    print(f"  ❌ Error de permisos de escritura: {e}")

# 8. Verificar DLLs (solo Windows)
if sys.platform == 'win32':
    print("\n[8] Verificando DLLs (Windows)...")
    try:
        from dll_debug import detect_openmp_runtimes, get_loaded_dlls
        dll_list = get_loaded_dlls()
        omp_info = detect_openmp_runtimes(dll_list)
        
        if omp_info['all_omp_dlls']:
            print(f"  ⚠️ Se detectaron {len(omp_info['all_omp_dlls'])} DLLs OpenMP:")
            for category, dlls in omp_info.items():
                if category != 'all_omp_dlls' and dlls:
                    print(f"    {category}: {len(dlls)} DLLs")
            
            total_runtimes = sum([
                len(omp_info['intel']) > 0,
                len(omp_info['msvc']) > 0,
                len(omp_info['gcc']) > 0,
                len(omp_info['other']) > 0
            ])
            if total_runtimes > 1:
                print("  ❌ CONFLICTO: Múltiples runtimes OpenMP detectados")
            else:
                print("  ✅ Solo un runtime OpenMP (sin conflicto)")
        else:
            print("  ✅ No se detectaron DLLs OpenMP")
    except ImportError:
        print("  ⚠️ No se pudo importar dll_debug (no crítico)")
    except Exception as e:
        print(f"  ⚠️ Error verificando DLLs: {e}")

# 9. Verificar estructura de carpetas esperada
print("\n[9] Verificando estructura de carpetas...")
expected_folders = [
    "03_予測",
    "04_パレート解"
]
for folder in expected_folders:
    if os.path.exists(folder):
        print(f"  ✅ Carpeta existe: {folder}")
        # Verificar permisos de escritura
        try:
            test_file = os.path.join(folder, "test_write.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"    ✅ Permisos de escritura OK")
        except Exception as e:
            print(f"    ❌ Sin permisos de escritura: {e}")
    else:
        print(f"  ⚠️ Carpeta no existe: {folder} (se creará automáticamente)")

# 10. Simular el entorno del subproceso
print("\n[10] Simulando entorno del subproceso...")
print("  Este es el entorno actual. Cuando se ejecuta desde subprocess:")
print("  - El directorio de trabajo puede ser diferente")
print("  - Las variables de entorno pueden ser diferentes")
print("  - sys.path puede no incluir todas las rutas necesarias")

print("\n" + "=" * 80)
print("DIAGNÓSTICO COMPLETADO")
print("=" * 80)
print("\nSi todos los checks pasaron (✅), el problema probablemente está en:")
print("  1. El entorno del subproceso (variables de entorno diferentes)")
print("  2. El directorio de trabajo cuando se ejecuta desde nonlinear_worker")
print("  3. Conflictos de DLLs que solo aparecen en subprocesos")
print("\nRecomendación: Revisa nonlinear_worker.py para asegurar que:")
print("  - sys.executable apunta al Python del venv")
print("  - PYTHONPATH incluye site-packages del venv")
print("  - Las variables de entorno no interfieren con xlsxwriter")





