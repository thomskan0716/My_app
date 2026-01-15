"""
Script de diagnóstico para verificar si los modelos se entrenaron correctamente
"""
import os
import sys
from pathlib import Path
import pandas as pd
import glob

# Agregar rutas al path
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PYTHON_CODE_FOLDER = PROJECT_ROOT / "00_Pythonコード"
if str(PYTHON_CODE_FOLDER) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE_FOLDER))

from config import Config

def find_prediction_files():
    """
    Busca archivos Prediction_output.xlsx en la estructura de análisis no lineal.
    La estructura es: output_folder/回帰_0817_DCV_shap/ (donde están los gráficos)
    y el archivo de predicción está en: output_folder/03_予測/Prediction_output.xlsx
    """
    prediction_files = []
    current_dir = Path.cwd()
    
    print(f"   🔍 Buscando desde: {current_dir}", flush=True)
    
    # Buscar todas las carpetas que contengan 回帰_0817_DCV_shap
    # Esta es la carpeta donde están los gráficos del análisis no lineal
    regression_folders = []
    
    # Buscar en el directorio actual y subdirectorios
    for pattern in ["**/回帰_0817_DCV_shap", "回帰_0817_DCV_shap"]:
        found = list(current_dir.glob(pattern))
        regression_folders.extend(found)
    
    # También buscar en subdirectorios comunes
    common_bases = [
        current_dir,
        current_dir / "Archivos_de_salida",
        current_dir.parent / "Archivos_de_salida",
    ]
    
    for base in common_bases:
        if base.exists():
            for pattern in ["**/回帰_0817_DCV_shap", "**/04_非線形回帰/**/回帰_0817_DCV_shap"]:
                found = list(base.glob(pattern))
                regression_folders.extend(found)
    
    # Eliminar duplicados
    regression_folders = list(set(regression_folders))
    
    print(f"   📁 Carpetas 回帰_0817_DCV_shap encontradas: {len(regression_folders)}", flush=True)
    
    for reg_folder in regression_folders:
        reg_path = Path(reg_folder)
        # El working_dir es el padre de 回帰_0817_DCV_shap
        working_dir = reg_path.parent
        # El archivo de predicción está en working_dir/03_予測/Prediction_output.xlsx
        prediction_path = working_dir / "03_予測" / "Prediction_output.xlsx"
        if prediction_path.exists():
            prediction_files.append(str(prediction_path))
            print(f"      ✅ Encontrado: {prediction_path}", flush=True)
        else:
            print(f"      ⚠️ No encontrado en: {prediction_path}", flush=True)
    
    # También buscar directamente
    search_patterns = [
        "**/03_予測/Prediction_output.xlsx"
    ]
    
    for pattern in search_patterns:
        files = list(current_dir.glob(pattern))
        for f in files:
            if str(f) not in prediction_files:
                prediction_files.append(str(f))
                print(f"      ✅ Encontrado (búsqueda directa): {f}", flush=True)
    
    # Eliminar duplicados y ordenar
    return sorted(set(prediction_files))

def check_models(prediction_path=None):
    """Verifica si los modelos están entrenados y disponibles"""
    print("=" * 80, flush=True)
    print("🔍 DIAGNÓSTICO DE MODELOS", flush=True)
    print("=" * 80, flush=True)
    
    # 1. Verificar configuración
    print("\n📋 CONFIGURACIÓN:")
    print(f"   TARGET_COLUMNS: {Config.TARGET_COLUMNS}")
    print(f"   MODEL_FOLDER: {Config.MODEL_FOLDER}")
    print(f"   FINAL_MODEL_PREFIX: {Config.FINAL_MODEL_PREFIX}")
    print(f"   PREDICTION_COLUMN_PREFIX: {Config.PREDICTION_COLUMN_PREFIX}")
    
    # 2. Verificar archivos de modelo
    print("\n📦 ARCHIVOS DE MODELO:")
    
    # Si tenemos una ruta de predicción, buscar modelos en la misma estructura
    model_search_paths = []
    if prediction_path:
        pred_path = Path(prediction_path)
        # El working_dir es el padre de 03_予測
        working_dir = pred_path.parent.parent
        model_folder_in_working = working_dir / Config.MODEL_FOLDER
        if model_folder_in_working.exists():
            model_search_paths.append(model_folder_in_working)
    
    # También buscar en la ruta por defecto
    default_model_folder = Path(Config.MODEL_FOLDER)
    if default_model_folder.exists():
        model_search_paths.append(default_model_folder)
    
    # Si no hay rutas, usar la ruta por defecto aunque no exista
    if not model_search_paths:
        model_search_paths.append(default_model_folder)
    
    model_files = {}
    for target in Config.TARGET_COLUMNS:
        model_filename = f"{Config.FINAL_MODEL_PREFIX}_{target}.pkl"
        found = False
        
        for model_folder_path in model_search_paths:
            model_path = model_folder_path / model_filename
            if model_path.exists():
                model_files[target] = {
                    'exists': True,
                    'path': str(model_path)
                }
                found = True
                status = "✅"
                print(f"   {status} {target}: {model_path}")
                size = model_path.stat().st_size / (1024 * 1024)  # MB
                print(f"      Tamaño: {size:.2f} MB")
                break
        
        if not found:
            model_files[target] = {
                'exists': False,
                'path': str(model_search_paths[0] / model_filename)
            }
            status = "❌"
            print(f"   {status} {target}: {model_search_paths[0] / model_filename}")
    
    # 3. Verificar archivo de predicción
    print("\n📊 ARCHIVO DE PREDICCIÓN:")
    
    # Si no se proporciona una ruta, buscar archivos
    if prediction_path is None:
        print(f"   🔍 Buscando archivos de predicción en estructura de análisis no lineal...")
        prediction_files = find_prediction_files()
        if prediction_files:
            print(f"   ✅ Se encontraron {len(prediction_files)} archivo(s) de predicción:")
            for i, pf in enumerate(prediction_files, 1):
                # Mostrar también la carpeta de gráficos asociada
                pf_path = Path(pf)
                graphics_folder = pf_path.parent.parent / "回帰_0817_DCV_shap"
                if graphics_folder.exists():
                    print(f"      {i}. {pf}")
                    print(f"         📁 Carpeta de gráficos: {graphics_folder}")
                else:
                    print(f"      {i}. {pf}")
            prediction_path = prediction_files[0]  # Usar el primero
            print(f"\n   📁 Analizando: {prediction_path}")
        else:
            # Intentar con la ruta por defecto
            prediction_folder = Config.PREDICTION_FOLDER
            prediction_file = Config.PREDICTION_OUTPUT_FILE
            prediction_path = os.path.join(prediction_folder, prediction_file)
            print(f"   ⚠️ No se encontraron archivos en estructura de análisis no lineal")
            print(f"   🔍 Intentando ruta por defecto: {prediction_path}")
    
    if os.path.exists(prediction_path):
        print(f"   ✅ Archivo encontrado: {prediction_path}")
        
        try:
            df = pd.read_excel(prediction_path)
            print(f"   📐 Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
            
            print(f"\n   📋 COLUMNAS ENCONTRADAS:")
            for col in df.columns:
                print(f"      - {col}")
            
            # Verificar columnas de predicción
            print(f"\n   🎯 COLUMNAS DE PREDICCIÓN ESPERADAS:")
            expected_cols = []
            for target in Config.TARGET_COLUMNS:
                pred_col = f"{Config.PREDICTION_COLUMN_PREFIX}_{target}"
                expected_cols.append(pred_col)
                exists = pred_col in df.columns
                status = "✅" if exists else "❌"
                print(f"      {status} {pred_col}")
            
            # Verificar 切削時間
            cutting_time_col = Config.CUTTING_TIME_COLUMN_NAME
            print(f"\n   ⏱️ COLUMNA DE TIEMPO DE CORTE:")
            exists = cutting_time_col in df.columns
            status = "✅" if exists else "❌"
            print(f"      {status} {cutting_time_col}")
            
            # Resumen
            print(f"\n   📊 RESUMEN:")
            found_pred_cols = sum(1 for col in expected_cols if col in df.columns)
            print(f"      Columnas de predicción encontradas: {found_pred_cols}/{len(expected_cols)}")
            print(f"      Columna 切削時間 encontrada: {'Sí' if cutting_time_col in df.columns else 'No'}")
            
            # Verificar para Pareto
            print(f"\n   🎯 VERIFICACIÓN PARA PARETO:")
            pareto_objectives = Config.PARETO_OBJECTIVES
            print(f"      Objetivos de Pareto configurados: {list(pareto_objectives.keys())}")
            
            pareto_found = []
            pareto_missing = []
            
            for obj_name in pareto_objectives.keys():
                # Para 切削時間, buscar directamente
                if obj_name == Config.CUTTING_TIME_COLUMN_NAME:
                    if obj_name in df.columns:
                        pareto_found.append(obj_name)
                    else:
                        pareto_missing.append(obj_name)
                else:
                    # Para otros, buscar con prefijo prediction_
                    pred_col = f"{Config.PREDICTION_COLUMN_PREFIX}_{obj_name}"
                    if pred_col in df.columns:
                        pareto_found.append(obj_name)
                    elif obj_name in df.columns:
                        pareto_found.append(obj_name)
                    else:
                        pareto_missing.append(obj_name)
            
            print(f"      ✅ Objetivos encontrados ({len(pareto_found)}): {pareto_found}")
            if pareto_missing:
                print(f"      ❌ Objetivos faltantes ({len(pareto_missing)}): {pareto_missing}")
            
            if len(pareto_found) < 2:
                print(f"\n   ⚠️ ADVERTENCIA: Solo se encontraron {len(pareto_found)} objetivos de Pareto.")
                print(f"      Se necesitan al menos 2 para el análisis de Pareto.")
            else:
                print(f"\n   ✅ OK: Se encontraron {len(pareto_found)} objetivos de Pareto (suficiente para análisis)")
                
        except Exception as e:
            print(f"   ❌ Error leyendo el archivo: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"   ❌ Archivo NO encontrado: {prediction_path}")
    
    # 4. Resumen final
    print("\n" + "=" * 80)
    print("📊 RESUMEN FINAL")
    print("=" * 80)
    
    models_ok = sum(1 for info in model_files.values() if info['exists'])
    print(f"   Modelos encontrados: {models_ok}/{len(Config.TARGET_COLUMNS)}")
    
    if models_ok == len(Config.TARGET_COLUMNS):
        print("   ✅ Todos los modelos están disponibles")
    else:
        print("   ⚠️ Faltan algunos modelos")
        for target, info in model_files.items():
            if not info['exists']:
                print(f"      ❌ Falta: {target}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser(description='Verificar modelos entrenados')
        parser.add_argument('--prediction-file', type=str, help='Ruta al archivo Prediction_output.xlsx')
        args = parser.parse_args()
        
        check_models(prediction_path=args.prediction_file)
    except Exception as e:
        print(f"❌ Error ejecutando script: {e}")
        import traceback
        traceback.print_exc()

