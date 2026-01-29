"""
Script de diagnóstico para verificar si los modelos se entrenaron correctamente
"""
import os
import sys
from pathlib import Path
import pandas as pd
import glob

# ES: Agregar rutas al path | EN: Add paths to sys.path | JA: sys.path にパスを追加
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
    
    print(f"   🔍 検索開始: {current_dir}", flush=True)
    
    # ES: Buscar todas las carpetas que contengan 回帰_0817_DCV_shap
    # EN: Find all folders containing 回帰_0817_DCV_shap
    # JA: 回帰_0817_DCV_shap を含むフォルダを探索
    # ES: Esta es la carpeta donde están los gráficos del análisis no lineal
    # EN: This is the folder where non-linear analysis graphs are stored
    # JA: 非線形解析のグラフが保存されるフォルダ
    regression_folders = []
    
    # ES: Buscar en el directorio actual y subdirectorios | EN: Search current directory and subdirectories | JA: 現在ディレクトリとサブディレクトリを探索
    for pattern in ["**/回帰_0817_DCV_shap", "回帰_0817_DCV_shap"]:
        found = list(current_dir.glob(pattern))
        regression_folders.extend(found)
    
    # ES: También buscar en subdirectorios comunes | EN: Also search common base directories | JA: よくあるベースディレクトリも探索
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
    
    # ES: Eliminar duplicados | EN: Remove duplicates | JA: 重複を除去
    regression_folders = list(set(regression_folders))
    
    print(f"   📁 回帰_0817_DCV_shap フォルダー数: {len(regression_folders)}", flush=True)
    
    for reg_folder in regression_folders:
        reg_path = Path(reg_folder)
        # ES: El working_dir es el padre de 回帰_0817_DCV_shap
        # EN: working_dir is the parent folder of 回帰_0817_DCV_shap
        # JA: working_dir は 回帰_0817_DCV_shap の親フォルダ
        working_dir = reg_path.parent
        # ES: El archivo de predicción está en working_dir/03_予測/Prediction_output.xlsx
        # EN: The prediction file is at working_dir/03_予測/Prediction_output.xlsx
        # JA: 予測ファイルは working_dir/03_予測/Prediction_output.xlsx
        prediction_path = working_dir / "03_予測" / "Prediction_output.xlsx"
        if prediction_path.exists():
            prediction_files.append(str(prediction_path))
            print(f"      ✅ 見つかりました: {prediction_path}", flush=True)
        else:
            print(f"      ⚠️ 見つかりません: {prediction_path}", flush=True)
    
    # ES: También buscar directamente | EN: Also search directly | JA: 直接探索も行う
    search_patterns = [
        "**/03_予測/Prediction_output.xlsx"
    ]
    
    for pattern in search_patterns:
        files = list(current_dir.glob(pattern))
        for f in files:
            if str(f) not in prediction_files:
                prediction_files.append(str(f))
                print(f"      ✅ 見つかりました（直接検索）: {f}", flush=True)
    
    # ES: Eliminar duplicados y ordenar | EN: De-duplicate and sort | JA: 重複排除してソート
    return sorted(set(prediction_files))

def check_models(prediction_path=None):
    """ES: Verifica si los modelos están entrenados y disponibles
    EN: Check if models are trained and available
    JA: モデルが学習済みで利用可能か確認"""
    print("=" * 80, flush=True)
    print("🔍 モデル診断", flush=True)
    print("=" * 80, flush=True)
    
    # ES: 1. Verificar configuración | EN: 1) Check configuration | JA: 1) 設定を確認
    print("\n📋 設定:")
    print(f"   TARGET_COLUMNS: {Config.TARGET_COLUMNS}")
    print(f"   MODEL_FOLDER: {Config.MODEL_FOLDER}")
    print(f"   FINAL_MODEL_PREFIX: {Config.FINAL_MODEL_PREFIX}")
    print(f"   PREDICTION_COLUMN_PREFIX: {Config.PREDICTION_COLUMN_PREFIX}")
    
    # ES: 2. Verificar archivos de modelo | EN: 2) Check model files | JA: 2) モデルファイルを確認
    print("\n📦 モデルファイル:")
    
    # ES: Si tenemos una ruta de predicción, buscar modelos en la misma estructura
    # EN: If a prediction path is provided, search models in the same structure
    # JA: 予測パスがあれば同じ構造内でモデルを探索
    model_search_paths = []
    if prediction_path:
        pred_path = Path(prediction_path)
        # ES: El working_dir es el padre de 03_予測 | EN: working_dir is the parent of 03_予測 | JA: working_dir は 03_予測 の親
        working_dir = pred_path.parent.parent
        model_folder_in_working = working_dir / Config.MODEL_FOLDER
        if model_folder_in_working.exists():
            model_search_paths.append(model_folder_in_working)
    
    # ES: También buscar en la ruta por defecto | EN: Also search the default path | JA: デフォルトパスも探索
    default_model_folder = Path(Config.MODEL_FOLDER)
    if default_model_folder.exists():
        model_search_paths.append(default_model_folder)
    
    # ES: Si no hay rutas, usar la ruta por defecto aunque no exista
    # EN: If no paths were found, fall back to the default path even if it doesn't exist
    # JA: パスが無ければ存在しなくてもデフォルトパスを使用
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
                print(f"      サイズ: {size:.2f} MB")
                break
        
        if not found:
            model_files[target] = {
                'exists': False,
                'path': str(model_search_paths[0] / model_filename)
            }
            status = "❌"
            print(f"   {status} {target}: {model_search_paths[0] / model_filename}")
    
    # ES: 3. Verificar archivo de predicción
    # EN: 3. Verify prediction file
    # JP: 3. 予測ファイルを確認
    print("\n📊 予測ファイル:")
    
    # ES: Si no se proporciona una ruta, buscar archivos
    # EN: If no path is provided, search for files
    # JP: パスが指定されていない場合はファイルを検索する
    if prediction_path is None:
        print(f"   🔍 非線形解析構造内の予測ファイルを検索中...")
        prediction_files = find_prediction_files()
        if prediction_files:
            print(f"   ✅ 予測ファイルを {len(prediction_files)} 件見つけました:")
            for i, pf in enumerate(prediction_files, 1):
                # ES: Mostrar también la carpeta de gráficos asociada
                # EN: Also show the associated charts folder
                # JP: 関連するグラフフォルダも表示する
                pf_path = Path(pf)
                graphics_folder = pf_path.parent.parent / "回帰_0817_DCV_shap"
                if graphics_folder.exists():
                    print(f"      {i}. {pf}")
                    print(f"         📁 グラフフォルダー: {graphics_folder}")
                else:
                    print(f"      {i}. {pf}")
            prediction_path = prediction_files[0]  # Usar el primero
            print(f"\n   📁 解析対象: {prediction_path}")
        else:
            # ES: Intentar con la ruta por defecto
            # EN: Try the default path
            # JP: デフォルトのパスを試す
            prediction_folder = Config.PREDICTION_FOLDER
            prediction_file = Config.PREDICTION_OUTPUT_FILE
            prediction_path = os.path.join(prediction_folder, prediction_file)
            print(f"   ⚠️ 非線形解析構造内にファイルが見つかりませんでした")
            print(f"   🔍 デフォルトパスを試行: {prediction_path}")
    
    if os.path.exists(prediction_path):
        print(f"   ✅ ファイルを見つけました: {prediction_path}")
        
        try:
            df = pd.read_excel(prediction_path)
            print(f"   📐 形状: {df.shape[0]} 行 × {df.shape[1]} 列")
            
            print(f"\n   📋 検出した列:")
            for col in df.columns:
                print(f"      - {col}")
            
            # ES: Verificar columnas de predicción
            # EN: Verify prediction columns
            # JP: 予測列を確認する
            print(f"\n   🎯 期待される予測列:")
            expected_cols = []
            for target in Config.TARGET_COLUMNS:
                pred_col = f"{Config.PREDICTION_COLUMN_PREFIX}_{target}"
                expected_cols.append(pred_col)
                exists = pred_col in df.columns
                status = "✅" if exists else "❌"
                print(f"      {status} {pred_col}")
            
            # Verificar 切削時間
            cutting_time_col = Config.CUTTING_TIME_COLUMN_NAME
            print(f"\n   ⏱️ 切削時間列:")
            exists = cutting_time_col in df.columns
            status = "✅" if exists else "❌"
            print(f"      {status} {cutting_time_col}")
            
            # Resumen
            print(f"\n   📊 要約:")
            found_pred_cols = sum(1 for col in expected_cols if col in df.columns)
            print(f"      予測列の検出数: {found_pred_cols}/{len(expected_cols)}")
            print(f"      切削時間 列: {'あり' if cutting_time_col in df.columns else 'なし'}")
            
            # ES: Verificar para Pareto
            # EN: Check compatibility for Pareto
            # JP: Pareto用の整合性を確認する
            print(f"\n   🎯 Pareto 用の検証:")
            pareto_objectives = Config.PARETO_OBJECTIVES
            print(f"      設定された Pareto 目的: {list(pareto_objectives.keys())}")
            
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
                    # ES: Para otros, buscar con prefijo prediction_
                    # EN: For other objectives, look for the prediction_ prefix
                    # JP: 他の目的変数は prediction_ プレフィックスで探す
                    pred_col = f"{Config.PREDICTION_COLUMN_PREFIX}_{obj_name}"
                    if pred_col in df.columns:
                        pareto_found.append(obj_name)
                    elif obj_name in df.columns:
                        pareto_found.append(obj_name)
                    else:
                        pareto_missing.append(obj_name)
            
            print(f"      ✅ 見つかった目的（{len(pareto_found)}件）: {pareto_found}")
            if pareto_missing:
                print(f"      ❌ 見つからない目的（{len(pareto_missing)}件）: {pareto_missing}")
            
            if len(pareto_found) < 2:
                print(f"\n   ⚠️ 警告: Pareto 目的が {len(pareto_found)} 件しか見つかりませんでした。")
                print(f"      Pareto 解析には少なくとも 2 件必要です。")
            else:
                print(f"\n   ✅ OK: Pareto 目的を {len(pareto_found)} 件見つけました（解析に十分）")
                
        except Exception as e:
            print(f"   ❌ ファイル読み込み中にエラー: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"   ❌ ファイルが見つかりません: {prediction_path}")
    
    # 4. Resumen final
    print("\n" + "=" * 80)
    print("📊 最終要約")
    print("=" * 80)
    
    models_ok = sum(1 for info in model_files.values() if info['exists'])
    print(f"   見つかったモデル: {models_ok}/{len(Config.TARGET_COLUMNS)}")
    
    if models_ok == len(Config.TARGET_COLUMNS):
        print("   ✅ すべてのモデルが利用可能です")
    else:
        print("   ⚠️ 一部のモデルが不足しています")
        for target, info in model_files.items():
            if not info['exists']:
                print(f"      ❌ 不足: {target}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser(description='学習済みモデルを確認')
        parser.add_argument('--prediction-file', type=str, help='Prediction_output.xlsx のパス')
        args = parser.parse_args()
        
        check_models(prediction_path=args.prediction_file)
    except Exception as e:
        print(f"❌ スクリプト実行中にエラー: {e}")
        import traceback
        traceback.print_exc()

