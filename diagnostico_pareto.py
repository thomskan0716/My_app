"""
ES: Script de diagnóstico para el error de Pareto Analysis.
EN: Diagnostic script for the Pareto Analysis error.
JA: Pareto Analysis エラーの診断スクリプト。

ES: Ejecuta este script para identificar la causa del problema.
EN: Run this script to identify the root cause.
JA: 原因特定のために実行。
"""
import sys
import os
from pathlib import Path

print("=" * 80)
print("Pareto解析エラーの診断")
print("=" * 80)

# ES: 1. Verificar xlsxwriter | EN: 1) Check xlsxwriter | JA: 1) xlsxwriter を確認
print("\n[1] xlsxwriter を確認中...")
try:
    import xlsxwriter
    print(f"  ✅ xlsxwriter インストール済み: バージョン {xlsxwriter.__version__}")
    print(f"  📍 場所: {xlsxwriter.__file__}")
except ImportError as e:
    print(f"  ❌ xlsxwriter がインストールされていません: {e}")
    sys.exit(1)
except Exception as e:
    print(f"  ❌ xlsxwriter のインポート中にエラー: {e}")
    sys.exit(1)

# ES: 2. Verificar pandas | EN: 2) Check pandas | JA: 2) pandas を確認
print("\n[2] pandas を確認中...")
try:
    import pandas as pd
    print(f"  ✅ pandas インストール済み: バージョン {pd.__version__}")
    print(f"  📍 場所: {pd.__file__}")
except ImportError as e:
    print(f"  ❌ pandas がインストールされていません: {e}")
    sys.exit(1)

# ES: 3. Probar ExcelWriter | EN: 3) Test ExcelWriter | JA: 3) ExcelWriter をテスト
print("\n[3] ExcelWriter をテスト中（engine='xlsxwriter'）...")
try:
    test_file = "test_pareto_diagnostico.xlsx"
    writer = pd.ExcelWriter(test_file, engine='xlsxwriter')
    print("  ✅ ExcelWriter を作成しました")
    
    # ES: Crear un DataFrame de prueba | EN: Create a test DataFrame | JA: テスト用DataFrameを作成
    test_df = pd.DataFrame({'test': [1, 2, 3]})
    test_df.to_excel(writer, sheet_name='test', index=False)
    writer.close()
    print("  ✅ Excelファイルを作成しました")
    
    # ES: Limpiar | EN: Cleanup | JA: 後片付け
    if os.path.exists(test_file):
        os.remove(test_file)
        print("  ✅ テストファイルを削除しました")
except Exception as e:
    print(f"  ❌ ExcelWriter 作成中にエラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ES: 4. Verificar entorno | EN: 4) Check environment | JA: 4) 環境を確認
print("\n[4] 環境を確認中...")
print(f"  Python 実行ファイル: {sys.executable}")
print(f"  Python バージョン: {sys.version}")
print(f"  現在のディレクトリ: {os.getcwd()}")

# ES: 5. Verificar sys.path | EN: 5) Check sys.path | JA: 5) sys.path を確認
print("\n[5] sys.path を確認中...")
venv_paths = [p for p in sys.path if '.venv' in p or 'venv' in p or 'site-packages' in p]
if venv_paths:
    print(f"  ✅ venv のパスを {len(venv_paths)} 件検出:")
    for p in venv_paths[:5]:  # Mostrar primeros 5
        print(f"    - {p}")
else:
    print("  ⚠️ sys.path に venv のパスが見つかりません")

# ES: 6. Verificar variables de entorno relevantes | EN: 6) Check relevant env vars | JA: 6) 関連する環境変数を確認
print("\n[6] 環境変数を確認中...")
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
    value = os.environ.get(var, '未設定')
    print(f"  {var}: {value}")

# ES: 7. Verificar permisos de escritura | EN: 7) Check write permissions | JA: 7) 書込み権限を確認
print("\n[7] 書き込み権限を確認中...")
try:
    test_write = "test_write_permissions.txt"
    with open(test_write, 'w') as f:
        f.write("test")
    os.remove(test_write)
    print("  ✅ 現在のディレクトリで書き込み権限OK")
except Exception as e:
    print(f"  ❌ 書き込み権限エラー: {e}")

# ES: 8. Verificar DLLs (solo Windows) | EN: 8) Check DLLs (Windows only) | JA: 8) DLL確認（Windowsのみ）
if sys.platform == 'win32':
    print("\n[8] DLL を確認中（Windows）...")
    try:
        from dll_debug import detect_openmp_runtimes, get_loaded_dlls
        dll_list = get_loaded_dlls()
        omp_info = detect_openmp_runtimes(dll_list)
        
        if omp_info['all_omp_dlls']:
            print(f"  ⚠️ OpenMP DLL を {len(omp_info['all_omp_dlls'])} 件検出:")
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
                print("  ❌ 競合: 複数のOpenMPランタイムを検出")
            else:
                print("  ✅ OpenMP ランタイムは1つのみ（競合なし）")
        else:
            print("  ✅ OpenMP DLL は検出されませんでした")
    except ImportError:
        print("  ⚠️ dll_debug をインポートできません（致命的ではありません）")
    except Exception as e:
        print(f"  ⚠️ DLL確認中にエラー: {e}")

# ES: 9. Verificar estructura de carpetas esperada | EN: 9) Check expected folder structure | JA: 9) 想定フォルダ構造を確認
print("\n[9] フォルダー構造を確認中...")
expected_folders = [
    "03_予測",
    "04_パレート解"
]
for folder in expected_folders:
    if os.path.exists(folder):
        print(f"  ✅ フォルダーあり: {folder}")
        # ES: Verificar permisos de escritura | EN: Check write permissions | JA: 書込み権限を確認
        try:
            test_file = os.path.join(folder, "test_write.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"    ✅ 書き込み権限OK")
        except Exception as e:
            print(f"    ❌ 書き込み権限なし: {e}")
    else:
        print(f"  ⚠️ フォルダーなし: {folder}（自動作成します）")

# ES: 10. Simular el entorno del subproceso | EN: 10) Simulate subprocess environment | JA: 10) サブプロセス環境を想定
print("\n[10] サブプロセス環境を想定中...")
print("  これは現在の環境です。subprocess から実行すると:")
print("  - 作業ディレクトリが異なる可能性があります")
print("  - 環境変数が異なる可能性があります")
print("  - sys.path に必要なパスが含まれない可能性があります")

print("\n" + "=" * 80)
print("診断完了")
print("=" * 80)
print("\nすべてのチェックが通った(✅)場合、問題は次の可能性が高いです:")
print("  1. サブプロセス環境（環境変数が異なる）")
print("  2. nonlinear_worker から実行したときの作業ディレクトリ")
print("  3. サブプロセスでのみ発生する DLL 競合")
print("\n推奨: nonlinear_worker.py を確認して次を保証してください:")
print("  - sys.executable が venv の Python を指している")
print("  - PYTHONPATH に venv の site-packages が含まれている")
print("  - 環境変数が xlsxwriter に影響しない")





