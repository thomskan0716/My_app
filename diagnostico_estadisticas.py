"""
ES: Script de diagnóstico para verificar el problema de estadísticas de clasificación.
EN: Diagnostic script to investigate the classification statistics issue.
JA: 分類統計の問題を調べる診断スクリプト。

ES: Ejecutar con: python diagnostico_estadisticas.py <ruta_carpeta_analisis>
EN: Run with: python diagnostico_estadisticas.py <analysis_folder_path>
JA: 実行: python diagnostico_estadisticas.py <解析フォルダパス>
"""
import os
import re
import sys
from pathlib import Path

def diagnosticar_estadisticas(output_folder):
    """
    ES: Diagnostica el problema con las estadísticas de clasificación.
    EN: Diagnose issues with classification statistics.
    JA: 分類統計の問題を診断。
    """
    print("=" * 80)
    print("分類統計の診断")
    print("=" * 80)
    
    print(f"\n📁 出力フォルダー: {output_folder}")
    if not os.path.exists(output_folder):
        print("❌ 出力フォルダーが存在しません")
        return False
    
    # ES: Verificar estructura de carpetas | EN: Check folder structure | JA: フォルダ構造を確認
    print("\n📂 フォルダー構造を確認中:")
    expected_paths = [
        ('02_本学習結果', os.path.join(output_folder, '02_本学習結果')),
        ('04_診断情報', os.path.join(output_folder, '02_本学習結果', '04_診断情報')),
        ('02_評価結果', os.path.join(output_folder, '02_本学習結果', '02_評価結果')),
    ]
    
    estructura_ok = True
    for name, path in expected_paths:
        exists = os.path.exists(path)
        is_dir = os.path.isdir(path) if exists else False
        status = '✅' if exists and is_dir else '❌'
        print(f"   {status} {name}: {path}")
        if not exists or not is_dir:
            estructura_ok = False
    
    # ES: Buscar diagnostic_report.txt | EN: Locate diagnostic_report.txt | JA: diagnostic_report.txt を探索
    print("\n📄 diagnostic_report.txt を検索中:")
    diagnostic_report_path = os.path.join(output_folder, '02_本学習結果', '04_診断情報', 'diagnostic_report.txt')
    alternative_path = os.path.join(output_folder, '02_本学習結果', '02_評価結果', 'diagnostic_report.txt')
    
    report_path = None
    if os.path.exists(diagnostic_report_path):
        report_path = diagnostic_report_path
        print(f"   ✅ 見つかりました: {diagnostic_report_path}")
    elif os.path.exists(alternative_path):
        report_path = alternative_path
        print(f"   ✅ 見つかりました（代替パス）: {alternative_path}")
    else:
        print(f"   ❌ 想定パスに見つかりませんでした")
        print(f"      検索先:")
        print(f"        1. {diagnostic_report_path}")
        print(f"        2. {alternative_path}")
        
        # ES: Buscar recursivamente | EN: Search recursively | JA: 再帰的に探索
        print(f"\n   🔍 diagnostic_report.txt を再帰検索:")
        encontrado = False
        for root, dirs, files in os.walk(output_folder):
            if 'diagnostic_report.txt' in files:
                found_path = os.path.join(root, 'diagnostic_report.txt')
                print(f"      ✅ 見つかりました: {found_path}")
                report_path = found_path
                encontrado = True
                break
        
        if not encontrado:
            print(f"      ❌ どこにも見つかりませんでした")
            return False
    
    # ES: Leer y analizar el contenido | EN: Read and analyze content | JA: 内容を読み取り解析
    print(f"\n📖 ファイルを解析中: {report_path}")
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"   ✅ ファイルを読み込みました")
        print(f"   サイズ: {len(content)} 文字")
        print(f"   行数: {len(content.splitlines())} 行")
        
        # ES: Mostrar primeras líneas | EN: Show first lines | JA: 先頭行を表示
        print(f"\n📋 先頭20行:")
        lines = content.splitlines()
        for i, line in enumerate(lines[:20], 1):
            print(f"   {i:3d}: {line}")
        
        # Probar expresiones regulares (igual que en 0sec.py)
        print(f"\n🔍 正規表現をテスト中（0sec.py と同等）:")
        diagnostic_data = {}
        problemas = []
        
        # [設定情報]
        print(f"\n   [設定情報]:")
        np_alpha_match = re.search(r'NP_ALPHA:\s*([\d.]+)', content)
        if np_alpha_match:
            diagnostic_data['np_alpha'] = np_alpha_match.group(1)
            print(f"      ✅ NP_ALPHA: {diagnostic_data['np_alpha']}")
        else:
            print(f"      ❌ NP_ALPHA: 見つかりません")
            alt_match = re.search(r'NP_ALPHA[:\s]+([\d.]+)', content, re.IGNORECASE)
            if alt_match:
                diagnostic_data['np_alpha'] = alt_match.group(1)
                print(f"         ⚠️  別形式を検出: {alt_match.group(1)}")
            else:
                problemas.append("NP_ALPHA が見つかりません")
        
        objective_match = re.search(r'目的変数:\s*(.+)', content)
        if objective_match:
            diagnostic_data['objective'] = objective_match.group(1).strip()
            print(f"      ✅ 目的変数: {diagnostic_data['objective']}")
        else:
            print(f"      ❌ 目的変数: 見つかりません")
            alt_match = re.search(r'目的変数[:\s]+(.+)', content)
            if alt_match:
                diagnostic_data['objective'] = alt_match.group(1).strip()
                print(f"         ⚠️  別形式を検出: {alt_match.group(1).strip()}")
            else:
                problemas.append("目的変数 が見つかりません")
        
        # [モデル情報]
        print(f"\n   [モデル情報]:")
        calibrator_match = re.search(r'Calibrator:\s*(.+)', content)
        if calibrator_match:
            diagnostic_data['calibrator'] = calibrator_match.group(1).strip()
            print(f"      ✅ Calibrator: {diagnostic_data['calibrator']}")
        else:
            print(f"      ❌ Calibrator: 見つかりません")
            problemas.append("Calibrator が見つかりません")
        
        # tau_pos con múltiples formatos
        tau_pos_match = re.search(r'τ\+\s*\(tau_pos\):\s*([\d.]+)', content)
        if not tau_pos_match:
            tau_pos_match = re.search(r'tau_pos[:\s]+([\d.]+)', content, re.IGNORECASE)
        if not tau_pos_match:
            tau_pos_match = re.search(r'τ\+[:\s]+([\d.]+)', content)
        if tau_pos_match:
            diagnostic_data['tau_pos'] = tau_pos_match.group(1)
            print(f"      ✅ τ+ (tau_pos): {diagnostic_data['tau_pos']}")
        else:
            print(f"      ❌ τ+ (tau_pos): 見つかりません")
            problemas.append("τ+ (tau_pos) が見つかりません")
        
        # tau_neg con múltiples formatos
        tau_neg_match = re.search(r'τ-\s*\(tau_neg\):\s*([\d.]+)', content)
        if not tau_neg_match:
            tau_neg_match = re.search(r'tau_neg[:\s]+([\d.]+)', content, re.IGNORECASE)
        if not tau_neg_match:
            tau_neg_match = re.search(r'τ-[:\s]+([\d.]+)', content)
        if tau_neg_match:
            diagnostic_data['tau_neg'] = tau_neg_match.group(1)
            print(f"      ✅ τ- (tau_neg): {diagnostic_data['tau_neg']}")
        else:
            print(f"      ❌ τ- (tau_neg): 見つかりません")
            problemas.append("τ- (tau_neg) が見つかりません")
        
        features_match = re.search(r'選択特徴量数:\s*(\d+)', content)
        if features_match:
            diagnostic_data['selected_features'] = features_match.group(1)
            print(f"      ✅ 選択特徴量数: {diagnostic_data['selected_features']}")
        else:
            print(f"      ❌ 選択特徴量数: 見つかりません")
        
        # [予測結果統計]
        print(f"\n   [予測結果統計]:")
        total_data_match = re.search(r'総データ数:\s*([\d,]+)', content)
        if total_data_match:
            diagnostic_data['total_data'] = total_data_match.group(1).replace(',', '')
            print(f"      ✅ 総データ数: {diagnostic_data['total_data']}")
        else:
            print(f"      ❌ 総データ数: 見つかりません")
        
        coverage_match = re.search(r'カバレッジ:\s*([\d.]+)%', content)
        if not coverage_match:
            coverage_match = re.search(r'カバレッジ[:\s]+([\d.]+)', content)
        if coverage_match:
            diagnostic_data['coverage'] = coverage_match.group(1)
            print(f"      ✅ カバレッジ: {diagnostic_data['coverage']}%")
        else:
            print(f"      ❌ カバレッジ: 見つかりません")
        
        # [ノイズ付加設定]
        print(f"\n   [ノイズ付加設定]:")
        noise_enabled_match = re.search(r'ノイズ付加:\s*(True|False)', content)
        if noise_enabled_match:
            diagnostic_data['noise_enabled'] = noise_enabled_match.group(1) == 'True'
            print(f"      ✅ ノイズ付加: {diagnostic_data['noise_enabled']}")
        else:
            print(f"      ❌ ノイズ付加: 見つかりません")
        
        noise_level_match = re.search(r'ノイズレベル:\s*([\d.]+)\s*ppm', content)
        if not noise_level_match:
            noise_level_match = re.search(r'ノイズレベル[:\s]+([\d.]+)', content)
        if noise_level_match:
            diagnostic_data['noise_level'] = noise_level_match.group(1)
            print(f"      ✅ ノイズレベル: {diagnostic_data['noise_level']} ppm")
        else:
            print(f"      ❌ ノイズレベル: 見つかりません")
        
        # Resumen de datos parseados
        print(f"\n📊 解析結果の要約:")
        print(f"   検出した項目数: {len(diagnostic_data)}")
        for key, value in diagnostic_data.items():
            print(f"      - {key}: {value}")
        
        # ES: Verificar qué se mostraría en la UI
        # EN: Verify what would be shown in the UI
        # JP: UIに何が表示されるか確認する
        print(f"\n🎨 UI 表示内容の推定:")
        has_basic_info = any([
            diagnostic_data.get('objective'),
            diagnostic_data.get('np_alpha'),
            diagnostic_data.get('total_data'),
            diagnostic_data.get('coverage'),
            diagnostic_data.get('selected_features'),
        ])
        print(f"   基本情報 (info_label): {'✅ あり' if has_basic_info else '❌ なし'}")
        if has_basic_info:
            print(f"      表示される想定: 解析完了時刻, 解析時間")
            if diagnostic_data.get('objective'):
                print(f"      + 目的変数: {diagnostic_data['objective']}")
            if diagnostic_data.get('np_alpha'):
                print(f"      + NP_ALPHA: {diagnostic_data['np_alpha']}")
            if diagnostic_data.get('total_data'):
                print(f"      + 総データ数: {diagnostic_data['total_data']}")
            if diagnostic_data.get('coverage'):
                print(f"      + カバレッジ: {diagnostic_data['coverage']}%")
            if diagnostic_data.get('selected_features'):
                print(f"      + 選択特徴量数: {diagnostic_data['selected_features']}")
        
        has_model_info = diagnostic_data.get('tau_pos') and diagnostic_data.get('tau_neg')
        print(f"   モデル情報 (metric_card): {'✅ あり' if has_model_info else '❌ なし'}")
        if not has_model_info:
            print(f"      ⚠️  必要: tau_pos と tau_neg（両方）")
            print(f"      tau_pos 検出: {bool(diagnostic_data.get('tau_pos'))}")
            print(f"      tau_neg 検出: {bool(diagnostic_data.get('tau_neg'))}")
        
        has_noise_info = diagnostic_data.get('noise_enabled')
        print(f"   ノイズ情報 (noise_card): {'✅ あり' if has_noise_info else '❌ なし'}")
        
        # Diagnóstico del problema
        print(f"\n🔍 問題診断:")
        if not has_basic_info and not has_model_info and not has_noise_info:
            print(f"   ❌ 重大: 表示に十分なデータが見つかりませんでした")
            print(f"      画面が空になる原因の可能性があります")
            print(f"      検出した問題: {len(problemas)}")
            for p in problemas:
                print(f"         - {p}")
            return False
        elif not has_model_info:
            print(f"   ⚠️  問題: モデル情報が不足しています（tau_pos または tau_neg）")
            print(f"      基本情報は表示されますが、モデルセクションは表示されません")
        else:
            print(f"   ✅ 問題はなさそうです。データは正常に表示されるはずです。")
            print(f"      それでも表示されない場合、原因は次の可能性があります:")
            print(f"        1. レイアウトが正しく更新されていない")
            print(f"        2. ウィジェットは作成されるが表示されない（CSS問題）")
            print(f"        3. container_layout が正しく接続されていない")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ファイル読み込み中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_folder = sys.argv[1]
    else:
        # ES: Buscar la última carpeta de análisis de clasificación
        # EN: Find the latest classification analysis folder
        # JP: 最新の分類解析フォルダを探す
        script_dir = Path(__file__).parent
        base_folder = script_dir / "05_分類"
        if not base_folder.exists():
            # ES: Intentar desde el directorio actual
            # EN: Try from the current directory
            # JP: カレントディレクトリから試す
            base_folder = Path("05_分類")
        
        if base_folder.exists():
            # ES: Buscar la carpeta más reciente
            # EN: Find the most recent folder
            # JP: 最新のフォルダを探す
            folders = [f for f in base_folder.iterdir() if f.is_dir() and "分類解析結果" in f.name]
            if folders:
                # Ordenar por fecha de modificación
                folders.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                output_folder = str(folders[0])
                print(f"📁 最新フォルダーを使用: {output_folder}")
            else:
                print("❌ 解析フォルダーが見つかりませんでした")
                print("   使い方: python diagnostico_estadisticas.py <出力フォルダーパス>")
                sys.exit(1)
        else:
            print("❌ ベースフォルダーが見つかりません: 05_分類")
            print("   使い方: python diagnostico_estadisticas.py <出力フォルダーパス>")
            sys.exit(1)
    
    success = diagnosticar_estadisticas(output_folder)
    sys.exit(0 if success else 1)














