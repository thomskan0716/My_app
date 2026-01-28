#!/usr/bin/env python
# coding: utf-8

"""
ES: Script de prueba para el módulo de análisis lineal avanzado.
EN: Test script for the advanced linear analysis module.
JA: 高度線形解析モジュールのテストスクリプト。
"""

import sys
import os
from pathlib import Path

# ES: Agregar el directorio actual al path
# EN: Add current directory to sys.path
# JA: 現在ディレクトリを sys.path に追加
sys.path.append(str(Path(__file__).parent))

try:
    from linear_analysis_advanced import run_advanced_linear_analysis_from_db
    from db_manager import DBManager
    print("✅ モジュールのインポートが完了しました")
except ImportError as e:
    print(f"❌ モジュールのインポート中にエラー: {e}")
    sys.exit(1)

def test_advanced_analysis():
    """ES: Prueba del análisis lineal avanzado
    EN: Test advanced linear analysis
    JA: 高度線形解析のテスト
    """
    print("🔧 高度線形解析のテストを開始...")
    
    try:
        # ES: Crear DBManager | EN: Create DBManager | JA: DBManager を作成
        db_manager = DBManager()
        print("✅ DBManager を作成しました")
        
        # ES: Verificar conexión | EN: Check connection | JA: 接続確認
        if not db_manager.conn:
            print("❌ データベースに接続できません")
            return False
        
        # ES: Ejecutar análisis sin filtros | EN: Run analysis without filters | JA: フィルタなしで解析実行
        print("🔧 フィルタなしで解析を実行中...")
        results = run_advanced_linear_analysis_from_db(db_manager)
        
        if results.get('success', False):
            print("✅ 解析が成功しました")
            print(f"📁 出力ディレクトリ: {results.get('output_directory', 'N/A')}")
            print(f"📊 データ形状: {results.get('data_shape', 'N/A')}")
            print(f"📈 Excel計算機: {results.get('excel_calculator', 'N/A')}")
            
            # ES: Verificar estructura de carpetas | EN: Check folder structure | JA: フォルダ構造を確認
            output_dir = Path(results.get('output_directory', ''))
            if output_dir.exists():
                print("\n📁 生成されたフォルダー構造:")
                for item in output_dir.rglob('*'):
                    if item.is_file():
                        print(f"  📄 {item.relative_to(output_dir)}")
                    elif item.is_dir():
                        print(f"  📁 {item.relative_to(output_dir)}/")
            
            return True
        else:
            print(f"❌ 解析中にエラー: {results.get('error', '不明なエラー')}")
            return False
            
    except Exception as e:
        print(f"❌ テスト中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 高度線形解析のテストを開始...")
    success = test_advanced_analysis()
    
    if success:
        print("\n✅ テスト成功: 高度線形解析は正常に動作しています")
    else:
        print("\n❌ テスト失敗: 高度線形解析に問題があります")
    
    print("\nテスト完了。")
